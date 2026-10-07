"""Package a multi-exit checkpoint with per-exit temperatures and its exit policy (v6.0-VL stage 5, v6.1-VL stage 5).

Nothing is trained or fitted here: the inputs are the head-stage checkpoint and the outputs
of scripts/select_exit_policy.py. The steps are those that built v6.0-VL's package, in order:

  1. temperatures: calibration.json gets cal_mode "temp"; calibration.safetensors gets
     cal_logT = the main exit's temperature fitted on the checkpoint's own DEV "cal" half
     (`seed` step). That buffer is the main exit's temperature when serving
     (rsijev.calibrate.load_calibration; serve/infer.py's adaptive finalizer).
     --cal-buffers carries the other cal-4b buffers of an existing calibration file along;
     under cal_mode "temp" they are not read.
  2. policy: calibration.json "exits" = the cascade selection's temperature per early exit (an exit the
     binding cascade skips gets logT 10, so it never stops a question), "main_logT" and
     "T" = the cascade selection's (recorded; serving reads the main exit's from the buffer above);
     meta.json "adaptive" = {exits, tau, conf "temp", policy, dev}; the serving fields
     spec.option_pool_own_tokens = true, spec.max_length = 32768 and
     spec.train_max_length = the head stage's cap.
  3. --drop-exit L: exit L's head is removed from aux_scorers.safetensors, from
     spec.arch_extra.aux_exits (and fit_extra.aux_exit_weights), from adaptive.exits
     and from calibration.json; a policy that never routes to it is unchanged.
  4. --thresholds: adaptive.auto_thresholds = the confirmed per-exit thresholds
     (effort=auto; serve/release.py auto_thresholds), tau stays the single-number default.

v6.1-VL (a weight average, scripts/soup_checkpoints.py) has no seed or cascade-selection step:
--temperatures takes every exit's temperature, the main exit's included, from the thresholds
step's refit ("T"), so cal_logT and calibration.json's main_logT are the same number, and
--single-tau takes the single-number tau from `thresholds --single` (its fallback included,
recorded as not confirmed).

The tower is linked, not copied. The output loads with serve/release.py like any release
checkpoint (base model from the Hub).

  python scripts/package_multiexit.py --ckpt D/heads-s1 --seed-metric D/policy/seed.json --seed s1 \\
      --policy D/policy/cascade.json --drop-exit 12 --thresholds D/policy/auto.json --out D/package
  python scripts/package_multiexit.py --ckpt D/soup --temperatures D/policy/auto.json \\
      --single-tau D/policy/single.json --drop-exit 12 --thresholds D/policy/auto.json --out D/package
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WEIGHTS = ("tower.safetensors", "scorer.safetensors", "aux_scorers.safetensors")


def calibration_block(main_exit: int, logT: dict, skip: dict, fit: str) -> dict:
    """calibration.json for per-exit temperatures. logT: {exit: logT} for every exit."""
    lt = {int(k): float(v) for k, v in logT.items()}
    lt.update({int(k): float(v) for k, v in (skip or {}).items()})
    ex = sorted(lt)
    return {"cal_mode": "temp", "exits": {str(L): {"cal_mode": "temp", "logT": lt[L]} for L in ex if L != main_exit},
            "main_exit": main_exit, "main_logT": lt[main_exit],
            "T": {str(L): round(2.718281828 ** lt[L], 4) for L in ex}, "fit": fit}


def adaptive_block(policy: dict) -> dict:
    S = policy["serve"]
    return {"exits": [int(x) for x in S["exits"]], "tau": float(S["tau"]), "conf": "temp", "policy": policy["binding"],
            "tuned_on": "policy development set: half A (temperatures, policy fit), half B (selection)",
            "dev": policy["dev"][policy["binding"]]["B"],
            "skip_exits": sorted(int(k) for k in (S.get("skip_exit_logT") or {}))}


def single_tau_block(exits: list, single: dict) -> dict:
    """meta.json adaptive for a cascade over `exits` with the single tau of `thresholds --single`."""
    sel = single["selection"]
    tau = float(sel["tau"]["16"])
    assert float(sel["tau"]["20"]) == tau, sel["tau"]
    blk = {"exits": sorted(int(x) for x in exits), "tau": tau, "conf": "temp", "policy": f"C16_t{tau}",
           "tuned_on": "policy development set: half A (temperatures, selection), half B (confirmation)",
           "confirmed": bool(sel.get("CONFIRMED"))}
    if sel.get("fallback"):
        blk["fallback"] = True
        blk["dev"] = sel.get("fallback_B")
    elif sel.get("winner"):
        blk["dev"] = {k: v for k, v in single["dev"][sel["winner"]]["B"].items() if k != "feas"}
    return blk


def patch_serving_fields(meta: dict) -> dict:
    """Own-token pooling, the 32k serving cap, and the head stage's training cap."""
    s = meta["spec"]
    hs = meta.get("head_stage") or {}
    s["option_pool_own_tokens"] = True
    s["max_length"] = int((meta.get("release") or {}).get("max_length_text") or 32768)
    s["train_max_length"] = int(hs.get("max_length") or (s.get("fit_extra") or {}).get("train_max_length"))
    return meta


def drop_exit(meta: dict, cal: dict, aux: dict, L: int):
    """(meta, cal, aux) without exit L."""
    keep = {k: v for k, v in aux.items() if not k.startswith(f"{L}.")}
    if len(keep) == len(aux):
        raise SystemExit(f"aux_scorers has no exit {L}")
    ax = meta["spec"]["arch_extra"]
    ax["aux_exits"] = [x for x in ax["aux_exits"] if int(x) != L]
    fe = meta["spec"].get("fit_extra")
    if isinstance(fe, dict) and "aux_exit_weights" in fe:
        fe["aux_exit_weights"] = {k: v for k, v in fe["aux_exit_weights"].items() if int(k) != L}
    ad = meta["adaptive"]
    ad["exits"] = [x for x in ad["exits"] if int(x) != L]
    ad.pop("skip_exits", None)
    ad["note"] = f"exit {L} head removed; policy unchanged: cascade {' -> '.join(str(x) for x in ad['exits'])} at tau {ad['tau']}"
    meta.setdefault("release_patches", []).append(f"aux exit {L} removed (head dropped; policy never routes to it)")
    cal["exits"] = {k: v for k, v in cal["exits"].items() if int(k) != L}
    cal["T"] = {k: v for k, v in cal["T"].items() if int(k) != L}
    return meta, cal, keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="the head-stage checkpoint (scripts/fit_head.py --heads all)")
    ap.add_argument("--seed-metric", help="select_exit_policy.py seed output")
    ap.add_argument("--seed", help="this checkpoint's name in --seed-metric")
    ap.add_argument("--policy", help="select_exit_policy.py policy output")
    ap.add_argument("--temperatures", help="instead of --seed-metric/--policy: a thresholds output whose \"T\" "
                    "gives every exit's logT, the main exit's included")
    ap.add_argument("--single-tau", help="with --temperatures: a `thresholds --single` output (the default tau)")
    ap.add_argument("--thresholds", default="", help="select_exit_policy.py thresholds output (CONFIRMED)")
    ap.add_argument("--drop-exit", type=int, action="append", default=[])
    ap.add_argument("--cal-buffers", default="", help="an existing calibration.safetensors to carry along")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from safetensors.torch import load_file, save_file
    src, out = Path(a.ckpt), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    main_L = int(meta["spec"]["arch_extra"]["exit_layer"])
    fit = "per exit: question-type-weighted NLL grid on half A of the policy development set"
    buf = load_file(a.cal_buffers) if a.cal_buffers else {}
    if a.temperatures:
        if not a.single_tau or a.policy or a.seed_metric:
            raise SystemExit("--temperatures goes with --single-tau, without --seed-metric / --policy")
        logT = {int(k): float(v) for k, v in json.loads(Path(a.temperatures).read_text())["T"].items()}
        # 1. one temperature per exit, the main exit's from the same fit
        buf["cal_logT"] = torch.tensor(logT[main_L], dtype=torch.float32)
        cal = calibration_block(main_L, logT, None, fit)
        # 2. the single-number tau
        meta["adaptive"] = single_tau_block(list(logT), json.loads(Path(a.single_tau).read_text()))
    else:
        if not (a.seed_metric and a.seed and a.policy):
            raise SystemExit("need --seed-metric, --seed and --policy (or --temperatures and --single-tau)")
        seed = json.loads(Path(a.seed_metric).read_text())["metrics"][a.seed]
        policy = json.loads(Path(a.policy).read_text())
        # 1. temperatures: the served main-exit temperature from the checkpoint's own DEV
        buf["cal_logT"] = torch.tensor(float(seed["per_exit"][str(main_L)]["logT"]), dtype=torch.float32)
        # 2. the policy
        S = policy["serve"]
        assert int(S["exits"][-1]) == main_L, (S["exits"], main_L)
        cal = calibration_block(main_L, S["logT"], S.get("skip_exit_logT"), fit)
        meta["adaptive"] = adaptive_block(policy)
    patch_serving_fields(meta)
    aux = load_file(str(src / "aux_scorers.safetensors"))
    # 3. exits the policy never uses
    for L in a.drop_exit:
        meta, cal, aux = drop_exit(meta, cal, aux, L)
    # 4. effort=auto thresholds
    if a.thresholds:
        sel = json.loads(Path(a.thresholds).read_text())["selection"]
        if not sel.get("CONFIRMED"):
            raise SystemExit("--thresholds: the selection was not confirmed")
        meta["adaptive"]["auto_thresholds"] = {str(k): float(v) for k, v in sel["tau"].items()}
    if not math.isfinite(float(buf["cal_logT"])):
        raise SystemExit("cal_logT is not finite")

    for f in WEIGHTS[:2]:
        d = out / f
        if d.exists() or d.is_symlink():
            d.unlink()
        os.symlink((src / f).resolve(), d)
    for f in src.iterdir():                     # tokenizer / image-processor files, if any
        if f.is_file() and f.name not in (*WEIGHTS, "meta.json") and not f.name.startswith("calibration"):
            shutil.copy2(f, out / f.name)
    save_file({k: v.float().contiguous() for k, v in aux.items()}, str(out / "aux_scorers.safetensors"))
    save_file({k: v.float().contiguous() for k, v in buf.items()}, str(out / "calibration.safetensors"))
    (out / "calibration.json").write_text(json.dumps(cal, indent=1) + "\n")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print(json.dumps({"out": out.name, "adaptive": meta["adaptive"], "exits": cal["exits"],
                      "main_cal_logT": float(buf["cal_logT"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
