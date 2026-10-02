"""Put an adaptive-exit policy into a release directory.

The tuning run (private tools/adaexit/measure.py) writes policy.pt: the exits, the tau
tuned on DEV and one cal-4b per aux exit. A release carries the same thing as files
the server reads without unpickling anything:

  adaptive_calibration.safetensors   "<exit>.<mean|W|mu|sd|w|b>" for every aux exit
  meta.json "adaptive"               {"exits", "tau", "calibration", "source"[, "serving"]}

    python scripts/pack_adaptive_policy.py --release DIR --policy policy.pt [--source TEXT]
        [--serving auto|on|off]

`--serving` records which requests the server runs adaptive (serve.release.adaptive_mode;
left out, the server's default, auto: multi-question requests). Repacking keeps a
"serving" already in meta.json unless --serving is given.

The release must already hold aux_scorers.safetensors and spec.arch_extra.aux_exits;
the policy's exits must be those aux exits plus the main exit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CAL_KEYS = ("mean", "W", "mu", "sd", "w", "b")


def pack(release: Path, policy: dict, source: str | None = None, serving: str | None = None) -> dict:
    from safetensors.torch import save_file
    meta_path = release / "meta.json"
    meta = json.loads(meta_path.read_text())
    ae = meta["spec"].get("arch_extra") or {}
    exits = [int(x) for x in policy["exits"]]
    want = sorted(int(x) for x in ae.get("aux_exits") or []) + [int(ae.get("exit_layer") or 0)]
    if exits != want:
        raise SystemExit(f"policy exits {exits} != the release's aux exits + exit_layer {want}")
    if not (release / "aux_scorers.safetensors").exists():
        raise SystemExit(f"{release} has no aux_scorers.safetensors")
    flat = {}
    for L in exits[:-1]:
        cal = policy["cal"][L] if L in policy["cal"] else policy["cal"][str(L)]
        missing = [k for k in CAL_KEYS if k not in cal]
        if missing:
            raise SystemExit(f"exit {L}: calibrator lacks {missing}")
        for k in CAL_KEYS:
            flat[f"{L}.{k}"] = cal[k].detach().float().contiguous().cpu().reshape(cal[k].shape)
    save_file(flat, str(release / "adaptive_calibration.safetensors"))
    if serving is not None and serving not in ("auto", "on", "off"):
        raise SystemExit(f"--serving must be auto, on or off, not {serving!r}")
    serving = serving if serving is not None else (meta.get("adaptive") or {}).get("serving")
    meta["adaptive"] = {"exits": exits, "tau": float(policy["tau"]),
                        "calibration": "adaptive_calibration.safetensors",
                        "source": source or "tau and per-exit cal-4b tuned on DEV (policy.pt)"}
    if serving is not None:
        meta["adaptive"]["serving"] = serving
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    return meta["adaptive"]


def main() -> int:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--source", default=None)
    ap.add_argument("--serving", default=None, choices=["auto", "on", "off"])
    a = ap.parse_args()
    pol = torch.load(a.policy, map_location="cpu", weights_only=True)
    print(json.dumps(pack(Path(a.release), pol, a.source, a.serving)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
