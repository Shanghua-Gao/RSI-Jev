"""Choose a multi-exit checkpoint's exit policy from per-exit dumps (v6.0-VL stages 5-6, v6.1-VL 4-5). CPU only.

The rules are in rsijev/exit_policy.py; the dumps come from scripts/dump_exits.py.

  seed        the seed-selection metric of each candidate seed (main-exit cal-4b ECE on
              its own DEV "tau" half) and the pick (lower; within .002 the first listed),
              plus each exit's DEV temperature (the package's main-exit cal_logT)
  policy      cascade selection: per-exit temperatures on half A of the policy development set,
              the binding servable policy on half B; with --eval-dump / --eval-vis-dump,
              the one read on the evaluation sets
  thresholds  effort=auto per-exit thresholds (tau16, tau20) on a held-out set and a
              shift-matched set scored with the Decision Index metric, selected on half A
              and confirmed on half B; with --eval, the one evaluation read of the
              confirmed winner (refused if that read already exists); with --single, one
              tau shared by both early exits, falling back to 0.95 when nothing is confirmed
  overlap     the development case ids found in training corpora (by case id, or by state
              and first question), for a refit on a checkpoint whose parents trained on them
  exclude     a copy of each dump without those case ids

  python scripts/select_exit_policy.py seed --dump s0=D/dumps/s0 --dump s1=D/dumps/s1 \\
      --out D/policy/seed.json
  python scripts/select_exit_policy.py policy --dev-dump D/dumps/s1/dev_dump.pt \\
      --text-dump D/dumps/pdtext.pt --vis-dump D/dumps/pdvis.pt --vis-cases D/policy-dev-v1/vision \\
      --eval-dump D/dumps/s1/eval_dump.pt --eval-vis-dump D/dumps/visexits.pt --out D/policy/cascade.json
  python scripts/select_exit_policy.py thresholds <the policy flags above, without --eval-*> \\
      --policy D/policy/cascade.json --heldout D/dumps/policy-dev-v2.pt D/policy-dev-v2 \\
      --heldout D/dumps/policy-dev-v2-fill.pt D/policy-dev-v2-fill --shift-matched D/dumps/policy-dev-sm.pt D/policy-dev-sm \\
      --bench-weights D/di_diag.json --chance KIT/data/index-0.2.json --out D/policy/auto.json
  python scripts/select_exit_policy.py thresholds ... --eval --eval-dump ... --eval-vis-dump ... --out D/policy/auto.json
  python scripts/select_exit_policy.py thresholds ... --single --out D/policy/single.json
  python scripts/select_exit_policy.py overlap --dev-dump D/dumps/dev_dump.pt --pd D/policy-dev-v1 ... \\
      --corpus D/domain-mix-2 --corpus D/image-corpus --out D/policy/excl.json
  python scripts/select_exit_policy.py exclude --ids D/policy/excl.json --pair D/dumps/pdtext.pt D/dumps-x/pdtext.pt ... \\
      --out D/policy/exclude.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rsijev import exit_policy as P  # noqa: E402

# The release bars the auto read is held to (the previous release's numbers, v5.0-VL):
# suite without open_jev_ood, held-out final set, final ECE, images at the last exit, MMLU-Pro.
BARS = {"b1x": ("suite_x_ood", ">=", .7625), "b2": ("final", ">=", .6889), "b3": ("final_ece", "<=", .078),
        "b4": ("vision_exit32", ">=", .821), "b7": ("mmlu_pro", ">=", .429)}


def _vis_case_files(d: str) -> list:
    root = Path(d)
    man = root / "manifest.json"
    if man.exists():
        return [root / b["file"] for b in json.loads(man.read_text())["benchmarks"].values()]
    return sorted(root.glob("probe_*.jsonl"))


def _v1(a):
    return P.load_v1(a.dev_dump, a.text_dump, a.vis_dump, _vis_case_files(a.vis_cases))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["seed", "policy", "thresholds", "overlap", "exclude"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", action="append", default=[], help="seed: NAME=DIR holding dev_dump.pt")
    ap.add_argument("--dev-dump")
    ap.add_argument("--text-dump", help="the policy development set's text dump (dump_exits.py cases)")
    ap.add_argument("--vis-dump", help="its image dump (dump_exits.py vision)")
    ap.add_argument("--vis-cases", help="the image dump's case directory (probe_*.jsonl, dump order)")
    ap.add_argument("--eval-dump")
    ap.add_argument("--eval-vis-dump")
    ap.add_argument("--policy", help="thresholds: the cascade selection's output (its temperatures must be reproduced)")
    ap.add_argument("--heldout", nargs=2, action="append", default=[], metavar=("DUMP", "PD"))
    ap.add_argument("--shift-matched", nargs=2, action="append", default=[], metavar=("DUMP", "PD"))
    ap.add_argument("--drop-slugs", default="home_sim,apibank",
                    help="held-out slugs replaced in the shift-matched set by its own renders")
    ap.add_argument("--bench-weights", help="per-benchmark diagnostic JSON (exit_policy.bench_weights)")
    ap.add_argument("--chance", help="the Decision Index index file with per-benchmark chance skill")
    ap.add_argument("--eval", action="store_true", help="thresholds: the one evaluation read")
    ap.add_argument("--single", action="store_true", help="thresholds: one tau shared by both early exits")
    ap.add_argument("--pd", action="append", default=[], help="overlap: a development set directory (*.jsonl)")
    ap.add_argument("--corpus", action="append", default=[], help="overlap: a training corpus directory (*.jsonl)")
    ap.add_argument("--ids", help="exclude: the overlap step's output")
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("IN", "OUT"),
                    help="exclude: a dump and where to write it without the excluded case ids")
    a = ap.parse_args()
    torch.set_num_threads(8)
    out = Path(a.out)

    if a.step == "seed":
        m = {}
        for s in a.dump:
            name, path = s.split("=", 1)
            m[name] = P.seed_metric(P.load(Path(path) / "dev_dump.pt"))
        res = {"metrics": m, "picked": P.pick_seed(m)}
        out.write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps({k: v["sel_metric"] for k, v in m.items()}), "picked", res["picked"])
        return 0

    if a.step == "overlap":
        own = list(P.load(a.dev_dump)["case_id"]) if a.dev_dump else []
        dev = [r for d in a.pd for r in P.jsonl_rows(d, skip=())]
        hit = P.overlap_case_ids(dev, (r for c in a.corpus for r in P.jsonl_rows(c)), own)
        where = {r["case_id"]: Path(d).name for d in a.pd for r in P.jsonl_rows(d, skip=())}
        per = {}
        for c in hit:
            k = where.get(c, "own DEV dump")
            per[k] = per.get(k, 0) + 1
        out.write_text(json.dumps({"excluded_case_ids": sorted(hit), "per_set": per}, indent=1))
        print("excluded case ids", len(hit), json.dumps(per))
        return 0

    if a.step == "exclude":
        ex = json.loads(Path(a.ids).read_text())["excluded_case_ids"]
        rep = {}
        for src, dst in a.pair:
            d = P.load(src)
            kept = P.drop_case_ids(d, ex)
            torch.save(kept, dst)
            n0 = len(d["case_id"]) if "case_id" in d else None
            rep[Path(src).name] = {"rows": n0, "kept": len(kept["case_id"]) if n0 is not None else None}
        out.write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep))
        return 0

    v1 = _v1(a)
    if a.step == "policy":
        res = P.select_cascade(v1, a.eval_dump, a.eval_vis_dump)
        out.write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps({"T": res["T"], "binding": res["binding"], "binding_dev": res["dev"][res["binding"]],
                          "winner_any": res["winner_any"]}))
        return 0

    T = P.fit_temperatures(v1)
    if a.policy:
        want = {int(k): float(v) for k, v in json.loads(Path(a.policy).read_text())["T"].items()}
        assert {L: T[L] for L in want} == want, (T, want)
    chance = {int(k): v["chance"] for k, v in json.loads(Path(a.chance).read_text())["chance"].items()}
    th = P.Thresholds(v1, T, P.bench_weights(json.loads(Path(a.bench_weights).read_text())), chance, single=a.single)
    if not a.eval:
        held = th.load_di([d for d, _ in a.heldout], [p for _, p in a.heldout])
        sm = th.load_di([d for d, _ in a.heldout + a.shift_matched], [p for _, p in a.heldout + a.shift_matched],
                        drop_slugs=tuple(x for x in a.drop_slugs.split(",") if x))
        res = th.select(held, sm)
        out.write_text(json.dumps(res, indent=1))
        print(json.dumps(res["selection"]))
        return 0
    sel = json.loads(out.read_text())["selection"]
    if not sel["CONFIRMED"]:
        raise SystemExit("the evaluation read is only for a CONFIRMED winner")
    eo = out.with_name(out.stem + ".eval.json")
    if eo.exists():
        raise SystemExit(f"{eo.name} exists: the evaluation is read once")
    ev = th.eval_read(sel, a.eval_dump, a.eval_vis_dump)
    w = ev[sel["winner"]]
    bars = {k: (w[f] >= v if op == ">=" else w[f] <= v) for k, (f, op, v) in BARS.items()}
    eo.write_text(json.dumps({"winner": sel["winner"], "tau": sel["tau"], "eval": ev, "bars": bars,
                              "all_text_bars_pass": all(bars.values())}, indent=1))
    print(json.dumps(ev[sel["winner"]]), "bars", json.dumps(bars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
