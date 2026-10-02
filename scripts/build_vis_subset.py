"""v5.0-VL image stage (vis-v4k): the new-data half of the corpus, one file per vision source.

The parent (b4-exit20, text only) has never seen an image, so every vision root is new to it.
The stage has a fixed question budget (steps x batch, about half text replay), so each source
gets a capped, seeded row sample (whole rows: a row's questions stay together). With --cap the
per-weight cap is fixed; without it the cap is water-filled until the total reaches --target.

Sources: vision_v1 (12 sources), vision_v2 (IconQA, Visual7W, TextVQA, OCR-VQA), vision_v3
(6 generators) and four generated roots: vision_spatial, vision_joint, vision_recast,
vision_sokoban. Gold is kept as built: one-hot everywhere except vis6_sokoban, whose soft label
spreads over equally optimal first moves (a designed probability, not annotator disagreement).

vis-v4k's image half (data/v5.0-vl_recipe/README.md):
    python scripts/build_vis_subset.py --root D --cap 600 \
        --weight vis3_change_detect=1.32 ... --weight vis2_iconqa=1 --weight vis5_abstain=3.96 --out D/vis_new
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SOURCES = {
    "vision_v1": ["vis_ai2d", "vis_aokvqa", "vis_chartqa", "vis_docvqa", "vis_hateful_memes",
                  "vis_nlvr2", "vis_rvlcdip", "vis_scienceqa", "vis_screen2words", "vis_tallyqa",
                  "vis_tqa", "vis_vqav2"],
    "vision_v2": ["vis2_iconqa", "vis2_visual7w", "vis2_textvqa", "vis2_ocrvqa"],
    "vision_v3": ["vis3_change_detect", "vis3_grid_games", "vis3_line_rule", "vis3_receipt_math",
                  "vis3_severity", "vis3_ui_state"],
    "vision_spatial": ["vis4_fp_obstacle", "vis4_grasp"],
    "vision_joint": ["vis5_record", "vis5_pair", "vis5_abstain"],
    "vision_recast": ["vis5_rc_count", "vis5_rc_colour"],
    "vision_sokoban": ["vis6_sokoban"],
}
WEIGHT = {"vis2_iconqa": 2.0}
SOFT_OK = {"vis6_sokoban"}


def nq(r):
    return len(r["questions"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the data directory holding the vision roots")
    ap.add_argument("--target", type=int, default=24000, help="vision questions in total")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, default=0.0,
                    help="fixed per-weight cap (skips the water-fill and --target); rows are seeded prefixes, so a "
                         "lower cap gives a subset of a higher cap's rows")
    ap.add_argument("--weight", action="append", default=[], metavar="SRC=W",
                    help="override/add a per-source weight (repeatable)")
    a = ap.parse_args()
    for kv in a.weight:
        k, v = kv.split("=")
        WEIGHT[k] = float(v)
    rows = {}
    for root, srcs in SOURCES.items():
        for s in srcs:
            f = Path(a.root) / root / f"{s}.jsonl"
            rs = [json.loads(l) for l in open(f) if l.strip()]
            for r in rs:
                for k, g in r["gold"].items():
                    soft = max(g) < 1 or sum(1 for x in g if x > 0) > 1
                    if soft and s not in SOFT_OK:
                        raise SystemExit(f"{s}: soft gold on {r['case_id']}/{k}; harden it first")
            random.Random(f"{a.seed}:{s}").shuffle(rs)
            rows[s] = (root, rs)
    total = {s: sum(map(nq, rs)) for s, (_, rs) in rows.items()}
    if a.cap:
        a.target = int(sum(min(total[s], a.cap * WEIGHT.get(s, 1.0)) for s in total))
    if sum(total.values()) < a.target:
        raise SystemExit(f"only {sum(total.values())} vision questions < target {a.target}")
    lo, hi = 1.0, float(max(total.values()))
    for _ in range(60):                              # water-fill the per-weight cap
        cap = (lo + hi) / 2
        got = sum(min(total[s], cap * WEIGHT.get(s, 1.0)) for s in total)
        lo, hi = (cap, hi) if got < a.target else (lo, cap)
    cap = a.cap or hi
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    man = {"target": a.target, "cap_per_weight": round(cap, 1), "seed": a.seed,
           "weights": {s: WEIGHT[s] for s in total if s in WEIGHT}, "files": {}}
    for s, (root, rs) in rows.items():
        lim = cap * WEIGHT.get(s, 1.0)
        keep, q = [], 0
        for r in rs:
            if q >= lim:
                break
            keep.append(r); q += nq(r)
        body = "".join(json.dumps(r) + "\n" for r in keep)
        (out / f"{s}.jsonl").write_text(body)
        man["files"][s] = {"root": root, "rows": len(keep), "questions": q,
                           "of_questions": total[s], "sha256": hashlib.sha256(body.encode()).hexdigest()}
    man["questions"] = sum(v["questions"] for v in man["files"].values())
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    for s, v in man["files"].items():
        print(f"{s:22s} {v['questions']:6d} / {v['of_questions']:6d} q ({v['rows']} rows)")
    print(f"total {man['questions']} vision questions, cap {cap:.0f}/weight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
