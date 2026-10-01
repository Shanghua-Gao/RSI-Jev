"""Depth-dial calibration data for calA-ce: soft targets from a parent's measured per-depth accuracy.

SYNTHETIC DATA, made by scripts/depthdial.py (own templates; the task families overlap
BIG-Bench Hard's web_of_lies, tracking_shuffled_objects and logical_deduction, which
the release card discloses). Nothing here downloads anything.

Three steps. Only step 2 needs a GPU and the parent checkpoint.

  1. folds    write the three evaluation folds (FOLDS below) into --folds DIR:
              gen_measure.jsonl (accuracy per cell), gen_dev.jsonl, gen_probe.jsonl.
  2. measure  score the parent on gen_measure / gen_probe with scripts/depthdial_measure.py,
              which writes <preds>/<parent>/gen_{measure,probe}.preds.jsonl (p_raw).
  3. build    per cell (gen, depth, k) of the MEASURE fold: n, acc, mean top conf,
              AUROC(conf -> correct), chance 1/k and a band:
                solve   acc >= .90                              -> one-hot target
                near    .50 <= acc < .90 and above chance       -> gold gets acc, the rest share 1 - acc
                mid     above chance but acc < .50              -> same soft rule
                beyond  not above chance (acc <= 1/k + 2 SE)    -> uniform target
              Training items come from a separate fold ("train", seed 2), minus any item
              whose exact text (state + question) is in the measure, dev or probe fold, so
              the accuracy that sets each target is out of fold. At most --n-train items per
              cell. Sources are named dd_<gen>_<band>. Probes (probe fold, TRUE one-hot gold,
              eval only, never trained on): probes/dd_probe_atchance.jsonl (beyond cells)
              and probes/dd_probe_near.jsonl (near cells).

    python scripts/build_depthdial.py folds  --folds FOLDS
    python scripts/depthdial_measure.py --ckpt PARENT_CKPT --folds FOLDS --out PREDS/vis-ct-3
    python scripts/build_depthdial.py build  --folds FOLDS --preds PREDS --parent vis-ct-3 \\
        --out DD/vis-ct-3 --n-train 300
    python scripts/build_depthdial.py report --folds FOLDS --preds PREDS --models a,b

calA-ce was built from vis-ct-3 with --n-train 300: 8 sources, 8,492 questions
(dd_lies_solve 300, dd_lies_near 300, dd_lies_beyond 2,192, dd_swap_near 900,
dd_swap_mid 1,200, dd_swap_beyond 2,100, dd_order_near 600, dd_order_mid 900),
514 training items dropped for text overlap with the evaluation folds, probes near 1,200
and at-chance 3,000.

The vis-ct-3 checkpoint is not published, so step 2 cannot be rerun from this repo for
that parent. `build --cells FILE` takes the per-cell table instead of the predictions:
FILE is either a manifest.json written by an earlier `build` or a JSON object
{"gen|depth|k": {"acc": ..., "band": ..., ...}}. vis-ct-3's table is
data/depthdial_cells_vis-ct-3.json; with it the build is byte-identical to the calA-ce
input (checked against the files training read, manifest included):

    python scripts/build_depthdial.py build --folds FOLDS --cells data/depthdial_cells_vis-ct-3.json \
        --parent vis-ct-3 --out DD/vis-ct-3 --n-train 300
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depthdial as DD   # noqa: E402

# (fold, n per cell, seed): the evaluation folds calA-ce's targets and overlap guard used.
FOLDS = (("measure", 200, 1), ("dev", 100, 1), ("probe", 200, 1))
TRAIN_SEED = 2


def auroc(conf, corr):
    pos = [c for c, y in zip(conf, corr) if y]; ns = sorted(c for c, y in zip(conf, corr) if not y)
    if not pos or not ns:
        return None
    s = sum(bisect.bisect_left(ns, p) + 0.5 * (bisect.bisect_right(ns, p) - bisect.bisect_left(ns, p)) for p in pos)
    return round(s / (len(pos) * len(ns)), 4)


def meta_of(folds: Path, fold: str):
    return {json.loads(l)["case_id"]: json.loads(l)["meta"] for l in open(folds / f"gen_{fold}.jsonl")}


def band_of(acc: float, k: int, n: int) -> str:
    ch = 1 / k
    above = acc > ch + 2 * math.sqrt(ch * (1 - ch) / n)
    return "solve" if acc >= .9 else "near" if (acc >= .5 and above) else "mid" if above else "beyond"


def cells(pred_file, folds: Path, fold="measure", key="p_raw"):
    M = meta_of(folds, fold)
    agg = defaultdict(lambda: {"conf": [], "corr": []})
    for l in open(pred_file):
        r = json.loads(l)
        m = M[r["case_id"]]
        p = r[key]
        i = max(range(len(p)), key=p.__getitem__)
        c = agg[(m["gen"], m["depth"], m["k"])]
        c["conf"].append(max(p)); c["corr"].append(i == r["y"])
    out = {}
    for (g, d, k), c in sorted(agg.items()):
        n = len(c["corr"]); acc = sum(c["corr"]) / n; ch = 1 / k
        out[f"{g}|{d}|{k}"] = {"gen": g, "depth": d, "k": k, "n": n, "acc": round(acc, 4), "conf": round(sum(c["conf"]) / n, 4),
                               "chance": round(ch, 4), "auroc": auroc(c["conf"], c["corr"]), "band": band_of(acc, k, n)}
    return out


def summary(cs, bands):
    sel = [c for c in cs.values() if c["band"] in bands]
    if not sel:
        return None
    n = sum(c["n"] for c in sel)
    return {"n": n, "acc": round(sum(c["acc"] * c["n"] for c in sel) / n, 4), "conf": round(sum(c["conf"] * c["n"] for c in sel) / n, 4),
            "chance": round(sum(c["chance"] * c["n"] for c in sel) / n, 4)}


def target(band: str, acc: float, k: int, y: int) -> list[float]:
    """The training target of one item: one-hot (solve), uniform (beyond), else gold = acc."""
    if band == "solve":
        g = [1.0 if j == y else 0.0 for j in range(k)]
    elif band == "beyond":
        g = [1.0 / k] * k
    else:
        acc = max(acc, 1.0 / k)
        g = [acc if j == y else (1 - acc) / (k - 1) for j in range(k)]
    s = sum(g); g = [round(x / s, 6) for x in g]; g[y] = round(1 - sum(g[:y]) - sum(g[y + 1:]), 6)
    return g


def load_cells(path: Path) -> dict:
    j = json.loads(Path(path).read_text())
    return j["cells"] if "cells" in j else j


def build(cs: dict, parent: str, folds: Path, out: Path, n_train: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    taken = set()
    for fold in ("measure", "dev", "probe"):
        for l in open(folds / f"gen_{fold}.jsonl"):
            r = json.loads(l); taken.add(r["state"] + "||" + r["questions"][0]["instructions"])
    train = DD.make("train", n_train + 50, seed=TRAIN_SEED)
    by_src, dropped = defaultdict(list), 0
    per_cell = defaultdict(int)
    for it in train:
        m = it["meta"]; c = cs[f"{m['gen']}|{m['depth']}|{m['k']}"]
        if it["state"] + "||" + it["questions"][0]["instructions"] in taken:
            dropped += 1; continue
        key = (m["gen"], m["depth"], m["k"])
        if per_cell[key] >= n_train:
            continue
        per_cell[key] += 1
        g = target(c["band"], c["acc"], m["k"], m["y"])
        it = {**it, "source": f"dd_{m['gen']}_{c['band']}", "gold": {"answer": g},
              "meta": {**m, "band": c["band"], "parent": parent, "parent_acc": c["acc"], "target": "hard" if c["band"] == "solve" else "soft"}}
        by_src[it["source"]].append(it)
    for s, L in by_src.items():
        with open(out / f"{s}.jsonl", "w") as fh:
            for it in L:
                fh.write(json.dumps(it) + "\n")
    # probes (true gold): at-chance = beyond cells, near = near cells; from the probe fold
    pr = defaultdict(list)
    for l in open(folds / "gen_probe.jsonl"):
        it = json.loads(l); m = it["meta"]; c = cs[f"{m['gen']}|{m['depth']}|{m['k']}"]
        it["meta"] = {**m, "band": c["band"], "parent": parent}
        if c["band"] in ("beyond", "near"):
            pr["atchance" if c["band"] == "beyond" else "near"].append(it)
    (out / "probes").mkdir(exist_ok=True)
    for b, L in pr.items():
        with open(out / "probes" / f"dd_probe_{b}.jsonl", "w") as fh:
            for it in L:
                fh.write(json.dumps(it) + "\n")
    man = {"parent": parent, "cells": cs, "sources": {s: len(L) for s, L in by_src.items()},
           "probes": {b: len(L) for b, L in pr.items()}, "dropped_overlap": dropped,
           "summary_parent": {b: summary(cs, (b,)) for b in ("solve", "near", "mid", "beyond")}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    return man


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["folds", "report", "build"])
    ap.add_argument("--folds", required=True, help="dir holding gen_{measure,dev,probe}.jsonl")
    ap.add_argument("--preds", default="", help="dir of <model>/gen_measure.preds.jsonl (depthdial_measure.py)")
    ap.add_argument("--cells", default="", help="per-cell table instead of --preds (build only)")
    ap.add_argument("--models", default="")
    ap.add_argument("--keys", default="p_raw")
    ap.add_argument("--parent", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--n-train", type=int, default=300)
    a = ap.parse_args()
    F = Path(a.folds)
    if a.cmd == "folds":
        for fold, n, seed in FOLDS:
            items = DD.make(fold, n, seed)
            DD.write(items, F / f"gen_{fold}.jsonl")
            print(fold, n, seed, len(items), "->", F / f"gen_{fold}.jsonl")
        return 0
    P = Path(a.preds) if a.preds else None
    if a.cmd == "report":
        rep = {}
        for m in a.models.split(","):
            for key in a.keys.split(","):
                f = P / m / "gen_measure.preds.jsonl"
                if not f.exists() or key not in open(f).readline():
                    continue
                rep[f"{m}:{key}"] = cells(f, F, "measure", key)
        print(json.dumps(rep))
        return 0
    if not a.out or not a.parent or not (a.cells or a.preds):
        ap.error("build needs --out, --parent and one of --preds / --cells")
    cs = load_cells(Path(a.cells)) if a.cells else cells(P / a.parent / "gen_measure.preds.jsonl", F, "measure")
    man = build(cs, a.parent, F, Path(a.out), a.n_train)
    print(json.dumps({k: man[k] for k in ("sources", "probes", "dropped_overlap", "summary_parent")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
