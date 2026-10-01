"""Score a parent checkpoint on the depth-dial folds: the input of build_depthdial.py.

Needs a GPU and the parent checkpoint (for calA-ce: vis-ct-3, which is not published;
its per-cell table is data/depthdial_cells_vis-ct-3.json). The model is scored in the
evaluator's conditions (rsijev.calibrate.collect: eval mode, fp32, canonical option
order, calibration off), on the text path only.

For each of gen_measure.jsonl and gen_probe.jsonl in --folds (build_depthdial.py folds)
it writes <out>/gen_<fold>.preds.jsonl, one row per question:
    {"case_id", "k", "y" (true index), "p_raw": [...]}  (+ "p_shipped" when the checkpoint
                                                          carries a calibration)
build_depthdial.py reads p_raw of the measure fold. The collected logits are cached as
<out>/rows.gen_<fold>.pt and reused on a rerun.

    python scripts/depthdial_measure.py --ckpt PARENT_CKPT --folds FOLDS --out PREDS/vis-ct-3 [--limit n]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="parent release checkpoint dir")
    ap.add_argument("--folds", required=True, help="dir holding gen_measure.jsonl / gen_probe.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="dry run: 3 * limit cases per fold")
    a = ap.parse_args()

    import torch
    from load_release import load_release
    import rsijev.calibrate as C
    from rsijev.calibrate import CAL_BUFFERS, load_calibration
    from rsijev.contract import load_cases, gold_label

    DEV, LIM = a.device, a.limit
    O = Path(a.out); O.mkdir(parents=True, exist_ok=True)
    model, tok, enc, meta = load_release(a.ckpt, DEV, vision=False)
    spec = meta["spec"]; MO = max(spec["max_options"], 160)
    SHIPPED = None
    if (Path(a.ckpt) / "calibration.safetensors").exists():
        load_calibration(model, a.ckpt)
        SHIPPED = {"mode": model.cal_mode, **{k: getattr(model, k).detach().clone() for k in CAL_BUFFERS}}
    model.cal_mode = "none"

    def fits(q):
        return len(q.options) <= MO

    def cases(path, n=None):
        cs = [c for c in load_cases(str(path)) if all(fits(q) for q in c.questions)]
        return cs[:n] if n else cs

    def coll(pairs, tag):
        f = O / f"rows.{tag}.pt"
        if f.exists():
            return torch.load(f, weights_only=False)
        d = C.collect(model, tok, pairs, enc, max_options=MO, device=DEV, batch_size=32)
        d["q"] = [(c.case_id, q.key, q.mode, len(q.options), tuple(q.options), gold_label(q, c.gold[q.key]))
                  for c, _ in pairs for q in c.questions]
        d = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
        torch.save(d, f)
        log("collected", tag, len(d["q"]))
        return d

    @torch.no_grad()
    def lt_linear(st, d):
        for k in CAL_BUFFERS:
            getattr(model, k).copy_(st[k])
        model.cal_mode = st["mode"]
        lt = model.cal_log_temperature(d["z"].to(DEV).float(), d["h"].to(DEV).float(), d["mode"].to(DEV))
        model.cal_mode = "none"
        return lt

    @torch.no_grad()
    def probs_of(ltf, d):
        z = d["z"].to(DEV).float()
        if ltf is None:
            return torch.softmax(z, -1).cpu()
        return torch.softmax(z / torch.exp(ltf(d))[:, None], -1).cpu()

    def dump_gen(tag, d, variants):
        P = {v: probs_of(f, d).tolist() for v, f in variants.items()}
        with open(O / f"gen_{tag}.preds.jsonl", "w") as fh:
            for i, (cid, key, mode, k, opts, gold) in enumerate(d["q"]):
                fh.write(json.dumps({"case_id": cid, "k": k, "y": opts.index(gold),
                                     **{f"p_{v}": [round(x, 5) for x in P[v][i][:k]] for v in P}}) + "\n")

    F = Path(a.folds)
    GEN = {t: coll([(c, "gen") for c in cases(F / f"gen_{t}.jsonl", 3 * LIM if LIM else None)], "gen_" + t)
           for t in ("measure", "probe")}
    variants = {"raw": None, **({"shipped": lambda d_: lt_linear(SHIPPED, d_)} if SHIPPED else {})}
    for t, d in GEN.items():
        dump_gen(t, d, variants)
    log("DONE", O)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
