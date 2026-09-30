"""Padding on a mixed-length workload: 32 questions with 2-40 options and short or long
instructions, on the 80- and 1,052-token documents.

    python docs/assets/bench_mixed_gb10.py --ckpt DIR --out mixed.json

For each switch setting (rows sorted by length, option slots trimmed) it reports the
served scorer's p50, the padded and real tokens the forward passes carry, the padded
and real option slots the scorer carries, and the answers against the default
setting (argmax agreement, max |dp|).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_gb10 import DOCS  # noqa: E402

WORDS = ("refund billing late order charge account shipping damaged policy customer agent "
         "escalate review urgent invoice tracking warranty return replace cancel").split()


def mixed_questions(seed: int = 0, n: int = 32) -> dict:
    rng = random.Random(seed)
    out = {}
    for i in range(n):
        k = rng.choice([2, 3, 4, 6, 10, 20, 40])
        ins = " ".join(rng.choice(WORDS) for _ in range(rng.choice([4, 12, 60]))).capitalize() + "?"
        kind = "noul" if k == 2 and rng.random() < 0.5 else rng.choice(["choice", "choice", "score"])
        if kind == "noul":
            out[f"m{i}"] = {"type": "noul", "instructions": ins}
        elif kind == "score":
            out[f"m{i}"] = {"type": "score", "instructions": ins,
                            "criteria": [f"Level {j}: " + " ".join(rng.choice(WORDS) for _ in range(rng.choice([1, 6])))
                                         for j in range(k)]}
        else:
            out[f"m{i}"] = {"type": "choice", "instructions": ins,
                            "criteria": {f"opt_{j}": " ".join(rng.choice(WORDS) for _ in range(rng.choice([1, 5, 15])))
                                         for j in range(k)}}
    return out


def padding(plan, batch_size, sort, trim, max_options):
    from serve.infer import row_order
    rows = plan["encoded"]
    npfx = len(plan["prefix"]) if plan["path"] in ("cached", "doc") else 0
    lens = [len(e["input_ids"]) - npfx for e in rows]
    ks = [len(e["option_index"]) for e in rows]
    tok_pad = slot_pad = 0
    for idx in row_order(lens, batch_size, sort):
        tok_pad += len(idx) * max(lens[i] for i in idx)
        slot_pad += len(idx) * (max(ks[i] for i in idx) if trim else max_options)
    return {"tokens_real": sum(lens), "tokens_padded": tok_pad,
            "slots_real": sum(ks), "slots_padded": slot_pad, "path": plan["path"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=20)
    a = ap.parse_args()
    import torch
    from serve.infer import plan_request, score_planned
    from serve.server import load_for_serving
    from serve.wire import parse_questions
    s = load_for_serving(a.ckpt, device="cuda", dtype="bf16")
    spec = s.meta["spec"]
    qs = parse_questions(mixed_questions())
    mo = max(spec["max_options"], max(len(q.options) for q in qs))
    res = {"questions": len(qs), "options": [len(q.options) for q in qs], "cells": {}}
    for doc, text in DOCS.items():
        plan = plan_request(s.tok, text, qs, s.enc)
        ref = None
        for sort, trim in ((False, False), (True, False), (False, True), (True, True)):
            def once():
                return score_planned(s.model, s.tok, plan, max_options=mo, device="cuda",
                                     batch_size=16, sort=sort, trim_options=trim)
            for _ in range(3):
                once()
            xs = []
            for _ in range(a.repeat):
                torch.cuda.synchronize(); t = time.perf_counter()
                preds, _ = once()
                torch.cuda.synchronize(); xs.append((time.perf_counter() - t) * 1000)
            probs = [list(p.probs) for p in preds]
            if ref is None:
                ref = probs
            agree = sum(max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__)
                        for x, y in zip(probs, ref))
            dp = max(max(abs(u - v) for u, v in zip(x, y)) for x, y in zip(probs, ref))
            cell = {"p50_ms": round(st.median(xs), 2), **padding(plan, 16, sort, trim, mo),
                    "argmax_vs_default": f"{agree}/{len(qs)}", "max_abs_dp_vs_default": dp}
            name = f"{doc}|sort={int(sort)}|trim={int(trim)}"
            res["cells"][name] = cell
            print(name, cell, flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
