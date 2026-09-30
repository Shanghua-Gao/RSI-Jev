"""Load an RSI-Jev release checkpoint and (optionally) verify it.

A checkpoint (written by run_arm_lib.save_release) holds the fine-tuned tower in
fp32 WITHOUT the embedding, which was frozen during training and is taken from
the public base model; the trained option scorer; and meta.json with the full
recipe. This file rebuilds the exact model that was scored.

    from load_release import load_release
    model, tok, enc, meta = load_release("path/to/ckpt", device="cuda")

    python scripts/load_release.py --ckpt DIR --verify [--record RUN.items.jsonl]

--ckpt also takes a Hugging Face repo id or an alias such as v3.0-2b.

--verify re-scores the full typed-decisions test set (canonical and reversed
order) and MMLU-Pro 1k from the checkpoint. With --record, it also compares
each per-question prediction with the training run's own items file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The loader lives in the serve package so that an installed `rsi-jev` has it;
# this script keeps the import path every other script and published checkpoint
# card uses, and the --verify entry point.
from rsijev.encode import EncodeConfig                                  # noqa: E402
from serve.release import artifact_name, load_release, resolve_ckpt  # noqa: E402,F401


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--record", default=None, help="the training run's .items.jsonl")
    ap.add_argument("--out", default=None, help="write verification JSON here")
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp32"],
                    help="tower precision for inference; the scorer is always fp32. "
                         "--verify ignores this and reports fp32 numbers.")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(args.dtype)
    model, tok, enc, meta = load_release(resolve_ckpt(args.ckpt), dev, infer_dtype=dt)
    print(f"loaded {args.ckpt}: base {meta['base_model']}, trained with "
          f"{meta['linear_attn_kernel']}, tower {args.dtype or 'fp32'}, "
          f"calibration {meta['calibration']}")
    if not args.verify:
        return 0
    from rsijev.contract import gold_label
    from rsijev.evaluate import predict
    from rsijev.targets import load_mmlu_pro_1k, load_typed_decisions
    spec = meta["spec"]
    targets = {"typed_decisions": load_typed_decisions("test"), "mmlu_pro_1k": load_mmlu_pro_1k()}
    rec = None
    if args.record:
        rec = {(r["target"], r["option_order"], r["case_id"], r["key"]): r["pred"]
               for r in map(json.loads, open(args.record))
               if r["role"] == "candidate"}
    result = {}
    for tname, cases in targets.items():
        for order in (["canonical", "reversed"] if tname == "typed_decisions" else ["canonical"]):
            e = EncodeConfig(layout=spec["layout"], option_pool=spec["option_pool"],
                             option_order=order)
            preds = predict(model, tok, cases, e, max_options=spec["max_options"],
                            device=dev, batch_size=spec["eval_batch_size"])
            hits, agree, n = 0, 0, 0
            for c, q, p in preds:
                # Exactly item_rows' definitions, so the numbers are the records' numbers.
                pred = q.options[max(range(len(p.probs)), key=p.probs.__getitem__)]
                hits += int(pred == gold_label(q, c.gold[q.key]))
                n += 1
                if rec is not None:
                    agree += int(rec.get((tname, order, c.case_id, q.key)) == pred)
            result[f"{tname}/{order}"] = {"pooled_top1": round(hits / n, 4), "n": n,
                                          **({"agreement_with_record": round(agree / n, 4)}
                                             if rec is not None else {})}
            print(f"  {tname:16s} {order:9s} pooled top-1 {hits / n:.4f}"
                  + (f"  agreement with training-run record {agree / n:.4f}" if rec is not None else ""))
    if args.out:
        Path(args.out).write_text(json.dumps({"ckpt": artifact_name(args.ckpt), "meta": meta,
                                              "verify": result}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
