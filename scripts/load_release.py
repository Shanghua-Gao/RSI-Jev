"""Load an RSI-Jev release checkpoint and (optionally) verify it.

A checkpoint (written by run_arm_lib.save_release) holds the fine-tuned tower in
fp32 WITHOUT the embedding, which was frozen during training and is taken from
the public base model; the trained option scorer; and meta.json with the full
recipe. This file rebuilds the exact model that was scored.

    from load_release import load_release
    model, tok, enc, meta = load_release("path/to/ckpt", device="cuda")

    python scripts/load_release.py --ckpt DIR --verify [--record RUN.items.jsonl]

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

from rsijev.arch import ArchConfig, DecisionModel      # noqa: E402
from rsijev.encode import EncodeConfig                 # noqa: E402


def load_release(ckpt: str | Path, device: str = "cuda", infer_dtype=None):
    """`infer_dtype` casts the TOWER for inference only: bf16 is 3-6x faster and
    moved pooled top-1 by at most 0.003 over the full test set. The scorer always
    stays fp32 -- running it in reduced precision is the bug that cost this
    project a whole version. Evaluation leaves this None and gets fp32.
    """
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt = Path(ckpt)
    meta = json.loads((ckpt / "meta.json").read_text())
    spec = meta["spec"]
    tok = AutoTokenizer.from_pretrained(meta["base_model"])
    # Load straight into the precision the tower will run in, rather than fp32
    # then a cast: the fp32 checkpoint rounds to the target once either way, and
    # this never materialises a second copy. That matters on a laptop, and on a
    # unified-memory box holding two checkpoints at once. The base weights are
    # bf16 on the Hub, so fp32 is an exact upcast and bf16 is a no-op.
    lm = AutoModelForCausalLM.from_pretrained(meta["base_model"],
                                              dtype=infer_dtype or torch.float32)
    tower = getattr(lm, "model", lm)
    missing, unexpected = tower.load_state_dict(load_file(str(ckpt / "tower.safetensors")),
                                                strict=False)
    bad = [k for k in missing if "embed_tokens" not in k]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint does not match {meta['base_model']}: "
                           f"missing {bad[:5]}, unexpected {list(unexpected)[:5]}")
    cfg = getattr(lm.config, "text_config", None) or lm.config
    arch = ArchConfig(readout=spec["readout"], readout_layer=spec["readout_layer"],
                      max_options=spec["max_options"], freeze_base=True,
                      option_pool=spec["option_pool"], residual=spec["residual"],
                      logit_cap=spec.get("logit_cap"),
                      head_input_norm=spec.get("head_input_norm", False),
                      **dict(spec.get("arch_extra") or {}))
    model = DecisionModel(tower, cfg.hidden_size, arch).to(device)
    model.scorer.load_state_dict(load_file(str(ckpt / "scorer.safetensors")))
    # v2.0 onward a checkpoint may ship a fitted calibration (calibration.safetensors
    # + calibration.json, rsijev/calibrate.py). It is part of the released model, not
    # an extra: the forward pass divides each question's logits by one positive
    # temperature, so the answer is unchanged and it is still one forward pass. A
    # checkpoint without those files -- v1.0 -- loads exactly as it always did.
    if (ckpt / "calibration.safetensors").exists():
        from rsijev.calibrate import load_calibration
        meta["calibration"] = load_calibration(model, ckpt)
    else:
        meta["calibration"] = "none"
    model.scorer.to(torch.float32)          # never follows the tower down
    model.eval()
    enc = EncodeConfig(layout=spec["layout"], option_pool=spec["option_pool"],
                       option_order="canonical")
    return model, tok, enc, meta


def artifact_name(path) -> str:
    """The checkpoint's own name, never where it lived.

    A record written here ships inside the published checkpoint, so an absolute
    path would publish the machine it was trained on. v1.0's verify.json went out
    carrying a lab filesystem path before this existed.
    """
    return Path(path).name


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
    model, tok, enc, meta = load_release(args.ckpt, dev, infer_dtype=dt)
    print(f"loaded {args.ckpt}: base {meta['base_model']}, kernel "
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
