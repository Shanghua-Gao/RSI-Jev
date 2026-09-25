"""Can a later position influence an earlier one?

Forward only, CPU fp32 is enough, seconds. Writes one JSON and changes nothing.

This is NOT a masking test, which is what it looks like at first. Under a causal
mask the decision position d attends only to positions <= d, so nothing appended
after d can reach it whether it is masked or not. What the appended arms detect
is **backward information flow**: a non-causal step in the chunked delta-rule
kernel (this model runs it on the reference path for 18 of its 24 layers,
chunk_size 64), the causal conv1d, or a sequence-wide normalisation.

Three conditions against an unpadded reference:

  append_masked     pads after the sequence, attention_mask 0. Must be ~0.
  append_unmasked   pads after the sequence, attention_mask 1. Must ALSO be ~0.
                    This is a second causality check, not a positive control --
                    a correct model ignores it for the same reason. If the two
                    append arms differ FROM EACH OTHER, the mask is being
                    applied where causality already made it unnecessary, which
                    is its own finding.
  prepend_real      real filler tokens BEFORE the sequence. Must move a lot.
                    THIS is the positive control: the tokens land where d can
                    see them. If it does not move, the readout is not reading
                    where we think and the whole run is void. Its magnitude is
                    not a clean "context effect" -- prepending also shifts every
                    real token's position -- but sensitivity is all it is for.

Verdict needs BOTH signatures for the serious branch: a difference at fp32 scale
(>=1e-4) AND ordering in the pad count. Rounding-scale noise that does not grow
with padding is the benign branch: bf16 GEMM reduction order follows tensor
shape, and the kernel pads to its own 64-grid before masking, so reassociation
can produce ~1e-6 even when everything is correct.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.arch import LogprobReadout                                   # noqa: E402
from rsijev.encode import EncodeConfig, encode_question, option_first_token_ids, iter_questions  # noqa: E402
from rsijev.targets import load_typed_decisions                          # noqa: E402

PAD_COUNTS = (0, 64, 256, 1024)


@torch.no_grad()
def logits_for(lm, tok, enc, opts, *, pad: int, mask_pads: bool, prepend: int,
               device, dtype):
    ids = list(enc["input_ids"])
    d = enc["decision_index"]
    filler = tok("the quick brown fox jumps over the lazy dog. " * 32,
                 add_special_tokens=False)["input_ids"]
    if prepend:
        ids = filler[:prepend] + ids
        d += prepend
    am = [1] * len(ids)
    if pad:
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ids = ids + [pad_id] * pad
        am = am + [1 if mask_pads is False else 0] * pad
    t = torch.tensor([ids], device=device)
    a = torch.tensor([am], device=device)
    out = lm(input_ids=t, attention_mask=a)
    row = out.logits[0, d].to(torch.float32)
    return row[torch.tensor(option_first_token_ids(tok, opts), device=device)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--out", default="padding_invariance.json")
    ap.add_argument("--questions", type=int, default=3)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, args.dtype)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()

    pairs = list(iter_questions(load_typed_decisions("test")))[: args.questions]
    rows = []
    for c, q in pairs:
        enc = encode_question(tok, c.state, q, EncodeConfig())
        ref = logits_for(lm, tok, enc, q.options, pad=0, mask_pads=True, prepend=0,
                         device=dev, dtype=dtype)
        for cond, kw in (("append_masked", dict(mask_pads=True, prepend=0)),
                         ("append_unmasked", dict(mask_pads=False, prepend=0))):
            for pad in PAD_COUNTS:
                got = logits_for(lm, tok, enc, q.options, pad=pad, device=dev,
                                 dtype=dtype, **kw)
                rows.append({"question": c.case_id, "condition": cond, "pad": pad,
                             "max_abs_logit_diff": float((ref - got).abs().max())})
        got = logits_for(lm, tok, enc, q.options, pad=0, mask_pads=True, prepend=64,
                         device=dev, dtype=dtype)
        rows.append({"question": c.case_id, "condition": "prepend_real", "pad": 64,
                     "max_abs_logit_diff": float((ref - got).abs().max())})

    def worst(cond):
        v = [r["max_abs_logit_diff"] for r in rows if r["condition"] == cond and r["pad"]]
        return max(v) if v else float("nan")

    def ordered(cond):
        by_q: dict[str, list[float]] = {}
        for r in rows:
            if r["condition"] == cond:
                by_q.setdefault(r["question"], []).append((r["pad"], r["max_abs_logit_diff"]))
        return all(all(x[1] <= y[1] + 1e-12 for x, y in zip(sorted(v), sorted(v)[1:]))
                   for v in by_q.values())

    wm, wu, wp = worst("append_masked"), worst("append_unmasked"), worst("prepend_real")
    if not (wp > 1e-2):
        verdict = "VOID: the positive control did not move; the readout is not where we think"
    elif max(wm, wu) >= 1e-4 and (ordered("append_masked") or ordered("append_unmasked")):
        verdict = "SERIOUS: backward information flow, scale and ordering both present"
    elif max(wm, wu) >= 1e-4:
        verdict = "SUSPECT: fp32-scale difference without ordering; investigate before trusting"
    else:
        verdict = "benign: rounding scale only, no backward flow"

    out = {"model": args.model, "device": dev, "dtype": args.dtype,
           "worst": {"append_masked": wm, "append_unmasked": wu, "prepend_real": wp},
           "append_arms_agree": abs(wm - wu) < 1e-6,
           "ordered_in_pad": {"append_masked": ordered("append_masked"),
                              "append_unmasked": ordered("append_unmasked")},
           "verdict": verdict, "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
