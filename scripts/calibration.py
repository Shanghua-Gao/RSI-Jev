"""Triage a queue of real tickets, then ask whether the confidence was honest.

    python scripts/calibration.py --model 2b --cases 100

Two things a decision model has to get right, shown together on real data:

1. **Throughput** — decisions per second, with nothing generated to parse.
2. **Calibration** — when it says 90%, is it right about 90% of the time? A
   confident wrong answer is worse than an unsure one, so the probability has to
   mean something. This bins every decision by confidence and reports the
   accuracy in each bin.

The questions come from `LocalLLaMA/typed-decisions`, which the models are never
trained on, so the accuracy below is genuinely out-of-sample.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

# The repository root must precede scripts/ on sys.path: scripts/serve.py
# would otherwise shadow the serve/ PACKAGE, and "from serve.infer import ..."
# fails with "serve is not a package". These scripts used to work only
# because load_release.py re-inserts the root when it is imported first.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPOS = {"0.8b": "shgao/rsi-jev-v1.0-qwen3.5-0.8b", "2b": "shgao/rsi-jev-v1.0-qwen3.5-2b"}
BINS = [(0.9, 1.01, "90-100%"), (0.7, 0.9, "70-90%"),
        (0.5, 0.7, "50-70%"), (0.0, 0.5, "under 50%")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="2b", choices=sorted(REPOS))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--device", default=None)
    ap.add_argument("--show", type=int, default=5, help="how many to print as they go")
    a = ap.parse_args()

    import torch
    from load_release import load_release
    from rsijev.contract import gold_label
    from rsijev.targets import load_typed_decisions
    from serve.infer import score_questions_cached
    from serve.wire import to_answer

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ckpt = a.ckpt
    if ckpt is None:
        from huggingface_hub import snapshot_download
        ckpt = snapshot_download(REPOS[a.model])
    model, tok, enc, meta = load_release(ckpt, device, infer_dtype=dtype)

    cases = load_typed_decisions("test")[:a.cases]
    total_q = sum(len(c.questions) for c in cases)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else device
    print(f"\n  {Path(ckpt).name} on {gpu}")
    print(f"  triaging {len(cases)} documents, {total_q} typed decisions, "
          f"none of it seen in training\n")

    hits = collections.Counter()
    seen = collections.Counter()
    right = wrong_confident = 0
    t0 = time.perf_counter()
    for n, c in enumerate(cases):
        qs = list(c.questions)
        preds, _ = score_questions_cached(model, tok, c.state, qs, enc, device=device)
        for q, p in zip(qs, preds):
            ans = to_answer(q, list(p.probs))
            pred = q.options[max(range(len(p.probs)), key=p.probs.__getitem__)]
            ok = pred == gold_label(q, c.gold[q.key])
            right += ok
            conf = max(p.probs)
            for lo, hi, label in BINS:
                if lo <= conf < hi:
                    seen[label] += 1
                    hits[label] += ok
                    if label == "90-100%" and not ok:
                        wrong_confident += 1
                    break
            if n < a.show:
                shown = (f"{ans['noul']:.2f} yes" if ans["type"] == "noul"
                         else f"{ans['choice']} {max(ans['probabilities'].values()):.2f}"
                         if ans["type"] == "choice"
                         else f"{ans['score']:.1f}/{len(ans['probabilities']) - 1}")
                print(f"    doc {n+1:03d}  {q.key:<14s} {shown:<18s} "
                      f"{'ok' if ok else 'MISS'}")
        if n < a.show:
            print()
    dt = time.perf_counter() - t0

    print(f"  {total_q} decisions in {dt:.1f}s — {total_q/dt:.0f} decisions/second, "
          f"{dt/total_q*1000:.1f} ms each")
    print(f"  0 tokens generated. Accuracy {right/total_q:.1%} "
          f"(always guessing the majority scores 51.9%)\n")

    print("  Is the confidence honest?\n")
    print(f"    {'model says':>12}   {'decisions':>9}   {'actually right':>14}")
    print(f"    {'-'*12}   {'-'*9}   {'-'*14}")
    for _, _, label in BINS:
        if seen[label]:
            print(f"    {label:>12}   {seen[label]:>9}   {hits[label]/seen[label]:>13.1%}")
    print(f"\n  Confidently wrong (said 90%+, was wrong): {wrong_confident} of {total_q} "
          f"({wrong_confident/total_q:.1%})")
    print("  Found one that looks unreasonable? "
          "https://github.com/Shanghua-Gao/RSI-Jev/issues\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
