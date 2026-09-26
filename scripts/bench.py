"""How fast is it on YOUR machine?

    python scripts/bench.py                      # 0.8B, downloads on first run
    python scripts/bench.py --model v2.0-2b

Times the served path — the same code the server calls — across document sizes
and question counts, because together those decide the cost: the document is
encoded once and every question continues from its cache, so latency grows far
more slowly than either does.

The second table is the one to quote. Fitting total time against question count
separates the fixed cost of reading the document from the marginal cost of one
more decision, which a single averaged per-decision figure hides -- and it does
that per document length, because the fixed cost is the part that grows with the
document while the marginal cost barely moves.

Reports the honest thing, not the flattering one: a median over repeats, on
named hardware, with the precision stated.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

# The repository root must precede scripts/ on sys.path: scripts/serve.py
# would otherwise shadow the serve/ PACKAGE, and "from serve.infer import ..."
# fails with "serve is not a package". These scripts used to work only
# because load_release.py re-inserts the root when it is imported first.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keyed by release, because a key that means "the 2B one" stops being useful the
# moment there are two of them.
REPOS = {"v2.0-2b": "shgao/rsi-jev-v2.0-qwen3.5-2b",
         "v1.0-2b": "shgao/rsi-jev-v1.0-qwen3.5-2b",
         "v1.0-0.8b": "shgao/rsi-jev-v1.0-qwen3.5-0.8b"}
QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "escalate": {"type": "noul", "instructions": "Should a human agent take this now?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, refunds and duplicate charges",
                                "technical": "Bugs, outages and login problems",
                                "shipping": "Delivery, tracking and lost parcels",
                                "retention": "Cancellations and win-back offers"}},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                "criteria": ["Routine", "Elevated", "Urgent", "Critical"]},
}
TICKET = ("Order #48213, placed 3 March, two-day shipping. Customer wrote on 11 March: "
          "\"This is the third time I'm writing. I was charged twice for the same order "
          "and nobody has replied. I need the duplicate refunded today or I'm disputing "
          "it with my bank.\" Account: 4 years old, 26 orders, no prior disputes.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v2.0-2b", choices=sorted(REPOS))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp32"])
    ap.add_argument("--repeat", type=int, default=10)
    a = ap.parse_args()

    import torch
    from load_release import load_release
    from serve.infer import score_questions, score_questions_cached
    from serve.wire import parse_questions

    device = a.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None)
                          and torch.backends.mps.is_available() else "cpu")
    dtype_name = a.dtype or ("bf16" if device == "cuda" else "fp32")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]

    ckpt = a.ckpt
    if ckpt is None:
        from huggingface_hub import snapshot_download
        ckpt = snapshot_download(REPOS[a.model])
    model, tok, enc, meta = load_release(ckpt, device, infer_dtype=dtype)

    name = (torch.cuda.get_device_name(0) if device == "cuda" else device)
    qs = parse_questions(QUESTIONS)
    print(f"\n  {Path(ckpt).name}  on {name}  ({dtype_name})")
    print(f"  {len(qs)} questions per document, median of {a.repeat}, no text generated\n")
    print(f"  {'document':>12}   {'per document':>13}   {'per decision':>13}   {'decisions/s':>11}")
    print(f"  {'-'*12}   {'-'*13}   {'-'*13}   {'-'*11}")

    for target, label in ((0, "short"), (400, "medium"), (1000, "long")):
        text = TICKET
        while len(tok(text, add_special_tokens=False)["input_ids"]) < target:
            text = TICKET + "\n\n" + text
        ntok = len(tok(text, add_special_tokens=False)["input_ids"])

        def once():
            score_questions_cached(model, tok, text, qs, enc, device=device)

        for _ in range(2):
            once()
        if device == "cuda":
            torch.cuda.synchronize()
        xs = []
        for _ in range(a.repeat):
            t = time.perf_counter()
            once()
            if device == "cuda":
                torch.cuda.synchronize()
            xs.append(time.perf_counter() - t)
        med = st.median(xs)
        print(f"  {ntok:>7} tok   {med*1000:>10.0f} ms   {med/len(qs)*1000:>10.1f} ms   "
              f"{len(qs)/med:>11.0f}")

    # How much does one more question cost? Fitting total time against question
    # count splits the fixed cost of reading the document from the marginal cost
    # of a decision. Averaging instead would attribute part of the read to every
    # question, making a 1-question request look cheap and a 32-question one
    # look expensive. Done per document length, because only the fixed half
    # depends on the document.
    counts = [1, 2, 4, 8, 16, 32]
    # The same four questions over and over under distinct keys: what is being
    # varied is how many decisions are asked for, not which.
    pool = [(f"{key}_{i}", q) for i in range(max(counts) // len(QUESTIONS) + 1)
            for key, q in QUESTIONS.items()]

    print(f"\n  cost of a decision, questions {counts[0]}-{counts[-1]}, "
          f"median of {a.repeat}\n")
    print(f"  {'document':>12}   {'read once':>11}   {'per decision':>13}   {'R2':>6}   "
          f"{'32 batched':>11}   {'32 apart':>11}")
    print(f"  {'-'*12}   {'-'*11}   {'-'*13}   {'-'*6}   {'-'*11}   {'-'*11}")

    for target in (0, 400, 1000):
        text = TICKET
        while len(tok(text, add_special_tokens=False)["input_ids"]) < target:
            text = TICKET + "\n\n" + text
        ntok = len(tok(text, add_special_tokens=False)["input_ids"])

        fit = []
        for n in counts:
            many = parse_questions(dict(pool[:n]))

            def once():
                score_questions_cached(model, tok, text, many, enc, device=device)

            for _ in range(2):
                once()
            if device == "cuda":
                torch.cuda.synchronize()
            xs = []
            for _ in range(a.repeat):
                t0 = time.perf_counter()
                once()
                if device == "cuda":
                    torch.cuda.synchronize()
                xs.append(time.perf_counter() - t0)
            fit.append((n, st.median(xs)))

        # Ordinary least squares, written out rather than pulling in numpy: this
        # script should run wherever the model does.
        mx = sum(n for n, _ in fit) / len(fit)
        my = sum(s for _, s in fit) / len(fit)
        sxx = sum((n - mx) ** 2 for n, _ in fit)
        slope = sum((n - mx) * (s - my) for n, s in fit) / sxx
        intercept = my - slope * mx
        ss_res = sum((s - (intercept + slope * n)) ** 2 for n, s in fit)
        ss_tot = sum((s - my) ** 2 for _, s in fit)
        r2 = 1 - ss_res / ss_tot if ss_tot else 1.0

        batched = dict(fit)[counts[-1]]
        apart = dict(fit)[1] * counts[-1]
        print(f"  {ntok:>7} tok   {intercept*1000:>8.0f} ms   {slope*1000:>10.1f} ms   "
              f"{r2:>6.3f}   {batched*1000:>8.0f} ms   {apart*1000:>8.0f} ms")

    print("\n  Reading the document is the fixed cost and grows with it; the marginal")
    print("  cost of one more decision barely moves. The last two columns are the")
    print("  same 32 decisions asked together and asked one request at a time.")
    print("  Numbers move with GPU load; treat them as approximate.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
