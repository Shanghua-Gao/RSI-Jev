"""One consistent before/after benchmark of the served path, for docs/inference.md.

Each configuration runs in its own process, one at a time, on the same checkpoint,
documents and questions. The scorer is the server's own: the configuration's
`serve.server.make_scorer` when it has one, else the closure its scripts/serve.py
used (v3.0 as released).

    # A: v3.0 as released (a checkout of 1767f6e, venv from its requirements.txt, no fla)
    python docs/assets/bench_gb10.py --label A --code /path/to/checkout --ckpt DIR --out a.json
    # B: now, default (this package installed with [fast])
    python docs/assets/bench_gb10.py --label B --ckpt DIR --out b.json
    # C: B + --profile server          D: B + --profile agent
    python docs/assets/bench_gb10.py --label C --profile server --ckpt DIR --out c.json
    python docs/assets/bench_gb10.py --label D --profile agent --ckpt DIR --out d.json
    python docs/assets/bench_gb10.py --combine a.json b.json c.json d.json --out bench_gb10.json

Times are in-process (no HTTP), CUDA-synchronised, p50 of --repeat runs after
--warmup runs, in milliseconds. Every grid cell also records its answers against
a reference read of the same model with no cache of any kind (`vs_reference`) and
against configuration A (`vs_A`).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

TICKET = ("Order #48213, placed 3 March, two-day shipping. Customer wrote on 11 March: "
          "\"This is the third time I'm writing. I was charged twice for the same order "
          "and nobody has replied. I need the duplicate refunded today or I'm disputing "
          "it with my bank.\" Account: 4 years old, 26 orders, no prior disputes.")
DOCS = {"80": TICKET, "1052": "\n\n".join([TICKET] * 13)}
ASKS = ["Which team should handle this ticket?", "What is the customer's main problem?",
        "What should support do first?", "Which channel should the reply use?",
        "What is the risk if nobody answers today?", "Which policy applies here?",
        "Who should approve the next step?", "How should this ticket be tagged?"]
OPTIONS = [{"billing": "Payments and refunds", "technical": "Bugs and outages",
            "shipping": "Delivery and tracking", "retention": "Cancellations"},
           {"refund": "Refund the duplicate charge", "replace": "Send a replacement",
            "escalate": "Escalate to a senior agent", "wait": "Wait for more information"},
           {"email": "Reply by email", "phone": "Call the customer",
            "chat": "Open a live chat", "letter": "Send a letter"},
           {"low": "Nothing happens", "medium": "A complaint",
            "high": "A chargeback", "critical": "Legal action"}]
QUESTIONS = {f"q{i}": {"type": "choice", "instructions": ASKS[i % 8],
                       "criteria": OPTIONS[i // 8]} for i in range(32)}
STEP = ("The customer adds: I checked my statement again this morning and both charges are "
        "still there, one on the 3rd and one on the 4th, each for the full order amount. "
        "I have attached the screenshot and my order number again. ")


def load(a):
    import torch
    if a.code:
        sys.path.insert(0, str(Path(a.code) / "scripts"))
        sys.path.insert(0, str(Path(a.code)))
    if a.profile:
        from serve.runtime import apply_profile
        apply_profile(a.profile)
    from serve.infer import score_questions
    try:
        from serve.server import load_for_serving, make_scorer
    except ImportError:                                   # v3.0 as released
        from load_release import load_release
        from serve.infer import score_questions_cached
        model, tok, enc, meta = load_release(a.ckpt, "cuda", infer_dtype=torch.bfloat16)
        spec = meta["spec"]

        def scorer(state, questions):                     # scripts/serve.py at 1767f6e
            preds, tokens = score_questions_cached(model, tok, state, questions, enc,
                                                   device="cuda", batch_size=16,
                                                   max_options=max(spec["max_options"],
                                                                   max(len(q.options) for q in questions)))
            return [list(p.probs) for p in preds], tokens
        applied = []
    else:
        s = load_for_serving(a.ckpt, device="cuda", dtype="bf16")
        model, tok, enc, spec, applied = s.model, s.tok, s.enc, s.meta["spec"], s.applied
        scorer = make_scorer(s, 16)

    def reference(state, questions):
        """Each question read with the whole state, no cache of any kind: the path
        the evaluator's predict() is pinned to."""
        preds, _ = score_questions(model, tok, state, questions, enc, device="cuda",
                                   batch_size=16, max_options=spec["max_options"])
        return [list(p.probs) for p in preds]
    return scorer, reference, tok, applied


def timed(fn, warmup, repeat):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs, out = [], None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        xs.append((time.perf_counter() - t) * 1000)
    return xs, out


def summary(xs):
    xs = sorted(xs)
    return {"p50_ms": round(st.median(xs), 2), "p10_ms": round(xs[len(xs) // 10], 2),
            "p90_ms": round(xs[(9 * len(xs)) // 10], 2), "n": len(xs)}


def run(a) -> dict:
    import torch
    import transformers
    scorer, reference, tok, applied = load(a)
    from serve.wire import parse_questions
    ntok = {k: len(tok(v, add_special_tokens=False)["input_ids"]) for k, v in DOCS.items()}
    res = {"label": a.label, "profile": a.profile, "applied": applied,
           "torch": torch.__version__, "transformers": transformers.__version__,
           "fla": _version("flash-linear-attention"), "gpu": torch.cuda.get_device_name(0),
           "env": {k: v for k, v in os.environ.items() if k.startswith("RSIJEV_")},
           "doc_tokens": ntok, "grid": {}, "repeat": {}, "growth": {}}

    if not a.skip_grid:
        for doc, text in DOCS.items():
            for n in (1, 8, 32):
                qs = parse_questions(dict(list(QUESTIONS.items())[:n]))
                xs, out = timed(lambda: scorer(text, qs), a.warmup, a.repeat)
                cell = {**summary(xs), "questions": n, "doc_tokens": ntok[doc],
                        "probs": [[round(p, 6) for p in row] for row in out[0]]}
                cell["vs_reference"] = agreement(out[0], reference(text, qs))
                cell["per_decision_ms"] = round(cell["p50_ms"] / n, 2)
                res["grid"][f"{doc}x{n}"] = cell
                print(a.label, "grid", doc, n, cell["p50_ms"], flush=True)

    # The same state asked again, as an agent re-checking its own context.
    for doc, text in DOCS.items():
        for n in (1, 4):
            qs = parse_questions(dict(list(QUESTIONS.items())[:n]))
            xs, _ = timed(lambda: scorer(text, qs), a.warmup, a.repeat)
            res["repeat"][f"{doc}x{n}"] = {**summary(xs), "questions": n, "doc_tokens": ntok[doc]}
            print(a.label, "repeat", doc, n, res["repeat"][f"{doc}x{n}"]["p50_ms"], flush=True)

    # A state growing by about 100 tokens per step, 17 steps, from about 240 to about
    # 1,870 tokens. Every trajectory starts with its own first line, so nothing is
    # reused across trajectories; the first is warm-up.
    for n in (1, 4):
        qs = parse_questions(dict(list(QUESTIONS.items())[:n]))
        xs, sizes = [], []
        for traj in range(a.trajectories + 1):
            state = f"Ticket thread {traj}-{n}. " + TICKET
            for step in range(17):
                state = state + " " + STEP + STEP
                torch.cuda.synchronize()
                t = time.perf_counter()
                scorer(state, qs)
                torch.cuda.synchronize()
                if traj > 0:
                    xs.append((time.perf_counter() - t) * 1000)
                    sizes.append(len(tok(state, add_special_tokens=False)["input_ids"]))
        res["growth"][f"x{n}"] = {**summary(xs), "questions": n, "steps": len(xs),
                                  "tokens_first": min(sizes), "tokens_last": max(sizes),
                                  "tokens_per_step": round((max(sizes) - min(sizes)) / 16, 1)}
        print(a.label, "growth", n, res["growth"][f"x{n}"]["p50_ms"], flush=True)
    return res


def agreement(got, ref) -> dict:
    same = sum(max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__)
               for x, y in zip(got, ref))
    dp = max(max(abs(p - q) for p, q in zip(x, y)) for x, y in zip(got, ref))
    return {"argmax_equal": f"{same}/{len(got)}", "max_abs_dp": round(dp, 5)}


def _version(dist):
    import importlib.metadata
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def combine(paths, out):
    runs = {r["label"]: r for r in (json.loads(Path(p).read_text()) for p in paths)}
    base = runs["A"]
    for label, r in runs.items():
        for cell, c in r["grid"].items():
            ref = base["grid"].get(cell)
            if ref is None:
                continue
            c["vs_A"] = agreement(c["probs"], ref["probs"])
            if label != "A":
                c["speedup_vs_A"] = round(ref["p50_ms"] / c["p50_ms"], 2)
        for group in ("repeat", "growth"):
            for cell, c in r[group].items():
                if label != "A" and cell in base[group]:
                    c["speedup_vs_A"] = round(base[group][cell]["p50_ms"] / c["p50_ms"], 2)
    Path(out).write_text(json.dumps({
        "what": "RSI-Jev v3.0-2B served path on one HP ZGX Nano (NVIDIA GB10), bf16 tower, "
                "fp32 scorer; in-process p50 in ms",
        "configs": {"A": "v3.0 as released (1767f6e), requirements.txt, no fla",
                    "B": "current code, rsi-jev[fast] (fla), default",
                    "C": "B + --profile server (RSIJEV_COMPILE=1)",
                    "D": "B + --profile agent (RSIJEV_DOC_CACHE=1)"},
        "questions": "4-option choice questions, distinct instructions",
        "runs": runs}, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label")
    ap.add_argument("--ckpt")
    ap.add_argument("--code", default=None, help="a checkout to import from, e.g. v3.0 as released")
    ap.add_argument("--profile", default=None, choices=["agent", "server"])
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--trajectories", type=int, default=2)
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--combine", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.combine:
        combine(a.combine, a.out)
        return 0
    Path(a.out).write_text(json.dumps(run(a), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
