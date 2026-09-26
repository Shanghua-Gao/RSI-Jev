"""What should you act on? Two routing questions, both measured.

    python scripts/routing.py                 # both models, 400 held-out documents
    python scripts/routing.py --cases 100     # quicker

An accuracy number tells you how often the model is right. It does not tell you
*which* answers to trust -- and that is the question a deployment actually asks.
A model with honest probabilities can answer it; a model without them cannot.

1. **Abstention.** Rank every decision by the probability behind it and keep only
   the top slice. If the confidence carries information, accuracy on the kept
   slice climbs as coverage falls. If it is noise, the line stays flat. This one
   works, and it is the reason to care about calibration.

2. **Cascade.** Let the 0.8B answer everything and send only the documents it was
   least sure about to the 2B. Unified-memory machines -- a DGX Spark, an Apple
   Silicon box -- hold both checkpoints resident, so escalation is a routing
   decision rather than a deployment one. On our hardware this one **loses**, and
   the table shows why rather than hiding it.

Nothing is simulated: both models really run, timings are wall clock, and every
row is scored on answers actually produced.
"""
from __future__ import annotations

import argparse
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

# The cascade needs a small model and a large one, so it runs v1.0's pair: v2.0 was
# only ever trained at 2B. It is also the honest pair for this script, whose claim
# that no test document was seen in training holds for v1.0 and not for v2.0.
CASCADE = ("v1.0-0.8b", "v1.0-2b")
COVERAGE = (1.0, 0.9, 0.8, 0.6, 0.4, 0.2)
# Escalate the least-confident share of documents, rather than thresholding on a
# probability. With several questions per document almost every document has one
# unsure answer, so any threshold above ~0.5 escalates everything and the table
# says nothing. A share sweeps the whole frontier by construction.
ESCALATE = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def run(model, tok, enc, cases, device, score):
    """Score every document once, keeping per-decision confidence and per-document
    wall time. Timing is per document because that is the unit a cascade escalates.

    The warm-up is not politeness: the first model to run pays for lazy kernel
    selection and allocator growth, which made the 2B look faster than the 0.8B
    purely because it went second."""
    import torch
    for c in cases[:3]:
        score(model, tok, c.state, list(c.questions), enc, device=device)
    if device == "cuda":
        torch.cuda.synchronize()
    out = []
    for c in cases:
        qs = list(c.questions)
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        preds, _ = score(model, tok, c.state, qs, enc, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        out.append({"dt": dt, "q": [(max(p.probs),
                                     max(range(len(p.probs)), key=p.probs.__getitem__))
                                    for p in preds]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp32"])
    a = ap.parse_args()

    import torch
    from load_release import load_release
    from rsijev.contract import gold_label
    from rsijev.targets import load_typed_decisions
    from serve.infer import score_questions_cached
    from huggingface_hub import snapshot_download

    device = a.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if getattr(torch.backends, "mps", None)
                          and torch.backends.mps.is_available() else "cpu")
    dtype_name = a.dtype or ("bf16" if device == "cuda" else "fp32")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]

    cases = load_typed_decisions("test")[:a.cases]
    gold = [[gold_label(q, c.gold[q.key]) for q in c.questions] for c in cases]
    total_q = sum(len(c.questions) for c in cases)

    name = torch.cuda.get_device_name(0) if device == "cuda" else device
    print(f"\n  {name}  ·  {dtype_name}  ·  {len(cases)} documents, {total_q} decisions")
    print("  none of it seen in training; both checkpoints held resident at once\n")

    # Both loaded before either runs, and both still loaded when both are done:
    # the cascade is a routing decision only if you are not paying to swap
    # checkpoints between documents.
    loaded, sizes = {}, {}
    for key in CASCADE:
        before = torch.cuda.memory_allocated() if device == "cuda" else 0
        loaded[key] = load_release(snapshot_download(REPOS[key]), device,
                                   infer_dtype=dtype)
        sizes[key] = ((torch.cuda.memory_allocated() - before) / 2**30
                      if device == "cuda" else 0.0)
    runs = {k: run(*loaded[k][:3], cases, device, score_questions_cached)
            for k in loaded}

    hits = {k: [[runs[k][d]["q"][i][1] == _index_of(cases[d].questions[i], gold[d][i])
                 for i in range(len(cases[d].questions))] for d in range(len(cases))]
            for k in runs}
    wall = {k: sum(r["dt"] for r in runs[k]) for k in runs}
    acc = {k: sum(map(sum, hits[k])) / total_q for k in runs}

    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  resident together: 0.8B {sizes['0.8b']:.1f} GiB + 2B {sizes['2b']:.1f} GiB "
              f"= {sizes['0.8b'] + sizes['2b']:.1f} GiB of {total / 2**30:.0f} GiB\n")

    print("  1. Which decisions can you act on?")
    print("     Rank the 2B's decisions by confidence and keep the top slice.\n")
    print(f"     {'coverage':>8}   {'decisions':>9}   {'accuracy':>8}")
    print(f"     {'-'*8}   {'-'*9}   {'-'*8}")
    ranked = sorted(((runs["2b"][d]["q"][i][0], hits["2b"][d][i])
                     for d in range(len(cases))
                     for i in range(len(cases[d].questions))), reverse=True)
    kept = {}
    for cov in COVERAGE:
        n = max(1, int(round(cov * total_q)))
        kept[cov] = sum(h for _, h in ranked[:n]) / n
        print(f"     {cov:>7.0%}   {n:>9}   {kept[cov]:>7.1%}")
    best = min(COVERAGE)
    print(f"\n     Reading: act on everything and {kept[1.0]:.1%} are right. Act only on the"
          f"\n     {best:.0%} it is surest about and {kept[best]:.1%} are right, with the other"
          f" {int(round((1-best)*total_q))}\n     going to a human. That trade only exists if"
          " the probability is honest.")

    print("\n  2. Does a second model earn its place?")
    print("     The 0.8B answers every document; the ones it was least sure about")
    print("     are re-answered by the 2B. Wall time is measured, not modelled.\n")
    print(f"     {'escalated':>9}   {'accuracy':>8}   {'wall':>7}   {'per decision':>12}")
    print(f"     {'-'*9}   {'-'*8}   {'-'*7}   {'-'*12}")
    # Least confident first, by the document's weakest answer.
    order = sorted(range(len(cases)), key=lambda d: min(c for c, _ in runs["0.8b"][d]["q"]))
    for share in ESCALATE:
        up = set(order[:int(round(share * len(cases)))])
        right = sum(sum(hits["2b" if d in up else "0.8b"][d]) for d in range(len(cases)))
        secs = wall["0.8b"] + sum(runs["2b"][d]["dt"] for d in up)
        tag = ("0.8B only" if share == 0 else "both, always" if share == 1
               else f"{share:.0%}")
        print(f"     {tag:>9}   {right/total_q:>7.1%}   {secs:>6.1f}s   "
              f"{secs/total_q*1000:>9.1f} ms")
    print(f"     {'2B only':>9}   {acc['2b']:>7.1%}   {wall['2b']:>6.1f}s   "
          f"{wall['2b']/total_q*1000:>9.1f} ms")
    ratio = wall["2b"] / wall["0.8b"]
    print(f"\n     A cascade pays only when the cheap model is much cheaper. Here the 2B")
    print(f"     costs {ratio:.2f}x the 0.8B, not the 5-10x a cascade needs, so every mixed")
    print(f"     row pays for the 0.8B pass and then pays the 2B anyway. If no row beats")
    print(f"     '2B only' on both columns, the honest answer is to run the 2B.\n")
    return 0


def _index_of(question, label) -> int:
    return question.options.index(label)


if __name__ == "__main__":
    raise SystemExit(main())
