"""How fast is a decision, and what is the cost actually spent on?

Measures the served path — `score_questions`, the same code the server and the
Space call — rather than a hand-rolled forward pass, so the numbers are the ones
a caller would see.

    python scripts/bench_inference.py --ckpt DIR [--json out.json]

Three things it answers:

1. **Latency per decision**, for a realistic multi-question case and for a
   single question, warm.
2. **What re-encoding the state costs.** Every question is encoded with the
   whole state in front of it, so a case with k questions pushes the state
   through the tower k times. This reports the implied waste.
3. **Whether bf16 inference is free.** The tower is fp32 at rest; bf16 should be
   roughly twice as fast. The question is whether it changes any answer, so this
   compares predictions rather than assuming.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

# The repository root must precede scripts/ on sys.path: scripts/serve.py would
# otherwise shadow the serve/ PACKAGE and "from serve.infer import ..." fails.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def timed(fn, repeat: int, warmup: int = 2):
    import torch
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    xs = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        xs.append(time.perf_counter() - t)
    return {"median_s": round(st.median(xs), 4), "min_s": round(min(xs), 4), "n": repeat}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--cases", type=int, default=40, help="cases for the bf16 agreement check")
    ap.add_argument("--compile", action="store_true",
                    help="also time torch.compile(mode='reduce-overhead'), which uses CUDA "
                         "graphs underneath, against the eager path")
    a = ap.parse_args()

    import torch
    from load_release import load_release
    from rsijev.targets import load_typed_decisions
    from serve.infer import score_questions, score_questions_cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, enc, meta = load_release(a.ckpt, device)
    enc = enc
    name = Path(a.ckpt).name
    cases = load_typed_decisions("test")
    report: dict = {"checkpoint": name, "device": device,
                    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                    "base": meta["base_model"], "kernel": meta["linear_attn_kernel"]}

    # A real case: one state, all of its questions, which is how the API is used.
    case = max(cases[:50], key=lambda c: len(c.questions))
    qs = list(case.questions)

    # Sweep the state length, because that is what decides whether caching it
    # pays: uncached costs Q*(P+S), cached costs P+Q*S, so the win grows with the
    # document relative to the questions. Longer states are built from other real
    # states so the text stays in distribution.
    filler = [c.state for c in cases[50:200]]

    def state_of(target_tokens: int) -> str:
        text = case.state
        k = 0
        while len(tok(text, add_special_tokens=False)["input_ids"]) < target_tokens and k < len(filler):
            text = filler[k] + "\n\n" + text
            k += 1
        return text

    report["questions"] = len(qs)
    report["sweep"] = []
    for target in (0, 400, 1000, 2000):
        st = state_of(target)
        ntok = len(tok(st, add_special_tokens=False)["input_ids"])
        row = {"state_tokens": ntok}
        for dtype_name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            model.tower.to(dtype)
            for path, fn in (("uncached", score_questions), ("cached", score_questions_cached)):
                r = timed(lambda fn=fn, st=st: fn(model, tok, st, qs, enc, device=device), a.repeat)
                row[f"{dtype_name}_{path}_ms"] = round(r["median_s"] * 1000, 1)
            row[f"{dtype_name}_cache_speedup"] = round(
                row[f"{dtype_name}_uncached_ms"] / row[f"{dtype_name}_cached_ms"], 2)
        model.tower.to(torch.float32)
        row["per_decision_ms_best"] = round(row["bf16_cached_ms"] / len(qs), 1)
        report["sweep"].append(row)
        print(f"  state {ntok:5d} tok  fp32 {row['fp32_uncached_ms']:7.1f} -> "
              f"{row['fp32_cached_ms']:7.1f} (x{row['fp32_cache_speedup']})   "
              f"bf16 {row['bf16_uncached_ms']:7.1f} -> {row['bf16_cached_ms']:7.1f} "
              f"(x{row['bf16_cache_speedup']})   best/decision {row['per_decision_ms_best']} ms",
              flush=True)

    if a.compile:
        # The regime is launch-bound: a pass costs about the same whatever it
        # contains. reduce-overhead captures CUDA graphs, which is the cheap way
        # to find out whether that is worth hand-rolling.
        st = state_of(400)
        model.tower.to(torch.bfloat16)
        base = timed(lambda: score_questions_cached(model, tok, st, qs, enc, device=device),
                     a.repeat)
        before = [list(p.probs) for p in
                  score_questions_cached(model, tok, st, qs, enc, device=device)[0]]
        try:
            t0 = time.perf_counter()
            model.tower.forward = torch.compile(model.tower.forward, mode="reduce-overhead")
            warm = timed(lambda: score_questions_cached(model, tok, st, qs, enc, device=device),
                         1, warmup=3)          # includes compilation
            after = [list(p.probs) for p in
                     score_questions_cached(model, tok, st, qs, enc, device=device)[0]]
            comp = timed(lambda: score_questions_cached(model, tok, st, qs, enc, device=device),
                         a.repeat)
            worst = max(abs(x - y) for u, v in zip(before, after) for x, y in zip(u, v))
            flips = sum(max(range(len(u)), key=u.__getitem__)
                        != max(range(len(v)), key=v.__getitem__)
                        for u, v in zip(before, after))
            report["compile"] = {
                "eager_ms": round(base["median_s"] * 1000, 1),
                "compiled_ms": round(comp["median_s"] * 1000, 1),
                "speedup": round(base["median_s"] / comp["median_s"], 2),
                "first_call_s": round(warm["median_s"], 1),
                "total_setup_s": round(time.perf_counter() - t0, 1),
                "worst_prob_delta": float(f"{worst:.2e}"), "argmax_flips": flips,
                "decisions": len(before)}
        except Exception as exc:
            report["compile"] = {"failed": f"{type(exc).__name__}: {str(exc)[:300]}"}
        finally:
            model.tower.to(torch.float32)
        print("  compile:", json.dumps(report["compile"]), flush=True)

    print(json.dumps(report, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
