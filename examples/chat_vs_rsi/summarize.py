"""rsi.json + chat.json -> results.json and results.md: the comparison's numbers.

    python examples/chat_vs_rsi/summarize.py [--rsi rsi.json] [--chat chat.json] [--out-dir .]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(mid - half, 3), round(mid + half, 3))


def lenient(text, q):
    """Post hoc reading of a chat reply: the last JSON object in it, and any of its values that is an
    allowed answer (e.g. under "answer" when the model echoed the question under the id "q")."""
    opts = ["yes", "no"] if q["spec"]["type"] == "noul" else list(q["spec"]["criteria"])
    lower = {o.lower(): o for o in opts}
    try:
        i = text.rindex("{")
        obj = json.loads(text[i: text.index("}", i) + 1])
    except Exception:
        return None
    vals = [obj.get(q["key"])] + [v for k, v in obj.items() if k != q["key"]]
    for v in vals:
        if isinstance(v, str) and v.strip().lower() in lower:
            a = lower[v.strip().lower()]
            return ("true" if a == "yes" else "false") if q["spec"]["type"] == "noul" else a
    return None


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 4)


def side(rows):
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    v = sum(r["valid"] for r in rows)
    ms = [r["ms_p50"] for r in rows]
    return dict(n=n, latency_ms_p50=round(statistics.median(ms), 1),
                latency_ms_min=round(min(ms), 1), latency_ms_max=round(max(ms), 1),
                valid_output_rate=round(v / n, 3), n_valid=v,
                accuracy=round(k / n, 3), n_correct=k, accuracy_95ci=wilson(k, n),
                confidence_available=all(r["confidence_available"] for r in rows))


def summarize(qs, rsi, chat) -> dict:
    R = {r["id"]: r for r in rsi["results"]}
    C = {r["id"]: r for r in chat["results"]}
    Q = {q["id"]: q for q in qs}
    ids = [q["id"] for q in qs if q["id"] in R and q["id"] in C]
    a, b = [R[i] for i in ids], [C[i] for i in ids]
    both = sum(R[i]["correct"] and C[i]["correct"] for i in ids)
    only_r = sum(R[i]["correct"] and not C[i]["correct"] for i in ids)
    only_c = sum(C[i]["correct"] and not R[i]["correct"] for i in ids)
    len_pick = {i: (C[i]["pick"] if C[i]["valid"] else lenient(C[i]["text"], Q[i])) for i in ids}
    len_valid = sum(v is not None for v in len_pick.values())
    len_correct = sum(len_pick[i] is not None and len_pick[i] in Q[i]["ref"] for i in ids)
    cats = {}
    for i in ids:
        c = cats.setdefault(Q[i]["category"], [0, 0, 0])
        c[0] += 1
        c[1] += R[i]["correct"]
        c[2] += C[i]["correct"]
    return dict(
        rsi_jev=dict(model=rsi.get("model"), path="rsi-jev serve over HTTP",
                     timing="whole HTTP call; per question 3 warm + 9 timed, p50; summary = median of per-question p50",
                     **side(a)),
        chat=dict(model=chat["model"], mode="non-thinking (enable_thinking=False), greedy, max_new_tokens=256, bf16",
                  timing="processor + generate; per question 2 warm + 5 timed, p50; summary = median of per-question p50",
                  lenient_post_hoc=dict(n_valid=len_valid, n_correct=len_correct,
                                        accuracy=round(len_correct / len(ids), 3),
                                        accuracy_95ci=wilson(len_correct, len(ids)),
                                        rule="last JSON object; any value that is an allowed answer"),
                  output_tokens_p50=statistics.median(r["output_tokens"] for r in b),
                  **side(b)),
        paired=dict(both_correct=both, only_rsi_correct=only_r, only_chat_correct=only_c,
                    neither=len(ids) - both - only_r - only_c, mcnemar_p=mcnemar(only_r, only_c)),
        by_category={k: dict(n=v[0], rsi_correct=v[1], chat_correct=v[2]) for k, v in cats.items()},
        per_question=[dict(id=i, ref=Q[i]["ref"], rsi_pick=R[i]["pick"], rsi_correct=R[i]["correct"],
                           rsi_ms=R[i]["ms_p50"], chat_text=C[i]["text"], chat_pick=C[i]["pick"],
                           chat_valid=C[i]["valid"], chat_correct=C[i]["correct"], chat_ms=C[i]["ms_p50"])
                      for i in ids],
        chat_prompt_example=chat.get("prompt_example"),
    )


def markdown(out: dict) -> str:
    r, c, p = out["rsi_jev"], out["chat"], out["paired"]
    lp = c["lenient_post_hoc"]
    n = r["n"]
    return f"""| | RSI-Jev v4.0-VL | Qwen3.5-2B chat (non-thinking) |
|---|---|---|
| latency p50 (median over questions) | {r['latency_ms_p50']:.0f} ms | {c['latency_ms_p50']:.0f} ms |
| latency range (per-question p50) | {r['latency_ms_min']:.0f}–{r['latency_ms_max']:.0f} ms | {c['latency_ms_min']:.0f}–{c['latency_ms_max']:.0f} ms |
| valid output | {r['n_valid']}/{n} | {c['n_valid']}/{n} |
| accuracy (invalid counts as wrong) | {r['n_correct']}/{n} = {r['accuracy']:.2f} | {c['n_correct']}/{n} = {c['accuracy']:.2f} |
| chat, lenient re-read (post hoc) | | valid {lp['n_valid']}/{n}, accuracy {lp['n_correct']}/{n} = {lp['accuracy']:.2f} |
| confidence available | {'yes: a calibrated probability per answer' if r['confidence_available'] else 'no'} | {'yes' if c['confidence_available'] else 'no: text only'} |

Paired: both right {p['both_correct']}, only RSI-Jev right {p['only_rsi_correct']}, only the chat model right \
{p['only_chat_correct']}, neither {p['neither']} (exact McNemar p = {p['mcnemar_p']:.2g}, strict reading).

By category (RSI-Jev / chat, correct of n): """ + ", ".join(
        f"{k} {v['rsi_correct']}/{v['chat_correct']} of {v['n']}" for k, v in out["by_category"].items()) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(HERE / "questions.json"))
    ap.add_argument("--rsi", default=str(HERE / "rsi.json"))
    ap.add_argument("--chat", default=str(HERE / "chat.json"))
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args(argv)
    out = summarize(json.load(open(a.questions)), json.load(open(a.rsi)), json.load(open(a.chat)))
    d = Path(a.out_dir)
    (d / "results.json").write_text(json.dumps(out, indent=1))
    md = markdown(out)
    (d / "results.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
