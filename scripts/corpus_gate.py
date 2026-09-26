"""Data-quality gate for custom training corpora.

A corpus that silently shifts the label prior loses on test, and looks like a
method failure when it is a data defect:
  - data-13: the Qwen3.5-4B direct-readout labels put the score argmax at level
    0 on 87% of questions (-0.132);
  - data-12: the two-teacher agreement filter moved the noul pool from 50% to
    61% "true" (-0.045).
This compares a corpus's per-mode label statistics against a reference (the
default synth corpus) and names every shift beyond tolerance. the arm runner
refuses to submit a corpus that fails, unless --allow-label-shift gives a reason
(which is recorded in the spec).

    python scripts/corpus_gate.py --corpus DIR_or_FILE [--reference scripts/corpus_reference_stats.json]
    python scripts/corpus_gate.py --make-reference synth.jsonl --out scripts/corpus_reference_stats.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TOL = {"noul_true_rate": 0.05, "score_mean_level": 0.25, "score_argmax_tv": 0.15,
       "choice_first_option_rate": 0.08, "mode_share_tv": 0.10}


def stats(cases) -> dict:
    n_mode = collections.Counter()
    noul_true, score_lvl, score_hist, choice_first = [], [], collections.Counter(), []
    for c in cases:
        for q in c.questions:
            g = c.gold[q.key]
            am = max(range(len(g)), key=g.__getitem__)
            n_mode[q.mode] += 1
            if q.mode == "noul":
                noul_true.append(g[1])
            elif q.mode == "score":
                score_lvl.append(sum(i * p for i, p in enumerate(g)))
                score_hist[str(am)] += 1
            elif q.mode == "choice":
                choice_first.append(am == 0)
    tot = sum(n_mode.values())
    ns = sum(score_hist.values())
    mean = lambda v: sum(v) / len(v) if v else None
    return {"questions": tot,
            "mode_share": {m: round(n / tot, 4) for m, n in sorted(n_mode.items())},
            "noul_true_rate": round(mean(noul_true), 4) if noul_true else None,
            "score_mean_level": round(mean(score_lvl), 4) if score_lvl else None,
            "score_argmax_hist": {k: round(v / ns, 4) for k, v in sorted(score_hist.items())} if ns else {},
            "choice_first_option_rate": round(mean(choice_first), 4) if choice_first else None}


def _load(path: str):
    from rsijev.contract import load_cases
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    return [c for f in files for c in load_cases(str(f))]


def _tv(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def check(corpus: str, reference: str) -> tuple[bool, list[str], dict]:
    ref = json.loads(Path(reference).read_text())
    s = stats(_load(corpus))
    fails = []
    if _tv(s["mode_share"], ref["mode_share"]) > TOL["mode_share_tv"]:
        fails.append(f"mode mix shifted: {s['mode_share']} vs {ref['mode_share']}")
    for k in ("noul_true_rate", "score_mean_level", "choice_first_option_rate"):
        if s[k] is not None and ref[k] is not None and abs(s[k] - ref[k]) > TOL[k]:
            fails.append(f"{k} {s[k]} vs reference {ref[k]} (tolerance {TOL[k]})")
    if s["score_argmax_hist"] and _tv(s["score_argmax_hist"], ref["score_argmax_hist"]) > TOL["score_argmax_tv"]:
        fails.append(f"score argmax histogram shifted (TV "
                     f"{_tv(s['score_argmax_hist'], ref['score_argmax_hist']):.3f}): "
                     f"{s['score_argmax_hist']} vs {ref['score_argmax_hist']}")
    return (not fails), fails, s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--reference", default=str(Path(__file__).with_name("corpus_reference_stats.json")))
    ap.add_argument("--make-reference")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.make_reference:
        s = stats(_load(a.make_reference))
        Path(a.out).write_text(json.dumps(s, indent=2) + "\n")
        print(json.dumps(s, indent=2))
        return 0
    ok, fails, s = check(a.corpus, a.reference)
    print(json.dumps(s, indent=2))
    print("CORPUS GATE:", "PASS" if ok else "FAIL")
    for f in fails:
        print("  -", f)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
