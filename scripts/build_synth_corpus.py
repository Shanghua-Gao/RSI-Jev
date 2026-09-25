"""Convert n4ze3m/typed-decisions-synth into training cases.

Generalist rule: none of its 149 domains is one of typed-decisions' four
workflows (agent_trace_observability, customer_service, invoice_processing,
security_incidents), and no state matches exactly. Nine domains are
keyword-ADJACENT (for example "procurement vendor vetting" next to invoice
processing). They go to a separate file, synth_adjacent.jsonl, so an arm can
leave them out and show the result does not depend on near-workflow data.

Gold is the TEACHER's soft distribution (DeepSeek, sampled x3 and averaged), the
same construction as typed-decisions' own gold, not the argmax label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.contract import Case, Question, dump_cases   # noqa: E402

ADJACENT = re.compile(r"agent|trace|observab|customer|support|ticket|invoice|bill|payable|"
                      r"procure|security|incident|alert|\bsoc\b|threat", re.I)
NOUL_DEFAULT = {"false": "The statement is false.", "true": "The statement is true."}


def convert(row) -> Case | None:
    qs, teacher = json.loads(row["questions"]), json.loads(row["teacher"])
    questions, gold = [], {}
    for key, q in qs.items():
        t = teacher.get(key)
        if t is None:
            continue
        raw = q.get("criteria")
        if q["type"] == "noul":
            if "noul" not in t:
                continue
            options = ("false", "true")
            criteria = raw if isinstance(raw, dict) and set(raw) >= {"false", "true"} else NOUL_DEFAULT
            p = float(t["noul"])
            vec = [1.0 - p, p]
        else:
            probs = t.get("probabilities")
            if not probs:
                continue
            if isinstance(raw, dict):
                options, criteria = tuple(raw), raw
            elif isinstance(raw, list):
                options = tuple(str(i) for i in range(len(raw)))
                criteria = {str(i): d for i, d in enumerate(raw)}
            else:
                continue
            vec = [float(probs.get(o, 0.0)) for o in options]
        s = sum(vec)
        if s <= 0:
            continue
        gold[key] = tuple(v / s for v in vec)
        questions.append(Question(key=key, mode=q["type"], instructions=q["instructions"],
                                  options=options, criteria=criteria))
    if not questions:
        return None
    return Case(case_id=row["state_id"], source="synth", state=row["state"],
                questions=tuple(questions), gold=gold)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from datasets import load_dataset
    d = load_dataset("n4ze3m/typed-decisions-synth")
    td = load_dataset("LocalLLaMA/typed-decisions", "all")
    td_states = {hashlib.sha256(r["state"].encode()).hexdigest() for sp in td for r in td[sp]}

    main_cases, adj_cases, skipped, overlap = [], [], 0, 0
    for sp in d:
        for row in d[sp]:
            if hashlib.sha256(row["state"].encode()).hexdigest() in td_states:
                overlap += 1
                continue
            c = convert(row)
            if c is None:
                skipped += 1
                continue
            if ADJACENT.search(row["domain"]):
                adj_cases.append(Case(c.case_id, "synth_adjacent", c.state, c.questions, c.gold))
            else:
                main_cases.append(c)
    out = Path(args.out)
    n1 = dump_cases(main_cases, str(out / "synth.jsonl"))
    n2 = dump_cases(adj_cases, str(out / "synth_adjacent.jsonl"))
    qt = Counter(q.mode for c in main_cases + adj_cases for q in c.questions)
    print(json.dumps({"synth_cases": n1, "synth_adjacent_cases": n2, "skipped": skipped,
                      "state_overlap_with_typed_decisions": overlap,
                      "questions_by_mode": dict(qt)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
