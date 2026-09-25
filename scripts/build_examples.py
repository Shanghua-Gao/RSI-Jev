"""Regenerate the demo's prefilled examples from the held-out split.

    python scripts/build_examples.py

The demo opens on a real held-out question so there is something to look at before
you type your own. This writes serve/examples.json: a handful of cases, their typed
questions, and the teacher's answers.

Two details are easy to get wrong, and were. The documents in this dataset are JSON
strings, so they are stored pretty-printed rather than as one 400-character line. And
the wire contract wants a score question's criteria as an ORDERED ARRAY, worst level
first, while the internal Question holds a mapping keyed by option -- shipping the
mapping made every prefilled example fail the moment anyone pressed Compare. Every
example is parsed against the real contract here before it is written.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Root before scripts/: scripts/serve.py shadows the serve/ package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="serve/examples.json")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    from rsijev.contract import gold_label
    from rsijev.targets import load_typed_decisions
    from serve.wire import parse_questions

    # One per source before a second from any: a picker of four near-identical agent
    # traces teaches nobody what the model is for. And the first one is what the page
    # opens on, so it has to be a situation a stranger recognises -- the reference
    # playground opens on a customer asking for a refund, not on an internal trace.
    FIRST = "customer_service"
    by_source: dict[str, list] = {}
    for c in load_typed_decisions("test"):
        by_source.setdefault(c.case_id.rsplit("_", 1)[0], []).append(c)
    picked = []
    order = sorted(by_source, key=lambda s: (s != FIRST, s))
    for depth in range(max(len(v) for v in by_source.values())):
        for source in order:
            if depth < len(by_source[source]) and len(picked) < a.n:
                picked.append(by_source[source][depth])
        if len(picked) >= a.n:
            break

    def spec(q) -> dict:
        s = {"type": q.mode, "instructions": q.instructions}
        if q.criteria:
            crit = dict(q.criteria)
            s["criteria"] = ([crit[k] for k in sorted(crit, key=int)] if q.mode == "score"
                             else crit)
        return s

    out, seen_titles = [], {}
    for c in picked:
        state = c.state
        try:                                  # these states are JSON; show them as JSON
            state = json.dumps(json.loads(state), indent=2, ensure_ascii=False)
        except (ValueError, TypeError):
            pass
        # A human name. Nobody can pick between agent_trace_observability_000000 and
        # customer_service_000050, and the playgrounds people actually understand label
        # their scenarios in words. Derived from the case id, not invented.
        source = c.case_id.rsplit("_", 1)[0].replace("_", " ").strip()
        title = source[:1].upper() + source[1:]
        seen_titles[title] = seen_titles.get(title, 0) + 1
        if seen_titles[title] > 1:
            title = f"{title} {seen_titles[title]}"
        out.append({"id": c.case_id, "title": title, "state": state,
                    "questions": {q.key: spec(q) for q in c.questions},
                    "gold": {q.key: gold_label(q, c.gold[q.key]) for q in c.questions}})

    for e in out:
        for q in parse_questions(e["questions"]):          # raises on an invalid spec
            if e["gold"][q.key] not in q.options:
                raise RuntimeError(f"{e['id']}/{q.key}: gold {e['gold'][q.key]!r} "
                                   f"is not one of {q.options}")

    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    modes = sorted({q["type"] for e in out for q in e["questions"].values()})
    print(f"  wrote {a.out} — {len(out)} examples, modes {modes}, "
          f"all parsed against the contract")
    for e in out:
        print(f"    {e['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
