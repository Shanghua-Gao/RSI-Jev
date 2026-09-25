"""Build the training corpus from public classification sources.

Generalist rule: these are workflows we do NOT evaluate on. Neither evaluation
target contributes a single training row.

Label names are read from each dataset's own ClassLabel feature rather than
hard-coded, so a renamed or reordered label cannot silently mislabel a corpus.
Where a source ships bare label names, the name is its own description; the
descriptions are part of the input, so a bare name is a weaker task, and which
sources get written descriptions is itself a data-axis question.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.contract import Case, Question, dump_cases   # noqa: E402
from rsijev.data import SOURCES                          # noqa: E402


def _label_names(ds, field: str, declared: tuple[str, ...]) -> tuple[str, ...]:
    feat = ds.features.get(field)
    names = getattr(feat, "names", None)
    if names:
        return tuple(str(n) for n in names)
    if declared:
        return declared
    vals = sorted({str(r[field]) for r in ds.select(range(min(len(ds), 2000)))})
    return tuple(vals)


def build_one(name: str, limit: int) -> list[Case]:
    from datasets import load_dataset

    src = SOURCES[name]
    ds = load_dataset(src.hf_id, src.config, split=src.split)
    if limit and len(ds) > limit:
        ds = ds.shuffle(seed=20260922).select(range(limit))
    raw_labels = _label_names(ds, src.label_field, src.labels)
    if src.label_map:
        labels = tuple(src.label_map.get(n, n) for n in raw_labels)
    elif src.mode == "noul":
        labels = ("false", "true")          # dataset order is (negative, positive)
    else:
        labels = raw_labels
    criteria = {o: (src.criteria.get(o) or o.replace("_", " ")) for o in labels}
    fields = (src.text_field,) if isinstance(src.text_field, str) else src.text_field

    out: list[Case] = []
    for i, row in enumerate(ds):
        raw = row[src.label_field]
        idx = int(raw) if not isinstance(raw, str) else (
            labels.index(raw) if raw in labels else None)
        if idx is None or not (0 <= idx < len(labels)):
            continue
        state = "\n\n".join(f"{f}: {row[f]}" for f in fields if row.get(f))
        if not state.strip():
            continue
        q = Question(key="label", mode=src.mode, instructions=src.instructions,
                     options=labels, criteria=criteria)
        gold = tuple(1.0 if k == idx else 0.0 for k in range(len(labels)))
        out.append(Case(f"{name}_{i:06d}", name, state, (q,), {"label": gold}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=3000, help="rows per source, 0 for all")
    ap.add_argument("--sources", default=",".join(SOURCES))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.sources.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            cases = build_one(name, args.limit)
        except Exception as exc:                       # a broken source must not
            summary[name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}   # kill the build
            print(f"{name:12s} FAILED  {summary[name]['error']}", flush=True)
            continue
        n = dump_cases(cases, str(out / f"{name}.jsonl"))
        q0 = cases[0].questions[0]
        summary[name] = {"cases": n, "mode": q0.mode, "options": len(q0.options),
                         "labels": list(q0.options)[:6]}
        print(f"{name:12s} {n:6d} cases  mode={q0.mode:6s} K={len(q0.options)}", flush=True)

    ok = {k: v for k, v in summary.items() if "error" not in v}
    summary["_totals"] = {"sources_ok": len(ok), "sources_failed": len(summary) - len(ok),
                          "cases": sum(v["cases"] for v in ok.values()),
                          "by_mode": dict(Counter(v["mode"] for v in ok.values()))}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n" + json.dumps(summary["_totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
