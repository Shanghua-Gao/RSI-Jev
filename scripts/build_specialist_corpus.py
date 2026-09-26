"""SPECIALIST TRACK corpus: champion synth + the typed-decisions TRAIN split (never test).

    OUT/synth.jsonl     copy of the champion's synth corpus (DeepSeek gold), unchanged
    OUT/td_train.jsonl  rsijev.targets.load_typed_decisions("train"), soft gold as published,
                        repeated --oversample times. Copies keep the SAME case_id, so the
                        runner's hash-by-case_id in-distribution holdout keeps or holds out
                        all copies of a case together (no copy of a held-out case is trained on).
    OUT/build_report.json

Verifies, with the runner's own state normalisation (whitespace-collapsed, lowercased
sha256), that ZERO corpus states occur in the typed-decisions TEST split, and refuses
to write otherwise. Train with --set sources='"synth,td_train"' --set specialist=true.

    python scripts/build_specialist_corpus.py --synth DS/synth.jsonl --out DIR [--oversample 3]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.contract import dump_cases, load_cases  # noqa: E402
from rsijev.targets import load_typed_decisions  # noqa: E402


def _norm(state: str) -> str:
    # Whitespace-collapsed, lower-cased, hashed. The overlap checks and the arm
    # runner must agree on this exactly, or a contamination check silently passes.
    return hashlib.sha256(re.sub(r"\s+", " ", state).strip().lower().encode()).hexdigest()


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--oversample", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    synth = load_cases(a.synth)
    train = load_typed_decisions("train")
    test = load_typed_decisions("test")
    assert all(c.source == "typed_decisions_train" for c in train)

    banned = {_norm(c.state) for c in test}
    test_ids = {c.case_id for c in test}
    hits_synth = sum(_norm(c.state) in banned for c in synth)
    hits_train = sum(_norm(c.state) in banned for c in train)
    id_hits = sum(c.case_id in test_ids for c in train)
    # substring check too: a test state embedded inside a longer train/synth state
    test_states = [re.sub(r"\s+", " ", c.state).strip().lower() for c in test]
    corpus_states = [re.sub(r"\s+", " ", c.state).strip().lower() for c in synth + train]
    joined = "\x00".join(corpus_states)
    substr_hits = sum(ts in joined for ts in test_states)
    if hits_synth or hits_train or id_hits or substr_hits:
        raise SystemExit(f"TEST OVERLAP: synth={hits_synth} train={hits_train} ids={id_hits} "
                         f"substr={substr_hits}; refusing to write")

    shutil.copy(a.synth, out / "synth.jsonl")
    td = [c for c in train for _ in range(a.oversample)]
    dump_cases(td, str(out / "td_train.jsonl"))

    def qc(cs):
        return dict(collections.Counter(q.mode for c in cs for q in c.questions))
    q_syn = sum(len(c.questions) for c in synth)
    q_td = sum(len(c.questions) for c in td)
    rep = {
        "track": "specialist",
        "synth": a.synth, "synth_md5": md5(a.synth), "synth_cases": len(synth),
        "synth_questions": q_syn, "synth_modes": qc(synth),
        "td_train_cases_unique": len(train), "oversample": a.oversample,
        "td_train_rows": len(td), "td_train_questions": q_td, "td_train_modes": qc(td),
        "td_train_md5": md5(out / "td_train.jsonl"),
        "test_overlap": {"exact_state_synth": hits_synth, "exact_state_train": hits_train,
                         "case_id": id_hits, "test_state_substring_in_corpus": substr_hits,
                         "test_cases_checked": len(test)},
        "steps_scaled": round(1500 * (q_syn + q_td) / q_syn),
    }
    try:
        from corpus_gate import check
        ok, fails, _ = check(str(out), str(Path(__file__).with_name("corpus_reference_stats.json")))
        rep["corpus_gate"] = {"pass": bool(ok), "fails": fails}
    except Exception as e:  # report, the submit path re-runs the gate anyway
        rep["corpus_gate"] = {"error": repr(e)}
    (out / "build_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
