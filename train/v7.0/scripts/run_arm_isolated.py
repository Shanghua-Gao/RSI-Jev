"""Run ONE arm against an agent's own code snapshot, in a fresh process.

The persistent worker runs arms in-process on the code it booted with. An
AutoScientists agent instead edits its own copy of the model, data and training
code, so each of its arms names a snapshot directory (`code_dir` in the spec) and
the worker hands it here. A fresh process is the only clean way to import a
different `rsijev` package; a model load costs ~30 s against ~13 min of training.

Two checks run before anything is loaded, and either one refuses the arm:

1. Evaluator integrity. The files that define WHAT is measured -- targets,
   metrics, scoring, the contract, the arm runner itself -- must be byte-identical
   to the deployed tree's. An agent may change how the model reads and trains;
   it may not change how the result is scored.
2. Test overlap. No training case may share a state with the typed-decisions TEST
   split, ever. The train split is refused too, unless the spec sets
   "specialist": true (run owner, 2026-09-24: a specialist track beside the
   generalist one, every result labelled with its track). Exact match after
   whitespace normalisation: paraphrase is not detectable here, which is why
   generated data also carries its own builder check. The eval suite and the
   frozen held-out set eval_final_v3 are banned the same way, always, and so is
   every question text of the knowledge guard kg_v1 (its rows have no state).

eval_final_v3 is scored as targets "final.<benchmark>" (canonical order, no
control), kept out of the suite weights; as_cluster pools them as final_clean.
kg_v1 replaces mmlu_pro_1k (DI test rows, KB-25 / A77) and is canonical-only.

Exit code 0 on success, 2 on a refused arm, anything else on a crash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

REF_ROOT = Path(__file__).resolve().parent.parent

# What an arm may not change. Everything under rsijev/ and scripts/ that is NOT
# listed here is the agents' search surface (arch.py, encode.py, fit.py,
# train.py, data.py, build_*.py).
PROTECTED = (
    "rsijev/evaluate.py",
    "rsijev/metrics.py",
    "rsijev/targets.py",
    "rsijev/contract.py",
    "rsijev/suite.py",
    "scripts/run_arm_lib.py",
    "scripts/run_arm_isolated.py",
    "scripts/worker.py",
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def integrity_violations(code_dir: Path) -> list[str]:
    bad = []
    for rel in PROTECTED:
        mine, ref = code_dir / rel, REF_ROOT / rel
        if not mine.exists():
            bad.append(f"{rel} missing")
        elif _sha(mine) != _sha(ref):
            bad.append(f"{rel} differs from the deployed evaluator")
    return bad


def _norm(state: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", state).strip().lower().encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, help="default corpus dir; spec corpus_dir overrides")
    args = ap.parse_args()

    code_dir = Path(args.code_dir).resolve()
    bad = integrity_violations(code_dir)
    if bad:
        print("EVALUATOR_INTEGRITY_VIOLATION: " + "; ".join(bad), file=sys.stderr)
        return 2

    # The snapshot's packages shadow the deployed ones for this process only.
    sys.path[:0] = [str(code_dir), str(code_dir / "scripts")]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import run_arm_lib as lib
    from rsijev.contract import load_cases
    from rsijev.targets import load_kg_v1, load_typed_decisions
    assert Path(lib.__file__).resolve().parent == code_dir / "scripts", lib.__file__

    spec = json.loads(Path(args.spec).read_text())
    corpus_dir = Path(spec.get("corpus_dir") or args.corpus)
    corpus = {f.stem: load_cases(str(f)) for f in sorted(corpus_dir.glob("*.jsonl"))}
    if not corpus:
        print(f"REFUSED: no *.jsonl corpus files in {corpus_dir}", file=sys.stderr)
        return 2

    td_test = load_typed_decisions("test")
    banned = {_norm(c.state) for c in td_test}
    hits = [(src, c.case_id) for src, cs in corpus.items() for c in cs if _norm(c.state) in banned]
    if hits:
        print(f"TEST_OVERLAP: {len(hits)} training cases share a state with the typed-decisions "
              f"TEST split, e.g. {hits[:3]}", file=sys.stderr)
        return 2
    # Run owner, 2026-09-24: two tracks. The train split is allowed only in arms
    # marked "specialist": true, which are labelled as such, as the card asks.
    td_train = {_norm(c.state) for c in load_typed_decisions("train")}
    n_train = sum(_norm(c.state) in td_train for cs in corpus.values() for c in cs)
    if n_train and not spec.get("specialist"):
        print(f"TEST_OVERLAP: {n_train} training cases come from the typed-decisions train split, "
              f"but the spec is not marked \"specialist\": true", file=sys.stderr)
        return 2
    print(f"track: {'specialist' if spec.get('specialist') else 'generalist'} "
          f"({n_train} typed-decisions train cases in the corpus)", flush=True)
    # The eval suite's test items are banned exactly like typed-decisions test.
    from rsijev.suite import load_eval_suite
    suite = load_eval_suite()      # always: no spec field can skip the suite or its overlap check
    suite_states = {_norm(c.state) for cs in suite.values() for c in cs}
    s_hits = [(src, c.case_id) for src, cs in corpus.items() for c in cs if _norm(c.state) in suite_states]
    if s_hits:
        print(f"TEST_OVERLAP: {len(s_hits)} training cases share a state with the eval suite, "
              f"e.g. {s_hits[:3]}", file=sys.stderr)
        return 2
    # The frozen held-out set eval_final_v3 (runbook A15/A77) is banned the same way.
    # Always loaded, sha256-verified, and no spec field can skip it or its check.
    from rsijev.suite import FINAL_PREFIX, load_final_set
    final = load_final_set()
    final_states = {_norm(c.state) for cs in final.values() for c in cs if c.state.strip()}
    f_hits = [(src, c.case_id) for src, cs in corpus.items() for c in cs if _norm(c.state) in final_states]
    if f_hits:
        print(f"TEST_OVERLAP: {len(f_hits)} training cases share a state with eval_final_v3, "
              f"e.g. {f_hits[:3]}", file=sys.stderr)
        return 2
    # kg_v1 rows carry the question in the instructions (state ""), so its ban is on
    # question text: no training state or question may repeat a kg_v1 question
    # (MMLU-Pro-style training rows put the question in the state).
    kg = load_kg_v1()
    kg_q = {_norm(q.instructions) for c in kg for q in c.questions}
    k_hits = [(src, c.case_id) for src, cs in corpus.items() for c in cs
              if _norm(c.state) in kg_q or any(_norm(q.instructions) in kg_q for q in c.questions)]
    if k_hits:
        print(f"TEST_OVERLAP: {len(k_hits)} training cases repeat a kg_v1 question, "
              f"e.g. {k_hits[:3]}", file=sys.stderr)
        return 2
    print(f"overlap check: corpus vs typed-decisions test, eval suite ({len(suite_states)} states), "
          f"eval_final_v3 ({len(final_states)} states), kg_v1 ({len(kg_q)} questions): clean", flush=True)
    # Targets "final.<benchmark>". Not in FULL_EVAL_TARGETS, so run_arm
    # scores them canonical-only without a control, like suite targets. The names
    # can never collide with a suite / KEEP target (as_cluster.SUITE_WEIGHTS).
    final_targets = {FINAL_PREFIX + b: cs for b, cs in final.items()}
    assert not (set(final_targets) & (set(suite) | lib.FULL_EVAL_TARGETS
                                      | {"typed_decisions", "kg_v1"})), final_targets.keys()
    assert "kg_v1" not in suite, suite.keys()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    targets = {"kg_v1": kg, "typed_decisions": td_test, **suite,
               **final_targets}
    print(f"[isolated {args.name}] code={code_dir} corpus={corpus_dir} "
          f"({sum(len(v) for v in corpus.values())} cases) loaded in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)

    out = Path(args.out)
    records = lib.run_arm(lm=lm, tok=tok, targets=targets, corpus=corpus, device=dev,
                          spec=spec, name=args.name,
                          items_path=out / f"{args.name}.items.jsonl")
    # A snapshot has no .git, so code_version reads "unknown"; stamp a content
    # hash of exactly the code that ran instead.
    h = hashlib.sha256()
    for f in sorted([*code_dir.glob("rsijev/*.py"), *code_dir.glob("scripts/*.py")]):
        h.update(f.relative_to(code_dir).as_posix().encode())
        h.update(f.read_bytes())
    for r in records:
        r["code_dir"] = str(code_dir)
        r["code_hash"] = h.hexdigest()[:12]
    with open(out / f"{args.name}.jsonl", "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
