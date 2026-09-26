"""SPECIALIST TRACK corpus: spec-1-x3's corpus + general multiple-choice replay.

    OUT/synth.jsonl      copied from --base (spec-1-x3: champion synth, DeepSeek gold), md5-checked
    OUT/td_train.jsonl   copied from --base (typed-decisions TRAIN x3), md5-checked
    OUT/mc_replay.jsonl  choice-mode cases sampled EVENLY from the --replay-sources pool_v2 files,
                         sized so replay = --frac of the corpus's total questions
    OUT/build_report.json

Replay filters (all counted in the report):
  * choice-mode, single-question cases only (the pool's rare score rows are dropped, so the
    replay does not touch the score prior);
  * dedup across sources on the normalised (state, question, options) text: MMLU's
    auxiliary_train is itself built from ARC / OBQA / RACE, so the same item can appear twice;
  * typed-decisions TEST: exact normalised state, and 8-gram containment >= 0.5 of any test state;
  * MMLU-Pro-1k (the guard): exact question, and 8-gram containment >= 0.5 of any MMLU-Pro-1k
    question or question+options.
The whole corpus (synth + td_train + replay) is then re-checked against the test split with the
runner's own normalisation (exact state, case_id, test-state substring) and refused on any hit.

    python scripts/build_specialist_replay_corpus.py --base DIR --pool DIR --out DIR --frac 0.15
Train with --set sources='"synth,td_train,mc_replay"' --set specialist=true.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.contract import dump_cases, load_cases  # noqa: E402
from rsijev.targets import load_mmlu_pro_1k, load_typed_decisions  # noqa: E402

SOURCES = "dc_mmlu,dc_arc,dc_openbookqa,dc_commonsense_qa,dc_sciq,dc_race"
BASE_MD5 = {"synth.jsonl": "8ddc122ad80babb2dd7860cf855e22b5",
            "td_train.jsonl": "5a8df853325f6165c21fbf8a42bc863d"}


def _norm(state: str) -> str:
    # Whitespace-collapsed, lower-cased, hashed. The overlap checks and the arm
    # runner must agree on this exactly, or a contamination check silently passes.
    return hashlib.sha256(re.sub(r"\s+", " ", state).strip().lower().encode()).hexdigest()


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _grams(s: str, n: int = 8) -> set[int]:
    w = re.findall(r"\w+", s.lower())
    return {hash(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


def md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Containment:
    """Flags a candidate text that contains >= thr of the 8-grams of any target text."""

    def __init__(self, targets: dict[str, str], thr: float = 0.5):
        self.thr, self.size, self.index = thr, {}, collections.defaultdict(list)
        for tid, text in targets.items():
            g = _grams(text)
            if not g:
                continue
            self.size[tid] = len(g)
            for x in g:
                self.index[x].append(tid)

    def hit(self, text: str) -> tuple[str, float] | None:
        cnt = collections.Counter(t for x in _grams(text) for t in self.index.get(x, ()))
        best = max(((t, c / self.size[t]) for t, c in cnt.items()), key=lambda z: z[1], default=None)
        return best if best and best[1] >= self.thr else None


def case_text(c) -> str:
    q = c.questions[0]
    return c.state + " " + q.instructions + " " + " ".join(q.criteria.get(o, "") for o in q.options)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="spec-1-x3 corpus dir (synth.jsonl, td_train.jsonl)")
    ap.add_argument("--pool", required=True, help="dir holding the pool_v2 replay source files")
    ap.add_argument("--sources", default=SOURCES)
    ap.add_argument("--frac", type=float, required=True, help="replay share of TOTAL questions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--base-steps", type=int, default=2610, help="spec-1-x3 steps")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    base = Path(a.base)
    for f, want in BASE_MD5.items():
        got = md5(base / f)
        if got != want:
            raise SystemExit(f"{base / f} md5 {got} != spec-1-x3's {want}")
    synth = load_cases(str(base / "synth.jsonl"))
    td = load_cases(str(base / "td_train.jsonl"))
    q_base = sum(len(c.questions) for c in synth + td)
    n_replay = round(a.frac * q_base / (1 - a.frac))

    test = load_typed_decisions("test")
    mmlu = load_mmlu_pro_1k()
    test_state = {_norm(c.state) for c in test}
    test_ct = Containment({c.case_id: c.state for c in test})
    mmlu_q = {_flat(c.questions[0].instructions) for c in mmlu}
    mmlu_ct = Containment({**{c.case_id + ":q": c.questions[0].instructions for c in mmlu},
                           **{c.case_id + ":qo": case_text(c) for c in mmlu}})

    names = a.sources.split(",")
    per = {n: n_replay // len(names) + (1 if i < n_replay % len(names) else 0) for i, n in enumerate(names)}
    seen: set[str] = set()
    drops = {n: collections.Counter() for n in names}
    examples = collections.defaultdict(list)
    replay, per_source = [], {}
    for n in names:
        cs = load_cases(str(Path(a.pool) / f"{n}.jsonl"))
        random.Random(f"{a.seed}:{n}").shuffle(cs)
        take = []
        for c in cs:
            if len(take) == per[n]:
                break
            if len(c.questions) != 1 or c.questions[0].mode != "choice":
                drops[n]["not_single_choice"] += 1
                continue
            q = c.questions[0]
            key = _norm(c.state + "\x00" + q.instructions + "\x00" + "\x00".join(q.criteria.get(o, "") for o in q.options))
            if key in seen:
                drops[n]["dup_across_sources"] += 1
                continue
            if _norm(c.state) in test_state:
                drops[n]["td_test_exact"] += 1
                continue
            h = test_ct.hit(c.state) if c.state else None
            if h:
                drops[n]["td_test_8gram"] += 1
                examples[n].append((c.case_id, h))
                continue
            if _flat(q.instructions) in mmlu_q:
                drops[n]["mmlu_pro_exact"] += 1
                continue
            h = mmlu_ct.hit(case_text(c))
            if h:
                drops[n]["mmlu_pro_8gram"] += 1
                examples[n].append((c.case_id, h))
                continue
            seen.add(key)
            take.append(c)
        if len(take) < per[n]:
            raise SystemExit(f"{n}: only {len(take)} clean cases, need {per[n]}")
        per_source[n] = {"in_file": len(cs), "taken": len(take), "dropped_while_scanning": dict(drops[n]),
                         "options_hist": dict(collections.Counter(len(x.questions[0].options) for x in take))}
        replay += take

    # Whole-corpus re-check with the runner's own normalisation (refuse on any hit).
    allc = synth + td + replay
    ids = {c.case_id for c in test}
    hits = {"exact_state": sum(_norm(c.state) in test_state for c in allc),
            "case_id": sum(c.case_id in ids for c in allc)}
    joined = "\x00".join(_flat(c.state) for c in allc)
    hits["test_state_substring_in_corpus"] = sum(_flat(c.state) in joined for c in test)
    hits["replay_mmlu_pro_exact_question"] = sum(_flat(c.questions[0].instructions) in mmlu_q for c in replay)
    if any(hits.values()):
        raise SystemExit(f"OVERLAP {hits}; refusing to write")

    shutil.copy(base / "synth.jsonl", out / "synth.jsonl")
    shutil.copy(base / "td_train.jsonl", out / "td_train.jsonl")
    dump_cases(replay, str(out / "mc_replay.jsonl"))

    def qc(cs):
        return dict(collections.Counter(q.mode for c in cs for q in c.questions))
    q_rep = sum(len(c.questions) for c in replay)
    total = q_base + q_rep
    rep = {
        "track": "specialist", "base": str(base), "base_md5": BASE_MD5, "base_questions": q_base,
        "frac_target": a.frac, "replay_questions": q_rep, "replay_frac_actual": round(q_rep / total, 4),
        "replay_modes": qc(replay), "replay_per_source": per_source, "replay_md5": md5(out / "mc_replay.jsonl"),
        "overlap_examples_dropped": {k: v[:5] for k, v in examples.items()},
        "overlap_final_check": {**hits, "test_cases_checked": len(test), "mmlu_pro_items_checked": len(mmlu)},
        "total_questions": total,
        "steps_scaled": round(a.base_steps * total / q_base),
        "seed": a.seed,
    }
    try:
        from corpus_gate import check
        ok, fails, _ = check(str(out), str(Path(__file__).with_name("corpus_reference_stats.json")))
        rep["corpus_gate"] = {"pass": bool(ok), "fails": fails}
    except Exception as e:
        rep["corpus_gate"] = {"error": repr(e)}
    (out / "build_report.json").write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
