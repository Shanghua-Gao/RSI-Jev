"""The two evaluation targets. PROTECTED — arms may not change what they are
scored on.

Both are GENERALIST evaluations: nothing from either target's own training data
enters training. For typed-decisions that is a deliberate cost — its train split
exists and using it would raise the headline, at the price of turning every
number into a specialist number that cannot be set beside Jev's.
"""
from __future__ import annotations

import json
import random
from typing import Sequence

from .contract import Case, Question

MMLU_PRO = "TIGER-Lab/MMLU-Pro"
TYPED_DECISIONS = "LocalLLaMA/typed-decisions"
PUBLISHED_1K_SEED = 42
PUBLISHED_1K_N = 1000


def _one_hot(n: int, i: int) -> tuple[float, ...]:
    return tuple(1.0 if k == i else 0.0 for k in range(n))


def load_mmlu_pro_1k(*, revision: str | None = None) -> list[Case]:
    """The 1,000 rows of the published OpenJev evaluation.

    Reproduced by construction, not by copying: `random.Random(42).sample(
    range(len(test)), 1000)` over the pinned test split, which is how the
    published subset is defined. This is the only set in the whole programme
    that has never taken part in any selection decision.
    """
    from datasets import load_dataset

    ds = load_dataset(MMLU_PRO, split="test", revision=revision)
    idx = random.Random(PUBLISHED_1K_SEED).sample(range(len(ds)), PUBLISHED_1K_N)
    cases: list[Case] = []
    for rank, i in enumerate(idx):
        row = ds[i]
        opts = [o for o in row["options"] if o is not None]
        letters = tuple("ABCDEFGHIJKLMNOP"[: len(opts)])
        q = Question(key="answer", mode="choice",
                     instructions=row["question"],
                     options=letters,
                     criteria=dict(zip(letters, opts)))
        gold_i = letters.index(row["answer"]) if row["answer"] in letters else row["answer_index"]
        cases.append(Case(case_id=f"mmlu_pro_{row['question_id']}", source="mmlu_pro_1k",
                          state="", questions=(q,),
                          gold={"answer": _one_hot(len(letters), gold_i)}))
    return cases


def load_typed_decisions(split: str = "test") -> list[Case]:
    """Cases with SOFT gold: the mean of three teacher samples per question.

    Read the ceilings before quoting any number from this set: majority 0.520,
    latent-factor bound 0.704, teacher self-agreement 0.735, and the dataset's
    own warning that much above 0.75 means the teacher's quirks were learned.
    """
    from datasets import load_dataset

    ds = load_dataset(TYPED_DECISIONS, "all", split=split)
    cases: list[Case] = []
    for row in ds:
        state = row["state"]
        qs_raw = json.loads(row["questions"])
        gold_raw = json.loads(row["gold"])
        questions, gold = [], {}
        for key, q in qs_raw.items():
            g = gold_raw.get(key)
            if g is None or "probabilities" not in g:
                continue
            probs = g["probabilities"]
            raw = q.get("criteria")
            if q["type"] == "noul":
                # 200 of the 600 noul questions carry no criteria at all -- they
                # are bare statements. Supplying a default keeps the option
                # blocks uniform instead of silently encoding two layouts.
                options = ("false", "true")
                criteria = raw if isinstance(raw, dict) else {
                    "false": "The statement is false.",
                    "true": "The statement is true."}
            elif isinstance(raw, dict):
                options = tuple(raw.keys())
                criteria = raw
            elif isinstance(raw, list):                 # score rubrics are ordered lists
                options = tuple(str(i) for i in range(len(raw)))
                criteria = {str(i): d for i, d in enumerate(raw)}
            else:
                continue
            vec = [float(probs.get(o, 0.0)) for o in options]
            total = sum(vec)
            if total <= 0:
                continue
            vec = [v / total for v in vec]
            questions.append(Question(key=key, mode=q["type"],
                                      instructions=q["instructions"],
                                      options=options, criteria=criteria))
            gold[key] = tuple(vec)
        if questions:
            cases.append(Case(case_id=row["id"], source=f"typed_decisions_{split}",
                              state=state, questions=tuple(questions), gold=gold))
    return cases


# The published reference points, so that no report has to look them up.
TYPED_DECISIONS_CEILINGS = {
    "majority_baseline": 0.520,
    "latent_factor_bound": 0.704,
    "teacher_self_agreement": 0.735,
    "quirk_line": 0.75,
}

EXTERNAL = {
    "mmlu_pro_1k": {"jev": 0.829, "ekzhang_sft": 0.71, "openjev": 0.588,
                    "frozen_35b_baseline_phase1": 0.584,
                    "kev_9b": 0.511, "kev_4b": 0.468},
    "typed_decisions": {"jev_generalist": 0.727,
                        "verdict2_specialist": 0.771, "laya_specialist": 0.766,
                        "tfidf_logreg": 0.661},
}
