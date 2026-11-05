"""The typed-decision contract. PROTECTED — editable files may not change this.

One `Case` is one state plus several typed questions, exactly the body of a
`POST /v1/systemone` request. A model answers every question of a case from one
encoding of the state, and the questions may not read each other.

A prediction is always a full distribution. The scalar readouts (`argmax` for
choice, `p(true)` for noul, `sum(i*p_i)` for score) are derived here so that no
model implementation can quietly use a different one.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

Mode = Literal["choice", "noul", "score"]
MODES: tuple[Mode, ...] = ("choice", "noul", "score")


@dataclass(frozen=True)
class Question:
    key: str
    mode: Mode
    instructions: str
    options: tuple[str, ...]          # ordered option keys; noul is ("false", "true")
    criteria: dict[str, str]          # option key -> its written description

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"{self.key}: unknown mode {self.mode!r}")
        if len(self.options) < 2:
            raise ValueError(f"{self.key}: needs at least two options")
        if self.mode == "noul" and tuple(self.options) != ("false", "true"):
            raise ValueError(f"{self.key}: noul options must be ('false','true')")
        missing = [o for o in self.options if o not in self.criteria]
        if missing:
            raise ValueError(f"{self.key}: options without criteria: {missing}")


@dataclass(frozen=True)
class Case:
    case_id: str
    source: str                        # which corpus it came from; used for deletion by source
    state: str                         # the document the questions are asked about
    questions: tuple[Question, ...]
    gold: dict[str, tuple[float, ...]]  # question key -> gold distribution over that question's options

    def __post_init__(self) -> None:
        for q in self.questions:
            g = self.gold.get(q.key)
            if g is None:
                raise ValueError(f"{self.case_id}/{q.key}: no gold")
            if len(g) != len(q.options):
                raise ValueError(f"{self.case_id}/{q.key}: gold has {len(g)} entries for "
                                 f"{len(q.options)} options")
            if abs(sum(g) - 1.0) > 1e-4:
                raise ValueError(f"{self.case_id}/{q.key}: gold sums to {sum(g)}")


@dataclass
class Prediction:
    """A distribution over one question's options, in the question's own order."""
    probs: tuple[float, ...]

    def __post_init__(self) -> None:
        if any(p < 0 for p in self.probs):
            raise ValueError("negative probability")
        total = sum(self.probs)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("probabilities do not sum to a positive finite number")
        if abs(total - 1.0) > 1e-4:
            self.probs = tuple(p / total for p in self.probs)

    # --- the three readouts, defined once ---
    def choice(self, options: Sequence[str]) -> str:
        return options[max(range(len(self.probs)), key=self.probs.__getitem__)]

    def noul(self) -> float:
        return self.probs[1]                       # options are ("false", "true")

    def score(self) -> float:
        return sum(i * p for i, p in enumerate(self.probs))

    def confidence(self) -> float:
        """Used only to RANK decisions for selective prediction, never to score them."""
        return max(self.probs)


def gold_label(q: Question, gold: Sequence[float]) -> str:
    return q.options[max(range(len(gold)), key=gold.__getitem__)]


def is_correct(q: Question, pred: Prediction, gold: Sequence[float]) -> bool:
    return pred.choice(q.options) == gold_label(q, gold)


def load_cases(path: str) -> list[Case]:
    """Read a JSONL corpus written by data.py."""
    out: list[Case] = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            qs = tuple(
                Question(key=q["key"], mode=q["mode"], instructions=q["instructions"],
                         options=tuple(q["options"]), criteria=q["criteria"])
                for q in r["questions"]
            )
            out.append(Case(case_id=r["case_id"], source=r["source"], state=r["state"],
                            questions=qs,
                            gold={k: tuple(v) for k, v in r["gold"].items()}))
    return out


def dump_cases(cases: Iterable[Case], path: str) -> int:
    n = 0
    with open(path, "w") as fh:
        for c in cases:
            fh.write(json.dumps({
                "case_id": c.case_id, "source": c.source, "state": c.state,
                "questions": [{"key": q.key, "mode": q.mode, "instructions": q.instructions,
                               "options": list(q.options), "criteria": q.criteria}
                              for q in c.questions],
                "gold": {k: list(v) for k, v in c.gold.items()},
            }) + "\n")
            n += 1
    return n
