"""EDITABLE — the data axis: collect, generate, DELETE.

Generalist rule: train on workflows we do not evaluate on.
`LocalLLaMA/typed-decisions` train is deliberately NOT a source here. Training
on it would turn every typed-decisions number into a specialist number and break
comparability with Jev. It is excluded precisely because it would raise the
headline.

Deletion is a first-class operation, not cleanup: in the closest published
replication 59% of errors came from three noisy-label sources that were only 37%
of the items.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .contract import Case, Question

# --------------------------------------------------------------------- sources
# Public classification corpora recast as typed decisions. These are the ten kev
# trains on; its headline is transfer to six never-trained sources.


@dataclass(frozen=True)
class Source:
    name: str
    hf_id: str
    config: str | None
    split: str
    text_field: str | tuple[str, ...]
    label_field: str
    mode: str
    labels: tuple[str, ...]
    criteria: dict[str, str]
    instructions: str
    # Dataset label name -> our option key. noul REQUIRES ("false","true"), and a
    # corpus whose own names are ("neg","pos") must be mapped, not renamed by
    # position, or the polarity silently inverts when a dataset reorders.
    label_map: dict[str, str] | None = None


def _c(labels: Sequence[str], descs: Sequence[str]) -> dict[str, str]:
    return dict(zip(labels, descs))


SENTIMENT5 = ("very negative", "negative", "neutral", "positive", "very positive")

SOURCES: dict[str, Source] = {
    "boolq": Source("boolq", "google/boolq", None, "train", ("passage", "question"),
                    "answer", "noul", ("false", "true"),
                    _c(("false", "true"), ("The passage does not support the statement.",
                                           "The passage supports the statement.")),
                    "The answer to the question is yes."),
    "ag_news": Source("ag_news", "fancyzhx/ag_news", None, "train", "text", "label",
                      "choice", ("world", "sports", "business", "sci_tech"),
                      _c(("world", "sports", "business", "sci_tech"),
                         ("International affairs, politics, conflict.",
                          "Sport, athletes, fixtures and results.",
                          "Companies, markets, trade and the economy.",
                          "Science and technology.")),
                      "Which desk should this story be filed under?"),
    # TREC is distributed as a loading script, which `datasets` no longer runs.
    # Emotion replaces it: same role (short text, many classes, noisy labels) and
    # it is one of the three noisy-label sources that produced 59% of errors in
    # the closest published replication, so it is exactly what the deletion axis
    # should be tested against.
    "emotion": Source("emotion", "dair-ai/emotion", "split", "train", "text", "label",
                      "choice", ("sadness", "joy", "love", "anger", "fear", "surprise"),
                      _c(("sadness", "joy", "love", "anger", "fear", "surprise"),
                         ("The writer feels sad, disappointed or low.",
                          "The writer feels happy, pleased or elated.",
                          "The writer feels affection or tenderness.",
                          "The writer feels angry, irritated or resentful.",
                          "The writer feels afraid, anxious or threatened.",
                          "The writer feels surprised or startled.")),
                      "What emotion does this text express?"),
    "sst5": Source("sst5", "SetFit/sst5", None, "train", "text", "label",
                   "score", tuple(str(i) for i in range(5)),
                   _c(tuple(str(i) for i in range(5)), SENTIMENT5),
                   "How positive is the sentiment of this text?"),
    "yelp": Source("yelp", "Yelp/yelp_review_full", None, "train", "text", "label",
                   "score", tuple(str(i) for i in range(5)),
                   _c(tuple(str(i) for i in range(5)),
                      ("One star: the reviewer was very unhappy.",
                       "Two stars: the reviewer was unhappy.",
                       "Three stars: the reviewer was mixed.",
                       "Four stars: the reviewer was happy.",
                       "Five stars: the reviewer was delighted.")),
                   "How many stars did this review give?"),
    "imdb": Source("imdb", "stanfordnlp/imdb", None, "train", "text", "label",
                   "noul", ("false", "true"),
                   _c(("false", "true"), ("The reviewer disliked the film.",
                                          "The reviewer liked the film.")),
                   "This review is positive.",
                   label_map={"neg": "false", "pos": "true"}),
    "dbpedia": Source("dbpedia", "fancyzhx/dbpedia_14", None, "train", "content", "label",
                      "choice", tuple(f"class_{i}" for i in range(14)),
                      _c(tuple(f"class_{i}" for i in range(14)), tuple("" for _ in range(14))),
                      "Which category does this article describe?"),
    "banking77": Source("banking77", "legacy-datasets/banking77", None, "train", "text",
                        "label", "choice", (), {}, "What is the customer asking about?"),
    "mnli": Source("mnli", "nyu-mll/multi_nli", None, "train", ("premise", "hypothesis"),
                   "label", "choice", ("entailment", "neutral", "contradiction"),
                   _c(("entailment", "neutral", "contradiction"),
                      ("The hypothesis follows from the premise.",
                       "The hypothesis is neither supported nor contradicted.",
                       "The hypothesis contradicts the premise.")),
                   "What is the relationship between premise and hypothesis?"),
    "amazon": Source("amazon", "SetFit/amazon_reviews_multi_en", None, "train",
                     ("title", "text"), "label", "score", tuple(str(i) for i in range(5)),
                     _c(tuple(str(i) for i in range(5)),
                        ("One star.", "Two stars.", "Three stars.", "Four stars.", "Five stars.")),
                     "How many stars did this review give?"),
}

# Eleventh source, ours, not public: generated policy cases. See `generate_policy`.


# ------------------------------------------------------------------ collection
def _hard(n: int, i: int) -> tuple[float, ...]:
    return tuple(1.0 if k == i else 0.0 for k in range(n))


def recast(source: Source, rows: Iterable[dict], limit: int | None = None) -> list[Case]:
    """One row of a classification corpus becomes one single-question case.

    Hard labels become one-hot gold. That is a CHOICE, not a fact: it asserts the
    annotator was certain. `soften` below is the counterfactual.
    """
    out: list[Case] = []
    fields = (source.text_field,) if isinstance(source.text_field, str) else source.text_field
    for n, row in enumerate(rows):
        if limit is not None and n >= limit:
            break
        state = "\n\n".join(f"{f}: {row[f]}" for f in fields)
        label = row[source.label_field]
        idx = int(label) if not isinstance(label, str) else source.labels.index(label)
        q = Question(key="label", mode=source.mode, instructions=source.instructions,
                     options=source.labels, criteria=source.criteria)
        cid = f"{source.name}_{hashlib.sha256(state.encode()).hexdigest()[:12]}"
        out.append(Case(cid, source.name, state, (q,),
                        {"label": _hard(len(source.labels), idx)}))
    return out


# -------------------------------------------------------------------- generate
def generate_unknowable(rng: random.Random, n: int,
                        template: Callable[[random.Random], tuple[str, Question]]) -> list[Case]:
    """Evidence-free cases whose gold is UNIFORM.

    The one data-side calibration intervention with a published positive result:
    it drove the share of evidence-free items answered at >= 0.9 confidence from
    0.19 to 0.00 at 4B, and 0.05 to 0.00 at 9B, with controls unchanged.
    """
    out = []
    for i in range(n):
        state, q = template(rng)
        k = len(q.options)
        out.append(Case(f"unknowable_{i:06d}", "generated_unknowable", state, (q,),
                        {q.key: tuple(1.0 / k for _ in range(k))}))
    return out


def soften(cases: Sequence[Case], eps: float) -> list[Case]:
    """Move mass off the one-hot label. NOT equivalent to a temperature.

    Label smoothing raised accuracy slightly and moved AURC +0.042 [+0.023,
    +0.065] in the published screen -- it destroyed ranking. Kept here because it
    is the control every soft-target claim must beat, not because it is expected
    to work.
    """
    out = []
    for c in cases:
        g = {k: tuple((1 - eps) * p + eps / len(v) for p in v) for k, v in c.gold.items()
             for v in (c.gold[k],)}
        out.append(Case(c.case_id, c.source, c.state, c.questions, g))
    return out


# -------------------------------------------------------------------- deletion
def delete_by_source(cases: Sequence[Case], drop: set[str]) -> list[Case]:
    return [c for c in cases if c.source not in drop]


def deduplicate(cases: Sequence[Case]) -> list[Case]:
    seen: set[str] = set()
    out = []
    for c in cases:
        h = hashlib.sha256(c.state.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(c)
    return out


def delete_by_gold_entropy(cases: Sequence[Case], *, keep_above: float | None = None,
                           keep_below: float | None = None) -> list[Case]:
    """Filter on how ambiguous the gold is (mean normalised entropy over questions).

    Both directions are hypotheses: dropping the ambiguous items removes label
    noise, and dropping the certain ones concentrates the budget on what is hard.
    """
    out = []
    for c in cases:
        hs = []
        for q in c.questions:
            g = c.gold[q.key]
            h = -sum(p * math.log(max(p, 1e-12)) for p in g) / math.log(len(g))
            hs.append(h)
        m = sum(hs) / len(hs)
        if keep_above is not None and m < keep_above:
            continue
        if keep_below is not None and m > keep_below:
            continue
        out.append(c)
    return out


def delete_by_difficulty(cases: Sequence[Case], per_case_loss: dict[str, float],
                         *, drop_fraction: float, hardest: bool) -> list[Case]:
    """Prune by a model's own loss. `hardest=True` drops what it cannot fit
    (suspected label noise); `hardest=False` drops what it already fits
    (suspected redundancy). Which one helps is an open question here."""
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError("drop_fraction must be in [0, 1)")
    ranked = sorted(cases, key=lambda c: per_case_loss.get(c.case_id, 0.0),
                    reverse=hardest)
    cut = int(len(ranked) * drop_fraction)
    return ranked[cut:]


# ---------------------------------------------------------------- the hard rule
def assert_disjoint(train: Sequence[Case], evaluation: Sequence[Case]) -> None:
    """No evaluation row, in any form, may appear in training. Verified, not assumed."""
    ev = {hashlib.sha256(c.state.encode()).hexdigest() for c in evaluation}
    hit = [c.case_id for c in train
           if hashlib.sha256(c.state.encode()).hexdigest() in ev]
    if hit:
        raise AssertionError(f"{len(hit)} training cases overlap evaluation, e.g. {hit[:5]}")
