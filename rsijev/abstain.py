"""Reserved abstention option (features, 2026-09-29). Torch-free, so the wire layer can
use it.

A question that allows abstention gets ONE extra option block, rendered by the encoder
exactly like any other option (`- <key>: <text>`), so the readout scores "cannot tell"
in the same single forward pass, with no extra head (knowledge/arch_goal_plan.md
advises against extra heads). The key and text are fixed: training recasts every
abstention source into exactly this block (tools/features/recast_abstain.py), and
serving strips it back out of the answer. A question without abstention is untouched,
so its encoding is byte-identical to before.

    unknown_probability = p(reserved option)      from the full softmax
    probabilities       = the real options, renormalized to sum to 1
    abstained           = unknown_probability >= tau, tau calibrated on a dev set
                          (tools/features/calibrate_tau.py; split-conformal bound on
                          the false-abstention rate over answerable questions)
"""
from __future__ import annotations

from typing import Sequence

from .contract import Question

ABSTAIN_KEY = "Not enough information"
ABSTAIN_TEXT = "The information given is not enough to choose any other option."


def is_abstain_question(q: Question) -> bool:
    return (len(q.options) >= 3 and q.options[-1] == ABSTAIN_KEY
            and q.criteria.get(ABSTAIN_KEY) == ABSTAIN_TEXT)


def with_abstain(q: Question) -> Question:
    """`q` plus the reserved option, appended last in canonical order.

    Choice questions only: noul's options are fixed to ("false", "true") by the
    contract, and a score question's levels are an ordinal scale that an extra
    non-level option would break. A question that already uses the reserved key is
    refused rather than silently merged."""
    if q.mode != "choice":
        raise ValueError(f"{q.key}: abstention is supported for choice questions only")
    if ABSTAIN_KEY in q.options:
        raise ValueError(f"{q.key}: option key {ABSTAIN_KEY!r} is reserved")
    return Question(key=q.key, mode=q.mode, instructions=q.instructions,
                    options=tuple(q.options) + (ABSTAIN_KEY,),
                    criteria={**q.criteria, ABSTAIN_KEY: ABSTAIN_TEXT})


def split_abstain(probs: Sequence[float]) -> tuple[list[float], float]:
    """Distribution over a `with_abstain` question -> (real options renormalized,
    unknown_probability). The reserved option is the last canonical option."""
    p = [float(x) for x in probs]
    unknown, real = p[-1], p[:-1]
    total = sum(real)
    real = [x / total for x in real] if total > 0 else [1.0 / len(real)] * len(real)
    return real, unknown


def conformal_tau(p_unknown_answerable: Sequence[float], alpha: float) -> float:
    """Split-conformal threshold: abstaining when p_unknown >= tau flags at most an
    alpha share of answerable questions (exchangeable with the calibration set),
    with the finite-sample (n+1) correction. Returns a value just above the
    ceil((n+1)(1-alpha))-th smallest calibration score; 1.0+ (never abstain) if the
    set is too small for alpha."""
    s = sorted(float(x) for x in p_unknown_answerable)
    n = len(s)
    import math
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return math.nextafter(s[k - 1], float("inf"))
