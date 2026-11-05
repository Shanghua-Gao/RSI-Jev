"""Metrics. PROTECTED — editable files may not change this.

Primary: tie-aware coverage at an error budget, and AURC. Secondary, reported
but never primary: accuracy, ECE, proper scores, per-mode decision scores.

Two things here are deliberate and were wrong in an earlier public
implementation of the same idea:

*  Selective metrics admit **whole tie groups**. Splitting a group of equally
   confident decisions makes the result depend on row order — the same
   predictions could read 0.94 or 0.00 under a permutation.
*  Coverage takes the **largest** admissible prefix, not the first violation.
   Risk is not monotone in coverage, so "stop at the first breach" understates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------- proper scores
def log_score(probs: Sequence[float], gold: Sequence[float]) -> float:
    """Expected log score under the gold distribution. Higher is better."""
    return sum(g * math.log(max(p, 1e-12)) for p, g in zip(probs, gold))


def brier(probs: Sequence[float], gold: Sequence[float]) -> float:
    """Squared error against the gold distribution. Lower is better."""
    return sum((p - g) ** 2 for p, g in zip(probs, gold))


def spherical(probs: Sequence[float], gold: Sequence[float]) -> float:
    norm = math.sqrt(sum(p * p for p in probs)) or 1e-12
    return sum(g * p for p, g in zip(probs, gold)) / norm


def composite_proper(probs: Sequence[float], gold: Sequence[float]) -> float:
    """log + spherical. Strictly proper; the reward used by the R0 diagnostic."""
    return log_score(probs, gold) + spherical(probs, gold)


# ------------------------------------------------------- selective prediction
@dataclass(frozen=True)
class RiskCoverage:
    coverage: tuple[float, ...]
    risk: tuple[float, ...]
    aurc: float


def _tie_groups(conf: Sequence[float], correct: Sequence[bool]):
    """Yield (cum_n, cum_errors) at the END of each tie group, most confident first."""
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    n = err = 0
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and conf[order[j]] == conf[order[i]]:
            n += 1
            err += 0 if correct[order[j]] else 1
            j += 1
        yield n, err
        i = j


def risk_coverage(conf: Sequence[float], correct: Sequence[bool]) -> RiskCoverage:
    """Risk-coverage curve over complete tie groups, plus its right-step integral."""
    total = len(conf)
    if total == 0:
        return RiskCoverage((), (), float("nan"))
    covs, risks = [], []
    for n, err in _tie_groups(conf, correct):
        covs.append(n / total)
        risks.append(err / n)
    aurc, prev_cov = 0.0, 0.0
    for c, r in zip(covs, risks):
        aurc += r * (c - prev_cov)      # right-step: the risk AT the group boundary
        prev_cov = c
    return RiskCoverage(tuple(covs), tuple(risks), aurc)


def coverage_at_error(conf: Sequence[float], correct: Sequence[bool],
                      budget: float = 0.05) -> float:
    """Largest fraction acceptable, in confidence order, with error <= budget.

    Whole tie groups only. Returns 0.0 if even the first group breaches."""
    rc = risk_coverage(conf, correct)
    ok = [c for c, r in zip(rc.coverage, rc.risk) if r <= budget + 1e-12]
    return max(ok) if ok else 0.0


# ------------------------------------------------------------------ calibration
def ece(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    if not conf:
        return float("nan")
    tot = len(conf)
    out = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf)
               if (c > lo or (b == 0 and c >= lo)) and c <= hi]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        avg = sum(conf[i] for i in idx) / len(idx)
        out += len(idx) / tot * abs(acc - avg)
    return out


# ----------------------------------------------------------- ranking statistics
def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        avg = (i + j - 1) / 2 + 1          # average rank, 1-based
        for k in range(i, j):
            r[order[k]] = avg
        i = j
    return r


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else float("nan")


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Rank AUC with ties counted as half, i.e. the Mann-Whitney statistic."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    r = _ranks(list(scores))
    rpos = sum(r[i] for i, y in enumerate(labels) if y)
    return (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# --------------------------------------------------------------- decision score
def decision_score(mode: str, *, accuracy: float | None = None,
                   base_rate: float | None = None, auc_value: float | None = None,
                   rho: float | None = None) -> float:
    """100 * (metric - guessing) / (perfect - guessing), per mode.

    choice needs the split's OWN base rate: option counts differ between corpora
    (MMLU-Pro has 10, typed-decisions has 4 or 5) and a decision score computed
    against the wrong base rate is not comparable to anything.
    """
    if mode == "choice":
        if accuracy is None or base_rate is None:
            raise ValueError("choice needs accuracy and base_rate")
        return 100.0 * (accuracy - base_rate) / (1.0 - base_rate)
    if mode == "noul":
        if auc_value is None:
            raise ValueError("noul needs auc_value")
        return 100.0 * (auc_value - 0.5) / 0.5
    if mode == "score":
        if rho is None:
            raise ValueError("score needs rho")
        return 100.0 * rho
    raise ValueError(f"unknown mode {mode!r}")
