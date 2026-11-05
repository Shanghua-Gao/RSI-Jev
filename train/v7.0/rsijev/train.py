"""EDITABLE — the training axis: objective, optimiser, schedule, calibration.

Every objective takes option LOGITS and a gold DISTRIBUTION and returns a scalar
loss to minimise, so arms differ in exactly one thing.

The RL objectives are here because reproducing or refuting RLCD is a headline
question, not because they are expected to win. `rlcd_std` -- the published form,
with the advantage standardised per example -- is known to destroy a strictly
proper reward's honesty incentive: on a constant input with truth p=0.30 it
converges to ~0.00008 while the direct gradient reaches 0.2999. `assert_r0_gate`
below runs that diagnostic on whatever objective an arm actually uses, and no RL
arm's result may be read before it passes.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch
import torch.nn.functional as F

Objective = Literal["hard_ce", "soft_ce", "proper_log", "proper_brier",
                    "proper_spherical", "proper_composite", "rlcd_std", "rlcd_loo"]


# ------------------------------------------------------------------ the scores
def _log_score(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    # `where` rather than a product: a masked option has p=0 and g=0, and
    # 0 * log(0) must be 0, not NaN.
    lp = p.clamp_min(1e-12).log()
    return torch.where(g > 0, g * lp, torch.zeros_like(lp)).sum(-1)


def _brier(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return -((p - g) ** 2).sum(-1)


def _spherical(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return (g * p).sum(-1) / p.pow(2).sum(-1).sqrt().clamp_min(1e-12)


REWARD: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "proper_log": _log_score,
    "proper_brier": _brier,
    "proper_spherical": _spherical,
    "proper_composite": lambda p, g: _log_score(p, g) + _spherical(p, g),
}


# -------------------------------------------------------------------- the loss
@dataclass
class RLConfig:
    group: int = 16                # samples per example
    sigma_start: float = 0.6
    sigma_end: float = 0.15
    reward: str = "proper_composite"


def prior_kl(logits: torch.Tensor, base_logits: torch.Tensor,
             direction: str = "base_to_model") -> torch.Tensor:
    """Anchor the head to the frozen base's own option distribution.

    The measured problem is not a cold start -- a zero-initialised residual head
    still pulled choice BELOW the zero-shot control out of distribution. Training
    on ten unrelated corpora overwrites a prior it never sees in its loss. This
    puts the prior in the loss, so departing from it costs something and the
    departure has to be paid for by the task term.

    Direction matters and the obvious one is wrong. KL(model || base) has a
    gradient of roughly p*(log p - log q + 1), which VANISHES once the head is
    confident: from logits [5, -5] against a base of [2, 1], sixty full-batch
    steps move the gap 10.000 -> 9.975. A head that has already overwritten the
    prior feels almost no pull back towards it, which is exactly the case this
    term exists for. KL(base || model) has gradient (p - q) with respect to the
    logits, which is largest precisely when the head has drifted furthest. That
    is the default here.
    """
    lp = F.log_softmax(logits, dim=-1)
    lq = F.log_softmax(base_logits.to(logits.dtype), dim=-1)
    # Masked options are -inf in both, and (-inf) - (-inf) is NaN, which `where`
    # still backpropagates from the branch it did not take. Neutralise the
    # operands BEFORE subtracting, not the result afterwards.
    finite = torch.isfinite(lp) & torch.isfinite(lq)
    zero = torch.zeros_like(lp)
    lp_s = torch.where(finite, lp, zero)
    lq_s = torch.where(finite, lq, zero)
    if direction == "model_to_base":
        w = torch.where(finite, lp_s.exp(), zero)
        return (w * (lp_s - lq_s)).sum(-1).mean()
    q = torch.where(finite, lq_s.exp(), zero)
    return (q * (lq_s - lp_s)).sum(-1).mean()


def objective_loss(name: Objective, logits: torch.Tensor, gold: torch.Tensor, *,
                   progress: float = 0.0, rl: RLConfig | None = None,
                   generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (B, K); gold (B, K) a distribution. Returns a scalar to MINIMISE."""
    if name == "hard_ce":
        return F.cross_entropy(logits, gold.argmax(-1))
    if name == "soft_ce":
        # Masked options carry logit -inf, so log_softmax is -inf there and gold
        # is 0: the product is 0 * -inf = NaN unless it is masked out first.
        lp = F.log_softmax(logits, dim=-1)
        return -torch.where(gold > 0, gold * lp, torch.zeros_like(lp)).sum(-1).mean()
    if name.startswith("proper_"):
        return -REWARD[name](F.softmax(logits, dim=-1), gold).mean()
    if name in ("rlcd_std", "rlcd_loo"):
        cfg = rl or RLConfig()
        reward = REWARD[cfg.reward]
        sigma = cfg.sigma_start + (cfg.sigma_end - cfg.sigma_start) * progress
        b, k = logits.shape
        # Masked options carry logit -inf, and (-inf + eps) - (-inf) is NaN in the
        # log-density below. Noise and density live on the VALID options only. The
        # R0 gate has no masked options, so it passed while every real batch --
        # which pads to max_options -- produced a NaN loss at step 0.
        valid = torch.isfinite(logits).unsqueeze(1)                  # (B, 1, K)
        eps = torch.randn((b, cfg.group, k), device=logits.device,
                          dtype=logits.dtype, generator=generator) * sigma
        eps = torch.where(valid, eps, torch.zeros_like(eps))
        sampled = logits.unsqueeze(1) + eps
        r = reward(F.softmax(sampled.detach(), dim=-1),
                   gold.unsqueeze(1).expand(-1, cfg.group, -1))      # (B, G)
        if name == "rlcd_std":
            # The published form. Dividing by the within-example std removes the
            # reward's derivative MAGNITUDE and leaves its sign, so the majority
            # outcome outvotes the minority even at the truthful forecast.
            adv = (r - r.mean(1, keepdim=True)) / r.std(1, keepdim=True).clamp_min(1e-8)
        else:
            loo = (r.sum(1, keepdim=True) - r) / (cfg.group - 1)
            adv = r - loo                                            # magnitude kept
        diff = torch.where(valid, sampled.detach() - logits.unsqueeze(1),
                           torch.zeros_like(sampled))
        logq = -(diff ** 2).sum(-1) / (2 * sigma ** 2)
        return -(adv.detach() * logq).mean()
    raise ValueError(f"unknown objective {name!r}")


# --------------------------------------------------------------- the R0 gate
def r0_probe(name: Objective, *, n_false: int = 70, n_true: int = 30, steps: int = 128,
             lr: float = 0.03, seed: int = 17, rl: RLConfig | None = None) -> float:
    """Constant input, 70 false / 30 true. Returns the learned p(true).

    A truthful learner reaches 0.30. Anything that does not is not optimising the
    reward it claims to optimise, whatever it does downstream.
    """
    torch.manual_seed(seed)
    z = torch.zeros(1, 2, requires_grad=True)
    opt = torch.optim.AdamW([z], lr=lr, weight_decay=0.0)
    gold = torch.zeros(n_false + n_true, 2)
    gold[:n_false, 0] = 1.0
    gold[n_false:, 1] = 1.0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = objective_loss(name, z.expand(len(gold), 2), gold,
                              progress=step / (steps - 1), rl=rl)
        loss.backward()
        opt.step()
    return F.softmax(z.detach(), -1)[0, 1].item()


def assert_r0_gate(name: Objective, *, tol: float = 0.02, seeds: Sequence[int] = (17, 29, 43),
                   rl: RLConfig | None = None) -> None:
    """No RL arm's downstream result may be read before this passes."""
    truth = 0.30
    got = [r0_probe(name, seed=s, rl=rl) for s in seeds]
    bad = [p for p in got if abs(p - truth) > tol]
    if bad:
        raise AssertionError(
            f"objective {name!r} failed the R0 diagnostic: p(true)={[round(p,6) for p in got]}, "
            f"truth {truth}. The reward is proper; the estimator is not.")


# ----------------------------------------------------- common random numbers
def seed_everything(seed: int) -> torch.Generator:
    """CRN: arms in one comparison share seed, init and data order, and differ in
    exactly one thing. The paired difference then has far lower variance than
    either arm, which is what makes a small seed budget usable."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def average_checkpoints(states: Sequence[dict]) -> dict:
    """Free variance reduction: averaging the last k checkpoints removes
    checkpoint-selection noise at no training cost."""
    if not states:
        raise ValueError("no checkpoints")
    out = {k: v.clone().float() for k, v in states[0].items()}
    for st in states[1:]:
        for k in out:
            out[k] += st[k].float()
    for k in out:
        out[k] /= len(states)
    return out


def fit_temperature(logits: torch.Tensor, gold: torch.Tensor, *, steps: int = 200,
                    lr: float = 0.05) -> float:
    """Fit ONE temperature on a held-out calibration partition, from raw logits.

    Reported as its own row, never folded into a raw number: a temperature moves
    ECE a great deal and selective coverage not at all, because it is monotone
    within a question.
    """
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = -(gold * F.log_softmax(logits / log_t.exp(), dim=-1)).sum(-1).mean()
        loss.backward()
        opt.step()
    return float(log_t.detach().exp())
