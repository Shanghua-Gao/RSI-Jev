"""RL stage (v3.0): continued-training arms whose
reward is NOT the per-item CE on labels.

fit.fit() hands over to fit_rl2() when FitConfig.rl2 is a non-empty dict. Every
step = one REPLAY batch (soft CE on the parent's corpus, identical stream in
every arm) + one RL-SIDE batch drawn from the sources that start with
rl2["source"]. What is done with the RL-side batch is the arm:

  listwise      Plackett-Luce ranking policy over one query's candidates (scores =
                the noul logit gap); reward NDCG@k of the gold; RLOO over sampled
                rankings. Non-decomposable: no per-candidate loss expresses it.
  pointwise     CONTROL for listwise: soft CE on each candidate's own label.
  consistency   exact expected agreement of the choice / noul / reworded forms of
                one decision (P(all forms pick the same side)). Uses NO gold.
  rlcd          RL from Contrastive Distillation (Yang et al. 2023): a teacher LM
                prompted with a POSITIVE vs a NEGATIVE system prompt gives the
                preferred / dispreferred option on UNLABELED decisions; DPO on the
                scorer's option log-probs against the frozen parent.
  rlcd_sft      CONTROL for rlcd: CE on the positive-prompt answer (context
                distillation) on the same items.
  replay_only   CONTROL where the RL side uses no labels: the replay batch alone.
                With cal_method="oof_head_scorefloor" it is control (b): the same
                steps, then the cal-4b head refitted on the rl_cal pool.

RLCD = Reinforcement Learning for Calibrated Decisions (Jev's name; undisclosed;
our hypotheses; docs/rl.md). RL-side pool rl_cal = the PARENT's own
1-in-10 in-distribution holdout (never trained on by the parent):
  select        H1 selective decisions: exact expected utility of a stochastic
                act/defer policy at the product threshold. Threshold utility, not a
                proper score: it only moves confidence across tau.
  setcal        H2 set-level calibration: squared calibration error of (cal group x
                mode x confidence band) cells, cell means tracked over steps; the
                gradient moves a whole cell's confidence toward its accuracy.
  agreecal      H3 calibration without gold: each form's confidence is pulled to the
                rate at which the OTHER forms of the same decision agree with it.

Every mode except replay_only carries the same KL(parent || policy) anchor on its
RL-side rows (frozen bf16 copy of the parent taken at step 0), and every mode
logs the A7 numbers: max |logit|, batch-mean KL to the parent, gated fraction.
A step whose KL exceeds rl2["kl_gate"] drops its RL/SFT-side term (KL still
applies), so an arm cannot run away from the parent.

RL-side source files mark their structure in plain Case fields (load_cases drops
anything else): group = case_id.split("#")[0]; a consistency question key is
"<key>@@<flag option>". consistency / rlcd never read a row's gold; the only
reader is a logged step-0 diagnostic (teacher accuracy where gold happens to exist).
"""
from __future__ import annotations

import copy
import math
import random
from collections import defaultdict

import torch
import torch.nn.functional as F

from .encode import collate, encode_question, gold_tensor, iter_questions, unpermute_logits

MATRIX = ("ce_full", "rl_binary", "rl_proper", "rlcr", "bandit", "bandit_sup")
MODES_RL2 = (*MATRIX, "packet", "packet_ce", "select", "setcal", "agreecal", "confrank",   # RLCD = RL for Calibrated Decisions
             "listwise", "pointwise", "consistency", "rlcd", "rlcd_sft", "replay_only")
# RLCD modes: which RL-side source they read by default
MODE_SOURCE = {**{m: "rl_slice" for m in MATRIX}, "packet": "rl_pkt", "packet_ce": "rl_pkt", "confrank": "rl_slice",
               "select": "rl_cal", "setcal": "rl_cal", "agreecal": "rl_cnc",
               "listwise": "rl_hrr", "pointwise": "rl_hrr", "consistency": "rl_cnc",
               "rlcd": "rl_cnc", "rlcd_sft": "rl_cnc", "replay_only": "rl_"}

DEFAULTS = dict(mode="replay_only", source=None, group_size=16, samples=16, ndcg_k=5,
                pl_tau=1.0, cases_per_step=4, rows_per_step=16, rl_coef=1.0, kl_coef=0.1,
                kl_gate=0.05, dpo_beta=0.5, teacher="Qwen/Qwen3.5-4B", teacher_batch=8,
                teacher_max_tokens=1536, pool_max=6000, log_every=50,
                # select (H1): act iff confidence >= tau; utility +1 right act, -c wrong
                # act, -d defer; the policy acts with P = sigmoid((logit m - logit tau)/T)
                tau=0.9, cost_wrong=9.0, cost_defer=0.0, act_temp=0.5,
                # setcal (H2): strata = cal group x mode, bands of top-1 confidence;
                # running (EMA) per-cell mean confidence and accuracy
                bands=(0.5, 0.7, 0.9), ema=0.98,
                # confrank: soft-AUC on the confidence ORDERING + a CE anchor.
                # rank_coef=0 reduces the arm exactly to ce_full (a built-in control).
                rank_coef=1.0, ce_anchor=1.0, margin_temp=1.0,
                # ctl-b: cal-4b refit on the rl_cal pool after training (calibrate.py)
                cal_method="", cal_save_dir="",
                # ---- RLCD reconstruction matrix (jev_rlcd_reproduction_notes s14/s19/s20) ----
                replay=True,          # False: the RL-side slice is the only training data
                G=8, lam=0.3,         # samples per state; calibration weight (rlcr / bandit)
                cal_grad=False,       # bandit: also differentiate the reward's p(a) (E-grad)
                explore="cat",        # cat | temp | gauss | gumbel
                explore_temp=1.5, sigma=0.1,
                cal_pool="rl_cal",    # cal-4b refit pool (source name)
                cal_activate=True,    # False: fit + save cal-4b, keep the arm's records NATIVE
                dev_pool="")          # source scored natively (and calibrated) at the end

POS_SYSTEM = ("You are a careful, expert reviewer. Read the whole text, check it against each "
              "option's description, and give the most accurate answer.")
NEG_SYSTEM = ("You are a careless reviewer who skims. Answer from first impressions and surface "
              "keywords, without checking the text against the options' descriptions.")


# ------------------------------------------------------------------ pure losses
def pl_listwise_loss(s: torch.Tensor, gold: int, *, samples: int, k: int, tau: float = 1.0,
                     generator: torch.Generator | None = None):
    """Plackett-Luce REINFORCE (RLOO) with reward NDCG@k of a single relevant item.

    s: (G,) candidate scores (with grad). Rankings are sampled with Gumbel-top-G
    on s/tau; the log-probability of the top-k PREFIX is exact under PL and is
    all the reward depends on. Returns (loss, mean reward, greedy hit@1)."""
    G = s.shape[0]
    z = s / tau
    u = torch.rand((samples, G), generator=generator, dtype=torch.float32).to(s.device)
    gumbel = -torch.log(-torch.log(u.clamp(1e-10, 1 - 1e-10)))
    perm = torch.argsort(z.detach().float().unsqueeze(0) + gumbel, dim=-1, descending=True)
    zp = z.float().unsqueeze(0).expand(samples, -1).gather(1, perm)            # (M, G)
    suffix = torch.logcumsumexp(zp.flip(-1), dim=-1).flip(-1)                  # lse of zp[j:]
    kk = min(k, G)
    logp = (zp[:, :kk] - suffix[:, :kk]).sum(-1)                               # (M,)
    rank = (perm == gold).float().argmax(-1)                                   # 0-based
    reward = torch.where(rank < k, 1.0 / torch.log2(rank.float() + 2.0),
                         torch.zeros_like(rank, dtype=torch.float32))
    if samples > 1:
        loo = (reward.sum() - reward) / (samples - 1)
        adv = reward - loo
    else:
        adv = reward
    loss = -(adv.detach() * logp).mean()
    hit1 = float(int(z.detach().argmax()) == gold)
    return loss, float(reward.mean()), hit1


def agreement_loss(q: torch.Tensor, groups: list[list[int]]):
    """Exact expected agreement. q: (R,) P(flag side) per row; groups: row index
    lists, one per decision. Reward of a decision = P(every form lands on the same
    side) = prod q + prod (1-q), computed exactly (RL_FRAMING s1: no sampling when
    the expectation is closed-form). Returns (loss, mean agreement)."""
    agr = []
    for g in groups:
        qg = q[g].clamp(1e-6, 1 - 1e-6)
        agr.append(qg.prod() + (1 - qg).prod())
    a = torch.stack(agr)
    return -a.mean(), float(a.detach().mean())


def dpo_option_loss(logits: torch.Tensor, ref_logits: torch.Tensor, pos: torch.Tensor,
                    neg: torch.Tensor, beta: float):
    """DPO on option log-probs: -log sigma(beta [(lp+ - lr+) - (lp- - lr-)])."""
    lp = F.log_softmax(logits.float(), -1)
    lr = F.log_softmax(ref_logits.float(), -1)
    g = lambda t, i: t.gather(1, i.view(-1, 1)).squeeze(1)
    d = beta * ((g(lp, pos) - g(lr, pos)) - (g(lp, neg) - g(lr, neg)))
    return -F.logsigmoid(d).mean(), float((d > 0).float().mean())


def kl_ref_policy(logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """Per-row KL(ref || policy) over the valid options. (R,)"""
    lp = F.log_softmax(logits.float(), -1)
    lq = F.log_softmax(ref_logits.float(), -1)
    fin = torch.isfinite(lp) & torch.isfinite(lq)
    z = torch.zeros_like(lp)
    lp, lq = torch.where(fin, lp, z), torch.where(fin, lq, z)
    return (torch.where(fin, lq.exp(), z) * (lq - lp)).sum(-1)


def top_conf_correct(logits: torch.Tensor, gold_idx: torch.Tensor):
    """(log m, log(1-m), correct) with m the top-1 probability (evaluator's first-max
    rule) and correct = top-1 == gold argmax."""
    lp = F.log_softmax(logits.float(), -1)
    z = logits.float().masked_fill(~torch.isfinite(logits), -1e30)
    top = z.argmax(-1)
    logm = lp.gather(1, top[:, None]).squeeze(1)
    log1m = torch.log(-torch.expm1(logm.clamp(max=-1e-7)))
    return logm, log1m, (top == gold_idx).float()


def selective_utility_loss(logits, gold_idx, *, tau: float, c: float, d: float, T: float):
    """-E[utility]/(1+c) of the act/defer policy; exact (closed form, RL_FRAMING s1).
    Returns (loss, mean utility, act rate)."""
    logm, log1m, correct = top_conf_correct(logits, gold_idx)
    zt = (logm - log1m) - math.log(tau / (1 - tau))
    pact = torch.sigmoid(zt / T)
    u = pact * (correct * (1 + c) - c) + (1 - pact) * (-d)
    return -(u / (1 + c)).mean(), float(u.detach().mean()), float(pact.detach().mean())


class CellStats:
    """Running per-cell (stratum x confidence band) sums of confidence and accuracy."""
    def __init__(self, bands, ema):
        self.bands, self.ema, self.s = tuple(bands), ema, {}

    def band(self, m: float) -> int:
        return sum(m >= b for b in self.bands)

    def update(self, keys, conf, acc):
        for cell in self.s.values():
            for i in range(3):
                cell[i] *= self.ema
        for k, c_, a in zip(keys, conf, acc):
            cell = self.s.setdefault(k, [0.0, 0.0, 0.0])
            cell[0] += 1.0; cell[1] += c_; cell[2] += a

    def gap(self, k) -> float:
        n, c_, a = self.s[k]
        return (c_ - a) / max(n, 1e-8)

    def ece(self) -> float:
        tot = sum(v[0] for v in self.s.values()) or 1.0
        return sum(v[0] / tot * abs(self.gap(k)) for k, v in self.s.items())


def setcal_loss(logits, gold_idx, strata, stats: CellStats):
    """Surrogate of the stratified squared calibration error sum_cell n (conf-acc)^2 / 2N:
    each row's confidence gets the (detached) gap of ITS cell, estimated from this
    batch plus the running cell sums. Returns (loss, running stratified ECE)."""
    logm, _, correct = top_conf_correct(logits, gold_idx)
    m = logm.exp()
    keys = [(s_, stats.band(float(v))) for s_, v in zip(strata, m.detach().tolist())]
    stats.update(keys, m.detach().tolist(), correct.tolist())
    gaps = torch.tensor([stats.gap(k) for k in keys], device=logits.device)
    return (gaps * m).mean(), stats.ece()


def conf_rank_loss(logits, gold_idx, *, margin_temp: float):
    """Soft-AUC surrogate for the ORDERING of top-1 confidence against correctness.

    AURC, coverage at a precision target and resolution depend only on that ordering,
    and are therefore exactly invariant under any strictly monotone map of the confidence
    SCALAR (temperature/Platt/isotonic on p; checked to 1e-9). They are NOT invariant under
    cal-4b: oof_head is a learned head over hidden-state features with per-group weights,
    so it can reorder rows and can move AURC. Nothing here is free -- confrank must be
    compared against the CE control WITH cal-4b applied post-hoc, not against native CE.
    What the objective adds is a term CE does not have: CE matches the full distribution,
    and only shapes the ordering indirectly.

    This is the pairwise logistic (RankNet) loss over (correct, wrong) pairs of
    s = logit-odds of the top-1 probability, which is monotone in confidence and unbounded.

    Returns (loss, soft AUC, n_pairs). A batch with no correct or no wrong row has no
    pair and contributes nothing.
    """
    logm, log1m, correct = top_conf_correct(logits, gold_idx)
    s = (logm - log1m) / margin_temp
    pos, neg = correct > 0.5, correct <= 0.5
    if not (bool(pos.any()) and bool(neg.any())):
        # No pair in this batch. Keep a (zero) gradient path so an anchor-free arm can
        # still call .backward(), but NEVER through the -inf option mask: logits.sum()
        # would be -inf there and -inf * 0 is NaN.
        safe = logits.float().masked_fill(~torch.isfinite(logits), 0.0)
        return safe.sum() * 0.0, float("nan"), 0
    d = s[pos][:, None] - s[neg][None, :]
    return F.softplus(-d).mean(), float((d.detach() > 0).float().mean()), int(d.numel())


def agreecal_loss(logits, rows_q, groups):
    """H3: form f's top-1 confidence -> the probability that the OTHER forms of the
    same decision land on f's side (detached). No gold. rows_q: the Question of each
    row (key carries the flag option). Returns (loss, mean target, mean agreement of
    argmax sides)."""
    p = F.softmax(logits.float(), -1)
    qf = torch.stack([p[j, list(q.options).index(flag_option(q.key))] for j, q in enumerate(rows_q)])
    side = (qf.detach() >= 0.5).float()                    # 1 = flagged side (argmax of 2)
    conf = torch.where(side > 0, qf, 1 - qf)
    losses, tg, agree = [], [], []
    for g in groups:
        for f in g:
            others = [o for o in g if o != f]
            po = qf.detach()[others]
            t = torch.where(side[f] > 0, po, 1 - po).mean()
            losses.append((conf[f] - t) ** 2)
            tg.append(float(t))
        sides = side[g]
        agree.append(float((sides == sides[0]).all()))
    return torch.stack(losses).mean(), sum(tg) / len(tg), sum(agree) / len(agree)


# ------------------------------------------------------------------ packet segmentation
def packet_loss(mode, logits, gold, keys, groups, *, G: int, generator):
    """Sequence-level RL for document splitting. Rows = the questions of a few packets;
    groups = per-packet row lists. boundary_* rows are the segmentation policy (independent
    Bernoulli per page from the noul P(true)); the reward of a sampled segmentation is 1 iff
    EVERY boundary of the packet is right (exact whole-packet split), RLOO over G per packet.
    category_* rows (and every row in packet_ce, the control) get soft CE."""
    z = logits.float()
    lp = F.log_softmax(z, -1)
    ce_rows = [j for j, k in enumerate(keys) if mode == "packet_ce" or not k.startswith("boundary_")]
    loss = torch.zeros((), device=z.device)
    if ce_rows:
        g = gold[ce_rows]
        l_ = lp[ce_rows]
        loss = loss - torch.where(g > 0, g * l_, torch.zeros_like(l_)).sum(-1).mean()
    rec = {}
    with torch.no_grad():
        ex = []
        for rows in groups:
            b = [j for j in rows if keys[j].startswith("boundary_")]
            if b:
                ex.append(float(all(int(z[j, 1] > z[j, 0]) == int(gold[j, 1] > 0.5) for j in b)))
        rec["rl_exact_greedy"] = sum(ex) / max(1, len(ex))
    if mode == "packet":
        terms, rews = [], []
        for rows in groups:
            b = [j for j in rows if keys[j].startswith("boundary_")]
            if not b:
                continue
            pt = lp[b, 1]; pf = lp[b, 0]                          # log p(true), log p(false)
            u = torch.rand((G, len(b)), generator=generator).to(z.device)
            smp = (u < pt.detach().exp().unsqueeze(0)).float()     # (G, n) sampled "starts a doc"
            gt = (gold[b, 1] > 0.5).float().unsqueeze(0)
            R = (smp == gt).all(-1).float()                        # exact packet split
            logpi = (smp * pt.unsqueeze(0) + (1 - smp) * pf.unsqueeze(0)).sum(-1)
            adv = R - (R.sum() - R) / max(1, G - 1)
            terms.append(-(adv.detach() * logpi).mean())
            rews.append(float(R.mean()))
        if terms:
            loss = loss + torch.stack(terms).mean()
            rec["rl_reward"] = sum(rews) / len(rews)
    return loss, rec


# ------------------------------------------------------------------ RLCD matrix
def _valid(z):
    return torch.isfinite(z)


def sample_actions(z: torch.Tensor, G: int, *, explore: str, T: float, sigma: float,
                   generator: torch.Generator):
    """G actions per row. Returns (a (B,G), logpi (B,G) with grad, p_act (B,G) detached =
    the reported probability of the chosen action). cat: a ~ softmax(z). temp: a ~
    softmax(z/T), the tempered policy is the one differentiated. gumbel: argmax(z + Gumbel),
    the same law as cat (kept as a check). gauss (Laya): z~ = z + sigma*eps on valid
    options, a = argmax z~, log-density of the Gaussian, p_act from softmax(z~)."""
    B, K = z.shape
    v = _valid(z)
    zz = z.float()
    if explore in ("cat", "temp", "gumbel"):
        pol = zz / (T if explore == "temp" else 1.0)
        lp = F.log_softmax(pol, -1)
        u = torch.rand((B, G, K), generator=generator).to(z.device).clamp(1e-10, 1 - 1e-10)
        gum = -torch.log(-torch.log(u))
        a = (lp.detach().unsqueeze(1) + gum).masked_fill(~v.unsqueeze(1), -1e30).argmax(-1)
        logpi = lp.gather(1, a)
        p_act = F.softmax(zz, -1).detach().gather(1, a)
        return a, logpi, p_act
    if explore == "gauss":
        eps = torch.randn((B, G, K), generator=generator).to(z.device) * sigma
        zs = zz.detach().unsqueeze(1) + eps
        zs = zs.masked_fill(~v.unsqueeze(1), -1e30)
        a = zs.argmax(-1)
        diff = torch.where(v.unsqueeze(1), zs - zz.unsqueeze(1), torch.zeros_like(zs))
        logpi = -(diff ** 2).sum(-1) / (2 * sigma ** 2)
        p_act = F.softmax(zs, -1).gather(2, a.unsqueeze(-1)).squeeze(-1).detach()
        return a, logpi, p_act
    raise ValueError(f"explore {explore!r}")


def _loo(R):
    G = R.shape[1]
    return R - (R.sum(1, keepdim=True) - R) / max(1, G - 1)


def _proper_reward(p, g, is_score):
    """log + spherical (+ RPS on score rows), per sampled distribution. p (B,G,K), g (B,K)."""
    gg = g.unsqueeze(1).expand_as(p)
    lp = p.clamp_min(1e-12).log()
    logs = torch.where(gg > 0, gg * lp, torch.zeros_like(lp)).sum(-1)
    sph = (gg * p).sum(-1) / p.pow(2).sum(-1).sqrt().clamp_min(1e-12)
    rps = ((p.cumsum(-1) - gg.cumsum(-1)) ** 2).sum(-1)
    return logs + sph - is_score.float().unsqueeze(1) * rps


def matrix_loss(mode, logits, gold, modes, r, generator):
    """One RLCD-matrix objective on a batch of states. logits (B,K) masked -inf; gold (B,K)
    the gold distribution. Outcome feedback y ~ Bernoulli(gold[a]) (a hard label gives the
    indicator): the environment reveals ONLY the chosen action's success. Returns (loss, rec)."""
    z = logits.float()
    p = F.softmax(z, -1)
    v = _valid(z)
    rec = {}
    with torch.no_grad():
        gidx = gold.argmax(-1)
        top = z.masked_fill(~v, -1e30).argmax(-1)
        ent = -(p * p.clamp_min(1e-12).log()).masked_fill(~v, 0).sum(-1)
        rec.update(rl_acc=float((top == gidx).float().mean()), rl_conf=float(p.max(-1).values.mean()),
                   rl_ent=float(ent.mean()))
    if mode == "ce_full":
        lp = F.log_softmax(z, -1)
        return -torch.where(gold > 0, gold * lp, torch.zeros_like(lp)).sum(-1).mean(), rec
    if mode == "rl_proper":          # Laya-style: Gaussian logit noise, proper score on the sample
        G, sig = r["G"], r["sigma"]
        eps = torch.randn((z.shape[0], G, z.shape[1]), generator=generator).to(z.device) * sig
        eps = torch.where(v.unsqueeze(1), eps, torch.zeros_like(eps))
        zs = z.detach().unsqueeze(1) + eps
        R = _proper_reward(F.softmax(zs, -1), gold, torch.tensor([m == "score" for m in modes],
                                                                 device=z.device))
        diff = torch.where(v.unsqueeze(1), zs - z.unsqueeze(1), torch.zeros_like(zs))
        logq = -(diff ** 2).sum(-1) / (2 * sig ** 2)
        rec["rl_reward"] = float(R.mean())
        return -(_loo(R).detach() * logq).mean(), rec
    a, logpi, p_act = sample_actions(z, r["G"], explore=r["explore"], T=r["explore_temp"],
                                     sigma=r["sigma"], generator=generator)
    with torch.no_grad():
        succ = gold.gather(1, a)
        y = (torch.rand(succ.shape, generator=generator).to(z.device) < succ).float()
    rec["rl_success"] = float(y.mean())
    if mode == "bandit_sup":         # E': the same chosen-action feedback, supervised
        pa = p.gather(1, a).clamp(1e-6, 1 - 1e-6)
        loss = -(y * pa.log() + (1 - y) * (1 - pa).log()).mean()
        return loss, rec
    if mode == "rl_binary":
        R = y
    elif mode == "bandit":           # E: gold hidden; R = y - lam (p(a) - y)^2
        R = y - r["lam"] * (p_act - y) ** 2
    elif mode == "rlcr":             # D: correctness by PG + lam * full-label Brier (exact)
        R = y
    else:
        raise ValueError(mode)
    rec["rl_reward"] = float(R.mean())
    loss = -(_loo(R).detach() * logpi).mean()
    if mode == "bandit" and r.get("cal_grad"):
        # E-grad: the reward's dependence on the REPORTED p(a) is differentiated too
        # (pathwise), not only through the sampling law. With the notes' detached
        # form a single-head policy collapses like binary RL (toy check, design note).
        pa = p.gather(1, a)
        loss = loss + r["lam"] * ((pa - y) ** 2).mean()
    if mode == "rlcr":
        brier = ((p - gold) ** 2).masked_fill(~v, 0).sum(-1).mean()
        loss = loss + r["lam"] * brier
        rec["rl_brier"] = float(brier.detach())
    return loss, rec


@torch.no_grad()
def dev_metrics(model, tokenizer, pairs, enc, *, max_options, device, bs=32) -> dict:
    """Native dev metrics (canonical order, eval mode): acc, NLL, Brier, ECE (10-bin and
    adaptive 15-quantile), conf-acc gap, AURC, coverage at 98% precision, resolution
    (mean conf correct - incorrect), mean entropy, conf>=.9 coverage / accuracy."""
    from .encode import EncodeConfig
    e = EncodeConfig(layout=enc.layout, option_pool=enc.option_pool, option_order="canonical")
    was = model.training
    model.eval()
    conf, corr, nll, bri, ent = [], [], [], [], []
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    for i in range(0, len(pairs), bs):
        ch = pairs[i:i + bs]
        b = collate(tokenizer, [encode_question(tokenizer, c.state, q, e) for c, q in ch],
                    max_options=max_options, device=device)
        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=dev_type == "cuda"):
            lg = model(**b)
        lg = unpermute_logits(lg.float(), b["option_perm"], b["option_mask"]).masked_fill(
            ~b["option_mask"], float("-inf"))
        p = F.softmax(lg, -1)
        g = gold_tensor([c for c, _ in ch], [q.key for _, q in ch], max_options, device=device)
        top = lg.masked_fill(~b["option_mask"], -1e30).argmax(-1)
        conf += p.max(-1).values.tolist()
        corr += (top == g.argmax(-1)).float().tolist()
        nll += (-torch.where(g > 0, g * p.clamp_min(1e-12).log(), torch.zeros_like(p)).sum(-1)).tolist()
        bri += ((p - g) ** 2).sum(-1).tolist()
        ent += (-(p * p.clamp_min(1e-12).log()).masked_fill(~b["option_mask"], 0).sum(-1)).tolist()
    model.train(was)
    n = len(conf)
    order = sorted(range(n), key=lambda i: -conf[i])
    def ece_bins(idx_groups):
        return sum(len(ix) / n * abs(sum(corr[i] for i in ix) / len(ix) - sum(conf[i] for i in ix) / len(ix))
                   for ix in idx_groups if ix)
    fixed = [[i for i in range(n) if (conf[i] > b / 10 or (b == 0 and conf[i] >= 0)) and conf[i] <= (b + 1) / 10]
             for b in range(10)]
    adapt = [order[k * n // 15:(k + 1) * n // 15] for k in range(15)]
    aurc, cov98, err = 0.0, 0.0, 0
    for k, i in enumerate(order, 1):
        err += 1 - corr[i]
        aurc += err / k / n
        if err / k <= 0.02:
            cov98 = k / n
    c1 = [conf[i] for i in range(n) if corr[i]]; c0 = [conf[i] for i in range(n) if not corr[i]]
    hi = [i for i in range(n) if conf[i] >= 0.9]
    return {"n": n, "acc": round(sum(corr) / n, 4), "nll": round(sum(nll) / n, 4),
            "brier": round(sum(bri) / n, 4), "ece": round(ece_bins(fixed), 4),
            "ece_adapt": round(ece_bins(adapt), 4),
            "conf_minus_acc": round(sum(conf) / n - sum(corr) / n, 4),
            "aurc": round(aurc, 4), "cov98": round(cov98, 4),
            "resolution": round((sum(c1) / max(1, len(c1))) - (sum(c0) / max(1, len(c0))), 4),
            "entropy": round(sum(ent) / n, 4),
            "conf90_cov": round(len(hi) / n, 4),
            "conf90_acc": round(sum(corr[i] for i in hi) / len(hi), 4) if hi else None}


# ------------------------------------------------------------------ data views
def cal_source(case_id: str) -> str:
    """rl_cal case ids are 'rl_cal|<parent corpus file stem>|<orig id>~<salt>'."""
    return case_id.split("|")[1] if case_id.startswith("rl_") and "|" in case_id else "?"


def cal_group(src: str) -> str:
    """cal-4b dev group (calibrate.GROUP_WEIGHTS) of a parent-corpus source."""
    if src == "td_train":
        return "td"
    if src in ("synth", "st_procedural_train"):
        return "synth"
    if src.startswith("mc_replay"):
        return "mc"
    return {"st_kev_hard_v1": "kev_hard", "st_kev_documents_v1": "kev_docs",
            "st_kev_devtools_v1": "kev_devtools", "st_nimble_train": "nimble_train"}.get(
        src, "ts" if src.startswith(("cov_", "ds2_")) else "nimble_up")


def group_of(case_id: str) -> str:
    return case_id.split("#")[0]


def flag_option(key: str) -> str | None:
    return key.split("@@", 1)[1] if "@@" in key else None


def build_listwise_groups(pairs):
    """pairs: [(case, q)] of noul rows. -> [(group, [row idx], gold row position)]"""
    by = defaultdict(list)
    for i, (c, q) in enumerate(pairs):
        by[group_of(c.case_id)].append(i)
    out = []
    for gname in sorted(by):
        rows = by[gname]
        golds = [j for j, i in enumerate(rows) if pairs[i][0].gold[pairs[i][1].key][1] > 0.5]
        if len(golds) == 1 and len(rows) >= 2:
            out.append((gname, rows, golds[0]))
    return out


def build_consistency_groups(pairs):
    by = defaultdict(list)
    for i, (c, q) in enumerate(pairs):
        if flag_option(q.key) is not None:
            by[c.case_id].append(i)
    return [(g, rows) for g, rows in sorted(by.items()) if len(rows) >= 2]


# ------------------------------------------------------------------ teacher (RLCD)
def teacher_user_text(state: str, q) -> str:
    opts = "\n".join(f"- {o}: {q.criteria.get(o, '')}".rstrip(": ") for o in q.options)
    head = f"{state}\n\n" if state.strip() else ""
    return f"{head}{q.instructions}\nOptions:\n{opts}\n\nReply with exactly one option key."


@torch.no_grad()
def teacher_scores(pairs, *, model_name: str, device, batch: int = 8, max_tokens: int = 1536,
                   system: str, lm=None, tok=None):
    """One next-token distribution over each question's option keys (first token
    of each key; a question whose keys collide is returned as None), from a chat
    teacher with `system` as its system prompt, thinking disabled."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .encode import option_first_token_ids
    own = lm is None
    if own:
        tok = AutoTokenizer.from_pretrained(model_name)
        lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out: list = [None] * len(pairs)
    todo = []
    for i, (c, q) in enumerate(pairs):
        ids = option_first_token_ids(tok, q.options, prefix="")
        if len(set(ids)) != len(ids):
            continue
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": teacher_user_text(c.state, q)}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t = tok(text, add_special_tokens=False)["input_ids"]
        if len(t) > max_tokens:            # cut the middle of the state, keep both ends
            t = t[: max_tokens // 4] + t[-(max_tokens - max_tokens // 4):]
        todo.append((i, t, ids))
    todo.sort(key=lambda r: len(r[1]))
    # RIGHT padding and a read at each row's own last token: nothing after a
    # position can reach it in a causal model, so padding cannot move the answer
    # (left padding is not safe for the linear-attention layers).
    for b in range(0, len(todo), batch):
        chunk = todo[b:b + batch]
        w = max(len(t) for _, t, _ in chunk)
        ids_t = torch.full((len(chunk), w), pad, dtype=torch.long)
        am = torch.zeros((len(chunk), w), dtype=torch.long)
        for r, (_, t, _) in enumerate(chunk):
            ids_t[r, :len(t)] = torch.tensor(t)
            am[r, :len(t)] = 1
        logits = lm(input_ids=ids_t.to(device), attention_mask=am.to(device)).logits
        for r, (i, t, oids) in enumerate(chunk):
            out[i] = torch.softmax(logits[r, len(t) - 1].float()[oids], -1).cpu()
    if own:
        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def rlcd_pairs(pairs, t_pos, t_neg):
    """Preferred = argmax under the positive prompt, dispreferred = argmax under the
    negative prompt; items where the two prompts agree carry no contrast and are
    dropped (their DPO term would be zero). -> [(row, y+, y-)]"""
    out = []
    for i in range(len(pairs)):
        if t_pos[i] is None or t_neg[i] is None:
            continue
        yp, yn = int(t_pos[i].argmax()), int(t_neg[i].argmax())
        if yp != yn:
            out.append((i, yp, yn))
    return out


# ------------------------------------------------------------------ the loop
def fit_rl2(model, tokenizer, cases, enc, cfg, *, max_options: int, seed: int, device,
            helpers) -> dict:
    """helpers: fit.py internals (param groups, bucketed stream, seed) so the replay
    stream and optimiser are exactly fit()'s."""
    from .train import objective_loss
    r = {**DEFAULTS, **dict(cfg.rl2)}
    mode = r["mode"]
    if mode not in MODES_RL2:
        raise ValueError(f"rl2.mode {mode!r} not in {MODES_RL2}")
    helpers["seed_everything"](seed)
    order_rng = random.Random(seed + 104729)
    rl_rng = random.Random(seed + 4242)
    gen = torch.Generator().manual_seed(seed + 99)
    src = r["source"] or MODE_SOURCE[mode]
    replay_cases = [c for c in cases if not c.source.startswith("rl_")]
    rl_cases = [c for c in cases if c.source.startswith(src)] if mode != "replay_only" else []
    pairs = list(iter_questions(replay_cases))
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    stream = None
    if cfg.length_bucket:
        stream = helpers["bucketed_stream"](pairs, order, cfg.steps * cfg.batch_size,
                                            cfg.batch_size, cfg.length_bucket,
                                            random.Random(seed + 7919))
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"

    # ---- RL-side data. Built BEFORE training, from the step-0 (= parent) model.
    rl_pairs = list(iter_questions(rl_cases))
    units: list = []
    info: dict = {"rl2_mode": mode, "rl2_rows": len(rl_pairs)}
    if mode in ("packet", "packet_ce"):
        by = defaultdict(list)
        for i, (c, q) in enumerate(rl_pairs):
            by[c.case_id].append(i)
        units = [(k, v) for k, v in sorted(by.items())
                 if any(rl_pairs[i][1].key.startswith("boundary_") for i in v)]
        info["rl2_groups"] = len(units)
    elif mode in ("select", "setcal", "confrank") or mode in MATRIX:
        units = list(range(len(rl_pairs)))
        info["rl2_groups"] = len(units)
    elif mode == "agreecal":
        units = build_consistency_groups(rl_pairs)
        info["rl2_groups"] = len(units)
    elif mode in ("listwise", "pointwise"):
        units = build_listwise_groups(rl_pairs)
        info["rl2_groups"] = len(units)
    elif mode == "consistency":
        units = build_consistency_groups(rl_pairs)
        info["rl2_groups"] = len(units)
    elif mode in ("rlcd", "rlcd_sft"):
        pool = list(range(len(rl_pairs)))
        random.Random(seed + 5).shuffle(pool)
        pool = sorted(pool[: r["pool_max"]])
        sub = [rl_pairs[i] for i in pool]
        t_pos = teacher_scores(sub, model_name=r["teacher"], device=device, batch=r["teacher_batch"],
                               max_tokens=r["teacher_max_tokens"], system=POS_SYSTEM,
                               lm=helpers.get("teacher_lm"), tok=helpers.get("teacher_tok"))
        t_neg = teacher_scores(sub, model_name=r["teacher"], device=device, batch=r["teacher_batch"],
                               max_tokens=r["teacher_max_tokens"], system=NEG_SYSTEM,
                               lm=helpers.get("teacher_lm"), tok=helpers.get("teacher_tok"))
        prs = rlcd_pairs(sub, t_pos, t_neg)
        units = [(pool[i], yp, yn) for i, yp, yn in prs]
        scored = sum(t is not None for t in t_pos)
        info.update(rl2_pool=len(sub), rl2_teacher_scored=scored, rl2_pairs=len(units),
                    rl2_pair_rate=round(len(units) / max(1, scored), 4))
        # diagnostic only (never a training signal): does the positive prompt beat the
        # negative one against whatever gold the pool happens to carry?
        gold_known = [(i, t_pos[i], t_neg[i]) for i in range(len(sub)) if t_pos[i] is not None
                      and max(sub[i][0].gold[sub[i][1].key]) > 0.99]
        if gold_known:
            ga = lambda i, t: float(int(t.argmax()) == max(range(len(sub[i][0].gold[sub[i][1].key])),
                                                           key=sub[i][0].gold[sub[i][1].key].__getitem__))
            info["rl2_diag_pos_acc"] = round(sum(ga(i, tp) for i, tp, _ in gold_known) / len(gold_known), 4)
            info["rl2_diag_neg_acc"] = round(sum(ga(i, tn) for i, _, tn in gold_known) / len(gold_known), 4)
    if mode != "replay_only" and not units:
        raise ValueError(f"rl2 mode {mode}: no RL-side units from source prefix {src!r}")
    print(f"    rl2: {info}", flush=True)

    ref = None
    if mode != "replay_only":
        ref = copy.deepcopy(model)
        if dev_type == "cuda":            # the scorer stays fp32 (never reduced precision)
            ref.tower.to(torch.bfloat16)
        ref.scorer.to(torch.float32)
        ref.eval()
        for p_ in ref.parameters():
            p_.requires_grad_(False)

    opt = torch.optim.AdamW(helpers["param_groups"](model, cfg), weight_decay=cfg.weight_decay)
    n_warm = max(1, int(cfg.warmup * cfg.steps))
    warm = lambda s: min(1.0, (s + 1) / n_warm)
    def warm_cosine(s):
        if s < n_warm:
            return warm(s)
        t = (s - n_warm) / max(1, cfg.steps - n_warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    base_fn = warm_cosine if cfg.base_schedule == "cosine" else warm
    head_fn = warm_cosine if cfg.head_schedule == "cosine" else warm
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [
        base_fn if gr["name"].startswith("base") else head_fn if gr["name"] == "head" else warm
        for gr in opt.param_groups])

    rl_order_rng = random.Random(seed + 271828)   # RL-side option shuffles: own stream,
                                                  # so the replay presentation is CRN-identical
    def forward(chunk, ref_too=False):
        rng_ = rl_order_rng if ref_too else order_rng
        batch = collate(tokenizer, [encode_question(tokenizer, c.state, q, enc, rng=rng_)
                                    for c, q in chunk], max_options=max_options, device=device)
        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cfg.autocast_bf16):
            lg = model(**batch)
        lg = unpermute_logits(lg.float(), batch["option_perm"], batch["option_mask"])
        lg = lg.masked_fill(~batch["option_mask"], float("-inf"))
        rl_ = None
        if ref_too:
            with torch.no_grad(), torch.autocast(dev_type, dtype=torch.bfloat16,
                                                 enabled=dev_type == "cuda"):
                rl_ = ref(**batch)
            rl_ = unpermute_logits(rl_.float(), batch["option_perm"], batch["option_mask"])
            rl_ = rl_.masked_fill(~batch["option_mask"], float("-inf"))
        return lg, rl_

    unit_order = list(range(len(units)))
    rl_rng.shuffle(unit_order)
    ucur = 0
    stats = CellStats(r["bands"], r["ema"])
    def next_units(n):
        nonlocal ucur, unit_order
        got = []
        for _ in range(n):
            if ucur >= len(unit_order):
                rl_rng.shuffle(unit_order)
                ucur = 0
            got.append(units[unit_order[ucur]]); ucur += 1
        return got

    use_replay = r["replay"] or mode == "replay_only"
    model.train()
    history, ckpts = [], []
    cursor = 0
    agg = defaultdict(float)
    agg_n = 0
    kl_sum = 0.0
    n_gated = n_rl = 0
    kl_max = 0.0
    logit_max_run = 0.0
    for step in range(cfg.steps):
        logging = step % cfg.log_every == 0 or step == cfg.steps - 1
        opt.zero_grad(set_to_none=True)
        loss_rep, lmax = torch.zeros(()), 0.0
        if use_replay:
            # ---------------- replay (identical stream in every mode that uses it)
            idx = (stream[cursor:cursor + cfg.batch_size] if stream is not None
                   else [order[(cursor + i) % len(order)] for i in range(cfg.batch_size)])
            cursor += cfg.batch_size
            chunk = [pairs[i] for i in idx]
            lg, _ = forward(chunk)
            gold = gold_tensor([c for c, _ in chunk], [q.key for _, q in chunk], max_options, device=device)
            loss_rep = objective_loss(cfg.objective, lg, gold)
            if not torch.isfinite(loss_rep):
                raise FloatingPointError(f"non-finite replay loss at step {step}")
            loss_rep.backward()
            fin = lg.detach()[torch.isfinite(lg.detach())]
            lmax = float(fin.abs().max()) if fin.numel() else 0.0
        # ---------------- RL side
        rec = {}
        if mode != "replay_only":
            if mode in ("packet", "packet_ce"):
                us = next_units(r["cases_per_step"])
                rchunk, groups = [], []
                for _, rows in us:
                    groups.append(list(range(len(rchunk), len(rchunk) + len(rows))))
                    rchunk += [rl_pairs[i] for i in rows]
            elif mode in MATRIX:
                rchunk = [rl_pairs[i] for i in next_units(r["rows_per_step"])]
            elif mode in ("select", "setcal", "confrank"):
                rchunk = [rl_pairs[i] for i in next_units(r["rows_per_step"])]
                gidx = torch.tensor([max(range(len(c.gold[q.key])), key=c.gold[q.key].__getitem__)
                                     for c, q in rchunk], device=device)
            elif mode == "agreecal":
                us = next_units(r["cases_per_step"])
                rchunk, groups = [], []
                for _, rows in us:
                    groups.append(list(range(len(rchunk), len(rchunk) + len(rows))))
                    rchunk += [rl_pairs[i] for i in rows]
            elif mode in ("listwise", "pointwise"):
                (gname, rows, gpos), = next_units(1)
                negs = [j for j in range(len(rows)) if j != gpos]
                rl_rng.shuffle(negs)
                keep = sorted([gpos] + negs[: r["group_size"] - 1])
                gi = keep.index(gpos)
                rchunk = [rl_pairs[rows[j]] for j in keep]
            elif mode == "consistency":
                us = next_units(r["cases_per_step"])
                rchunk, groups = [], []
                for _, rows in us:
                    groups.append(list(range(len(rchunk), len(rchunk) + len(rows))))
                    rchunk += [rl_pairs[i] for i in rows]
            else:
                us = next_units(r["rows_per_step"])
                rchunk = [rl_pairs[i] for i, _, _ in us]
                ypos = torch.tensor([a for _, a, _ in us], device=device)
                yneg = torch.tensor([b for _, _, b in us], device=device)
            rlg, rref = forward(rchunk, ref_too=True)
            kl_rows = kl_ref_policy(rlg, rref)
            kl = kl_rows.mean()
            if mode in ("packet", "packet_ce"):
                g_ = gold_tensor([c for c, _ in rchunk], [q.key for _, q in rchunk], max_options,
                                 device=device)
                term, mrec = packet_loss(mode, rlg, g_, [q.key for _, q in rchunk], groups,
                                         G=r["G"], generator=gen)
                rec.update(mrec)
            elif mode in MATRIX:
                g_ = gold_tensor([c for c, _ in rchunk], [q.key for _, q in rchunk], max_options,
                                 device=device)
                term, mrec = matrix_loss(mode, rlg, g_, [q.mode for _, q in rchunk], r, gen)
                rec.update(mrec)
            elif mode == "select":
                term, util, pact = selective_utility_loss(rlg, gidx, tau=r["tau"], c=r["cost_wrong"],
                                                          d=r["cost_defer"], T=r["act_temp"])
                rec.update(rl_util=util, rl_act=pact)
            elif mode == "confrank":
                g_ = gold_tensor([c for c, _ in rchunk], [q.key for _, q in rchunk],
                                 max_options, device=device)
                rank, auc, npair = conf_rank_loss(rlg, gidx, margin_temp=r["margin_temp"])
                term = r["rank_coef"] * rank
                if r["ce_anchor"]:
                    term = term + r["ce_anchor"] * objective_loss("soft_ce", rlg, g_)
                rec.update(rl_auc=auc, rl_pairs=npair)
            elif mode == "setcal":
                strata = [(cal_group(cal_source(c.case_id)), q.mode) for c, q in rchunk]
                term, sece = setcal_loss(rlg, gidx, strata, stats)
                rec.update(rl_strat_ece=sece)
            elif mode == "agreecal":
                term, tgt, agr = agreecal_loss(rlg, [q for _, q in rchunk], groups)
                rec.update(rl_target=tgt, rl_agree=agr)
            elif mode == "listwise":
                s = rlg[:, 1] - rlg[:, 0]
                term, rew, hit = pl_listwise_loss(s, gi, samples=r["samples"], k=r["ndcg_k"],
                                                  tau=r["pl_tau"], generator=gen)
                rec.update(rl_reward=rew, rl_hit1=hit)
            elif mode == "pointwise":
                g2 = gold_tensor([c for c, _ in rchunk], [q.key for _, q in rchunk], max_options,
                                 device=device)
                term = objective_loss("soft_ce", rlg, g2)
                rec.update(rl_hit1=float(int((rlg[:, 1] - rlg[:, 0]).detach().argmax()) == gi))
            elif mode == "consistency":
                p = F.softmax(rlg, -1)
                qf = torch.stack([p[j, list(q.options).index(flag_option(q.key))]
                                  for j, (_, q) in enumerate(rchunk)])
                term, agr = agreement_loss(qf, groups)
                rec.update(rl_agree=agr)
            elif mode == "rlcd":
                term, win = dpo_option_loss(rlg, rref, ypos, yneg, r["dpo_beta"])
                rec.update(rl_pref_win=win)
            else:   # rlcd_sft
                term = -F.log_softmax(rlg, -1).gather(1, ypos.view(-1, 1)).mean()
            gated = float(kl.detach()) > r["kl_gate"]
            n_rl += 1
            n_gated += int(gated)
            kl_max = max(kl_max, float(kl.detach()))
            loss_rl = r["kl_coef"] * kl + (0.0 if gated else r["rl_coef"]) * term
            if not torch.isfinite(loss_rl):
                raise FloatingPointError(f"non-finite rl2 loss at step {step} ({mode})")
            loss_rl.backward()
            finr = rlg.detach()[torch.isfinite(rlg.detach())]
            lmax = max(lmax, float(finr.abs().max()) if finr.numel() else 0.0)
            rec.update(rl_term=float(term.detach()), rl_kl=float(kl.detach()), rl_gated=int(gated))
            kl_sum += float(kl.detach())
            agg_n += 1
            for k_, v in rec.items():
                agg[k_] += v
        logit_max_run = max(logit_max_run, lmax)
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_([p for gr in opt.param_groups for p in gr["params"]],
                                           cfg.grad_clip)
        opt.step()
        sched.step()
        if logging:
            row = {"step": step, "loss": float(loss_rep.detach()), "logit_absmax": round(lmax, 2),
                   "logit_absmax_run": round(logit_max_run, 2)}
            if mode != "replay_only":
                row.update({f"{k_}_avg": round(v / max(1, agg_n), 5) for k_, v in agg.items()})
                row.update(rl_gated_frac=round(n_gated / max(1, n_rl), 4), rl_kl_max=round(kl_max, 5))
                agg.clear()
                agg_n = 0
            if step == 0:
                row.update(info)
            history.append(row)
            print("    rl2 " + " ".join(f"{k_}={v}" for k_, v in row.items()), flush=True)
        if step >= cfg.steps - cfg.keep_last_k:
            ckpts.append({k_: v.detach().cpu().clone() for k_, v in model.scorer.state_dict().items()})

    kl_mean = kl_sum / max(1, n_rl)
    gfrac = n_gated / max(1, n_rl)
    # A7: max logit < 100, batch-mean KL to the parent < 0.05, gated fraction <= 50%.
    a7 = {"a7_logit_max": round(logit_max_run, 2), "a7_kl_mean": round(kl_mean, 5),
          "a7_kl_max": round(kl_max, 5), "a7_gated_frac": round(gfrac, 4),
          "a7_stable": bool(logit_max_run < 100 and kl_mean < 0.05 and gfrac <= 0.5)}
    history.append({"step": cfg.steps, "loss": history[-1]["loss"], **a7, **info})
    print(f"    rl2 A7: {a7}", flush=True)
    del ref
    averaged = {k_: torch.stack([c[k_].float() for c in ckpts]).mean(0) for k_ in ckpts[0]}
    model.scorer.load_state_dict({k_: v.to(next(model.scorer.parameters()).dtype)
                                  for k_, v in averaged.items()})
    dev_pairs = (list(iter_questions([c for c in cases if c.source == r["dev_pool"]]))
                 if r["dev_pool"] else [])
    if dev_pairs:
        dm = dev_metrics(model, tokenizer, dev_pairs, enc, max_options=max_options, device=device)
        history[-1].update({f"dev_native_{k_}": v for k_, v in dm.items()})
        print(f"    rl2 DEV native {dm}", flush=True)
    if r["cal_method"]:
        # cal-4b (the current post-hoc method) refitted on the cal pool. With
        # cal_activate=False it is fitted and saved but the arm's records stay NATIVE;
        # the post-hoc numbers come from the dev pool here and from a rescoring job.
        from .calibrate import calibrate, save_calibration
        dev = [(c, cal_group(cal_source(c.case_id))) for c in cases if c.source == r["cal_pool"]]
        if not dev:
            raise ValueError(f"cal_method needs the {r['cal_pool']} pool in the arm's sources")
        rep = calibrate(model, tokenizer, dev, enc, method=r["cal_method"],
                        max_options=max_options, device=device)
        print("    CALIBRATION " + " ".join(f"{k_}={v}" for k_, v in rep.items()), flush=True)
        if r["cal_save_dir"]:
            save_calibration(model, r["cal_save_dir"], rep)
        history.insert(0, {"step": -1, "loss": rep["cal_dev_wbce_cal"], **rep})
        if dev_pairs:
            dm = dev_metrics(model, tokenizer, dev_pairs, enc, max_options=max_options, device=device)
            history[-1].update({f"dev_cal_{k_}": v for k_, v in dm.items()})
            print(f"    rl2 DEV cal-4b {dm}", flush=True)
        if not r["cal_activate"]:
            model.cal_mode = "none"
    return {"history": history, "averaged_over": len(ckpts),
            "examples_seen": cfg.steps * cfg.batch_size, "seed": seed}
