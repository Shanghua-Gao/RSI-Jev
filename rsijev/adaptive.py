"""Adaptive per-question early exit (jevtr_v1_adaexit).

A model with arch.aux_exits has a readout head at every exit in `exits` (e.g. 12, 16 and
the main exit 20). At inference a question stops at the FIRST exit whose calibrated
confidence reaches tau; the last exit always answers.

  * calibration: one cal-4b per exit (per-input temperature log tau = w . f + b on
    cal_features of that exit's logits and decision state; log tau >= 0 on score rows),
    fitted on DEV only. The math is rsijev.calibrate (v4 0ce56aa, oof_head_scorefloor),
    copied here so this module has no dependency on a release tree.
  * tau: one threshold on the calibrated top-1 probability, the same at every exit
    (calibration is what makes the exits comparable). Tuned on DEV (tools/adaexit).
  * staged execution (`staged_scores`): the text model runs exit-to-exit on the rows
    still undecided. Each stage is the text model with its layer list swapped for
    layers[lo:hi] and the final norm for the identity, fed the previous stage's
    un-normed state as inputs_embeds, so the decoder layers see exactly the inputs of
    one uninterrupted pass. Stage boundaries must be multiples of the layer-type period
    (Qwen3.5: 4), because the text model picks each layer's mask by its position in
    the list.

Only torch is imported, so the serving adapter can load this file from a different
rsijev tree (importlib, by path).

Ported from the private branch adaexit (4dc31ff). The one addition is `finalize` on
staged_scores: the served path uses it to give each row the calibration of the exit
that answered it (serve/infer.score_adaptive). Without it, behaviour is the original.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("choice", "noul", "score")
SCORE = MODES.index("score")
CAL_PCA_DIM = 16
CAL_LOGT_CLAMP = 3.0
# group -> weight (rsijev.calibrate GROUP_WEIGHTS + "other", as cal5.py)
GROUP_WEIGHTS = {"td": 0.20, "nimble_up": 0.25, "nimble_train": 0.05, "mc": 0.10, "kev_hard": 0.16,
                 "kev_docs": 0.05, "kev_devtools": 0.05, "synth": 0.14, "ts": 0.08, "other": 0.05}
GROUP_OF = [("td_train", "td"), ("synth", "synth"), ("mc_replay", "mc"), ("st_kev_hard", "kev_hard"),
            ("st_kev_documents", "kev_docs"), ("st_kev_devtools", "kev_devtools"),
            ("st_nimble_train", "nimble_train"), ("cov_tasksource", "ts"), ("ds2_tasksource", "ts"),
            ("st_procedural", "synth"), ("cov_open_jev", "synth"), ("ds2_open_jev", "synth")]


def group_of(src: str) -> str:
    """cal5.py's DEV grouping of a training source."""
    src = src[3:] if src.startswith("rp_") else src
    for p, g in GROUP_OF:
        if src.startswith(p):
            return g
    return "nimble_up" if src.startswith(("st_", "jsg_", "cov_")) else "other"


def hash_int(tag: str, cid: str) -> int:
    return int(hashlib.sha256((tag + cid).encode()).hexdigest(), 16)


# ---------------------------------------------------------------- calibration (cal-4b)
def cal_features(logits, decision_h, mode_id, pca_mean, pca_W):
    """rsijev.arch.cal_features (v4): PCA of the decision state + p_top, top-2 gap,
    normalised entropy, log K, mode one-hot."""
    z = logits.float()
    finite = torch.isfinite(z)
    k = finite.sum(-1).clamp_min(2).float()
    p = torch.softmax(z, dim=-1)
    top2 = torch.topk(z.masked_fill(~finite, -1e9), 2, dim=-1).values
    gap = (top2[:, 0] - top2[:, 1]).clamp(0, 30)
    ent = -(p * torch.log(p.clamp_min(1e-12))).masked_fill(~finite, 0).sum(-1) / torch.log(k)
    ptop = p.max(-1).values
    if mode_id is None:
        mode_id = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
    onehot = F.one_hot(mode_id.long(), len(MODES)).float()
    proj = (decision_h.float() - pca_mean) @ pca_W
    return torch.cat([proj, ptop[:, None], gap[:, None], ent[:, None], torch.log(k)[:, None], onehot], -1)


def _weights(groups, device):
    n = {}
    for g in groups:
        n[g] = n.get(g, 0) + 1
    present = {g: GROUP_WEIGHTS.get(g, 0.0) for g in n}
    tot = sum(present.values()) or 1.0
    w = torch.tensor([present[g] / tot / n[g] for g in groups], device=device, dtype=torch.float32)
    return w / w.sum()


def _top_terms(z):
    finite = torch.isfinite(z)
    zz = z.masked_fill(~finite, -1e30)
    top = zz.argmax(-1)
    zmax = zz.gather(1, top[:, None])
    d = (z - zmax).masked_fill(~finite, -1e4)
    is_top = torch.zeros_like(finite)
    is_top.scatter_(1, top[:, None], True)
    return d, is_top, top


def _logp_top(d, is_top, log_t):
    s = d / torch.exp(log_t)[:, None]
    s_all = torch.logsumexp(s, -1)
    s_rest = torch.logsumexp(s.masked_fill(is_top, float("-inf")), -1)
    return -s_all, s_rest - s_all


def _bce(d, is_top, correct, w, log_t):
    lp, lq = _logp_top(d, is_top, log_t)
    lq = torch.nan_to_num(lq, neginf=-1e4)
    return -(w * (correct * lp + (1 - correct) * lq)).sum()


def _grid_logt(d, is_top, correct, w) -> float:
    grid = torch.linspace(-2.5, 2.5, 501, device=d.device)
    vals = [float(_bce(d, is_top, correct, w, lt.expand(d.shape[0]))) for lt in grid]
    return float(grid[min(range(len(vals)), key=vals.__getitem__)])


def _floor(lt, floor_mask):
    return lt if floor_mask is None else torch.where(floor_mask, lt.clamp_min(0.0), lt)


def _fit_head(f, d, is_top, correct, w, lam, b0, steps=600, floor_mask=None):
    wv = torch.zeros(f.shape[1], device=f.device, requires_grad=True)
    b = torch.tensor(b0, device=f.device, requires_grad=True)
    opt = torch.optim.Adam([wv, b], lr=0.03)
    for _ in range(steps):
        lt = _floor((f @ wv + b).clamp(-CAL_LOGT_CLAMP, CAL_LOGT_CLAMP), floor_mask)
        loss = _bce(d, is_top, correct, w / w.sum(), lt) + lam * (wv ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return wv.detach(), b.detach()


def ece(conf, correct, w=None, bins=15) -> float:
    w = torch.ones_like(conf) if w is None else w
    w = w / w.sum()
    b = (conf * bins).long().clamp(max=bins - 1)
    return float(sum(abs(float((w[b == k] * (correct[b == k] - conf[b == k])).sum()))
                     for k in range(bins) if (b == k).any()))


@torch.enable_grad()
def fit_cal4b(z, h, mode, y, groups, case_ids, lam_grid=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1), folds=5) -> dict:
    """cal-4b (oof_head_scorefloor) on one exit's DEV logits z (canonical order), decision
    states h, modes, gold indices y. Ridge by K-fold CV over cases. Returns the params
    `calibrate_logt` needs (all tensors on z's device) plus fit diagnostics."""
    z, h = z.float(), h.float()
    d, is_top, top = _top_terms(z)
    correct = (top == y).float()
    w = _weights(groups, z.device)
    lt_g = _grid_logt(d, is_top, correct, w)
    mean = h.mean(0)
    _, _, V = torch.pca_lowrank(h - mean, q=CAL_PCA_DIM, center=False)
    W = V[:, :CAL_PCA_DIM]
    f = cal_features(z, h, mode, mean, W)
    mu = (w[:, None] * f).sum(0)
    sd = ((w[:, None] * (f - mu) ** 2).sum(0)).sqrt().clamp_min(1e-4)
    fs = (f - mu) / sd
    fm = mode == SCORE
    fold = torch.tensor([hash_int("cal-fold:", c) % folds for c in case_ids], device=z.device)
    cv = {}
    for lam in lam_grid:
        tot = 0.0
        for k in range(folds):
            tr, te = fold != k, fold == k
            wv, b = _fit_head(fs[tr], d[tr], is_top[tr], correct[tr], w[tr] / w[tr].sum(), lam, lt_g,
                              floor_mask=fm[tr])
            lte = _floor((fs[te] @ wv + b).clamp(-CAL_LOGT_CLAMP, CAL_LOGT_CLAMP), fm[te])
            tot += float(_bce(d[te], is_top[te], correct[te], w[te], lte))
        cv[lam] = tot
    lam = min(cv, key=cv.get)
    wv, b = _fit_head(fs, d, is_top, correct, w, lam, lt_g, floor_mask=fm)
    cal = {"mean": mean, "W": W, "mu": mu, "sd": sd, "w": wv, "b": b}
    lt = calibrate_logt(z, h, mode, cal)
    conf0 = torch.softmax(z, -1).max(-1).values
    conf1 = torch.softmax(z / torch.exp(lt)[:, None], -1).max(-1).values
    diag = {"lambda": lam, "cv": cv, "T_global": round(math.exp(lt_g), 4), "n": int(z.shape[0]),
            "dev_acc": round(float((w * correct).sum()), 4),
            "dev_ece_raw": round(ece(conf0, correct, w), 4), "dev_ece_cal": round(ece(conf1, correct, w), 4)}
    return {"cal": cal, "diag": diag}


def calibrate_logt(z, h, mode, cal):
    f = (cal_features(z, h, mode, cal["mean"], cal["W"]) - cal["mu"]) / cal["sd"]
    lt = (f @ cal["w"] + cal["b"]).clamp(-CAL_LOGT_CLAMP, CAL_LOGT_CLAMP)
    return _floor(lt, mode == SCORE)


def calibrated_conf(z, h, mode, cal):
    """Top-1 probability of softmax(z / tau(x)): the exit statistic."""
    z = z.float()
    lt = calibrate_logt(z, h, mode, cal)
    return torch.softmax(z / torch.exp(lt)[:, None], -1).max(-1).values


def cal_to(cal: dict, device) -> dict:
    return {k: v.to(device=device, dtype=torch.float32) for k, v in cal.items()}


# ---------------------------------------------------------------- policy (offline)
def simulate(conf: dict, correct: dict, tau: float, exits: list[int]):
    """Per-row exit under threshold tau, from every exit's precomputed confidence and
    correctness ({exit: (N,) tensor}). Returns (exit index per row, correct per row)."""
    n = next(iter(correct.values())).shape[0]
    dev = next(iter(correct.values())).device
    chosen = torch.full((n,), exits[-1], dtype=torch.long, device=dev)
    ok = correct[exits[-1]].clone()
    undecided = torch.ones(n, dtype=torch.bool, device=dev)
    for L in exits[:-1]:
        stop = undecided & (conf[L] >= tau)
        chosen[stop] = L
        ok[stop] = correct[L][stop]
        undecided &= ~stop
    return chosen, ok


# ---------------------------------------------------------------- staged execution
@dataclass
class Policy:
    """Exits shallow to deep (the last always answers), per-exit cal-4b params for every
    exit but the last, and the threshold. tau > 1 never exits early."""
    exits: list
    cal: dict = field(default_factory=dict)       # {exit: cal dict}
    tau: float = 2.0


def layer_period(tm) -> int:
    types = list(tm.config.layer_types)
    for p in range(1, len(types) + 1):
        if all(types[i] == types[i % p] for i in range(len(types))):
            return p
    return len(types)


class _Stages:
    """Cached ModuleLists of layers[lo:hi] (plain attributes: nothing enters a state_dict)."""

    def __init__(self, tm):
        self.tm = tm
        self.full = tm.layers
        self.cache = {}

    def get(self, lo, hi):
        if (lo, hi) not in self.cache:
            self.cache[(lo, hi)] = nn.ModuleList(list(self.full)[lo:hi])
        return self.cache[(lo, hi)]


def _stages(tm) -> _Stages:
    st = getattr(tm, "_adaexit_stages", None)
    if st is None or st.full is not tm.layers:
        st = _Stages(tm)
        object.__setattr__(tm, "_adaexit_stages", st)
    return st


def run_layers(tm, lo: int, hi: int, *, input_ids=None, inputs_embeds=None, attention_mask=None,
               **extra) -> torch.Tensor:
    """Decoder layers lo..hi-1 of the text model, un-normed output (B, T, H)."""
    per = layer_period(tm)
    if lo % per:
        raise ValueError(f"stage start {lo} is not a multiple of the layer-type period {per}")
    st = _stages(tm)
    full_layers, full_norm = tm.layers, tm.norm
    tm.layers, tm.norm = st.get(lo, hi), nn.Identity()
    try:
        out = tm(input_ids=input_ids if inputs_embeds is None else None, inputs_embeds=inputs_embeds,
                 attention_mask=attention_mask, **extra)
    finally:
        tm.layers, tm.norm = full_layers, full_norm
    return out.last_hidden_state


def readout(scorer, h, batch, rows=None, option_pool="mean", logit_cap=None):
    """DecisionModel._readout (no residual / base term) on normed states h (B, T, H).
    Returns (logits (B, K), decision_h fp32 (B, H))."""
    def g(k):
        v = batch.get(k)
        return v if (v is None or rows is None) else v[rows]
    di, mask = g("decision_index"), g("option_mask")
    b = torch.arange(h.shape[0], device=h.device)
    decision_h = h[b, di]
    option_h = None
    s0, s1 = g("option_span_start"), g("option_span_end")
    if s0 is not None and option_pool == "mean":
        t = torch.arange(h.shape[1], device=h.device).view(1, 1, -1)
        inside = ((t >= s0.unsqueeze(-1)) & (t < s1.unsqueeze(-1))).to(h.dtype)
        denom = inside.sum(-1, keepdim=True).clamp_min(1.0)
        option_h = torch.einsum("bkt,bth->bkh", inside, h) / denom
    elif g("option_index") is not None:
        option_h = h[b.unsqueeze(1), g("option_index")]
    dtype = next(scorer.parameters()).dtype
    decision_h = decision_h.to(dtype)
    if option_h is not None:
        option_h = option_h.to(dtype)
    with torch.autocast(decision_h.device.type, enabled=False):
        logits = scorer(decision_h=decision_h, option_h=option_h, option_mask=mask,
                        mode_id=g("mode_id"), option_perm=g("option_perm"))
    if logit_cap:
        c = float(logit_cap)
        logits = (c * torch.tanh(logits / c)).masked_fill(~mask, float("-inf"))
    return logits, decision_h.float()


_REPLICA_ATTRS = ("conv_states", "recurrent_states", "is_conv_states_initialized",
                  "is_recurrent_states_initialized", "has_previous_state", "conv_kernel_size")


class StageReplica:
    """The served prefix cache for the staged path, replicated ONCE per layer (AX-2).

    The current path builds a fresh full replica (every cached layer, all rows) for every
    stage. Stages run disjoint layer ranges [lo, hi), so a stage only needs the rows it runs
    for ITS layers: view(lo, hi, n) replicates layers lo..hi-1 to n rows (the same
    index-0 reorder as serve/infer._replicate, so the tensors are bitwise the same) and
    leaves every other layer as the caller's untouched batch-1 prefix. The model reads the
    prefix length (get_seq_length / mask sizes) and has_previous_state from those prefix
    layers, which no stage writes, so each view sees exactly what a fresh replica shows.
    Across a request each cached layer is replicated once, for the rows that reach it."""

    def __init__(self, cache, device):
        self.base, self.device = cache, device
        self.replicated_layers = 0          # bookkeeping for the latency report

    def view(self, lo: int, hi: int, n: int):
        import copy
        v = copy.copy(self.base)
        v.layers = list(self.base.layers)
        idx = torch.zeros(n, dtype=torch.long, device=self.device)
        for j in range(lo, min(hi, len(self.base.layers))):
            src = self.base.layers[j]
            dst = copy.copy(src)
            for attr in _REPLICA_ATTRS:
                val = getattr(src, attr, None)
                if isinstance(val, (list, dict)):
                    setattr(dst, attr, val.copy())
            dst.reorder_cache(idx)
            v.layers[j] = dst
            self.replicated_layers += 1
        return v

    __call__ = view


@torch.no_grad()
def staged_scores(tm, scorers: dict, policy: Policy, batch: dict, *, cache_factory=None,
                  position_ids=None, option_pool="mean", logit_cap=None, stats: dict | None = None,
                  stage_cache=None, finalize=None):
    """Adaptive-exit logits for a collated batch (presented order, -inf where masked) and
    the exit index each row stopped at.

    scorers: {exit: OptionScorer}. cache_factory(n) -> a fresh prefix cache for n rows (the
    served cached path; each stage gets its own replica, so its layers read the prefix
    state alone), with position_ids the rows' absolute positions. Rows that stop are
    dropped from the next stage. stage_cache(lo, hi, n) (a StageReplica) replaces
    cache_factory: one replica per layer range instead of a full replica per stage.

    finalize(L, z, decision_h, rows) -> logits, if given, maps the logits a row is answered
    with at exit L (after the stop decision, which always reads the raw z): the served
    path applies that exit's calibration there. None keeps z."""
    ids, am = batch["input_ids"], batch.get("attention_mask")
    B = ids.shape[0]
    rows = torch.arange(B, device=ids.device)
    out = None
    depth = torch.full((B,), policy.exits[-1], dtype=torch.long, device=ids.device)
    h, lo = None, 0
    for i, L in enumerate(policy.exits):
        last = i == len(policy.exits) - 1
        extra = {}
        if stage_cache is not None:
            extra = {"past_key_values": stage_cache(lo, L, int(rows.numel())), "use_cache": True,
                     "position_ids": position_ids[rows]}
        elif cache_factory is not None:
            extra = {"past_key_values": cache_factory(int(rows.numel())), "use_cache": True,
                     "position_ids": position_ids[rows]}
        h = run_layers(tm, lo, L, input_ids=ids[rows] if lo == 0 else None,
                       inputs_embeds=h if lo > 0 else None,
                       attention_mask=None if am is None else am[rows], **extra)
        z, dh = readout(scorers[L], tm.norm(h), batch, rows=rows, option_pool=option_pool,
                        logit_cap=logit_cap)
        if out is None:
            out = torch.full((B, z.shape[1]), float("-inf"), dtype=z.dtype, device=z.device)
        if stats is not None:
            stats.setdefault("rows_at", {})[L] = stats.get("rows_at", {}).get(L, 0) + int(rows.numel())
        if last:
            out[rows] = z if finalize is None else finalize(L, z, dh, rows).to(out.dtype)
            break
        conf = calibrated_conf(z, dh, batch["mode_id"][rows], policy.cal[L])
        stop = conf >= policy.tau
        if bool(stop.any()):
            zs = z if finalize is None else finalize(L, z, dh, rows).to(out.dtype)
            out[rows[stop]] = zs[stop]
            depth[rows[stop]] = L
            keep = ~stop
            rows, h = rows[keep], h[keep]
        if rows.numel() == 0:
            break
        lo = L
    return out, depth


@torch.no_grad()
def staged_scores_bucketed(tm, scorers: dict, policy: Policy, batch: dict, *, group: int = 16,
                           stage_cache=None, position_ids=None, option_pool="mean", logit_cap=None):
    """Multi-question staged exit with first-stage-confidence depth bucketing (AX-2).

    batch holds ALL rows of one request (one collate, common width). Stage 1 (layers
    0..exits[0]) runs in row groups of `group` in request order and its calibrated
    confidence is the depth predictor: its cost is the real first stage, nothing extra.
    Rows that continue are pooled ACROSS the request, ordered by that confidence (most
    confident first, so rows likely to stop at the next exit share groups), and every later
    stage runs the pooled rows in groups of `group` -- instead of each 16-row chunk walking
    its own stages with its own replica. Same layers, same heads, same calibrators and tau
    as staged_scores; only the batch composition of the deeper stages differs."""
    ids, am = batch["input_ids"], batch.get("attention_mask")
    B = ids.shape[0]
    exits = policy.exits
    depth = torch.full((B,), exits[-1], dtype=torch.long, device=ids.device)
    out = None

    def stage(rows, lo, L, x):
        extra = {}
        if stage_cache is not None:
            extra = {"past_key_values": stage_cache(lo, L, int(rows.numel())), "use_cache": True,
                     "position_ids": position_ids[rows]}
        h = run_layers(tm, lo, L, input_ids=x if lo == 0 else None, inputs_embeds=x if lo > 0 else None,
                       attention_mask=None if am is None else am[rows], **extra)
        z, dh = readout(scorers[L], tm.norm(h), batch, rows=rows, option_pool=option_pool,
                        logit_cap=logit_cap)
        return h, z, dh

    rows_all = torch.arange(B, device=ids.device)
    lo, rows, h, key = 0, rows_all, None, None
    for i, L in enumerate(exits):
        last = i == len(exits) - 1
        if i == 1:                      # bucket once, by first-stage confidence; order kept after
            order = torch.argsort(key, descending=True, stable=True)
            rows, h = rows[order], h[order]
        nr, nh, nk = [], [], []
        for g0 in range(0, int(rows.numel()), group):
            r = rows[g0:g0 + group]
            hg, z, dh = stage(r, lo, L, ids[r] if lo == 0 else h[g0:g0 + group])
            if out is None:
                out = torch.full((B, z.shape[1]), float("-inf"), dtype=z.dtype, device=z.device)
            if last:
                out[r] = z
                continue
            conf = calibrated_conf(z, dh, batch["mode_id"][r], policy.cal[L])
            stop = conf >= policy.tau
            if bool(stop.any()):
                out[r[stop]] = z[stop]
                depth[r[stop]] = L
            keep = ~stop
            nr.append(r[keep]); nh.append(hg[keep])
            if i == 0:
                nk.append(conf[keep])
        if last:
            break
        rows = torch.cat(nr)
        if rows.numel() == 0:
            break
        h = torch.cat(nh)
        if i == 0:
            key = torch.cat(nk)
        lo = L
    return out, depth
