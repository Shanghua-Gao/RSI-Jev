"""Adaptive per-question early exit.

A model with arch.aux_exits has a readout head at every exit in `exits` (e.g. 12, 16 and
the main exit 20). At inference a question stops at the FIRST exit whose calibrated
confidence reaches tau; the last exit always answers.

  * calibration, per aux exit, fitted on DEV only, one of:
      - a scalar temperature ({"logT": ...}): log tau = logT for every row, the same
        arithmetic as the main head's cal_mode "temp", so argmax-preserving;
      - a cal-4b (per-input temperature log tau = w . f + b on cal_features of that
        exit's logits and decision state; log tau >= 0 on score rows). The math is
        rsijev.calibrate's oof_head_scorefloor, copied here so this module has no
        dependency on a release tree.
  * tau: one threshold on the calibrated top-1 probability, the same at every exit
    (calibration is what makes the exits comparable). Tuned on DEV.
  * staged execution (`staged_scores`): the text model runs exit-to-exit on the rows
    still undecided. Each stage is the text model with its layer list swapped for
    layers[lo:hi] and the final norm for the identity, fed the previous stage's
    un-normed state as inputs_embeds, so the decoder layers see exactly the inputs of
    one uninterrupted pass. Stage boundaries must be multiples of the layer-type period
    (Qwen3.5: 4), because the text model picks each layer's mask by its position in
    the list.

Only torch is imported, so the serving adapter can load this file from a different
rsijev tree (importlib, by path).

Ported from the research tree (staged_scores, StageReplica, staged_scores_bucketed,
and staged_scores_fast, the "opt" path: masks and
rotary once per request, direct layer calls, one host read per exit, the exact calibrator
skip, no aux heads at tau > 1; bitwise vs staged_scores). The served path
(serve/infer.score_adaptive) runs staged_scores_fast, which hands the document-cache path
to staged_scores. `finalize` (on both) is this tree's: it gives each row the calibration
of the exit that answered it. Without it, behaviour is the original.
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


def temp_cal(logT: float) -> dict:
    """A scalar-temperature calibrator for one exit: log tau = logT on every row."""
    if not math.isfinite(float(logT)):
        raise ValueError(f"logT must be finite, got {logT!r}")
    return {"logT": torch.tensor(float(logT), dtype=torch.float32)}


def calibrate_logt(z, h, mode, cal):
    """log tau per row, (B,): a scalar temperature ({"logT"}) or a cal-4b."""
    if "logT" in cal:
        return cal["logT"].expand(z.shape[0])
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
    # per-exit thresholds {exit: tau} overriding `tau` at those exits (empty: `tau` everywhere)
    taus: dict = field(default_factory=dict)

    def tau_at(self, L) -> float:
        return self.taus.get(int(L), self.tau)


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
    """The served prefix cache for the staged path, replicated ONCE per layer .

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
        stop = conf >= policy.tau_at(L)
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
    """Multi-question staged exit with first-stage-confidence depth bucketing .

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
            stop = conf >= policy.tau_at(L)
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


def _pool_cache(batch, T, dtype, device):
    """readout()'s option-span indicator and its denominator for the full batch: the same
    tensors at every exit (they depend on spans, width and dtype only), built once."""
    s0, s1 = batch.get("option_span_start"), batch.get("option_span_end")
    if s0 is None:
        return None
    t = torch.arange(T, device=device).view(1, 1, -1)
    inside = ((t >= s0.unsqueeze(-1)) & (t < s1.unsqueeze(-1))).to(dtype)
    return inside, inside.sum(-1, keepdim=True).clamp_min(1.0)


def _readout_pooled(scorer, h, batch, rows, pool, option_pool="mean", logit_cap=None):
    """readout() with the pooling indicator from _pool_cache (rows selects it): same math."""
    if pool is None or option_pool != "mean":
        return readout(scorer, h, batch, rows=rows, option_pool=option_pool, logit_cap=logit_cap)
    def g(k):
        v = batch.get(k)
        return v if (v is None or rows is None) else v[rows]
    di, mask = g("decision_index"), g("option_mask")
    inside, denom = pool if rows is None else (pool[0][rows], pool[1][rows])
    b = torch.arange(h.shape[0], device=h.device)
    decision_h = h[b, di]
    option_h = torch.einsum("bkt,bth->bkh", inside, h) / denom
    dtype = next(scorer.parameters()).dtype
    decision_h, option_h = decision_h.to(dtype), option_h.to(dtype)
    with torch.autocast(decision_h.device.type, enabled=False):
        logits = scorer(decision_h=decision_h, option_h=option_h, option_mask=mask,
                        mode_id=g("mode_id"), option_perm=g("option_perm"))
    if logit_cap:
        c = float(logit_cap)
        logits = (c * torch.tanh(logits / c)).masked_fill(~mask, float("-inf"))
    return logits, decision_h.float()


CONF_BOUND_MARGIN = 1e-5


def conf_upper_bound(z, mode, cal=None):
    """An upper bound on calibrated_conf for any cal-4b: the top-1 probability is
    non-increasing in the temperature, and a cal-4b's log tau is clamped to
    >= -CAL_LOGT_CLAMP (>= 0 on score rows). Rows whose bound is below
    tau - CONF_BOUND_MARGIN cannot stop, whatever the calibrator says. A scalar
    temperature ({"logT"}) is neither clamped nor floored, so for it the exact confidence
    (as cheap as the bound) is returned instead."""
    if cal is not None and "logT" in cal:
        return calibrated_conf(z, None, mode, cal)
    z = z.float()
    lt = torch.full((z.shape[0],), -CAL_LOGT_CLAMP, device=z.device)
    lt = _floor(lt, mode == SCORE)
    return torch.softmax(z / torch.exp(lt)[:, None], -1).max(-1).values


# ---------------------------------------------------------------- fast staged path
class _StageRunner:
    """The text model's forward, split at the exits, with the per-request work done ONCE.

    `run_layers` re-enters tm.forward for every stage, and each entry rebuilds the causal and
    recurrent masks (each with a host-synchronising `.all()` padding check), the rotary
    tables and a DynamicCache. Here those are built once for the request's rows and reused
    by every stage while the row set is unchanged; when rows drop (B > 1) they are rebuilt
    for the remaining rows, which is exactly what the current path computes for them. The
    decoder layers get the same arguments tm.forward gives them (merge_with_config_defaults:
    use_cache and is_causal from the config; a fresh DynamicCache per stage when use_cache,
    so the masks see an empty cache, as in the current path)."""

    def __init__(self, tm, input_ids, attention_mask):
        import sys
        self.tm, self.cfg = tm, tm.config
        self.mod = sys.modules[type(tm).__module__]
        self.layers = list(_stages(tm).full)
        self.types = list(self.cfg.layer_types)
        self.use_cache = getattr(self.cfg, "use_cache", None)
        self.kw = {}
        if getattr(self.cfg, "is_causal", None) is not None:
            self.kw["is_causal"] = self.cfg.is_causal
        self.ids, self.am = input_ids, attention_mask
        self.key = None

    def _cache(self):
        if not self.use_cache:
            return None
        from transformers.cache_utils import DynamicCache
        return DynamicCache(config=self.cfg)

    def prepare(self, rows, x, full: bool):
        """Masks / rotary for `rows` (x: their (n, T, H) stage input); rebuilt only when the
        row set changed since the last stage."""
        if self.key is not None:
            return
        n, T = x.shape[0], x.shape[1]
        am = None if self.am is None else (self.am if full else self.am[rows])
        cache = self._cache()
        pos = torch.arange(T, device=x.device).view(1, 1, -1).expand(4, n, -1)
        self.tpos, rpos = pos[0], pos[1:]
        mk = {"config": self.cfg, "inputs_embeds": x, "attention_mask": am, "past_key_values": cache,
              "position_ids": self.tpos}
        # transformers >= 5.17 builds the linear-attention mask with
        # create_recurrent_attention_mask; earlier 5.x forwards call the text model's own
        # _update_linear_attn_mask(attention_mask, cache). Either way, what tm.forward uses.
        rec = getattr(self.mod, "create_recurrent_attention_mask", None)
        self.masks = {"full_attention": self.mod.create_causal_mask(**mk),
                      "linear_attention": rec(**mk) if rec is not None
                      else self.tm._update_linear_attn_mask(am, cache)}
        self.pe = self.tm.rotary_emb(x, rpos)
        self.key = True

    def run(self, lo, hi, x):
        cache = self._cache()
        for j in range(lo, hi):
            x = self.layers[j](x, position_embeddings=self.pe, attention_mask=self.masks[self.types[j - lo]],
                               position_ids=self.tpos, past_key_values=cache, use_cache=self.use_cache,
                               **self.kw)
        return x


@torch.no_grad()
def staged_scores_fast(tm, scorers: dict, policy: Policy, batch: dict, *, cache_factory=None,
                       position_ids=None, option_pool="mean", logit_cap=None, stats: dict | None = None,
                       stage_cache=None, finalize=None):
    """staged_scores (same arguments, same results) with the stage-boundary overhead cut:
      * masks, rotary tables built once per request (_StageRunner), not once per stage;
      * the decoder layers are called directly (no tm.forward re-entry per stage);
      * the exit decision is ONE device->host read per non-final exit (stop flags for all rows
        in a single copy); a single-row request then answers / continues with no boolean
        indexing (each boolean index is another hidden sync), B > 1 drops rows with index
        tensors built from that one host copy;
      * the readout at an exit runs on the rows still undecided, as before.
    The document-cache path (stage_cache / cache_factory) is delegated to staged_scores."""
    if stage_cache is not None or cache_factory is not None:
        return staged_scores(tm, scorers, policy, batch, cache_factory=cache_factory,
                             position_ids=position_ids, option_pool=option_pool, logit_cap=logit_cap,
                             stats=stats, stage_cache=stage_cache, finalize=finalize)
    ids, am = batch["input_ids"], batch.get("attention_mask")
    B = ids.shape[0]
    dev = ids.device
    per = layer_period(tm)
    if any(L % per for L in policy.exits[:-1]):
        raise ValueError(f"exits {policy.exits} are not multiples of the layer-type period {per}")
    R = _StageRunner(tm, ids, am)
    rows = torch.arange(B, device=dev)
    rows_host = list(range(B))
    full = True
    out = None
    depth = torch.full((B,), policy.exits[-1], dtype=torch.long, device=dev)
    h = tm.embed_tokens(ids)
    lo = 0
    pool = None
    # tau > 1: no exit but the last can ever be taken (conf <= 1), so no aux head runs
    exits = policy.exits if any(policy.tau_at(L) <= 1.0 for L in policy.exits[:-1]) else policy.exits[-1:]
    for i, L in enumerate(exits):
        last = i == len(exits) - 1
        R.prepare(rows, h, full)
        h = R.run(lo, L, h)
        rsel = None if full else rows
        if pool is None:
            pool = _pool_cache(batch, h.shape[1], h.dtype, h.device)
        z, dh = _readout_pooled(scorers[L], tm.norm(h), batch, rsel, pool, option_pool=option_pool,
                                logit_cap=logit_cap)
        if out is None:
            out = torch.full((B, z.shape[1]), float("-inf"), dtype=z.dtype, device=z.device)
        if stats is not None:
            stats.setdefault("rows_at", {})[L] = stats.get("rows_at", {}).get(L, 0) + len(rows_host)
        if last:
            zf = z if finalize is None else finalize(L, z, dh, rows).to(out.dtype)
            if full:
                out.copy_(zf)
            else:
                out[rows] = zf
            break
        mode = batch["mode_id"] if full else batch["mode_id"][rows]
        # exact skip: if no row can reach tau under ANY calibration, the calibrator is not run
        if not any((conf_upper_bound(z, mode, policy.cal[L]) >= policy.tau_at(L) - CONF_BOUND_MARGIN).tolist()):
            if stats is not None:
                stats["cal_skipped"] = stats.get("cal_skipped", 0) + 1
            lo = L
            continue
        conf = calibrated_conf(z, dh, mode, policy.cal[L])
        stop_host = (conf >= policy.tau_at(L)).tolist()          # the one host read at this exit
        if any(stop_host):
            zs = z if finalize is None else finalize(L, z, dh, rows).to(out.dtype)
            si = [k for k, s in enumerate(stop_host) if s]
            ki = [k for k, s in enumerate(stop_host) if not s]
            if not ki:                                      # every row answered here
                if full:
                    out.copy_(zs)
                    depth.fill_(L)
                else:
                    out[rows] = zs
                    depth[rows] = L
                break
            sidx = torch.tensor(si, device=dev)
            kidx = torch.tensor(ki, device=dev)
            out[rows[sidx]] = zs[sidx]
            depth[rows[sidx]] = L
            rows, h = rows[kidx], h[kidx]
            rows_host = [rows_host[k] for k in ki]
            full = False
            R.key = None                                    # row set changed: rebuild masks/rotary
        lo = L
    return out, depth
