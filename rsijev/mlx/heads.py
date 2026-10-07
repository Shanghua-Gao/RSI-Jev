"""The decision heads in MLX (fp32): option pooling, the option scorer, calibration.

Ports of rsijev/arch.py, numerically as the PyTorch path runs them:

  * `pool`: decision state = the state at decision_index; option states = the mean over each
    option's span (own-token spans, encode.py), computed in the tower dtype as
    DecisionModel._readout does, then cast to fp32
  * `OptionScorer` (readout option_xattn): q/k/v projections, multi-head cross-attention of
    the decision query over the options (torch.nn.MultiheadAttention: in_proj, 1/sqrt(d_head)
    scaling, key padding mask, out_proj), combined per option as "sum" / "mlp" / "bilinear" /
    "bilinear_norm" (ArchConfig.xattn_combine), -inf on padded slots; optional head-input
    RMS norm and logit cap
  * calibration: a scalar temperature (cal_mode "temp"), per mode ("temp_mode"), or the
    per-input head (oof_head, oof_head_scorefloor, oof_head_joint) on cal_features
"""
from __future__ import annotations

import math

import mlx.core as mx

F32 = mx.float32
MODES = ("choice", "noul", "score")
CAL_LOGT_CLAMP = 3.0


def _gelu(x):
    """torch.nn.GELU() (exact, erf)."""
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def _rms(x):
    return x * mx.rsqrt((x * x).mean(-1, keepdims=True) + 1e-6)


class OptionScorer:
    """rsijev.arch.OptionScorer for readout option_xattn, from its state dict (fp32)."""

    def __init__(self, sd: dict, *, heads: int = 4, combine: str = "sum",
                 head_input_norm: bool = False):
        g = lambda k: sd[k].astype(F32)                                 # noqa: E731
        self.q_w, self.q_b = g("q.weight"), g("q.bias")
        self.k_w, self.k_b = g("k.weight"), g("k.bias")
        self.v_w, self.v_b = g("v.weight"), g("v.bias")
        d = self.q_w.shape[0]
        ipw, ipb = g("attn.in_proj_weight"), g("attn.in_proj_bias")
        self.wq, self.wk, self.wv = ipw[:d], ipw[d:2 * d], ipw[2 * d:]
        self.bq, self.bk, self.bv = ipb[:d], ipb[d:2 * d], ipb[2 * d:]
        self.out_w, self.out_b = g("attn.out_proj.weight"), g("attn.out_proj.bias")
        self.proj_w, self.proj_b = g("proj.weight"), g("proj.bias")
        self.heads, self.d, self.combine = heads, d, combine
        self.head_input_norm = head_input_norm
        if combine == "mlp":
            self.c0_w, self.c0_b = g("comb.0.weight"), g("comb.0.bias")
            self.c2_w, self.c2_b = g("comb.2.weight"), g("comb.2.bias")
        elif combine in ("bilinear", "bilinear_norm"):
            self.bil_w = g("bil.weight")
        elif combine != "sum":
            raise ValueError(f"xattn_combine {combine!r}")

    def __call__(self, decision_h: mx.array, option_h: mx.array, option_mask: mx.array) -> mx.array:
        """decision_h (B, D), option_h (B, K, D) fp32, option_mask (B, K) bool -> (B, K)."""
        if self.head_input_norm:
            decision_h, option_h = _rms(decision_h), _rms(option_h)
        q = decision_h @ self.q_w.T + self.q_b                          # (B, D)
        k = option_h @ self.k_w.T + self.k_b                            # (B, K, D)
        v = option_h @ self.v_w.T + self.v_b
        B, K, D = v.shape
        h, dh = self.heads, D // self.heads
        Q = (q @ self.wq.T + self.bq).reshape(B, 1, h, dh).transpose(0, 2, 1, 3)
        Kh = (k @ self.wk.T + self.bk).reshape(B, K, h, dh).transpose(0, 2, 1, 3)
        Vh = (v @ self.wv.T + self.bv).reshape(B, K, h, dh).transpose(0, 2, 1, 3)
        s = (Q @ Kh.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(dh))     # (B, h, 1, K)
        s = mx.where(option_mask[:, None, None, :], s, -mx.inf)
        a = mx.softmax(s, axis=-1)
        ctx = (a @ Vh).transpose(0, 2, 1, 3).reshape(B, 1, D)
        ctx = ctx @ self.out_w.T + self.out_b                           # (B, 1, D)
        if self.combine == "sum":
            logits = ((v + ctx) @ self.proj_w.T + self.proj_b)[..., 0]
        elif self.combine == "mlp":
            c = mx.broadcast_to(ctx, v.shape)
            x = mx.concatenate([v, v * c, v - c], axis=-1)
            logits = (_gelu(x @ self.c0_w.T + self.c0_b) @ self.c2_w.T + self.c2_b)[..., 0]
        elif self.combine == "bilinear":
            logits = (v @ self.proj_w.T + self.proj_b)[..., 0] + ((v @ self.bil_w.T) * ctx).sum(-1)
        else:
            bil = ((_rms(v) @ self.bil_w.T) * _rms(ctx)).sum(-1) / v.shape[-1]
            logits = (v @ self.proj_w.T + self.proj_b)[..., 0] + bil
        return mx.where(option_mask, logits, -mx.inf)


def pool(h: mx.array, decision_index: mx.array, span_start: mx.array, span_end: mx.array,
         option_index: mx.array, option_pool: str = "mean"):
    """(decision state (B, D) fp32, option states (B, K, D) fp32) from states h (B, T, D)."""
    B, T, D = h.shape
    b = mx.arange(B)
    decision = h[b, decision_index]
    if option_pool == "mean":
        t = mx.arange(T)[None, None, :]
        inside = ((t >= span_start[..., None]) & (t < span_end[..., None])).astype(h.dtype)
        denom = mx.maximum(inside.sum(-1, keepdims=True), 1.0).astype(h.dtype)
        options = (inside @ h) / denom
    else:
        options = h[b[:, None], option_index]
    return decision.astype(F32), options.astype(F32)


def cap_logits(logits: mx.array, cap: float | None, mask: mx.array) -> mx.array:
    if not cap:
        return logits
    c = float(cap)
    return mx.where(mask, c * mx.tanh(logits / c), -mx.inf)


# --------------------------------------------------------------------------- calibration
def cal_features(logits, decision_h, mode_id, pca_mean, pca_W):
    """rsijev.arch.cal_features: PCA of the decision state, p_top, top-2 gap, normalised
    entropy, log K, mode one-hot."""
    z = logits.astype(F32)
    finite = mx.isfinite(z)
    k = mx.maximum(finite.sum(-1), 2).astype(F32)
    p = mx.softmax(z, axis=-1)
    zz = mx.where(finite, z, -1e9)
    top2 = mx.sort(zz, axis=-1)[:, -2:]
    gap = mx.clip(top2[:, 1] - top2[:, 0], 0, 30)
    ent = mx.where(finite, -(p * mx.log(mx.maximum(p, 1e-12))), 0).sum(-1) / mx.log(k)
    ptop = p.max(-1)
    if mode_id is None:
        mode_id = mx.zeros((z.shape[0],), dtype=mx.int32)
    onehot = (mode_id[:, None] == mx.arange(len(MODES))[None, :]).astype(F32)
    proj = (decision_h.astype(F32) - pca_mean) @ pca_W
    return mx.concatenate([proj, ptop[:, None], gap[:, None], ent[:, None], mx.log(k)[:, None],
                           onehot], axis=-1)


class Calibration:
    """The main head's calibration (calibration.safetensors + calibration.json cal_mode)."""

    def __init__(self, mode: str = "none", buffers: dict | None = None):
        self.mode = mode
        self.b = {k: v.astype(F32) for k, v in (buffers or {}).items()}

    def log_t(self, logits, decision_h, mode_id) -> mx.array:
        B = logits.shape[0]
        m = self.mode
        if m == "temp":
            return mx.broadcast_to(self.b["cal_logT"], (B,))
        if m == "temp_mode":
            return self.b["cal_logT_mode"][mode_id]
        if m in ("oof_head", "oof_head_scorefloor", "oof_head_joint"):
            f = cal_features(logits, decision_h, mode_id, self.b["cal_pca_mean"], self.b["cal_pca_W"])
            f = (f - self.b["cal_feat_mu"]) / self.b["cal_feat_sd"]
            lt = mx.clip(f @ self.b["cal_w"] + self.b["cal_b"], -CAL_LOGT_CLAMP, CAL_LOGT_CLAMP)
            if m == "oof_head_scorefloor":
                lt = mx.where(mode_id == MODES.index("score"), mx.maximum(lt, 0.0), lt)
            return lt
        raise ValueError(f"unknown cal_mode {m!r}")

    def apply(self, logits, decision_h, mode_id) -> mx.array:
        if self.mode == "none":
            return logits
        return logits.astype(F32) / mx.exp(self.log_t(logits, decision_h, mode_id))[:, None]


def exit_logt(cal: dict, z, decision_h, mode_id) -> mx.array:
    """An aux exit's calibrator (rsijev.adaptive.calibrate_logt): {"logT": x} or a cal-4b."""
    if "logT" in cal:
        return mx.broadcast_to(mx.array(float(cal["logT"]), dtype=F32), (z.shape[0],))
    f = (cal_features(z, decision_h, mode_id, cal["mean"], cal["W"]) - cal["mu"]) / cal["sd"]
    lt = mx.clip(f @ cal["w"] + cal["b"], -CAL_LOGT_CLAMP, CAL_LOGT_CLAMP)
    return mx.where(mode_id == MODES.index("score"), mx.maximum(lt, 0.0), lt)


def calibrated_conf(z, decision_h, mode_id, cal) -> mx.array:
    """Top-1 probability of softmax(z / tau(x)) (rsijev.adaptive.calibrated_conf)."""
    z = z.astype(F32)
    lt = exit_logt(cal, z, decision_h, mode_id)
    return mx.softmax(z / mx.exp(lt)[:, None], axis=-1).max(-1)
