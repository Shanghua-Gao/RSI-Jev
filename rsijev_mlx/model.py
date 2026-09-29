"""The v3.0 decision model in MLX: tower, option scorer, calibration.

The tower is mlx-lm's own Qwen3.5 text model (`mlx_lm.models.qwen3_5`), which
already implements the hybrid stack -- Gated DeltaNet layers and gated full
attention every fourth layer. What this file adds is the part mlx-lm does not
have: loading a release checkpoint into it, and the readout that turns the final
hidden states into one logit per option (`rsijev/arch.py`, option_xattn with the
"mlp" combine) and the fitted calibration (`cal_mode` oof_head_scorefloor).

The scorer and the calibration always run in fp32, as in PyTorch. The tower
runs in the dtype passed to `load` (fp32 by default, which is what the parity
test checks).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_5 import GatedDeltaNet, Qwen3_5TextModel, TextModelArgs

from rsijev.contract import MODES

CAL_LOGT_CLAMP = 3.0                      # rsijev/arch.py
CONVERTED = "mlx_model.safetensors"       # written by rsijev_mlx.convert
# Qwen3.5's RMSNorm scales by (1 + weight); mlx-lm's nn.RMSNorm scales by weight.
# The gated norm inside each DeltaNet layer (linear_attn.norm) is a plain weight
# in both, so it is not in this list.
_SHIFTED_NORMS = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                  ".q_norm.weight", ".k_norm.weight")
DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}


def resolve(ckpt: str | Path) -> Path:
    """A local directory, or a Hugging Face repo id to download."""
    p = Path(ckpt).expanduser()
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(ckpt)))


def base_dir(base_model: str) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(base_model, allow_patterns=[
        "*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]))


def sanitize_tower(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """PyTorch layout -> mlx-lm layout for the text tower (keys unprefixed)."""
    out = {}
    for k, v in weights.items():
        if k.endswith("conv1d.weight") and v.shape[-1] != 1:
            v = v.moveaxis(2, 1)                    # (C, 1, K) -> (C, K, 1)
        if k == "norm.weight" or k.endswith(_SHIFTED_NORMS):
            v = v.astype(mx.float32) + 1.0
        out[k] = v
    return out


def release_tower_weights(ckpt: Path, meta: dict) -> tuple[dict, dict]:
    """(sanitized tower weights, text config) straight from a release directory.

    The release ships the tower without its embedding, which was frozen in
    training and is taken from the public base model, exactly as
    scripts/load_release.py does.
    """
    base = base_dir(meta["base_model"])
    cfg = json.loads((base / "config.json").read_text())
    text_cfg = cfg.get("text_config", cfg)
    tower = mx.load(str(ckpt / "tower.safetensors"))
    embed = None
    for f in sorted(base.glob("*.safetensors")):
        w = mx.load(str(f))
        for key in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight"):
            if key in w:
                embed = w[key]
                break
        if embed is not None:
            break
    if embed is None:
        raise RuntimeError(f"no embed_tokens in {meta['base_model']}")
    tower["embed_tokens.weight"] = embed
    return sanitize_tower(tower), text_cfg


class ExactGatedDeltaNet(GatedDeltaNet):
    """mlx-lm's GatedDeltaNet with the q/k normalisation of the reference.

    PyTorch (and fla) L2-normalise q and k as x * rsqrt(sum(x^2) + 1e-6).
    mlx-lm uses rms_norm, i.e. rsqrt(mean(x^2) + 1e-6), which is the same up to a
    constant factor except that the epsilon is effectively 128x larger. After the
    conv and SiLU, mean(q^2) is ~0.015 here, so that epsilon is not negligible:
    it moved the delta-rule output by ~1e-3 relative and the final probabilities
    by up to 2e-3. This is the only change; no cache, since nothing here decodes.
    """

    def __call__(self, inputs: mx.array, mask=None, cache=None) -> mx.array:
        if cache is not None:
            raise ValueError("ExactGatedDeltaNet runs full sequences only")
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b, a = self.in_proj_b(inputs), self.in_proj_a(inputs)
        conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
        conv_out = nn.silu(self.conv1d(mx.concatenate([conv_state, qkv], axis=1)))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim])]

        def l2norm(t):
            t32 = t.astype(mx.float32)
            return (t32 * mx.rsqrt((t32 * t32).sum(-1, keepdims=True) + 1e-6)).astype(t.dtype)

        q = (self.head_k_dim ** -0.5) * l2norm(q)
        k = l2norm(k)
        out, _ = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, None, None,
                                    use_kernel=not self.training)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


class DecisionModelMLX:
    """Tower + option_xattn scorer + calibration. Call `logits(batch)`."""

    def __init__(self, tower: Qwen3_5TextModel, scorer: dict, cal: dict | None,
                 cal_mode: str, heads: int = 4):
        self.tower = tower
        self.s = {k: v.astype(mx.float32) for k, v in scorer.items()}
        self.cal = None if cal is None else {k: v.astype(mx.float32) for k, v in cal.items()}
        self.cal_mode = cal_mode if cal is not None else "none"
        self.heads = heads
        if self.cal_mode not in ("none", "temp", "temp_mode", "oof_head",
                                 "oof_head_scorefloor"):
            raise ValueError(f"unsupported cal_mode {self.cal_mode!r}")

    # --- scorer: rsijev.arch.OptionScorer, readout option_xattn, combine mlp ---
    def _lin(self, x, name):
        return x @ self.s[f"{name}.weight"].T + self.s[f"{name}.bias"]

    def scorer(self, decision_h: mx.array, option_h: mx.array,
               option_mask: mx.array) -> mx.array:
        s = self.s
        q = self._lin(decision_h, "q")                      # (B, D)
        k, v = self._lin(option_h, "k"), self._lin(option_h, "v")   # (B, K, D)
        B, K, D = v.shape
        H, hd = self.heads, D // self.heads
        w_in, b_in = s["attn.in_proj_weight"], s["attn.in_proj_bias"]
        qh = (q @ w_in[:D].T + b_in[:D]).reshape(B, 1, H, hd)
        kh = (k @ w_in[D:2 * D].T + b_in[D:2 * D]).reshape(B, K, H, hd)
        vh = (v @ w_in[2 * D:].T + b_in[2 * D:]).reshape(B, K, H, hd)
        att = (qh * kh).sum(-1) / math.sqrt(hd)             # (B, K, H)
        att = mx.where(option_mask[..., None], att, -mx.inf)
        att = mx.softmax(att, axis=1)
        ctx = (att[..., None] * vh).sum(1).reshape(B, D)    # (B, D)
        ctx = ctx @ s["attn.out_proj.weight"].T + s["attn.out_proj.bias"]
        c = mx.broadcast_to(ctx[:, None, :], v.shape)
        x = mx.concatenate([v, v * c, v - c], axis=-1)
        x = nn.gelu(self._lin(x, "comb.0"))                 # exact (erf) GELU, as nn.GELU()
        logits = self._lin(x, "comb.2")[..., 0]
        return mx.where(option_mask, logits, -mx.inf)

    # --- calibration: rsijev.arch.cal_features / cal_log_temperature ---
    def log_temperature(self, logits, decision_h, mode_id):
        c = self.cal
        if self.cal_mode == "temp":
            return mx.broadcast_to(c["cal_logT"], (logits.shape[0],))
        if self.cal_mode == "temp_mode":
            return c["cal_logT_mode"][mode_id]
        finite = mx.isfinite(logits)
        kk = mx.maximum(finite.sum(-1), 2).astype(mx.float32)
        p = mx.softmax(logits, axis=-1)
        z = mx.where(finite, logits, -1e9)
        top2 = mx.sort(z, axis=-1)[:, ::-1][:, :2]
        gap = mx.clip(top2[:, 0] - top2[:, 1], 0, 30)
        ent = -mx.where(finite, p * mx.log(mx.maximum(p, 1e-12)), 0).sum(-1) / mx.log(kk)
        ptop = p.max(-1)
        onehot = (mode_id[:, None] == mx.arange(len(MODES))[None, :]).astype(mx.float32)
        proj = (decision_h - c["cal_pca_mean"]) @ c["cal_pca_W"]
        f = mx.concatenate([proj, ptop[:, None], gap[:, None], ent[:, None],
                            mx.log(kk)[:, None], onehot], axis=-1)
        f = (f - c["cal_feat_mu"]) / c["cal_feat_sd"]
        lt = mx.clip(f @ c["cal_w"] + c["cal_b"], -CAL_LOGT_CLAMP, CAL_LOGT_CLAMP)
        if self.cal_mode == "oof_head_scorefloor":
            lt = mx.where(mode_id == MODES.index("score"), mx.maximum(lt, 0.0), lt)
        return lt

    def logits(self, input_ids: np.ndarray, decision_index: np.ndarray,
               span_start: np.ndarray, span_end: np.ndarray, option_mask: np.ndarray,
               mode_id: np.ndarray) -> mx.array:
        """Calibrated logits in PRESENTED option order, (B, K), -inf where masked.

        Right padding needs no attention mask: every layer is causal, so a pad
        token after a sequence cannot change any position the readout reads.
        """
        h = self.tower(mx.array(input_ids)).astype(mx.float32)     # (B, T, H) final, normed
        B, T, _ = h.shape
        b = mx.arange(B)
        decision_h = h[b, mx.array(decision_index)]
        t = np.arange(T)[None, None, :]
        inside = ((t >= span_start[..., None]) & (t < span_end[..., None])).astype(np.float32)
        denom = np.maximum(inside.sum(-1, keepdims=True), 1.0)
        option_h = (mx.array(inside) @ h) / mx.array(denom)          # mean over each block
        om = mx.array(option_mask)
        logits = self.scorer(decision_h, option_h, om)
        if self.cal_mode != "none":
            lt = self.log_temperature(logits, decision_h, mx.array(mode_id))
            logits = logits / mx.exp(lt)[:, None]
        return logits


def load(ckpt: str | Path, dtype: str = "float32"):
    """Load a release (or a converted directory). Returns (model, tokenizer, meta).

    `ckpt` is a release directory, a converted directory (rsijev_mlx.convert), or
    a Hugging Face repo id such as shgao/rsi-jev-v3.0-qwen3.5-2b.
    """
    from transformers import AutoTokenizer
    ckpt = resolve(ckpt)
    meta = json.loads((ckpt / "meta.json").read_text())
    spec = meta["spec"]
    if (spec["readout"], (spec.get("arch_extra") or {}).get("xattn_combine")) != \
            ("option_xattn", "mlp") or spec.get("readout_layer", -1) != -1 \
            or spec.get("option_pool") != "mean" or spec.get("residual") \
            or spec.get("logit_cap") or spec.get("head_input_norm"):
        raise ValueError("rsijev_mlx implements the v3.0 head: option_xattn + mlp combine, "
                         "final layer, mean pooling, no residual/cap/input norm")
    if (ckpt / CONVERTED).exists():
        weights = mx.load(str(ckpt / CONVERTED))
        text_cfg = json.loads((ckpt / "mlx_config.json").read_text())
        tok = AutoTokenizer.from_pretrained(str(ckpt))
    else:
        weights, text_cfg = release_tower_weights(ckpt, meta)
        tok = AutoTokenizer.from_pretrained(meta["base_model"])
    args = TextModelArgs.from_dict(text_cfg)
    tower = Qwen3_5TextModel(args)
    for layer in tower.layers:
        if layer.is_linear:
            layer.linear_attn.__class__ = ExactGatedDeltaNet
    dt = DTYPES[dtype]
    weights = {k: (v if k.endswith("A_log") else v.astype(dt)) for k, v in weights.items()}
    tower.load_weights(list(weights.items()), strict=True)
    tower.eval()
    mx.eval(tower.parameters())
    scorer = mx.load(str(ckpt / "scorer.safetensors"))
    cal, cal_mode = None, "none"
    if (ckpt / "calibration.safetensors").exists():
        cal = mx.load(str(ckpt / "calibration.safetensors"))
        cal_mode = json.loads((ckpt / "calibration.json").read_text())["cal_mode"]
    model = DecisionModelMLX(tower, scorer, cal, cal_mode)
    meta["calibration"] = model.cal_mode
    meta["mlx_dtype"] = dtype
    return model, tok, meta
