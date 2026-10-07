"""Images on the MLX path: the Qwen3.5 vision tower, image preparation, M-RoPE positions.

  * `VisionTower`: transformers' Qwen3_5VisionModel (ViT + 2x2 patch merger) in MLX, bf16,
    from the release's visual.safetensors. Patch embedding as one matmul (the Conv3d's kernel
    equals its stride), learned position table resampled bilinearly (align_corners) per
    image, 2-D rotary in fp32, attention within each image, LayerNorm, GELU (tanh) MLP,
    merger LayerNorm -> 4096 -> GELU -> text width. What rsijev.vision.load_visual runs.
  * `ImagePrep`: rsijev.vision.ImagePrep without torch: the release's own image processor
    config (transformers' Qwen2-VL processor; the torchvision backend when torchvision is
    installed, so pixels match the PyTorch path exactly, else the PIL backend), the same
    per-request token budget split evenly over the images, numpy out.
  * `mrope_positions`: transformers' Qwen3_5Model.get_rope_index in numpy: text tokens count
    up, an image's tokens take (t, h, w) from its merged grid starting at the running
    position, and the text after it continues from the grid's larger side.
"""
from __future__ import annotations

import json
import math
from itertools import groupby
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

F32 = mx.float32


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def _gelu(x):
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def _layer_norm(x, w, b, eps=1e-6):
    return mx.fast.layer_norm(x, w, b, eps)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


class VisionTower:
    def __init__(self, cfg: dict, params: dict, dtype=mx.bfloat16):
        vc = cfg["vision_config"]
        self.dtype = dtype
        self.hidden = vc["hidden_size"]
        self.heads = vc["num_heads"]
        self.depth = vc["depth"]
        self.patch = vc["patch_size"]
        self.tps = vc["temporal_patch_size"]
        self.merge = vc["spatial_merge_size"]
        self.side = int(vc["num_position_embeddings"] ** 0.5)
        theta = float((vc.get("rope_parameters") or {}).get("rope_theta", 10000.0))
        hd = self.hidden // self.heads
        spatial = hd // 2
        self.inv_freq = 1.0 / (theta ** (mx.arange(0, spatial, 2, dtype=F32) / spatial))
        g = lambda k: params[k].astype(dtype)                           # noqa: E731
        pw = params["patch_embed.proj.weight"]                          # (O, C, T, P, P)
        self.patch_w = pw.reshape(pw.shape[0], -1).astype(dtype)
        self.patch_b = g("patch_embed.proj.bias")
        self.pos_embed = g("pos_embed.weight")
        self.blocks = []
        for i in range(self.depth):
            p = f"blocks.{i}."
            self.blocks.append({k: g(p + k) for k in (
                "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "attn.qkv.weight",
                "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias", "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias", "mlp.linear_fc2.weight", "mlp.linear_fc2.bias")})
        self.merger = {k: g("merger." + k) for k in (
            "norm.weight", "norm.bias", "linear_fc1.weight", "linear_fc1.bias",
            "linear_fc2.weight", "linear_fc2.bias")}
        self.out_hidden = self.merger["linear_fc2.weight"].shape[0]

    # -- per-image geometry (numpy), as transformers.vision_utils computes it
    def _positions(self, grid: np.ndarray) -> np.ndarray:
        m = self.merge
        out = []
        for t, h, w in grid.tolist():
            hp, wp = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            shp = (h // m, m, w // m, m)
            hp = hp.reshape(shp).transpose(0, 2, 1, 3).reshape(-1)
            wp = wp.reshape(shp).transpose(0, 2, 1, 3).reshape(-1)
            out.append(np.tile(np.stack([hp, wp], -1), (t, 1)))
        return np.concatenate(out, 0)

    def _interp(self, grid: np.ndarray):
        """Bilinear (align_corners) taps + weights into the side x side position table, per
        patch in merge-block order: (N, 4) indices, (N, 4) fp32 weights."""
        side, m = self.side, self.merge
        idx, wts = [], []
        for t, h, w in grid.tolist():
            n = h * w
            within = np.arange(n)
            bw = w // m
            in_col = within % m
            in_row = (within // m) % m
            bcol = (within // (m * m)) % bw
            brow = within // (m * m * bw)
            row = (brow * m + in_row).astype(np.float32)
            col = (bcol * m + in_col).astype(np.float32)

            def taps(i, size):
                src = i * np.float32(side - 1) / np.float32(max(size - 1, 1))
                fl = np.floor(src)
                t0 = np.clip(fl.astype(np.int64)[:, None] + np.arange(2), 0, side - 1)
                d = np.abs(src[:, None] - fl[:, None] - np.arange(2, dtype=np.float32))
                return t0, np.clip(1 - d, 0, None).astype(np.float32)
            ht, hw_ = taps(row, h)
            wt, ww = taps(col, w)
            ii = (ht[:, :, None] * side + wt[:, None, :]).reshape(-1, 4)
            ww4 = (hw_[:, :, None] * ww[:, None, :]).reshape(-1, 4)
            idx.append(np.tile(ii, (t, 1)))
            wts.append(np.tile(ww4, (t, 1)))
        return np.concatenate(idx, 0), np.concatenate(wts, 0)

    def __call__(self, pixel_values: np.ndarray, grid: np.ndarray) -> mx.array:
        """pixel_values (N, C*T*P*P) fp32, grid (n_images, 3) -> features (N / merge^2, out)."""
        grid = np.asarray(grid, dtype=np.int64)
        x = mx.array(np.asarray(pixel_values, dtype=np.float32)).astype(self.dtype)
        x = x @ self.patch_w.T + self.patch_b
        ii, ww = self._interp(grid)
        pe = self.pos_embed[mx.array(ii)].astype(F32) * mx.array(ww)[:, :, None]
        x = x + pe.sum(1).astype(self.dtype)
        pos = mx.array(self._positions(grid).astype(np.float32))
        f = pos[:, :, None] * self.inv_freq                            # (N, 2, hd/4)
        f = mx.concatenate([f[:, 0], f[:, 1]], axis=-1)
        emb = mx.concatenate([f, f], axis=-1)                           # (N, hd)
        cos, sin = mx.cos(emb)[:, None, :], mx.sin(emb)[:, None, :]
        lens = (grid[:, 1] * grid[:, 2]).repeat(grid[:, 0]).tolist()
        bounds = np.concatenate([[0], np.cumsum(lens)]).tolist()
        N = x.shape[0]
        hd = self.hidden // self.heads
        for blk in self.blocks:
            h = _layer_norm(x, blk["norm1.weight"], blk["norm1.bias"])
            qkv = (h @ blk["attn.qkv.weight"].T + blk["attn.qkv.bias"]).reshape(N, 3, self.heads, hd)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            qf, kf = q.astype(F32), k.astype(F32)
            q = (qf * cos + _rotate_half(qf) * sin).astype(self.dtype)
            k = (kf * cos + _rotate_half(kf) * sin).astype(self.dtype)
            outs = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                qi = q[a:b].transpose(1, 0, 2)[None]
                ki = k[a:b].transpose(1, 0, 2)[None]
                vi = v[a:b].transpose(1, 0, 2)[None]
                o = mx.fast.scaled_dot_product_attention(qi, ki, vi, scale=hd ** -0.5)
                outs.append(o[0].transpose(1, 0, 2).reshape(b - a, self.hidden))
            o = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
            x = x + (o @ blk["attn.proj.weight"].T + blk["attn.proj.bias"])
            h = _layer_norm(x, blk["norm2.weight"], blk["norm2.bias"])
            h = _gelu_tanh((h @ blk["mlp.linear_fc1.weight"].T + blk["mlp.linear_fc1.bias"]).astype(F32)).astype(self.dtype)
            x = x + (h @ blk["mlp.linear_fc2.weight"].T + blk["mlp.linear_fc2.bias"])
        mg = self.merger
        h = _layer_norm(x, mg["norm.weight"], mg["norm.bias"]).reshape(-1, self.hidden * self.merge ** 2)
        h = _gelu((h @ mg["linear_fc1.weight"].T + mg["linear_fc1.bias"]).astype(F32)).astype(self.dtype)
        return h @ mg["linear_fc2.weight"].T + mg["linear_fc2.bias"]


# ------------------------------------------------------------------------- preprocessing
class ImagePrep:
    """PIL images -> (pixel_values np fp32, grid np (n, 3), LLM tokens per image), budgeted
    exactly as rsijev.vision.ImagePrep: tokens per image = max(min_tokens, budget // n)."""

    def __init__(self, src: str | Path, budget: int = 1024, min_tokens_per_image: int = 64,
                 backend: str | None = None):
        self.src = str(src)
        self.image_token_budget, self.min_tokens_per_image = int(budget), int(min_tokens_per_image)
        self._cls = _processor_class(backend)
        self._base = self._cls.from_pretrained(self.src)
        self.unit = self._base.patch_size * self._base.merge_size
        self.merge = self._base.merge_size
        self._procs: dict[int, object] = {}
        self.backend = type(self._base).__name__

    def tokens_per_image(self, n_images: int) -> int:
        return max(self.min_tokens_per_image, self.image_token_budget // max(1, n_images))

    def _proc(self, max_tokens: int):
        if max_tokens not in self._procs:
            u2 = self.unit * self.unit
            mn = min(self.min_tokens_per_image, max_tokens) * u2
            self._procs[max_tokens] = self._cls.from_pretrained(
                self.src, size={"shortest_edge": mn, "longest_edge": max_tokens * u2})
        return self._procs[max_tokens]

    def __call__(self, images: Sequence):
        per = self.tokens_per_image(len(images))
        out = self._proc(per)([im.convert("RGB") for im in images], return_tensors="np")
        pv = np.asarray(out["pixel_values"], dtype=np.float32)
        grid = np.asarray(out["image_grid_thw"], dtype=np.int64)
        m = self.merge
        return pv, grid, [int(np.prod(g)) // (m * m) for g in grid]


def _processor_class(backend: str | None = None):
    """Qwen2-VL's image processor: the torchvision backend (what the PyTorch path serves
    with, so pixels are identical) when torch and torchvision are installed, else the PIL
    backend (numpy; resizes with PIL's bicubic, which differs from torchvision's by a
    rounding step: see docs/inference.md "On a Mac (MLX)")."""
    if backend is None:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            backend = "torchvision"
        except ImportError:
            backend = "pil"
    if backend == "torchvision":
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        return Qwen2VLImageProcessor
    if backend == "pil":
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        return Qwen2VLImageProcessorPil
    raise ValueError(f"image processor backend must be torchvision or pil, got {backend!r}")


def mrope_positions(input_ids: np.ndarray, attention_mask: np.ndarray, grid: np.ndarray,
                    image_token_id: int, spatial_merge_size: int = 2) -> np.ndarray:
    """(3, B, T) int positions: Qwen3_5Model.get_rope_index for images (padding gets 0)."""
    B, T = input_ids.shape
    pos = np.zeros((3, B, T), dtype=np.int64)
    grids = iter(np.asarray(grid).tolist())
    m = spatial_merge_size
    for b in range(B):
        keep = attention_mask[b].astype(bool)
        ids = input_ids[b][keep]
        types = (ids == image_token_id).astype(int).tolist()
        cur = 0
        parts = []
        i0 = 0
        for key, grp in groupby(types):
            n = len(list(grp))
            if key == 0:
                parts.append(np.broadcast_to(np.arange(n) + cur, (3, n)))
                cur += n
            else:
                t, h, w = next(grids)
                lt, lh, lw = t, h // m, w // m
                tt, hh, ww = np.meshgrid(np.arange(lt), np.arange(lh) + cur, np.arange(lw) + cur,
                                         indexing="ij")
                vp = np.stack([tt, hh, ww]).reshape(3, -1)
                vp[0] += cur
                if vp.shape[1] != n:
                    raise ValueError(f"{n} image tokens for a {lt}x{lh}x{lw} grid")
                parts.append(vp)
                cur += max(h, w) // m
            i0 += n
        allp = np.concatenate(parts, axis=1) if parts else np.zeros((3, 0), dtype=np.int64)
        pos[:, b, keep] = allp
    return pos


def load_vision(path: str | Path, dtype=mx.bfloat16) -> VisionTower:
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    return VisionTower(cfg, mx.load(str(path / "visual.safetensors")), dtype=dtype)
