"""Images on the MLX path: the Qwen3.5 vision tower, image preparation, M-RoPE positions.

  * `VisionTower`: transformers' Qwen3_5VisionModel (ViT + 2x2 patch merger) in MLX, bf16,
    from the release's visual.safetensors. Patch embedding as one matmul (the Conv3d's kernel
    equals its stride), learned position table resampled bilinearly (align_corners) per
    image, 2-D rotary in fp32, attention within each image, LayerNorm, GELU (tanh) MLP,
    merger LayerNorm -> 4096 -> GELU -> text width. What rsijev.vision.load_visual runs.
  * `ImagePrep`: rsijev.vision.ImagePrep without torch: the release's own image processor
    config (transformers' Qwen2-VL processor with the torchvision backend when torchvision
    is installed, so pixels match the PyTorch path exactly, else `NativeProcessor`, Pillow and
    numpy, equal to transformers' PIL backend), the same per-request token budget split
    evenly over the images, numpy out.
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

from .text import Linear

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
    def __init__(self, cfg: dict, params: dict, dtype=mx.bfloat16, group: int | None = None):
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

        def lin(name):
            """A block's linear, plain in `dtype` or MLX affine-quantized (`group` from mlx.json)."""
            m = Linear.from_params(params, name, group)
            m.b = m.b.astype(dtype)
            if not m.quantized:
                m.w = m.w.astype(dtype)
            return m

        pw = params["patch_embed.proj.weight"]                          # (O, C, T, P, P)
        self.patch_w = pw.reshape(pw.shape[0], -1).astype(dtype)
        self.patch_b = g("patch_embed.proj.bias")
        self.pos_embed = g("pos_embed.weight")
        self.blocks = []
        for i in range(self.depth):
            p = f"blocks.{i}."
            blk = {k: g(p + k) for k in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias")}
            blk.update({k: lin(p + k) for k in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")})
            self.blocks.append(blk)
        self.merger = {k: g("merger." + k) for k in ("norm.weight", "norm.bias")}
        self.merger.update({k: lin("merger." + k) for k in ("linear_fc1", "linear_fc2")})
        self.out_hidden = params["merger.linear_fc2.bias"].shape[0]

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
            qkv = blk["attn.qkv"](h).reshape(N, 3, self.heads, hd)
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
            x = x + blk["attn.proj"](o)
            h = _layer_norm(x, blk["norm2.weight"], blk["norm2.bias"])
            h = _gelu_tanh(blk["mlp.linear_fc1"](h).astype(F32)).astype(self.dtype)
            x = x + blk["mlp.linear_fc2"](h)
            # one block at a time: a lazy graph over all blocks keeps every block's attention
            # scores alive (28 GB peak for one 1,024-token image)
            mx.eval(x)
        mg = self.merger
        h = _layer_norm(x, mg["norm.weight"], mg["norm.bias"]).reshape(-1, self.hidden * self.merge ** 2)
        h = _gelu(mg["linear_fc1"](h).astype(F32)).astype(self.dtype)
        return mg["linear_fc2"](h)


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


class NativeProcessor:
    """Qwen2VLImageProcessorPil of transformers 5.17 in Pillow and numpy: bicubic resize to
    multiples of patch x merge, scale to [0, 1], normalize, one temporal copy per patch."""

    def __init__(self, config: dict, size: dict | None = None):
        size = size or config["size"]
        self.min_pixels, self.max_pixels = int(size["shortest_edge"]), int(size["longest_edge"])
        self.patch_size, self.merge_size = int(config["patch_size"]), int(config["merge_size"])
        self.temporal_patch_size = int(config["temporal_patch_size"])
        self.mean = np.array(config["image_mean"], dtype=np.float32)[:, None, None]
        self.std = np.array(config["image_std"], dtype=np.float32)[:, None, None]
        self.config = config

    @classmethod
    def from_pretrained(cls, src: str, size: dict | None = None) -> "NativeProcessor":
        return cls(json.loads((Path(src) / "preprocessor_config.json").read_text()), size)

    def _resized(self, height: int, width: int) -> tuple[int, int]:
        """transformers' smart_resize."""
        if max(height, width) / min(height, width) > 200:
            raise ValueError(f"absolute aspect ratio must be smaller than 200, got "
                             f"{max(height, width) / min(height, width)}")
        factor = self.patch_size * self.merge_size
        h, w = round(height / factor) * factor, round(width / factor) * factor
        if h * w > self.max_pixels:
            beta = math.sqrt((height * width) / self.max_pixels)
            h = max(factor, math.floor(height / beta / factor) * factor)
            w = max(factor, math.floor(width / beta / factor) * factor)
        elif h * w < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
        return h, w

    def __call__(self, images: Sequence, return_tensors: str = "np") -> dict:
        from PIL import Image
        patches, grids = [], []
        p, m, t = self.patch_size, self.merge_size, self.temporal_patch_size
        for im in images:
            h, w = self._resized(im.height, im.width)
            a = np.asarray(im.resize((w, h), resample=Image.Resampling.BICUBIC)).transpose(2, 0, 1)
            a = (a.astype(np.float64) * (1 / 255)).astype(np.float32)
            a = (a - self.mean) / self.std
            gh, gw = h // p, w // p
            x = a.reshape(3, gh // m, m, p, gw // m, m, p).transpose(1, 4, 2, 5, 0, 3, 6)
            x = np.broadcast_to(x[:, :, :, :, :, None, :, :], (*x.shape[:5], t, *x.shape[5:]))
            patches.append(x.reshape(gh * gw, 3 * t * p * p))
            grids.append([1, gh, gw])
        return {"pixel_values": np.concatenate(patches, axis=0),
                "image_grid_thw": np.array(grids, dtype=np.int64)}


def _processor_class(backend: str | None = None):
    """The torchvision backend of Qwen2-VL's image processor (what the PyTorch path serves
    with, so pixels are identical) when torch and torchvision are installed, else
    NativeProcessor, which resizes with PIL's bicubic: that differs from torchvision's by a
    rounding step (see docs/inference.md "On a Mac (MLX)"). "pil" is transformers' PIL backend."""
    if backend is None:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            backend = "torchvision"
        except ImportError:
            backend = "native"
    if backend == "native":
        return NativeProcessor
    if backend == "torchvision":
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        return Qwen2VLImageProcessor
    if backend == "pil":
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        return Qwen2VLImageProcessorPil
    raise ValueError(f"image processor backend must be native, torchvision or pil, got {backend!r}")


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
    rec = json.loads((path / "mlx.json").read_text()) if (path / "mlx.json").exists() else {}
    return VisionTower(cfg, mx.load(str(path / "visual.safetensors")), dtype=dtype,
                       group=(rec.get("vision") or {}).get("group_size"))
