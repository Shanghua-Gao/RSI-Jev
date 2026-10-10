"""MLX affine group quantization, computed in PyTorch, with MLX's on-disk packing.

The scheme is `mx.quantize` in mode "affine": per row, per contiguous group of `g` input
columns, w ~ scale * q + bias with q in [0, 2^bits - 1]. The group's larger-magnitude edge
(min or max) is represented exactly:
    scale = +-(max - min) / n_bins (sign so that edge / scale >= 0), q0 = round(edge / scale),
    scale = edge / q0 if q0 != 0, bias = edge (or 0 if q0 == 0).
Codes are computed with the float32 (unrounded) scale and bias; the stored scale and bias are
then rounded to the weight dtype (bf16), and dequantization is (q * scale) + bias in that dtype.

This is tools/quant/mlxq.py of the quantization study, which was checked bit for bit against
mlx 0.32.3 (codes, scales, biases and dequantized weights, 8/6/4-bit, g32/64/128, on real
tower rows), plus `pack` / `unpack`: MLX stores the codes of a row as uint32 words, 32 // bits
codes per word, the first code in the lowest bits (bits 8 / 4 / 2; 3, 5 and 6 bits are packed
as a little-endian bit stream, which `pack` also handles). tests/test_mlx_convert.py checks
`quantize` + `pack` against `mx.quantize` itself.

Computing it in PyTorch keeps the converter independent of where MLX runs, and lets codes,
scales and biases from another quantizer (GPTQ, DWQ: the study's tools) be packed the same way.
"""
from __future__ import annotations

import numpy as np
import torch


def affine_params(w: torch.Tensor, bits: int, g: int):
    """w (..., n) -> float32 scale, bias (..., n // g) (not rounded)."""
    n_bins = (1 << bits) - 1
    wg = w.float().reshape(*w.shape[:-1], w.shape[-1] // g, g)
    w_max = wg.amax(-1)
    w_min = wg.amin(-1)
    mask = w_min.abs() > w_max.abs()
    scale = torch.clamp((w_max - w_min) / n_bins, min=1e-7)
    scale = torch.where(mask, scale, -scale)
    edge = torch.where(mask, w_min, w_max)
    q0 = torch.round(edge / scale)
    scale = torch.where(q0 != 0, edge / q0, scale)
    bias = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    return scale, bias


def quantize(w: torch.Tensor, bits: int, g: int, scale=None, bias=None):
    """(codes uint8 (same shape as w), scale, bias (..., n // g) in w's dtype)."""
    if w.shape[-1] % g:
        raise ValueError(f"last dim {w.shape[-1]} not divisible by group size {g}")
    if scale is None:
        scale, bias = affine_params(w, bits, g)
    n_bins = (1 << bits) - 1
    wg = w.float().reshape(*w.shape[:-1], w.shape[-1] // g, g)
    s, b = scale.float().unsqueeze(-1), bias.float().unsqueeze(-1)
    q = torch.clamp(torch.round((wg - b) / s), 0, n_bins)
    return q.reshape(w.shape).to(torch.uint8), scale.to(w.dtype), bias.to(w.dtype)


def dequantize(q: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, g: int,
               dtype=torch.bfloat16) -> torch.Tensor:
    qg = q.to(dtype).reshape(*q.shape[:-1], q.shape[-1] // g, g)
    w = qg * scale.to(dtype).unsqueeze(-1) + bias.to(dtype).unsqueeze(-1)
    return w.reshape(q.shape)


def qdq(w: torch.Tensor, bits: int, g: int, scale=None, bias=None) -> torch.Tensor:
    """The effective (dequantized) weight in w's dtype: what an MLX checkpoint computes with."""
    q, s, b = quantize(w, bits, g, scale, bias)
    return dequantize(q, s, b, g, w.dtype)


def packed_bytes(shape, bits: int, g: int, param_bytes: int = 2) -> int:
    """On-disk bytes of an MLX quantized matrix: packed uint32 codes + scale and bias per group."""
    n = 1
    for d in shape:
        n *= d
    return n * bits // 8 + 2 * param_bytes * (n // g)


def pack(q: np.ndarray, bits: int) -> np.ndarray:
    """Codes (..., n) (integers < 2^bits) -> MLX's uint32 words (..., n * bits // 32)."""
    q = np.asarray(q).astype(np.uint64)
    n = q.shape[-1]
    if (n * bits) % 32:
        raise ValueError(f"{n} codes of {bits} bits do not fill whole uint32 words")
    if 32 % bits == 0:
        per = 32 // bits
        qg = q.reshape(*q.shape[:-1], n // per, per)
        shifts = np.arange(per, dtype=np.uint64) * np.uint64(bits)
        return (qg << shifts).sum(-1).astype(np.uint32)
    # 3/5/6 bits: a little-endian bit stream; every 32 codes fill exactly `bits` words
    if n % 32:
        raise ValueError(f"{bits}-bit packing needs a multiple of 32 codes per row, got {n}")
    qg = q.reshape(*q.shape[:-1], n // 32, 32)
    words = np.zeros((*qg.shape[:-1], bits), dtype=np.uint64)
    mask32 = np.uint64(0xFFFFFFFF)
    for i in range(32):
        p = i * bits
        w, sh = p // 32, p % 32
        words[..., w] |= (qg[..., i] << np.uint64(sh)) & mask32
        if sh + bits > 32:
            words[..., w + 1] |= qg[..., i] >> np.uint64(32 - sh)
    return words.reshape(*q.shape[:-1], n * bits // 32).astype(np.uint32)


def unpack(words: np.ndarray, bits: int, n: int) -> np.ndarray:
    """MLX uint32 words (..., n * bits // 32) -> codes (..., n) uint8."""
    w = np.asarray(words).astype(np.uint64)
    m = np.uint64((1 << bits) - 1)
    if 32 % bits == 0:
        per = 32 // bits
        shifts = np.arange(per, dtype=np.uint64) * np.uint64(bits)
        q = (w[..., None] >> shifts) & m
        return q.reshape(*w.shape[:-1], n).astype(np.uint8)
    wg = w.reshape(*w.shape[:-1], n // 32, bits)
    out = np.zeros((*wg.shape[:-1], 32), dtype=np.uint64)
    for i in range(32):
        p = i * bits
        k, sh = p // 32, p % 32
        v = wg[..., k] >> np.uint64(sh)
        if sh + bits > 32:
            v = v | (wg[..., k + 1] << np.uint64(32 - sh))
        out[..., i] = v & m
    return out.reshape(*w.shape[:-1], n).astype(np.uint8)
