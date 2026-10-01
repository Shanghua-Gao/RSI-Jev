"""A fused, bit-exact rotary embedding for the vision tower: `RSIJEV_VISION_ROPE`.

The vision tower's attention rotates its queries and keys with
`apply_rotary_pos_emb_vision`: both are cast to fp32, rotated with `rotate_half`
(a slice, a negation and a concatenation), multiplied by cos and sin, added, and
cast back to bf16. That is ten memory-bound kernels per block over 16 heads x 64
dims x every patch, and at 1,600 px it is ~40 ms of the tower's ~110 ms on a GB10.

This kernel does the same arithmetic in one pass per tensor: the same fp32 values,
`x * cos` and `rotate_half(x) * sin` each rounded once, their sum rounded once (FMA
contraction is turned off, so no product skips its rounding), negation exact, and
the same round-to-nearest-even cast to bf16. Each output element is therefore the
bit pattern the eager code produces, which tests/test_vision_rope.py checks on the
GPU. It takes only what it was checked for (bf16 q/k, fp32 or bf16 cos/sin, a
power-of-two head count and head size, CUDA); anything else runs the original.
"""
from __future__ import annotations

import os

import torch

_ORIG = None


def _flag() -> bool:
    return os.environ.get("RSIJEV_VISION_ROPE", "1").strip().lower() not in (
        "", "0", "false", "no", "off")


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, sx_t, sx_h, sc_t, ss_t,
                     H: tl.constexpr, D: tl.constexpr):
        t = tl.program_id(0)
        h = tl.arange(0, H)[:, None]
        d = tl.arange(0, D)[None, :]
        half = D // 2
        dr = tl.where(d < half, d + half, d - half)
        c = tl.load(cos_ptr + t * sc_t + d).to(tl.float32)
        s = tl.load(sin_ptr + t * ss_t + d).to(tl.float32)
        base = x_ptr + t * sx_t + h * sx_h
        x = tl.load(base + d).to(tl.float32)
        xr = tl.load(base + dr).to(tl.float32)
        rot = tl.where(d < half, -xr, xr)
        out = x * c + rot * s
        tl.store(out_ptr + (t * H + h) * D + d, out.to(tl.bfloat16))

    _HAVE_TRITON = True
except Exception:                                    # no Triton: the eager code runs
    _HAVE_TRITON = False


def _pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _ok(x, cos) -> bool:
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 3 and x.stride(-1) == 1
            and cos.dim() == 2 and cos.shape[0] == x.shape[0] and cos.shape[1] == x.shape[2]
            and cos.stride(-1) == 1 and cos.dtype in (torch.float32, torch.bfloat16)
            and _pow2(x.shape[1]) and _pow2(x.shape[2]) and x.shape[2] >= 2)


def rope_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    T, H, D = x.shape
    out = torch.empty((T, H, D), dtype=x.dtype, device=x.device)
    if T:
        _rope_kernel[(T,)](x, cos, sin, out, x.stride(0), x.stride(1), cos.stride(0),
                           sin.stride(0), H=H, D=D, enable_fp_fusion=False)
    return out


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Drop-in for transformers' Qwen3.5 `apply_rotary_pos_emb_vision`."""
    if _flag() and _HAVE_TRITON and _ok(q, cos) and _ok(k, cos) and sin.shape == cos.shape \
            and sin.dtype == cos.dtype and sin.stride(-1) == 1:
        return rope_one(q, cos, sin), rope_one(k, cos, sin)
    return _ORIG(q, k, cos, sin)


def install() -> bool:
    """Route the Qwen3.5 vision tower's rotary embedding through the fused kernel
    (process-wide; RSIJEV_VISION_ROPE=0 at call time runs the original). Returns
    whether it is installed."""
    global _ORIG
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m
    except Exception:
        return False
    if not _HAVE_TRITON or not hasattr(m, "apply_rotary_pos_emb_vision"):
        return False
    if m.apply_rotary_pos_emb_vision is apply_rotary_pos_emb_vision:
        return True
    _ORIG = m.apply_rotary_pos_emb_vision
    m.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision
    return True
