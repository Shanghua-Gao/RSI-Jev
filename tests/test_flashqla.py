"""RSIJEV_FLASHQLA: the DeltaNet chunked forward routed to FlashQLA.

The switch is checked without flash_qla; the kernel tests skip unless flash_qla
(and a CUDA card it supports) is present. What it was measured to cost in
agreement, calibration and speed is in serve/README.md.

    python -m pytest tests/test_flashqla.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve import accel                                    # noqa: E402

HAVE_FQ = importlib.util.find_spec("flash_qla") is not None and torch.cuda.is_available()
needs_fq = pytest.mark.skipif(not HAVE_FQ, reason="needs flash_qla and a CUDA card")


def _ref(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
         use_qk_l2norm_in_kernel=False, cu_seqlens=None):
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    o, s = torch_chunk_gated_delta_rule(q.float(), k.float(), v.float(), g=g.float(),
                                        beta=beta.float(), initial_state=initial_state,
                                        output_final_state=output_final_state,
                                        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
    return o.to(q.dtype), s


class _Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk_gated_delta_rule = _ref
        self.calls = 0


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = nn.ModuleList([_Mixer(), nn.Linear(4, 4), _Mixer()])
        self.scorer = nn.Linear(4, 1)


def test_flag_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(accel, "enable_flashqla", lambda m: calls.append("fq") or 18)
    monkeypatch.delenv("RSIJEV_COMPILE", raising=False)
    monkeypatch.delenv("RSIJEV_FLASHQLA", raising=False)
    assert accel.apply_env(_Model()) == [] and calls == []
    monkeypatch.setenv("RSIJEV_FLASHQLA", "1")
    assert accel.apply_env(_Model()) == ["flashqla (18 DeltaNet layers)"] and calls == ["fq"]


def test_missing_flash_qla_raises(monkeypatch):
    if importlib.util.find_spec("flash_qla") is not None:
        pytest.skip("flash_qla is installed")
    with pytest.raises(ImportError):
        accel.enable_flashqla(_Model())


def _inputs(B, T, dtype=torch.bfloat16, h0=True, seed=0):
    g_ = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g_)   # noqa: E731
    q, k, v = (r(B, T, 16, 128).to(dtype) for _ in range(3))
    g = -0.05 * F.softplus(r(B, T, 16))
    beta = r(B, T, 16).sigmoid().to(dtype)
    s0 = r(B, 16, 128, 128) * 0.1 if h0 else None
    return q, k, v, g, beta, s0


@needs_fq
def test_swaps_every_mixer_once():
    m = _Model()
    assert accel.enable_flashqla(m) == 2
    assert accel.enable_flashqla(m) == 0              # already switched
    assert m._rsijev_numerics == "eager+flashqla"


@needs_fq
@pytest.mark.parametrize("B,T", [(1, 80), (4, 300), (2, 1052)])
def test_matches_reference(B, T):
    m = _Model()
    accel.enable_flashqla(m)
    q, k, v, g, beta, s0 = _inputs(B, T)
    kw = dict(initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    with torch.inference_mode():
        o, s = m.tower[0].chunk_gated_delta_rule(q, k, v, g=g, beta=beta, **kw)
        o_ref, s_ref = _ref(q, k, v, g=g, beta=beta, **kw)
    # bf16 output: one rounding of values of order 0.1-1
    assert (o.float() - o_ref.float()).abs().max() < 2e-2
    assert (s - s_ref).abs().max() / s_ref.abs().max() < 1e-2


@needs_fq
def test_fp32_and_grad_fall_back():
    m = _Model()
    accel.enable_flashqla(m)
    q, k, v, g, beta, _ = _inputs(1, 64, dtype=torch.float32, h0=False)
    with torch.inference_mode():
        o, _ = m.tower[0].chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
                                                 use_qk_l2norm_in_kernel=True)
        o_ref, _ = _ref(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    assert torch.equal(o, o_ref)                      # fp32 went to the original kernel
    q = q.bfloat16().requires_grad_()
    o, _ = m.tower[0].chunk_gated_delta_rule(q, k.bfloat16(), v.bfloat16(), g=g,
                                             beta=beta.bfloat16(), use_qk_l2norm_in_kernel=True)
    assert o.requires_grad
