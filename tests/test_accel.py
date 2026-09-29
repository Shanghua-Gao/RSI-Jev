"""The opt-in speed paths stay off unless asked for, and touch only what they may.

The measured cost of each (agreement, calibration, speed) is in serve/README.md;
these pin the switches and the layer filter, with no weights and no GPU.

    python -m pytest tests/test_accel.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve import accel                                    # noqa: E402


class _Model(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.tower = nn.Sequential(nn.Embedding(1000, 256), nn.Linear(256, 512)).to(dtype)
        self.scorer = nn.Linear(256, 1)


def test_nothing_applied_by_default(monkeypatch):
    for k in ("RSIJEV_FP8", "RSIJEV_COMPILE"):
        monkeypatch.delenv(k, raising=False)
    m = _Model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    assert accel.apply_env(m) == []
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert not hasattr(m, "_rsijev_numerics")


def test_flags_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(accel, "enable_fp8", lambda m, **k: calls.append("fp8") or 3)
    monkeypatch.setattr(accel, "enable_compile", lambda m, **k: calls.append(("compile", k)) or 2)
    monkeypatch.setenv("RSIJEV_FP8", "1")
    monkeypatch.setenv("RSIJEV_COMPILE", "1")
    monkeypatch.setenv("RSIJEV_COMPILE_MODE", "max-autotune-no-cudagraphs")
    done = accel.apply_env(_Model())
    assert calls == ["fp8", ("compile", {"mode": "max-autotune-no-cudagraphs"})] and len(done) == 2
    monkeypatch.setenv("RSIJEV_FP8", "0")
    monkeypatch.setenv("RSIJEV_COMPILE", "false")
    calls.clear()
    assert accel.apply_env(_Model()) == [] and calls == []


def test_fp8_filter_keeps_embeddings_and_small_projections_out():
    assert accel._fp8_eligible(nn.Linear(2048, 6144), "mlp.up_proj")
    assert not accel._fp8_eligible(nn.Linear(2048, 16), "linear_attn.in_proj_a")
    assert not accel._fp8_eligible(nn.Linear(2050, 2048), "odd")
    assert not accel._fp8_eligible(nn.Embedding(1000, 2048), "embed_tokens")
    assert not accel._fp8_eligible(nn.Conv1d(64, 64, 4, groups=64), "linear_attn.conv1d")


def test_fp8_needs_a_bf16_tower():
    with pytest.raises(ValueError, match="bf16"):
        accel.enable_fp8(_Model(torch.float32))


def test_compile_refuses_cuda_graphs():
    with pytest.raises(ValueError, match="reduce-overhead"):
        accel.enable_compile(_Model(), mode="reduce-overhead")
