"""The opt-in speed path stays off unless asked for.

The measured cost (agreement, calibration, speed) is in serve/README.md; these
pin the switch, with no weights and no GPU.

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
    monkeypatch.delenv("RSIJEV_COMPILE", raising=False)
    m = _Model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    assert accel.apply_env(m) == []
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert not hasattr(m, "_rsijev_numerics")


def test_flags_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(accel, "enable_compile", lambda m, **k: calls.append(("compile", k)) or 2)
    monkeypatch.setenv("RSIJEV_COMPILE", "1")
    monkeypatch.setenv("RSIJEV_COMPILE_MODE", "max-autotune-no-cudagraphs")
    done = accel.apply_env(_Model())
    assert calls == [("compile", {"mode": "max-autotune-no-cudagraphs"})] and len(done) == 1
    monkeypatch.setenv("RSIJEV_COMPILE", "false")
    calls.clear()
    assert accel.apply_env(_Model()) == [] and calls == []


def test_compile_refuses_cuda_graphs():
    with pytest.raises(ValueError, match="reduce-overhead"):
        accel.enable_compile(_Model(), mode="reduce-overhead")
