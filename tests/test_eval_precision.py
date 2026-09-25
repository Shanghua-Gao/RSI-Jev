"""Regression tests for eval_precision. CPU, no model, under a second.

The bug these exist for: DecisionModel is mixed precision on purpose -- the
tower is bf16 and the trained scorer is fp32 -- and restoring the module with a
single `module.to(dtype_of_first_parameter)` sends the scorer back as bf16.
Rounding a trained head is not a restore. In a real arm that corrupted the head
after the first evaluation target, so targets two and three were scored by a
different model than target one.

The guard missed it because it checksummed only the first few parameters, which
on a real DecisionModel are all tower. The corrupted part sat outside the
checked set. So the second test here is that the guard can still FIRE -- a check
that cannot fail is worth less than no check, because it is believed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.evaluate import _bitwise_checksum, eval_precision  # noqa: E402


def mixed(n_tower_linears: int) -> nn.Module:
    """The real layout: many bf16 tower tensors first, one fp32 scorer last."""
    m = nn.Module()
    m.tower = nn.Sequential(*[nn.Linear(16, 16) for _ in range(n_tower_linears)]).to(torch.bfloat16)
    m.tower.register_buffer("running", torch.ones(4, dtype=torch.bfloat16))
    m.scorer = nn.Linear(16, 4).to(torch.float32)
    return m


def test_mixed_precision_round_trip() -> None:
    # n=6 gives 12 tower parameters, so the scorer is far outside any prefix.
    for n in (1, 6):
        m = mixed(n)
        w0 = m.scorer.weight.detach().clone()
        before = _bitwise_checksum(m)
        with eval_precision(m, torch.float32):
            assert m.scorer.weight.dtype is torch.float32
            assert m.tower[0].weight.dtype is torch.float32, "tower not cast"
            assert m.tower.running.dtype is torch.float32, "buffer not cast"
        assert m.scorer.weight.dtype is torch.float32, f"n={n}: scorer came back bf16"
        assert m.tower[0].weight.dtype is torch.bfloat16, f"n={n}: tower not restored"
        assert m.tower.running.dtype is torch.bfloat16, f"n={n}: buffer not restored"
        assert torch.equal(m.scorer.weight.detach(), w0), f"n={n}: trained weights changed"
        assert _bitwise_checksum(m) == before, f"n={n}: round trip not bit-exact"


def test_guard_fires_on_corruption() -> None:
    """A tensor perturbed during the pass must be caught on restore."""
    m = mixed(6)
    try:
        with eval_precision(m, torch.float32):
            with torch.no_grad():
                m.scorer.weight.add_(1.0)      # the part a prefix checksum missed
        raise AssertionError("guard did not fire on a corrupted scorer")
    except RuntimeError as exc:
        assert "bit-exact" in str(exc), exc


def test_downcast_is_refused_at_entry() -> None:
    """Qwen3.5 loads as 321 bf16 parameters plus 2 fp32 rotary buffers, so a
    bf16 request would compute RoPE from bf16 frequencies -- a numeric path no
    training step ever used. fp32 -> bf16 -> fp32 is not a round trip: it
    discards 16 mantissa bits, and 63 of 64 inv_freq values change.
    """
    m = mixed(6)
    d, base = 128, 1000000.0
    inv = 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    m.register_buffer("inv_freq", inv)
    assert not torch.equal(inv, inv.to(torch.bfloat16).to(torch.float32)), \
        "premise wrong: fp32->bf16->fp32 would be exact"
    try:
        with eval_precision(m, torch.bfloat16):
            pass
        raise AssertionError("a down-cast was allowed")
    except ValueError as exc:
        assert "DOWN-cast" in str(exc), exc
    # and the up-cast still works on the same model
    before = _bitwise_checksum(m)
    with eval_precision(m, torch.float32):
        assert m.tower[0].weight.dtype is torch.float32
    assert _bitwise_checksum(m) == before
    assert m.inv_freq.dtype is torch.float32


def test_disabled_is_a_noop() -> None:
    m = mixed(3)
    before = _bitwise_checksum(m)
    with eval_precision(m, None):
        assert m.tower[0].weight.dtype is torch.bfloat16
    assert _bitwise_checksum(m) == before


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name}: PASS")
