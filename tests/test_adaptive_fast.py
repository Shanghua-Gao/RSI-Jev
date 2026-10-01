"""AX-4: staged_scores_fast (the served adaptive path) == staged_scores, bitwise (logits
and depth), on a tiny Qwen3.5. Ported from the private AX-4 test (eb42d54).

Exits 4/8/12 of a 16-layer tower (period 4). Policies: forced exit at each exit (tau -1 on
[4, 12] / [8, 12], tau 2), mixed taus; single-row requests and a 24-row batch (rows drop
between stages); with and without a finalizer.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_adaptive_fast.py -q
"""
from __future__ import annotations

import pytest
import torch

from test_adaptive_exit import A, H, _arch, _batch, _text_config, tok  # noqa: F401

from rsijev.arch import DecisionModel, _text_model

LAYERS, EXITS = 16, (4, 8, 12)


@pytest.fixture(scope="module")
def ada(tok):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    torch.manual_seed(0)
    tower = Qwen3_5TextModel(_text_config(LAYERS, len(tok))).eval()
    torch.manual_seed(17)
    m = DecisionModel(tower, H, _arch(exit_layer=EXITS[-1], aux_exits=EXITS[:-1])).eval()
    torch.manual_seed(3)
    for p in m.aux_scorers.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    return m


def _cal(seed):
    g = torch.Generator().manual_seed(seed)
    n = A.CAL_PCA_DIM + 7
    return {"mean": torch.zeros(H), "W": torch.randn(H, A.CAL_PCA_DIM, generator=g) / H ** .5,
            "mu": torch.zeros(n), "sd": torch.ones(n), "w": torch.randn(n, generator=g) * .1,
            "b": torch.tensor(0.0)}


def _fin(L, z, dh, rows):
    return z.float() * (1.0 + L / 100.0)


def _run(fn, ada, pol, b, fin, **kw):
    tm = _text_model(ada.tower)
    sc = {L: ada.exit_scorer(L) for L in pol.exits}
    with torch.no_grad():
        return fn(tm, sc, pol, b, option_pool="mean", finalize=fin, **kw)


def _rows(b, i):
    return {k: (v[i:i + 1] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == b["input_ids"].shape[0] else v)
            for k, v in b.items()}


def _policies(conf_taus):
    cal = {4: _cal(1), 8: _cal(2)}
    out = [("force4", A.Policy([4, 12], {4: cal[4]}, -1.0)),
           ("force8", A.Policy([8, 12], {8: cal[8]}, -1.0)),
           ("force12", A.Policy(list(EXITS), cal, 2.0)),
           ("first4", A.Policy(list(EXITS), cal, -1.0))]
    out += [(f"tau{t:.3f}", A.Policy(list(EXITS), cal, t)) for t in conf_taus]
    return out


@pytest.mark.parametrize("fin", [None, _fin])
def test_fast_equals_current_bitwise(tok, ada, fin):
    b = _batch(tok)
    n = b["input_ids"].shape[0]
    stats = {}
    for name, pol in _policies((0.3, 0.45, 0.6, 0.9, 0.99, 0.9999, 1.5)):
        z0, d0 = _run(A.staged_scores, ada, pol, b, fin)
        z1, d1 = _run(A.staged_scores_fast, ada, pol, b, fin)
        assert torch.equal(d0, d1), name
        assert torch.equal(z0, z1), (name, float((z0 - z1)[torch.isfinite(z0)].abs().max()))
        stats[name] = {L: int((d1 == L).sum()) for L in EXITS}
        for i in range(0, n, 5):                                   # one-row requests
            bi = _rows(b, i)
            z0, d0 = _run(A.staged_scores, ada, pol, bi, fin)
            z1, d1 = _run(A.staged_scores_fast, ada, pol, bi, fin)
            assert torch.equal(d0, d1) and torch.equal(z0, z1), (name, i)
    assert stats["force4"][4] == n and stats["force8"][8] == n and stats["force12"][12] == n
    # at least one tau mixes exits (rows drop between stages in the batched call)
    assert any(sum(v[L] > 0 for L in EXITS) >= 2 for k, v in stats.items() if k.startswith("tau")), stats


def test_tau_above_one_runs_no_aux_head(tok, ada):
    """tau > 1 can never stop early: the fast path runs no aux head and no calibrator."""
    b = _batch(tok)
    pol = _policies(())[2][1]
    seen = []
    hooks = [ada.aux_scorers[str(L)].register_forward_hook(lambda *a, L=L: seen.append(L)) for L in EXITS[:-1]]
    try:
        stats = {}
        _, d = _run(A.staged_scores_fast, ada, pol, b, None, stats=stats)
    finally:
        for h in hooks:
            h.remove()
    assert seen == [] and stats["rows_at"] == {EXITS[-1]: b["input_ids"].shape[0]}
    assert bool((d == EXITS[-1]).all())
