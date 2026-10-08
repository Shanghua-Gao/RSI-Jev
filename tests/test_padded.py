"""Padded serving (serve/padded.py) on the tiny random Qwen3.5 of test_adaptive_exit, on CPU.

  * off by default: a package without meta.json `serving.pad_multiple` serves unpadded, and
    no padded runner is ever built; RSIJEV_PAD_MULTIPLE overrides the package either way;
  * load_release reads `serving.pad_multiple` and reports it in meta["serving"];
  * padding preserves the eager answers: the fixed exit, the staged exits (auto, low,
    high) and both, through the served entry points, give the unpadded path's argmax and
    depth, with probabilities equal to float rounding;
  * the masks the padded pass builds are transformers' own on every real position.

The padded path runs on CUDA only (`active` is 0 elsewhere); here `active` is forced on and
CUDA graphs off, so the same `_Runner._stage` function runs eager on CPU.

    python -m pytest tests/test_padded.py -q
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from rsijev import adaptive as A
from serve import infer
from serve import padded as P
from tests.test_adaptive_exit import (AUX, ENC, EXIT, QS, STATES, _cal, _main_cal,  # noqa: F401
                                      _write_release, models, stub_lm, tok)

M = 64
PLAIN = dict(min_saved_tokens=10 ** 9, doc_cache=False)


@pytest.fixture
def forced(monkeypatch):
    """The padded path on CPU: `active` ignores the device, graphs off."""
    monkeypatch.setenv(P.ENV_GRAPHS, "0")
    monkeypatch.delenv(P.ENV_PAD, raising=False)
    monkeypatch.setattr(P, "active", lambda model, device: P.pad_multiple(model) if P.supported(model) else 0)


def _release(models, tau=None, calibrated=True):
    _, _, ada, _ = models
    m = copy.deepcopy(ada)
    if calibrated:
        _main_cal(m)
    m.adaptive_policy = None if tau is None else A.Policy(m.exit_indices(), {AUX: _cal(11)}, tau)
    m.adaptive_mode = "auto"
    return m


def _score(model, tok, state, qs=QS, effort=None):
    plan = infer.plan_request(tok, state, qs, ENC, **PLAIN)
    assert plan["path"] == "plain"
    if effort is not None:
        plan["effort"] = effort
    got, _ = infer.score_planned(model, tok, plan, max_options=8, device="cpu", batch_size=2)
    return [p.probs for p in got], plan.get("depth")


def _close(a, b, tol=1e-5):
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        assert max(range(len(pa)), key=pa.__getitem__) == max(range(len(pb)), key=pb.__getitem__)
        assert max(abs(x - y) for x, y in zip(pa, pb)) < tol


# -- off by default ------------------------------------------------------------------------

def test_no_pad_multiple_means_unpadded(monkeypatch):
    monkeypatch.delenv(P.ENV_PAD, raising=False)
    assert P.pad_multiple(SimpleNamespace()) == 0
    assert P.pad_multiple(SimpleNamespace(serving_pad_multiple=0)) == 0
    assert P.pad_multiple(SimpleNamespace(serving_pad_multiple=M)) == M
    assert not P.pads(1, 10, 0)


def test_the_environment_overrides_the_package(monkeypatch):
    monkeypatch.setenv(P.ENV_PAD, "0")
    assert P.pad_multiple(SimpleNamespace(serving_pad_multiple=M)) == 0
    monkeypatch.setenv(P.ENV_PAD, "32")
    assert P.pad_multiple(SimpleNamespace()) == 32


def test_only_short_batches_pad():
    assert P.pads(1, 700, M) and P.pads(2, 300, M)            # 768, 2 x 320
    assert not P.pads(2, 400, M) and not P.pads(1, 769, M)     # 2 x 448, 832


def test_cpu_never_pads(models):
    m = _release(models)
    m.serving_pad_multiple = M
    assert P.supported(m) and P.active(m, "cpu") == 0


def test_a_package_without_pad_multiple_builds_no_runner(stub_lm, tmp_path, models, monkeypatch):
    from serve.release import load_release
    monkeypatch.delenv(P.ENV_PAD, raising=False)
    _, _, ada, _ = models
    model, tok_, _, meta = load_release(_write_release(tmp_path / "rel", ada), "cpu")
    assert model.serving_pad_multiple == 0 and "pad_multiple" not in meta["serving"]
    for state in STATES[:2]:
        _score(model, tok_, state)
    assert getattr(model, "_rsijev_padded", None) is None


def test_load_release_reads_pad_multiple(stub_lm, tmp_path, models, monkeypatch):
    from serve.release import load_release
    monkeypatch.delenv(P.ENV_PAD, raising=False)
    _, _, ada, _ = models
    d = _write_release(tmp_path / "rel", ada)
    meta = json.loads((d / "meta.json").read_text())
    meta["serving"] = {"pad_multiple": M}
    (d / "meta.json").write_text(json.dumps(meta))
    model, _, _, meta = load_release(d, "cpu")
    assert model.serving_pad_multiple == M
    assert meta["serving"]["pad_multiple"] == M and meta["serving"]["pad_multiple_served"] == M
    monkeypatch.setenv(P.ENV_PAD, "0")
    assert P.pad_multiple(model) == 0


# -- padding preserves the eager answers ---------------------------------------------------

@pytest.mark.parametrize("calibrated", [False, True])
def test_fixed_exit_padded_equals_eager(tok, models, forced, calibrated):
    eager = _release(models, calibrated=calibrated)
    padded = _release(models, calibrated=calibrated)
    padded.serving_pad_multiple = M
    for state in STATES:
        for qs in (QS, QS[:1]):
            want, _ = _score(eager, tok, state, qs)
            got, _ = _score(padded, tok, state, qs)
            _close(got, want)
    assert P.runner(padded).stats["eager"] > 0                 # the padded function did run
    assert getattr(eager, "_rsijev_padded", None) is None


@pytest.mark.parametrize("effort,tau", [("auto", 0.9), ("low", 0.9), ("high", 0.9), (None, 0.6)])
def test_staged_exit_padded_equals_eager(tok, models, forced, effort, tau):
    eager = _release(models, tau=tau)
    padded = _release(models, tau=tau)
    for m in (eager, padded):
        m.effort_base = m.adaptive_policy
    padded.serving_pad_multiple = M
    depths = set()
    for state in STATES:
        want, dw = _score(eager, tok, state, effort=effort)
        got, dg = _score(padded, tok, state, effort=effort)
        assert dg == dw
        depths.update(dw)
        _close(got, want)
    assert P.runner(padded).stats["eager"] > 0
    if effort == "low":
        assert depths == {AUX}
    if effort == "high":
        assert depths == {EXIT}


def test_long_batches_stay_eager(tok, models, forced, monkeypatch):
    """A batch over the pad limit runs the unpadded code: bitwise the eager answer."""
    monkeypatch.setattr(P, "PAD_MAX_TOKENS", 0)
    eager = _release(models)
    padded = _release(models)
    padded.serving_pad_multiple = M
    for state in STATES[:3]:
        assert _score(padded, tok, state)[0] == _score(eager, tok, state)[0]
    assert getattr(padded, "_rsijev_padded", None) is None


# -- masks -----------------------------------------------------------------------------------

def test_masks_match_transformers_on_real_positions(models):
    tower = models[0]
    T, lengths = 2 * M, torch.tensor([M + 5, 2 * M - 9, 17])
    x = torch.zeros(len(lengths), T, tower.config.hidden_size)
    want = P.hf_masks(tower, x, lengths)
    got = P.own_masks(T, lengths, True, "cpu")
    assert torch.equal(got["linear_attention"], want["linear_attention"].long())
    full = want["full_attention"]
    if full.dtype != torch.bool:                               # additive float mask
        full = full == 0
    for r, L in enumerate(lengths.tolist()):
        assert torch.equal(got["full_attention"][r, :, :L], full[r, :, :L])
    one = P.own_masks(T, lengths[:1], True, "cpu")
    assert one["full_attention"] is None                       # one row: is_causal, no mask
    assert P.own_masks(T, lengths, False, "cpu") == {"full_attention": None, "linear_attention": None}
