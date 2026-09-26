"""The v2.0 confidence head: it may change how sharp an answer is, never which
option wins.

That is the whole safety property of post-hoc calibration here. Every method
divides a row of logits by ONE positive number, so argmax is preserved by
construction -- but "by construction" is what the score-mode floor, the feature
standardisation and the clamp each get a chance to break, so it is checked.

    python -m pytest tests/test_calibration.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsijev.arch import (CAL_LOGT_CLAMP, CAL_N_SCALAR, CAL_PCA_DIM,   # noqa: E402
                         ArchConfig, DecisionModel, cal_features)
from rsijev.contract import MODES                                     # noqa: E402

HIDDEN = 8
SCORE = MODES.index("score")


def a_model() -> DecisionModel:
    """A DecisionModel with a stub tower. Nothing here runs the tower: the
    calibration reads the logits and the decision state, both given directly."""
    return DecisionModel(torch.nn.Identity(), HIDDEN, ArchConfig())


def some_logits(n: int = 6, k: int = 4, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, k, generator=torch.Generator().manual_seed(seed))


def test_an_uncalibrated_model_is_the_identity():
    m = a_model()
    assert m.cal_mode == "none"
    with pytest.raises(ValueError):
        m.cal_log_temperature(some_logits(), torch.zeros(6, HIDDEN), None)


def test_a_temperature_never_changes_the_answer():
    m = a_model()
    m.cal_mode = "temp"
    z = some_logits()
    for log_t in (-2.0, -0.5, 0.0, 0.5, 2.0):
        m.cal_logT.fill_(log_t)
        lt = m.cal_log_temperature(z, torch.zeros(len(z), HIDDEN), None)
        assert torch.equal((z / torch.exp(lt).unsqueeze(-1)).argmax(-1), z.argmax(-1))


def test_the_per_question_head_never_changes_the_answer():
    """A random head, so the temperature genuinely varies row to row."""
    m = a_model()
    m.cal_mode = "oof_head"
    g = torch.Generator().manual_seed(7)
    m.cal_pca_W.copy_(torch.randn(HIDDEN, CAL_PCA_DIM, generator=g))
    m.cal_w.copy_(torch.randn(CAL_PCA_DIM + CAL_N_SCALAR, generator=g))
    m.cal_b.fill_(0.3)
    z, h = some_logits(32, 5, seed=3), torch.randn(32, HIDDEN, generator=g)
    mode = torch.randint(0, len(MODES), (32,), generator=g)
    lt = m.cal_log_temperature(z, h, mode)
    assert lt.shape == (32,)
    assert (torch.exp(lt) > 0).all()
    assert torch.equal((z / torch.exp(lt).unsqueeze(-1)).argmax(-1), z.argmax(-1))


def test_the_score_floor_only_ever_softens():
    """cal-4b: score rows may soften (tau >= 1) but never sharpen, which is what
    keeps their distributions close to the teacher's spread. Other modes are free."""
    m = a_model()
    g = torch.Generator().manual_seed(11)
    m.cal_pca_W.copy_(torch.randn(HIDDEN, CAL_PCA_DIM, generator=g))
    m.cal_w.copy_(torch.randn(CAL_PCA_DIM + CAL_N_SCALAR, generator=g))
    m.cal_b.fill_(-1.0)                      # bias toward sharpening, so the floor bites
    z, h = some_logits(64, 4, seed=5), torch.randn(64, HIDDEN, generator=g)
    mode = torch.randint(0, len(MODES), (64,), generator=g)

    m.cal_mode = "oof_head"
    free = m.cal_log_temperature(z, h, mode)
    m.cal_mode = "oof_head_scorefloor"
    floored = m.cal_log_temperature(z, h, mode)

    is_score = mode == SCORE
    assert (floored[is_score] >= 0).all(), "a score row was allowed to sharpen"
    assert torch.equal(floored[~is_score], free[~is_score]), "a non-score row was touched"
    assert (free[is_score] < 0).any(), "this bias should have made the floor bite"


def test_the_temperature_is_clamped():
    """Without the clamp one outlying feature can drive tau to 0 and make an
    answer one-hot, which is exactly the failure a calibration must not have."""
    m = a_model()
    m.cal_mode = "oof_head"
    m.cal_w.fill_(50.0)
    m.cal_b.fill_(50.0)
    lt = m.cal_log_temperature(some_logits(8, 3), torch.randn(8, HIDDEN), None)
    assert (lt.abs() <= CAL_LOGT_CLAMP + 1e-6).all()


def test_the_head_reads_only_this_forward_pass():
    """The features are the option distribution's shape, the mode, the option
    count and a projection of the decision state -- so calibrating costs no
    second pass. A change in width here means the saved buffers no longer fit."""
    f = cal_features(some_logits(4, 3), torch.randn(4, HIDDEN), torch.zeros(4, dtype=torch.long),
                     torch.zeros(HIDDEN), torch.zeros(HIDDEN, CAL_PCA_DIM))
    assert f.shape == (4, CAL_PCA_DIM + CAL_N_SCALAR)


def test_masked_options_do_not_reach_the_features():
    """Padded options come in as -inf. If they counted, K would be wrong and every
    feature that divides by it would drift with the batch's padding."""
    z = torch.tensor([[2.0, 1.0, 0.5, float("-inf")]])
    f = cal_features(z, torch.zeros(1, HIDDEN), None, torch.zeros(HIDDEN),
                     torch.zeros(HIDDEN, CAL_PCA_DIM))
    assert torch.isfinite(f).all()
    k = torch.exp(f[0, CAL_PCA_DIM + 3])                  # log K feature
    assert abs(k.item() - 3.0) < 1e-4, f"counted the padded option: K={k.item()}"


# ---------------------------------------------------------------------------
# The serving path. Calibration lives inside _readout, so serve/ needed no change --
# but only because the batch carries mode_id. Drop that and a score-floored model
# stops serving, so it is pinned here rather than left to the GPU smoke test.
# ---------------------------------------------------------------------------

def test_the_serving_batch_carries_mode_id():
    """`serve/infer.py` calls `model(**batch)` with whatever `collate` built. If
    mode_id stops being in it, every calibrated checkpoint raises on its first
    score question."""
    src = (ROOT / "rsijev" / "encode.py").read_text()
    assert '"mode_id"' in src, "collate no longer emits mode_id"
    src_serve = (ROOT / "serve" / "infer.py").read_text()
    assert "model(**batch)" in src_serve, (
        "serve/infer.py no longer passes the whole batch through; check mode_id still "
        "reaches the model or calibration silently stops applying")


def test_a_missing_mode_id_is_loud_not_silent():
    """The failure has to be an exception. A score row that quietly skips its floor
    would serve sharpened probabilities that no published number describes."""
    m = a_model()
    m.cal_mode = "oof_head_scorefloor"
    with pytest.raises(ValueError, match="mode_id"):
        m.cal_log_temperature(some_logits(), torch.zeros(6, HIDDEN), None)
    m.cal_mode = "temp_mode"
    with pytest.raises(ValueError, match="mode_id"):
        m.cal_log_temperature(some_logits(), torch.zeros(6, HIDDEN), None)


def test_the_api_reports_whether_it_is_calibrated():
    """A client cannot tell a calibrated answer from an uncalibrated one by looking at
    it, so GET /v1/limits says which it is getting."""
    app_src = (ROOT / "serve" / "app.py").read_text()
    assert '"calibration": calibration' in app_src
    serve_src = (ROOT / "scripts" / "serve.py").read_text()
    assert 'calibration=meta.get("calibration"' in serve_src, (
        "scripts/serve.py must pass the loaded checkpoint's calibration to create_app")
