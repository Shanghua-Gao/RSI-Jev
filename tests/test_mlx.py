"""The MLX port (rsijev_mlx) against the PyTorch model. Skipped without mlx.

    pip install mlx mlx-lm                     # on Linux: mlx[cpu] or mlx[cuda]
    python -m pytest tests/test_mlx.py -q      # head + calibration, random weights
    RSIJEV_CKPT=shgao/rsi-jev-v3.0-qwen3.5-2b python -m pytest tests/test_mlx.py -q

The second form also loads the release and checks it against
tests/fixtures/mlx_parity_v3.0.json, which holds PyTorch fp32 probabilities for
125 typed-decisions questions. That part needs no torch and no dataset.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("MLX_ENABLE_TF32", "0")      # CUDA backend: true fp32 matmuls
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm.models.qwen3_5")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "mlx_parity_v3.0.json"
# fp32 on both sides. Measured 4.1e-6 on MLX CUDA (GB10), the same size as PyTorch
# CUDA vs PyTorch CPU (4.6e-6); the bound leaves room for Metal's own kernels.
MAX_ABS_DP = 1e-4


def test_head_and_calibration_match_torch():
    """option_xattn + mlp combine + oof_head_scorefloor, on random weights."""
    torch = pytest.importorskip("torch")
    import numpy as np
    from rsijev.arch import ArchConfig, DecisionModel, OptionScorer
    from rsijev_mlx.model import DecisionModelMLX

    torch.manual_seed(0)
    H, B, K = 64, 5, 7
    cfg = ArchConfig(readout="option_xattn", xattn_combine="mlp", xattn_mlp_hidden=32)
    ts = OptionScorer(H, cfg).eval()
    dec, opt = torch.randn(B, H), torch.randn(B, K, H)
    mask = torch.ones(B, K, dtype=torch.bool)
    mask[1, 3:] = False
    mask[4, 2:] = False
    mode = torch.tensor([0, 1, 2, 2, 0])
    cal = {"cal_pca_mean": torch.randn(H), "cal_pca_W": torch.randn(H, 16) * 0.1,
           "cal_feat_mu": torch.randn(23) * 0.1, "cal_feat_sd": torch.rand(23) + 0.5,
           "cal_w": torch.randn(23) * 0.3, "cal_b": torch.tensor(0.1),
           "cal_logT": torch.tensor(0.0), "cal_logT_mode": torch.zeros(3)}
    with torch.no_grad():
        ref = ts(decision_h=dec, option_h=opt, option_mask=mask)
        fake = SimpleNamespace(cal_mode="oof_head_scorefloor", **cal)
        lt = DecisionModel.cal_log_temperature(fake, ref, dec, mode)
        ref_cal = ref / torch.exp(lt).unsqueeze(-1)

    m = DecisionModelMLX(None, {k: mx.array(v.numpy()) for k, v in ts.state_dict().items()},
                         {k: mx.array(v.numpy()) for k, v in cal.items()},
                         "oof_head_scorefloor")
    got = m.scorer(mx.array(dec.numpy()), mx.array(opt.numpy()), mx.array(mask.numpy()))
    got_lt = m.log_temperature(got, mx.array(dec.numpy()), mx.array(mode.numpy()))
    got_cal = got / mx.exp(got_lt)[:, None]
    fin = mask.numpy()
    np.testing.assert_allclose(np.array(got)[fin], ref.numpy()[fin], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.array(got_lt), lt.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.array(got_cal)[fin], ref_cal.numpy()[fin], rtol=1e-4, atol=1e-5)
    assert np.all(np.isneginf(np.array(got)[~fin]))


@pytest.fixture(scope="module")
def release():
    ckpt = os.environ.get("RSIJEV_CKPT")
    if not ckpt:
        pytest.skip("set RSIJEV_CKPT to a v3.0 release directory or repo id")
    from rsijev_mlx import load
    return load(ckpt)


def test_release_matches_pytorch_fixture(release):
    sys.path.insert(0, str(ROOT / "scripts"))
    from mlx_parity import compare
    model, tok, meta = release
    fixture = json.loads(FIXTURE.read_text())
    assert meta["release"]["name"] == fixture["release"]
    r = compare(fixture, model, tok, meta)
    assert r["n"] >= 100
    assert r["argmax_agreement"] == 1.0, r["flips"]
    assert r["max_abs_dp"] < MAX_ABS_DP, r["max_abs_dp"]


def test_decide_has_the_served_shape(release):
    from rsijev_mlx import decide
    model, tok, meta = release
    body = decide(model, tok, meta, {"ticket": "RV-1", "body": "Charged twice, please refund."}, {
        "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
        "team": {"type": "choice", "instructions": "Which department?",
                 "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
        "urgency": {"type": "score", "instructions": "How urgent?",
                    "criteria": ["Routine", "Urgent", "Emergency"]}})
    a = body["answers"]
    assert set(a) == {"refund", "team", "urgency"}
    assert set(a["refund"]) == {"type", "noul"} and 0 <= a["refund"]["noul"] <= 1
    assert a["team"]["choice"] in ("billing", "technical")
    assert abs(sum(a["team"]["probabilities"].values()) - 1) < 1e-5
    assert 0 <= a["urgency"]["score"] <= 2 and set(a["urgency"]["legend"]) == {"0", "1", "2"}
    assert body["usage"]["output_tokens"] == 3
