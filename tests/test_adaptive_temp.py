"""Per-exit scalar temperature for adaptive exit.

calibration.json carries the main head's calibration (cal_mode "temp", cal_logT in
calibration.safetensors) and one scalar temperature per aux exit:
"exits": {"<exit>": {"cal_mode": "temp", "logT": x}}. An aux exit calibrated this way must
answer exactly as a single-exit model built at that exit with the same temperature, and
the adaptive path must give every question exactly the output of the exit it stopped at.
"""
import copy
import json
import math

import pytest
import torch

from rsijev import adaptive as A
from serve import infer
from tests.test_adaptive_exit import (AUX, ENC, EXIT, QS, STATES, PATHS, _dc, _write_release,  # noqa: F401
                                      models, stub_lm, tok)

T_MAIN, T_AUX = math.log(2.4), math.log(2.3)


def _temp(model, logT):
    model.cal_mode = "temp"
    model.cal_logT.fill_(logT)
    return model


def _pair(models, tau, mode="auto"):
    """(single-exit model at AUX, fixed model at EXIT, adaptive model), temperatures on."""
    _, _, ada, short = models
    sh = _temp(copy.deepcopy(short), T_AUX)
    fx = _temp(copy.deepcopy(ada), T_MAIN)
    fx.adaptive_policy = None
    ad = _temp(copy.deepcopy(ada), T_MAIN)
    ad.adaptive_policy = A.Policy(ad.exit_indices(), {AUX: A.temp_cal(T_AUX)}, tau)
    ad.adaptive_mode = mode
    return sh, fx, ad


def _served(model, state, path, qs=QS):
    cfg = PATHS[path]
    plan = infer.plan_request(tok_ref[0], state, qs, ENC, min_saved_tokens=cfg["min_saved_tokens"],
                              doc_cache=_dc(cfg["doc"]))
    assert plan["path"] == path
    got, _ = infer.score_planned(model, tok_ref[0], plan, max_options=8, device="cpu", batch_size=2)
    return plan, [p.probs for p in got]


tok_ref = []


@pytest.fixture(autouse=True)
def _keep_tok(tok):
    tok_ref[:] = [tok]


@pytest.mark.parametrize("path", list(PATHS))
def test_a_temperature_exit_is_the_single_exit_model_bitwise(models, path):
    """tau -1: every question stops at AUX; its probabilities are, bitwise, those of a model
    built at exit AUX with cal_mode temp and the same logT."""
    sh, _, ad = _pair(models, -1.0)
    for state in STATES[:4]:
        plan, got = _served(ad, state, path)
        assert plan["depth"] == [AUX] * len(QS)
        _, ref = _served(sh, state, path)
        assert got == ref, (state[:20], path)


@pytest.mark.parametrize("path", list(PATHS))
def test_each_question_gets_exactly_its_exits_output(models, path):
    """A tau between the AUX confidences: some questions stop at AUX, the rest at EXIT.
    A question answered at AUX gets, bitwise, the single-exit model's output there. One
    answered at EXIT gets the fixed model's output up to fp32 batch-composition noise:
    the rows that stopped leave its batch, so the deep layers run on fewer rows (the same
    tolerance as the staged-execution tests)."""
    sh, fx, _ = _pair(models, 2.0)
    confs = []
    for state in STATES:
        _, probs = _served(sh, state, path)
        confs += [max(p) for p in probs]
    tau = sorted(confs)[len(confs) // 2]
    _, _, ad = _pair(models, tau)
    seen = set()
    for state in STATES:
        plan, got = _served(ad, state, path)
        _, at_aux = _served(sh, state, path)
        _, at_main = _served(fx, state, path)
        for d, g, a, m in zip(plan["depth"], got, at_aux, at_main):
            if d == AUX:
                assert g == a, (state[:20], d)
            else:
                assert torch.allclose(torch.tensor(g), torch.tensor(m), atol=1e-5), (state[:20], d)
            seen.add(d)
    assert seen == {AUX, EXIT}


def test_tau_above_one_with_temperatures_is_the_fixed_path(models):
    _, fx, ad = _pair(models, 2.0)
    for path in PATHS:
        plan, got = _served(ad, STATES[2], path)
        assert plan["depth"] == [EXIT] * len(QS)
        assert got == _served(fx, STATES[2], path)[1]


def test_temp_cal_refuses_non_finite():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            A.temp_cal(bad)


# ---------------------------------------------------------------------------
# the package
# ---------------------------------------------------------------------------

def _temp_release(d, ada, exits=None, main_buffers=("cal_logT",), main_mode="temp", tau=0.9):
    from safetensors.torch import save_file
    d = _write_release(d, ada, tau=tau, with_policy=False)
    meta = json.loads((d / "meta.json").read_text())
    meta["adaptive"] = {"exits": [AUX, EXIT], "tau": tau}
    (d / "meta.json").write_text(json.dumps(meta))
    save_file({k: torch.tensor(T_MAIN) if k == "cal_logT" else getattr(ada, k).detach().clone()
               for k in main_buffers}, str(d / "calibration.safetensors"))
    cj = {"cal_mode": main_mode}
    cj["exits"] = {str(AUX): {"cal_mode": "temp", "logT": T_AUX}} if exits is None else exits
    (d / "calibration.json").write_text(json.dumps(cj))
    return d


def test_a_temperature_package_loads(stub_lm, tmp_path, models):
    from serve.release import load_release
    _, _, ada, _ = models
    d = _temp_release(tmp_path / "t", ada)
    model, _, _, meta = load_release(d, "cpu")
    assert model.cal_mode == "temp" and float(model.cal_logT) == float(torch.tensor(T_MAIN))
    pol = model.adaptive_policy
    assert pol.exits == [AUX, EXIT] and pol.tau == 0.9
    assert set(pol.cal[AUX]) == {"logT"} and float(pol.cal[AUX]["logT"]) == float(torch.tensor(T_AUX))
    assert meta["serving"]["adaptive"]["tau"] == 0.9


@pytest.mark.parametrize("exits", [
    {},                                                              # an aux exit missing
    {str(AUX): {"cal_mode": "temp", "logT": T_AUX}, str(EXIT): {"cal_mode": "temp", "logT": 0.1}},
    {str(AUX): {"cal_mode": "oof_head", "logT": T_AUX}},             # not a temperature
    {str(AUX): {"cal_mode": "temp"}},                                # no logT
    {str(AUX): {"cal_mode": "temp", "logT": "0.8"}},                 # not a number
    {str(AUX): {"cal_mode": "temp", "logT": float("inf")}},
    {str(AUX): {"cal_mode": "temp", "logT": T_AUX, "extra": 1}},
    [T_AUX],
])
def test_a_malformed_exits_block_is_refused(stub_lm, tmp_path, models, exits):
    from serve.release import load_release
    _, _, ada, _ = models
    d = _temp_release(tmp_path / "bad", ada, exits=exits)
    with pytest.raises(RuntimeError, match="exits"):
        load_release(d, "cpu")
    with pytest.raises(RuntimeError, match="exits"):                 # also when serving fixed
        load_release(d, "cpu", fixed_exit=True)


def test_one_aux_calibration_source_only(stub_lm, tmp_path, models):
    from safetensors.torch import save_file
    from serve.release import ADAPTIVE_CAL_FILE, load_release
    _, _, ada, _ = models
    d = _temp_release(tmp_path / "two", ada)
    save_file({"x": torch.zeros(1)}, str(d / ADAPTIVE_CAL_FILE))
    with pytest.raises(RuntimeError, match="one source only"):
        load_release(d, "cpu")


def test_exit_temperatures_need_a_calibrated_main_head(stub_lm, tmp_path, models):
    from serve.release import load_release
    _, _, ada, _ = models
    from rsijev.calibrate import CAL_BUFFERS
    d = _temp_release(tmp_path / "raw", ada, main_mode="none", main_buffers=CAL_BUFFERS)
    with pytest.raises(RuntimeError, match="main head"):
        load_release(d, "cpu")


def test_a_temperature_main_head_needs_only_its_buffer(stub_lm, tmp_path, models):
    from serve.release import load_release
    _, _, ada, _ = models
    d = _temp_release(tmp_path / "nobuf", ada, main_buffers=())
    with pytest.raises(RuntimeError, match="cal_logT"):
        load_release(d, "cpu")
