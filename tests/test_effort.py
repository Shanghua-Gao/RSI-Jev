"""`effort` (serve/effort.py): a request's depth on a multi-exit release, mapped onto its heads.

low = the shallowest aux exit for every question, medium = the deepest aux exit,
high = the main exit (all layers), auto = the release's cascade on every text request. Each must give,
bitwise, what the existing paths give for the same depth; with no effort anywhere,
serving is unchanged.
"""
import copy
import math

import pytest
import torch

from rsijev import adaptive as A
from rsijev.arch import DecisionModel
from serve import effort as E
from serve import infer
from tests.test_adaptive_exit import (AUX, ENC, EXIT, H, N_LAYERS, QS, STATES, PATHS, _arch, _dc,  # noqa: F401
                                      _write_release, models, stub_lm, tok)

T_MAIN, T_AUX, T_AUX2 = math.log(2.4), math.log(2.3), math.log(1.9)
tok_ref = []


@pytest.fixture(autouse=True)
def _keep_tok(tok):
    tok_ref[:] = [tok]


def _temp(model, logT):
    model.cal_mode = "temp"
    model.cal_logT.fill_(logT)
    return model


def _ada(models, tau, mode="auto"):
    _, _, ada, _ = models
    m = _temp(copy.deepcopy(ada), T_MAIN)
    m.adaptive_policy = A.Policy(m.exit_indices(), {AUX: A.temp_cal(T_AUX)}, tau)
    m.effort_base = m.adaptive_policy
    m.adaptive_mode = mode
    return m


def _run(model, state, path, effort=None, qs=QS):
    cfg = PATHS[path]
    plan = infer.plan_request(tok_ref[0], state, qs, ENC, min_saved_tokens=cfg["min_saved_tokens"],
                              doc_cache=_dc(cfg["doc"]))
    assert plan["path"] == path
    if effort is not None:
        plan["effort"] = effort
    got, _ = infer.score_planned(model, tok_ref[0], plan, max_options=8, device="cpu", batch_size=2)
    return plan, [p.probs for p in got]


# -- parsing ----------------------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [("low", "low"), ("LOW", "low"), (" Medium ", "medium"),
                                      ("high", "high"), ("auto", "auto"), (None, None), ("", None)])
def test_effort_names(raw, want):
    assert E.canonical(raw) == want


@pytest.mark.parametrize("raw", ["fast", "1", "on", "off", "light", "balanced", "full", "max"])
def test_an_unknown_effort_is_refused_with_the_valid_names(raw):
    with pytest.raises(ValueError, match="effort must be one of low, medium, high, auto"):
        E.canonical(raw)


def test_server_default_reads_the_environment(monkeypatch):
    monkeypatch.setenv("RSIJEV_EFFORT", "medium")
    assert E.default_effort() == "medium"
    assert E.default_effort("low") == "low"               # the flag wins
    monkeypatch.delenv("RSIJEV_EFFORT")
    assert E.default_effort() is None


# -- each effort is bitwise its existing path --------------------------------------------

@pytest.mark.parametrize("path", list(PATHS))
def test_low_is_the_tau_minus_one_path_bitwise(models, path):
    ad = _ada(models, 0.9)
    ref = _ada(models, -1.0, mode="on")                    # the verified "tau -1, adaptive on" run
    for state in STATES[:4]:
        p1, got = _run(ad, state, path, "low")
        p2, want = _run(ref, state, path)
        assert got == want and p1["depth"] == p2["depth"] == [AUX] * len(QS)


@pytest.mark.parametrize("path", list(PATHS))
def test_high_is_the_fixed_exit_bitwise(models, path):
    ad = _ada(models, 0.9)
    fx = _ada(models, 0.9)
    fx.adaptive_policy, fx.adaptive_mode = None, "off"     # --adaptive off
    for state in STATES[:4]:
        p1, got = _run(ad, state, path, "high")
        p2, want = _run(fx, state, path)
        assert got == want and p1["depth"] == p2["depth"] == [EXIT] * len(QS)


@pytest.mark.parametrize("path", list(PATHS))
def test_auto_is_adaptive_on_for_every_request(models, path):
    ad = _ada(models, 0.9)                                 # served default: auto mode
    on = _ada(models, 0.9, mode="on")
    for state in STATES[:4]:
        for qs in ((QS, QS[:1]) if path == "plain" else (QS,)):   # one question included
            p1, got = _run(ad, state, path, "auto", qs)
            p2, want = _run(on, state, path, qs=qs)
            assert got == want and p1["depth"] == p2["depth"]


def test_no_effort_leaves_serving_unchanged(models):
    """A single question on the default (auto) mode takes the fixed exit, as before."""
    ad = _ada(models, -1.0)
    plan, _ = _run(ad, STATES[0], "plain", qs=QS[:1])
    assert plan["depth"] == [EXIT] and "effort" not in plan
    plan, _ = _run(ad, STATES[0], "plain")
    assert plan["depth"] == [AUX] * len(QS)                # multi-question: the cascade


# -- medium on a two-aux-exit model: exit 8 of (4, 8, 12) == a model with aux exits (8,) --

@pytest.fixture(scope="module")
def two_aux(models):
    tower = models[0]
    torch.manual_seed(21)
    m = DecisionModel(tower, H, _arch(exit_layer=N_LAYERS, aux_exits=(4, 8))).eval()
    torch.manual_seed(5)
    for p in m.aux_scorers.parameters():
        p.data.add_(0.05 * torch.randn_like(p))
    only8 = DecisionModel(tower, H, _arch(exit_layer=N_LAYERS, aux_exits=(8,))).eval()
    only8.scorer.load_state_dict(m.scorer.state_dict())
    only8.aux_scorers["8"].load_state_dict(m.aux_scorers["8"].state_dict())
    for x in (m, only8):
        _temp(x, T_MAIN)
    m.effort_base = A.Policy([4, 8, N_LAYERS], {4: A.temp_cal(T_AUX), 8: A.temp_cal(T_AUX2)}, 0.7)
    m.adaptive_policy, m.adaptive_mode = m.effort_base, "auto"
    only8.adaptive_policy = A.Policy([8, N_LAYERS], {8: A.temp_cal(T_AUX2)}, -1.0)
    only8.adaptive_mode = "on"
    return m, only8


@pytest.mark.parametrize("path", list(PATHS))
def test_medium_is_the_deepest_aux_exit_model_bitwise(two_aux, path):
    m, only8 = two_aux
    for state in STATES[:4]:
        p1, got = _run(m, state, path, "medium")
        p2, want = _run(only8, state, path)
        assert got == want and p1["depth"] == p2["depth"] == [8] * len(QS)
    _, low = _run(m, STATES[0], path, "low")
    assert _run(m, STATES[0], path, "low")[0]["depth"] == [4] * len(QS) and low


# -- refusals, images, pooling -------------------------------------------------------------

def test_a_single_exit_release_serves_full_only(models):
    fixed = models[1]
    E.check_supported(fixed, "high")
    E.check_supported(fixed, None)
    for e in E.NEEDS_EXITS:
        with pytest.raises(ValueError, match="aux exits"):
            E.check_supported(fixed, e)


def _runner(model):
    from serve.batcher import ModelRunner
    return ModelRunner(model, tok_ref[0], ENC, spec_max_options=8, device="cpu")


def test_the_runner_resolves_validates_and_does_not_pool(models):
    from serve.wire import RequestError
    ad = _ada(models, 0.9)
    r = _runner(ad)
    assert "effort" not in r.plan(STATES[0], QS)           # unchanged when unset
    p = r.plan(STATES[0], QS, effort="medium")
    assert p["effort"] == "medium" and not r.poolable(p)
    assert r.poolable(r.plan(STATES[0], QS, effort="high")) == r.poolable(r.plan(STATES[0], QS))
    with pytest.raises(RequestError, match="effort must be one of"):
        r.plan(STATES[0], QS, effort="fast")
    with pytest.raises(RequestError, match="aux exits"):
        _runner(models[1]).plan(STATES[0], QS, effort="low")
    rd = _runner(ad)
    rd.default_effort = "low"
    assert rd.plan(STATES[0], QS)["effort"] == "low"
    assert rd.plan(STATES[0], QS, effort="high")["effort"] == "high"    # the request wins


def test_image_requests_run_at_full_depth(models, monkeypatch):
    ad = _ada(models, 0.9)
    r = _runner(ad)
    monkeypatch.setattr(r, "_plan_images", lambda s, q, i: {"path": "image", "encoded": []})
    assert r.plan(STATES[0], QS, images=["x"], effort="low")["effort"] == "high"


def test_effort_survives_adaptive_off_at_load(stub_lm, tmp_path, models):
    from serve.release import load_release
    _, _, ada, _ = models
    d = _write_release(tmp_path / "ada", ada, tau=0.9)
    m, _, _, meta = load_release(d, "cpu", fixed_exit=True)
    assert m.adaptive_policy is None and m.effort_base is not None
    assert m.effort_base.exits == [AUX, EXIT] and m.effort_base.tau == 0.9
    m2, _, _, _ = load_release(d, "cpu")
    assert m2.effort_base is m2.adaptive_policy


# -- the HTTP route --------------------------------------------------------------------------

def _client(calls):
    from fastapi.testclient import TestClient
    from serve.app import create_app

    def scorer(state, questions, images=None, **kw):
        calls.append(kw.get("effort"))
        extras = {"effort": kw["effort"]} if kw.get("effort") else {}
        return [[0.3, 0.7] for _ in questions], 5, extras
    return TestClient(create_app(scorer, served_model_name="m"))


def _body(**kw):
    return {"model": "m", "state": "hi",
            "questions": {"a": {"type": "noul", "instructions": "x?",
                                "criteria": {"true": "yes", "false": "no"}}}, **kw}


def test_the_route_passes_effort_and_reports_it():
    calls = []
    c = _client(calls)
    r = c.post("/v1/systemone", json=_body(effort="Medium"))
    assert r.status_code == 200 and r.json()["usage"]["effort"] == "medium" and calls == ["medium"]
    r = c.post("/v1/systemone", json=_body())
    assert r.status_code == 200 and "effort" not in r.json()["usage"] and calls[-1] is None
    for bad in ("turbo", "full", "light"):
        r = c.post("/v1/systemone", json=_body(effort=bad))
        assert r.status_code == 422 and "effort must be one of low, medium, high, auto" in r.text


# -- confidence_threshold and usage.confidence ----------------------------------------------

def _with_threshold(model, state, path, effort, thr, qs=QS):
    from serve.effort import resolve
    eff, t = resolve(model, effort, thr, False)
    cfg = PATHS[path]
    plan = infer.plan_request(tok_ref[0], state, qs, ENC, min_saved_tokens=cfg["min_saved_tokens"],
                              doc_cache=_dc(cfg["doc"]))
    if eff is not None:
        plan["effort"] = eff
    if t is not None:
        plan["threshold"] = t
    got, _ = infer.score_planned(model, tok_ref[0], plan, max_options=8, device="cpu", batch_size=2)
    return plan, [p.probs for p in got]


@pytest.mark.parametrize("path", list(PATHS))
def test_the_release_tau_as_threshold_is_auto_bitwise(models, path):
    ad = _ada(models, 0.9)
    for state in STATES[:4]:
        p1, got = _with_threshold(ad, state, path, "auto", 0.9)
        p2, want = _run(ad, state, path, "auto")
        assert got == want and p1["depth"] == p2["depth"]


@pytest.mark.parametrize("path", list(PATHS))
def test_threshold_one_is_high(models, path):
    ad = _ada(models, 0.9)
    for state in STATES[:4]:
        for eff in ("auto", None):
            p1, got = _with_threshold(ad, state, path, eff, 1.0)
            p2, want = _run(ad, state, path, "high")
            assert got == want and p1["depth"] == [EXIT] * len(QS) and p1["effort"] == "high"


@pytest.mark.parametrize("path", list(PATHS))
def test_a_tiny_threshold_is_low(models, path):
    ad = _ada(models, 0.9)
    for state in STATES[:4]:
        p1, got = _with_threshold(ad, state, path, "auto", 1e-6)
        p2, want = _run(ad, state, path, "low")
        assert got == want and p1["depth"] == p2["depth"] == [AUX] * len(QS)


def test_threshold_is_validated_and_scoped(models):
    from serve.effort import resolve, threshold
    for bad in (0, -0.1, 1.5, "x", float("nan")):
        with pytest.raises(ValueError):
            threshold(bad)
    assert threshold(1) == 1.0 and threshold(None) is None
    ad = _ada(models, 0.9)
    for eff in ("low", "medium", "high"):
        with pytest.raises(ValueError, match="applies to effort 'auto'"):
            resolve(ad, eff, 0.5, False)
    with pytest.raises(ValueError, match="multi-exit"):
        resolve(models[1], None, 0.5, False)
    assert resolve(ad, None, None, True) == (None, None)            # images, nothing asked
    assert resolve(ad, "low", None, True) == ("high", None)       # images run at full depth
    assert resolve(ad, "auto", 0.5, True) == ("high", None)


@pytest.mark.parametrize("effort", [None, "low", "medium", "high", "auto"])
def test_confidence_is_the_calibrated_top1_at_the_exit_used(models, effort):
    ad = _ada(models, 0.9)
    sh = _temp(copy.deepcopy(models[3]), T_AUX)                      # the single-exit model at AUX
    fx = _ada(models, 0.9)
    fx.adaptive_policy, fx.adaptive_mode = None, "off"
    for state in STATES[:4]:
        plan, got = _run(ad, state, "plain", effort)
        conf = plan["confidence"]
        assert conf == [float(max(p)) for p in got]
        _, at_aux = _run(sh, state, "plain")
        _, at_main = _run(fx, state, "plain")
        for d, c, a, m in zip(plan["depth"], conf, at_aux, at_main):
            if d == AUX:
                assert c == float(max(a))                           # softmax(z / e^logT).max at AUX
            elif effort in ("high", None):
                assert c == float(max(m))


def test_confidence_on_the_image_and_pooled_paths(models):
    ad = _ada(models, 0.9)
    from rsijev.contract import Prediction
    plan = {"path": "image", "encoded": []}
    infer.record_confidence(ad, plan, [Prediction((0.2, 0.8)), Prediction((0.6, 0.3, 0.1))])
    assert plan["confidence"] == [0.8, 0.6]
    infer.record_confidence(models[1], plan2 := {}, [Prediction((0.2, 0.8))])
    assert "confidence" not in plan2                                 # single-exit: shape unchanged
    r = _runner(ad)
    plans = [r.plan(s, QS[:1]) for s in STATES[:3]]
    out = r._pooled(plans)
    for p, (probs, _) in zip(plans, out):
        assert p["confidence"] == [float(max(x)) for x in probs] and p["depth"] == [EXIT]


def test_the_route_validates_the_threshold():
    calls = []
    c = _client(calls)
    assert c.post("/v1/systemone", json=_body(confidence_threshold=1.2)).status_code == 422
    assert c.post("/v1/systemone", json=_body(confidence_threshold=0)).status_code == 422


# -- per-exit thresholds for effort auto (owner option A) -----------------------------------

@pytest.mark.parametrize("path", list(PATHS))
def test_an_object_threshold_equal_to_the_number_is_bitwise_the_number(models, path):
    ad = _ada(models, 0.9)
    for state in STATES[:4]:
        p1, got = _with_threshold(ad, state, path, "auto", {str(AUX): 0.7})
        p2, want = _with_threshold(ad, state, path, "auto", 0.7)
        assert got == want and p1["depth"] == p2["depth"]


@pytest.mark.parametrize("path", list(PATHS))
def test_one_at_an_exit_never_stops_there(models, two_aux, path):
    ad = _ada(models, 0.9)
    for state in STATES[:4]:
        p1, got = _with_threshold(ad, state, path, "auto", {str(AUX): 1.0})
        p2, want = _run(ad, state, path, "high")
        assert got == want and p1["depth"] == [EXIT] * len(QS)
    m, _ = two_aux
    seen = set()
    for state in STATES:
        p, _ = _with_threshold(m, state, path, "auto", {"4": 1.0, "8": 1e-6})
        assert 4 not in p["depth"] and set(p["depth"]) <= {8}
        seen |= set(p["depth"])
    assert seen == {8}


def test_a_partial_object_fills_from_the_default(models, two_aux):
    m, _ = two_aux                                                    # tau 0.7 at 4 and 8
    for state in STATES[:4]:
        p1, got = _with_threshold(m, state, "plain", "auto", {"8": 0.7})
        p2, want = _run(m, state, "plain", "auto")
        assert got == want and p1["depth"] == p2["depth"]


@pytest.mark.parametrize("bad", [{"5": 0.5}, {"x": 0.5}, {str(AUX): 1.5}, {str(AUX): 0}, {}, True])
def test_bad_threshold_objects_are_refused(models, bad):
    from serve.effort import resolve, threshold
    ad = _ada(models, 0.9)
    with pytest.raises(ValueError):
        resolve(ad, "auto", threshold(bad), False)


def test_an_object_needs_effort_auto(models):
    from serve.effort import resolve, threshold
    ad = _ada(models, 0.9)
    with pytest.raises(ValueError, match="needs effort 'auto'"):
        resolve(ad, None, threshold({str(AUX): 0.5}), False)


def test_auto_uses_the_release_auto_thresholds_and_unset_ignores_them(models):
    ad = _ada(models, 0.9)
    plain = _ada(models, 0.9)
    ad.effort_auto_taus = {AUX: 0.6}
    for state in STATES[:4]:
        p1, got = _run(ad, state, "plain", "auto")
        p2, want = _with_threshold(plain, state, "plain", "auto", 0.6)
        assert got == want and p1["depth"] == p2["depth"]
        p3, unset = _run(ad, state, "plain")                         # default path: the single tau
        p4, before = _run(plain, state, "plain")
        assert unset == before and p3["depth"] == p4["depth"]


def test_the_release_reads_auto_thresholds(stub_lm, tmp_path, models):
    import json
    from serve.release import load_release
    _, _, ada, _ = models
    d = _write_release(tmp_path / "ada", ada, tau=0.9)
    meta = json.loads((d / "meta.json").read_text())
    meta["adaptive"]["auto_thresholds"] = {str(AUX): 0.75}
    (d / "meta.json").write_text(json.dumps(meta))
    m, _, _, _ = load_release(d, "cpu")
    assert m.effort_auto_taus == {AUX: 0.75} and m.adaptive_policy.tau == 0.9
    meta["adaptive"]["auto_thresholds"] = {"3": 0.5}
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="auto_thresholds"):
        load_release(d, "cpu")


def test_the_route_takes_a_threshold_object():
    calls = []
    c = _client(calls)
    assert c.post("/v1/systemone", json=_body(effort="auto", confidence_threshold={"16": 0.7})).status_code == 200
    assert c.post("/v1/systemone", json=_body(confidence_threshold=0.5)).status_code == 200
    assert c.post("/v1/systemone", json=_body(confidence_threshold={"16": 2})).status_code == 422
