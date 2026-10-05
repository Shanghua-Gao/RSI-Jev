"""`effort` (serve/effort.py): a request's depth on a multi-exit release, mapped onto its heads.

light = the shallowest aux exit for every question, balanced = the deepest aux exit,
full = the main exit, auto = the release's cascade on every text request. Each must give,
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

@pytest.mark.parametrize("raw,want", [("light", "light"), ("LOW", "light"), ("medium", "balanced"),
                                      ("balanced", "balanced"), ("high", "full"), ("max", "full"),
                                      (" Full ", "full"), ("auto", "auto"), (None, None), ("", None)])
def test_effort_names_and_aliases(raw, want):
    assert E.canonical(raw) == want


@pytest.mark.parametrize("raw", ["fast", "1", "on", "off"])
def test_an_unknown_effort_is_refused(raw):
    with pytest.raises(ValueError, match="effort must be one of"):
        E.canonical(raw)


def test_server_default_reads_the_environment(monkeypatch):
    monkeypatch.setenv("RSIJEV_EFFORT", "medium")
    assert E.default_effort() == "balanced"
    assert E.default_effort("light") == "light"           # the flag wins
    monkeypatch.delenv("RSIJEV_EFFORT")
    assert E.default_effort() is None


# -- each effort is bitwise its existing path --------------------------------------------

@pytest.mark.parametrize("path", list(PATHS))
def test_light_is_the_tau_minus_one_path_bitwise(models, path):
    ad = _ada(models, 0.9)
    ref = _ada(models, -1.0, mode="on")                    # the verified "tau -1, adaptive on" run
    for state in STATES[:4]:
        p1, got = _run(ad, state, path, "light")
        p2, want = _run(ref, state, path)
        assert got == want and p1["depth"] == p2["depth"] == [AUX] * len(QS)


@pytest.mark.parametrize("path", list(PATHS))
def test_full_is_the_fixed_exit_bitwise(models, path):
    ad = _ada(models, 0.9)
    fx = _ada(models, 0.9)
    fx.adaptive_policy, fx.adaptive_mode = None, "off"     # --adaptive off
    for state in STATES[:4]:
        p1, got = _run(ad, state, path, "full")
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


# -- balanced on a two-aux-exit model: exit 8 of (4, 8, 12) == a model with aux exits (8,) --

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
def test_balanced_is_the_deepest_aux_exit_model_bitwise(two_aux, path):
    m, only8 = two_aux
    for state in STATES[:4]:
        p1, got = _run(m, state, path, "balanced")
        p2, want = _run(only8, state, path)
        assert got == want and p1["depth"] == p2["depth"] == [8] * len(QS)
    _, light = _run(m, STATES[0], path, "light")
    assert _run(m, STATES[0], path, "light")[0]["depth"] == [4] * len(QS) and light


# -- refusals, images, pooling -------------------------------------------------------------

def test_a_single_exit_release_serves_full_only(models):
    fixed = models[1]
    E.check_supported(fixed, "full")
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
    assert p["effort"] == "balanced" and not r.poolable(p)
    assert r.poolable(r.plan(STATES[0], QS, effort="full")) == r.poolable(r.plan(STATES[0], QS))
    with pytest.raises(RequestError, match="effort must be one of"):
        r.plan(STATES[0], QS, effort="fast")
    with pytest.raises(RequestError, match="aux exits"):
        _runner(models[1]).plan(STATES[0], QS, effort="light")
    rd = _runner(ad)
    rd.default_effort = "light"
    assert rd.plan(STATES[0], QS)["effort"] == "light"
    assert rd.plan(STATES[0], QS, effort="high")["effort"] == "full"    # the request wins


def test_image_requests_run_at_full_depth(models, monkeypatch):
    ad = _ada(models, 0.9)
    r = _runner(ad)
    monkeypatch.setattr(r, "_plan_images", lambda s, q, i: {"path": "image", "encoded": []})
    assert r.plan(STATES[0], QS, images=["x"], effort="light")["effort"] == "full"


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
    assert r.status_code == 200 and r.json()["usage"]["effort"] == "balanced" and calls == ["balanced"]
    r = c.post("/v1/systemone", json=_body())
    assert r.status_code == 200 and "effort" not in r.json()["usage"] and calls[-1] is None
    r = c.post("/v1/systemone", json=_body(effort="turbo"))
    assert r.status_code == 422 and "effort must be one of" in r.text
