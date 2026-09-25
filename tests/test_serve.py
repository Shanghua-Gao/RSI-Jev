"""Wire-contract tests for the Jev-compatible API.

These assert the same things the reference's own suite asserts
(`reference/openjev-sglang/tests/test_api.py`), against a fake scorer, so the
contract is pinned without a GPU.

    python -m pytest tests/test_serve.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve.app import create_app          # noqa: E402
from serve.wire import confidence         # noqa: E402

MODEL = "rsi-jev-v1.0-qwen3.5-2b"
calls: list[tuple[str, list]] = []


def fake_scorer(state, questions):
    """Deterministic, distinguishable probabilities; records that it was called."""
    calls.append((state, questions))
    probs = []
    for q in questions:
        n = len(q.options)
        raw = [i + 1 for i in range(n)]          # increasing, so argmax is the LAST option
        total = sum(raw)
        probs.append([v / total for v in raw])
    return probs, 123


@pytest.fixture
def client():
    calls.clear()
    return TestClient(create_app(fake_scorer, served_model_name=MODEL))


def body(**questions):
    return {"model": "jev-latest", "state": "I was charged twice.", "questions": questions}


NOUL = {"type": "noul", "instructions": "Does the user request a refund?"}
CHOICE = {"type": "choice", "instructions": "Which department?",
          "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}}
SCORE = {"type": "score", "instructions": "How urgent?",
         "criteria": ["Routine", "Urgent", "Emergency"]}


def test_answer_shapes(client):
    r = client.post("/v1/systemone", json=body(refund=NOUL, department=CHOICE, urgency=SCORE))
    assert r.status_code == 200, r.text
    data = r.json()
    assert set(data) == {"model", "answers", "usage"}
    assert data["model"] == "jev-latest"                      # the REQUESTED name is echoed
    assert list(data["answers"]) == ["refund", "department", "urgency"]   # key order preserved

    noul = data["answers"]["refund"]
    assert noul == {"type": "noul", "noul": pytest.approx(2 / 3)}   # options ("false","true")

    choice = data["answers"]["department"]
    assert set(choice) == {"type", "choice", "probabilities", "confidence"}
    assert choice["choice"] == "technical"                    # last option has the mass
    assert set(choice["probabilities"]) == {"billing", "technical"}
    assert sum(choice["probabilities"].values()) == pytest.approx(1.0)

    score = data["answers"]["urgency"]
    assert set(score) == {"type", "score", "legend", "probabilities", "confidence"}
    assert score["legend"] == {"0": "Routine", "1": "Urgent", "2": "Emergency"}
    assert set(score["probabilities"]) == {"0", "1", "2"}
    p = [score["probabilities"][k] for k in ("0", "1", "2")]
    assert score["score"] == pytest.approx(sum(i * v for i, v in enumerate(p)))
    assert 0 <= score["score"] <= 2

    assert data["usage"] == {"input_tokens": 123, "output_tokens": 3}
    assert len(r.headers["x-typesafe-request-id"]) == 32
    assert r.headers["x-rsijev-model"] == MODEL


def test_questions_are_independent(client):
    """One state, one scorer call, every question carried in that call."""
    client.post("/v1/systemone", json=body(a=NOUL, b=CHOICE))
    assert len(calls) == 1
    state, questions = calls[0]
    assert state == "I was charged twice."
    assert [q.key for q in questions] == ["a", "b"]


@pytest.mark.parametrize("count,expected", [(1, 422), (2, 200), (64, 200), (65, 422)])
def test_choice_option_limits(client, count, expected):
    criteria = {f"k{i}": f"description {i}" for i in range(count)}
    r = client.post("/v1/systemone", json=body(q={**CHOICE, "criteria": criteria}))
    assert r.status_code == expected
    if expected == 422:
        assert not calls, "a rejected request must not reach the model"
    else:
        assert len(r.json()["answers"]["q"]["probabilities"]) == count


@pytest.mark.parametrize("payload", [
    {"model": "jev-latest", "questions": {"q": NOUL}},                      # no state
    {"model": "other", "state": "s", "questions": {"q": NOUL}},             # unknown model
    {"model": 1, "state": "s", "questions": {"q": NOUL}},                   # not coerced
    {"model": "jev-latest", "state": "s", "questions": {}},                 # empty
    {"model": "jev-latest", "state": "s", "questions": {"q": {"type": "nope",
                                                              "instructions": "i"}}},
    {"model": "jev-latest", "state": "s", "questions": {"q": {**SCORE, "criteria": ["only"]}}},
    {"model": "jev-latest", "state": "s", "questions": {f"q{i}": NOUL for i in range(65)}},
    {"model": "jev-latest", "state": "s", "questions": {"q": NOUL}, "extra": 1},  # unknown field
])
def test_invalid_requests_never_reach_the_model(client, payload):
    r = client.post("/v1/systemone", json=payload)
    assert r.status_code == 422, r.text
    assert not calls


def test_unknown_model_uses_the_error_envelope(client):
    r = client.post("/v1/systemone",
                    json={"model": "other", "state": "s", "questions": {"q": NOUL}})
    assert r.json() == {"error": {"message": "Unknown model: other"}}


def test_served_name_is_accepted_too(client):
    r = client.post("/v1/systemone",
                    json={"model": MODEL, "state": "s", "questions": {"q": NOUL}})
    assert r.status_code == 200
    assert r.json()["model"] == MODEL


def test_noul_criteria_use_true_false_keys(client):
    r = client.post("/v1/systemone", json=body(
        q={**NOUL, "criteria": {"true": "Yes it does", "false": "No it does not"}}))
    assert r.status_code == 200
    _, questions = calls[0]
    assert questions[0].options == ("false", "true")
    assert questions[0].criteria == {"false": "No it does not", "true": "Yes it does"}


def test_noul_criteria_default_to_yes_no(client):
    client.post("/v1/systemone", json=body(q=NOUL))
    assert calls[0][1][0].criteria == {"false": "No", "true": "Yes"}


def test_choice_accepts_null_and_empty_descriptions(client):
    r = client.post("/v1/systemone", json=body(
        q={**CHOICE, "criteria": {"a": None, "b": "", "c": "described"}}))
    assert r.status_code == 200
    assert set(r.json()["answers"]["q"]["probabilities"]) == {"a", "b", "c"}


def test_structured_state_is_serialized_as_compact_json(client):
    client.post("/v1/systemone", json={"model": "jev-latest", "questions": {"q": NOUL},
                                       "state": {"ticket": 12, "body": "charged twice"}})
    assert calls[0][0] == '{"ticket":12,"body":"charged twice"}'


def test_chat_state_rejects_non_text_content(client):
    r = client.post("/v1/systemone", json={
        "model": "jev-latest", "questions": {"q": NOUL},
        "state": [{"role": "user", "content": [{"type": "image_url", "url": "x"}]}]})
    assert r.status_code == 422
    assert r.json()["error"]["message"] == "Jev state supports text content only"
    assert not calls


def test_chat_state_rejects_unknown_role(client):
    r = client.post("/v1/systemone", json={
        "model": "jev-latest", "questions": {"q": NOUL},
        "state": [{"role": "robot", "content": "hi"}]})
    assert r.status_code == 422
    assert r.json()["error"]["message"] == "Chat state has an unsupported message role"


def test_confidence_is_the_published_peak_statistic():
    """(K * p_max - 1) / (K - 1), which TypeSafe documents for three options as
    "(3 x largest probability - 1) / 2".

    The endpoints below are NOT enough on their own: 1 - H(p)/ln(K) satisfies all
    three of them too, which is exactly how this shipped wrong. The interior
    points are the test -- they are where the two formulas disagree.
    """
    assert confidence([0.5, 0.5]) == pytest.approx(0.0)              # uniform
    assert confidence([1.0, 0.0]) == pytest.approx(1.0)              # one-hot
    assert confidence([1 / 3] * 3) == pytest.approx(0.0)             # uniform, K=3

    # The documented worked example: three options, peak 0.6 -> (3*0.6-1)/2 = 0.4
    assert confidence([0.6, 0.2, 0.2]) == pytest.approx(0.4)
    # Two options: 2p - 1.
    assert confidence([0.9, 0.1]) == pytest.approx(0.8)
    # Four options, peak 0.5. Entropy would say 0.104; the peak statistic says 1/3.
    assert confidence([0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3]) == pytest.approx(1 / 3)

    # Only the peak matters, not how the remainder is spread -- the property that
    # distinguishes this from any entropy-based statistic.
    assert confidence([0.5, 0.5, 0.0, 0.0]) == pytest.approx(confidence([0.5, 0.25, 0.25, 0.0]))


def test_noul_answers_carry_no_confidence():
    """The reference does not put a confidence on a noul answer, and neither do we:
    with two outcomes the probability already is the confidence."""
    from serve.wire import to_answer, to_question
    q = to_question("refund", {"type": "noul", "instructions": "Refund?"})
    assert set(to_answer(q, [0.3, 0.7])) == {"type", "noul"}


def test_models_limits_and_health(client):
    models = client.get("/v1/models").json()
    assert models["data"][0]["id"] == "jev-latest"
    assert [m["name"] for m in models["models"]] == ["jev-latest", MODEL]

    lim = client.get("/v1/limits").json()
    assert lim["max_questions"] == 64
    assert lim["max_answers_per_question"] == 64
    # This deployment shows option keys to the model; the reference hides them.
    assert lim["option_keys_visible_to_model"] is True

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/health/live").json() == {"status": "ok"}


def test_auth_protects_inference_but_not_health():
    app = create_app(fake_scorer, served_model_name=MODEL, api_key="secret")
    c = TestClient(app)
    assert c.post("/v1/systemone", json=body(q=NOUL)).status_code == 401
    assert c.get("/health").status_code == 200
    ok = c.post("/v1/systemone", json=body(q=NOUL), headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
