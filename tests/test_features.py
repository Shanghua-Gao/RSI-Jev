"""Features (2026-09-29): structured criteria objects and calibrated abstention.

No GPU, no weights; the encoder checks use a tiny character tokenizer so they run
anywhere. The byte-identity of string criteria against main's encode.py on real
corpora is tests/test_criteria_render.py.

    python -m pytest tests/test_features.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.abstain import (ABSTAIN_KEY, ABSTAIN_TEXT, conformal_tau,  # noqa: E402
                            split_abstain, with_abstain)
from rsijev.contract import Question                                   # noqa: E402
from rsijev.encode import EncodeConfig, criterion_text, encode_question, render  # noqa: E402
from serve.app import create_app                                       # noqa: E402

MODEL = "rsi-jev-test"
calls: list = []


def fake_scorer(state, questions):
    calls.append((state, questions))
    out = []
    for q in questions:
        raw = [i + 1 for i in range(len(q.options))]     # argmax = last canonical option
        out.append([v / sum(raw) for v in raw])
    return out, 7


@pytest.fixture
def client():
    calls.clear()
    return TestClient(create_app(fake_scorer, served_model_name=MODEL, abstain_tau=0.3))


def body(**qs):
    return {"model": "jev-latest", "state": "An email.", "questions": qs}


SPAM = {"what": "Unsolicited bulk email", "includes": ["ads from strangers", "lottery scams"],
        "excludes": ["newsletters the recipient signed up for"]}
OBJ_CHOICE = {"type": "choice", "instructions": "Is `email` legitimate or spam?",
              "criteria": {"legitimate": {"what": "Email the recipient expects",
                                          "includes": ["receipts"]},
                           "spam": SPAM}}


class CharTok:
    """Deterministic stand-in tokenizer: one id per character."""
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(ch) for ch in text]}


# ---------------------------------------------------------------- structured criteria
def test_renderer_deterministic_and_strings_unchanged():
    assert criterion_text("plain") == "plain"
    t = criterion_text(SPAM)
    assert t == ("Unsolicited bulk email. Includes: ads from strangers; lottery scams. "
                 "Excludes: newsletters the recipient signed up for.")
    # key order in the object does not matter; the same object always renders the same
    shuffled = {"excludes": SPAM["excludes"], "includes": SPAM["includes"], "what": SPAM["what"]}
    assert criterion_text(shuffled) == t
    assert criterion_text(json.loads(json.dumps(SPAM))) == t


def test_object_criteria_accepted_and_rendered(client):
    r = client.post("/v1/systemone", json=body(category=OBJ_CHOICE))
    assert r.status_code == 200, r.text
    q = calls[0][1][0]
    assert q.criteria["spam"] == SPAM                        # passed through as an object
    _, parts = render("An email.", q, EncodeConfig())
    assert parts[1] == "- spam: " + criterion_text(SPAM)
    a = r.json()["answers"]["category"]
    assert set(a) == {"type", "choice", "probabilities", "confidence"}


def test_object_equals_its_rendered_string_encoding(client):
    """An object and the string it renders to give the identical token sequence."""
    client.post("/v1/systemone", json=body(category=OBJ_CHOICE))
    q_obj = calls[0][1][0]
    as_str = {**OBJ_CHOICE, "criteria": {k: criterion_text(v)
                                         for k, v in OBJ_CHOICE["criteria"].items()}}
    client.post("/v1/systemone", json=body(category=as_str))
    q_str = calls[1][1][0]
    tok = CharTok()
    assert (encode_question(tok, "An email.", q_obj, EncodeConfig())
            == encode_question(tok, "An email.", q_str, EncodeConfig()))


def test_mixed_null_string_object(client):
    crit = {"a": None, "b": "described", "c": {"what": "w"}}
    r = client.post("/v1/systemone", json=body(q={**OBJ_CHOICE, "criteria": crit}))
    assert r.status_code == 200, r.text
    assert calls[0][1][0].criteria == {"a": "", "b": "described", "c": {"what": "w"}}


@pytest.mark.parametrize("bad", [
    {"what": "w", "colour": "red"},          # unknown field
    {},                                      # empty object
    {"what": 3},                             # wrong type
    {"includes": [1, 2]},                    # list of non-strings
])
def test_bad_objects_rejected(client, bad):
    crit = {"a": "x", "b": bad}
    r = client.post("/v1/systemone", json=body(q={**OBJ_CHOICE, "criteria": crit}))
    assert r.status_code == 422
    assert not calls


# ---------------------------------------------------------------- abstention
def test_default_encoding_unchanged():
    q = Question(key="q", mode="choice", instructions="Which?", options=("a", "b"),
                 criteria={"a": "A", "b": "B"})
    tok = CharTok()
    base = encode_question(tok, "s", q, EncodeConfig())
    qa = with_abstain(q)
    assert qa.options == ("a", "b", ABSTAIN_KEY)
    enc = encode_question(tok, "s", qa, EncodeConfig())
    # the real options' blocks are a prefix-identical part of the abstain encoding
    assert enc["input_ids"][:base["option_index"][-1] + 1] == \
        base["input_ids"][:base["option_index"][-1] + 1]
    _, parts = render("s", qa, EncodeConfig())
    assert parts[2] == f"- {ABSTAIN_KEY}: {ABSTAIN_TEXT}"


def test_abstain_answer(client):
    q = {**OBJ_CHOICE, "allow_abstain": True}
    r = client.post("/v1/systemone", json=body(category=q, plain=OBJ_CHOICE))
    assert r.status_code == 200, r.text
    sent = {x.key: x for x in calls[0][1]}
    assert sent["category"].options[-1] == ABSTAIN_KEY
    assert ABSTAIN_KEY not in sent["plain"].options
    a = r.json()["answers"]
    # fake scorer: p = (1, 2, 3) / 6, so unknown = .5 and the real options are (1/3, 2/3)
    assert a["category"]["unknown_probability"] == pytest.approx(0.5)
    assert a["category"]["abstained"] is True                  # tau = .3
    assert a["category"]["choice"] == "spam"
    assert set(a["category"]["probabilities"]) == {"legitimate", "spam"}
    assert sum(a["category"]["probabilities"].values()) == pytest.approx(1.0)
    assert a["category"]["probabilities"]["spam"] == pytest.approx(2 / 3)
    assert "unknown_probability" not in a["plain"] and "abstained" not in a["plain"]


def test_abstain_type_and_reserved_key_errors(client):
    for q in ({"type": "noul", "instructions": "x", "allow_abstain": True},
              {"type": "score", "instructions": "x", "criteria": ["lo", "hi"], "allow_abstain": True},
              {**OBJ_CHOICE, "criteria": {"a": "x", ABSTAIN_KEY: "y"}, "allow_abstain": True},
              {**OBJ_CHOICE, "allow_abstain": "yes"}):
        r = client.post("/v1/systemone", json=body(q=q))
        assert r.status_code == 422, (q, r.text)
    assert not calls
    # the reserved key is an ordinary option when abstention is off
    r = client.post("/v1/systemone", json=body(q={**OBJ_CHOICE, "criteria": {"a": "x", ABSTAIN_KEY: "y"}}))
    assert r.status_code == 200
    assert "unknown_probability" not in r.json()["answers"]["q"]


def test_bundling_invariance_with_abstain(client):
    """A question's answer does not depend on which other questions share the request."""
    q = {**OBJ_CHOICE, "allow_abstain": True}
    one = client.post("/v1/systemone", json=body(a=q)).json()["answers"]["a"]
    many = client.post("/v1/systemone", json=body(
        a=q, b=OBJ_CHOICE, c={"type": "noul", "instructions": "x"})).json()["answers"]["a"]
    assert one == many


def test_limits_report_features(client):
    lim = client.get("/v1/limits").json()
    assert lim["structured_criteria"] == ["what", "includes", "excludes"]
    assert lim["abstain"]["question_types"] == ["choice"]
    assert lim["abstain"]["threshold"] == 0.3


def test_split_and_conformal():
    real, u = split_abstain([0.2, 0.2, 0.6])
    assert u == pytest.approx(0.6) and real == pytest.approx([0.5, 0.5])
    scores = [i / 100 for i in range(100)]           # answerable dev p_unknown
    tau = conformal_tau(scores, alpha=0.05)
    assert sum(s >= tau for s in scores) / len(scores) <= 0.05
    assert conformal_tau([0.1, 0.2], alpha=0.05) == float("inf")   # too few to certify


def test_abstain_uses_one_option_slot(client):
    from serve.wire import MAX_ANSWERS
    crit = {f"k{i}": "d" for i in range(MAX_ANSWERS)}
    r = client.post("/v1/systemone", json=body(q={**OBJ_CHOICE, "criteria": crit, "allow_abstain": True}))
    assert r.status_code == 422 and not calls
    crit = {f"k{i}": "d" for i in range(MAX_ANSWERS - 1)}
    r = client.post("/v1/systemone", json=body(q={**OBJ_CHOICE, "criteria": crit, "allow_abstain": True}))
    assert r.status_code == 200 and len(calls[0][1][0].options) == MAX_ANSWERS
