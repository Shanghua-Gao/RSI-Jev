"""Serving never cuts an input: past the cap a question is refused with a 422.

Training and the benchmark scripts keep cutting the state from the left at 2,048
tokens (EncodeConfig.truncate defaults to "left"), so published numbers reproduce.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsijev.contract import Question                       # noqa: E402
from rsijev.encode import EncodeConfig, InputTooLong, encode_question  # noqa: E402
from serve.app import create_app                           # noqa: E402


class CharTokenizer:
    """One token per character: enough to exercise the length logic."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 1000 for c in text]}


Q = Question(key="q", mode="choice", instructions="Which one?",
             options=("a", "b"), criteria={"a": "first", "b": "second"})


def test_serving_refuses_instead_of_cutting():
    cfg = EncodeConfig(max_length=200, truncate="none")
    assert len(encode_question(CharTokenizer(), "x" * 50, Q, cfg)["input_ids"]) <= 200
    with pytest.raises(InputTooLong, match="maximum context length of 200"):
        encode_question(CharTokenizer(), "x" * 500, Q, cfg)


def test_training_still_cuts_the_state_from_the_left():
    row = encode_question(CharTokenizer(), "x" * 500, Q, EncodeConfig(max_length=200))
    assert len(row["input_ids"]) == 200


def test_too_long_is_a_422_the_benchmark_kits_recognise():
    def scorer(state, questions):
        raise InputTooLong("question q is 40000 tokens, over the maximum context length of 32768")

    c = TestClient(create_app(scorer, served_model_name="m"))
    r = c.post("/v1/systemone", json={"model": "jev-latest", "state": "s",
                                      "questions": {"q": {"type": "noul", "instructions": "i"}}})
    assert r.status_code == 422
    assert "maximum context length" in r.json()["error"]["message"]
