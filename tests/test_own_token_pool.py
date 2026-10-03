"""option_pool_own_tokens: an option's span leaves out its leading "\\n- key:" tokens.

In a causal tower those tokens follow the previous option and carry it; pooling them
made the 4B exit line pick the option after the right one on long numbered lists.
Both encoding paths (encode_question and the read-once plan) must agree.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsijev.contract import Question                       # noqa: E402
from rsijev.encode import EncodeConfig, encode_question     # noqa: E402

KEYS = tuple(f"option_{i}" for i in range(12))
Q = Question(key="q", mode="choice", instructions="Classify the request.",
             options=KEYS, criteria={k: f"intent number {i}" for i, k in enumerate(KEYS)})
Q2 = dataclasses.replace(Q, key="q2", instructions="Classify it again.")


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")


def test_spans_skip_the_key_prefix(tok):
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
    whole = encode_question(tok, "user: book me a table", Q, enc)
    own = encode_question(tok, "user: book me a table", Q,
                          dataclasses.replace(enc, option_pool_own_tokens=True))
    assert own["input_ids"] == whole["input_ids"]
    for (a, b), (a2, b2), key in zip(whole["option_span"], own["option_span"], KEYS):
        assert b2 == b and a < a2 < b
        text = tok.decode(own["input_ids"][a2:b2])
        assert key not in text and "intent number" in text


def test_read_once_path_matches(tok):
    from serve.infer import plan_request
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical",
                       option_pool_own_tokens=True)
    state = "user: book me a table for two tonight " * 200
    plan = plan_request(tok, state, [Q, Q2], enc)
    direct = [encode_question(tok, state, q, enc) for q in (Q, Q2)]
    assert plan["path"] in ("cached", "doc")          # the state is long enough to share
    for r, d in zip(plan["encoded"], direct):
        assert r["input_ids"] == d["input_ids"]
        assert r["option_span"] == d["option_span"]
