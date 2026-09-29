"""Scoring and the `decide` entry point, on MLX.

Encoding is the repo's own `rsijev.encode.encode_question` -- the same prompt,
truncation and option spans as training, evaluation and `scripts/serve.py` --
and the wire mapping is `serve.wire`, so a decision here has the same shape as
an answer from `POST /v1/systemone`.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import mlx.core as mx
import numpy as np

from rsijev.contract import MODES, Question
from rsijev.encode import EncodeConfig, encode_question
from serve.wire import parse_questions, state_to_text, to_answer


def encode_config(meta: dict) -> EncodeConfig:
    spec = meta["spec"]
    return EncodeConfig(layout=spec["layout"], option_pool=spec["option_pool"],
                        option_order="canonical")


def _pad(tokenizer, encoded: list[dict]):
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    B = len(encoded)
    T = max(len(e["input_ids"]) for e in encoded)
    K = max(len(e["option_index"]) for e in encoded)
    ids = np.full((B, T), pad, dtype=np.int32)
    di = np.zeros(B, dtype=np.int32)
    ss = np.zeros((B, K), dtype=np.int64)
    se = np.zeros((B, K), dtype=np.int64)
    om = np.zeros((B, K), dtype=bool)
    mid = np.zeros(B, dtype=np.int32)
    for r, e in enumerate(encoded):
        n, k = len(e["input_ids"]), len(e["option_index"])
        ids[r, :n] = e["input_ids"]
        di[r] = e["decision_index"]
        ss[r, :k] = [a for a, _ in e["option_span"]]
        se[r, :k] = [b for _, b in e["option_span"]]
        om[r, :k] = True
        mid[r] = MODES.index(e["mode"])
    return ids, di, ss, se, om, mid


def score_questions(model, tokenizer, state: str, questions: Sequence[Question],
                    enc: EncodeConfig, *, batch_size: int = 8) -> tuple[list[list[float]], int]:
    """Probabilities per question in the question's own option order, and the
    prompt tokens encoded. Each question is encoded with the state on its own,
    as in serve.infer.score_questions."""
    out: list[list[float]] = []
    tokens = 0
    for i in range(0, len(questions), batch_size):
        chunk = list(questions[i:i + batch_size])
        encoded = [encode_question(tokenizer, state, q, enc) for q in chunk]
        tokens += sum(len(e["input_ids"]) for e in encoded)
        logits = model.logits(*_pad(tokenizer, encoded))
        probs = np.array(mx.softmax(logits, axis=-1).astype(mx.float32))
        for r, (q, e) in enumerate(zip(chunk, encoded)):
            k = len(q.options)
            canon = [0.0] * k
            for pos, idx in enumerate(e["option_perm"]):      # presented -> canonical
                canon[idx] = float(probs[r, pos])
            out.append(canon)
    return out, tokens


def decide(model, tokenizer, meta: dict, state: Any, questions: dict,
           *, batch_size: int = 8) -> dict:
    """The body `POST /v1/systemone` returns, for the same `state` and `questions`.

    `state` is anything the wire accepts (text, JSON, a chat transcript);
    `questions` is the wire's {key: {type, instructions, criteria}} object.
    """
    qs = parse_questions(questions)
    text = state_to_text(state)
    t0 = time.perf_counter()
    probs, tokens = score_questions(model, tokenizer, text, qs, encode_config(meta),
                                    batch_size=batch_size)
    ms = (time.perf_counter() - t0) * 1000
    return {"model": meta.get("release", {}).get("name", "rsi-jev"),
            "answers": {q.key: to_answer(q, p) for q, p in zip(qs, probs)},
            "usage": {"input_tokens": tokens, "output_tokens": len(qs)},
            "timing_ms": round(ms, 1)}
