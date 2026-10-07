"""Batching encoded rows for the MLX path, in numpy (no torch at inference).

`collate` is rsijev.encode.collate without torch and without `option_token_ids` (only the
residual readout reads them, and no MLX-served release uses it): right padding, explicit
positions, the same keys and the same values. `unpermute` is rsijev.encode.unpermute_logits.
tests/test_mlx_parity.py checks both against the torch versions.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from rsijev.contract import MODES


def collate(tokenizer, examples: Sequence[dict], max_options: int) -> dict[str, np.ndarray]:
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    n = len(examples)
    width = max(len(e["input_ids"]) for e in examples)
    ids = np.full((n, width), pad, dtype=np.int32)
    am = np.zeros((n, width), dtype=np.int32)
    oi = np.zeros((n, max_options), dtype=np.int32)
    os_ = np.zeros((n, max_options), dtype=np.int32)
    oe = np.zeros((n, max_options), dtype=np.int32)
    om = np.zeros((n, max_options), dtype=bool)
    mid = np.zeros(n, dtype=np.int32)
    perm = np.zeros((n, max_options), dtype=np.int32)
    di = np.zeros(n, dtype=np.int32)
    for r, e in enumerate(examples):
        t = len(e["input_ids"])
        ids[r, :t] = e["input_ids"]
        am[r, :t] = 1
        di[r] = e["decision_index"]
        k = len(e["option_index"])
        if k > max_options:
            raise ValueError(f"{k} options exceeds max_options={max_options}")
        oi[r, :k] = e["option_index"]
        sp = e.get("option_span")
        if sp:
            os_[r, :k] = [a for a, _ in sp]
            oe[r, :k] = [b for _, b in sp]
        om[r, :k] = True
        if e.get("option_perm"):
            perm[r, :k] = e["option_perm"]
        if e.get("mode"):
            mid[r] = MODES.index(e["mode"])
    return {"input_ids": ids, "attention_mask": am, "decision_index": di, "option_index": oi,
            "option_span_start": os_, "option_span_end": oe, "option_mask": om,
            "mode_id": mid, "option_perm": perm}


def unpermute(logits: np.ndarray, option_perm: np.ndarray, option_mask: np.ndarray) -> np.ndarray:
    """Logits from PRESENTED to canonical option order (-inf where masked)."""
    B, K = logits.shape
    out = np.full((B, K + 1), -np.inf, dtype=logits.dtype)
    idx = np.where(option_mask, option_perm, K)
    src = np.where(option_mask, logits, -np.inf)
    rows = np.arange(B)[:, None]
    out[rows, idx] = src
    return out[:, :K]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64)
    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - m)
    return (e / e.sum(-1, keepdims=True)).astype(np.float32)


def suffixes(encoded: list[dict], npfx: int) -> list[dict]:
    """Each row re-indexed onto its suffix after a shared prefix of `npfx` tokens."""
    return [{**e,
             "input_ids": e["input_ids"][npfx:],
             "option_index": [i - npfx for i in e["option_index"]],
             "option_span": [(a - npfx, b - npfx) for a, b in e["option_span"]],
             "decision_index": e["decision_index"] - npfx} for e in encoded]
