"""The serving forward pass.

This is `rsijev.evaluate.predict` without the `Case` wrapper. `predict` takes
`Case` objects, and a `Case` requires gold; a served request has no gold, and
inventing a uniform one would put a fabricated label in a field the whole
project treats as ground truth.

So this calls the same primitives in the same order -- `encode_question`,
`collate`, the model, `unpermute_logits`, softmax -- and returns the same
`Prediction` objects. `tests/test_serve.py::test_matches_evaluate_predict`
asserts the two paths agree bit for bit on the same inputs, so they cannot
drift apart silently.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Sequence

import torch
import torch.nn.functional as F

from rsijev.contract import Prediction, Question
from rsijev.encode import EncodeConfig, collate, encode_question, unpermute_logits

DEFAULT_MAX_OPTIONS = 80

# Redundant prefix tokens ((Q-1) * prefix) that must be avoidable before the
# cached path is worth its fixed cost. See score_questions_cached.
#
# The break-even point is not a property of the model, it is a property of how
# expensive one tower pass is on the machine -- so it differs by an order of
# magnitude depending on whether the fused linear-attention kernels are present.
# With `fla` installed on an A100, a tower pass is cheap and the fixed cost of a
# second sequential pass plus replicating the cache dominates until the prefix
# is large: a 98-token state measured 0.55x, an outright loss. On the plain
# torch fallback the tower pass is the whole cost, and caching won at every size
# we measured on a GB10: 1.09x at 161 tokens rising to 3.10x at 1,052.
#
# 480 is the smallest saving we have actually measured a win at (161 tokens over
# four questions), not an extrapolation toward zero.
MIN_SAVED_TOKENS_FUSED = 2048
MIN_SAVED_TOKENS_FALLBACK = 480


def default_min_saved_tokens() -> int:
    """Pick the threshold for this machine. `RSIJEV_MIN_SAVED_TOKENS` overrides,
    including with 0 to force caching whenever a prefix is shared."""
    override = os.environ.get("RSIJEV_MIN_SAVED_TOKENS")
    if override is not None:
        return int(override)
    fused = importlib.util.find_spec("fla") is not None
    return MIN_SAVED_TOKENS_FUSED if fused else MIN_SAVED_TOKENS_FALLBACK


MIN_SAVED_TOKENS = default_min_saved_tokens()


@torch.no_grad()
def score_questions(model, tokenizer, state: str, questions: Sequence[Question],
                    enc: EncodeConfig, *, max_options: int | None = None,
                    device: str = "cuda", batch_size: int = 16,
                    temperature: float = 1.0) -> tuple[list[Prediction], int]:
    """Answer every question about one state. Returns (predictions, prompt_tokens).

    Questions are independent: each is encoded with the state on its own and the
    model never sees another question or its answer, which is what the API
    promises.
    """
    model.eval()
    if max_options is None:
        max_options = max(DEFAULT_MAX_OPTIONS, max(len(q.options) for q in questions))
    out: list[Prediction] = []
    prompt_tokens = 0
    for i in range(0, len(questions), batch_size):
        chunk = list(questions[i:i + batch_size])
        encoded = [encode_question(tokenizer, state, q, enc) for q in chunk]
        prompt_tokens += sum(len(e["input_ids"]) for e in encoded)
        batch = collate(tokenizer, encoded, max_options=max_options, device=device)
        logits = unpermute_logits(model(**batch), batch["option_perm"],
                                  batch["option_mask"]) / temperature
        probs = F.softmax(logits, dim=-1)
        for r, q in enumerate(chunk):
            out.append(Prediction(tuple(probs[r, : len(q.options)].float().tolist())))
    return out, prompt_tokens


def _shared_prefix(tokenizer, state: str, enc: EncodeConfig, encoded: list[dict]):
    """The token prefix every question of this state shares, or None.

    `render()` builds `state + "\n\n" + instructions + ...` for the default
    layout, so the state is a prefix of every question's sequence. Returning None
    means "do not use the cache": the layouts differ, the state is empty, the
    tokenizer did not split at the boundary, or `encode_question` truncated the
    state (which it does per question, so the prefix would no longer be shared).
    """
    if enc.layout != "state_first" or not state.strip():
        return None
    ids = tokenizer(f"{state}\n\n", add_special_tokens=False)["input_ids"]
    if not ids:
        return None
    for e in encoded:
        if e["input_ids"][:len(ids)] != ids:
            return None                    # boundary moved, or the state was truncated
        if e["decision_index"] < len(ids) or min(e["option_index"], default=0) < len(ids):
            return None                    # nothing the readout needs may sit in the prefix
    return ids


def _replicate(cache, rows: int, device):
    """One state's cache, viewed by `rows` questions.

    The cache is hybrid: attention layers hold keys and values, the DeltaNet
    layers hold convolution and recurrent state. Shallow-copy the container and
    each layer so the caller's cache is untouched, then broadcast batch 1 to
    `rows` by reordering onto index 0 -- the same trick a beam search uses.
    """
    import copy
    replica = copy.copy(cache)
    replica.layers = [copy.copy(layer) for layer in cache.layers]
    for src, dst in zip(cache.layers, replica.layers):
        for attr in ("conv_states", "recurrent_states", "is_conv_states_initialized",
                     "is_recurrent_states_initialized", "has_previous_state",
                     "conv_kernel_size"):
            val = getattr(src, attr, None)
            if isinstance(val, list):
                setattr(dst, attr, val.copy())
    replica.reorder_cache(torch.zeros(rows, dtype=torch.long, device=device))
    return replica


@torch.no_grad()
def score_questions_cached(model, tokenizer, state: str, questions: Sequence[Question],
                           enc: EncodeConfig, *, max_options: int | None = None,
                           device: str = "cuda", batch_size: int = 16,
                           temperature: float = 1.0,
                           min_saved_tokens: int | None = None):
    """`score_questions`, but the state is encoded once instead of per question.

    Only when that pays. Uncached costs Q*(P+S) and cached costs P + Q*S plus a
    fixed cost -- a second sequential pass and replicating the cache across the
    batch -- so the saving is (Q-1)*P and it has to clear that fixed cost.
    Whether that pays depends on the machine, so the default threshold is chosen
    per machine -- see MIN_SAVED_TOKENS above. With fused kernels on an A100 and
    five questions: 98 tokens is 0.55x (a real loss), 402 is break-even, 1,022 is
    2.3x. On the torch fallback on a GB10 with four questions, caching won
    everywhere measured: 2.02x at 404 tokens, 3.10x at 1,052. Pass
    `min_saved_tokens` (or set RSIJEV_MIN_SAVED_TOKENS) to override.

    Falls back whenever the prefix is not provably shared, so a caller always
    gets an answer. Returns (predictions, prompt_tokens), where prompt_tokens
    counts the prefix once, because it is computed once.
    """
    model.eval()
    if min_saved_tokens is None:
        min_saved_tokens = default_min_saved_tokens()
    if max_options is None:
        max_options = max(DEFAULT_MAX_OPTIONS, max(len(q.options) for q in questions))
    encoded = [encode_question(tokenizer, state, q, enc) for q in questions]
    prefix = _shared_prefix(tokenizer, state, enc, encoded)
    if prefix is not None and (len(questions) - 1) * len(prefix) < min_saved_tokens:
        prefix = None                      # real, but too small to pay for itself
    if prefix is None or len(questions) < 2:
        return score_questions(model, tokenizer, state, questions, enc,
                               max_options=max_options, device=device,
                               batch_size=batch_size, temperature=temperature)

    npfx = len(prefix)
    pids = torch.tensor([prefix], dtype=torch.long, device=device)
    cache = model.encode_prefix(pids)

    # Re-index onto the suffix: the cache supplies everything before it.
    suffix = [{**e,
               "input_ids": e["input_ids"][npfx:],
               "option_index": [i - npfx for i in e["option_index"]],
               "option_span": [(a - npfx, b - npfx) for a, b in e["option_span"]],
               "decision_index": e["decision_index"] - npfx} for e in encoded]

    out: list[Prediction] = []
    for i in range(0, len(questions), batch_size):
        chunk, part = list(questions[i:i + batch_size]), suffix[i:i + batch_size]
        batch = collate(tokenizer, part, max_options=max_options, device=device)
        width = batch["input_ids"].shape[1]
        batch["attention_mask"] = torch.cat(
            [torch.ones((len(part), npfx), dtype=batch["attention_mask"].dtype, device=device),
             batch["attention_mask"]], dim=1)
        pos = (torch.arange(width, device=device) + npfx).unsqueeze(0).expand(len(part), width)
        logits = unpermute_logits(
            model(**batch, past_key_values=_replicate(cache, len(part), device),
                  position_ids=pos),
            batch["option_perm"], batch["option_mask"]) / temperature
        probs = F.softmax(logits, dim=-1)
        for r, q in enumerate(chunk):
            out.append(Prediction(tuple(probs[r, : len(q.options)].float().tolist())))

    tokens = npfx + sum(len(e["input_ids"]) for e in suffix)
    return out, tokens
