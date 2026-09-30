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

import hashlib
import importlib.util
import os
import threading
from collections import OrderedDict
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
#
# The fused kernels alone do not move a GB10 into the A100's regime. With `fla`
# on a GB10 the tower pass is still the larger cost, and break-even sits at the
# same ~400-480 saved tokens as the fallback: 0.75x at 280, 1.06x at 400, 1.28x
# at 480, 1.97x at 1,200. The A100 threshold there would give up 1.3-2x on
# every request between 480 and 2,048 saved tokens. So the fused threshold
# applies only on the data-centre GPUs it was measured on.
MIN_SAVED_TOKENS_FUSED = 2048
MIN_SAVED_TOKENS_FALLBACK = 480


def default_min_saved_tokens() -> int:
    """Pick the threshold for this machine. `RSIJEV_MIN_SAVED_TOKENS` overrides,
    including with 0 to force caching whenever a prefix is shared."""
    override = os.environ.get("RSIJEV_MIN_SAVED_TOKENS")
    if override is not None:
        return int(override)
    fused = importlib.util.find_spec("fla") is not None
    return MIN_SAVED_TOKENS_FUSED if fused and _datacentre_gpu() else MIN_SAVED_TOKENS_FALLBACK


def _datacentre_gpu() -> bool:
    """An A100/H100-class GPU (compute capability 8.0 or 9.x), where the fused
    tower pass is cheap enough that the A100 threshold holds."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in {(8, 0), (9, 0)}


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
            # Lists up to transformers 5.16, dicts keyed by state index from 5.17.
            # Missing the dicts shared them with the caller's cache, so reordering the
            # replica rewrote the original: every batch after the first, and every
            # document-cache hit, continued from a state that already held another
            # question.
            if isinstance(val, (list, dict)):
                setattr(dst, attr, val.copy())
    replica.reorder_cache(torch.zeros(rows, dtype=torch.long, device=device))
    return replica


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


def model_fingerprint(model) -> str:
    """Identify the weights a cache was computed with.

    Names, shapes, dtypes and tensor types of every parameter and buffer, plus a
    float64 sum and the leading values of each. Computed once per model object
    and kept on it; anything that changes the weights in place (quantization,
    say) drops `_rsijev_fingerprint` so it is recomputed.
    """
    fp = getattr(model, "_rsijev_fingerprint", None)
    if fp is not None:
        return fp
    h = hashlib.sha256()
    sums = []
    with torch.no_grad():
        for name, t in model.state_dict().items():
            h.update(f"{name}|{tuple(t.shape)}|{t.dtype}|{type(t).__name__};".encode())
            x = t.dequantize() if type(t) is not torch.Tensor and hasattr(t, "dequantize") else t
            x = x.detach()
            if x.numel() == 0:
                continue
            if not x.is_floating_point():
                x = x.to(torch.float32)
            sums.append(torch.sum(x, dtype=torch.float64).reshape(1).cpu())
            sums.append(x.reshape(-1)[:256].to(torch.float64).cpu())
    h.update(torch.cat(sums).numpy().tobytes())
    fp = h.hexdigest()[:24]
    model._rsijev_fingerprint = fp
    return fp


def _cache_nbytes(cache) -> int:
    n = 0
    for layer in cache.layers:
        for attr in ("keys", "values", "conv_states", "recurrent_states"):
            t = getattr(layer, attr, None)
            if isinstance(t, torch.Tensor):
                n += t.numel() * t.element_size()
    return n


class DocCache:
    """Document caches that outlive a request, for agents that ask about the same
    state again and again, or about a state that only grows.

    Keyed on (model fingerprint, exact token ids) -- never on text, because two
    texts can tokenize the same and one text can tokenize differently in context.
    A repeated state reuses its cache with no document pass. A state whose ids
    strictly extend a cached state's ids runs only the new tail, on a copy of the
    cached cache: attention keys and values and the DeltaNet recurrent and
    convolution state all continue exactly. If the tokenizer re-merged across the
    old/new boundary the ids are no longer a prefix, and the state is read in full.

    `holdback` tokens at the end of each state are left out of the cached ids and
    re-read with every question. The state is followed by a blank line and the
    question, and a growing state changes its own last few tokens (a closing
    bracket that becomes a comma); holding them back is what lets the next, longer
    state still find this one as a token prefix. It costs `holdback` tokens per
    question and does not affect exactness, which the token-prefix check alone
    guarantees.

    Bounded in entries and in bytes (RSIJEV_DOC_CACHE_ENTRIES, RSIJEV_DOC_CACHE_MB,
    RSIJEV_DOC_CACHE_HOLDBACK). Entries are never mutated: callers get a cache to
    read and must `_replicate` it before running anything on it, as the question
    pass already does.
    """

    def __init__(self, max_entries: int | None = None, max_bytes: int | None = None,
                 holdback: int | None = None):
        env = os.environ.get
        self.max_entries = int(max_entries if max_entries is not None
                               else env("RSIJEV_DOC_CACHE_ENTRIES", 32))
        self.max_bytes = int(max_bytes if max_bytes is not None
                             else float(env("RSIJEV_DOC_CACHE_MB", 2048)) * 2**20)
        self.holdback = int(holdback if holdback is not None
                            else env("RSIJEV_DOC_CACHE_HOLDBACK", 8))
        self._entries: OrderedDict = OrderedDict()      # (fp, ids) -> (cache, nbytes)
        self._bytes = 0
        self._lock = threading.Lock()
        self.stats = {"hit": 0, "extend": 0, "miss": 0, "extended_tokens": 0,
                      "read_tokens": 0, "evicted": 0}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def _longest_prefix(self, fp: str, ids: tuple):
        best = None
        for key in self._entries:
            kfp, kids = key
            if kfp == fp and len(kids) < len(ids) and ids[:len(kids)] == kids:
                if best is None or len(kids) > len(best[1]):
                    best = key
        return best

    def _store(self, key, cache, nbytes: int) -> None:
        if nbytes > self.max_bytes or self.max_entries < 1:
            return                                         # too big to keep: use once
        self._entries[key] = (cache, nbytes)
        self._bytes += nbytes
        while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
            _, (_, b) = self._entries.popitem(last=False)
            self._bytes -= b
            self.stats["evicted"] += 1

    @torch.no_grad()
    def get(self, model, ids: Sequence[int], device):
        """The cache for exactly `ids`, built, extended or reused. Read-only."""
        fp = f"{model_fingerprint(model)}|{getattr(model, '_rsijev_numerics', 'eager')}|{device}"
        ids = tuple(int(i) for i in ids)
        key = (fp, ids)
        with self._lock:
            got = self._entries.get(key)
            if got is not None:
                self._entries.move_to_end(key)
                self.stats["hit"] += 1
                return got[0]
            base = self._longest_prefix(fp, ids)
            if base is not None:
                self._entries.move_to_end(base)
                start = len(base[1])
                cache = _replicate(self._entries[base][0], 1, device)
                tail = torch.tensor([ids[start:]], dtype=torch.long, device=device)
                cache = model.extend_prefix(cache, tail, start)
                self.stats["extend"] += 1
                self.stats["extended_tokens"] += len(ids) - start
            else:
                cache = model.encode_prefix(torch.tensor([ids], dtype=torch.long, device=device))
                self.stats["miss"] += 1
                self.stats["read_tokens"] += len(ids)
            self._store(key, cache, _cache_nbytes(cache))
            return cache


_DOC_CACHE: DocCache | None = None


def default_doc_cache() -> DocCache | None:
    """The process-wide cache when RSIJEV_DOC_CACHE=1, else None (off by default)."""
    global _DOC_CACHE
    if not _flag("RSIJEV_DOC_CACHE"):
        return None
    if _DOC_CACHE is None:
        _DOC_CACHE = DocCache()
    return _DOC_CACHE


def _score_on_cache(model, tokenizer, questions, encoded, cache, npfx: int, *,
                    max_options: int, device, batch_size: int, temperature: float):
    """Continue every question from a prefix cache of its first `npfx` tokens."""
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
    return out, suffix


@torch.no_grad()
def score_questions_cached(model, tokenizer, state: str, questions: Sequence[Question],
                           enc: EncodeConfig, *, max_options: int | None = None,
                           device: str = "cuda", batch_size: int = 16,
                           temperature: float = 1.0,
                           min_saved_tokens: int | None = None,
                           doc_cache: "DocCache | bool | None" = None):
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

    `doc_cache` keeps document caches across calls (see DocCache). None means the
    process-wide one if RSIJEV_DOC_CACHE=1 and none otherwise; False turns it off.
    With a doc cache the state always goes through it, whatever the threshold,
    because the point is the next request, not this one.

    Falls back whenever the prefix is not provably shared, so a caller always
    gets an answer. Returns (predictions, prompt_tokens), where prompt_tokens
    counts the prefix once, because it is computed once.
    """
    model.eval()
    if doc_cache is None:
        doc_cache = default_doc_cache()
    elif doc_cache is False:
        doc_cache = None
    if min_saved_tokens is None:
        min_saved_tokens = default_min_saved_tokens()
    if max_options is None:
        max_options = max(DEFAULT_MAX_OPTIONS, max(len(q.options) for q in questions))
    encoded = [encode_question(tokenizer, state, q, enc) for q in questions]
    prefix = _shared_prefix(tokenizer, state, enc, encoded)
    kw = dict(max_options=max_options, device=device, batch_size=batch_size,
              temperature=temperature)

    if doc_cache is not None and prefix is not None and len(prefix) > doc_cache.holdback:
        npfx = len(prefix) - doc_cache.holdback
        cache = doc_cache.get(model, prefix[:npfx], device)
        out, suffix = _score_on_cache(model, tokenizer, questions, encoded, cache, npfx, **kw)
        return out, npfx + sum(len(e["input_ids"]) for e in suffix)

    if prefix is not None and (len(questions) - 1) * len(prefix) < min_saved_tokens:
        prefix = None                      # real, but too small to pay for itself
    if prefix is None or len(questions) < 2:
        return score_questions(model, tokenizer, state, questions, enc, **kw)

    npfx = len(prefix)
    pids = torch.tensor([prefix], dtype=torch.long, device=device)
    cache = model.encode_prefix(pids)
    out, suffix = _score_on_cache(model, tokenizer, questions, encoded, cache, npfx, **kw)
    return out, npfx + sum(len(e["input_ids"]) for e in suffix)


@torch.no_grad()
def score_image_questions(model, tokenizer, prep, state: str, images: Sequence,
                          questions: Sequence[Question], enc: EncodeConfig, *,
                          max_options: int | None = None, device: str = "cuda",
                          batch_size: int = 16, temperature: float = 1.0):
    """Answer every question about one state that carries images.

    `enc` is the image encoder config (the text max_length plus the image-token
    budget). Each question is encoded with the state and its images on its own,
    exactly as `rsijev.vision.encode_vision_question` does offline. Two things are
    done once per request rather than per question, and neither changes a number:
    the image processor, and the vision tower's pass (its features are copied into
    every question's row).

    Neither the prefix cache nor the document cache is used: both hold text-only
    tower states, and an image request goes through `inputs_embeds` with M-RoPE
    positions, which they do not cover. This path always reads the whole state.

    Returns (predictions, prompt_tokens), image tokens included. Raises
    ValueError when the state is so long that truncation would cut an image.
    """
    from rsijev.vision import encode_vision_question, mrope_position_ids
    model.eval()
    if max_options is None:
        max_options = max(DEFAULT_MAX_OPTIONS, max(len(q.options) for q in questions))
    pv, grid, ntok = prep(images)
    encoded = [encode_vision_question(tokenizer, prep, state, images, q, enc,
                                      prepared=(pv, grid, ntok)) for q in questions]
    feats = model.image_embeds(pv.to(device), grid.to(device))
    out: list[Prediction] = []
    prompt_tokens = 0
    for i in range(0, len(questions), batch_size):
        chunk, part = list(questions[i:i + batch_size]), encoded[i:i + batch_size]
        prompt_tokens += sum(len(e["input_ids"]) for e in part)
        batch = collate(tokenizer, part, max_options=max_options, device="cpu")
        rows = len(part)
        batch["position_ids"] = mrope_position_ids(
            batch["input_ids"], batch["attention_mask"], grid.repeat(rows, 1),
            model.image_token_id)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = unpermute_logits(model(**batch, image_embeds=feats.repeat(rows, 1)),
                                  batch["option_perm"], batch["option_mask"]) / temperature
        probs = F.softmax(logits, dim=-1)
        for r, q in enumerate(chunk):
            out.append(Prediction(tuple(probs[r, : len(q.options)].float().tolist())))
    return out, prompt_tokens
