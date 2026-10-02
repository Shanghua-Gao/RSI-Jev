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
import json
import os
import threading
from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn.functional as F

from rsijev.contract import Prediction, Question
from rsijev.encode import (EncodeConfig, collate, encode_question, option_permutation, render,
                           unpermute_logits)

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


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


def _needs_option_tokens(model) -> bool:
    """Only the residual readout (and prior anchoring, which serving never asks for)
    reads `option_token_ids`. Anything that is not a DecisionModel keeps them."""
    cfg = getattr(model, "cfg", None)
    return cfg is None or bool(getattr(cfg, "residual", True))


# Padded tokens per forward pass (rows x longest row, prefix included); see run_rows.
FORWARD_MAX_TOKENS = int(os.environ.get("RSIJEV_FORWARD_MAX_TOKENS", "32768"))


def row_order(lengths: Sequence[int], batch_size: int, sort: bool = False,
              max_tokens: int | None = None) -> list[list[int]]:
    """Which rows go through the model together, as lists of indices.

    Default: in request order, `batch_size` at a time, as the evaluator batches.
    `sort`: longest first, so rows of similar length share a batch and pad less.
    Results always go back to their own index, so the order a caller sees never
    changes. Sorting is not bit-exact: a bf16 row's numbers depend slightly on what
    it is batched with (see docs/assets/profile_gb10.md), so it is opt-in."""
    idx = list(range(len(lengths)))
    if sort:
        idx.sort(key=lambda i: -lengths[i])
    if max_tokens is None:
        return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]
    # Also close a batch before its padded size (rows x longest row) passes max_tokens.
    out, cur, width = [], [], 0
    for i in idx:
        w = max(width, lengths[i])
        if cur and (len(cur) >= batch_size or (len(cur) + 1) * w > max_tokens):
            out.append(cur)
            cur, w = [], lengths[i]
        cur.append(i)
        width = w
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def run_rows(model, tokenizer, rows: Sequence[dict], *, max_options: int, device,
             batch_size: int, temperature: float = 1.0, cache=None, npfx: int = 0,
             sort: bool = False, trim_options: bool = False,
             max_tokens: int | None = None, positions: Sequence | None = None,
             row_embeds: torch.Tensor | None = None) -> list[list[float]]:
    """Probabilities for encoded rows, in the order given.

    With `cache`, every row continues from that one prefix cache of `npfx` tokens
    (rows are already re-indexed onto their suffix). One GPU->CPU copy per batch.
    `trim_options` pads the option slots to the widest question in the batch
    instead of `max_options`; the scorer's cost scales with the slots, and the
    result moves by ~1e-6 (fp32 reduction order), so it is opt-in.

    Image states (score_image_planned): `positions` gives each row its own (3, L)
    M-RoPE positions, sliced from its whole sequence's, in place of `npfx + arange`;
    padding gets 0, as `mrope_position_ids` gives it. `row_embeds` are the image
    features every row carries in its suffix (a held-back image tail), in order."""
    ks = [len(e["option_index"]) for e in rows]
    out: list[list[float] | None] = [None] * len(rows)
    tokens = _needs_option_tokens(model)
    # A forward pass holds at most FORWARD_MAX_TOKENS padded tokens, each row counted
    # with the prefix it reads, so many long questions split over more passes instead
    # of running the GPU out of memory. Rows under the budget batch exactly as before.
    if max_tokens is None:
        max_tokens = FORWARD_MAX_TOKENS
    lengths = [len(e["input_ids"]) + npfx for e in rows]
    for idx in row_order(lengths, batch_size, sort, max_tokens):
        part = [rows[i] for i in idx]
        width = max(ks[i] for i in idx) if trim_options else max_options
        batch = collate(tokenizer, part, max_options=width, device=device, option_tokens=tokens)
        if cache is not None:
            w = batch["input_ids"].shape[1]
            batch["attention_mask"] = torch.cat(
                [torch.ones((len(part), npfx), dtype=batch["attention_mask"].dtype,
                            device=device), batch["attention_mask"]], dim=1)
            batch["past_key_values"] = _replicate(cache, len(part), device)
            batch["position_ids"] = (torch.arange(w, device=device) + npfx
                                     ).unsqueeze(0).expand(len(part), w)
        if positions is not None:
            w = batch["input_ids"].shape[1]
            pos = torch.zeros((3, len(part), w), dtype=torch.long)
            for r, i in enumerate(idx):
                pos[:, r, :positions[i].shape[1]] = positions[i]
            batch["position_ids"] = pos.to(device)
        if row_embeds is not None:
            batch["image_embeds"] = row_embeds.repeat(len(part), 1)
        logits = unpermute_logits(model(**batch), batch["option_perm"],
                                  batch["option_mask"]) / temperature
        probs = F.softmax(logits, dim=-1).float().cpu()
        for r, i in enumerate(idx):
            out[i] = probs[r, : ks[i]].tolist()
    return out


@torch.no_grad()
def score_questions(model, tokenizer, state: str, questions: Sequence[Question],
                    enc: EncodeConfig, *, max_options: int | None = None,
                    device: str = "cuda", batch_size: int = 16,
                    temperature: float = 1.0, encoded: list[dict] | None = None,
                    sort: bool = False, trim_options: bool = False
                    ) -> tuple[list[Prediction], int]:
    """Answer every question about one state. Returns (predictions, prompt_tokens).

    Questions are independent: each is encoded with the state on its own and the
    model never sees another question or its answer, which is what the API
    promises. `encoded` takes rows already made by `encode_question` (or
    `encode_questions`, which gives the same rows)."""
    model.eval()
    if max_options is None:
        max_options = max(DEFAULT_MAX_OPTIONS, max(len(q.options) for q in questions))
    if encoded is None:
        encoded = [encode_question(tokenizer, state, q, enc) for q in questions]
    probs = run_rows(model, tokenizer, encoded, max_options=max_options, device=device,
                     batch_size=batch_size, temperature=temperature, sort=sort,
                     trim_options=trim_options)
    return ([Prediction(tuple(p)) for p in probs],
            sum(len(e["input_ids"]) for e in encoded))


# The Qwen2/Qwen3.5 pre-tokenizer. With it, "\n\n" followed by a printable,
# non-whitespace character is always a pre-token boundary: no alternative of the
# pattern can match across it (letters, digits and punctuation runs stop at a
# newline; the whitespace alternatives stop at a non-space), the pattern has no
# look-behind or anchors, so matching resumes after the boundary exactly as it would
# on the text alone, and byte-level BPE never merges across pre-tokens. NFC cannot
# compose or reorder across a newline, and no added token of this tokenizer contains
# one or strips whitespace. So tok(state + "\n\n" + rest) == tok(state + "\n\n") +
# tok(rest). `encode_questions` relies on that only when this exact configuration is
# what the tokenizer reports; anything else takes `encode_question` per question.
_QWEN_SPLIT = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+"
               "[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
_QWEN_PRE = {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex": _QWEN_SPLIT}, "behavior": "Isolated", "invert": False},
    {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]}


def splits_after_blank_line(tokenizer) -> bool:
    """Whether `tokenizer` is one `encode_questions` may tokenize the state once for."""
    ok = getattr(tokenizer, "_rsijev_splits_after_blank_line", None)
    if ok is not None:
        return ok
    ok = False
    try:
        j = json.loads(tokenizer.backend_tokenizer.to_str())
        ok = (j.get("normalizer") in (None, {"type": "NFC"})
              and j.get("pre_tokenizer") == _QWEN_PRE
              and j.get("model", {}).get("type") == "BPE"
              and not getattr(tokenizer, "split_special_tokens", False)
              and all(not (t.get("lstrip") or t.get("rstrip")) and "\n" not in t["content"]
                      and "\r" not in t["content"] for t in j.get("added_tokens", [])))
    except Exception:                      # a slow tokenizer, or an unknown format
        ok = False
    try:
        tokenizer._rsijev_splits_after_blank_line = ok
    except Exception:
        pass
    return ok


def encode_questions(tokenizer, state: str, questions: Sequence[Question],
                     enc: EncodeConfig) -> tuple[list[dict], list[int] | None]:
    """`[encode_question(tokenizer, state, q, enc) for q in questions]`, with the
    state tokenized once per request instead of once per question, and every other
    piece (instructions, option blocks, the cue) in one batched tokenizer call.

    Returns (rows, ids of state + "\n\n"), or (rows, None) when it fell back to
    encode_question for the whole request. The rows are identical to
    encode_question's: see `splits_after_blank_line` for why, and
    tests/test_serve_speed.py for the check on the real tokenizer. A question whose
    text after the blank line starts with whitespace, or that needs its state
    truncated, is encoded by encode_question on its own."""
    lead = f"{state}\n\n"
    if not (enc.layout == "state_first" and enc.option_order in ("canonical", "reversed")
            and state.strip() and splits_after_blank_line(tokenizer)):
        return [encode_question(tokenizer, state, q, enc) for q in questions], None
    plans = []
    texts: dict[str, None] = {lead: None}
    for q in questions:
        order = option_permutation(q, enc)
        head, parts = render(state, q, enc, order)
        rest = head[len(lead):]
        if not (head.startswith(lead) and rest[:1].isprintable() and rest[:1].strip()):
            plans.append(None)
            continue
        blocks = ["\n" + b for b in parts[:-1]]
        plans.append((q, order, rest, blocks, parts[-1]))
        for t in (rest, *blocks, parts[-1]):
            texts.setdefault(t)
    keys = list(texts)
    ids = dict(zip(keys, tokenizer(keys, add_special_tokens=False)["input_ids"]))
    lead_ids = ids[lead]
    rows = []
    for q, plan in zip(questions, plans):
        if plan is None:
            rows.append(encode_question(tokenizer, state, q, enc))
            continue
        _, order, rest, blocks, tail = plan
        row = lead_ids + ids[rest]
        spans = []
        for b in blocks:
            start = len(row)
            row = row + ids[b]
            spans.append((start, len(row)))
        row = row + ids[tail]
        if len(row) > enc.max_length:              # truncation: let the original do it
            rows.append(encode_question(tokenizer, state, q, enc))
            continue
        rows.append({"input_ids": row, "option_index": [e - 1 for _, e in spans],
                     "option_span": spans, "decision_index": len(row) - 1,
                     "options": [q.options[i] for i in order],
                     "option_perm": order, "mode": q.mode})
    return rows, list(lead_ids)


def _shared_prefix(tokenizer, state: str, enc: EncodeConfig, encoded: list[dict],
                   ids: list[int] | None = None):
    """The token prefix every question of this state shares, or None.

    `render()` builds `state + "\n\n" + instructions + ...` for the default
    layout, so the state is a prefix of every question's sequence. Returning None
    means "do not use the cache": the layouts differ, the state is empty, the
    tokenizer did not split at the boundary, or `encode_question` truncated the
    state (which it does per question, so the prefix would no longer be shared).
    """
    if enc.layout != "state_first" or not state.strip():
        return None
    if ids is None:
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

    Keyed on (model fingerprint, exact token ids, images) -- never on text, because
    two texts can tokenize the same and one text can tokenize differently in
    context. `images` is None for a text state; for an image state it is one key
    per image the ids reach (a hash of the decoded pixels plus the grid and token
    budget it was prepared at), because an image's tokens are all the same pad id
    and the ids alone do not say which picture they hold.
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
        self._entries: OrderedDict = OrderedDict()      # (fp, ids, images) -> (cache, nbytes)
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

    def _longest_prefix(self, fp: str, ids: tuple, imgs: tuple | None):
        best = None
        for key in self._entries:
            kfp, kids, kimgs = key
            if kfp != fp or len(kids) >= len(ids) or ids[:len(kids)] != kids:
                continue
            # A text entry serves only text; an image entry only a request whose
            # images it read, in the same order (the ids fix how many that is).
            if (kimgs is None) != (imgs is None):
                continue
            if kimgs is not None and imgs[:len(kimgs)] != kimgs:
                continue
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
    def get(self, model, ids: Sequence[int], device, image: "_ImageState | None" = None):
        """The cache for exactly `ids`, built, extended or reused. Read-only.

        `image` makes it an image state's cache (see score_image_questions_cached):
        the key then also holds the images read so far, and the tower runs on
        embeddings with the image features and the sequence's M-RoPE positions. A
        hit runs neither the tower nor the vision tower."""
        fp = f"{model_fingerprint(model)}|{getattr(model, '_rsijev_numerics', 'eager')}|{device}"
        ids = tuple(int(i) for i in ids)
        imgs = None if image is None else image.keys_within(ids)
        key = (fp, ids, imgs)
        with self._lock:
            got = self._entries.get(key)
            if got is not None:
                self._entries.move_to_end(key)
                self.stats["hit"] += 1
                return got[0]
            base = self._longest_prefix(fp, ids, None if image is None else image.keys)
            start = 0 if base is None else len(base[1])
            cache = None
            if base is not None:
                self._entries.move_to_end(base)
                cache = _replicate(self._entries[base][0], 1, device)
            if image is not None:
                cache = image.run(model, ids, start, cache, device)
            elif base is not None:
                tail = torch.tensor([ids[start:]], dtype=torch.long, device=device)
                cache = model.extend_prefix(cache, tail, start)
            else:
                cache = model.encode_prefix(torch.tensor([ids], dtype=torch.long, device=device))
            if base is not None:
                self.stats["extend"] += 1
                self.stats["extended_tokens"] += len(ids) - start
            else:
                self.stats["miss"] += 1
                self.stats["read_tokens"] += len(ids)
            self._store(key, cache, _cache_nbytes(cache))
            return cache


class VisionCache:
    """Vision-tower outputs that outlive a request: the same images asked about with
    other text skip the image processor and the vision tower. Opt-in,
    `RSIJEV_VISION_CACHE=1`.

    Keyed on the decoded pixels of every image of the request, in order (mode,
    size, bytes, as `image_keys` hashes them), and on the token budget each was
    prepared at. The image processor and the vision tower are deterministic
    functions of exactly that, so a hit returns the tensor the miss computed: the
    answer is the same, bit for bit. Separate from the document cache, which keys
    on the whole state's token ids as well.

    An entry holds the features (on the model's device) and the grid and token
    counts the plan needs; it is never mutated. Bounded in entries and bytes
    (RSIJEV_VISION_CACHE_ENTRIES, default 64; RSIJEV_VISION_CACHE_MB, default 1024).
    """

    def __init__(self, max_entries: int | None = None, max_bytes: int | None = None):
        env = os.environ.get
        self.max_entries = int(max_entries if max_entries is not None
                               else env("RSIJEV_VISION_CACHE_ENTRIES", 64))
        self.max_bytes = int(max_bytes if max_bytes is not None
                             else float(env("RSIJEV_VISION_CACHE_MB", 1024)) * 2**20)
        self._entries: OrderedDict = OrderedDict()     # key -> (feats, grid, ntok, nbytes)
        self._bytes = 0
        self._lock = threading.Lock()
        self.stats = {"hit": 0, "miss": 0, "evicted": 0}

    @staticmethod
    def key(images: Sequence, tokens_per_image: int, tag: str = "") -> str:
        h = hashlib.sha256(f"{tag}|{tokens_per_image}|{len(images)}|".encode())
        for im in images:
            h.update(f"{im.mode}|{im.size}|".encode())
            h.update(im.tobytes())
        return h.hexdigest()

    def get(self, key: str):
        with self._lock:
            got = self._entries.get(key)
            if got is None:
                self.stats["miss"] += 1
                return None
            self._entries.move_to_end(key)
            self.stats["hit"] += 1
            return got[:3]

    def put(self, key: str, feats: torch.Tensor, grid: torch.Tensor, ntok: list) -> None:
        nbytes = feats.numel() * feats.element_size()
        with self._lock:
            if key in self._entries or nbytes > self.max_bytes or self.max_entries < 1:
                return
            self._entries[key] = (feats, grid, list(ntok), nbytes)
            self._bytes += nbytes
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                _, (_, _, _, b) = self._entries.popitem(last=False)
                self._bytes -= b
                self.stats["evicted"] += 1


def default_vision_cache(model=None) -> VisionCache | None:
    """The model's vision-output cache when RSIJEV_VISION_CACHE=1, else None. Kept on
    the model object, so a cache never serves features of other weights."""
    if not _flag("RSIJEV_VISION_CACHE"):
        return None
    holder = model if model is not None else default_vision_cache
    vc = getattr(holder, "_rsijev_vision_cache", None)
    if vc is None:
        vc = VisionCache()
        try:
            holder._rsijev_vision_cache = vc
        except AttributeError:             # an object that cannot hold one: no cache
            return None
    return vc


_DOC_CACHE: DocCache | None = None


def default_doc_cache() -> DocCache | None:
    """The process-wide cache when RSIJEV_DOC_CACHE=1, else None (off by default)."""
    global _DOC_CACHE
    if not _flag("RSIJEV_DOC_CACHE"):
        return None
    if _DOC_CACHE is None:
        _DOC_CACHE = DocCache()
    return _DOC_CACHE


def _suffixes(encoded: list[dict], npfx: int) -> list[dict]:
    """Re-index each question onto its suffix: a cache supplies the first `npfx`."""
    return [{**e,
             "input_ids": e["input_ids"][npfx:],
             "option_index": [i - npfx for i in e["option_index"]],
             "option_span": [(a - npfx, b - npfx) for a, b in e["option_span"]],
             "decision_index": e["decision_index"] - npfx} for e in encoded]


def _score_on_cache(model, tokenizer, questions, encoded, cache, npfx: int, *,
                    max_options: int, device, batch_size: int, temperature: float,
                    sort: bool = False, trim_options: bool = False):
    """Continue every question from a prefix cache of its first `npfx` tokens."""
    suffix = _suffixes(encoded, npfx)
    probs = run_rows(model, tokenizer, suffix, max_options=max_options, device=device,
                     batch_size=batch_size, temperature=temperature, cache=cache,
                     npfx=npfx, sort=sort, trim_options=trim_options)
    return [Prediction(tuple(p)) for p in probs], suffix


@torch.no_grad()
def score_questions_cached(model, tokenizer, state: str, questions: Sequence[Question],
                           enc: EncodeConfig, *, max_options: int | None = None,
                           device: str = "cuda", batch_size: int = 16,
                           temperature: float = 1.0,
                           min_saved_tokens: int | None = None,
                           doc_cache: "DocCache | bool | None" = None,
                           sort: bool | None = None, trim_options: bool | None = None):
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
    plan = plan_request(tokenizer, state, questions, enc, min_saved_tokens=min_saved_tokens,
                        doc_cache=doc_cache)
    if max_options is None:
        max_options = max(DEFAULT_MAX_OPTIONS, max(len(q.options) for q in questions))
    return score_planned(model, tokenizer, plan, max_options=max_options, device=device,
                         batch_size=batch_size, temperature=temperature, sort=sort,
                         trim_options=trim_options)


def _env_on(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None or not v.strip() else _flag(name)


def speed_options(sort: bool | None = None, trim_options: bool | None = None,
                  fast_encode: bool | None = None) -> dict[str, bool]:
    """The serving path's switches, from the environment unless given.

    RSIJEV_FAST_ENCODE (on)   tokenize the state once per request; exact
    RSIJEV_SORT_ROWS (off)    batch rows of similar length together
    RSIJEV_TRIM_OPTIONS (off) pad option slots to the batch, not to max_options
    The last two move probabilities slightly (bf16 / fp32 batch composition), so
    they are opt-in; `--profile server` turns them on."""
    return {"sort": _env_on("RSIJEV_SORT_ROWS", False) if sort is None else sort,
            "trim_options": (_env_on("RSIJEV_TRIM_OPTIONS", False) if trim_options is None
                             else trim_options),
            "fast_encode": _env_on("RSIJEV_FAST_ENCODE", True) if fast_encode is None
            else fast_encode}


def plan_request(tokenizer, state: str, questions: Sequence[Question], enc: EncodeConfig, *,
                 min_saved_tokens: int | None = None,
                 doc_cache: "DocCache | bool | None" = None,
                 fast_encode: bool | None = None) -> dict:
    """Encode one request and decide how it runs, without touching the model.

    Returns {"encoded", "prefix", "path", "doc_cache"}; path is "doc" (through the
    document cache), "cached" (state read once for this request) or "plain" (each
    question read in full). score_planned runs it; the micro-batcher pools the
    "plain" ones of several requests."""
    if doc_cache is None:
        doc_cache = default_doc_cache()
    elif doc_cache is False:
        doc_cache = None
    if min_saved_tokens is None:
        min_saved_tokens = default_min_saved_tokens()
    if speed_options(fast_encode=fast_encode)["fast_encode"]:
        encoded, lead = encode_questions(tokenizer, state, questions, enc)
    else:
        encoded, lead = [encode_question(tokenizer, state, q, enc) for q in questions], None
    prefix = _shared_prefix(tokenizer, state, enc, encoded, ids=lead)
    if doc_cache is not None and prefix is not None and len(prefix) > doc_cache.holdback:
        path = "doc"
    elif prefix is None or len(questions) < 2 or \
            (len(questions) - 1) * len(prefix) < min_saved_tokens:
        path = "plain"                     # no shared prefix, or too small to pay for itself
    else:
        path = "cached"
    return {"encoded": encoded, "prefix": prefix, "path": path, "doc_cache": doc_cache,
            "options": [len(q.options) for q in questions]}


@torch.no_grad()
def score_planned(model, tokenizer, plan: dict, *, max_options: int, device,
                  batch_size: int = 16, temperature: float = 1.0,
                  sort: bool | None = None, trim_options: bool | None = None):
    """Run a plan_request plan. Returns (predictions, prompt_tokens)."""
    opt = speed_options(sort=sort, trim_options=trim_options)
    kw = dict(max_options=max_options, device=device, batch_size=batch_size,
              temperature=temperature, sort=opt["sort"], trim_options=opt["trim_options"])
    encoded, prefix = plan["encoded"], plan["prefix"]
    if plan["path"] == "plain":
        probs = run_rows(model, tokenizer, encoded, **kw)
        return [Prediction(tuple(p)) for p in probs], sum(len(e["input_ids"]) for e in encoded)
    if plan["path"] == "doc":
        doc_cache = plan["doc_cache"]
        npfx = len(prefix) - doc_cache.holdback
        cache = doc_cache.get(model, prefix[:npfx], device)
    else:
        npfx = len(prefix)
        cache = model.encode_prefix(torch.tensor([prefix], dtype=torch.long, device=device))
    out, suffix = _score_on_cache(model, tokenizer, None, encoded, cache, npfx, **kw)
    return out, npfx + sum(len(e["input_ids"]) for e in suffix)


@torch.no_grad()
def score_image_questions(model, tokenizer, prep, state: str, images: Sequence,
                          questions: Sequence[Question], enc: EncodeConfig, *,
                          max_options: int | None = None, device: str = "cuda",
                          batch_size: int = 16, temperature: float = 1.0):
    """Answer every question about one state that carries images, each question
    reading the whole state: the reference `score_image_questions_cached` (what
    the server runs) is gated against.

    `enc` is the image encoder config (the text max_length plus the image-token
    budget). Each question is encoded with the state and its images on its own,
    exactly as `rsijev.vision.encode_vision_question` does offline. Two things are
    done once per request rather than per question, and neither changes a number:
    the image processor, and the vision tower's pass (its features are copied into
    every question's row).

    Returns (predictions, prompt_tokens), image tokens included. Raises
    ValueError when the state is so long that truncation would cut an image.
    """
    if max_options is None:
        max_options = max(DEFAULT_MAX_OPTIONS, max(len(q.options) for q in questions))
    plan = plan_image_request(tokenizer, prep, state, images, questions, enc, read_once=False)
    return score_image_planned(model, tokenizer, plan, max_options=max_options, device=device,
                               batch_size=batch_size, temperature=temperature)


@torch.no_grad()
def score_image_questions_cached(model, tokenizer, prep, state: str, images: Sequence,
                                 questions: Sequence[Question], enc: EncodeConfig, *,
                                 max_options: int | None = None, device: str = "cuda",
                                 batch_size: int = 16, temperature: float = 1.0,
                                 min_saved_tokens: int | None = None,
                                 doc_cache: "DocCache | bool | None" = None,
                                 sort: bool | None = None, trim_options: bool | None = None):
    """`score_image_questions`, but the state -- image tokens included -- is read
    once and every question continues from its cache, as `score_questions_cached`
    does for text. The vision tower runs once either way.

    The trap is position. Image tokens take M-RoPE positions (t, h, w) from the
    image's grid, and the text after an image continues from the grid's extent,
    not from the token count, so the positions of a question's tail cannot be
    re-derived from where the prefix stopped. They are not: every position used
    here, prefix and suffix, is a slice of `mrope_position_ids` over the question's
    WHOLE sequence -- the very tensor the uncached path feeds the tower. Under the
    causal mask the cached pass therefore computes the same function;
    tests/test_image_cache.py checks it at fp32 on CPU.

    The same threshold as text decides whether one read pays (MIN_SAVED_TOKENS).
    With a document cache (`--profile agent`) the state always goes through it,
    keyed on the ids plus each image's pixel hash (`image_keys`): the same image
    and state asked about again runs neither tower on the state. Falls back to the
    uncached path whenever the prefix is not provably shared.
    """
    model.eval()
    if max_options is None:
        max_options = max(DEFAULT_MAX_OPTIONS, max(len(q.options) for q in questions))
    plan = plan_image_request(tokenizer, prep, state, images, questions, enc,
                              min_saved_tokens=min_saved_tokens, doc_cache=doc_cache,
                              vision_cache=default_vision_cache(model))
    return score_image_planned(model, tokenizer, plan, max_options=max_options, device=device,
                               batch_size=batch_size, temperature=temperature, sort=sort,
                               trim_options=trim_options)


def image_keys(images: Sequence, grid: torch.Tensor, tokens_per_image: int) -> tuple:
    """One key per image for the document cache: a hash of the decoded pixels
    (mode, size, bytes) and of how they were prepared (grid, token budget), so
    the same picture sent as PNG or as a data URL keys the same, and a picture
    prepared at another resolution does not."""
    keys = []
    for im, g in zip(images, grid):
        h = hashlib.sha256(f"{im.mode}|{im.size}|{tuple(int(x) for x in g)}|"
                           f"{tokens_per_image}|".encode())
        h.update(im.tobytes())
        keys.append(h.hexdigest()[:32])
    return tuple(keys)


def plan_image_request(tokenizer, prep, state: str, images: Sequence,
                       questions: Sequence[Question], enc: EncodeConfig, *,
                       min_saved_tokens: int | None = None,
                       doc_cache: "DocCache | bool | None" = None,
                       read_once: bool = True,
                       vision_cache: "VisionCache | None" = None) -> dict:
    """The model-free half of an image request, run in the request's thread as
    `plan_request` is for text, so the model's thread only runs the model: the
    image processor once, every question encoded, and the choice of path.

    `image_path` is "plain" (each question reads the whole state), "cached" (the
    state is read once for this request) or "doc" (through the document cache),
    by the same rules as text. For the last two the plan also carries the M-RoPE
    positions of every whole sequence, computed here on the CPU and sliced into
    the prefix's and each suffix's, and, for "doc", each image's key.
    With `vision_cache` (VisionCache), images seen before skip the image processor
    here and the vision tower on the model's thread: the plan carries their
    features. Raises ValueError when truncation would cut an image."""
    from rsijev.vision import IMAGE_PAD, encode_vision_question, expand_state, mrope_position_ids
    feats = vkey = None
    if vision_cache is not None:
        vkey = vision_cache.key(images, prep.tokens_per_image(len(images)),
                                f"{prep.model_id}|{prep.revision}")
        hit = vision_cache.get(vkey)
        if hit is not None:
            feats, grid, ntok = hit
            pv = None
    if feats is None:
        pv, grid, ntok = prep(images)
    encoded = [encode_vision_question(tokenizer, prep, state, images, q, enc,
                                      prepared=(pv, grid, ntok)) for q in questions]
    plan = {"path": "image", "image_path": "plain", "encoded": encoded, "pixel_values": pv,
            "grid": grid, "options": [len(q.options) for q in questions],
            "ntok": list(ntok), "image_feats": feats, "vision_cache": vision_cache,
            "vision_key": vkey}
    if not read_once:
        return plan
    if doc_cache is None:
        doc_cache = default_doc_cache()
    elif doc_cache is False:
        doc_cache = None
    if min_saved_tokens is None:
        min_saved_tokens = default_min_saved_tokens()
    prefix = _shared_prefix(tokenizer, expand_state(state, ntok), enc, encoded)
    if doc_cache is not None and prefix is not None and len(prefix) > doc_cache.holdback:
        npfx = len(prefix) - doc_cache.holdback
        plan["image_path"] = "doc"
    elif prefix is None or len(questions) < 2 or \
            (len(questions) - 1) * len(prefix) < min_saved_tokens:
        return plan
    else:
        npfx = len(prefix)
        plan["image_path"] = "cached"
    pad = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    lengths = [len(e["input_ids"]) for e in encoded]
    ids = torch.zeros((len(encoded), max(lengths)), dtype=torch.long)
    am = torch.zeros_like(ids)
    for r, e in enumerate(encoded):
        ids[r, :lengths[r]] = torch.tensor(e["input_ids"])
        am[r, :lengths[r]] = 1
    pos = mrope_position_ids(ids, am, grid.repeat(len(encoded), 1), pad)
    plan.update(prefix=list(prefix[:npfx]), npfx=npfx, doc_cache=doc_cache,
                prefix_positions=pos[:, :1, :npfx].clone(),
                suffix=_suffixes(encoded, npfx),
                suffix_positions=[pos[:, r, npfx:lengths[r]].clone() for r in range(len(encoded))],
                prefix_image_tokens=sum(1 for t in prefix[:npfx] if t == pad),
                image_tokens=sum(ntok),
                keys=(image_keys(images, grid, prep.tokens_per_image(len(images)))
                      if doc_cache is not None else None))
    return plan


class _ImageState:
    """What it takes to run any stretch of an image state's prefix: the M-RoPE
    positions of the whole sequence, the image features (computed on first need, so
    a document-cache hit never runs the vision tower) and the image keys."""

    def __init__(self, model, tokenizer, keys: tuple | None, positions: torch.Tensor, feats_fn):
        from rsijev.vision import IMAGE_PAD, VISION_START
        self.keys, self.positions, self._feats_fn, self._feats = keys, positions, feats_fn, None
        self.pad = model.image_token_id
        self.start = tokenizer.convert_tokens_to_ids(VISION_START)
        assert self.pad == tokenizer.convert_tokens_to_ids(IMAGE_PAD)

    def feats(self) -> torch.Tensor:
        if self._feats is None:
            self._feats = self._feats_fn()
        return self._feats

    def keys_within(self, ids: Sequence[int]) -> tuple:
        """The keys of the images whose runs start inside `ids`."""
        return self.keys[:sum(1 for t in ids if t == self.start)]

    def pads(self, ids: Sequence[int]) -> int:
        return sum(1 for t in ids if t == self.pad)

    def run(self, model, ids: Sequence[int], start: int, cache, device):
        """Run ids[start:] on `cache` (None: from scratch). Mutates `cache`."""
        a, b = self.pads(ids[:start]), self.pads(ids)
        emb = self.feats()[a:b] if b > a else None
        chunk = torch.tensor([list(ids[start:])], dtype=torch.long, device=device)
        pos = self.positions[:, :, start:len(ids)].to(device)
        return model.encode_image_prefix(chunk, pos, emb, cache=cache)


def _image_feats(model, plan: dict, device) -> torch.Tensor:
    """The vision tower's output for a plan's images: from the vision cache when the
    plan found them there, else computed (and kept, when the plan has a cache)."""
    feats = plan.get("image_feats")
    if feats is not None:
        return feats
    feats = model.image_embeds(plan["pixel_values"].to(device), plan["grid"].to(device))
    vc = plan.get("vision_cache")
    if vc is not None:
        vc.put(plan["vision_key"], feats, plan["grid"], plan["ntok"])
    return feats


@torch.no_grad()
def score_image_planned(model, tokenizer, plan: dict, *, max_options: int, device,
                        batch_size: int = 16, temperature: float = 1.0,
                        sort: bool | None = None, trim_options: bool | None = None):
    """Run a `plan_image_request` plan on the model's thread: the vision tower, then
    either every whole question ("plain"), or the prefix once and each question's
    suffix through `run_rows` ("cached", "doc"). Returns (predictions, prompt_tokens)."""
    from rsijev.vision import mrope_position_ids
    model.eval()
    encoded, grid = plan["encoded"], plan["grid"]
    if plan.get("image_path", "plain") != "plain":
        opt = speed_options(sort=sort, trim_options=trim_options)
        img = _ImageState(model, tokenizer, plan["keys"], plan["prefix_positions"],
                          lambda: _image_feats(model, plan, device))
        prefix, npfx = plan["prefix"], plan["npfx"]
        if plan["image_path"] == "doc":
            cache = plan["doc_cache"].get(model, prefix, device, image=img)
        else:
            cache = img.run(model, prefix, 0, None, device)
        n_pre = plan["prefix_image_tokens"]
        tail = img.feats()[n_pre:] if n_pre < plan["image_tokens"] else None
        probs = run_rows(model, tokenizer, plan["suffix"], max_options=max_options,
                         device=device, batch_size=batch_size, temperature=temperature,
                         cache=cache, npfx=npfx, sort=opt["sort"],
                         trim_options=opt["trim_options"], positions=plan["suffix_positions"],
                         row_embeds=tail)
        return ([Prediction(tuple(p)) for p in probs],
                npfx + sum(len(e["input_ids"]) for e in plan["suffix"]))
    feats = _image_feats(model, plan, device)
    out: list[Prediction] = []
    prompt_tokens = 0
    for i in range(0, len(encoded), batch_size):
        part, nopt = encoded[i:i + batch_size], plan["options"][i:i + batch_size]
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
        for r, n in enumerate(nopt):
            out.append(Prediction(tuple(probs[r, :n].float().tolist())))
    return out, prompt_tokens
