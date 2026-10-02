"""Turning a Case into model input. PROTECTED — the layout is an ARCH choice,
but the bookkeeping that makes positions correct is not, and a silent off-by-one
here would corrupt every arm identically and invisibly.

One encoded example carries, besides the token ids:

  decision_index : the position whose hidden state is the readout's query
  option_index   : one position per option, the LAST token of that option's block
  option_mask    : which of the K slots are real

The option block layout is deliberately the thing an arm can vary (options
before the question, after it, descriptions stripped) while the index
bookkeeping stays fixed.
"""
from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from .contract import MODES, Case, Question

Layout = Literal["state_first", "options_first"]


@dataclass
class EncodeConfig:
    layout: Layout = "state_first"
    # The order options are PRESENTED in. A causal encoder gives option k a
    # representation that has seen options 1..k-1 and not k+1..K, so position
    # carries information the task does not intend. The trained readout collapses
    # onto that off-distribution: last option on 800/800 typed-decisions score
    # questions, first option on about half of MMLU-Pro.
    #   canonical  as written
    #   reversed   the diagnostic: if the head still answers by POSITION it is
    #              positional, if it follows content it is semantic
    #   shuffled   the augmentation, per example, from the encoder's rng
    # Logits come back in PRESENTED order and are mapped to canonical order by
    # `unpermute_logits` before anything in contract.py sees them, so
    # Prediction.score()'s sum(i*p_i) keeps indexing levels and not positions.
    option_order: Literal["canonical", "reversed", "shuffled"] = "canonical"
    # How an option's block becomes one vector. "mean" pools the whole block;
    # "last" takes its final token, which is usually punctuation and can carry
    # almost no option identity in a model with weak context mixing.
    option_pool: Literal["mean", "last"] = "mean"
    include_criteria: bool = True      # the descriptions are part of the task
    max_length: int = 2048
    answer_cue: str = "Answer:"
    # What happens to a STATE that does not fit max_length. The question, the
    # option labels and the cue are never cut.
    #   "middle"  keep the state's head and tail, drop the middle and put
    #             `cut_marker` where it was. The tail gets 1 - head_frac of the
    #             kept tokens (evidence tends to sit near the end: kev_hard),
    #             and the head always keeps the state's first line (a leading
    #             "Query: ..." line) when it fits in half the room. If the
    #             question and options alone leave the state no room, the
    #             option DESCRIPTIONS are trimmed evenly (labels intact) and
    #             the row reports `options_cut`; the encoder never refuses a
    #             row whose labels, question and cue fit.
    #   "left"    drop the state's start, silently (v1.0 - v4.0-VL behaviour; a
    #             leading query was the first thing lost).
    #   "none"    cut nothing: a longer question raises InputTooLong. Serving uses
    #             this, so no request is answered on input it never saw; training
    #             and the benchmark scripts keep "left" so published numbers reproduce.
    # Either way the encoded row reports `state_cut` = state tokens dropped
    # (0 when the row fits, which is then encoded exactly as before).
    # "left" is the default: every release so far was trained and gated with it.
    # "middle" is the long-context encoder (private branch longctx e1069bf); a
    # release trained with it says so in meta.json (`spec.truncate`), and
    # serve/release.py then serves it that way. In "left" mode a row is encoded
    # byte for byte as by the v1.0 - v4.0-VL encoder, structured option
    # descriptions included (see `render`).
    truncate: Literal["middle", "left", "none"] = "left"
    head_frac: float = 0.25          # share of the kept state taken from its head
    cut_marker: str = "\n[... {n} tokens of the state omitted ...]\n"
    desc_marker: str = " ..."        # appended to a trimmed option description


class _NoRoom(ValueError):
    """The question and options leave the state no room (middle mode only)."""


class InputTooLong(ValueError):
    """A question longer than EncodeConfig.max_length with truncation off."""


def option_first_token_ids(tokenizer, options: Sequence[str],
                           prefix: str = " ") -> list[int]:
    """The tokens that actually distinguish the options at the answer position.

    A leading space is right for word options (" stop" is what follows
    "Answer:") and catastrophic for short numeric ones: this tokenizer emits the
    space as its own token, so " 0", " 1", " 2", " 3" all begin with token 220
    and every score question would score identically. Fall back to the bare form
    when the prefixed form collides.
    """
    def first(o: str, pre: str) -> int:
        return tokenizer(pre + o, add_special_tokens=False)["input_ids"][0]
    ids = [first(o, prefix) for o in options]
    if len(set(ids)) == len(ids) or not prefix:
        return ids
    bare = [first(o, "") for o in options]
    return bare if len(set(bare)) == len(bare) else ids


def option_permutation(q: Question, cfg: EncodeConfig,
                       rng: random.Random | None = None) -> list[int]:
    """presented position -> canonical option index."""
    order = list(range(len(q.options)))
    if cfg.option_order == "reversed":
        order.reverse()
    elif cfg.option_order == "shuffled":
        (rng or random.Random(0)).shuffle(order)
    return order


def criterion_text(c) -> str:
    """One option's description as text. A string is returned unchanged, so string
    criteria encode byte-identically to before. A structured description is rendered
    deterministically: {"what": ..., "includes": [...], "excludes": [...]} ->
    "<what>. Includes: a; b. Excludes: c." Other keys follow in their given order as
    "<Key>: <value>"; lists join with "; "; numbers and booleans as JSON; None -> ""."""
    if isinstance(c, str):
        return c
    if c is None:
        return ""
    if isinstance(c, (list, tuple)):
        return "; ".join(t for t in (criterion_text(x) for x in c) if t)
    if isinstance(c, dict):
        first = [k for k in ("what", "includes", "excludes") if k in c]
        parts = []
        for k in first + [k for k in c if k not in first]:
            v = criterion_text(c[k]).strip()
            if v:
                v = v if k == "what" else f"{str(k)[:1].upper()}{str(k)[1:]}: {v}"
                parts.append(v if v[-1] in ".!?" else v + ".")
        return " ".join(parts)
    return json.dumps(c, ensure_ascii=False)


def _criterion(c, cfg: EncodeConfig):
    """A description as the encoder in force renders it: `criterion_text` for the
    long-context encoder (e1069bf, as the 4B line was trained), the value as it is
    (the v1.0 - v4.0-VL encoder) otherwise. A string is the same either way."""
    return criterion_text(c) if cfg.truncate == "middle" else c


def render(state: str, q: Question, cfg: EncodeConfig,
           order: Sequence[int] | None = None) -> tuple[str, list[str]]:
    """Return the prompt prefix and the per-option blocks, in PRESENTED order."""
    order = list(range(len(q.options))) if order is None else list(order)
    blocks = [
        f"- {q.options[i]}: {_criterion(q.criteria[q.options[i]], cfg)}"
        if cfg.include_criteria and q.criteria.get(q.options[i]) else f"- {q.options[i]}"
        for i in order
    ]
    # MMLU-Pro carries its question in `instructions` and has no separate state,
    # so an unguarded template would start every one of those prompts with two
    # blank lines -- a formatting difference between corpora that no arm asked for.
    if cfg.layout == "options_first":
        head = f"{q.instructions}\nOptions:\n"
        tail = (f"\n\n{state}\n\n{cfg.answer_cue}" if state.strip()
                else f"\n\n{cfg.answer_cue}")
    else:
        head = (f"{state}\n\n{q.instructions}\nOptions:\n" if state.strip()
                else f"{q.instructions}\nOptions:\n")
        tail = f"\n\n{cfg.answer_cue}"
    return head, blocks + [tail]


def encode_question(tokenizer, state: str, q: Question, cfg: EncodeConfig,
                    rng: random.Random | None = None) -> dict[str, list[int]]:
    """Token ids plus the positions the readout needs. No padding here.

    Every row carries `state_cut` (state tokens dropped) and `options_cut`
    (option-description tokens dropped); both are 0 for a row that fits, which
    is then encoded exactly as by the v1.0-v2.1 encoder."""
    order = option_permutation(q, cfg, rng)
    try:
        out = _encode(tokenizer, state, q, cfg, order)
        out["options_cut"] = 0
        return out
    except _NoRoom:
        if cfg.truncate != "middle":
            raise
    return _encode_trimmed(tokenizer, state, q, cfg, order)


def _encode(tokenizer, state: str, q: Question, cfg: EncodeConfig, order) -> dict:
    head, parts = render(state, q, cfg, order)
    ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    spans: list[tuple[int, int]] = []
    for block in parts[:-1]:                       # one per option
        start = len(ids)
        ids += tokenizer("\n" + block, add_special_tokens=False)["input_ids"]
        spans.append((start, len(ids)))            # [start, end) of this option block
    option_index = [e - 1 for _, e in spans]
    ids += tokenizer(parts[-1], add_special_tokens=False)["input_ids"]
    decision_index = len(ids) - 1

    if len(ids) > cfg.max_length and cfg.truncate == "none":
        raise InputTooLong(f"question {q.key} is {len(ids)} tokens, over the maximum "
                           f"context length of {cfg.max_length}")
    state_cut = 0
    if len(ids) > cfg.max_length:
        # Truncate the STATE, never the options, the question or the cue:
        # dropping an option silently changes the task, and dropping the cue
        # moves the readout.
        overflow = len(ids) - cfg.max_length
        head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
        if cfg.truncate == "left":
            if overflow >= len(head_ids):
                raise _NoRoom(f"cannot fit {q.key}: options alone exceed max_length")
            keep = head_ids[overflow:]
            ids = keep + ids[len(head_ids):]
            option_index = [i - overflow for i in option_index]
            spans = [(a - overflow, b - overflow) for a, b in spans]
            decision_index -= overflow
            state_cut = overflow
        else:
            ids, option_index, spans, decision_index, state_cut = _middle_cut(
                tokenizer, state, q, cfg, head_ids, ids, option_index, spans)
    return {"input_ids": ids, "option_index": option_index,
            "option_span": spans, "decision_index": decision_index,
            "options": [q.options[i] for i in order],
            "option_perm": order, "mode": q.mode, "state_cut": state_cut}


def _cut_state(tokenizer, state: str, budget: int, cfg: EncodeConfig) -> tuple[list[int], int]:
    """The state's token ids cut to at most `budget` tokens: head + marker + tail.
    Returns (ids, state tokens dropped)."""
    s_ids = tokenizer(state, add_special_tokens=False)["input_ids"]
    if len(s_ids) <= budget:
        return s_ids, 0
    # the marker's length depends on the count it prints; size it for the
    # largest count first, so the kept head + tail can only get more room
    room = budget - len(tokenizer(cfg.cut_marker.format(n=len(s_ids)),
                                  add_special_tokens=False)["input_ids"])
    if room < MIN_STATE_ROOM:
        raise _NoRoom("state budget is smaller than the cut marker")
    h = int(room * cfg.head_frac)
    # the first line (a leading "Query: ..." line) is always in the head
    first = state.split("\n", 1)[0]
    n_first = len(tokenizer(first, add_special_tokens=False)["input_ids"]) + 1
    if n_first <= room // 2:
        h = max(h, n_first)
    t = room - h
    n = len(s_ids) - h - t
    marker = tokenizer(cfg.cut_marker.format(n=n), add_special_tokens=False)["input_ids"]
    return s_ids[:h] + marker + (s_ids[len(s_ids) - t:] if t else []), n


def _middle_cut(tokenizer, state, q, cfg, head_ids, ids, option_index, spans):
    """Re-encode an over-long row with the middle of its state replaced by a
    marker. The state is tokenised on its own here; everything outside it keeps
    the ids it had."""
    if not state.strip():
        raise _NoRoom(f"cannot fit {q.key}: question and options alone exceed max_length")
    if cfg.layout == "options_first":
        # state sits in the tail: ...options | "\n\n" state "\n\n" cue
        n_opt = spans[-1][1] if spans else len(head_ids)
        pre = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        post = tokenizer(f"\n\n{cfg.answer_cue}", add_special_tokens=False)["input_ids"]
        budget = cfg.max_length - n_opt - len(pre) - len(post)
        if budget <= 0:
            raise _NoRoom(f"cannot fit {q.key}: question and options alone exceed max_length")
        s_ids, cut = _cut_state(tokenizer, state, budget, cfg)
        ids = ids[:n_opt] + pre + s_ids + post
        return ids, option_index, spans, len(ids) - 1, cut
    rest = tokenizer(f"\n\n{q.instructions}\nOptions:\n", add_special_tokens=False)["input_ids"]
    fixed = len(rest) + len(ids) - len(head_ids)       # question + options + cue
    budget = cfg.max_length - fixed
    if budget <= 0:
        raise _NoRoom(f"cannot fit {q.key}: question and options alone exceed max_length")
    s_ids, cut = _cut_state(tokenizer, state, budget, cfg)
    new_head = s_ids + rest
    shift = len(head_ids) - len(new_head)
    ids = new_head + ids[len(head_ids):]
    option_index = [i - shift for i in option_index]
    spans = [(a - shift, b - shift) for a, b in spans]
    return ids, option_index, spans, len(ids) - 1, cut


MIN_STATE_ROOM = 32     # state tokens kept around the marker before options are trimmed


def _ntok(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _encode_trimmed(tokenizer, state, q, cfg, order) -> dict:
    """The question and options leave the state no room: trim every option
    description to the same token allowance (shorter ones stay whole, labels
    are never touched), so the state keeps min(its length, max(256, cap/8))
    tokens. If the labels, question and cue alone still do not fit, drop the
    descriptions, then cut the middle of the question text."""
    cap = cfg.max_length
    s_tok = _ntok(tokenizer, state) if state.strip() else 0
    reserve = min(s_tok, max(256, cap // 8)) if s_tok else 0
    desc = {o: criterion_text(q.criteria.get(o)) if cfg.include_criteria else "" for o in q.options}
    d_ids = {o: tokenizer(d, add_special_tokens=False)["input_ids"] for o, d in desc.items()}
    # question + options + cue, untruncated
    fixed = len(_encode(tokenizer, "", q, dataclasses.replace(cfg, max_length=10 ** 9), order)["input_ids"])
    total = sum(len(v) for v in d_ids.values())

    def with_allowance(a: int) -> Question:
        crit = {}
        for o in q.options:
            v = d_ids[o]
            crit[o] = desc[o] if len(v) <= a else (
                tokenizer.decode(v[:a]).rstrip() + cfg.desc_marker if a > 0 else "")
        return dataclasses.replace(q, criteria=crit)

    need = fixed - (cap - reserve) + 16
    lo, hi = 0, max((len(v) for v in d_ids.values()), default=0)
    while lo < hi:                              # largest allowance under budget
        mid = (lo + hi + 1) // 2
        if total - sum(min(len(v), mid) for v in d_ids.values()) >= need:
            lo = mid
        else:
            hi = mid - 1
    a = lo
    for _ in range(12):
        q2 = with_allowance(a)
        try:
            out = _encode(tokenizer, state, q2, cfg, order)
            out["options_cut"] = total - sum(min(len(v), a) for v in d_ids.values())
            out["options"] = [q.options[i] for i in order]
            return out
        except _NoRoom:
            if a == 0:
                break
            a = int(a * 0.8)
    # labels + question + cue do not fit: cut the middle of the question text
    q2 = with_allowance(0)
    ins = tokenizer(q.instructions, add_special_tokens=False)["input_ids"]
    big = dataclasses.replace(cfg, max_length=10 ** 9)
    over = len(_encode(tokenizer, "", q2, big, order)["input_ids"]) + min(s_tok, MIN_STATE_ROOM + 32) - cap + 24
    keep = len(ins) - over
    if keep < 16:
        raise ValueError(f"cannot fit {q.key}: the option labels alone exceed max_length")
    h = keep // 2
    text = (tokenizer.decode(ins[:h]) + " [...] " + tokenizer.decode(ins[len(ins) - (keep - h):]))
    q3 = dataclasses.replace(q2, instructions=text)
    out = _encode(tokenizer, state, q3, cfg, order)
    out["options_cut"] = total
    out["question_cut"] = len(ins) - keep
    out["options"] = [q.options[i] for i in order]
    return out


def truncation_report(tokenizer, pairs, cfg: EncodeConfig) -> list[int]:
    """state_cut per (case, question) pair, for records written outside the
    protected evaluator (0 = the row fits and is unchanged)."""
    return [encode_question(tokenizer, c.state, q, cfg)["state_cut"] for c, q in pairs]


def collate(tokenizer, examples: Sequence[dict], max_options: int,
            device: str | torch.device = "cpu",
            option_tokens: bool = True) -> dict[str, torch.Tensor]:
    """Right-pad. Positions are explicit, so padding side cannot shift them.

    `option_tokens=False` leaves `option_token_ids` at zero instead of looking up
    each option's first token. Only the residual readout and prior anchoring read
    that tensor; the lookups are two tokenizer calls per option, which a server
    answering a trained readout pays for nothing."""
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    width = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), width), pad, dtype=torch.long)
    am = torch.zeros((len(examples), width), dtype=torch.long)
    oi = torch.zeros((len(examples), max_options), dtype=torch.long)
    os_ = torch.zeros((len(examples), max_options), dtype=torch.long)
    oe = torch.zeros((len(examples), max_options), dtype=torch.long)
    om = torch.zeros((len(examples), max_options), dtype=torch.bool)
    oti = torch.zeros((len(examples), max_options), dtype=torch.long)
    mid = torch.zeros(len(examples), dtype=torch.long)
    perm = torch.zeros((len(examples), max_options), dtype=torch.long)
    di = torch.zeros(len(examples), dtype=torch.long)
    for r, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[r, :n] = torch.tensor(e["input_ids"])
        am[r, :n] = 1
        di[r] = e["decision_index"]
        k = len(e["option_index"])
        if k > max_options:
            raise ValueError(f"{k} options exceeds max_options={max_options}")
        oi[r, :k] = torch.tensor(e["option_index"])
        sp = e.get("option_span")
        if sp:
            os_[r, :k] = torch.tensor([a for a, _ in sp])
            oe[r, :k] = torch.tensor([b for _, b in sp])
        om[r, :k] = True
        if e.get("option_perm"):
            perm[r, :k] = torch.tensor(e["option_perm"])
        if e.get("mode"):
            mid[r] = MODES.index(e["mode"])
        if option_tokens and e.get("options"):
            oti[r, :k] = torch.tensor(option_first_token_ids(tokenizer, e["options"]))
    return {k: v.to(device) for k, v in
            {"input_ids": ids, "attention_mask": am, "decision_index": di,
             "option_index": oi, "option_span_start": os_, "option_span_end": oe,
             "option_token_ids": oti, "option_mask": om, "mode_id": mid,
             "option_perm": perm}.items()}


def gold_tensor(cases: Sequence[Case], keys: Sequence[str], max_options: int,
                device: str | torch.device = "cpu") -> torch.Tensor:
    g = torch.zeros((len(cases), max_options))
    for r, (c, k) in enumerate(zip(cases, keys)):
        v = c.gold[k]
        g[r, :len(v)] = torch.tensor(v)
    return g.to(device)


def iter_questions(cases: Sequence[Case]):
    """Flatten cases into (case, question) pairs. One question is one example;
    packing several questions onto one encoded state is an ARCH option and needs
    a 4-D attention mask so the questions cannot read each other."""
    for c in cases:
        for q in c.questions:
            yield c, q


def unpermute_logits(logits: torch.Tensor, option_perm: torch.Tensor,
                     option_mask: torch.Tensor) -> torch.Tensor:
    """Map logits from PRESENTED order back to canonical option order.

    Everything downstream -- the loss against gold, Prediction.choice(),
    Prediction.score()'s sum(i*p_i) -- indexes `q.options`. If a permuted
    presentation reached them unmapped, `score` would average POSITIONS rather
    than levels and the gold would be compared against the wrong option. This is
    the single point where the permutation is undone.
    """
    k = logits.shape[1]
    # Padded slots carry option_perm = 0, so a plain scatter sends every one of
    # them to canonical index 0 and overwrites a real option. Send them to a
    # scratch column instead and drop it.
    idx = torch.where(option_mask, option_perm,
                      torch.full_like(option_perm, k))
    src = torch.where(option_mask, logits, torch.full_like(logits, float("-inf")))
    out = torch.full((logits.shape[0], k + 1), float("-inf"),
                     dtype=logits.dtype, device=logits.device)
    out.scatter_(1, idx, src)
    return out[:, :k]
