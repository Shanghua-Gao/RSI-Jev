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


def render(state: str, q: Question, cfg: EncodeConfig,
           order: Sequence[int] | None = None) -> tuple[str, list[str]]:
    """Return the prompt prefix and the per-option blocks, in PRESENTED order."""
    order = list(range(len(q.options))) if order is None else list(order)
    blocks = [
        f"- {q.options[i]}: {q.criteria[q.options[i]]}"
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
    """Token ids plus the positions the readout needs. No padding here."""
    order = option_permutation(q, cfg, rng)
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

    if len(ids) > cfg.max_length:
        # Truncate the STATE, never the options or the cue: dropping an option
        # silently changes the task, and dropping the cue moves the readout.
        overflow = len(ids) - cfg.max_length
        head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
        if overflow >= len(head_ids):
            raise ValueError(f"cannot fit {q.key}: options alone exceed max_length")
        keep = head_ids[overflow:]
        ids = keep + ids[len(head_ids):]
        option_index = [i - overflow for i in option_index]
        spans = [(a - overflow, b - overflow) for a, b in spans]
        decision_index -= overflow
    return {"input_ids": ids, "option_index": option_index,
            "option_span": spans, "decision_index": decision_index,
            "options": [q.options[i] for i in order],
            "option_perm": order, "mode": q.mode}


def collate(tokenizer, examples: Sequence[dict], max_options: int,
            device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Right-pad. Positions are explicit, so padding side cannot shift them."""
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
        if e.get("options"):
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
