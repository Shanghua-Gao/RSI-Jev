"""The torch-free half of serving: encoding a request and choosing how it runs.

Moved out of serve/infer.py (which re-exports every name here) so that a backend without
torch -- the MLX path, rsijev/mlx -- plans requests with exactly the code the PyTorch path
uses. Nothing here touches a model.
"""
from __future__ import annotations

import json
import os
from typing import Sequence

from rsijev.encode import EncodeConfig, encode_question, option_permutation, own_token_spans, render


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


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
        if enc.option_pool_own_tokens:
            spans = own_token_spans(tokenizer, spans, [q.options[i] for i in order])
        if len(row) > enc.max_length:              # truncation: let the original do it
            rows.append(encode_question(tokenizer, state, q, enc))
            continue
        rows.append({"input_ids": row, "option_index": [e - 1 for _, e in spans],
                     "option_span": spans, "decision_index": len(row) - 1,
                     "options": [q.options[i] for i in order],
                     "option_perm": order, "mode": q.mode, "state_cut": 0, "options_cut": 0})
    return rows, list(lead_ids)


def truncation_report(encoded: Sequence[dict], enc: EncodeConfig) -> dict | None:
    """What the encoder cut from one request, for the response's `truncated` field,
    or None when this model is not served with the long-context encoder (the field
    is then left out, as it always was). The state is one text, so its count is the
    most any question lost of it; description and question counts add up over the
    questions. All 0 when nothing was cut."""
    if getattr(enc, "truncate", "left") != "middle":
        return None
    return {"state_tokens_omitted": max((int(e.get("state_cut", 0)) for e in encoded), default=0),
            "option_desc_tokens_omitted": sum(int(e.get("options_cut", 0)) for e in encoded),
            "question_tokens_omitted": sum(int(e.get("question_cut", 0)) for e in encoded),
            "max_length": int(enc.max_length)}


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


def reports_depth(model) -> bool:
    """Whether responses carry the exit depth: models with aux exits only (every other
    release's response keeps its shape)."""
    return bool(getattr(getattr(model, "cfg", None), "aux_exits", ()))


def record_confidence(model, plan: dict, preds) -> None:
    """plan["confidence"]: each question's calibrated top-1 probability at the exit that
    answered it (the value the cascade compares with tau), for models with aux exits."""
    if reports_depth(model):
        plan["confidence"] = [float(max(p.probs)) for p in preds]


def fixed_depth(model, plan: dict) -> None:
    """Record the exit every question of `plan` used on a fixed-exit path."""
    if reports_depth(model):
        plan["depth"] = [int(model.cfg.exit_layer)] * len(plan["encoded"])


def adaptive_applies(model, plan: dict) -> bool:
    """Whether a text plan runs with adaptive exit (score_adaptive).

    Needs a policy (serve.release.adaptive_policy) and model.adaptive_mode:
      auto  multi-question requests (read in full, read once or through the document
            cache); a single question takes the fixed exit. GB10 five-repeat check:
            adaptive 6-9% faster on 8- and 32-question batches, 1-7% slower
            on single questions
      on    every text plan, one question included
      off   none (load_release then loads no policy at all)
    Pooled (micro-batched) rows and image requests never come here: see
    serve/batcher.ModelRunner._pooled and score_image_planned."""
    effort = plan.get("effort")
    if effort is not None:
        # serve/effort.py: high is the fixed exit; low, medium and auto run the staged
        # path on every text plan, one question included, with the effort's own policy
        return effort != "high" and plan.get("path") in ("plain", "cached", "doc")
    if getattr(model, "adaptive_policy", None) is None:
        return False
    mode = getattr(model, "adaptive_mode", "auto")
    if mode == "off" or plan.get("path") not in ("plain", "cached", "doc"):
        return False
    return mode == "on" or len(plan["encoded"]) > 1
