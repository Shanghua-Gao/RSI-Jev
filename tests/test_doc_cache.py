"""The cross-request document cache must answer exactly what a fresh read answers.

A repeated state reuses its cache; a state that grows runs only its new tail on a
copy of the cached one. Both are exact under a causal mask, so the bar is the one
tests/test_prefix_cache.py sets: every argmax equal, probabilities within
reassociation noise. The fast tests pin the bookkeeping (keys, bounds, never
mutating an entry) on a stand-in model; the slow ones run real weights.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_doc_cache.py -q -m slow
    RSIJEV_TEST_DEVICE=cuda python -m pytest tests/test_doc_cache.py -q -m slow   # fp32, fused kernels
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve import infer                                    # noqa: E402
from serve.infer import DocCache, score_questions, score_questions_cached  # noqa: E402


# --------------------------------------------------------------------------- fakes

class _Layer:
    def __init__(self, keys):
        self.keys = keys

    def reorder_cache(self, idx):
        self.keys = self.keys.index_select(0, idx)


class _Cache:
    """Holds the ids it has read as one float per token."""

    def __init__(self, keys):
        self.layers = [_Layer(keys)]

    def reorder_cache(self, idx):
        for layer in self.layers:
            layer.reorder_cache(idx)


class _Model(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.full((4,), scale))
        self.reads = []                                   # token counts actually run

    def encode_prefix(self, ids):
        self.reads.append(("full", ids.shape[1]))
        return _Cache(ids.to(torch.float32))

    def extend_prefix(self, cache, ids, start):
        assert cache.layers[0].keys.shape[1] == start
        self.reads.append(("tail", ids.shape[1]))
        cache.layers[0].keys = torch.cat([cache.layers[0].keys, ids.to(torch.float32)], dim=1)
        return cache


def _ids(cache):
    return cache.layers[0].keys[0].long().tolist()


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("RSIJEV_DOC_CACHE", raising=False)
    assert infer.default_doc_cache() is None
    monkeypatch.setenv("RSIJEV_DOC_CACHE", "0")
    assert infer.default_doc_cache() is None
    monkeypatch.setenv("RSIJEV_DOC_CACHE", "1")
    assert isinstance(infer.default_doc_cache(), DocCache)


def test_env_bounds(monkeypatch):
    monkeypatch.setenv("RSIJEV_DOC_CACHE_ENTRIES", "3")
    monkeypatch.setenv("RSIJEV_DOC_CACHE_MB", "1.5")
    monkeypatch.setenv("RSIJEV_DOC_CACHE_HOLDBACK", "5")
    c = DocCache()
    assert (c.max_entries, c.max_bytes, c.holdback) == (3, int(1.5 * 2**20), 5)


def test_repeat_extend_and_fallback():
    m, c = _Model(), DocCache(max_entries=8, max_bytes=1 << 30)
    a = c.get(m, [1, 2, 3], "cpu")
    assert _ids(a) == [1, 2, 3] and m.reads == [("full", 3)]
    assert c.get(m, [1, 2, 3], "cpu") is a                 # repeat: no pass at all
    assert m.reads == [("full", 3)]
    b = c.get(m, [1, 2, 3, 4, 5], "cpu")                   # growth: the tail only
    assert _ids(b) == [1, 2, 3, 4, 5] and m.reads[-1] == ("tail", 2)
    assert _ids(a) == [1, 2, 3], "extending must not touch the entry it started from"
    c.get(m, [1, 2, 3, 4, 5, 6], "cpu")                    # longest cached prefix wins
    assert m.reads[-1] == ("tail", 1)
    c.get(m, [1, 2, 9, 4], "cpu")                          # boundary re-merged: full read
    assert m.reads[-1] == ("full", 4)
    assert c.stats["hit"] == 1 and c.stats["extend"] == 2 and c.stats["miss"] == 2


def test_keyed_on_weights_not_just_ids():
    c = DocCache(max_entries=8, max_bytes=1 << 30)
    m1, m2 = _Model(1.0), _Model(2.0)
    c.get(m1, [1, 2, 3], "cpu")
    c.get(m2, [1, 2, 3], "cpu")
    assert m2.reads == [("full", 3)], "another model's cache must never be reused"
    c.get(m2, [1, 2, 3, 4], "cpu")
    assert m2.reads[-1] == ("tail", 1)


def test_bounded_in_entries_and_bytes():
    m = _Model()
    c = DocCache(max_entries=2, max_bytes=1 << 30)
    for n in (3, 4, 5):
        c.get(m, list(range(100, 100 + n)) + [n], "cpu")
    assert len(c) == 2 and c.stats["evicted"] == 1
    per_token = 4                                          # one float32 per token
    c = DocCache(max_entries=100, max_bytes=10 * per_token)
    c.get(m, list(range(6)), "cpu")
    c.get(m, list(range(50, 56)), "cpu")                   # 6 + 6 tokens > 10
    assert len(c) == 1 and c.nbytes == 6 * per_token
    c.get(m, list(range(200, 220)), "cpu")                 # larger than the whole budget
    assert len(c) == 1, "an entry over the byte budget is used once, not stored"


# --------------------------------------------------------------------------- real weights

MODEL = "Qwen/Qwen3.5-0.8B-Base"
DEVICE = os.environ.get("RSIJEV_TEST_DEVICE", "cpu")


@pytest.fixture(scope="module")
def model_and_tok():
    from rsijev.arch import ArchConfig, DecisionModel
    from rsijev.train import seed_everything
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    hidden = (getattr(lm.config, "text_config", None) or lm.config).hidden_size
    seed_everything(17)
    model = DecisionModel(lm.model, hidden,
                          ArchConfig(readout="option_xattn", readout_layer=-1,
                                     max_options=80, freeze_base=True,
                                     option_pool="mean", residual=False))
    model.scorer.to(torch.float32)
    return model.to(DEVICE).eval(), tok


def _agree(a, b, where):
    worst = 0.0
    assert len(a) == len(b)
    for x, y in zip(a, b):
        worst = max(worst, max(abs(p - q) for p, q in zip(x.probs, y.probs)))
        assert (max(range(len(x.probs)), key=x.probs.__getitem__)
                == max(range(len(y.probs)), key=y.probs.__getitem__)), f"{where}: argmax moved"
    assert worst < 1e-3, f"{where}: probabilities drifted by {worst:.2e}"
    return worst


def _snapshot(cache):
    out = []
    for layer in cache.layers:
        for attr in ("keys", "values", "conv_states", "recurrent_states"):
            t = getattr(layer, attr, None)
            if isinstance(t, torch.Tensor):
                out.append(t.clone())
    return out


def _agent_states():
    from serve.wire import state_to_text
    turns = [{"role": "user", "content": "My order #48213 arrived damaged and two days late."},
             {"role": "assistant", "content": "Sorry about that. Can you describe the damage?"},
             {"role": "user", "content": "The box was crushed and the lamp base is cracked."},
             {"role": "assistant", "content": "Thanks. I can offer a replacement or a refund."},
             {"role": "user", "content": "I was also charged twice, so refund both please."}]
    return [state_to_text(turns[:k]) for k in range(2, len(turns) + 1)]


def _questions():
    from serve.wire import parse_questions
    return parse_questions({
        "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Payments and refunds", "shipping": "Delivery",
                              "technical": "Bugs"}},
        "urgency": {"type": "score", "instructions": "How urgent is it?",
                    "criteria": ["Routine", "Urgent", "Emergency"]}})


@pytest.mark.slow
def test_repeated_state_matches_a_fresh_read(model_and_tok):
    from rsijev.encode import EncodeConfig
    model, tok = model_and_tok
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
    state, qs = _agent_states()[-1], list(_questions())
    fresh, _ = score_questions(model, tok, state, qs, enc, device=DEVICE)
    c = DocCache(max_entries=4, max_bytes=1 << 30)
    first, _ = score_questions_cached(model, tok, state, qs, enc, device=DEVICE, doc_cache=c)
    (entry, _), = c._entries.values()
    before = _snapshot(entry)
    again, _ = score_questions_cached(model, tok, state, qs, enc, device=DEVICE, doc_cache=c)
    single, _ = score_questions_cached(model, tok, state, qs[:1], enc, device=DEVICE, doc_cache=c)
    assert c.stats == {**c.stats, "miss": 1, "hit": 2, "extend": 0}
    w = max(_agree(fresh, first, "first"), _agree(fresh, again, "repeat"),
            _agree(fresh[:1], single, "single question"))
    assert all(torch.equal(x, y) for x, y in zip(before, _snapshot(entry))), \
        "answering from an entry must not change it"
    print(f"\nrepeat: worst |dprob| {w:.2e}")


@pytest.mark.slow
def test_growing_state_matches_a_fresh_read(model_and_tok):
    from rsijev.encode import EncodeConfig
    model, tok = model_and_tok
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
    qs = list(_questions())
    c = DocCache(max_entries=8, max_bytes=1 << 30)
    worst = 0.0
    for step, state in enumerate(_agent_states()):
        fresh, _ = score_questions(model, tok, state, qs, enc, device=DEVICE)
        got, _ = score_questions_cached(model, tok, state, qs, enc, device=DEVICE, doc_cache=c)
        worst = max(worst, _agree(fresh, got, f"step {step}"))
    assert c.stats["miss"] == 1, "every later turn should extend the one before"
    assert c.stats["extend"] == len(_agent_states()) - 1
    print(f"\ngrowth: {c.stats}, worst |dprob| {worst:.2e}")


@pytest.mark.slow
def test_boundary_re_merge_falls_back_to_a_full_read(model_and_tok):
    """With no holdback, 'deliv' + 'ered' re-tokenizes as one word: the old ids are
    no longer a prefix of the new ones, and the cache must not be extended."""
    from rsijev.encode import EncodeConfig
    model, tok = model_and_tok
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
    qs = list(_questions())
    a = "Order #48213 was deliv"
    b = a + "ered late and the lamp base is cracked."
    ia = tok(a + "\n\n", add_special_tokens=False)["input_ids"]
    ib = tok(b + "\n\n", add_special_tokens=False)["input_ids"]
    assert ib[:len(ia)] != ia, "the premise: the boundary re-merged"
    c = DocCache(max_entries=8, max_bytes=1 << 30, holdback=0)
    score_questions_cached(model, tok, a, qs, enc, device=DEVICE, doc_cache=c)
    got, _ = score_questions_cached(model, tok, b, qs, enc, device=DEVICE, doc_cache=c)
    fresh, _ = score_questions(model, tok, b, qs, enc, device=DEVICE)
    assert c.stats["miss"] == 2 and c.stats["extend"] == 0
    _agree(fresh, got, "re-merged")
