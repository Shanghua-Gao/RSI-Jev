"""The cached path must answer exactly what the uncached path answers.

Encoding the state once and continuing each question on its cache is an *exact*
optimisation under a causal mask, not an approximation — so the bar is
agreement, not closeness. Floating point still reassociates (different GEMM
shapes, a different chunking of the DeltaNet scan), so probabilities are
compared with a tolerance while every argmax must match.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_prefix_cache.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.arch import ArchConfig, DecisionModel          # noqa: E402
from rsijev.encode import EncodeConfig                     # noqa: E402
from rsijev.targets import load_typed_decisions            # noqa: E402
from rsijev.train import seed_everything                   # noqa: E402
from serve.infer import score_questions, score_questions_cached  # noqa: E402
from serve.wire import state_to_text                       # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B-Base"


@pytest.fixture(scope="module")
def model_and_tok():
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
    return model.eval(), tok


@pytest.mark.slow
def test_cached_matches_uncached(model_and_tok):
    model, tok = model_and_tok
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
    cases = [c for c in load_typed_decisions("test")[:6] if len(c.questions) > 1]
    assert cases, "need multi-question cases for this to mean anything"

    worst = 0.0
    compared = 0
    for c in cases:
        state = state_to_text(c.state)
        qs = list(c.questions)
        plain, plain_tokens = score_questions(model, tok, state, qs, enc, device="cpu")
        # min_saved_tokens=0 forces the cached path on. Without it the heuristic
        # would decline these short states and this would silently test nothing.
        cached, cached_tokens = score_questions_cached(model, tok, state, qs, enc,
                                                      device="cpu", min_saved_tokens=0)

        assert len(plain) == len(cached) == len(qs)
        for q, a, b in zip(qs, plain, cached):
            assert len(a.probs) == len(b.probs)
            worst = max(worst, max(abs(x - y) for x, y in zip(a.probs, b.probs)))
            assert (max(range(len(a.probs)), key=a.probs.__getitem__)
                    == max(range(len(b.probs)), key=b.probs.__getitem__)), \
                f"{c.case_id}/{q.key}: the cached path chose a different option"
            compared += 1
        # The prefix is counted once, so the cached path reports fewer tokens.
        assert cached_tokens < plain_tokens

    print(f"\n{compared} decisions, worst |dprob| {worst:.2e}")
    assert worst < 1e-3, f"probabilities drifted by {worst:.2e}; expected reassociation only"


@pytest.mark.slow
def test_heuristic_declines_a_prefix_too_small_to_pay(model_and_tok):
    """Caching is a real loss on short documents: two sequential passes and a
    replicated cache cost more than they save. The default threshold declines
    them, which shows up as the prefix being counted once per question again."""
    model, tok = model_and_tok
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
    c = next(c for c in load_typed_decisions("test")[:6] if len(c.questions) > 1)
    state = state_to_text(c.state)
    qs = list(c.questions)
    _, plain = score_questions(model, tok, state, qs, enc, device="cpu")
    _, declined = score_questions_cached(model, tok, state, qs, enc, device="cpu")
    _, forced = score_questions_cached(model, tok, state, qs, enc, device="cpu",
                                       min_saved_tokens=0)
    assert declined == plain, "a short state should fall back to the uncached path"
    assert forced < plain, "forcing the cache should stop re-counting the prefix"


@pytest.mark.slow
def test_falls_back_when_the_prefix_is_not_shared(model_and_tok):
    """No state means no shared prefix, and the caller still gets answers."""
    model, tok = model_and_tok
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
    c = next(c for c in load_typed_decisions("test")[:6] if len(c.questions) > 1)
    preds, _ = score_questions_cached(model, tok, "   ", list(c.questions), enc, device="cpu")
    assert len(preds) == len(c.questions)


def test_the_gate_is_chosen_for_the_machine(monkeypatch):
    """The break-even prefix length is a property of the hardware and kernel
    stack, not of the model: with fused kernels a tower pass is cheap and the
    cache has to amortise a large fixed cost, without them the tower pass IS the
    cost and caching pays far sooner. Shipping one constant meant a GB10 refused
    a measured 2.0x speed-up at 404 tokens."""
    import importlib.util

    from serve import infer

    monkeypatch.delenv("RSIJEV_MIN_SAVED_TOKENS", raising=False)
    real = importlib.util.find_spec

    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a, **k: object() if n == "fla" else real(n, *a, **k))
    assert infer.default_min_saved_tokens() == infer.MIN_SAVED_TOKENS_FUSED

    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda n, *a, **k: None if n == "fla" else real(n, *a, **k))
    assert infer.default_min_saved_tokens() == infer.MIN_SAVED_TOKENS_FALLBACK
    assert infer.MIN_SAVED_TOKENS_FALLBACK < infer.MIN_SAVED_TOKENS_FUSED

    monkeypatch.setenv("RSIJEV_MIN_SAVED_TOKENS", "0")
    assert infer.default_min_saved_tokens() == 0
