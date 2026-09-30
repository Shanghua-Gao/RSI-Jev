"""Image states read once: `score_image_questions_cached` against the uncached path.

The cached path runs the shared prefix -- the state with its image tokens -- once
and continues every question from its cache. The uncached path (`score_image_
questions`, itself pinned to the offline encoder in test_vision_model.py) reads
the whole sequence per question. At fp32 on CPU the two must agree to float
noise (~1e-6), on the same tiny random Qwen3.5 as test_vision_model.py.

The trap is M-RoPE: after an image, text positions continue from the image's
grid extent, not from its token count. `test_the_position_trap_is_live` checks
the fixtures really exercise that, so the equality tests are not vacuous.

    python -m pytest tests/test_image_cache.py -q
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

pytest.importorskip("PIL")
pytest.importorskip("torchvision")
from PIL import Image                                               # noqa: E402

from test_vision_model import QS, _images, parts                    # noqa: E402,F401

from rsijev.vision import encode_vision_question, expand_state, mrope_position_ids  # noqa: E402
from serve.infer import (DocCache, score_image_questions,           # noqa: E402
                         score_image_questions_cached)

ATOL = 1e-5

STATES = [
    ("Left: <image>\nRight: <image>\nA note about both pictures.", 2),
    ("A short note.", 1),                                    # no marker: image first
    ("Screenshot of the settings page: <image> The user says it froze.", 1),
    ("Receipt photo <image>", 1),                            # the image ends the state
]


def _imgs(n):
    return _images()[:n] if n <= 2 else _images() * 2


def _uncached(p, state, images, qs=QS, **kw):
    tok, prep, vm, _, enc = p
    return score_image_questions(vm, tok, prep, state, images, qs, enc, max_options=8,
                                 device="cpu", **kw)


def _cached(p, state, images, qs=QS, **kw):
    tok, prep, vm, _, enc = p
    kw.setdefault("min_saved_tokens", 0)
    kw.setdefault("doc_cache", False)
    return score_image_questions_cached(vm, tok, prep, state, images, qs, enc, max_options=8,
                                        device="cpu", **kw)


def _worst(a, b):
    return max(max(abs(x - y) for x, y in zip(pa.probs, pb.probs)) for pa, pb in zip(a, b))


def _argmaxes(preds):
    return [max(range(len(p.probs)), key=p.probs.__getitem__) for p in preds]


def test_the_position_trap_is_live(parts):
    """After an image, the text's M-RoPE position is not its token index."""
    tok, prep, vm, _, enc = parts
    images = _imgs(2)
    e = encode_vision_question(tok, prep, STATES[0][0], images, QS[0], enc)
    ids = torch.tensor([e["input_ids"]])
    pos = mrope_position_ids(ids, torch.ones_like(ids), prep(images)[1], vm.image_token_id)
    assert int(pos[0, 0, -1]) != ids.shape[1] - 1


@pytest.mark.parametrize("state,n", STATES)
@pytest.mark.parametrize("batch_size", [1, 16])
def test_read_once_equals_reading_per_question(parts, state, n, batch_size):
    images = _imgs(n)
    ref, ref_tokens = _uncached(parts, state, images, batch_size=batch_size)
    got, tokens = _cached(parts, state, images, batch_size=batch_size)
    assert tokens < ref_tokens                                # the prefix really was shared
    assert _argmaxes(got) == _argmaxes(ref)
    assert _worst(got, ref) < ATOL, _worst(got, ref)


def test_below_the_threshold_it_reads_per_question(parts):
    state, n = STATES[0]
    ref = _uncached(parts, state, _imgs(n))
    got = _cached(parts, state, _imgs(n), min_saved_tokens=10**9)
    assert got[1] == ref[1]
    assert [p.probs for p in got[0]] == [p.probs for p in ref[0]]


def test_one_question_reads_per_question(parts):
    state, n = STATES[2]
    ref = _uncached(parts, state, _imgs(n), qs=QS[:1])
    got = _cached(parts, state, _imgs(n), qs=QS[:1])
    assert [p.probs for p in got[0]] == [p.probs for p in ref[0]]


def _counting(vm):
    calls = []
    orig = vm.image_embeds

    def wrapped(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    return calls, wrapped


@pytest.mark.parametrize("holdback", [0, 8])
def test_document_cache_repeat_reads_nothing_new(parts, monkeypatch, holdback):
    tok, prep, vm, _, enc = parts
    calls, wrapped = _counting(vm)
    monkeypatch.setattr(vm, "image_embeds", wrapped)
    c = DocCache(max_entries=8, max_bytes=1 << 30, holdback=holdback)
    for state, n in STATES:
        images = _imgs(n)
        ref = _uncached(parts, state, images)[0]
        first = _cached(parts, state, images, doc_cache=c)[0]
        before = (dict(c.stats), len(calls))
        again = _cached(parts, state, images, doc_cache=c)[0]
        assert c.stats["hit"] == before[0]["hit"] + 1
        assert c.stats["read_tokens"] == before[0]["read_tokens"]
        # A hit runs the vision tower only if the held-back tail reaches into an image.
        ids = tok(f"{expand_state(state, prep(images)[2])}\n\n",
                  add_special_tokens=False)["input_ids"]
        tail_in_image = holdback > 0 and vm.image_token_id in ids[-holdback:]
        assert len(calls) == before[1] + (1 if tail_in_image else 0)
        for got in (first, again):
            assert _argmaxes(got) == _argmaxes(ref)
            assert _worst(got, ref) < ATOL


def test_document_cache_tells_images_apart(parts):
    """Two same-sized images tokenize to the same ids; the cache must not mix them."""
    red = Image.new("RGB", (96, 64), (220, 20, 20))
    green = Image.new("RGB", (96, 64), (20, 200, 20))
    green.paste((0, 0, 0), (10, 10, 40, 40))
    c = DocCache(max_entries=8, max_bytes=1 << 30)
    state = "<image> A picture from the camera roll."
    _cached(parts, state, [red], doc_cache=c)
    got = _cached(parts, state, [green], doc_cache=c)[0]
    ref = _uncached(parts, state, [green])[0]
    assert c.stats["hit"] == 0 and c.stats["miss"] == 2
    assert _worst(got, ref) < ATOL
    assert _worst(got, _uncached(parts, state, [red])[0]) > 1e-4


def test_document_cache_extends_a_growing_image_state(parts):
    c = DocCache(max_entries=8, max_bytes=1 << 30)
    images = _imgs(1)
    base = "Screenshot: <image> Log so far: opened settings."
    grown = base + " Clicked save. A dialog appeared saying the disk is full."
    _cached(parts, base, images, doc_cache=c)
    got = _cached(parts, grown, images, doc_cache=c)[0]
    assert c.stats["extend"] == 1
    ref = _uncached(parts, grown, images)[0]
    assert _argmaxes(got) == _argmaxes(ref)
    assert _worst(got, ref) < ATOL


def test_text_and_image_entries_do_not_mix(parts):
    """A text request never continues from an image entry, nor the reverse."""
    tok, prep, vm, _, enc = parts
    from serve.infer import score_questions_cached
    c = DocCache(max_entries=8, max_bytes=1 << 30)
    state = "<image> A note that is long enough to be cached."
    _cached(parts, state, _imgs(1), doc_cache=c)
    text_enc = dataclasses.replace(enc)
    score_questions_cached(vm, tok, "A note that is long enough to be cached. More.", QS,
                           text_enc, max_options=8, device="cpu", doc_cache=c)
    assert c.stats["hit"] == 0 and c.stats["extend"] == 0


def test_token_index_positions_would_be_caught(parts, monkeypatch):
    """The negative control: continue the questions at token-index positions (what
    a text-only continuation would do) and the answers move far past float noise."""
    import rsijev.vision as V
    state, n = STATES[2]
    ref = _uncached(parts, state, _imgs(n))[0]
    real = V.mrope_position_ids

    def token_index(ids, am, grid, tid, **k):
        pos = real(ids, am, grid, tid).clone()
        if ids.shape[0] > 1:                       # the per-chunk suffix positions only
            for r, n_real in enumerate(am.sum(1).tolist()):
                pos[:, r, :n_real] = torch.arange(n_real)
        return pos
    monkeypatch.setattr(V, "mrope_position_ids", token_index)
    got = _cached(parts, state, _imgs(n))[0]
    assert _worst(got, ref) > 10 * ATOL           # read-once itself is ~3e-8 here
