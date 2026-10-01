"""Early-exit serving (the 4B releases: Qwen3.5-4B-Base cut at layer 20 of 32), on a
tiny random Qwen3.5 (text + vision), on CPU in seconds.

An exit model runs only its first `exit_layer` decoder layers, then the text
model's final norm, and the head reads that state. What is pinned here:

  * exit at k IS the full model truncated at k: the same logits as a tower built
    with k layers and the same weights, for text and image states; with
    exit_norm=False, the same as a readout_layer=k tap on the full tower;
  * layers >= k never run, in the scoring pass, the prefix cache, the document
    cache or an image prefix, and the module is whole afterwards;
  * read-once is exact for exit models: text and image states read once (and
    through the document cache) give what reading per question gives;
  * the served image path equals the offline encoder for exit models;
  * load_release builds all of this from meta.json alone, and a release without
    exit_layer / max_length / truncate loads exactly as before.

Needs the Qwen3.5 tokenizer and image processor (config files only, no weights).

    python -m pytest tests/test_big4b_exit.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("PIL")
pytest.importorskip("torchvision")
from PIL import Image                                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve.runtime import keep_fused_kernels_off                   # noqa: E402

keep_fused_kernels_off("cpu")

from rsijev.arch import ArchConfig, DecisionModel, _text_model     # noqa: E402
from rsijev.contract import Question                               # noqa: E402
from rsijev.encode import EncodeConfig, unpermute_logits          # noqa: E402
from rsijev.vision import (IMAGE_PAD, PINNED_REVISIONS, ImagePrep,  # noqa: E402
                           VisionConfig, VisionDecisionModel, encode_vision_question,
                           vision_collate)
from serve.infer import (DocCache, score_image_questions,          # noqa: E402
                         score_image_questions_cached, score_questions,
                         score_questions_cached)

BASE4 = "Qwen/Qwen3.5-4B-Base"
N_LAYERS, H = 8, 64
EXITS = [4, 6]          # 4 follows a full-attention layer (as 16/20/24 of 32 do); 6 a linear one
ATOL = 1e-5

QS = [Question("a", "noul", "Is the square red?", ("false", "true"),
               {"false": "No.", "true": "Yes."}),
      Question("b", "choice", "Which colour is the square, and what shape is it really?",
               ("red", "green", "blue"), {"red": "Red.", "green": "Green.", "blue": "Blue."}),
      Question("c", "score", "How bright?", ("0", "1", "2", "3"),
               {"0": "Dark", "1": "Dim", "2": "Bright", "3": "Glaring"})]
TEXT_STATE = "The report says revenue grew 12% while costs fell. " * 12
IMAGE_STATES = [("Left: <image>\nRight: <image>\nA note about both pictures.", 2),
                ("A short note.", 1),                                    # no marker: image first
                ("Screenshot of the settings page: <image> The user says it froze.", 1)]


def _text_config(n_layers: int, vocab: int):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    tc = Qwen3_5TextConfig(vocab_size=vocab, hidden_size=H, intermediate_size=128,
                           num_hidden_layers=n_layers, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=16, linear_value_head_dim=16,
                           layer_types=[("full_attention" if (i + 1) % 4 == 0 else "linear_attention")
                                        for i in range(n_layers)],
                           rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                                            "partial_rotary_factor": 0.25,
                                            "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    tc._attn_implementation = "sdpa"
    return tc


def _tower(vocab: int, seed: int = 0):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    torch.manual_seed(seed)
    tc = _text_config(N_LAYERS, vocab)
    return Qwen3_5TextModel(tc).eval(), tc


def _truncated(tower, k: int):
    """A genuinely k-layer Qwen3.5 holding the full tower's first k layers, its
    embedding and its final norm: the reference an exit at k must equal."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    tc = _text_config(k, tower.config.vocab_size)
    t = Qwen3_5TextModel(tc).eval()
    sd = {n: v for n, v in tower.state_dict().items()
          if not n.startswith("layers.") or int(n.split(".")[1]) < k}
    t.load_state_dict(sd, strict=True)
    return t


def _visual(seed: int = 1):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    torch.manual_seed(seed)
    vc = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                             out_hidden_size=H, patch_size=16, spatial_merge_size=2,
                             temporal_patch_size=2, in_channels=3, num_position_embeddings=64)
    vc._attn_implementation = "sdpa"
    return Qwen3_5VisionModel(vc).eval()


def _arch(**kw):
    return ArchConfig(max_options=8, freeze_base=True, readout="option_xattn",
                      xattn_combine="mlp", xattn_mlp_hidden=16, **kw)


@pytest.fixture(scope="module")
def tok_prep():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE4, revision=PINNED_REVISIONS[BASE4])
        prep = ImagePrep(BASE4, VisionConfig(image_token_budget=64, min_tokens_per_image=16))
    except Exception as e:                                      # offline, no cache
        pytest.skip(f"Qwen3.5-4B tokenizer / image processor unavailable: {e}")
    return tok, prep


@pytest.fixture(scope="module")
def parts(tok_prep):
    tok, prep = tok_prep
    tower, _ = _tower(len(tok))
    visual = _visual()
    pad = tok.convert_tokens_to_ids(IMAGE_PAD)
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical",
                       max_length=2048 + 64)
    models = {}
    for k in EXITS:
        torch.manual_seed(5)
        ex = VisionDecisionModel(tower, H, _arch(readout_layer=-1, exit_layer=k), visual=visual,
                                 image_token_id=pad).eval()
        tr = VisionDecisionModel(_truncated(tower, k), H, _arch(readout_layer=-1), visual=visual,
                                 image_token_id=pad).eval()
        tr.scorer.load_state_dict(ex.scorer.state_dict())
        models[k] = (ex, tr)
    return tok, prep, tower, visual, enc, models


def _images():
    a = Image.new("RGB", (96, 64), (220, 20, 20))
    b = Image.new("RGB", (64, 64), (20, 20, 220))
    b.paste((250, 250, 250), (16, 16, 48, 48))
    return [a, b]


def _imgs(n):
    return _images()[:n]


def _worst(a, b):
    return max(max(abs(x - y) for x, y in zip(pa.probs, pb.probs)) for pa, pb in zip(a, b))


def _argmaxes(preds):
    return [max(range(len(p.probs)), key=p.probs.__getitem__) for p in preds]


@torch.no_grad()
def _offline(tok, prep, model, state, images, q, enc):
    """The gated offline image path: one question, its own ViT pass."""
    bt = vision_collate(tok, [encode_vision_question(tok, prep, state, images, q, enc)],
                        8, device="cpu")
    z = unpermute_logits(model(**bt).float(), bt["option_perm"], bt["option_mask"])
    return torch.softmax(z, -1)[0, :len(q.options)]


# ---------------------------------------------------------------------------
# exit at k == the full model truncated at k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", EXITS)
def test_exit_equals_the_model_truncated_at_k(parts, k):
    tok, prep, _, _, enc, models = parts
    ex, tr = models[k]
    for state, images in [(TEXT_STATE, []), *[(s, _imgs(n)) for s, n in IMAGE_STATES]]:
        rows = [encode_vision_question(tok, prep, state, images, q, enc) for q in QS]
        b = vision_collate(tok, rows, 8, device="cpu")
        with torch.no_grad():
            a, c = ex(**b), tr(**b)
        m = b["option_mask"]
        assert torch.equal(a[m], c[m]), (state[:20], float((a[m] - c[m]).abs().max()))


@pytest.mark.parametrize("k", EXITS)
def test_raw_exit_equals_a_tap_at_k_on_the_full_tower(parts, k):
    """exit_norm=False reads the un-normalised state: readout_layer=k on the full tower
    (the internal tests/test_early_exit.py, on the tiny tower)."""
    tok, prep, tower, visual, enc, _ = parts
    pad = tok.convert_tokens_to_ids(IMAGE_PAD)
    torch.manual_seed(5)
    tap = VisionDecisionModel(tower, H, _arch(readout_layer=k), visual=visual, image_token_id=pad).eval()
    raw = VisionDecisionModel(tower, H, _arch(readout_layer=-1, exit_layer=k, exit_norm=False),
                              visual=visual, image_token_id=pad).eval()
    raw.scorer.load_state_dict(tap.scorer.state_dict())
    for state, images in [(TEXT_STATE, []), (IMAGE_STATES[0][0], _imgs(2))]:
        b = vision_collate(tok, [encode_vision_question(tok, prep, state, images, q, enc)
                                 for q in QS], 8, device="cpu")
        with torch.no_grad():
            a, c = tap(**b), raw(**b)
        m = b["option_mask"]
        assert torch.equal(a[m], c[m]), float((a[m] - c[m]).abs().max())


def test_exit_settings_are_checked(parts):
    _, _, tower, _, _, _ = parts
    for bad in (dict(readout_layer=4, exit_layer=4), dict(readout_layer=-1, exit_layer=N_LAYERS),
                dict(readout_layer=-1, exit_layer=4, residual=True)):
        with pytest.raises(ValueError):
            DecisionModel(tower, H, _arch(**bad))


@pytest.mark.parametrize("k", EXITS)
def test_upper_layers_never_run_and_the_module_stays_whole(parts, k):
    tok, prep, tower, _, enc, models = parts
    ex, _ = models[k]
    tm = _text_model(ex.tower)
    calls = []
    hooks = [tm.layers[i].register_forward_hook(lambda *a, i=i: calls.append(i))
             for i in range(N_LAYERS)]
    keys = list(ex.state_dict().keys())
    try:
        score_questions(ex, tok, TEXT_STATE, QS, enc, max_options=8, device="cpu")
        score_questions_cached(ex, tok, TEXT_STATE, QS, enc, max_options=8, device="cpu",
                               min_saved_tokens=0, doc_cache=False)
        c = DocCache(max_entries=4, max_bytes=1 << 30, holdback=8)
        for st in (TEXT_STATE, TEXT_STATE + "Then it fell."):        # build, then extend
            score_questions_cached(ex, tok, st, QS, enc, max_options=8, device="cpu", doc_cache=c)
        assert c.stats["extend"] == 1
        score_image_questions_cached(ex, tok, prep, IMAGE_STATES[0][0], _imgs(2), QS, enc,
                                     max_options=8, device="cpu", min_saved_tokens=0,
                                     doc_cache=False)
    finally:
        for h in hooks:
            h.remove()
    assert calls and max(calls) == k - 1
    assert list(ex.state_dict().keys()) == keys
    assert len(tm.layers) == N_LAYERS and not isinstance(tm.norm, torch.nn.Identity)
    with torch.no_grad():   # a direct tower call outside the model still runs full depth
        out = ex.tower(input_ids=torch.tensor([[1, 2, 3]]), output_hidden_states=True)
    assert len(out.hidden_states) == N_LAYERS + 1


@pytest.mark.parametrize("k", EXITS)
def test_prefix_caches_hold_only_the_exit_layers(parts, k):
    tok, prep, _, _, _, models = parts
    ex, _ = models[k]
    ids = torch.tensor([tok("A state to keep.", add_special_tokens=False)["input_ids"]])
    cache = ex.encode_prefix(ids)
    assert len(cache.layers) == k and cache.has_previous_state()
    ex.extend_prefix(cache, ids, ids.shape[1])
    assert len(cache.layers) == k


# ---------------------------------------------------------------------------
# read once, document cache: exact for exit models
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", EXITS)
@pytest.mark.parametrize("batch_size", [1, 16])
def test_text_read_once_is_exact(parts, k, batch_size):
    tok, _, _, _, enc, models = parts
    ex, tr = models[k]
    kw = dict(max_options=8, device="cpu", batch_size=batch_size)
    ref, ref_tokens = score_questions(tr, tok, TEXT_STATE, QS, enc, **kw)
    plain, _ = score_questions(ex, tok, TEXT_STATE, QS, enc, **kw)
    got, tokens = score_questions_cached(ex, tok, TEXT_STATE, QS, enc, min_saved_tokens=0,
                                         doc_cache=False, **kw)
    assert tokens < ref_tokens                                      # the prefix really was shared
    assert [p.probs for p in plain] == [p.probs for p in ref]       # exit == truncated, bitwise
    assert _argmaxes(got) == _argmaxes(ref) and _worst(got, ref) < ATOL


@pytest.mark.parametrize("k", EXITS)
def test_text_document_cache_is_exact(parts, k):
    tok, _, _, _, enc, models = parts
    ex, tr = models[k]
    c = DocCache(max_entries=8, max_bytes=1 << 30, holdback=8)
    for st in (TEXT_STATE, TEXT_STATE, TEXT_STATE + "Margins held at 30%."):
        ref = score_questions(tr, tok, st, QS, enc, max_options=8, device="cpu")[0]
        got = score_questions_cached(ex, tok, st, QS, enc, max_options=8, device="cpu",
                                     doc_cache=c)[0]
        assert _argmaxes(got) == _argmaxes(ref) and _worst(got, ref) < ATOL
    assert c.stats["hit"] == 1 and c.stats["extend"] == 1


@pytest.mark.parametrize("k", EXITS)
def test_served_image_path_equals_the_offline_encoder(parts, k):
    tok, prep, _, _, enc, models = parts
    ex, tr = models[k]
    for state, n in IMAGE_STATES:
        preds, tokens = score_image_questions(ex, tok, prep, state, _imgs(n), QS, enc,
                                              max_options=8, device="cpu", batch_size=16)
        for q, p in zip(QS, preds):
            ref = _offline(tok, prep, ex, state, _imgs(n), q, enc)
            assert torch.allclose(torch.tensor(p.probs), ref, atol=ATOL), (q.key, p.probs, ref)
            trunc = _offline(tok, prep, tr, state, _imgs(n), q, enc)
            assert torch.equal(ref, trunc)
        assert tokens > sum(prep(_imgs(n))[2])


@pytest.mark.parametrize("k", EXITS)
@pytest.mark.parametrize("batch_size", [1, 16])
def test_image_read_once_is_exact(parts, k, batch_size):
    tok, prep, _, _, enc, models = parts
    ex, _ = models[k]
    for state, n in IMAGE_STATES:
        kw = dict(max_options=8, device="cpu", batch_size=batch_size)
        ref, ref_tokens = score_image_questions(ex, tok, prep, state, _imgs(n), QS, enc, **kw)
        got, tokens = score_image_questions_cached(ex, tok, prep, state, _imgs(n), QS, enc,
                                                   min_saved_tokens=0, doc_cache=False, **kw)
        assert tokens < ref_tokens
        assert _argmaxes(got) == _argmaxes(ref) and _worst(got, ref) < ATOL, _worst(got, ref)


@pytest.mark.parametrize("k", EXITS)
@pytest.mark.parametrize("holdback", [0, 8])
def test_image_document_cache_is_exact(parts, k, holdback):
    tok, prep, _, _, enc, models = parts
    ex, _ = models[k]
    c = DocCache(max_entries=8, max_bytes=1 << 30, holdback=holdback)
    for state, n in IMAGE_STATES:
        ref = score_image_questions(ex, tok, prep, state, _imgs(n), QS, enc, max_options=8,
                                    device="cpu")[0]
        for _ in range(2):                                          # a miss, then a hit
            got = score_image_questions_cached(ex, tok, prep, state, _imgs(n), QS, enc,
                                               max_options=8, device="cpu", doc_cache=c)[0]
            assert _argmaxes(got) == _argmaxes(ref) and _worst(got, ref) < ATOL
    assert c.stats["hit"] == len(IMAGE_STATES)


# ---------------------------------------------------------------------------
# load_release from meta.json alone (weights stubbed with the tiny model)
# ---------------------------------------------------------------------------

def _spec(**extra):
    return {"readout": "option_xattn", "readout_layer": -1, "max_options": 8,
            "option_pool": "mean", "residual": False, "layout": "state_first",
            "eval_batch_size": 16, **extra}


def _write_ckpt(d: Path, meta: dict, tower, scorer_sd) -> Path:
    from safetensors.torch import save_file
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tower.state_dict().items() if "embed_tokens" not in k},
              str(d / "tower.safetensors"))
    save_file({k: v.contiguous() for k, v in scorer_sd.items()}, str(d / "scorer.safetensors"))
    (d / "meta.json").write_text(json.dumps(meta))
    return d


@pytest.fixture
def stub_weights(monkeypatch, tok_prep):
    """from_pretrained -> a fresh tiny tower (same seed, so the same embedding), and
    the vision tower -> the tiny ViT; records the revision it was asked for."""
    import transformers
    from rsijev import vision
    tok, _ = tok_prep
    seen = {"visual": []}

    def fake_lm(model_id, **kw):
        tower, tc = _tower(len(tok))
        return SimpleNamespace(model=tower, config=SimpleNamespace(text_config=tc))

    def fake_visual(model_id, dtype=torch.bfloat16, revision=None):
        seen["visual"].append((model_id, revision))
        return _visual()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_lm))
    monkeypatch.setattr(vision, "load_visual", fake_visual)
    for k in ("RSIJEV_MAX_LENGTH", "RSIJEV_TRUNCATE"):
        monkeypatch.delenv(k, raising=False)
    return seen


# The meta.json shapes of the published releases (no exit, no cap, no policy), and of
# the 4B exit candidates (vis-v4k: exit 20 in arch_extra, vision in fit_extra).
OLD_METAS = {
    "v1.0-2b": {"base_model": "Qwen/Qwen3.5-2B-Base",
                "spec": _spec(arch_extra={"xattn_combine": "mlp", "xattn_mlp_hidden": 16})},
    "v4.0-vl-2b": {"base_model": "Qwen/Qwen3.5-2B-Base",
                   "spec": _spec(arch_extra={"xattn_combine": "mlp", "xattn_mlp_hidden": 16}),
                   "release": {"name": "v4.0-VL", "vision": {"budget": 64}}},
}


@pytest.mark.parametrize("name", sorted(OLD_METAS))
def test_old_releases_load_as_before(stub_weights, tmp_path, parts, name):
    from serve.release import load_release
    tok, _, tower, _, _, _ = parts
    meta = OLD_METAS[name]
    torch.manual_seed(5)
    ref = DecisionModel(tower, H, _arch()).eval()
    d = _write_ckpt(tmp_path / name, meta, tower, ref.scorer.state_dict())
    model, tok2, enc, m = load_release(d, "cpu")
    assert model.cfg.exit_layer is None
    assert enc.max_length == 2048 and enc.truncate == "left"
    assert m["serving"] == {"max_length": 2048, "truncate": "left", "exit_layer": None, "adaptive": None}
    if "release" in meta:
        assert isinstance(model, VisionDecisionModel)
        assert stub_weights["visual"] == [(meta["base_model"], PINNED_REVISIONS[meta["base_model"]])]
    ids = torch.tensor([tok("A state.", add_special_tokens=False)["input_ids"]])
    assert len(model.encode_prefix(ids).layers) == N_LAYERS        # the whole tower, as before
    got = score_questions(model, tok2, TEXT_STATE, QS, enc, max_options=8, device="cpu")[0]
    want = score_questions(ref, tok, TEXT_STATE, QS, enc, max_options=8, device="cpu")[0]
    assert [p.probs for p in got] == [p.probs for p in want]


def _exit_meta(**spec_extra):
    return {"base_model": BASE4,
            "spec": _spec(arch_extra={"xattn_combine": "mlp", "xattn_mlp_hidden": 16,
                                      "exit_layer": 4},
                          fit_extra={"retention_at_exit": True,
                                     "vision": {"roots": ["/x"], "budget": 64, "model": BASE4}},
                          **spec_extra),
            "release": {"name": "vis-v4k", "cal_mode": "none"}}


def test_an_exit_release_loads_from_meta_alone(stub_weights, tmp_path, parts):
    from serve.release import load_release
    tok, prep, _, _, enc, models = parts
    ex, tr = models[4]
    d = _write_ckpt(tmp_path / "x20", _exit_meta(), ex.tower, ex.scorer.state_dict())
    model, tok2, enc2, meta = load_release(d, "cpu")
    assert isinstance(model, VisionDecisionModel)
    assert model.cfg.exit_layer == 4 and model.cfg.exit_norm is True
    assert meta["serving"] == {"max_length": 2048, "truncate": "left", "exit_layer": 4, "adaptive": None}
    assert meta["vision"]["revision"] == PINNED_REVISIONS[BASE4]
    assert stub_weights["visual"] == [(BASE4, PINNED_REVISIONS[BASE4])]
    got = score_questions(model, tok2, TEXT_STATE, QS, enc2, max_options=8, device="cpu")[0]
    want = score_questions(tr, tok, TEXT_STATE, QS, enc2, max_options=8, device="cpu")[0]
    assert [p.probs for p in got] == [p.probs for p in want]
    venc = EncodeConfig(layout="state_first", option_pool="mean", max_length=2048 + 64)
    got = score_image_questions_cached(model, tok2, prep, IMAGE_STATES[0][0], _imgs(2), QS, venc,
                                       max_options=8, device="cpu", min_saved_tokens=0,
                                       doc_cache=False)[0]
    for q, p in zip(QS, got):
        ref = _offline(tok, prep, tr, IMAGE_STATES[0][0], _imgs(2), q, venc)
        assert torch.allclose(torch.tensor(p.probs), ref, atol=ATOL)


def test_the_trained_cap_and_policy_come_from_meta(stub_weights, tmp_path, parts, monkeypatch):
    from serve.release import load_release
    _, _, _, _, _, models = parts
    ex, _ = models[4]
    d = _write_ckpt(tmp_path / "lc", _exit_meta(max_length=32768, truncate="middle"),
                    ex.tower, ex.scorer.state_dict())
    _, _, enc, meta = load_release(d, "cpu", vision=False)
    assert (enc.max_length, enc.truncate) == (32768, "middle")
    assert "vision" not in meta
    _, _, enc, _ = load_release(d, "cpu", vision=False, max_length=4096, truncate="left")
    assert (enc.max_length, enc.truncate) == (4096, "left")
    monkeypatch.setenv("RSIJEV_MAX_LENGTH", "8192")
    monkeypatch.setenv("RSIJEV_TRUNCATE", "left")
    _, _, enc, _ = load_release(d, "cpu", vision=False)
    assert (enc.max_length, enc.truncate) == (8192, "left")


def test_a_vision_tower_of_another_size_is_refused(stub_weights, tmp_path, parts):
    from serve.release import load_release
    _, _, _, _, _, models = parts
    ex, _ = models[4]
    meta = _exit_meta()
    meta["spec"]["fit_extra"]["vision"]["model"] = "Qwen/Qwen3.5-2B-Base"
    d = _write_ckpt(tmp_path / "bad", meta, ex.tower, ex.scorer.state_dict())
    with pytest.raises(RuntimeError, match="vision tower"):
        load_release(d, "cpu")
    tok = parts[0]
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    wide = Qwen3_5VisionModel(Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64,
                                                  num_heads=2, out_hidden_size=2 * H))
    with pytest.raises(ValueError, match="wide"):
        VisionDecisionModel(ex.tower, H, _arch(readout_layer=-1, exit_layer=4), visual=wide,
                            image_token_id=tok.convert_tokens_to_ids(IMAGE_PAD))


# ---------------------------------------------------------------------------
# the `truncated` response field
# ---------------------------------------------------------------------------

def _served(stub_weights, tmp_path, parts, **spec_extra):
    from serve.server import load_for_serving
    _, _, _, _, _, models = parts
    ex, _ = models[4]
    d = _write_ckpt(tmp_path / "srv", _exit_meta(**spec_extra), ex.tower, ex.scorer.state_dict())
    return load_for_serving(str(d), device="cpu", dtype="fp32")


LONG_REQ = {"model": "srv",
            "state": "Query: where did I park the blue car?\n\n"
                     + " ".join(f"Sentence {i} of a long note." for i in range(400))
                     + "\n\nLAST LINE: level 3, row F.",
            "questions": {"a": {"type": "noul", "instructions": "Does it say where the car is?"},
                          "b": {"type": "choice", "instructions": "Which level?",
                                "criteria": {"one": "Level 1", "three": "Level 3"}}}}


@pytest.mark.parametrize("worker", [False, True])
def test_middle_cut_reports_what_it_cut(stub_weights, tmp_path, parts, worker):
    from fastapi.testclient import TestClient
    from serve.app import create_app
    from serve.batcher import model_worker
    from serve.decider import Decider
    from serve.server import make_scorer
    s = _served(stub_weights, tmp_path, parts, max_length=512, truncate="middle")
    assert (s.enc.max_length, s.enc.truncate) == (512, "middle")
    scorer = model_worker(s) if worker else make_scorer(s)
    body = TestClient(create_app(scorer, served_model_name="srv")).post(
        "/v1/systemone", json=LONG_REQ).json()
    t = body["truncated"]
    assert t["state_tokens_omitted"] > 0 and t["max_length"] == 512
    assert t["option_desc_tokens_omitted"] == 0 and t["question_tokens_omitted"] == 0
    d = Decider.from_scorer(make_scorer(s), name="srv")            # the Python API says the same
    assert d.request(LONG_REQ["state"], LONG_REQ["questions"])["truncated"] == t
    short = TestClient(create_app(scorer, served_model_name="srv")).post(
        "/v1/systemone", json={**LONG_REQ, "state": "Level 3."}).json()
    assert short["truncated"]["state_tokens_omitted"] == 0


def test_left_cut_responses_keep_their_shape(stub_weights, tmp_path, parts):
    from fastapi.testclient import TestClient
    from serve.app import create_app
    from serve.batcher import model_worker
    s = _served(stub_weights, tmp_path, parts, max_length=512)
    assert s.enc.truncate == "left"
    from serve.server import make_scorer
    for scorer in (model_worker(s), make_scorer(s)):
        body = TestClient(create_app(scorer, served_model_name="srv")).post(
            "/v1/systemone", json=LONG_REQ).json()
        assert set(body) == {"model", "answers", "usage"}


def test_an_exit_call_first_leaves_full_depth_calls_whole(tok_prep):
    """transformers installs its hidden-state hooks on the layers present at the first
    output_hidden_states call; an exit call must not be the one that decides."""
    tok, _ = tok_prep
    tower, _ = _tower(len(tok), seed=3)                 # fresh: no hooks installed yet
    ex = DecisionModel(tower, H, _arch(readout_layer=-1, exit_layer=4)).eval()
    ids = torch.tensor([tok("A state.", add_special_tokens=False)["input_ids"]])
    with torch.no_grad():
        ex._run_tower(ids, torch.ones_like(ids))
        out = tower(input_ids=ids, output_hidden_states=True)
    assert len(out.hidden_states) == N_LAYERS + 1
