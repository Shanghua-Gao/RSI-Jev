"""The image path on a tiny random Qwen3.5 (text + vision), on CPU in seconds.

Pins the two things the served image path does differently from the offline
encoder the vision releases were gated with (`encode_vision_question` +
`vision_collate` + the model, one question at a time):

  * the vision tower runs once per request and its features are copied into
    every question's row, and the questions run as one padded batch;
  * a text batch through `VisionDecisionModel` is the text model's own pass.

Needs the Qwen3.5 tokenizer and image processor (config files only, no weights);
skips when they are not in the Hugging Face cache and cannot be fetched.

    python -m pytest tests/test_vision_model.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("PIL")
pytest.importorskip("torchvision")
from PIL import Image                                              # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve.runtime import keep_fused_kernels_off                   # noqa: E402

keep_fused_kernels_off("cpu")       # the tiny model runs on CPU: torch reference DeltaNet

from rsijev.arch import ArchConfig, DecisionModel                  # noqa: E402
from rsijev.contract import Question                               # noqa: E402
from rsijev.encode import EncodeConfig, unpermute_logits           # noqa: E402
from rsijev.vision import (IMAGE_PAD, ImagePrep, VisionConfig,     # noqa: E402
                           VisionDecisionModel, encode_vision_question, vision_collate)
from serve.infer import score_image_questions, score_questions     # noqa: E402

BASE = "Qwen/Qwen3.5-2B-Base"


@pytest.fixture(scope="module")
def parts():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(BASE)
        prep = ImagePrep(BASE, VisionConfig(image_token_budget=64, min_tokens_per_image=16))
    except Exception as e:                                      # offline, no cache
        pytest.skip(f"Qwen3.5 tokenizer / image processor unavailable: {e}")
    from transformers.models.qwen3_5.configuration_qwen3_5 import (Qwen3_5TextConfig,
                                                                   Qwen3_5VisionConfig)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5VisionModel
    torch.manual_seed(0)
    tc = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                           num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=16, linear_value_head_dim=16,
                           layer_types=["linear_attention"] * 3 + ["full_attention"],
                           rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                                            "partial_rotary_factor": 0.25,
                                            "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    tc._attn_implementation = "sdpa"
    tower = Qwen3_5TextModel(tc).eval()
    vc = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                             out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                             temporal_patch_size=2, in_channels=3, num_position_embeddings=64)
    vc._attn_implementation = "sdpa"
    visual = Qwen3_5VisionModel(vc).eval()
    arch = ArchConfig(max_options=8, freeze_base=True, readout="option_xattn",
                      xattn_combine="mlp", xattn_mlp_hidden=16)
    vm = VisionDecisionModel(tower, 64, arch, visual=visual,
                             image_token_id=tok.convert_tokens_to_ids(IMAGE_PAD),
                             vcfg=VisionConfig(image_token_budget=64)).eval()
    tm = DecisionModel(tower, 64, arch).eval()
    tm.scorer.load_state_dict(vm.scorer.state_dict())
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical",
                       max_length=2048 + 64)
    return tok, prep, vm, tm, enc


QS = [Question("a", "noul", "Is the square red?", ("false", "true"),
               {"false": "No.", "true": "Yes."}),
      Question("b", "choice", "Which colour is the square, and what shape is it really?",
               ("red", "green", "blue"), {"red": "Red.", "green": "Green.", "blue": "Blue."}),
      Question("c", "score", "How bright?", ("0", "1", "2", "3"),
               {"0": "Dark", "1": "Dim", "2": "Bright", "3": "Glaring"})]


def _images():
    a = Image.new("RGB", (96, 64), (220, 20, 20))
    b = Image.new("RGB", (64, 64), (20, 20, 220))
    b.paste((250, 250, 250), (16, 16, 48, 48))
    return [a, b]


@torch.no_grad()
def _offline(tok, prep, model, state, images, q, enc):
    """The gated offline path: one question, its own ViT pass."""
    bt = vision_collate(tok, [encode_vision_question(tok, prep, state, images, q, enc)],
                        8, device="cpu")
    z = unpermute_logits(model(**bt).float(), bt["option_perm"], bt["option_mask"])
    return torch.softmax(z, -1)[0, :len(q.options)]


def test_served_image_path_equals_the_offline_encoder(parts):
    tok, prep, vm, _, enc = parts
    for state, images in [("Left: <image>\nRight: <image>\nA note.", _images()),
                          ("A short note.", _images()[:1])]:             # no marker: image first
        preds, tokens = score_image_questions(vm, tok, prep, state, images, QS, enc,
                                              max_options=8, device="cpu", batch_size=16)
        for q, p in zip(QS, preds):
            ref = _offline(tok, prep, vm, state, images, q, enc)
            assert torch.allclose(torch.tensor(p.probs), ref, atol=1e-5), (q.key, p.probs, ref)
        assert tokens > sum(prep(images)[2])          # image tokens are counted


def test_batch_size_does_not_move_an_answer(parts):
    tok, prep, vm, _, enc = parts
    state, images = "<image> then <image>", _images()
    one = score_image_questions(vm, tok, prep, state, images, QS, enc, max_options=8,
                                device="cpu", batch_size=1)[0]
    all_ = score_image_questions(vm, tok, prep, state, images, QS, enc, max_options=8,
                                 device="cpu", batch_size=16)[0]
    for a, b in zip(one, all_):
        assert torch.allclose(torch.tensor(a.probs), torch.tensor(b.probs), atol=1e-5)


def test_the_image_reaches_the_decision(parts):
    tok, prep, vm, _, enc = parts
    red = score_image_questions(vm, tok, prep, "<image>", [_images()[0]], QS, enc,
                                max_options=8, device="cpu")[0]
    blue = score_image_questions(vm, tok, prep, "<image>", [_images()[1]], QS, enc,
                                 max_options=8, device="cpu")[0]
    assert any(max(abs(x - y) for x, y in zip(a.probs, b.probs)) > 1e-4 for a, b in zip(red, blue))


def test_a_text_batch_is_the_text_models_pass(parts):
    tok, _, vm, tm, _ = parts
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
    state = "A plain text state with <image> written in it."
    a, _ = score_questions(vm, tok, state, QS, enc, max_options=8, device="cpu")
    b, _ = score_questions(tm, tok, state, QS, enc, max_options=8, device="cpu")
    assert [p.probs for p in a] == [p.probs for p in b]            # bitwise


def test_a_state_too_long_for_its_image_is_refused(parts):
    tok, prep, vm, _, enc = parts
    import dataclasses
    short = dataclasses.replace(enc, max_length=80)
    with pytest.raises(ValueError, match="cut into an image"):
        score_image_questions(vm, tok, prep, "<image> " + "word " * 200, _images()[:1], QS[:1],
                              short, max_options=8, device="cpu")
