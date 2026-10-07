"""The MLX backend (rsijev/mlx) against the PyTorch path, on a tiny random Qwen3.5 release
(text tower 8 layers with an aux exit at 4, option_xattn heads with the "mlp" combine,
per-exit temperatures, a 1-block vision tower), on CPU in seconds. No Apple hardware.

  * the DeltaNet core: chunked == one-token-at-a-time == transformers' torch reference (fp32)
  * the text tower, layer by layer, equals transformers' Qwen3_5TextModel (fp32), with right
    padding and continuing from a prefix cache
  * the decision model: every exit's logits (forward_exits), text and image states, equal
    load_release's model (fp32); 8-bit MLX equals the PyTorch model with the MLX weights
    simulated (qdq)
  * numpy collate / unpermute / M-RoPE positions equal the torch ones
  * serving: Decider(backend="mlx") gives the PyTorch Decider's answers, usage.depth and
    usage.confidence for every effort, the unset default (adaptive auto), single and
    multi-question requests, the read-once path, and images
  * the MLX path imports and answers with torch unavailable

Needs mlx (pip install "mlx[cpu]" on Linux), torch, transformers, Pillow, torchvision and
the Qwen3.5 tokenizer / image processor (config files only, no weights).

    python -m pytest tests/test_mlx_parity.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("torchvision")
from PIL import Image                                                 # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from serve.runtime import keep_fused_kernels_off                      # noqa: E402

keep_fused_kernels_off("cpu")
mx.set_default_device(mx.cpu)

from rsijev.contract import Question                                  # noqa: E402
from rsijev.mlx import text as T                                      # noqa: E402

BASE4 = "Qwen/Qwen3.5-4B-Base"
H, N_LAYERS, AUX = 64, 8, 4

QS = [Question("a", "noul", "Is the square red?", ("false", "true"),
               {"false": "No.", "true": "Yes."}),
      Question("b", "choice", "Which colour is the square, and what shape is it really?",
               ("red", "green", "blue"), {"red": "Red.", "green": "Green.", "blue": "Blue."}),
      Question("c", "score", "How bright?", ("0", "1", "2", "3"),
               {"0": "Dark", "1": "Dim", "2": "Bright", "3": "Glaring"})]
TEXT_STATE = "The report says revenue grew 12% while costs fell. " * 12
IMAGE_STATES = [("Left: <image>\nRight: <image>\nA note about both pictures.", 2),
                ("A short note.", 1)]


def _images():
    a = Image.new("RGB", (96, 64), (220, 20, 20))
    b = Image.new("RGB", (64, 64), (20, 20, 220))
    b.paste((250, 250, 250), (16, 16, 48, 48))
    return [a, b]


def _np(a):
    return np.array(a.astype(mx.float32)) if isinstance(a, mx.array) else a.detach().float().numpy()


# ------------------------------------------------------------------------------- DeltaNet
def _delta_inputs(B=2, Tn=150, H_=4, D=16, seed=0, correlated=False):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Tn, H_, D, generator=g)
    k = torch.randn(B, Tn, H_, D, generator=g)
    if correlated:              # nearly parallel keys: the ill-conditioned case
        k = k[:, :1] + 0.05 * k
    v = torch.randn(B, Tn, H_, D, generator=g)
    gl = -torch.rand(B, Tn, H_, generator=g) * 0.3
    beta = torch.rand(B, Tn, H_, generator=g)
    s0 = torch.randn(B, H_, D, D, generator=g) * 0.1
    return q, k, v, gl, beta, s0


@pytest.mark.parametrize("correlated", [False, True])
def test_chunked_delta_rule_equals_recurrent_and_transformers(correlated):
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    q, k, v, gl, beta, s0 = _delta_inputs(correlated=correlated)
    ref, ref_s = torch_chunk_gated_delta_rule(q, k, v, g=gl, beta=beta, initial_state=s0,
                                              output_final_state=True, use_qk_l2norm_in_kernel=True)
    m = [mx.array(x.numpy()) for x in (q, k, v, gl, beta, s0)]
    out, st = T.chunk_gated_delta(*m[:5], state=m[5])
    rec, rst = T.recurrent_gated_delta(*m[:5], state=m[5])
    scale = float(ref.abs().max())
    assert np.abs(_np(out) - ref.numpy()).max() < 1e-5 * max(1, scale)
    assert np.abs(_np(rec) - ref.numpy()).max() < 1e-5 * max(1, scale)
    assert np.abs(_np(st) - ref_s.numpy()).max() < 1e-4
    assert np.abs(_np(rst) - ref_s.numpy()).max() < 1e-4


def test_unit_lower_inverse():
    rng = np.random.default_rng(0)
    k = rng.normal(size=(64, 32))
    k = k[0] + 0.05 * k
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    S = np.tril(rng.uniform(0.5, 1, 64)[:, None] * (k @ k.T), -1)
    got = np.array(T._unit_lower_inverse(mx.array(S[None].astype(np.float32))))[0]
    assert np.abs(got - np.linalg.inv(np.eye(64) + S)).max() < 1e-5


# ------------------------------------------------------------------------------- text tower
def _text_config(vocab: int, n_layers: int = N_LAYERS):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    tc = Qwen3_5TextConfig(vocab_size=vocab, hidden_size=H, intermediate_size=128,
                           num_hidden_layers=n_layers, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
                           linear_key_head_dim=16, linear_value_head_dim=16,
                           layer_types=[("full_attention" if (i + 1) % 4 == 0 else "linear_attention")
                                        for i in range(n_layers)],
                           rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                                            "partial_rotary_factor": 0.25,
                                            "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    tc._attn_implementation = "sdpa"
    return tc


def _stress(model, seed=0):
    """Weights large enough that every op matters (random init leaves a near-identity tower)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim >= 2:
                p.mul_(4.0)
            elif "norm" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
            else:
                p.add_(torch.randn(p.shape, generator=g) * 0.3)


@pytest.fixture(scope="module")
def hf_tower():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    torch.manual_seed(0)
    tc = _text_config(300)
    m = Qwen3_5TextModel(tc).eval()
    _stress(m)
    sd = {k: mx.array(v.numpy()) for k, v in m.state_dict().items()}
    tw = T.TextTower(T.TextConfig.from_config(tc.to_dict()), sd, dtype=mx.float32)
    return m, tw


def test_text_tower_layer_by_layer(hf_tower):
    m, tw = hf_tower
    ids = torch.randint(0, 300, (2, 150), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = m(input_ids=ids, output_hidden_states=True)
    h = tw.embed(mx.array(ids.numpy()))
    rope = tw.rotary(T.text_positions(2, 150))
    for L in range(N_LAYERS):
        h = tw.run(h, L, L + 1, rope)
        ref = out.hidden_states[L + 1] if L + 1 < N_LAYERS else None
        if ref is not None:
            err = np.abs(_np(h) - ref.numpy()).max() / max(1.0, float(ref.abs().max()))
            assert err < 1e-5, (L, err)
    err = np.abs(_np(tw.norm(h)) - out.last_hidden_state.numpy()).max()
    assert err < 1e-4 * max(1.0, float(out.last_hidden_state.abs().max()))


def test_right_padding_and_prefix_cache(hf_tower):
    """A short row padded to a long one reads what it reads alone; a prefix cache plus the
    suffix reads what the whole sequence reads (fp32)."""
    m, tw = hf_tower
    g = torch.Generator().manual_seed(2)
    a = torch.randint(0, 300, (1, 130), generator=g)
    b = torch.randint(0, 300, (1, 70), generator=g)
    pad = torch.cat([b, torch.zeros(1, 60, dtype=torch.long)], 1)
    both = mx.array(torch.cat([a, pad]).numpy())
    hb = tw.run(tw.embed(both), 0, N_LAYERS, tw.rotary(T.text_positions(2, 130)))
    hs = tw.run(tw.embed(mx.array(b.numpy())), 0, N_LAYERS, tw.rotary(T.text_positions(1, 70)))
    assert np.abs(_np(hb[1, :70]) - _np(hs[0])).max() < 1e-4
    # prefix 90 + suffix 40, two rows continuing the same prefix
    full = mx.array(a.numpy())
    hf = tw.run(tw.embed(full), 0, N_LAYERS, tw.rotary(T.text_positions(1, 130)))
    rec = T.PrefixCache()
    tw.run(tw.embed(full[:, :90]), 0, N_LAYERS, tw.rotary(T.text_positions(1, 90)), record=rec)
    assert rec.length == 90
    suf = mx.concatenate([full[:, 90:], full[:, 90:]], axis=0)
    hc = tw.run(tw.embed(suf), 0, N_LAYERS, tw.rotary(T.text_positions(2, 40, 90)), cache=rec)
    for r in range(2):
        assert np.abs(_np(hc[r]) - _np(hf[0, 90:])).max() < 1e-4


def test_vision_tower_fp32():
    """The MLX ViT (+ merger) equals transformers' Qwen3_5VisionModel in fp32, two images."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from rsijev.mlx.vision import VisionTower
    torch.manual_seed(1)
    vc = Qwen3_5VisionConfig(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                             out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                             temporal_patch_size=2, in_channels=3, num_position_embeddings=64)
    vc._attn_implementation = "sdpa"
    m = Qwen3_5VisionModel(vc).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    grid = torch.tensor([[1, 4, 6], [1, 2, 2]])
    pv = torch.randn(int(grid.prod(-1).sum()), 3 * 2 * 16 * 16)
    with torch.no_grad():
        ref = m(pv, grid_thw=grid, return_dict=True).pooler_output.numpy()
    vt = VisionTower({"vision_config": vc.to_dict()},
                     {k: mx.array(v.numpy()) for k, v in m.state_dict().items() if "rotary" not in k},
                     dtype=mx.float32)
    assert np.abs(np.array(vt(pv.numpy(), grid.numpy())) - ref).max() < 1e-5 * max(1, np.abs(ref).max())


# ------------------------------------------------------------------------------- a release
@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        from rsijev.vision import PINNED_REVISIONS
        return AutoTokenizer.from_pretrained(BASE4, revision=PINNED_REVISIONS[BASE4])
    except Exception as e:                                      # offline, no cache
        pytest.skip(f"Qwen3.5-4B tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def release(tok, tmp_path_factory):
    """A self-contained tiny release (what scripts/package_multiexit.py ships, shrunk), plus
    its MLX conversions (bf16-compatible full precision, and 8-bit g32)."""
    from safetensors.torch import save_file
    from transformers import AutoImageProcessor, AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import (Qwen3_5Config,
                                                                   Qwen3_5VisionConfig)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    from rsijev.arch import ArchConfig, DecisionModel
    from rsijev.mlx.convert import convert
    from rsijev.vision import PINNED_REVISIONS
    pkg = tmp_path_factory.mktemp("rel") / "rsi-jev-tiny-mlxtest"
    pkg.mkdir()
    tc = _text_config(len(tok))
    vc = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                             out_hidden_size=H, patch_size=16, spatial_merge_size=2,
                             temporal_patch_size=2, in_channels=3, num_position_embeddings=64)
    cfg = Qwen3_5Config(text_config=tc.to_dict(), vision_config=vc.to_dict(),
                        image_token_id=tok.convert_tokens_to_ids("<|image_pad|>"),
                        video_token_id=tok.convert_tokens_to_ids("<|video_pad|>"),
                        vision_start_token_id=tok.convert_tokens_to_ids("<|vision_start|>"),
                        vision_end_token_id=tok.convert_tokens_to_ids("<|vision_end|>"),
                        tie_word_embeddings=True)
    cfg.save_pretrained(pkg)
    torch.manual_seed(0)
    lm = AutoModelForCausalLM.from_config(cfg)
    tower = getattr(lm, "model", lm)
    _stress(tower)
    save_file({k: v.contiguous() for k, v in tower.state_dict().items()}, str(pkg / "tower.safetensors"))
    torch.manual_seed(1)
    vcfg = Qwen3_5VisionConfig(**vc.to_dict())
    vcfg._attn_implementation = "sdpa"
    visual = Qwen3_5VisionModel(vcfg)
    save_file({k: v.contiguous().bfloat16() for k, v in visual.state_dict().items()
               if "rotary" not in k}, str(pkg / "visual.safetensors"))
    torch.manual_seed(5)
    arch = ArchConfig(readout="option_xattn", readout_layer=-1, max_options=8, freeze_base=True,
                      option_pool="mean", xattn_combine="mlp", exit_layer=N_LAYERS, aux_exits=(AUX,))
    dm = DecisionModel(tower, H, arch)
    with torch.no_grad():
        for p in list(dm.scorer.parameters()) + list(dm.aux_scorers.parameters()):
            p.mul_(3.0)
    save_file({k: v.contiguous() for k, v in dm.scorer.state_dict().items()}, str(pkg / "scorer.safetensors"))
    save_file({f"{L}.{k}": v.contiguous() for L, sc in dm.aux_scorers.items()
               for k, v in sc.state_dict().items()}, str(pkg / "aux_scorers.safetensors"))
    save_file({"cal_logT": torch.tensor(0.05)}, str(pkg / "calibration.safetensors"))
    (pkg / "calibration.json").write_text(json.dumps(
        {"cal_mode": "temp", "exits": {str(AUX): {"cal_mode": "temp", "logT": -0.2}},
         "main_exit": N_LAYERS, "main_logT": 0.05, "T": {str(AUX): 0.8187, str(N_LAYERS): 1.0513}}))
    tok.save_pretrained(pkg)
    AutoImageProcessor.from_pretrained(BASE4, revision=PINNED_REVISIONS[BASE4]).save_pretrained(pkg)
    meta = {"base_model": BASE4, "self_contained": True,
            "spec": {"readout": "option_xattn", "readout_layer": -1, "max_options": 8,
                     "option_pool": "mean", "residual": False, "logit_cap": None,
                     "head_input_norm": False, "layout": "state_first",
                     "option_pool_own_tokens": True, "max_length": 4096,
                     "arch_extra": {"aux_exits": [AUX], "exit_layer": N_LAYERS, "xattn_combine": "mlp"},
                     "fit_extra": {"vision": {"budget": 64, "model": BASE4}}},
            "adaptive": {"exits": [AUX, N_LAYERS], "tau": 0.6, "auto_thresholds": {str(AUX): 0.5}}}
    (pkg / "meta.json").write_text(json.dumps(meta))
    out16 = pkg.parent / "mlx-full"
    out8 = pkg.parent / "mlx-8bit"
    convert(pkg, out16, bits=16, log=lambda *a: None)
    convert(pkg, out8, bits=8, group=32, log=lambda *a: None)
    return pkg, out16, out8


@pytest.fixture(scope="module")
def models(release):
    from serve.release import load_release
    from rsijev.mlx.model import MLXDecisionModel
    pkg, out16, out8 = release
    tm, tok_, enc, meta = load_release(pkg, "cpu")              # fp32 tower and heads
    tm.cal_mode = "none"
    m16 = MLXDecisionModel(out16, dtype=mx.float32)
    m8 = MLXDecisionModel(out8, dtype=mx.float32)
    m16.calibration.mode = m8.calibration.mode = "none"
    m16.cal_mode = m8.cal_mode = "none"
    return tm, tok_, enc, meta, m16, m8


def _text_batch(tok, enc, state=TEXT_STATE, qs=QS):
    from rsijev.encode import encode_question
    return [encode_question(tok, state, q, enc) for q in qs]


def test_collate_and_unpermute_equal_torch(models):
    from rsijev.encode import collate as tcollate, unpermute_logits
    from rsijev.mlx.collate import collate, unpermute
    tm, tok, enc, *_ = models
    rows = _text_batch(tok, enc)
    rows[1] = {**rows[1], "option_perm": [2, 0, 1]}
    a, b = tcollate(tok, rows, 8, option_tokens=False), collate(tok, rows, 8)
    for k, v in b.items():
        assert np.array_equal(a[k].numpy(), v), k
    z = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    ref = unpermute_logits(torch.from_numpy(z), a["option_perm"], a["option_mask"]).numpy()
    assert np.array_equal(unpermute(z, b["option_perm"], b["option_mask"]), ref)


def _close(a, b, mask, tol):
    a, b = _np(a)[mask], _np(b)[mask]
    return float(np.abs(a - b).max()) < tol * max(1.0, float(np.abs(b).max())), float(np.abs(a - b).max())


def test_every_exit_equals_torch_text(models):
    from rsijev.encode import collate as tcollate
    from rsijev.mlx.collate import collate
    tm, tok, enc, _, m16, _ = models
    rows = _text_batch(tok, enc) + _text_batch(tok, enc, "Short state.")
    with torch.no_grad():
        ref = tm.forward_exits(**tcollate(tok, rows, 8))
    got = m16.forward_exits(collate(tok, rows, 8))
    mask = collate(tok, rows, 8)["option_mask"]
    assert sorted(ref) == sorted(got) == [AUX, N_LAYERS]
    for L in ref:
        ok, err = _close(got[L], ref[L], mask, 1e-4)
        assert ok, (L, err)


def test_every_exit_equals_torch_images(models, release):
    from rsijev.encode import EncodeConfig
    from rsijev.mlx.collate import collate
    from rsijev.mlx.vision import ImagePrep, mrope_positions
    from rsijev.vision import ImagePrep as TPrep, VisionConfig, encode_vision_question, vision_collate
    pkg, out16, _ = release
    tm, tok, _, _, m16, _ = models
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical",
                       max_length=4096 + 64, option_pool_own_tokens=True)
    tprep = TPrep(str(pkg), VisionConfig(image_token_budget=64))
    mprep = ImagePrep(out16, budget=64)
    for state, n in IMAGE_STATES:
        ims = _images()[:n]
        pv_t, g_t, nt_t = tprep(ims)
        pv_m, g_m, nt_m = mprep(ims)
        assert np.array_equal(pv_t.numpy(), pv_m) and np.array_equal(g_t.numpy(), g_m) and nt_t == nt_m
        rows = [encode_vision_question(tok, tprep, state, ims, q, enc) for q in QS]
        tb = vision_collate(tok, rows, 8)
        with torch.no_grad():
            feats_t = tm.image_embeds(tb["pixel_values"], tb["image_grid_thw"])
            ref = tm.forward_exits(**tb)
        feats_m = m16.image_embeds(pv_m, g_m)
        # each row carries the request's images once; the batch repeats them per row
        ft = _np(feats_t)
        assert np.abs(_np(mx.concatenate([feats_m] * len(QS))) - ft).max() < 2e-2 * max(1.0, np.abs(ft).max())
        mb = collate(tok, rows, 8)
        mb["position_ids"] = mrope_positions(mb["input_ids"], mb["attention_mask"],
                                             np.tile(g_m, (len(QS), 1)), m16.image_token_id)
        assert np.array_equal(mb["position_ids"], tb["position_ids"].numpy())
        # the text tower on the same image features: fp32, tight
        got = m16.forward_exits(mb, image_embeds=mx.array(feats_t.float().numpy()))
        for L in ref:
            ok, err = _close(got[L], ref[L], mb["option_mask"], 1e-4)
            assert ok, (state[:12], L, err)
        # end to end (the ViT runs in bf16 on both sides): same answers
        got = m16.forward_exits(mb, image_embeds=mx.concatenate([feats_m] * len(QS)))
        for L in ref:
            m_ = mb["option_mask"]
            assert (np.where(m_, _np(got[L]), -np.inf).argmax(-1) ==
                    np.where(m_, _np(ref[L]), -np.inf).argmax(-1)).all(), (state[:12], L)


def test_8bit_equals_torch_with_the_mlx_weights(models, release):
    """MLX 8-bit (g32) vs the PyTorch model holding the dequantized MLX weights."""
    from rsijev.encode import collate as tcollate
    from rsijev.mlx.collate import collate
    from rsijev.mlx.convert import is_quantized_linear
    from rsijev.mlx.quant import qdq
    from serve.release import load_release
    pkg, _, _ = release
    tm, tok, enc, _, _, m8 = models
    tq, *_ = load_release(pkg, "cpu")
    tq.cal_mode = "none"
    with torch.no_grad():
        for name, p in tq.tower.named_parameters():
            if is_quantized_linear(name) and p.shape[-1] % 32 == 0:
                p.copy_(qdq(p.data.bfloat16(), 8, 32).float())
    rows = _text_batch(tok, enc)
    b = tcollate(tok, rows, 8)
    with torch.no_grad():
        ref, base = tq.forward_exits(**b), tm.forward_exits(**b)
    got = m8.forward_exits(collate(tok, rows, 8))
    mask = b["option_mask"].numpy()
    for L in ref:
        _, err = _close(got[L], ref[L], mask, 1)
        _, quant = _close(base[L], ref[L], mask, 1)
        # MLX's CPU quantized matmul accumulates in reduced precision; still well inside
        # the quantization's own effect (8-bit vs full precision)
        assert err < max(2e-2, 0.5 * quant) * max(1.0, float(np.abs(_np(ref[L])[mask]).max())), (L, err, quant)
        r = np.sort(np.where(mask, _np(ref[L]), -np.inf), -1)
        clear = (r[:, -1] - r[:, -2]) > 2 * err          # not a near-tie at this precision
        assert (_np(got[L]).argmax(-1) == _np(ref[L]).argmax(-1))[clear].all(), (L, err, r[:, -2:])


# ------------------------------------------------------------------------------- serving
@pytest.fixture(scope="module")
def deciders(release):
    from serve.decider import Decider
    pkg, out16, _ = release
    os.environ["RSIJEV_MIN_SAVED_TOKENS"] = "100000"            # plain path unless a test asks
    t = Decider(str(pkg), device="cpu", dtype="fp32")
    m = Decider(str(out16), backend="mlx", dtype="fp32")
    yield t, m
    os.environ.pop("RSIJEV_MIN_SAVED_TOKENS", None)


QUESTIONS = {"a": {"type": "noul", "instructions": "Is the square red?"},
             "b": {"type": "choice", "instructions": "Which colour is the square?",
                   "criteria": {"red": "Red.", "green": "Green.", "blue": "Blue."}},
             "c": {"type": "score", "instructions": "How bright?", "criteria": ["Dark", "Dim", "Bright"]}}


def _ask(d, state, qs, effort=None, images=None):
    """The full response body for a request carrying a request-level `effort`."""
    if effort is None:
        return d.request(state, qs, images=images)
    from serve.app import SystemOneRequest, answer_request
    req = SystemOneRequest.model_validate({"state": state, "model": d.name, "questions": qs,
                                           "effort": effort})
    answers, usage, _, _ = answer_request(d._scorer, req)
    return {"model": d.name, "answers": answers, "usage": usage}


def _same(rt, rm, tol=2e-4):
    assert rt.keys() - {"model"} == rm.keys() - {"model"}
    ut, um = rt["usage"], rm["usage"]
    assert ut.get("depth") == um.get("depth"), (ut, um)
    assert ut.get("effort") == um.get("effort")
    for k in ut.get("confidence", {}):
        assert abs(ut["confidence"][k] - um["confidence"][k]) < tol
    assert ut["prompt_tokens"] == um["prompt_tokens"] if "prompt_tokens" in ut else True
    at, am = json.dumps(rt["answers"], sort_keys=True), json.dumps(rm["answers"], sort_keys=True)
    ja, jm = json.loads(at), json.loads(am)
    def eq(v, w, where):
        if isinstance(v, dict):
            assert v.keys() == w.keys(), where
            for k in v:
                eq(v[k], w[k], where + (k,))
        elif isinstance(v, float):
            assert abs(v - w) < tol, (where, v, w)
        else:
            assert v == w, (where, v, w)
    eq(ja, jm, ())


@pytest.mark.parametrize("effort", [None, "low", "medium", "high", "auto"])
@pytest.mark.parametrize("which", ["one", "three"])
def test_decider_matches_torch(deciders, effort, which):
    t, m = deciders
    qs = {"a": QUESTIONS["a"]} if which == "one" else QUESTIONS
    rt, rm = _ask(t, TEXT_STATE, qs, effort), _ask(m, TEXT_STATE, qs, effort)
    _same(rt, rm)
    assert "depth" in rm["usage"] and "confidence" in rm["usage"]


def test_decider_read_once_path_matches_torch(deciders, monkeypatch):
    t, m = deciders
    monkeypatch.setenv("RSIJEV_MIN_SAVED_TOKENS", "0")
    import serve.infer as I
    monkeypatch.setattr(I, "MIN_SAVED_TOKENS", 0)
    for effort in (None, "low", "auto"):
        _same(_ask(t, TEXT_STATE, QUESTIONS, effort), _ask(m, TEXT_STATE, QUESTIONS, effort))


def test_decider_images_match_torch(deciders):
    t, m = deciders
    for state, n in IMAGE_STATES:
        ims = _images()[:n]
        rt, rm = t.request(state, QUESTIONS, images=ims), m.request(state, QUESTIONS, images=ims)
        # the ViT runs in bf16 on both sides (as released); its rounding moves this stressed
        # tiny tower's probabilities by up to a few 1e-2 (fp32 parity: test_vision_tower_fp32)
        _same(rt, rm, tol=5e-2)
        assert rm["usage"]["depth"] == {k: N_LAYERS for k in QUESTIONS}


def test_effort_paths_are_forced_exits_and_the_cascade(models, deciders):
    """low / medium = the aux exit for every question; high = the main exit; auto = the
    cascade (stop at the aux exit when its calibrated top-1 reaches auto_thresholds)."""
    from rsijev.mlx.collate import collate, softmax, unpermute
    from rsijev.mlx.serving import policy_for
    _, m = deciders
    model = m.served.model
    tok, enc = m.served.tok, m.served.enc
    rows = _text_batch(tok, enc) + _text_batch(tok, enc, "Short state.")
    b = collate(tok, rows, 8)
    raw = model.forward_exits(b, calibrate=False)
    temps = {AUX: float(np.exp(-0.2)), N_LAYERS: float(np.exp(0.05))}
    probs = {L: softmax(unpermute(_np(z), b["option_perm"], b["option_mask"]) / temps[L]) for L, z in raw.items()}
    for effort, L in (("low", AUX), ("medium", AUX)):
        z, d = model.staged(b, policy_for(model, effort))
        assert d == [L] * len(rows)
        assert np.abs(softmax(unpermute(_np(z), b["option_perm"], b["option_mask"])) - probs[L]).max() < 1e-5
    z, d = model.staged(b, policy_for(model, "auto"))
    conf = probs[AUX].max(-1)
    want = [AUX if c >= 0.5 else N_LAYERS for c in conf]
    assert d == want
    got = softmax(unpermute(_np(z), b["option_perm"], b["option_mask"]))
    for r, L in enumerate(want):
        assert np.abs(got[r] - probs[L][r]).max() < 1e-5


def test_mlx_path_runs_without_torch(release, tmp_path):
    """Import, load and answer (text and an image) with torch unavailable."""
    _, _, out8 = release
    img = tmp_path / "a.png"
    _images()[0].save(img)
    code = f"""
import sys
sys.modules['torch'] = None
sys.modules['torchvision'] = None
sys.path.insert(0, {str(ROOT)!r})
import mlx.core as mx
mx.set_default_device(mx.cpu)
from PIL import Image
from serve.decider import Decider
d = Decider({str(out8)!r}, backend="mlx")
r = d.request("A state.", {{"a": {{"type": "noul", "instructions": "Is it?"}},
                            "b": {{"type": "noul", "instructions": "Is it not?"}}}})
assert set(r["usage"]["depth"]) == {{"a", "b"}}, r
r = d.request("<image> A picture.", {{"a": {{"type": "noul", "instructions": "Red?"}}}},
              images=[Image.open({str(img)!r})])
assert r["usage"]["depth"]["a"] == {N_LAYERS}, r
assert "torch" not in [m for m in sys.modules if sys.modules[m] is not None and m.split('.')[0] == 'torch']
print("ok")
"""
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0 and p.stdout.strip().endswith("ok"), p.stderr[-3000:]
