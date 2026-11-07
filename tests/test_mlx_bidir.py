"""The MLX backend on a release with meta.json serve_env (v7.0-VL on): option echo and
prefix-bidirectional attention, against the PyTorch path, on the tiny release of
test_mlx_parity.py (CPU, fp32, seconds).

  * the mask: every (query, key) pair as rsijev/arch.py _sdpa_e2 defines it
  * every exit, text and image rows, right padding, query blocks: equal to the PyTorch model
  * a pad token never reaches a real one
  * rows: echoed, the same spans and option_index as the PyTorch encoder; P from the batch
  * serving: the Decider's answers, depth and confidence equal PyTorch's for every effort, the
    cascade drops rows mid-pass, every plan is "plain", cached paths are refused
  * serve_env: applied at load, a conflicting environment refused, unknown switches refused
  * no torch on the path
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_mlx_parity import (AUX, IMAGE_STATES, N_LAYERS, QS, QUESTIONS, TEXT_STATE, _ask, _close,  # noqa: E402
                             _images, _np, _same, _text_batch, release, tok)  # noqa: E402,F401

from rsijev.mlx import text as T                                      # noqa: E402

FLAGS = {"RSIJEV_OPTION_ECHO": "1", "RSIJEV_PREFIX_BIDIR": "1"}


def _with_env(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, copy_function=os.link)
    (dst / "meta.json").unlink()
    meta = json.loads((src / "meta.json").read_text())
    meta["serve_env"] = FLAGS
    (dst / "meta.json").write_text(json.dumps(meta))
    return dst


@pytest.fixture(scope="module")
def v7(release, tmp_path_factory):  # noqa: F811
    """The tiny release, its bf16 and 8-bit MLX builds, with the v7.0-VL serve_env. The
    switches are process environment, so they hold for this module and are restored after."""
    pkg, out16, out8 = release
    root = tmp_path_factory.mktemp("v7")
    dirs = [_with_env(p, root / n) for p, n in ((pkg, "rsi-jev-tiny-v7"), (out16, "mlx16"), (out8, "mlx8"))]
    with pytest.MonkeyPatch.context() as mp:
        for k, v in FLAGS.items():
            mp.setenv(k, v)
        mp.setenv("RSIJEV_MIN_SAVED_TOKENS", "0")           # the cached path if nothing stops it
        yield dirs


@pytest.fixture(scope="module")
def models(v7):
    from rsijev.mlx.model import MLXDecisionModel
    from serve.release import load_release
    pkg, out16, out8 = v7
    tm, tok_, enc, meta = load_release(pkg, "cpu")
    tm.cal_mode = "none"
    m16 = MLXDecisionModel(out16, dtype=mx.float32)
    m16.calibration.mode = m16.cal_mode = "none"
    return tm, tok_, enc, meta, m16


@pytest.fixture(scope="module")
def deciders(v7):
    from serve.decider import Decider
    pkg, out16, _ = v7
    t = Decider(str(pkg), device="cpu", dtype="fp32")
    m = Decider(str(out16), backend="mlx", dtype="fp32")
    return t, m


def _rows(tok, enc):  # noqa: F811
    return _text_batch(tok, enc) + _text_batch(tok, enc, "Short state.")


# ------------------------------------------------------------------------------- the mask
def test_mask_is_the_reference_rule():
    rows = [(0, 12), (5, 12), (12, 12), (3, 7)]              # (P, n), n only pads the row
    P = mx.array([p for p, _ in rows])
    got = np.array(T.prefix_bidir_mask(P, 0, 12, 12))
    assert got.shape == (4, 1, 12, 12)
    for r, (p, n) in enumerate(rows):
        for i in range(n):                                   # real queries; pads are unread
            want = [j < p if i < p else j <= i for j in range(12)]
            assert got[r, 0, i].tolist() == want, (p, i)
            assert not got[r, 0, i, n:].any(), "a real query reads a pad"
    part = np.array(T.prefix_bidir_mask(P, 4, 9, 11))
    assert np.array_equal(part, got[:, :, 4:9, :11])


# ------------------------------------------------------------------------------- rows
def test_rows_are_echoed_with_the_pytorch_spans(models):
    from rsijev.encode import collate as tcollate
    from rsijev.mlx.collate import collate
    from rsijev.mlx.model import prefix_rows
    tm, tok, enc, _, m16 = models
    assert enc.option_echo and m16.prefix_bidir
    assert m16.serve_env == FLAGS
    e = _text_batch(tok, enc)[1]
    text = tok.decode(e["input_ids"])
    assert text.count("Options (again):") == 1
    echo = len(tok(text[: text.index("Options (again):")], add_special_tokens=False)["input_ids"])
    assert all(a >= echo for a, _ in e["option_span"]), "option spans sit on the first copy"
    rows = _rows(tok, enc)
    a, b = tcollate(tok, rows, 8, option_tokens=False), collate(tok, rows, 8)
    for k, v in b.items():
        assert np.array_equal(a[k].numpy(), v), k
    P = np.array(prefix_rows(b))
    want = np.where(b["option_mask"], b["option_span_end"], 0).max(1)
    assert np.array_equal(P, want) and (P <= b["attention_mask"].sum(1)).all()
    assert (P > 0).all() and (P < b["attention_mask"].sum(1)).all(), "the cue stays outside P"


# ------------------------------------------------------------------------------- the pass
def test_every_exit_equals_torch_text_and_differs_from_causal(models):
    from rsijev.encode import collate as tcollate
    from rsijev.mlx.collate import collate
    tm, tok, enc, _, m16 = models
    rows = _rows(tok, enc)
    with torch.no_grad():
        ref = tm.forward_exits(**tcollate(tok, rows, 8))
    b = collate(tok, rows, 8)
    got = m16.forward_exits(b)
    assert set(got) == {AUX, N_LAYERS}
    for L in ref:
        ok, err = _close(got[L], ref[L], b["option_mask"], 1e-4)
        assert ok, (L, err)
    m16.prefix_bidir = False                                 # the check can see the mask
    try:
        causal = m16.forward_exits(b)
    finally:
        m16.prefix_bidir = True
    assert not _close(causal[N_LAYERS], ref[N_LAYERS], b["option_mask"], 1e-3)[0]


def test_a_pad_token_never_reaches_a_real_one(models):
    from rsijev.mlx.collate import collate
    _, tok, enc, _, m16 = models
    rows = _rows(tok, enc)
    short = rows[-1]
    alone = m16.forward_exits(collate(tok, [short], 8))
    pad = m16.forward_exits(collate(tok, [short, rows[0]], 8))
    other = dict(collate(tok, [short, rows[0]], 8))
    other["input_ids"] = other["input_ids"].copy()
    n = len(short["input_ids"])
    other["input_ids"][0, n:] = 17                           # different pad contents
    swapped = m16.forward_exits(other)
    for L in alone:
        assert np.allclose(_np(pad[L])[0], _np(alone[L])[0], atol=1e-4, equal_nan=True), L
        assert np.allclose(_np(swapped[L])[0], _np(alone[L])[0], atol=1e-4, equal_nan=True), L


def test_blocks_equal_one_block(models, monkeypatch):
    """Long inputs: prefix queries see keys past their block."""
    from rsijev.mlx.collate import collate
    _, tok, enc, _, m16 = models
    b = collate(tok, _rows(tok, enc), 8)
    ref = m16.forward_exits(b)
    monkeypatch.setattr(T, "ATTN_BLOCK", 16)
    got = m16.forward_exits(b)
    for L in ref:
        assert np.allclose(_np(got[L]), _np(ref[L]), atol=1e-4, equal_nan=True), L


def test_image_rows_equal_torch(models, v7):
    from rsijev.encode import EncodeConfig
    from rsijev.mlx.collate import collate
    from rsijev.mlx.model import prefix_rows
    from rsijev.mlx.vision import ImagePrep, mrope_positions
    from rsijev.vision import ImagePrep as TPrep, VisionConfig, encode_vision_question, vision_collate
    pkg, out16, _ = v7
    tm, tok, _, _, m16 = models
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical",
                       max_length=4096 + 64, option_pool_own_tokens=True)
    assert enc.option_echo
    tprep = TPrep(str(pkg), VisionConfig(image_token_budget=64))
    mprep = ImagePrep(out16, budget=64)
    for state, n in IMAGE_STATES:
        ims = _images()[:n]
        pv, grid, _ = mprep(ims)
        rows = [encode_vision_question(tok, tprep, state, ims, q, enc) for q in QS]
        tb = vision_collate(tok, rows, 8)
        with torch.no_grad():
            feats = tm.image_embeds(tb["pixel_values"], tb["image_grid_thw"])
            ref = tm.forward_exits(**tb)
        mb = collate(tok, rows, 8)
        mb["position_ids"] = mrope_positions(mb["input_ids"], mb["attention_mask"],
                                             np.tile(grid, (len(QS), 1)), m16.image_token_id)
        last = np.nonzero(mb["input_ids"] == m16.image_token_id)[1].max()
        assert (np.array(prefix_rows(mb)) > last).all(), "the image sits inside the prefix"
        got = m16.forward_exits(mb, image_embeds=mx.array(feats.float().numpy()))
        for L in ref:
            ok, err = _close(got[L], ref[L], mb["option_mask"], 1e-4)
            assert ok, (state[:12], L, err)


# ------------------------------------------------------------------------------- serving
@pytest.mark.parametrize("effort", [None, "low", "medium", "high", "auto"])
@pytest.mark.parametrize("which", ["one", "three"])
def test_decider_matches_torch(deciders, effort, which):
    t, m = deciders
    qs = {"a": QUESTIONS["a"]} if which == "one" else QUESTIONS
    rt, rm = _ask(t, TEXT_STATE, qs, effort), _ask(m, TEXT_STATE, qs, effort)
    _same(rt, rm)
    if effort == "low":
        assert set(rm["usage"]["depth"].values()) == {AUX}
    if effort in ("medium", "high"):                # one aux exit: medium is the main exit
        assert set(rm["usage"]["depth"].values()) == {N_LAYERS}


def test_decider_images_match_torch_at_full_depth(deciders):
    t, m = deciders
    for state, n in IMAGE_STATES:
        ims = _images()[:n]
        rt, rm = t.request(state, QUESTIONS, images=ims), m.request(state, QUESTIONS, images=ims)
        _same(rt, rm, tol=5e-2)                 # the ViT runs in bf16 on both sides, as released
        assert rm["usage"]["depth"] == {k: N_LAYERS for k in QUESTIONS}


def test_cascade_drops_rows_and_keeps_their_prefixes(models, deciders):
    """Rows of different P stop at different exits: each row's answer equals its forced exit."""
    from rsijev.mlx.collate import collate, softmax, unpermute
    from rsijev.mlx.model import Policy
    _, m = deciders
    model = m.served.model
    tok, enc = m.served.tok, m.served.enc
    rows = _rows(tok, enc)
    b = collate(tok, rows, 8)
    raw = model.forward_exits(b, calibrate=False)
    temps = {AUX: float(np.exp(-0.2)), N_LAYERS: float(np.exp(0.05))}
    probs = {L: softmax(unpermute(_np(z), b["option_perm"], b["option_mask"]) / temps[L]) for L, z in raw.items()}
    conf = probs[AUX].max(-1)
    tau = float(np.median(conf))
    assert (conf >= tau).any() and (conf < tau).any()
    base = model.effort_base
    z, d = model.staged(b, Policy(list(base.exits), dict(base.cal), tau))
    assert d == [AUX if c >= tau else N_LAYERS for c in conf]
    got = softmax(unpermute(_np(z), b["option_perm"], b["option_mask"]))
    for r, L in enumerate(d):
        assert np.abs(got[r] - probs[L][r]).max() < 1e-4


def test_no_cached_path(models, deciders):
    from rsijev.mlx.collate import collate
    from rsijev.mlx.serving import plan_request
    _, tok, enc, _, m16 = models
    plan = plan_request(tok, TEXT_STATE, QS, enc)
    assert plan["path"] == "plain" and plan["prefix"] is None
    with pytest.raises(RuntimeError, match="RSIJEV_PREFIX_BIDIR"):
        m16.encode_prefix(np.arange(8))
    cache = T.PrefixCache()
    with pytest.raises(ValueError, match="no prefix cache"):
        m16.forward(collate(tok, _rows(tok, enc), 8), cache=cache)
    with pytest.raises(ValueError, match="no prefix cache"):
        m16.tower.run(mx.zeros((1, 4, 64)), 0, 1, m16._rope({}, 1, 4), record=T.PrefixCache(),
                      prefix=mx.array([2]))


# ------------------------------------------------------------------------------- serve_env
def test_conflicting_environment_refused(v7, monkeypatch):
    from rsijev.mlx.model import MLXDecisionModel
    from rsijev.mlx.serving import load_for_serving
    monkeypatch.setenv("RSIJEV_PREFIX_BIDIR", "0")
    with pytest.raises(RuntimeError, match="needs RSIJEV_PREFIX_BIDIR=1"):
        MLXDecisionModel(v7[1])
    with pytest.raises(RuntimeError, match="needs RSIJEV_PREFIX_BIDIR=1"):
        load_for_serving(str(v7[1]))


def test_unknown_switch_refused(v7, tmp_path):
    from rsijev.mlx.model import MLXDecisionModel
    d = tmp_path / "x"
    shutil.copytree(v7[1], d, copy_function=os.link)
    (d / "meta.json").unlink()
    meta = json.loads((v7[1] / "meta.json").read_text())
    meta["serve_env"]["RSIJEV_SOMETHING"] = "1"
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="unknown switches"):
        MLXDecisionModel(d)


def test_serving_record_and_8bit_build(v7, models):
    from rsijev.mlx.serving import load_for_serving
    s = load_for_serving(str(v7[2]), dtype="fp32")
    assert s.enc.option_echo and s.venc.option_echo and s.model.prefix_bidir
    assert s.meta["serving"]["serve_env"] == FLAGS


def test_a_release_without_serve_env_stays_causal(release, tmp_path):  # noqa: F811
    """Fresh process, no switches: v6.1-style rows and the causal path."""
    _, out16, _ = release
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import mlx.core as mx
mx.set_default_device(mx.cpu)
from rsijev.mlx.serving import load_for_serving
s = load_for_serving({str(out16)!r}, dtype="fp32")
assert not s.model.prefix_bidir and not s.enc.option_echo and "serve_env" not in s.meta["serving"]
print("ok")
"""
    env = {k: v for k, v in os.environ.items() if k not in FLAGS}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0 and p.stdout.strip().endswith("ok"), p.stderr[-3000:]


def test_runs_without_torch(v7, tmp_path):
    """Import, load and answer (text and an image) with torch unavailable."""
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
d = Decider({str(v7[2])!r}, backend="mlx")
r = d.request("A state.", {{"a": {{"type": "noul", "instructions": "Is it?"}},
                            "b": {{"type": "noul", "instructions": "Is it not?"}}}})
assert set(r["usage"]["depth"]) == {{"a", "b"}}, r
r = d.request("<image> A picture.", {{"a": {{"type": "noul", "instructions": "Red?"}}}},
              images=[Image.open({str(img)!r})])
assert r["usage"]["depth"]["a"] == {N_LAYERS}, r
print("ok")
"""
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0 and p.stdout.strip().endswith("ok"), p.stderr[-3000:]


# ------------------------------------------------------------------------------- smaller builds
@pytest.fixture
def dense_matmul(monkeypatch):
    """Quantized linears multiply the dequantized matrix. mx.quantized_matmul accumulates in
    reduced precision on CPU, which test_8bit_equals_torch_with_the_mlx_weights bounds; these
    tests check that a build holds the right weights, so they compare without that rounding."""
    from rsijev.mlx.text import Linear
    plain = Linear.__call__

    def call(self, x):
        if self.scales is None:
            return plain(self, x)
        w = mx.dequantize(self.w, self.scales, self.biases, group_size=self.group, bits=self.bits)
        y = x @ w.astype(x.dtype).T
        return y if self.b is None else y + self.b

    monkeypatch.setattr(Linear, "__call__", call)


def test_quantized_vision_and_heads_equal_their_effective_package(v7, tmp_path, dense_matmul):
    """8-bit vision tower and heads: the MLX build answers as the PyTorch package holding the
    same weights, images included; a mixed-width codes file loads; no vision = text only."""
    from serve.decider import Decider
    from rsijev.mlx.convert import convert
    pkg = v7[0]
    out = tmp_path / "small"
    rec = convert(pkg, out, bits=8, group=32, vision_bits=8, vision_group=16, heads_bits=8,
                  heads_group=32, log=lambda *a: None)
    assert rec["vision"] == {"bits": 8, "group_size": 16} and rec["head_quant"]["bits"] == 8
    eff = tmp_path / "rsi-jev-tiny-eff"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "mlx_effective_pkg.py"), str(out), str(eff)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    t, m = Decider(str(eff), device="cpu", dtype="fp32"), Decider(str(out), backend="mlx", dtype="fp32")
    _same(_ask(t, TEXT_STATE, QUESTIONS, "high"), _ask(m, TEXT_STATE, QUESTIONS, "high"), tol=1e-3)
    state, n = IMAGE_STATES[1]
    ims = _images()[:n]
    _same(t.request(state, QUESTIONS, images=ims), m.request(state, QUESTIONS, images=ims), tol=5e-3)
    noimg = tmp_path / "text"
    convert(pkg, noimg, bits=8, group=32, vision=False, log=lambda *a: None)
    assert not (noimg / "visual.safetensors").exists()
    d = Decider(str(noimg), backend="mlx", dtype="fp32")
    assert d.served.model.visual is None and d.request("A state.", {"a": QUESTIONS["a"]})["answers"]


def test_pq_embeddings(v7, tmp_path, dense_matmul):
    """Product-quantized embeddings: the lookup returns the decoded table exactly, the MLX build
    answers as the PyTorch package holding that table, and the file holds no dense table."""
    from safetensors.torch import load_file
    from serve.decider import Decider
    from rsijev.mlx.convert import convert
    from rsijev.mlx.model import MLXDecisionModel
    from rsijev.mlx.quant import pq_decode
    out = tmp_path / "pq"
    rec = convert(v7[0], out, bits=8, group=32, embed_pq=(4, 16), log=lambda *a: None)
    assert rec["embed_tokens"]["pq"] == {"m": 4, "k": 16}
    sd = load_file(str(out / "tower.safetensors"))
    assert "embed_tokens.weight" not in sd
    table = pq_decode(sd["embed_tokens.pq_codes"], sd["embed_tokens.pq_books"])
    m = MLXDecisionModel(out, dtype=mx.float32)
    ids = np.array([[0, 1, 5], [7, 2, 9]])
    got = np.array(m.tower.embed(mx.array(ids)).astype(mx.float32))
    assert np.array_equal(got, table[torch.from_numpy(ids)].float().numpy())
    eff = tmp_path / "rsi-jev-tiny-pq-eff"
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "mlx_effective_pkg.py"), str(out), str(eff)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    t, d = Decider(str(eff), device="cpu", dtype="fp32"), Decider(str(out), backend="mlx", dtype="fp32")
    _same(_ask(t, TEXT_STATE, QUESTIONS, "high"), _ask(d, TEXT_STATE, QUESTIONS, "high"), tol=1e-3)
