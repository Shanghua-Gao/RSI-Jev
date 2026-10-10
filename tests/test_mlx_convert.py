"""The MLX converter (rsijev/mlx/convert.py, scripts/convert_mlx.py) on CPU, no weights.

  * the PyTorch quantizer + MLX packing equal `mx.quantize` bit for bit (codes as packed
    uint32 words, bf16 scales and biases), 2-8 bits, groups 32 / 64 / 128
  * pack / unpack round-trip (numpy only)
  * a converted package: every decoder-layer linear quantized (and only those; embeddings
    with --embed-bits), every other tensor copied unchanged, layers past the exit dropped,
    heads / calibration / vision / tokenizer files copied byte for byte, mlx.json written,
    and the stored codes dequantize to exactly the weights the study simulated (`qdq`)

    python -m pytest tests/test_mlx_convert.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.mlx.convert import LINEAR_NAMES, convert, is_quantized_linear  # noqa: E402
from rsijev.mlx.quant import pack, qdq, quantize, unpack                 # noqa: E402


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group", [32, 64, 128])
def test_quantize_and_pack_equal_mx_quantize(bits, group):
    mx = pytest.importorskip("mlx.core")
    torch.manual_seed(bits * 1000 + group)
    w = (torch.randn(24, 512) * torch.rand(24, 1) * 0.1).bfloat16()
    w[3, :64] = 0.0                                  # an all-zero group
    w[5, 64:128] = w[5, 64:128].abs()                # a one-signed group
    q, s, b = quantize(w, bits, group)
    qm, sm, bm = mx.quantize(mx.array(w.float().numpy()).astype(mx.bfloat16), group_size=group, bits=bits)
    assert np.array_equal(pack(q.numpy(), bits), np.array(qm))
    assert np.array_equal(np.array(sm.astype(mx.float32)), s.float().numpy())
    assert np.array_equal(np.array(bm.astype(mx.float32)), b.float().numpy())
    deq = mx.dequantize(qm, sm, bm, group_size=group, bits=bits)
    assert np.array_equal(np.array(deq.astype(mx.float32)), qdq(w, bits, group).float().numpy())


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_pack_unpack_round_trip(bits):
    rng = np.random.default_rng(bits)
    q = rng.integers(0, 1 << bits, size=(7, 256)).astype(np.uint8)
    words = pack(q, bits)
    assert words.dtype == np.uint32 and words.shape == (7, 256 * bits // 32)
    assert np.array_equal(unpack(words, bits, 256), q)


def test_which_tensors_are_quantized():
    for n in LINEAR_NAMES:
        mixer = "mlp" if n.endswith(("gate_proj", "up_proj", "down_proj")) else (
            "self_attn" if n in ("q_proj", "k_proj", "v_proj", "o_proj") else "linear_attn")
        assert is_quantized_linear(f"layers.7.{mixer}.{n}.weight")
    for k in ("embed_tokens.weight", "norm.weight", "layers.0.input_layernorm.weight",
              "layers.0.linear_attn.conv1d.weight", "layers.0.linear_attn.A_log",
              "layers.0.linear_attn.dt_bias", "layers.0.linear_attn.norm.weight",
              "layers.3.self_attn.q_norm.weight", "layers.3.self_attn.k_norm.weight",
              "layers.0.mlp.gate_proj.bias"):
        assert not is_quantized_linear(k), k


# ---------------------------------------------------------------------------- a package
H, I, V, N = 64, 128, 512, 8


def _tower_sd(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: (torch.randn(*s, generator=g) * 0.05).bfloat16()       # noqa: E731
    sd = {"embed_tokens.weight": r(V, H), "norm.weight": r(H)}
    for i in range(N):
        p = f"layers.{i}."
        sd[p + "input_layernorm.weight"] = r(H)
        sd[p + "post_attention_layernorm.weight"] = r(H)
        sd[p + "mlp.gate_proj.weight"] = r(I, H)
        sd[p + "mlp.up_proj.weight"] = r(I, H)
        sd[p + "mlp.down_proj.weight"] = r(H, I)
        if (i + 1) % 4:
            a = p + "linear_attn."
            sd[a + "in_proj_qkv.weight"] = r(3 * 32, H)
            sd[a + "in_proj_z.weight"] = r(32, H)
            sd[a + "in_proj_b.weight"] = r(2, H)
            sd[a + "in_proj_a.weight"] = r(2, H)
            sd[a + "out_proj.weight"] = r(H, 32)
            sd[a + "conv1d.weight"] = r(96, 1, 4)
            sd[a + "A_log"] = r(2)
            sd[a + "dt_bias"] = r(2)
            sd[a + "norm.weight"] = r(16)
        else:
            a = p + "self_attn."
            sd[a + "q_proj.weight"] = r(2 * 2 * 32, H)
            sd[a + "k_proj.weight"] = r(32, H)
            sd[a + "v_proj.weight"] = r(32, H)
            sd[a + "o_proj.weight"] = r(H, 2 * 32)
            sd[a + "q_norm.weight"] = r(32)
            sd[a + "k_norm.weight"] = r(32)
    return sd


@pytest.fixture()
def package(tmp_path):
    from safetensors.torch import save_file
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    sd = _tower_sd()
    save_file(sd, str(pkg / "tower.safetensors"))
    save_file({"q.weight": torch.randn(H, H)}, str(pkg / "scorer.safetensors"))
    save_file({"4.q.weight": torch.randn(H, H)}, str(pkg / "aux_scorers.safetensors"))
    save_file({"cal_logT": torch.tensor(0.1)}, str(pkg / "calibration.safetensors"))
    save_file({"blocks.0.attn.qkv.weight": torch.randn(96, 32).bfloat16()}, str(pkg / "visual.safetensors"))
    (pkg / "calibration.json").write_text(json.dumps({"cal_mode": "temp"}))
    (pkg / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (pkg / "tokenizer.json").write_text("{}")
    (pkg / "SHA256SUMS").write_text("abc123  tower.safetensors\n")
    meta = {"spec": {"arch_extra": {"exit_layer": 6, "aux_exits": [4]}}}
    (pkg / "meta.json").write_text(json.dumps(meta))
    return pkg, sd


@pytest.mark.parametrize("bits,group,embed_bits", [(8, 64, None), (8, 32, 8), (4, 32, None), (16, None, None)])
def test_convert_package(package, tmp_path, bits, group, embed_bits):
    from safetensors import safe_open
    pkg, sd = package
    out = tmp_path / f"out{bits}"
    rec = convert(pkg, out, bits=bits, group=group or 64, embed_bits=embed_bits,
                  embed_group=group or 64, log=lambda *a: None)
    with safe_open(str(out / "tower.safetensors"), "pt") as fh:
        got = {k: fh.get_tensor(k) for k in fh.keys()}
    # layers 6 and 7 (past exit_layer 6) never run: dropped
    assert not any(k.startswith(("layers.6.", "layers.7.")) for k in got)
    assert rec["layers_dropped"] == sum(1 for k in sd if k.startswith(("layers.6.", "layers.7.")))
    kept = {k: v for k, v in sd.items() if not k.startswith(("layers.6.", "layers.7."))}
    n_q = 0
    for k, w in kept.items():
        quant = ((bits < 16 and is_quantized_linear(k)) or (k == "embed_tokens.weight" and embed_bits)) \
            and w.shape[1] % group == 0 if group else False
        if not quant:
            assert torch.equal(got[k], w), k                         # copied unchanged
            assert not any(x in got for x in (k[:-7] + ".scales",))
            continue
        n_q += 1
        b = embed_bits if k == "embed_tokens.weight" else bits
        name = k[: -len(".weight")]
        words = got[f"{name}.weight"]
        assert words.dtype == torch.uint32 and words.shape == (w.shape[0], w.shape[1] * b // 32)
        q = unpack(words.view(torch.int32).numpy().view(np.uint32), b, w.shape[1])
        s, z = got[f"{name}.scales"], got[f"{name}.biases"]
        assert s.dtype == z.dtype == torch.bfloat16 and s.shape == (w.shape[0], w.shape[1] // group)
        deq = (torch.from_numpy(q).bfloat16().reshape(w.shape[0], -1, group)
               * s.unsqueeze(-1) + z.unsqueeze(-1)).reshape(w.shape)
        assert torch.equal(deq, qdq(w, b, group)), k                 # what the study simulated
    n_div = sum(is_quantized_linear(k) and w.shape[1] % (group or 1) == 0 for k, w in kept.items())
    assert n_q == (n_div if bits < 16 else 0) + (1 if embed_bits else 0)
    for f in ("scorer.safetensors", "aux_scorers.safetensors", "calibration.safetensors",
              "visual.safetensors", "calibration.json", "config.json", "tokenizer.json", "meta.json"):
        assert (out / f).read_bytes() == (pkg / f).read_bytes(), f
    _check_sums(out)
    m = json.loads((out / "mlx.json").read_text())
    assert m["size"]["weights_bytes"] == sum(p.stat().st_size for p in out.glob("*.safetensors"))
    assert m["format"] == "rsijev-mlx" and m["tower"]["bits"] == bits
    assert m["source_tower_sha256"] == "abc123" and m["embed_tokens"]["bits"] == embed_bits
    # a width the group does not divide stays dense (here out_proj / in_proj with 32 inputs at g64)
    assert set(m["not_quantized"]) == ({k for k, w in kept.items() if is_quantized_linear(k)
                                        and w.shape[1] % group} if bits < 16 else set())


def test_convert_loads_in_mlx(package, tmp_path):
    """mx.load reads the converted tower; nn.QuantizedLinear-style (weight, scales, biases)
    triples give mx.dequantize == qdq."""
    mx = pytest.importorskip("mlx.core")
    pkg, sd = package
    out = tmp_path / "o"
    convert(pkg, out, bits=8, group=32, log=lambda *a: None)
    t = mx.load(str(out / "tower.safetensors"))
    k = "layers.0.mlp.down_proj"
    deq = mx.dequantize(t[k + ".weight"], t[k + ".scales"], t[k + ".biases"], group_size=32, bits=8)
    assert np.array_equal(np.array(deq.astype(mx.float32)),
                          qdq(sd[k + ".weight"], 8, 32).float().numpy())


def test_refuses_a_package_without_config(package, tmp_path):
    pkg, _ = package
    (pkg / "config.json").unlink()
    with pytest.raises(SystemExit):
        convert(pkg, tmp_path / "x", log=lambda *a: None)


def _check_sums(out: Path):
    import hashlib
    lines = (out / "SHA256SUMS").read_text().split("\n")
    files = {ln.split("  ")[1]: ln.split("  ")[0] for ln in lines if ln}
    assert set(files) == {p.name for p in out.iterdir() if p.name != "SHA256SUMS"}
    for name, h in files.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == h, name


def _codes_file(sd, path, bits=4, group=32, seed=3):
    """A GPTQ-like result: codes not equal to round-to-nearest, fp32 scales / biases."""
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, w in sd.items():
        if not is_quantized_linear(k) or w.shape[1] % group:
            continue
        n = k[: -len(".weight")]
        q, s, b = quantize(w, bits, group)
        flip = torch.rand(q.shape, generator=g) < 0.2         # move 20% of codes by one step
        q = torch.where(flip, (q.long() + 1).clamp(max=(1 << bits) - 1), q.long()).to(torch.uint8)
        out[n + ".q"], out[n + ".s"], out[n + ".b"] = q, s.float() * 1.001, b.float()
    save_file(out, str(path))
    return out


def test_convert_from_codes_packs_them_as_is(package, tmp_path):
    """--codes: every tower linear from the file, bit for bit (no re-quantization); bits and
    group read from the file; embeddings at their own group; heads stored bf16."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    pkg, sd = package
    # out_proj (32 inputs) is divisible by g32, so the file covers every tower linear
    codes = _codes_file(sd, tmp_path / "codes.safetensors")
    out = tmp_path / "o4"
    rec = convert(pkg, out, bits=8, group=64, codes=tmp_path / "codes.safetensors", embed_bits=8,
                  embed_group=64, heads_dtype="bf16", log=lambda *a: None)
    assert rec["tower"]["bits"] == 4 and rec["tower"]["group_size"] == 32
    assert rec["tower"]["method"].startswith("precomputed") and len(rec["tower"]["codes_sha256"]) == 64
    assert rec["embed_tokens"] == {"bits": 8, "group_size": 64, "method": "round-to-nearest (mx.quantize)"}
    with safe_open(str(out / "tower.safetensors"), "pt") as fh:
        got = {k: fh.get_tensor(k) for k in fh.keys()}
    n = 0
    for k in sd:
        name = k[: -len(".weight")]
        if name + ".q" not in codes or k.startswith(("layers.6.", "layers.7.")):
            continue
        q = unpack(got[k].view(torch.int32).numpy().view(np.uint32), 4, sd[k].shape[1])
        assert np.array_equal(q, codes[name + ".q"].numpy()), k
        assert torch.equal(got[name + ".scales"], codes[name + ".s"].bfloat16())
        assert torch.equal(got[name + ".biases"], codes[name + ".b"].bfloat16())
        n += 1
    assert n == sum(is_quantized_linear(k) for k in sd if not k.startswith(("layers.6.", "layers.7.")))
    e = got["embed_tokens.weight"]
    assert e.shape == (V, H * 8 // 32) and got["embed_tokens.scales"].shape == (V, H // 64)
    for f in ("scorer.safetensors", "aux_scorers.safetensors"):
        a, b = load_file(str(pkg / f)), load_file(str(out / f))
        assert all(b[k].dtype == torch.bfloat16 and torch.equal(b[k], a[k].bfloat16()) for k in a)
    assert (out / "calibration.safetensors").read_bytes() == (pkg / "calibration.safetensors").read_bytes()
    _check_sums(out)


def test_codes_must_cover_every_linear(package, tmp_path):
    from safetensors.torch import load_file, save_file
    pkg, sd = package
    _codes_file(sd, tmp_path / "c.safetensors")
    c = load_file(str(tmp_path / "c.safetensors"))
    drop = "layers.0.mlp.up_proj"
    save_file({k: v for k, v in c.items() if not k.startswith(drop + ".")}, str(tmp_path / "c2.safetensors"))
    with pytest.raises(SystemExit, match="no entry"):
        convert(pkg, tmp_path / "x", codes=tmp_path / "c2.safetensors", log=lambda *a: None)


def test_codes_dequantize_in_mlx_to_the_measured_weights(package, tmp_path):
    """mx.dequantize of the packed file == the effective bf16 weights the codes define."""
    mx = pytest.importorskip("mlx.core")
    from rsijev.mlx.quant import dequantize
    pkg, sd = package
    codes = _codes_file(sd, tmp_path / "c.safetensors")
    out = tmp_path / "o"
    convert(pkg, out, codes=tmp_path / "c.safetensors", log=lambda *a: None)
    t = mx.load(str(out / "tower.safetensors"))
    for n in sorted({k.rsplit(".", 1)[0] for k in codes}):
        if n.startswith(("layers.6.", "layers.7.")):
            continue
        deq = mx.dequantize(t[n + ".weight"], t[n + ".scales"], t[n + ".biases"], group_size=32, bits=4)
        ref = dequantize(codes[n + ".q"], codes[n + ".s"].bfloat16(), codes[n + ".b"].bfloat16(), 32)
        assert np.array_equal(np.array(deq.astype(mx.float32)), ref.float().numpy()), n
