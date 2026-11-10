"""MLX memory savings that keep the arithmetic: quantized heads, segmented delta rule."""
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from rsijev.mlx.heads import OptionScorer, QuantWeight, load_heads  # noqa: E402


def _sd(d, rng):
    w = lambda *s: mx.array(rng.standard_normal(s).astype(np.float32) * 0.05)  # noqa: E731
    return {"q.weight": w(d, d), "q.bias": w(d), "k.weight": w(d, d), "k.bias": w(d),
            "v.weight": w(d, d), "v.bias": w(d), "attn.in_proj_weight": w(3 * d, d),
            "attn.in_proj_bias": w(3 * d), "attn.out_proj.weight": w(d, d),
            "attn.out_proj.bias": w(d), "proj.weight": w(1, d), "proj.bias": w(1)}


def test_quantized_heads_equal_heads_dequantized_at_load(tmp_path):
    rng = np.random.default_rng(0)
    d, group = 128, 64
    sd = _sd(d, rng)
    packed = {}
    for k, v in sd.items():
        if v.ndim == 2 and v.shape[-1] % group == 0:
            q, s, b = mx.quantize(v, group_size=group, bits=8)
            packed.update({k: q, k + ".scales": s, k + ".biases": b})
        else:
            packed[k] = v
    mx.save_safetensors(str(tmp_path / "h.safetensors"), packed)
    lazy = load_heads(tmp_path / "h.safetensors", group)
    assert isinstance(lazy["attn.in_proj_weight"], QuantWeight)
    assert lazy["attn.in_proj_weight"].shape == (3 * d, d)
    dense = {k: (v.dense() if isinstance(v, QuantWeight) else v) for k, v in lazy.items()}
    a, b = OptionScorer(lazy), OptionScorer(dense)
    dh = mx.array(rng.standard_normal((3, d)).astype(np.float32))
    oh = mx.array(rng.standard_normal((3, 5, d)).astype(np.float32))
    mask = mx.array(np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], bool))
    za, zb = np.array(a(dh, oh, mask)), np.array(b(dh, oh, mask))
    assert np.array_equal(za, zb)


def test_segmented_delta_rule_equals_one_piece():
    from rsijev.mlx.text import chunk_gated_delta
    rng = np.random.default_rng(1)
    B, T, H, D = 1, 300, 2, 16
    a = lambda *s: mx.array(rng.standard_normal(s).astype(np.float32))  # noqa: E731
    q, k, v = a(B, T, H, D), a(B, T, H, D), a(B, T, H, D)
    g = -mx.abs(a(B, T, H)) * 0.1
    beta = mx.sigmoid(a(B, T, H))
    o1, s1 = chunk_gated_delta(q, k, v, g, beta, segment=0)
    o2, s2 = chunk_gated_delta(q, k, v, g, beta, segment=128)
    assert np.array_equal(np.array(o1), np.array(o2)) and np.array_equal(np.array(s1), np.array(s2))

