"""The fused vision rotary kernel (serve/kernels.py) is bit-identical to transformers'
`apply_rotary_pos_emb_vision`. Needs CUDA and Triton."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _qk(T, H=16, D=64, scale=1.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    qkv = (torch.randn(T, 3 * H * D, device="cuda", generator=g) * scale).to(torch.bfloat16)
    q, k, _ = qkv.reshape(T, 3, H, D).permute(1, 0, 2, 3).unbind(0)
    return q, k


@pytest.mark.parametrize("T,scale", [(1, 1.0), (577, 1.0), (4096, 1.0), (1024, 300.0), (256, 1e-30)])
@pytest.mark.parametrize("cos_dtype", [torch.float32, torch.bfloat16])
def test_bit_identical(T, scale, cos_dtype):
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    from serve import kernels
    kernels.install()
    ref = kernels._ORIG
    q, k = _qk(T, scale=scale)
    pos = torch.arange(T, device="cuda", dtype=torch.float32)[:, None]
    freqs = pos * (1.0 / (10000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64)))[None]
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos().to(cos_dtype), emb.sin().to(cos_dtype)
    rq, rk = ref(q, k, cos, sin)
    fq, fk = m.apply_rotary_pos_emb_vision(q, k, cos, sin)
    assert fq.dtype == rq.dtype and fq.shape == rq.shape
    assert torch.equal(fq.view(torch.int16), rq.view(torch.int16))
    assert torch.equal(fk.view(torch.int16), rk.view(torch.int16))


def test_falls_back_off_cuda_shapes(monkeypatch):
    from serve import kernels
    kernels.install()
    monkeypatch.setenv("RSIJEV_VISION_ROPE", "0")
    q, k = _qk(8)
    cos = torch.ones(8, 64, device="cuda"); sin = torch.zeros(8, 64, device="cuda")
    a = kernels.apply_rotary_pos_emb_vision(q, k, cos, sin)
    b = kernels._ORIG(q, k, cos, sin)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
