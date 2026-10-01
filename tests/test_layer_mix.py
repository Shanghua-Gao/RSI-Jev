"""A layer-mixture model: forward mixes per mode, base_logits reads the final state,
and the single-layer hidden_states() refuses with a clear error (it used to raise
NameError on names that only exist in _compute)."""
import pytest
import torch

from rsijev.arch import ArchConfig, DecisionModel


@pytest.fixture(scope="module")
def mix_model():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from serve.runtime import keep_fused_kernels_off
    keep_fused_kernels_off("cpu")
    torch.manual_seed(0)
    tc = Qwen3_5TextConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=16, linear_value_head_dim=16,
                           layer_types=["linear_attention"] * 3 + ["full_attention"])
    tc._attn_implementation = "sdpa"
    tower = Qwen3_5TextModel(tc).eval()
    lm_head = torch.nn.Linear(64, 128, bias=False)
    cfg = ArchConfig(readout="option_marker", layer_mix=(2, 3, -1))
    return DecisionModel(tower, 64, cfg, lm_head=lm_head).eval()


def _batch():
    ids = torch.randint(0, 128, (2, 12))
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                decision_index=torch.tensor([11, 11]),
                option_token_ids=torch.tensor([[5, 6], [7, 8]]),
                option_mask=torch.ones(2, 2, dtype=torch.bool))


def test_hidden_states_refuses_a_layer_mixture(mix_model):
    b = _batch()
    with pytest.raises(ValueError, match="layer mixture"):
        mix_model.hidden_states(b["input_ids"], b["attention_mask"])


def test_base_logits_works_on_a_layer_mixture(mix_model):
    out = mix_model.base_logits(**_batch())
    assert out.shape == (2, 2) and torch.isfinite(out).all()
