"""The vLLM backend (serve/vllm_backend.py): the parts that do not need vLLM.

`VllmDecisionModel` replaces the tower pass with rows returned by an engine and
then runs the checkpoint's own `_readout`. So given the tower's true hidden
states it must return exactly what `DecisionModel.forward` returns -- including
when the engine hands back only the rows after a cached prefix. That is checked
here against a tiny random Qwen3.5 tower on CPU, with a fake engine standing in
for vLLM. The engine itself is exercised by tests/test_vllm_parity.py.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_vllm_backend.py -q
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.arch import ArchConfig, DecisionModel                    # noqa: E402
from rsijev.encode import EncodeConfig, collate, encode_question     # noqa: E402
from rsijev.train import seed_everything                             # noqa: E402
from serve import vllm_backend as vb                                 # noqa: E402
from serve.wire import parse_questions                               # noqa: E402

QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
    "team": {"type": "choice", "instructions": "Which department?",
             "criteria": {"billing": "Payments and refunds", "technical": "Software bugs",
                          "shipping": "Parcels"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["Routine", "Urgent", "Emergency"]},
}


class _Tok:
    """A byte-level stand-in tokenizer: enough for encode_question and collate."""
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1 + (b % 200) for b in text.encode()]}


def _tiny_tower():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    cfg = Qwen3_5TextConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                            head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                            linear_key_head_dim=16, linear_value_head_dim=16,
                            layer_types=["linear_attention"] * 3 + ["full_attention"])
    torch.manual_seed(0)
    return Qwen3_5TextModel(cfg).eval(), cfg


class _FakeTower:
    """Plays VllmTower.hidden_rows from a real HF tower, optionally dropping a
    'cached' prefix the way a prefix-cache hit does."""

    def __init__(self, tower, cached: int = 0):
        self.tower, self.cached, self.calls = tower, cached, []

    @torch.no_grad()
    def hidden_rows(self, seqs, need_from, lead=None):
        out = []
        for s, need in zip(seqs, need_from):
            h = self.tower(input_ids=torch.tensor([s])).last_hidden_state[0]
            off = min(self.cached, need)
            self.calls.append((len(s), off, need))
            out.append((off, h[off:]))
        return out


def _models(cached=0):
    tower, cfg = _tiny_tower()
    arch = ArchConfig(readout="option_xattn", readout_layer=-1, max_options=8,
                      freeze_base=True, option_pool="mean", residual=False,
                      xattn_combine="mlp")
    seed_everything(17)
    ref = DecisionModel(tower, cfg.hidden_size, arch).eval()
    fake = _FakeTower(tower, cached)
    vm = vb.VllmDecisionModel(fake, cfg.hidden_size, arch, hidden_dtype=torch.float32).eval()
    vm.scorer.load_state_dict(ref.scorer.state_dict())
    # a non-trivial calibration, so the cal path is compared too
    for m in (ref, vm):
        m.cal_mode = "temp"
        m.cal_logT.fill_(0.3)
    return ref, vm, fake


def _batch(state="Ticket 7: charged twice, please refund the duplicate charge now."):
    tok = _Tok()
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
    qs = parse_questions(QUESTIONS)
    return collate(tok, [encode_question(tok, state, q, enc) for q in qs], max_options=8)


@pytest.mark.parametrize("cached", [0, 16])
def test_forward_matches_decision_model(cached):
    ref, vm, fake = _models(cached)
    batch = _batch()
    with torch.no_grad():
        want = ref(**batch)
        got = vm(**batch)
    assert torch.equal(torch.isfinite(want), torch.isfinite(got))
    fin = torch.isfinite(want)
    assert torch.allclose(want[fin], got[fin], atol=1e-5, rtol=0), (want - got).abs().max()
    if cached:
        assert all(off == cached for _, off, _ in fake.calls)


def test_need_from_is_first_readout_position():
    _, vm, fake = _models(0)
    batch = _batch()
    with torch.no_grad():
        vm(**batch)
    for r, (_, _, need) in enumerate(fake.calls):
        m = batch["option_mask"][r]
        assert need == min(int(batch["decision_index"][r]),
                           int(batch["option_span_start"][r][m].min()))


def test_refuses_non_final_readout():
    arch = ArchConfig(readout="option_xattn", readout_layer=12, max_options=8,
                      freeze_base=True, option_pool="mean", residual=False)
    with pytest.raises(ValueError, match="final normed"):
        vb.VllmDecisionModel(None, 64, arch)


def test_rows_reruns_when_cache_reaches_the_readout():
    """A cache hit that covers a readout position must be re-run uncached."""
    t = vb.VllmTower.__new__(vb.VllmTower)
    t.prefix_caching = True
    t.stats = {"requests": 0, "rerun_uncached": 0, "rows": 0, "prompt_tokens": 0}
    seen = []

    async def one(ids, read_cache):
        seen.append(read_cache)
        return torch.zeros((len(ids) - (544 if read_cache else 0), 4))
    t._one = one
    off, rows = asyncio.run(t._rows(list(range(600)), need_from=500))
    assert seen == [True, False] and off == 0 and rows.shape[0] == 600
    seen.clear()
    off, rows = asyncio.run(t._rows(list(range(600)), need_from=560))
    assert seen == [True] and off == 544


def test_refuses_large_memory_share():
    with pytest.raises(ValueError, match="0.25"):
        vb.VllmTower("unused", gpu_memory_utilization=0.9)


def test_build_model_dir(tmp_path):
    from safetensors.torch import load_file, save_file
    base, ck = tmp_path / "base", tmp_path / "ckpt"
    base.mkdir(), ck.mkdir()
    text = {"model_type": "qwen3_5_text", "num_hidden_layers": 2, "hidden_size": 4}
    (base / "config.json").write_text(json.dumps({"architectures": ["X"], "text_config": text,
                                                  "tie_word_embeddings": True}))
    emb = "model.language_model.embed_tokens.weight"
    save_file({emb: torch.randn(10, 4).to(torch.bfloat16)}, str(base / "w.safetensors"))
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {emb: "w.safetensors"}}))
    (base / "tokenizer.json").write_text("{}")
    tower = {f"layers.{i}.mlp.up_proj.weight": torch.randn(8, 4) for i in range(2)}
    tower["norm.weight"] = torch.randn(4)
    save_file(tower, str(ck / "tower.safetensors"))
    (ck / "meta.json").write_text(json.dumps({"base_model": "unused",
                                              "spec": {"readout_layer": -1}}))
    out = vb.build_model_dir(ck, tmp_path / "out", base_dir=base)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["architectures"] == [vb.ARCHITECTURE] and cfg["model_type"] == "qwen3_5_text"
    w = load_file(str(out / "model.safetensors"))
    assert set(w) == {"model." + k for k in tower} | {"model.embed_tokens.weight"}
    assert all(v.dtype == torch.bfloat16 for v in w.values())
    assert torch.equal(w["model.norm.weight"], tower["norm.weight"].to(torch.bfloat16))
    assert (out / "tokenizer.json").exists()
    # idempotent for the same checkpoint, refuses a different one
    assert vb.build_model_dir(ck, out, base_dir=base) == out
    tower["norm.weight"] = torch.randn(4)
    save_file(tower, str(ck / "tower.safetensors"))
    with pytest.raises(FileExistsError):
        vb.build_model_dir(ck, out, base_dir=base)
