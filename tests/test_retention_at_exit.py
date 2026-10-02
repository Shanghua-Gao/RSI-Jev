"""fit.retention_at_exit: the LM-retention term runs the EXIT model.

  * base targets == log_softmax(norm(base hidden_states[L]) @ W.T) top-k, i.e. the
    base model cut at the exit layer (and != the full-depth targets);
  * after fit() steps: decoder layers >= L never run (forward hooks), are frozen
    and bitwise unchanged; lower layers and the final norm train;
  * the scorer head's input width follows the tower's hidden size (2560 for 4B).

    python -m pytest tests/test_retention_at_exit.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from serve.runtime import keep_fused_kernels_off                    # noqa: E402

keep_fused_kernels_off("cpu")      # the torch DeltaNet reference on CPU, not fla's Triton kernels

from rsijev.arch import ArchConfig, DecisionModel, _text_model   # noqa: E402
from rsijev.contract import Case, Question                          # noqa: E402
from rsijev.encode import EncodeConfig                              # noqa: E402
from rsijev.fit import FitConfig, _retention_rows, _retention_targets, fit   # noqa: E402
from rsijev.train import seed_everything                            # noqa: E402

BASE = "Qwen/Qwen3.5-2B-Base"       # tokenizer + layer mix; the 4B shares both
L = 4                                # of 8 layers: output of layer 3 (full attention)


def _tiny(hidden=64, layers=8):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(BASE)
    tc = getattr(cfg, "text_config", cfg)
    for k, v in dict(num_hidden_layers=layers, hidden_size=hidden, intermediate_size=128,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=32,
                     linear_num_key_heads=2, linear_num_value_heads=2,
                     linear_key_head_dim=16, linear_value_head_dim=16, vocab_size=None).items():
        if v is not None and hasattr(tc, k):
            setattr(tc, k, v)
    tc.layer_types = [("full_attention" if (i + 1) % 4 == 0 else "linear_attention")
                      for i in range(layers)]
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(tc).float().eval()


def _arch(**kw):
    return ArchConfig(readout="option_xattn", max_options=8, freeze_base=False,
                      option_pool="mean", residual=False, xattn_combine="mlp", **kw)


def _cases(n=12):
    out = []
    for i in range(n):
        q = Question(key="q", mode="noul", instructions=f"Is item {i} relevant to the request?",
                     options=("false", "true"),
                     criteria={"false": "not relevant", "true": "relevant"})
        out.append(Case(case_id=f"c{i}", source="dc_test",
                        state=f"User asks about topic {i}. " * 6, questions=(q,),
                        gold={"q": (1.0, 0.0) if i % 2 else (0.0, 1.0)}))
    return out


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE)


def _model(lm, exit_layer=L):
    tower = lm.model
    for p in tower.parameters():
        p.requires_grad_(True)
    seed_everything(17)
    return DecisionModel(tower, lm.config.get_text_config().hidden_size,
                         _arch(readout_layer=-1, exit_layer=exit_layer))


def test_targets_are_base_cut_at_exit(tok):
    lm = _tiny()
    m = _model(lm)
    cfg = FitConfig(retention_kl=0.5, retention_pool=4, retention_topk=16, retention_at_exit=True,
                    autocast_bf16=False)
    rows = _retention_rows(tok, _cases(), cfg, 17)
    got = _retention_targets(m, tok, rows, cfg, "cpu", "cpu")
    tm = _text_model(m.tower)
    W = m.tower.get_input_embeddings().weight
    full = _retention_targets(m, tok, rows, FitConfig(**{**cfg.__dict__, "retention_at_exit": False}),
                              "cpu", "cpu")
    with torch.no_grad():
        for r, row in enumerate(rows):
            ids = torch.tensor([row])
            hs = m.tower(input_ids=ids, output_hidden_states=True).hidden_states
            assert len(hs) == 9                              # direct call is full depth again
            n = len(row) - 1
            lp = F.log_softmax(tm.norm(hs[L])[0, :n].float() @ W.float().T, dim=-1)
            v, ix = lp.topk(16, dim=-1)
            assert torch.allclose(got[r][0], v, atol=1e-5), float((got[r][0] - v).abs().max())
            assert torch.equal(got[r][1].long(), ix)
            assert not torch.allclose(full[r][0], v, atol=1e-3)   # the full-depth LM differs


def test_fit_never_runs_upper_layers(tok):
    lm = _tiny()
    m = _model(lm)
    tm = _text_model(m.tower)
    calls = [0] * len(tm.layers)
    for i, layer in enumerate(tm.layers):
        layer.register_forward_hook(lambda mod, a, o, i=i: calls.__setitem__(i, calls[i] + 1))
    before = {n: p.detach().clone() for n, p in m.tower.named_parameters()}
    cfg = FitConfig(steps=3, batch_size=4, retention_kl=0.5, retention_rows=2, retention_pool=4,
                    retention_topk=16, retention_at_exit=True, autocast_bf16=False, log_every=1,
                    lr_base=1e-3)
    fit(m, tok, _cases(), EncodeConfig(layout="state_first", option_pool="mean"), cfg,
        max_options=8, seed=17, device="cpu")
    assert all(calls[i] == 0 for i in range(L, 8)), calls
    assert all(calls[i] > 0 for i in range(L)), calls
    import re
    pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    changed = {n for n, p in m.tower.named_parameters() if not torch.equal(p, before[n])}
    n_upper = 0
    for n, p in m.tower.named_parameters():
        mm = pat.search(n)
        if mm and int(mm.group(1)) >= L:
            n_upper += 1
            assert not p.requires_grad and n not in changed, n
    assert n_upper > 0
    assert any((mm := pat.search(n)) and int(mm.group(1)) == 0 for n in changed), sorted(changed)
    assert any(not pat.search(n) and n.endswith("norm.weight") for n in changed), sorted(changed)


def test_head_width_follows_hidden_2560():
    lm = _tiny(hidden=2560, layers=4)
    m = DecisionModel(lm.model, lm.config.get_text_config().hidden_size,
                      _arch(readout_layer=-1, exit_layer=2))
    assert m.scorer.q.in_features == 2560 and m.scorer.k.in_features == 2560
