"""v6.0-VL's recipe is expressible by this repo's code, and its new pieces do what they say, on CPU.

* The stage specs in data/v6.0-vl_recipe/ (1 and 3 are release_train specs) are accepted key by
  key: the run-arm spec, FitConfig, ArchConfig; release_train resolves the exit-head paths.
* The encoder keys (max_length, truncate, option_pool_own_tokens) change nothing when unset.
* forward_exits(aux_detach=True) returns the same logits, and the aux loss then reaches only
  the aux heads; fit() with detached aux exits moves the tower and the main head exactly as a
  fit without aux exits does, while the aux heads train at lr_head (lr_aux 0).
* grad_accum splits a step into micro-batches with the same update; freeze_lower_n freezes the
  lowest layers; the conf-rank, aux and non-finite helpers match their definitions.
* The stage 2 / 5 / 6 helpers (refit_exit_heads, dump_exits, rsijev.exit_policy,
  package_multiexit) on small synthetic tensors, and the package they write is read by
  serve/release.py.

    python -m pytest tests/test_v6_recipe.py -q
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))        # after the root: scripts/serve.py would shadow serve/

from serve.runtime import keep_fused_kernels_off                    # noqa: E402

keep_fused_kernels_off("cpu")

from rsijev import exit_policy as P                                # noqa: E402
from rsijev.arch import ArchConfig, DecisionModel                  # noqa: E402
from rsijev.contract import Case, Question                         # noqa: E402
from rsijev.encode import EncodeConfig, collate, encode_question   # noqa: E402
from rsijev.fit import (FitConfig, _clip_groups, _group_grad_norms, _nonfinite_decision,  # noqa: E402
                        aux_exit_term, conf_rank_loss, fit)
from rsijev.train import seed_everything                           # noqa: E402

RECIPE = ROOT / "data" / "v6.0-vl_recipe"
BASE = "Qwen/Qwen3.5-2B-Base"
H = 64


def _spec(name):
    return json.loads((RECIPE / f"{name}.json").read_text())


# ---------------------------------------------------------------- the specs
@pytest.mark.parametrize("stage", ["1-trunk-exits", "3-images"])
def test_stage_spec_is_accepted_key_by_key(stage):
    import run_arm_lib as lib
    spec = _spec(stage)
    assert lib.unknown_spec_keys(spec) == []
    cfg = {**lib.DEFAULTS, **spec}
    fe = dict(cfg["fit_extra"])
    fc = FitConfig(**fe)
    for k, v in fe.items():
        assert getattr(fc, k) == v, k
    ArchConfig(max_options=cfg["max_options"], **dict(cfg["arch_extra"]))
    enc = lib.encode_config(cfg, cfg["option_order"])
    assert (enc.max_length, enc.truncate, enc.option_pool_own_tokens) == (32768, "middle", True)


def test_stage_1_trains_detached_exits_at_lr_head():
    fe = _spec("1-trunk-exits")["fit_extra"]
    assert fe["aux_detach_tower"] is True and fe["lr_aux"] == 0.0 and fe["aux_distill"] == 0.5
    assert _spec("1-trunk-exits")["arch_extra"]["aux_exits"] == [12, 16, 20]
    assert _spec("2-exit-head-refit.json".removesuffix(".json"))["aux_distill"] == 0.5


def test_stage_3_chains_from_stage_2():
    s3 = _spec("3-images")
    assert s3["fit_extra"]["init_from"] == s3["fit_extra"]["aux_init_from"] == s3["init_parent"]["path"] == "trunk-refit/s17"
    assert s3["fit_extra"]["aux_exit_weights"] == {"12": 0.0, "16": 0.0, "20": 0.0}
    assert s3["init_parent"]["parent_steps"] == _spec("1-trunk-exits")["steps"]


def test_release_train_resolves_exit_head_paths(tmp_path):
    import release_train
    fe = release_train.resolve_paths(_spec("3-images"), tmp_path)["fit_extra"]
    assert fe["aux_init_from"] == fe["init_from"] == str(tmp_path / "trunk-refit/s17")
    assert fe["aux_save_dir"] == str(tmp_path / "images/s17")
    assert release_train.resolve_paths(_spec("1-trunk-exits"), tmp_path)["fit_extra"]["aux_save_dir"] == str(tmp_path / "trunk/s17")


def test_encoder_keys_change_nothing_when_unset():
    import run_arm_lib as lib
    cfg = {**lib.DEFAULTS}
    for order in ("canonical", "shuffled"):
        assert lib.encode_config(cfg, order) == EncodeConfig(layout=cfg["layout"], option_pool=cfg["option_pool"],
                                                             option_order=order)


# ---------------------------------------------------------------- tiny model
@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(BASE)
    except Exception as e:                                      # offline, no cache
        pytest.skip(f"Qwen3.5 tokenizer unavailable: {e}")


def _tiny_lm(layers=8):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(BASE)
    tc = getattr(cfg, "text_config", cfg)
    for k, v in dict(num_hidden_layers=layers, hidden_size=H, intermediate_size=128, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                     linear_key_head_dim=16, linear_value_head_dim=16).items():
        if hasattr(tc, k):
            setattr(tc, k, v)
    tc.layer_types = [("full_attention" if (i + 1) % 4 == 0 else "linear_attention") for i in range(layers)]
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(tc).float().eval()


def _arch(**kw):
    return ArchConfig(readout="option_xattn", max_options=8, freeze_base=False, option_pool="mean",
                      residual=False, xattn_combine="mlp", readout_layer=-1, **kw)


def _model(tower, **kw):
    tower = copy.deepcopy(tower)
    for p in tower.parameters():
        p.requires_grad_(True)
    seed_everything(17)
    return DecisionModel(tower, H, _arch(**kw))


def _cases(n=12):
    out = []
    for i in range(n):
        q = Question(key="q", mode="choice", instructions=f"Which team handles ticket {i}?",
                     options=("billing", "tech", "sales"),
                     criteria={"billing": "payments", "tech": "bugs", "sales": "plans"})
        g = [0.0, 0.0, 0.0]
        g[i % 3] = 1.0
        out.append(Case(case_id=f"c{i}", source="dc_test" if i % 2 else "pf_canon", state=f"Ticket {i}: help. " * (2 + i % 3),
                        questions=(q,), gold={"q": tuple(g)}))
    return out


ENC = EncodeConfig(layout="state_first", option_pool="mean", option_order="shuffled")


def test_forward_exits_aux_detach(tok):
    m = _model(_tiny_lm().model, exit_layer=8, aux_exits=(4,)).eval()
    rows = [encode_question(tok, c.state, c.questions[0], ENC) for c in _cases(4)]
    b = collate(tok, rows, max_options=8, device="cpu")
    a, d = m.forward_exits(**b), m.forward_exits(**b, aux_detach=True)
    assert all(torch.equal(a[L], d[L]) for L in a)
    assert torch.equal(a[8], m(**b))                                   # the main exit is forward()
    for detach, reaches in ((True, False), (False, True)):
        m.zero_grad(set_to_none=True)
        z = m.forward_exits(**b, aux_detach=detach)[4]
        z.masked_fill(~torch.isfinite(z), 0).sum().backward()
        tower_g = [p.grad for p in m.tower.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        assert bool(tower_g) is reaches
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.aux_scorers.parameters())


def test_detached_exits_leave_the_main_path_unchanged(tok):
    """fit() with aux exits + aux_detach_tower: tower and main head move exactly as without aux exits."""
    tower = _tiny_lm().model
    plain, multi = _model(tower, exit_layer=8), _model(tower, exit_layer=8, aux_exits=(4,))
    assert all(torch.equal(p, q) for p, q in zip(plain.scorer.parameters(), multi.scorer.parameters()))
    aux0 = {k: v.clone() for k, v in multi.aux_scorers.state_dict().items()}
    common = dict(steps=3, batch_size=4, lr_base=1e-3, lr_head=1e-3, autocast_bf16=False, log_every=1, keep_last_k=1)
    fit(plain, tok, _cases(), ENC, FitConfig(**common), max_options=8, seed=17, device="cpu")
    h = fit(multi, tok, _cases(), ENC, FitConfig(**common, aux_weight_total=1.0, aux_distill=0.5, aux_detach_tower=True,
                                                aux_zloss=1e-4, nonfinite_skip_max=5),
            max_options=8, seed=17, device="cpu")
    for (n, p), q in zip(plain.tower.named_parameters(), multi.tower.parameters()):
        assert torch.allclose(p, q, atol=1e-6, rtol=0), n
    for p, q in zip(plain.scorer.parameters(), multi.scorer.parameters()):
        assert torch.allclose(p, q, atol=1e-6, rtol=0)
    assert any(not torch.equal(aux0[k], v) for k, v in multi.aux_scorers.state_dict().items())
    rec = h["history"][-1]
    assert {"aux4_loss", "aux4_kl", "aux4_z", "gn_aux"} <= set(rec) and rec["nf_skips"] == 0


def test_aux_heads_have_their_own_group_at_lr_head(tok):
    from rsijev.fit import _lr_lambdas, _param_groups
    m = _model(_tiny_lm().model, exit_layer=8, aux_exits=(4,))
    for lr_aux, want in ((0.0, 3e-4), (5e-5, 5e-5)):
        cfg = FitConfig(lr_head=3e-4, lr_aux=lr_aux, head_schedule="cosine", steps=10)
        gr = _param_groups(m, cfg)
        aux = next(g for g in gr if g["name"] == "aux")
        assert aux["lr"] == want and len(aux["params"]) == len(list(m.aux_scorers.parameters()))
        fns = _lr_lambdas(gr, cfg)
        names = [g["name"] for g in gr]
        assert fns[names.index("aux")](9) == fns[names.index("head")](9)


def test_aux_init_from_and_save(tok, tmp_path):
    from safetensors.torch import save_file
    src = _model(_tiny_lm().model, exit_layer=8, aux_exits=(4,))
    for p in src.aux_scorers.parameters():
        p.data.normal_()
    save_file({k: v.contiguous() for k, v in src.aux_scorers.state_dict().items()}, str(tmp_path / "aux_scorers.safetensors"))
    m = _model(_tiny_lm().model, exit_layer=8, aux_exits=(4,))
    fit(m, tok, _cases(4), ENC, FitConfig(steps=1, batch_size=2, lr_head=0.0, lr_base=0.0, autocast_bf16=False,
                                          aux_init_from=str(tmp_path), aux_init_from_main=False,
                                          aux_exit_weights={"4": 0.0}, aux_save_dir=str(tmp_path / "out")),
        max_options=8, seed=17, device="cpu")
    from safetensors.torch import load_file
    got = load_file(str(tmp_path / "out" / "aux_scorers.safetensors"))
    assert all(torch.equal(got[k], v) for k, v in src.aux_scorers.state_dict().items())


def test_grad_accum_is_the_same_step(tok, monkeypatch):
    """Two micro-batches give the one-pass step's gradients (row-share weighting)."""
    grads = []
    real = torch.optim.AdamW.step

    def spy(self, *a, **k):
        grads.append([p.grad.detach().clone() if p.grad is not None else None
                      for g in self.param_groups for p in g["params"]])
        return real(self, *a, **k)
    monkeypatch.setattr(torch.optim.AdamW, "step", spy)
    tower = _tiny_lm().model
    a, b = _model(tower, exit_layer=8), _model(tower, exit_layer=8)
    common = dict(steps=1, batch_size=4, lr_base=1e-3, lr_head=1e-3, autocast_bf16=False, keep_last_k=1, grad_clip=0.0)
    fit(a, tok, _cases(), ENC, FitConfig(**common), max_options=8, seed=17, device="cpu")
    h = fit(b, tok, _cases(), ENC, FitConfig(**common, grad_accum=2), max_options=8, seed=17, device="cpu")
    assert h["split_steps"] == 1 and len(grads) == 2
    n = 0
    for g1, g2 in zip(*grads):
        if g1 is None:
            assert g2 is None
            continue
        n += 1
        assert torch.allclose(g1, g2, rtol=1e-4, atol=1e-7), float((g1 - g2).abs().max())
    assert n > 0


def test_grad_accum_min_tokens_keeps_short_steps_whole(tok):
    m = _model(_tiny_lm().model, exit_layer=8)
    h = fit(m, tok, _cases(), ENC, FitConfig(steps=2, batch_size=4, autocast_bf16=False, grad_accum=2,
                                             grad_accum_min_tokens=100000), max_options=8, seed=17, device="cpu")
    assert h["split_steps"] == 0


def test_canonical_order_sources(tok, monkeypatch):
    import rsijev.fit as F
    seen = []
    real = F.encode_question

    def spy(tokenizer, state, q, enc, rng=None):
        seen.append((state, enc.option_order))
        return real(tokenizer, state, q, enc, rng=rng)
    monkeypatch.setattr(F, "encode_question", spy)
    m = _model(_tiny_lm().model, exit_layer=8)
    fit(m, tok, _cases(), ENC, FitConfig(steps=2, batch_size=4, autocast_bf16=False, canonical_order_sources="pf_canon"),
        max_options=8, seed=17, device="cpu")
    src = {c.state: c.source for c in _cases()}
    assert seen and all((o == "canonical") == (src[s] == "pf_canon") for s, o in seen)


def test_freeze_lower_n():
    m = _model(_tiny_lm().model, exit_layer=8, freeze_lower_n=2)
    import re
    pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    for n, p in m.tower.named_parameters():
        mm = pat.search(n)
        if mm:
            assert p.requires_grad is (int(mm.group(1)) >= 2), n


# ---------------------------------------------------------------- loss helpers
def test_conf_rank_loss_is_ranknet_on_top1_logit_odds():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(10, 4, generator=g)
    z[0, 3] = float("-inf")
    y = torch.randint(0, 3, (10,), generator=g)
    loss, auc, n = conf_rank_loss(z, y, margin_temp=0.5)
    p = torch.softmax(z, -1)
    top = z.argmax(-1)
    m = p.max(-1).values.double()
    s = torch.log(m / (1 - m)) / 0.5
    c = top == y
    d = s[c][:, None] - s[~c][None, :]
    assert n == d.numel() and abs(float(loss) - float(torch.nn.functional.softplus(-d).mean())) < 1e-5
    assert abs(auc - float((d > 0).float().mean())) < 1e-9
    z2 = z.clone().requires_grad_(True)
    loss0, auc0, n0 = conf_rank_loss(z2, z.argmax(-1), margin_temp=1.0)     # every row correct: no pair
    loss0.backward()
    assert n0 == 0 and float(loss0.detach()) == 0.0 and math.isnan(auc0) and torch.isfinite(z2.grad).all()


def test_aux_exit_term():
    g = torch.Generator().manual_seed(1)
    om = torch.tensor([[True, True, True], [True, True, False]])
    main = torch.randn(2, 3, generator=g).masked_fill(~om, float("-inf"))
    aux = torch.randn(2, 3, generator=g).masked_fill(~om, float("-inf"))
    gold = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    cfg = FitConfig(aux_distill=0.5, aux_distill_temp=2.0, aux_zloss=0.1)
    term, parts = aux_exit_term(cfg, main, aux, gold, om, progress=0.0)
    ce = -(gold * torch.log_softmax(aux, -1).masked_fill(~om, 0)).sum(-1).mean()
    pt = torch.softmax(main / 2, -1)
    kl = (pt * (torch.log_softmax(main / 2, -1) - torch.log_softmax(aux / 2, -1))).masked_fill(~om, 0).sum(-1).mean()
    zl = (torch.logsumexp(aux, -1) ** 2).mean()
    assert torch.allclose(term, ce + 0.5 * 4 * kl + 0.1 * zl, atol=1e-6)
    assert torch.allclose(parts["kl"], kl, atol=1e-6) and torch.allclose(parts["z"], zl)


def test_nonfinite_decision_and_safe_clip():
    assert _nonfinite_decision(False, {"head": 1.0}, True) == "skip"
    assert _nonfinite_decision(True, {"head": 1.0, "aux": 2.0}, True) == "ok"
    assert _nonfinite_decision(True, {"head": 1.0, "aux": float("inf")}, True) == "skip_aux"
    assert _nonfinite_decision(True, {"head": 1.0, "aux": float("inf")}, False) == "skip"
    assert _nonfinite_decision(True, {"head": float("nan"), "aux": float("inf")}, True) == "skip"
    w = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.SGD([{"params": [w], "name": "head"}], lr=1.0)
    w.grad = torch.full((4,), 3e19)                        # finite, but sum(g^2) overflows fp32
    scaled = set()
    gn = _group_grad_norms(opt, safe=True, scaled=scaled)
    assert scaled == {"head"} and math.isfinite(gn["head"]) and abs(gn["head"] / 6e19 - 1) < 1e-4
    _clip_groups(opt, lambda g_: True, 1.0, gn, scaled)
    assert torch.isfinite(w.grad).all() and abs(float(w.grad.norm()) - 1.0) < 1e-3


# ---------------------------------------------------------------- stage 2 helpers
def test_refit_helpers(tmp_path):
    import refit_exit_heads as R
    q = R.waterfill({"a": 5, "b": 100, "c": 100}, 105)
    assert q == {"a": 5, "b": 50, "c": 50}
    bs = R.token_batches([5, 1, 3, 8, 2], tok_budget=10, max_bs=2, rng=__import__("random").Random(0))
    assert sorted(i for b in bs for i in b) == list(range(5)) and all(len(b) <= 2 for b in bs)
    assert all(len(b) * max([5, 1, 3, 8, 2][i] for i in b) <= 10 or len(b) == 1 for b in bs)
    f = R.warmup_cosine(50, 3000)
    assert f(0) == pytest.approx(1 / 50 * 0.5 * (1 + 1)) and f(49) == pytest.approx(1.0, abs=1e-3) and f(3000) == pytest.approx(0.0)
    om = torch.tensor([[True, True, False]])
    perm = torch.tensor([[0, 1, 2]])
    ex = {32: torch.tensor([[2.0, 0.0, float("-inf")]]), 16: torch.tensor([[0.0, 1.0, float("-inf")]])}
    gold = torch.tensor([[1.0, 0.0, 0.0]])
    loss, rec = R.refit_loss(ex, 32, gold, perm, om, distill=0.5)
    ps, pt = torch.softmax(torch.tensor([0.0, 1.0]), -1), torch.softmax(torch.tensor([2.0, 0.0]), -1)
    want = -torch.log(ps[0]) + 0.5 * (pt * (pt.log() - ps.log())).sum()
    assert float(loss) == pytest.approx(float(want), abs=1e-6) and set(rec) == {"ce16", "kl16"}
    src, aux = tmp_path / "src", tmp_path / "aux.safetensors"
    src.mkdir()
    for n in ("meta.json", "tower.safetensors", "scorer.safetensors", "aux_scorers.safetensors"):
        (src / n).write_text(n)
    aux.write_text("new")
    R.link_checkpoint(src, tmp_path / "ck", aux)
    assert (tmp_path / "ck" / "tower.safetensors").read_text() == "tower.safetensors"
    assert (tmp_path / "ck" / "aux_scorers.safetensors").read_text() == "new"


# ---------------------------------------------------------------- stage 5-6 dumps
def test_dump_collect_reads_every_exit(tok):
    import dump_exits as DX
    m = _model(_tiny_lm().model, exit_layer=8, aux_exits=(4,)).eval()
    rows = [(c, c.questions[0], f"g|{'cal' if i % 2 else 'tau'}") for i, c in enumerate(_cases(6))]
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
    R = DX.collect(m, tok, enc, rows, max_options=8, device="cpu", tok_budget=4096, batch_size=4, want_h=True)
    assert R["exits"] == [4, 8] and R["z"][8].shape[0] == 6 and R["h"][4].shape == (6, H)
    b = collate(tok, [encode_question(tok, c.state, q, enc) for c, q, _ in rows], max_options=8, device="cpu")
    with torch.no_grad():
        ref = m(**b)[:, :R["z"][8].shape[1]]
        ex = m.forward_exits(**b)
    assert torch.allclose(R["z"][8], ref, atol=1e-5) and torch.allclose(R["z"][4], ex[4][:, :R["z"][4].shape[1]], atol=1e-5)
    assert R["y"].tolist() == [i % 3 for i in range(6)]
    assert DX.token_budget_batches([3, 1, 2], 100, 2) == [[1, 2], [0]]


# ---------------------------------------------------------------- exit policy
def _synthetic_dump(n, exits, seed, tags, case_ids, K=3, sharp=None):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, K, (n,), generator=g)
    z = {}
    for i, L in enumerate(exits):
        s = (sharp or {}).get(L, 1.0 + i)
        zz = torch.randn(n, K, generator=g) + s * torch.nn.functional.one_hot(y, K).float()
        z[L] = torch.nn.functional.pad(zz, (0, 5 - K), value=float("-inf"))
    corr = {L: (P.clean(v).argmax(-1) == y).float() for L, v in z.items()}
    return {"exits": exits, "z": z, "mode": torch.zeros(n, dtype=torch.long), "y": y, "tag": tags,
            "case_id": case_ids, "correct": corr,
            "conf": {L: torch.softmax(P.clean(v), -1).max(-1).values for L, v in z.items()}}


EX4 = [12, 16, 20, 32]


@pytest.fixture(scope="module")
def v1(tmp_path_factory):
    d = tmp_path_factory.mktemp("pd")
    n = 400
    dev = _synthetic_dump(n, EX4, 0, [f"td|{'cal' if i % 2 else 'tau'}" for i in range(n)], [f"d{i}" for i in range(n)])
    txt = _synthetic_dump(n, EX4, 1, [f"{'mmlupro' if i % 4 == 0 else 'misc'}|{P.half_of(f'p{i}')}" for i in range(n)],
                          [f"p{i}" for i in range(n)])
    vis = _synthetic_dump(80, EX4, 2, [""] * 80, [""] * 80)
    vis = {"z": vis["z"], "mode": vis["mode"], "gold": vis["y"], "bench": ["vb1" if i % 2 else "vb2" for i in range(80)]}
    f = d / "probe_a.jsonl"
    f.write_text("\n".join(json.dumps({"case_id": f"v{i}", "questions": [{}]}) for i in range(80)) + "\n")
    return P.load_v1(dev, txt, vis, [f])


def test_temperature_fit_and_weights(v1):
    for h in "AB":
        w = v1["H"][h]["wt"]
        assert abs(float(w.sum()) - 1) < 1e-6 and abs(float(v1["H"][h]["wv"].sum()) - 1) < 1e-6
    T = P.fit_temperatures(v1)
    assert set(T) == set(EX4) and all(-2 <= t <= 3 for t in T.values())
    g = torch.Generator().manual_seed(3)
    z = 3 * torch.randn(20000, 4, generator=g)
    y = torch.multinomial(torch.softmax(z / 2.0, -1), 1, generator=g).squeeze(1)
    assert abs(P.fit_T_unweighted(z, y) - math.log(2.0)) < 0.05
    assert abs(P.fit_T(lambda t: P.logp(z, t), y, torch.full((20000,), 1 / 20000)) - math.log(2.0)) < 0.05


def test_cascade_and_seed_pick():
    Pp = {16: torch.tensor([[.96, .04], [.6, .4], [.5, .5]]), 20: torch.tensor([[.5, .5], [.7, .3], [.5, .5]]),
          32: torch.tensor([[.5, .5]] * 3)}
    assert P.cascade(Pp, 0.65, [16, 20], 32).tolist() == [16, 20, 32]
    assert P.pick_seed({"s0": {"sel_metric": .0342}, "s1": {"sel_metric": .0300}}) == "s1"
    assert P.pick_seed({"s0": {"sel_metric": .0310}, "s1": {"sel_metric": .0300}}) == "s0"


def test_seed_metric(v1):
    n = 400
    dev = _synthetic_dump(n, EX4, 0, [f"td|{'cal' if i % 2 else 'tau'}" for i in range(n)], [f"d{i}" for i in range(n)])
    m = P.seed_metric(dev)
    assert m["sel_metric"] == m["per_exit"]["32"]["dev_tau"]["cal4b_ece"] and m["n_dev_cal"] == 200


def test_cascade_selection_runs_and_binds_a_servable_policy(v1):
    r = P.select_cascade(v1)
    assert r["binding"] == "fixed32" or r["binding"].startswith(("C_t", "C16_t"))
    S = r["serve"]
    assert S["exits"] == EX4 and set(S["logT"]) == {str(L) for L in EX4}
    if r["binding"].startswith("C16_t"):
        assert S["skip_exit_logT"] == {"12": 10.0} and S["tau"] == float(r["binding"][5:])
    b = r["dev"][r["binding"]]["B"]
    ref = r["dev"]["fixed32"]["B"]
    assert b["ECE_TXT"] <= P.ECE_MAX and all(b[c] >= ref[c] - P.TOL_COMP for c in ("TXT", "KN", "VIS"))


def test_threshold_routing_and_di_metric(v1, tmp_path):
    T = P.fit_temperatures(v1)
    th = P.Thresholds(v1, T, {5: 1.0, 9: 1.0, 7: 0.5}, {5: 0.0})
    pk = {"n": 3, "pm": torch.tensor([[.9, .9, .9], [.97, .5, .5], [.5, .6, .5], [.5, .5, .5]])}
    assert th.route(pk, {"kind": "pm", "tau": {16: .95, 20: .59}}).tolist() == [1, 2, 3]
    assert th.route(pk, {"kind": "fixed", "L": 20}).tolist() == [2, 2, 2]
    pd = tmp_path / "pd"
    (pd / "text").mkdir(parents=True)
    n = 8
    rows = [{"case_id": f"h{c}", "questions": [{"options": ["a", "b", "c"], "criteria": {}}] * 2} for c in range(4)]
    (pd / "text" / "di9.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:2]) + "\n")
    (pd / "text" / "clinc_clean.jsonl").write_text("\n".join(json.dumps(r) for r in rows[2:]) + "\n")
    dump = _synthetic_dump(n, EX4, 5, ["di9|A"] * 4 + ["clinc_clean|A"] * 4, [f"h{i // 2}" for i in range(n)])
    d = th.load_di([dump], [pd])
    assert d["bench"].tolist() == [9] * 4 + [5] * 4
    pkd = th.pack_di(d)
    ch = torch.full((n,), 3)
    sk9, dep = th.bench_metric(pkd, 9, ch)          # case-level exact match, chance (1/3)^2 per two-question case
    ok = pkd["ok"][3][:4]
    exact = float(((ok[0::2] * ok[1::2]) > 0).float().mean())
    assert sk9 == pytest.approx(max(0.0, (exact - 1 / 9) / (1 - 1 / 9))) and dep == 32.0
    sk5, _ = th.bench_metric(pkd, 5, ch)            # macro-F1 over the labels, chance 0
    assert 0.0 <= sk5 <= 1.0
    U, per = th.m_sm(pkd, ch)
    assert U["U"] == pytest.approx((sk9 + sk5) / 2) and U["depth"] == 32.0


def test_bench_weights():
    WB = P.bench_weights({"bench": {"3": {"skill": {"exit32": 0.5}, "DI_contrib": {"exit32": 2.0}},
                                    "40": {"skill": {"exit32": 0.0}, "DI_contrib": {"exit32": 0.0}}}})
    assert WB == {3: 0.04, 40: 0.0}


# ---------------------------------------------------------------- the package
def test_package_is_read_by_serving(tmp_path):
    from safetensors.torch import save_file
    import package_multiexit as PM
    from serve.release import adaptive_block, auto_thresholds, exit_temperatures
    ck = tmp_path / "heads"
    ck.mkdir()
    save_file({"w": torch.zeros(2)}, str(ck / "tower.safetensors"))
    save_file({"w": torch.zeros(2)}, str(ck / "scorer.safetensors"))
    save_file({f"{L}.w": torch.full((2,), float(L)) for L in (12, 16, 20)}, str(ck / "aux_scorers.safetensors"))
    (ck / "meta.json").write_text(json.dumps({"spec": {"arch_extra": {"aux_exits": [12, 16, 20], "exit_layer": 32},
                                                       "fit_extra": {"aux_exit_weights": {"12": 0.0, "16": 0.0, "20": 0.0}}},
                                              "head_stage": {"max_length": 4096}}))
    (tmp_path / "seed.json").write_text(json.dumps({"metrics": {"s1": {"per_exit": {"32": {"logT": 0.5}}}}}))
    (tmp_path / "p3.json").write_text(json.dumps({
        "binding": "C16_t0.59", "dev": {"C16_t0.59": {"B": {"U": 0.76}}},
        "serve": {"exits": [12, 16, 20, 32], "tau": 0.59, "logT": {"12": .43, "16": .63, "20": .68, "32": .8},
                  "skip_exit_logT": {"12": 10.0}}}))
    (tmp_path / "auto.json").write_text(json.dumps({"selection": {"CONFIRMED": True, "tau": {"16": 0.95, "20": 0.59}}}))
    out = tmp_path / "pkg"
    sys.argv = ["package_multiexit.py", "--ckpt", str(ck), "--seed-metric", str(tmp_path / "seed.json"), "--seed", "s1",
                "--policy", str(tmp_path / "p3.json"), "--drop-exit", "12", "--thresholds", str(tmp_path / "auto.json"),
                "--out", str(out)]
    assert PM.main() == 0
    from safetensors.torch import load_file
    assert sorted(load_file(str(out / "aux_scorers.safetensors"))) == ["16.w", "20.w"]
    assert float(load_file(str(out / "calibration.safetensors"))["cal_logT"]) == 0.5
    meta = json.loads((out / "meta.json").read_text())
    cal = json.loads((out / "calibration.json").read_text())
    assert meta["spec"]["arch_extra"]["aux_exits"] == [16, 20] and meta["spec"]["train_max_length"] == 4096
    assert meta["spec"]["option_pool_own_tokens"] is True and meta["spec"]["max_length"] == 32768
    blk = adaptive_block(meta)
    assert blk["exits"] == [16, 20, 32] and blk["tau"] == 0.59 and "skip_exits" not in blk
    assert auto_thresholds(blk, blk["exits"]) == {16: 0.95, 20: 0.59}
    temps = exit_temperatures(out, blk["exits"])
    assert {L: round(float(v["logT"]), 4) for L, v in temps.items()} == {16: 0.63, 20: 0.68}
    assert cal["cal_mode"] == "temp" and cal["main_logT"] == 0.8
