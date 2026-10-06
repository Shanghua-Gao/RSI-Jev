"""Adaptive (per-question) early exit, served: aux exits + rsijev.adaptive, on a tiny random
Qwen3.5 (12 layers, aux exit 4, main exit 8), on CPU in seconds.

Ported from the research tree's tests (the arch and staged-execution parts), plus the
serving path:

  * with the aux heads ignored the model IS the fixed-exit model, bitwise;
  * each aux exit is bitwise a fixed exit-L model whose scorer is that aux head;
  * tau = 2 (never exits early) is bitwise the fixed served path for multi-question
    requests read in full, read once and through the document cache (and for one
    question under --adaptive on), with the release's calibration on and off;
  * a mixed tau: stopped rows answer with their exit's logits, are dropped from the
    next stage (the deeper layers run on the other rows only), and the deep rows
    agree with the fixed path; StageReplica is bitwise a fresh replica per stage;
  * the policy: auto sends one question to the fixed exit and several to adaptive exit,
    on sends both, off neither; pooled requests take the fixed exit;
  * load_release takes tau and the per-exit calibrators from the checkpoint; --adaptive,
    RSIJEV_ADAPTIVE, --fixed-exit / RSIJEV_FIXED_EXIT and meta.json adaptive.serving;
  * usage.depth reports the layers run per question, for models with aux exits only.

    python -m pytest tests/test_adaptive_exit.py -q
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve.runtime import keep_fused_kernels_off                   # noqa: E402

keep_fused_kernels_off("cpu")

from rsijev import adaptive as A                                   # noqa: E402
from rsijev.arch import ArchConfig, DecisionModel, _text_model      # noqa: E402
from rsijev.contract import Question                               # noqa: E402
from rsijev.encode import EncodeConfig, collate, encode_question   # noqa: E402
from serve import infer                                            # noqa: E402

BASE = "Qwen/Qwen3.5-2B-Base"
N_LAYERS, EXIT, AUX, H = 12, 8, 4, 64      # exits after full-attention layers 3 and 7
ENC = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")


def _text_config(n_layers, vocab):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    tc = Qwen3_5TextConfig(vocab_size=vocab, hidden_size=H, intermediate_size=128,
                           num_hidden_layers=n_layers, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=16, linear_value_head_dim=16,
                           layer_types=[("full_attention" if (i + 1) % 4 == 0 else "linear_attention")
                                        for i in range(n_layers)])
    tc._attn_implementation = "sdpa"
    return tc


def _tower(vocab, seed=0):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    torch.manual_seed(seed)
    tc = _text_config(N_LAYERS, vocab)
    return Qwen3_5TextModel(tc).eval(), tc


def _arch(**kw):
    return ArchConfig(readout="option_xattn", max_options=8, freeze_base=True, option_pool="mean",
                      residual=False, xattn_combine="mlp", xattn_mlp_hidden=16, readout_layer=-1, **kw)


def _cal(seed, b=0.3):
    """A valid, non-trivial cal-4b (identity-ish PCA, small random head)."""
    g = torch.Generator().manual_seed(seed)
    k = A.CAL_PCA_DIM + 7
    return {"mean": torch.zeros(H), "W": torch.eye(H)[:, :A.CAL_PCA_DIM], "mu": torch.zeros(k),
            "sd": torch.ones(k), "w": 0.05 * torch.randn(k, generator=g), "b": torch.tensor(b)}


def _main_cal(model, seed=7):
    """Turn on the release's own cal-4b at the main exit with random buffers."""
    g = torch.Generator().manual_seed(seed)
    model.cal_mode = "oof_head_scorefloor"
    model.cal_pca_W.copy_(torch.eye(H)[:, :model.cal_pca_W.shape[1]])
    model.cal_w.copy_(0.05 * torch.randn(model.cal_w.shape, generator=g))
    model.cal_b.fill_(-0.2)


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(BASE)
    except Exception as e:                                      # offline, no cache
        pytest.skip(f"Qwen3.5 tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def models(tok):
    tower, _ = _tower(len(tok))
    torch.manual_seed(17)
    fixed = DecisionModel(tower, H, _arch(exit_layer=EXIT)).eval()
    torch.manual_seed(17)
    ada = DecisionModel(tower, H, _arch(exit_layer=EXIT, aux_exits=(AUX,))).eval()
    ada.scorer.load_state_dict(fixed.scorer.state_dict())
    torch.manual_seed(3)
    for p in ada.aux_scorers.parameters():                   # an aux head unlike the main one
        p.data.add_(0.05 * torch.randn_like(p))
    short = DecisionModel(tower, H, _arch(exit_layer=AUX)).eval()
    short.scorer.load_state_dict(ada.aux_scorers[str(AUX)].state_dict())
    return tower, fixed, ada, short


QS = [Question("a", "noul", "Is the user asking for a refund?", ("false", "true"),
               {"false": "No.", "true": "Yes."}),
      Question("b", "choice", "Which team should answer?", ("billing", "tech", "sales"),
               {"billing": "Payments.", "tech": "Bugs.", "sales": "Plans."}),
      Question("c", "score", "How urgent is it?", ("0", "1", "2", "3"),
               {"0": "Not", "1": "Low", "2": "High", "3": "Now"})]
STATES = [f"User {i} writes: my card was charged twice for order {i}, please help. " * (1 + i % 4)
          for i in range(8)]


def _batch(tok, states=STATES, qs=QS):
    rows = [encode_question(tok, s, q, ENC) for s in states for q in qs]
    return collate(tok, rows, max_options=8, device="cpu", option_tokens=False)


def _policy(model, tau, b=0.3):
    return A.Policy(model.exit_indices(), {AUX: _cal(11, b)}, tau)


def _scorers(model):
    return {L: model.exit_scorer(L) for L in model.exit_indices()}


# ---------------------------------------------------------------------------
# the arch: aux heads change nothing unless read
# ---------------------------------------------------------------------------

def test_main_exit_is_the_fixed_model_bitwise(tok, models):
    _, fixed, ada, _ = models
    b = _batch(tok)
    with torch.no_grad():
        want = fixed(**b)
        assert torch.equal(ada(**b), want)
        ex = ada.forward_exits(**b)
    assert sorted(ex) == [AUX, EXIT] and torch.equal(ex[EXIT], want)
    assert set(ada.scorer.state_dict()) == set(fixed.scorer.state_dict())
    assert ada.exit_indices() == [AUX, EXIT] and fixed.aux_scorers is None


def test_each_aux_exit_is_a_fixed_exit_model_bitwise(tok, models):
    _, _, ada, short = models
    b = _batch(tok)
    with torch.no_grad():
        assert torch.equal(ada.forward_exits(**b)[AUX], short(**b))
        z, depth = A.staged_scores(_text_model(ada.tower), _scorers(ada), _policy(ada, -1.0), b)
    assert bool((depth == AUX).all()) and torch.equal(z, short(**b))


def test_aux_exits_are_checked(models):
    tower = models[0]
    for bad in (dict(aux_exits=(4,)), dict(exit_layer=EXIT, aux_exits=(EXIT,)),
                dict(exit_layer=EXIT, exit_norm=False, aux_exits=(4,))):
        with pytest.raises(ValueError):
            DecisionModel(tower, H, _arch(**bad))


# ---------------------------------------------------------------------------
# staged execution
# ---------------------------------------------------------------------------

def test_tau_2_is_the_fixed_path_bitwise(tok, models):
    _, fixed, ada, _ = models
    b = _batch(tok)
    with torch.no_grad():
        z, depth = A.staged_scores(_text_model(ada.tower), _scorers(ada), _policy(ada, 2.0), b)
        assert bool((depth == EXIT).all()) and torch.equal(z, fixed(**b))


def test_stopped_rows_are_dropped(tok, models):
    """Mixed tau: rows over tau answer at the aux exit with its logits; only the others
    run the deeper layers, and they agree with the fixed path."""
    tower, fixed, ada, short = models
    tm = _text_model(tower)
    b = _batch(tok)
    pol = _policy(ada, 2.0)
    with torch.no_grad():
        ex = ada.forward_exits(**b)
        hs = None
        with ada.exit_tower():
            hs = tower(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                       output_hidden_states=True).hidden_states
        h = tm.norm(hs[AUX])
        dh = h[torch.arange(h.shape[0]), b["decision_index"]].float()
        conf = A.calibrated_conf(ex[AUX], dh, b["mode_id"], pol.cal[AUX])
        pol.tau = float(conf.median())
        early = conf >= pol.tau
        assert 0 < int(early.sum()) < len(early)
        seen = []
        hook = tm.layers[EXIT - 1].register_forward_hook(lambda m, a, o: seen.append(a[0].shape[0]))
        stats = {}
        try:
            z, depth = A.staged_scores(tm, _scorers(ada), pol, b, stats=stats)
        finally:
            hook.remove()
    assert torch.equal(depth == AUX, early)
    assert stats["rows_at"] == {AUX: len(early), EXIT: int((~early).sum())}
    assert seen == [int((~early).sum())]                    # the deep layers saw only the rest
    assert torch.equal(z[early], ex[AUX][early])
    fin = torch.isfinite(ex[EXIT])
    assert torch.allclose(z[~early][fin[~early]], ex[EXIT][~early][fin[~early]], atol=1e-5)


def test_stage_replica_is_a_fresh_replica_per_stage(tok, models):
    """The cached path: one replica per layer range (only that stage's layers) is
    bitwise a full fresh replica per stage, and the caller's prefix is untouched."""
    _, _, ada, _ = models
    tm = _text_model(ada.tower)
    state = STATES[3]
    rows = [encode_question(tok, state, q, ENC) for q in QS]
    pids = tok(f"{state}\n\n", add_special_tokens=False)["input_ids"]
    npfx = len(pids)
    cache = ada.encode_prefix(torch.tensor([pids]))
    assert len(cache.layers) == EXIT
    sb = collate(tok, infer._suffixes(rows, npfx), max_options=8, device="cpu", option_tokens=False)
    w = sb["input_ids"].shape[1]
    sb["attention_mask"] = torch.cat([torch.ones((len(rows), npfx), dtype=torch.long), sb["attention_mask"]], 1)
    pos = (torch.arange(w) + npfx).unsqueeze(0).expand(len(rows), w)
    before = copy.deepcopy(cache)
    with torch.no_grad():
        for tau in (2.0, -1.0, 0.5):
            pol = _policy(ada, tau)
            ref, dr = A.staged_scores(tm, _scorers(ada), pol, sb, position_ids=pos,
                                      cache_factory=lambda n: infer._replicate(cache, n, "cpu"))
            rep = A.StageReplica(cache, "cpu")
            got, dg = A.staged_scores(tm, _scorers(ada), pol, sb, stage_cache=rep, position_ids=pos)
            assert torch.equal(dr, dg) and torch.equal(ref, got), tau
            assert rep.replicated_layers <= EXIT
    for lb, la in zip(before.layers, cache.layers):
        for k, v in vars(lb).items():
            if torch.is_tensor(v):
                assert torch.equal(v, getattr(la, k)), k


# ---------------------------------------------------------------------------
# the served path
# ---------------------------------------------------------------------------

def _served_pair(models, tau, calibrated=True, mode="auto"):
    """(fixed-exit release, the same release with adaptive exit at tau, served `mode`)."""
    tower, _, ada, _ = models
    fx = copy.deepcopy(ada)
    ad = copy.deepcopy(ada)
    for m in (fx, ad):
        if calibrated:
            _main_cal(m)
        m.adaptive_policy = None
    ad.adaptive_policy = _policy(ad, tau)
    ad.adaptive_mode = mode
    return fx, ad


# min_saved_tokens / doc cache that put a multi-question request on each text path
PATHS = {"plain": dict(min_saved_tokens=10 ** 9, doc=False),
         "cached": dict(min_saved_tokens=0, doc=False),
         "doc": dict(min_saved_tokens=0, doc=True)}


def _dc(doc):
    return infer.DocCache(max_entries=4, max_bytes=1 << 30, holdback=8) if doc else False


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("path", list(PATHS))
def test_served_tau_2_is_the_fixed_served_path_bitwise(tok, models, calibrated, path):
    """Multi-question requests under auto, in several row batches (batch_size 2 over
    3 questions), on every text path: tau 2 gives the fixed path's probabilities."""
    fx, ad = _served_pair(models, 2.0, calibrated)
    cfg = PATHS[path]
    for state in STATES[:4]:
        kw = dict(max_options=8, device="cpu", batch_size=2)
        ref, rt = infer.score_questions_cached(fx, tok, state, QS, ENC, doc_cache=_dc(cfg["doc"]),
                                               min_saved_tokens=cfg["min_saved_tokens"], **kw)
        plan = infer.plan_request(tok, state, QS, ENC, min_saved_tokens=cfg["min_saved_tokens"],
                                  doc_cache=_dc(cfg["doc"]))
        assert plan["path"] == path and infer.adaptive_applies(ad, plan)
        got, gt = infer.score_planned(ad, tok, plan, **kw)
        assert plan["depth"] == [EXIT] * len(QS) and gt == rt
        assert [p.probs for p in got] == [p.probs for p in ref], (state[:20], path)


@pytest.mark.parametrize("doc", [False, True])
def test_served_tau_2_single_question_on_is_the_fixed_path_bitwise(tok, models, doc):
    fx, ad = _served_pair(models, 2.0, mode="on")
    for state in STATES[:4]:
        for q in QS:
            kw = dict(max_options=8, device="cpu", min_saved_tokens=0)
            ref, rt = infer.score_questions_cached(fx, tok, state, [q], ENC, doc_cache=_dc(doc), **kw)
            plan = infer.plan_request(tok, state, [q], ENC, min_saved_tokens=0, doc_cache=_dc(doc))
            assert plan["path"] == ("doc" if doc else "plain") and infer.adaptive_applies(ad, plan)
            got, gt = infer.score_planned(ad, tok, plan, max_options=8, device="cpu")
            assert plan["depth"] == [EXIT] and gt == rt
            assert [p.probs for p in got] == [p.probs for p in ref], (state[:20], q.key)


def _aux_answer(tok, short, ad, row):
    """What one encoded question answered at AUX with AUX's calibration should get."""
    b = collate(tok, [row], max_options=8, device="cpu", option_tokens=False)
    with torch.no_grad():
        z = short(**b)                                        # the aux exit, uncalibrated
        with short.exit_tower():
            hs = short.tower(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        dh = hs[torch.arange(1), b["decision_index"]].float()
        lt = A.calibrate_logt(z.float(), dh, b["mode_id"], ad.adaptive_policy.cal[AUX])
        zc = infer.unpermute_logits(z.float() / torch.exp(lt)[:, None], b["option_perm"], b["option_mask"])
    assert float(lt[0]) != 0.0                                # the calibrator really acted
    return torch.softmax(zc, -1)[0, :len(row["option_index"])]


@pytest.mark.parametrize("path", list(PATHS))
def test_served_early_answers_carry_their_exits_calibration(tok, models, path):
    """tau -1: every question of a multi-question request stops at AUX and answers with
    AUX's calibrator, on every text path."""
    _, _, _, short = models
    fx, ad = _served_pair(models, -1.0)
    cfg = PATHS[path]
    plan = infer.plan_request(tok, STATES[1], QS, ENC, min_saved_tokens=cfg["min_saved_tokens"],
                              doc_cache=_dc(cfg["doc"]))
    assert plan["path"] == path
    got, _ = infer.score_planned(ad, tok, plan, max_options=8, device="cpu", batch_size=2)
    assert plan["depth"] == [AUX] * len(QS)
    for p, row in zip(got, plan["encoded"]):
        assert torch.allclose(torch.tensor(p.probs), _aux_answer(tok, short, ad, row), atol=1e-5)


def test_policy_routes_by_question_count(tok, models):
    """auto: one question -> fixed exit (bitwise the fixed release), several -> adaptive;
    on: both adaptive; off: neither. tau -1 makes the adaptive route visible (depth AUX)."""
    fx, ad = _served_pair(models, -1.0)
    one, many = [QS[0]], QS
    for mode, want_one, want_many in (("auto", EXIT, AUX), ("on", AUX, AUX), ("off", EXIT, EXIT)):
        ad.adaptive_mode = mode
        for qs, want in ((one, want_one), (many, want_many)):
            for path, cfg in PATHS.items():
                plan = infer.plan_request(tok, STATES[2], qs, ENC, min_saved_tokens=cfg["min_saved_tokens"],
                                          doc_cache=_dc(cfg["doc"]))
                assert infer.adaptive_applies(ad, plan) == (want == AUX), (mode, len(qs), path)
                got, _ = infer.score_planned(ad, tok, plan, max_options=8, device="cpu")
                assert plan["depth"] == [want] * len(qs), (mode, len(qs), path)
                if want == EXIT:
                    ref, _ = infer.score_questions_cached(fx, tok, STATES[2], qs, ENC, max_options=8,
                                                          device="cpu", doc_cache=_dc(cfg["doc"]),
                                                          min_saved_tokens=cfg["min_saved_tokens"])
                    assert [p.probs for p in got] == [p.probs for p in ref], (mode, len(qs), path)
    ad.adaptive_policy = None                                 # no policy: never adaptive
    ad.adaptive_mode = "on"
    assert not infer.adaptive_applies(ad, infer.plan_request(tok, STATES[2], many, ENC, doc_cache=False))


def test_pooled_requests_keep_the_fixed_exit(tok, models):
    from serve.batcher import ModelRunner
    fx, ad = _served_pair(models, -1.0, mode="on")
    runner = ModelRunner(ad, tok, ENC, spec_max_options=8, device="cpu")
    plans = [runner.plan(STATES[i], QS[:2]) for i in range(3)]
    assert all(runner.poolable(p) for p in plans)
    out = runner.run(plans)
    for i, (p, (probs, _)) in enumerate(zip(plans, out)):
        assert p["depth"] == [EXIT, EXIT]
        ref, _ = infer.score_questions_cached(fx, tok, STATES[i], QS[:2], ENC, max_options=8, device="cpu",
                                              doc_cache=False)
        for got, r in zip(probs, ref):
            assert torch.allclose(torch.tensor(got), torch.tensor(r.probs), atol=1e-5)


# ---------------------------------------------------------------------------
# release loading, flags, and the response
# ---------------------------------------------------------------------------

def _write_release(d: Path, ada, tau=0.9, with_policy=True):
    from safetensors.torch import save_file
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from pack_adaptive_policy import pack
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in ada.tower.state_dict().items() if "embed_tokens" not in k},
              str(d / "tower.safetensors"))
    save_file({k: v.contiguous() for k, v in ada.scorer.state_dict().items()}, str(d / "scorer.safetensors"))
    save_file({k: v.contiguous() for k, v in ada.aux_scorers.state_dict().items()},
              str(d / "aux_scorers.safetensors"))
    meta = {"base_model": BASE,
            "spec": {"readout": "option_xattn", "readout_layer": -1, "max_options": 8, "option_pool": "mean",
                     "residual": False, "layout": "state_first", "eval_batch_size": 16,
                     "arch_extra": {"xattn_combine": "mlp", "xattn_mlp_hidden": 16, "exit_layer": EXIT,
                                    "aux_exits": [AUX]}}}
    (d / "meta.json").write_text(json.dumps(meta))
    if with_policy:
        pack(d, {"exits": [AUX, EXIT], "tau": tau, "cal": {AUX: _cal(11)}})
    return d


@pytest.fixture
def stub_lm(monkeypatch, tok):
    import transformers

    def fake_lm(model_id, **kw):
        tower, tc = _tower(len(tok))
        return SimpleNamespace(model=tower, config=SimpleNamespace(text_config=tc))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_lm))
    monkeypatch.delenv("RSIJEV_FIXED_EXIT", raising=False)
    monkeypatch.delenv("RSIJEV_ADAPTIVE", raising=False)


def test_the_policy_is_read_from_the_checkpoint(stub_lm, tmp_path, models, monkeypatch):
    from serve.release import load_release
    _, _, ada, _ = models
    d = _write_release(tmp_path / "ada", ada, tau=0.8725)
    model, _, _, meta = load_release(d, "cpu")
    assert meta["adaptive"]["tau"] == 0.8725
    assert meta["serving"]["adaptive"] == {"exits": [AUX, EXIT], "tau": 0.8725, "mode": "auto"}
    assert model.adaptive_mode == "auto"
    pol = model.adaptive_policy
    assert pol.tau == 0.8725 and pol.exits == [AUX, EXIT]
    assert all(torch.equal(pol.cal[AUX][k], v.float()) for k, v in _cal(11).items())
    for k, v in ada.aux_scorers.state_dict().items():
        assert torch.equal(model.aux_scorers.state_dict()[k], v)
    m2, _, _, meta2 = load_release(d, "cpu", fixed_exit=True)
    assert m2.adaptive_policy is None and "forced" in meta2["serving"]["adaptive_off"]
    monkeypatch.setenv("RSIJEV_FIXED_EXIT", "1")
    assert load_release(d, "cpu")[0].adaptive_policy is None


def test_aux_exits_without_a_tau_serve_the_fixed_exit(stub_lm, tmp_path, models):
    from serve.release import load_release
    _, _, ada, _ = models
    d = _write_release(tmp_path / "notau", ada, with_policy=False)
    model, _, _, meta = load_release(d, "cpu")
    assert model.adaptive_policy is None and meta["serving"]["adaptive"] is None
    assert "tau" in meta["serving"]["adaptive_off"]
    (d / "aux_scorers.safetensors").unlink()
    with pytest.raises(RuntimeError, match="aux_scorers"):
        load_release(d, "cpu")


def _set_serving(d: Path, mode):
    meta = json.loads((d / "meta.json").read_text())
    meta["adaptive"]["serving"] = mode
    (d / "meta.json").write_text(json.dumps(meta))


def test_adaptive_flags(stub_lm, tmp_path, models, monkeypatch):
    """--adaptive / RSIJEV_ADAPTIVE beat --fixed-exit / RSIJEV_FIXED_EXIT (an alias for
    off), which beat meta.json adaptive.serving, which beats the default auto."""
    from serve.release import load_release
    from serve.server import add_serve_args
    import argparse
    _, _, ada, _ = models
    d = _write_release(tmp_path / "flags", ada, tau=0.9)

    def mode(**kw):
        m, _, _, meta = load_release(d, "cpu", **kw)
        got = m.adaptive_mode
        assert (m.adaptive_policy is None) == (got == "off")
        assert (meta["serving"]["adaptive"] or {}).get("mode", "off") == got
        return got
    assert mode() == "auto"
    for m in ("auto", "on", "off"):
        assert mode(adaptive=m) == m
    assert mode(fixed_exit=True) == "off" and mode(fixed_exit=True, adaptive="off") == "off"
    with pytest.raises(ValueError, match="contradicts"):
        load_release(d, "cpu", fixed_exit=True, adaptive="on")
    with pytest.raises(ValueError, match="adaptive must be"):
        load_release(d, "cpu", adaptive="sometimes")
    monkeypatch.setenv("RSIJEV_ADAPTIVE", "on")
    assert mode() == "on" and mode(adaptive="off") == "off"
    monkeypatch.delenv("RSIJEV_ADAPTIVE")
    monkeypatch.setenv("RSIJEV_FIXED_EXIT", "1")
    assert mode() == "off" and mode(fixed_exit=False) == "auto"
    monkeypatch.delenv("RSIJEV_FIXED_EXIT")
    _set_serving(d, "on")                                     # the release's own default
    assert mode() == "on" and mode(adaptive="auto") == "auto" and mode(fixed_exit=True) == "off"
    _set_serving(d, "off")
    m, _, _, meta = load_release(d, "cpu")
    assert m.adaptive_policy is None and "adaptive.serving" in meta["serving"]["adaptive_off"]
    assert mode(adaptive="auto") == "auto"
    # the server's arguments
    ap = argparse.ArgumentParser()
    add_serve_args(ap, positional=False)
    assert ap.parse_args([]).adaptive is None and not ap.parse_args([]).fixed_exit
    assert ap.parse_args(["--adaptive", "on"]).adaptive == "on"
    assert ap.parse_args(["--fixed-exit"]).fixed_exit
    with pytest.raises(SystemExit):
        ap.parse_args(["--adaptive", "sometimes"])
    monkeypatch.setenv("RSIJEV_ADAPTIVE", "off")
    ap = argparse.ArgumentParser()
    add_serve_args(ap, positional=False)
    assert ap.parse_args([]).adaptive == "off"


def test_responses_report_the_depth_used(stub_lm, tmp_path, models):
    from fastapi.testclient import TestClient
    from serve.app import create_app
    from serve.batcher import model_worker
    from serve.decider import Decider
    from serve.server import load_for_serving, make_scorer
    _, _, ada, _ = models
    d = _write_release(tmp_path / "srv", ada, tau=-1.0)      # every adaptive question stops at AUX
    one = {"model": "srv", "state": "My card was charged twice.",
           "questions": {"refund": {"type": "noul", "instructions": "Does the user want a refund?"}}}
    many = {**one, "questions": {**one["questions"],
                                 "team": {"type": "choice", "instructions": "Which team?",
                                          "criteria": {"billing": "Payments", "tech": "Bugs"}}}}
    want = {"auto": ({"refund": EXIT}, {"refund": AUX, "team": AUX}),
            "on": ({"refund": AUX}, {"refund": AUX, "team": AUX}),
            "off": ({"refund": EXIT}, {"refund": EXIT, "team": EXIT})}
    for mode, (w1, wn) in want.items():
        s = load_for_serving(str(d), device="cpu", dtype="fp32", adaptive=mode)
        for scorer in (model_worker(s), make_scorer(s)):
            c = TestClient(create_app(scorer, served_model_name="srv"))
            assert c.post("/v1/systemone", json=one).json()["usage"]["depth"] == w1, mode
            assert c.post("/v1/systemone", json=many).json()["usage"]["depth"] == wn, mode
        dec = Decider.from_scorer(make_scorer(s), name="srv")
        assert dec.request(one["state"], many["questions"])["usage"]["depth"] == wn, mode
    fixed = load_for_serving(str(d), device="cpu", dtype="fp32", fixed_exit=True)
    c = TestClient(create_app(model_worker(fixed), served_model_name="srv"))
    assert c.post("/v1/systemone", json=many).json()["usage"]["depth"] == {"refund": EXIT, "team": EXIT}


def test_models_without_aux_exits_keep_their_usage(tok, models):
    from fastapi.testclient import TestClient
    from serve.app import create_app
    from serve.batcher import ModelRunner, GpuWorker
    _, fixed, _, _ = models
    fixed.adaptive_policy = None
    runner = ModelRunner(fixed, tok, ENC, spec_max_options=8, device="cpu")
    body = TestClient(create_app(GpuWorker(runner), served_model_name="m")).post(
        "/v1/systemone", json={"model": "m", "state": "Hello.",
                               "questions": {"a": {"type": "noul", "instructions": "Greeting?"}}}).json()
    assert set(body["usage"]) == {"input_tokens", "output_tokens"} and set(body) == {"model", "answers", "usage"}
