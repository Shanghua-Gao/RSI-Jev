"""v4.0-VL's recipe is expressible by this repo's code, on CPU.

* Every stage spec in data/v4.0-vl_recipe/ -- asym-calA.json is the released
  checkpoint's own meta.json spec, verbatim -- is accepted key by key: the
  run-arm spec, FitConfig (with its `vision` and `rl2` blocks), ArchConfig and
  fit_rl2's config. A key this code does not route is an error here, because
  `FitConfig(**fit_extra)` or a DEFAULTS merge would otherwise train a different
  recipe without saying so.
* asym_loss equals the implementation the stage was trained with, on random
  inputs, and its optimum is the closed form in its docstring.
* VisionFeed feeds a training batch that scores exactly like the gated offline
  image path, on the tiny random Qwen3.5 of tests/test_vision_model.py.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))        # after the root: scripts/serve.py would shadow serve/

RECIPE = ROOT / "data" / "v4.0-vl_recipe"
STAGES = ["v4-rep-b-s17ck", "vis-ct-1", "rl5-listwise-vis", "vis-ct-3", "calA-ce", "asym-calA"]


def _spec(name):
    return json.loads((RECIPE / f"{name}.json").read_text())


# ---------------------------------------------------------------- spec acceptance
@pytest.mark.parametrize("stage", STAGES)
def test_stage_spec_is_accepted_key_by_key(stage):
    import run_arm_lib as lib
    from rsijev.arch import ArchConfig
    from rsijev.fit import FitConfig
    from rsijev.rl2 import MODES_RL2, unknown_rl2_keys
    from rsijev.train import RLConfig

    spec = _spec(stage)
    assert lib.unknown_spec_keys(spec) == []
    cfg = {**lib.DEFAULTS, **spec}
    fe = dict(cfg["fit_extra"])
    fc = FitConfig(**fe)                          # raises on any key FitConfig lacks
    for k, v in fe.items():
        assert getattr(fc, k) == v, k
    ArchConfig(max_options=cfg["max_options"], **dict(cfg["arch_extra"]))
    RLConfig(**dict(cfg["rl_extra"] or {}))
    if fc.rl2:
        assert unknown_rl2_keys(fc.rl2) == []
        assert fc.rl2["mode"] in MODES_RL2
    if fc.vision:
        roots = fc.vision.get("roots") or [fc.vision["root"]]
        assert all(isinstance(r, str) for r in roots) and int(fc.vision["budget"]) > 0


def test_released_spec_routes_to_the_asym_stage():
    from rsijev.rl2 import DEFAULTS, MODE_SOURCE
    spec = _spec("asym-calA")
    r = {**DEFAULTS, **spec["fit_extra"]["rl2"]}
    assert r["mode"] == "asym" and (r["lam_over"], r["lam_under"]) == (4.0, 1.0)
    assert r["replay"] is False and MODE_SOURCE["asym"] == "rl_slice"
    # rl_slice_vis (the image rows) is trained through the rl_slice prefix; the
    # calibration and dev pools are not, and none of the four is replay
    srcs = spec["sources"].split(",")
    assert [s for s in srcs if s.startswith(MODE_SOURCE["asym"])] == ["rl_slice", "rl_slice_vis"]
    assert all(s.startswith("rl_") for s in srcs)
    assert spec["fit_extra"]["vision"]["roots"] == ["vision_v1", "vision_v2", "vision_v3"]


def test_stages_chain_through_init_parent():
    for parent, child in zip(STAGES, STAGES[1:]):
        c = _spec(child)
        assert c["fit_extra"]["init_from"] == f"{parent}/s17" == c["init_parent"]["path"]
        assert c["init_parent"]["parent_steps"] == _spec(parent)["steps"]


def test_release_train_resolves_relative_paths(tmp_path):
    import release_train
    spec = release_train.resolve_paths(_spec("asym-calA"), tmp_path)
    fe = spec["fit_extra"]
    assert fe["init_from"] == str(tmp_path / "calA-ce/s17")
    assert fe["vision"]["roots"] == [str(tmp_path / r) for r in ("vision_v1", "vision_v2", "vision_v3")]
    assert fe["rl2"]["cal_save_dir"] == str(tmp_path / "asym-calA/s17")
    assert _spec("asym-calA")["fit_extra"]["init_from"] == "calA-ce/s17"      # not mutated


def test_unknown_keys_are_reported():
    import run_arm_lib as lib
    from rsijev.rl2 import unknown_rl2_keys
    assert lib.unknown_spec_keys({"steps": 1, "why": "x", "lam_ovr": 4}) == ["lam_ovr"]
    assert unknown_rl2_keys({"mode": "asym", "lam_ovr": 4.0}) == ["lam_ovr"]


# ---------------------------------------------------------------- asym_loss
def _asym_reference(logits, gold_idx, lam_over, lam_under):
    """The trained implementation, restated: first-max top-1, R = c - lo*max(0,p-c)^2
    - lu*max(0,c-p)^2, loss = -mean R."""
    lp = torch.log_softmax(logits.double(), -1)
    z = logits.double().masked_fill(~torch.isfinite(logits), -1e30)
    top = z.argmax(-1)
    p = lp.gather(1, top[:, None]).squeeze(1).exp()
    c = (top == gold_idx).double()
    R = c - lam_over * torch.clamp(p - c, min=0) ** 2 - lam_under * torch.clamp(c - p, min=0) ** 2
    return -R.mean(), R.mean(), p.mean()


@pytest.mark.parametrize("lam", [(4.0, 1.0), (1.0, 1.0), (2.5, 0.5)])
def test_asym_loss_matches_reference(lam):
    from rsijev.rl2 import asym_loss
    g = torch.Generator().manual_seed(0)
    for n, k in [(16, 2), (16, 4), (7, 10)]:
        logits = torch.randn(n, k, generator=g) * 3
        logits[0, -1] = float("-inf")                       # a padded option slot
        gold = torch.randint(0, k - 1, (n,), generator=g)
        loss, rew, conf = asym_loss(logits, gold, lam_over=lam[0], lam_under=lam[1])
        rl, rr, rc = _asym_reference(logits, gold, *lam)
        assert torch.allclose(loss.double(), rl, atol=1e-6)
        assert abs(rew - float(rr)) < 1e-6 and abs(conf - float(rc)) < 1e-6


def test_asym_loss_gradient_and_optimum():
    from rsijev.rl2 import asym_loss
    logits = torch.randn(32, 4, requires_grad=True)
    gold = torch.randint(0, 4, (32,))
    loss, _, _ = asym_loss(logits, gold, lam_over=4.0, lam_under=1.0)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    # expected reward at confidence p when P(correct) = q is maximised at
    # p* = q*lu / (q*lu + (1-q)*lo)
    lo, lu = 4.0, 1.0
    ps = torch.linspace(0.001, 0.999, 9999, dtype=torch.float64)
    for q in (0.3, 0.6, 0.9):
        er = q * (1 - lu * (1 - ps) ** 2) + (1 - q) * (-lo * ps ** 2)
        assert abs(float(ps[er.argmax()]) - q * lu / (q * lu + (1 - q) * lo)) < 1e-3


# ---------------------------------------------------------------- VisionFeed
BASE = "Qwen/Qwen3.5-2B-Base"


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("PIL")
    pytest.importorskip("torchvision")
    from serve.runtime import keep_fused_kernels_off
    keep_fused_kernels_off("cpu")
    try:
        from transformers import AutoTokenizer
        from rsijev.vision import ImagePrep, VisionConfig
        tok = AutoTokenizer.from_pretrained(BASE)
        ImagePrep(BASE, VisionConfig(image_token_budget=64))
    except Exception as e:                                      # offline, no cache
        pytest.skip(f"Qwen3.5 tokenizer / image processor unavailable: {e}")
    from transformers.models.qwen3_5.configuration_qwen3_5 import (Qwen3_5TextConfig,
                                                                   Qwen3_5VisionConfig)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5VisionModel
    from rsijev.arch import ArchConfig, DecisionModel
    torch.manual_seed(0)
    tc = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                           num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                           head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                           linear_key_head_dim=16, linear_value_head_dim=16,
                           layer_types=["linear_attention"] * 3 + ["full_attention"],
                           rope_parameters={"rope_type": "default", "rope_theta": 1e7,
                                            "partial_rotary_factor": 0.25,
                                            "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    tc._attn_implementation = "sdpa"
    tower = Qwen3_5TextModel(tc).eval()
    vc = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
                             out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                             temporal_patch_size=2, in_channels=3, num_position_embeddings=64)
    vc._attn_implementation = "sdpa"
    visual = Qwen3_5VisionModel(vc).to(torch.bfloat16).eval()
    arch = ArchConfig(max_options=8, freeze_base=True, readout="option_xattn",
                      xattn_combine="mlp", xattn_mlp_hidden=16)
    tm = DecisionModel(tower, 64, arch).eval()
    return tok, tm, visual, arch


def _vision_root(tmp_path):
    from PIL import Image
    from rsijev.contract import Case, Question
    root = tmp_path / "vision_x"
    (root / "images").mkdir(parents=True)
    Image.new("RGB", (96, 64), (220, 20, 20)).save(root / "images" / "a.png")
    b = Image.new("RGB", (64, 64), (20, 20, 220))
    b.paste((250, 250, 250), (16, 16, 48, 48))
    b.save(root / "images" / "b.png")
    q1 = Question("q", "noul", "Is the square red?", ("false", "true"),
                  {"false": "No.", "true": "Yes."})
    q2 = Question("c", "choice", "Which colour is it?", ("red", "green", "blue"),
                  {"red": "Red.", "green": "Green.", "blue": "Blue."})
    img = Case("vis_x:1", "vis_x", "Left: <image>\nRight: <image>\nA note.", (q1, q2),
               {"q": (0.0, 1.0), "c": (1.0, 0.0, 0.0)})
    txt = Case("txt:1", "st_x", "A plain note about a red square.", (q1,), {"q": (0.0, 1.0)})
    with open(root / "vis_x.jsonl", "w") as fh:
        fh.write(json.dumps({"case_id": img.case_id, "images": ["images/a.png", "images/b.png"]}) + "\n")
    return root, img, txt


def test_vision_feed_batch_equals_the_offline_image_path(tiny, tmp_path, monkeypatch):
    import rsijev.vision_fit as vf
    from rsijev.encode import EncodeConfig, collate, encode_question, unpermute_logits
    from rsijev.vision import (IMAGE_PAD, ImagePrep, VisionConfig, VisionDecisionModel,
                               encode_vision_question, vision_collate)
    from PIL import Image
    tok, tm, visual, arch = tiny
    monkeypatch.setattr(vf, "load_visual", lambda mid: visual)
    root, img, txt = _vision_root(tmp_path)
    feed = vf.VisionFeed({"roots": [str(root)], "budget": 64, "model": BASE}, tok, tm, "cpu")
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical",
                       max_length=2048 + 64)
    chunk = [(img, img.questions[0]), (img, img.questions[1]), (txt, txt.questions[0])]
    with torch.no_grad():
        b = feed.batch(chunk, enc, random.Random(0), 8)
        assert "inputs_embeds" in b and "pixel_values" not in b and feed.n_image_rows == 2
        got = unpermute_logits(tm(**b).float(), b["option_perm"], b["option_mask"])

        # the gated offline path: VisionDecisionModel runs the ViT itself, per question
        vm = VisionDecisionModel(tm.tower, 64, arch, visual=visual,
                                 image_token_id=tok.convert_tokens_to_ids(IMAGE_PAD),
                                 vcfg=VisionConfig(image_token_budget=64)).eval()
        vm.scorer.load_state_dict(tm.scorer.state_dict())
        prep = ImagePrep(BASE, VisionConfig(image_token_budget=64))
        ims = [Image.open(root / p).convert("RGB") for p in ("images/a.png", "images/b.png")]
        for r, (c, q) in enumerate(chunk[:2]):
            bt = vision_collate(tok, [encode_vision_question(tok, prep, c.state, ims, q, enc)],
                                8, device="cpu")
            ref = unpermute_logits(vm(**bt).float(), bt["option_perm"], bt["option_mask"])[0]
            n = len(q.options)
            assert torch.allclose(got[r, :n], ref[:n], atol=1e-4), (got[r, :n], ref[:n])
        # a text row in a mixed batch is the plain encoder's row
        bt = collate(tok, [encode_question(tok, txt.state, txt.questions[0], enc)], 8, device="cpu")
        ref = unpermute_logits(tm(**bt).float(), bt["option_perm"], bt["option_mask"])[0]
        assert torch.allclose(got[2, :2], ref[:2], atol=1e-4)


def test_vision_feed_text_only_batch_is_the_plain_batch(tiny, tmp_path, monkeypatch):
    import rsijev.vision_fit as vf
    from rsijev.encode import EncodeConfig, collate, encode_question
    tok, tm, visual, _ = tiny
    monkeypatch.setattr(vf, "load_visual", lambda mid: visual)
    root, _, txt = _vision_root(tmp_path)
    feed = vf.VisionFeed({"root": str(root), "budget": 64, "model": BASE}, tok, tm, "cpu")
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="shuffled")
    b = feed.batch([(txt, txt.questions[0])], enc, random.Random(3), 8)
    ref = collate(tok, [encode_question(tok, txt.state, txt.questions[0], enc, rng=random.Random(3))],
                  8, device="cpu")
    assert set(b) == set(ref) and all(torch.equal(b[k], ref[k]) for k in ref)
    assert feed.n_image_rows == 0


def test_vision_feed_refuses_a_case_id_in_two_roots(tiny, tmp_path, monkeypatch):
    import shutil
    import rsijev.vision_fit as vf
    tok, tm, visual, _ = tiny
    monkeypatch.setattr(vf, "load_visual", lambda mid: visual)
    root, _, _ = _vision_root(tmp_path)
    shutil.copytree(root, tmp_path / "vision_y")
    with pytest.raises(ValueError, match="two vision roots"):
        vf.VisionFeed({"roots": [str(root), str(tmp_path / "vision_y")], "budget": 64,
                       "model": BASE}, tok, tm, "cpu")


def test_asym_stage_trains_image_rows_end_to_end(tiny, tmp_path, monkeypatch):
    """fit() with the released rl2 + vision blocks, on the tiny model: routes to fit_rl2's
    asym mode, trains the rl_slice and rl_slice_vis rows (images through the feed), never
    the calibration or dev pools, fits and saves cal-4b without activating it."""
    import copy
    import rsijev.vision_fit as vf
    from rsijev.arch import DecisionModel
    from rsijev.contract import Case, Question
    from rsijev.encode import EncodeConfig
    from rsijev.fit import FitConfig, fit
    tok, tm, visual, arch = tiny
    monkeypatch.setattr(vf, "load_visual", lambda mid: visual)
    root, img, _ = _vision_root(tmp_path)
    q = Question("q", "noul", "Is it true?", ("false", "true"), {"false": "No.", "true": "Yes."})

    def case(cid, src, i):
        return Case(cid, src, f"Note {i}: the claim is {'true' if i % 2 else 'false'}.", (q,),
                    {"q": (0.0, 1.0) if i % 2 else (1.0, 0.0)})
    vis_row = Case(img.case_id, "rl_slice_vis", img.state, img.questions, img.gold)
    cases = ([case(f"s{i}", "rl_slice", i) for i in range(6)] + [vis_row]
             + [case(f"rl_calp|td_train|c{i}~0", "rl_calp", i) for i in range(40)]
             + [case(f"d{i}", "rl_dev", i) for i in range(4)])
    model = DecisionModel(copy.deepcopy(tm.tower).float(), 64,
                          type(arch)(**{**arch.__dict__, "freeze_base": False})).float()
    rl2 = dict(_spec("asym-calA")["fit_extra"]["rl2"], rows_per_step=4,
               cal_save_dir=str(tmp_path / "cal"))
    cfg = FitConfig(steps=3, batch_size=4, lr_head=5e-5, lr_base=2e-6, keep_last_k=1,
                    lower_layers_n=2, lower_layers_lr_scale=0.1, log_every=1,
                    vision={"roots": [str(root)], "budget": 64, "model": BASE}, rl2=rl2)
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="shuffled",
                       max_length=2048 + 64)
    out = fit(model, tok, cases, enc, cfg, max_options=8, seed=17, device="cpu")
    rows = [h for h in out["history"] if h.get("step", -1) >= 0]
    assert any("rl_reward_avg" in h for h in rows)
    assert max(h.get("image_rows", 0) for h in rows) > 0      # the image row went through the feed
    summary = rows[-1]
    assert summary["rl2_mode"] == "asym" and summary["rl2_rows"] == 8     # 6 text + the image case's 2
    assert (tmp_path / "cal" / "calibration.json").exists() and model.cal_mode == "none"


# ---------------------------------------------------------------- calibration pool (v4.0-VL)
def _held_ids(n, want):
    import fit_release_calibration as F
    return [f"{'h' if want else 't'}{i}" for i in range(1000)
            if F.held(f"{'h' if want else 't'}{i}") == want][:n]


def test_calibration_lineage_guard_and_image_holdout(tmp_path):
    import fit_release_calibration as F
    assert F.group_of("rp_rp_rp_st_kev_hard_v1") == "kev_hard" == F.group_of("st_kev_hard_v1")

    def row(cid, state, images=None):
        r = {"case_id": cid, "source": "x", "state": state, "questions": [
            {"key": "q", "mode": "noul", "instructions": "?", "options": ["false", "true"],
             "criteria": {"false": "No.", "true": "Yes."}}], "gold": {"q": [0.0, 1.0]}}
        if images:
            r["images"] = images
        return json.dumps(r) + "\n"
    held, trained = _held_ids(3, True), _held_ids(3, False)
    # parent: trained text state "S-up" and trained image "b.png"; child: an RL stage whose
    # cal/dev pools are never trained
    (tmp_path / "p_corpus").mkdir()
    (tmp_path / "p_corpus" / "st_a.jsonl").write_text(row(trained[0], "S-up") + row(held[0], "S-held"))
    (tmp_path / "p_corpus" / "vis_a.jsonl").write_text(row(trained[1], "I", ["images/b.png"]))
    (tmp_path / "c_corpus").mkdir()
    (tmp_path / "c_corpus" / "rl_slice.jsonl").write_text(row(trained[2], "S-child"))
    (tmp_path / "c_corpus" / "rl_calp.jsonl").write_text(row("rl_calp|x|" + trained[0], "S-cal"))
    (tmp_path / "parent" / "s17").mkdir(parents=True)
    (tmp_path / "parent" / "s17" / "meta.json").write_text(json.dumps({"spec": {
        "corpus_dir": "p_corpus", "sources": "st_a,vis_a", "fit_extra": {}}}))
    spec = {"corpus_dir": "c_corpus", "sources": "rl_slice,rl_calp",
            "fit_extra": {"rl2": {"cal_pool": "rl_calp", "dev_pool": "rl_dev"}},
            "init_parent": {"path": "parent/s17"}}
    chain = F.lineage(spec, tmp_path)
    assert [c.name for _, c, _ in chain] == ["c_corpus", "p_corpus"]
    assert chain[0][2] == ["rl_slice"]                      # the cal pool is not trained
    for sp_, corpus_, _ in chain:
        sp_["_corpus"] = str(corpus_)
    states = F.lineage_trained_states(chain)
    assert {"S-up", "S-child"} <= states and "S-held" not in states and "S-cal" not in states

    # image holdout: held cases of the dev root, minus any whose image set was trained upstream
    vroot = tmp_path / "vision_v1"
    vroot.mkdir()
    (vroot / "vis_a.jsonl").write_text(row(held[1], "I1", ["images/a.png"])
                                       + row(held[2], "I2", ["images/b.png"])
                                       + row(trained[1], "I3", ["images/b.png"]))
    idx = F.image_index([vroot])
    timgs = F.lineage_trained_images(chain, idx)
    assert ("b.png",) in timgs
    dev_v, nq, n_held, vleak = F.image_dev_pool(vroot, idx, timgs, 2400)
    assert [c.case_id for c, _, _ in dev_v] == [held[1]] and (nq, n_held, vleak) == (1, 1, 1)
    assert dev_v[0][1] == [str(vroot / "images/a.png")] and dev_v[0][2] == "vis"
