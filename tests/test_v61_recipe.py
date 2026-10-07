"""v6.1-VL's recipe is expressible by this repo's code, and its new pieces do what they say, on CPU.

* The member-B stage spec in data/v6.1-vl_recipe/ is a release_train spec accepted key by key; the
  calibration and policy records agree with each other and with what the package carries.
* scripts/soup_checkpoints.py: the uniform mean of two checkpoints, tower in its own dtype, heads
  in fp32; members whose forward pass or tensor names differ are refused.
* rsijev.exit_policy: the training-overlap exclusion (overlap_case_ids, drop_case_ids) and the
  single-tau threshold family with its 0.95 fallback.
* scripts/package_multiexit.py --temperatures / --single-tau: one temperature per exit from the
  refit (the main exit's in calibration.safetensors and calibration.json alike), read by serving.

    python -m pytest tests/test_v61_recipe.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))        # after the root: scripts/serve.py would shadow serve/

from rsijev import exit_policy as P                                # noqa: E402
from rsijev.arch import ArchConfig                                 # noqa: E402
from rsijev.fit import FitConfig                                   # noqa: E402

RECIPE = ROOT / "data" / "v6.1-vl_recipe"


def _spec(name):
    return json.loads((RECIPE / f"{name}.json").read_text())


# ---------------------------------------------------------------- the specs
def test_member_b_stage_spec_is_accepted_key_by_key():
    import run_arm_lib as lib
    spec = _spec("1-member-b-domain-mix")
    assert lib.unknown_spec_keys(spec) == []
    cfg = {**lib.DEFAULTS, **spec}
    fe = dict(cfg["fit_extra"])
    fc = FitConfig(**fe)
    for k, v in fe.items():
        assert getattr(fc, k) == v, k
    ArchConfig(max_options=cfg["max_options"], **dict(cfg["arch_extra"]))
    enc = lib.encode_config(cfg, cfg["option_order"])
    assert (enc.max_length, enc.truncate, enc.option_pool_own_tokens) == (32768, "middle", True)
    assert fe["aux_detach_tower"] is True and fe["aux_exit_weights"] == {"12": 0.3, "16": 0.3, "20": 0.3}
    assert fe["init_from"] == fe["aux_init_from"] == spec["init_parent"]["path"]


def test_release_train_resolves_member_b_paths(tmp_path):
    import release_train
    fe = release_train.resolve_paths(_spec("1-member-b-domain-mix"), tmp_path)["fit_extra"]
    assert fe["init_from"] == str(tmp_path / "b-domain-1/s17") and fe["aux_save_dir"] == str(tmp_path / "b-domain-2/s17")


def test_records_agree():
    cal, pol, soup = _spec("4-calibration"), _spec("5-exit-policy"), _spec("3-soup")
    assert soup["weights"] == [0.5, 0.5] and len(soup["members"]) == 2
    lt = cal["fitted"]["logT"]
    assert lt == {"12": -0.25, "16": -0.14, "20": -0.47, "32": 0.02}
    assert pol["auto"]["result"]["tau"] == {"16": 0.85, "20": 0.5} and pol["auto"]["result"]["CONFIRMED"] is True
    single = pol["single"]["result"]
    assert single["CONFIRMED"] is False and single["fallback"] is True and single["tau"] == {"16": 0.95, "20": 0.95}
    pkg = cal["package"]
    assert pkg["calibration.safetensors cal_logT"] == pkg["calibration.json main_logT"] == lt["32"]


# ---------------------------------------------------------------- the soup
def _ckpt(d: Path, seed: int, spec: dict, tower_dtype=torch.bfloat16, aux=True):
    from safetensors.torch import save_file
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(seed)
    save_file({"layers.0.w": torch.randn(4, 3, generator=g).to(tower_dtype), "norm.w": torch.randn(3, generator=g).to(tower_dtype)},
              str(d / "tower.safetensors"))
    save_file({"q.w": torch.randn(3, 2, generator=g)}, str(d / "scorer.safetensors"))
    if aux:
        save_file({"16.q.w": torch.randn(3, 2, generator=g), "20.q.w": torch.randn(3, 2, generator=g)},
                  str(d / "aux_scorers.safetensors"))
    (d / "meta.json").write_text(json.dumps({"spec": spec}))


SPEC = {"readout": "option_xattn", "layout": "state_first", "option_pool": "mean", "max_options": 160,
        "arch_extra": {"aux_exits": [16, 20], "exit_layer": 32}, "residual": False, "logit_cap": None,
        "head_input_norm": False, "readout_layer": -1}


def test_soup_is_the_mean_in_each_files_dtype(tmp_path):
    import soup_checkpoints as S
    from safetensors.torch import load_file
    _ckpt(tmp_path / "a", 0, SPEC)
    _ckpt(tmp_path / "b", 1, SPEC)
    rep = S.soup(tmp_path / "s", [tmp_path / "a", tmp_path / "b"])
    assert rep["files"] == list(S.FILES) and rep["soup"] == {"members": ["a", "b"], "weights": [0.5, 0.5], "kind": "uniform"}
    for fn in S.FILES:
        a, b, s = (load_file(str(tmp_path / x / fn)) for x in ("a", "b", "s"))
        for k in a:
            want = ((a[k].float() + b[k].float()) / 2).to(a[k].dtype if fn == "tower.safetensors" else torch.float32)
            assert s[k].dtype == want.dtype and torch.equal(s[k], want), (fn, k)
    meta = json.loads((tmp_path / "s" / "meta.json").read_text())
    assert meta["spec"] == SPEC and meta["soup"]["kind"] == "uniform"
    w = S.average([{"x": torch.tensor([0.0])}, {"x": torch.tensor([1.0])}], [0.3, 0.7])
    assert float(w["x"]) == pytest.approx(0.7)


def test_soup_refuses_members_that_differ(tmp_path):
    import soup_checkpoints as S
    _ckpt(tmp_path / "a", 0, SPEC)
    _ckpt(tmp_path / "b", 1, {**SPEC, "option_pool": "first"})
    with pytest.raises(ValueError, match="option_pool"):
        S.soup(tmp_path / "s", [tmp_path / "a", tmp_path / "b"])
    with pytest.raises(ValueError, match="tensor names"):
        S.average([{"x": torch.zeros(1)}, {"y": torch.zeros(1)}], [0.5, 0.5])
    with pytest.raises(ValueError, match="sum to 1"):
        S.average([{"x": torch.zeros(1)}, {"x": torch.zeros(1)}], [0.5, 0.6])
    _ckpt(tmp_path / "c", 2, SPEC, aux=False)
    with pytest.raises(ValueError, match="aux_scorers"):
        S.soup(tmp_path / "s2", [tmp_path / "a", tmp_path / "c"])


# ---------------------------------------------------------------- the exclusion
def test_overlap_by_case_id_and_by_content(tmp_path):
    q = [{"instructions": "Is it spam?", "options": ["yes", "no"]}]
    dev = [{"case_id": "d1", "state": "s1", "questions": q},
           {"case_id": "d2", "state": "s2", "questions": q},
           {"case_id": "d3", "state": "s3", "questions": q}]
    corpus = [{"case_id": "d1", "state": "other", "questions": q},          # same id
              {"case_id": "x9", "state": "s2", "questions": q},             # same content, new id
              {"case_id": "x8", "state": "s3", "questions": [{"instructions": "Other?", "options": ["a", "b"]}]}]
    assert P.overlap_case_ids(dev, corpus) == {"d1", "d2"}
    assert P.overlap_case_ids(dev, [{"case_id": "own7"}], dev_case_ids=["own7"]) == {"own7"}
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "train.jsonl").write_text("\n".join(json.dumps(r) for r in corpus) + "\nnot json\n")
    (root / "held.jsonl").write_text(json.dumps({"case_id": "d3", "state": "s3", "questions": q}) + "\n")
    assert [r["case_id"] for r in P.jsonl_rows(root)] == ["d1", "x9", "x8"]       # held-out file left out


def test_drop_case_ids_cuts_every_aligned_field():
    d = {"case_id": ["a", "b", "c"], "y": torch.tensor([0, 1, 2]), "tag": ["t0", "t1", "t2"],
         "z": {16: torch.arange(6.).view(3, 2), 32: torch.arange(6.).view(3, 2) + 10}, "exits": [16, 32]}
    k = P.drop_case_ids(d, {"b"})
    assert k["case_id"] == ["a", "c"] and k["y"].tolist() == [0, 2] and k["tag"] == ["t0", "t2"]
    assert k["z"][32].tolist() == [[10.0, 11.0], [14.0, 15.0]] and k["exits"] == [16, 32]
    vis = {"z": {32: torch.zeros(2, 2)}, "gold": torch.tensor([0, 1])}
    assert P.drop_case_ids(vis, {"b"}) is vis


# ---------------------------------------------------------------- the single tau
EX4 = [12, 16, 20, 32]


def _dump(n, seed, tags, case_ids, K=3):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, K, (n,), generator=g)
    z = {L: torch.nn.functional.pad(torch.randn(n, K, generator=g) + (1.0 + i) * torch.nn.functional.one_hot(y, K).float(),
                                    (0, 5 - K), value=float("-inf")) for i, L in enumerate(EX4)}
    return {"exits": EX4, "z": z, "mode": torch.zeros(n, dtype=torch.long), "y": y, "tag": tags, "case_id": case_ids}


@pytest.fixture(scope="module")
def th_sets(tmp_path_factory):
    d = tmp_path_factory.mktemp("pd")
    n = 400
    dev = _dump(n, 0, [f"td|{'cal' if i % 2 else 'tau'}" for i in range(n)], [f"d{i}" for i in range(n)])
    txt = _dump(n, 1, [f"{'mmlupro' if i % 4 == 0 else 'misc'}|{P.half_of(f'p{i}')}" for i in range(n)],
                [f"p{i}" for i in range(n)])
    vis = _dump(80, 2, [""] * 80, [""] * 80)
    vis = {"z": vis["z"], "mode": vis["mode"], "gold": vis["y"], "bench": ["vb1" if i % 2 else "vb2" for i in range(80)]}
    f = d / "probe_a.jsonl"
    f.write_text("\n".join(json.dumps({"case_id": f"v{i}", "questions": [{}]}) for i in range(80)) + "\n")
    v1 = P.load_v1(dev, txt, vis, [f])
    pd = d / "pd"
    (pd / "text").mkdir(parents=True)
    ncase = 60
    rows = [{"case_id": f"h{c}", "questions": [{"options": ["a", "b", "c"], "criteria": {}}] * 2} for c in range(ncase)]
    (pd / "text" / "di9.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:ncase // 2]) + "\n")
    (pd / "text" / "clinc_clean.jsonl").write_text("\n".join(json.dumps(r) for r in rows[ncase // 2:]) + "\n")
    m = 2 * ncase
    held = _dump(m, 5, ["di9|x"] * (m // 2) + ["clinc_clean|x"] * (m // 2), [f"h{i // 2}" for i in range(m)])
    return v1, held, pd


def test_single_tau_family_and_its_fallback(th_sets, monkeypatch):
    v1, held, pd = th_sets
    T = P.fit_temperatures(v1)
    th = P.Thresholds(v1, T, {5: 1.0, 9: 1.0}, {5: 0.0}, single=True)
    assert [p["tau"][16] for p in th.grid] == P.Thresholds.G_SINGLE and all(p["tau"][16] == p["tau"][20] for p in th.grid)
    assert th.c16 in th.pols and "t16=0.95,t20=0.95" in th.pols
    d = th.load_di([held], [pd])
    res = th.select(d, d)
    sel = res["selection"]
    assert sel["family"] == "single"
    if sel["CONFIRMED"]:
        assert sel["tau"]["16"] == sel["tau"]["20"] and "fallback" not in sel
    else:
        assert sel["fallback"] is True and sel["tau"] == {"16": 0.95, "20": 0.95}
    monkeypatch.setattr(P.Thresholds, "DCAP", 0.0)                   # nothing feasible on A
    res = th.select(d, d)
    sel = res["selection"]
    assert sel["feasible_A"] == [] and sel["CONFIRMED"] is False and sel["fallback"] is True
    assert sel["tau"] == {"16": 0.95, "20": 0.95} and set(sel["fallback_B"]) >= {"suite_U", "suite_depth", "sm_U"}
    per_exit = P.Thresholds(v1, T, {5: 1.0, 9: 1.0}, {5: 0.0})
    assert len(per_exit.grid) == len(P.Thresholds.G16) * len(P.Thresholds.G20) and "family" not in per_exit.select(d, d)["selection"]


# ---------------------------------------------------------------- the package
def test_package_from_refit_temperatures_is_read_by_serving(tmp_path):
    from safetensors.torch import load_file, save_file
    import package_multiexit as PM
    from serve.release import adaptive_block, auto_thresholds, exit_temperatures
    ck = tmp_path / "soup"
    ck.mkdir()
    save_file({"w": torch.zeros(2)}, str(ck / "tower.safetensors"))
    save_file({"w": torch.zeros(2)}, str(ck / "scorer.safetensors"))
    save_file({f"{L}.w": torch.full((2,), float(L)) for L in (12, 16, 20)}, str(ck / "aux_scorers.safetensors"))
    (ck / "meta.json").write_text(json.dumps({"spec": {"arch_extra": {"aux_exits": [12, 16, 20], "exit_layer": 32},
                                                       "fit_extra": {"aux_exit_weights": {"12": 0.3, "16": 0.3, "20": 0.3}}},
                                              "head_stage": {"max_length": 4096}}))
    T = {"12": -0.25, "16": -0.14, "20": -0.47, "32": 0.02}
    (tmp_path / "auto.json").write_text(json.dumps({"T": T, "selection": {"CONFIRMED": True, "tau": {"16": 0.85, "20": 0.5}}}))
    (tmp_path / "single.json").write_text(json.dumps({"T": T, "dev": {}, "selection": {
        "family": "single", "CONFIRMED": False, "fallback": True, "tau": {"16": 0.95, "20": 0.95},
        "fallback_B": {"suite_U": 0.7852, "suite_depth": 26.8}}}))
    out = tmp_path / "pkg"
    sys.argv = ["package_multiexit.py", "--ckpt", str(ck), "--temperatures", str(tmp_path / "auto.json"),
                "--single-tau", str(tmp_path / "single.json"), "--drop-exit", "12",
                "--thresholds", str(tmp_path / "auto.json"), "--out", str(out)]
    assert PM.main() == 0
    assert sorted(load_file(str(out / "aux_scorers.safetensors"))) == ["16.w", "20.w"]
    assert float(load_file(str(out / "calibration.safetensors"))["cal_logT"]) == pytest.approx(0.02)
    meta = json.loads((out / "meta.json").read_text())
    cal = json.loads((out / "calibration.json").read_text())
    assert cal["main_logT"] == 0.02 and set(cal["exits"]) == {"16", "20"}
    blk = adaptive_block(meta)
    assert blk["exits"] == [16, 20, 32] and blk["tau"] == 0.95 and blk["fallback"] is True and blk["confirmed"] is False
    assert auto_thresholds(blk, blk["exits"]) == {16: 0.85, 20: 0.5}
    temps = exit_temperatures(out, blk["exits"])
    assert {L: round(float(v["logT"]), 4) for L, v in temps.items()} == {16: -0.14, 20: -0.47}
    sys.argv = ["package_multiexit.py", "--ckpt", str(ck), "--temperatures", str(tmp_path / "auto.json"),
                "--out", str(tmp_path / "pkg2")]
    with pytest.raises(SystemExit):
        PM.main()
