"""The v4.0-VL stage-corpus builders, on tiny synthetic corpora (CPU, no model, no data).

  ct_build_corpus.py      replay budget, source-stratified proportions, seeded determinism
  build_stage_corpora.py  rep-b cap, rl5-vis vision share, calasym-stage rl_slice_vis rows
  build_rl2_calpool.py    --splits takes only the parent's holdout, ids leave the holdout
  depthdial.py            valid rsijev Cases whose labels an independent solver agrees with
  build_depthdial.py      bands and the soft-target rules

    python -m pytest tests/test_stage_builders.py -q
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_depthdial as BD      # noqa: E402
import build_stage_corpora as BS  # noqa: E402
import depthdial as DD            # noqa: E402
from rsijev.contract import load_cases  # noqa: E402


def held(cid: str) -> bool:
    return int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 10 == 0


def case(cid, source, nq=1, k=2, y=0, **extra):
    qs = [{"key": f"q{i}", "mode": "choice", "instructions": "pick", "options": [f"o{j}" for j in range(k)],
           "criteria": {f"o{j}": f"o{j}" for j in range(k)}} for i in range(nq)]
    return {"case_id": cid, "source": source, "state": f"state of {cid}", "questions": qs,
            "gold": {f"q{i}": [1.0 if j == y else 0.0 for j in range(k)] for i in range(nq)}, **extra}


def write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def nq(path: Path) -> int:
    return sum(len(json.loads(l)["questions"]) for l in open(path) if l.strip())


def ct(parent, sources, steps, out, new=(), seed=17, batch=16):
    cmd = [sys.executable, str(SCRIPTS / "ct_build_corpus.py"), "--parent-corpus", str(parent),
           "--parent-sources", sources, "--steps", str(steps), "--batch", str(batch),
           "--out", str(out), "--seed", str(seed)]
    if new:
        cmd += ["--new", *map(str, new)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads((out / "manifest.json").read_text())


def files(d: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(d.glob("*.jsonl"))}


# ---------------------------------------------------------------- replay mixing
@pytest.fixture
def parent(tmp_path):
    p = tmp_path / "parent"
    for s, n in (("a", 600), ("b", 300), ("c", 100)):
        write(p / f"{s}.jsonl", [case(f"{s}:{i}", s) for i in range(n)])
    return p


def test_replay_fills_the_budget_in_source_proportion(tmp_path, parent):
    man = ct(parent, "a,b,c", 25, tmp_path / "ctl")          # 400 of 1,000 questions
    assert man["budget_questions"] == 400 == man["replay_questions"]
    got = {k: v["questions"] for k, v in man["files"].items()}
    assert set(got) == {"rp_a", "rp_b", "rp_c"}
    for s, share in (("rp_a", .6), ("rp_b", .3), ("rp_c", .1)):
        assert abs(got[s] - 400 * share) <= 1, got
    assert man["sources"] == "rp_a,rp_b,rp_c"


def test_replay_is_deterministic_and_seeded(tmp_path, parent):
    ct(parent, "a,b,c", 25, tmp_path / "x")
    ct(parent, "a,b,c", 25, tmp_path / "y")
    ct(parent, "a,b,c", 25, tmp_path / "z", seed=18)
    assert files(tmp_path / "x") == files(tmp_path / "y")
    assert files(tmp_path / "x") != files(tmp_path / "z")


def test_data_arm_is_new_files_plus_a_prefix_of_the_controls_replay(tmp_path, parent):
    new = write(tmp_path / "new" / "pf_x__new.jsonl", [case(f"n:{i}", "pf_x__new") for i in range(150)])
    arm = ct(parent, "a,b,c", 25, tmp_path / "arm", new=[new])
    ctl = ct(parent, "a,b,c", 25, tmp_path / "ctl")
    assert arm["new_questions"] == 150 and arm["replay_questions"] == 250
    assert (tmp_path / "arm" / "pf_x__new.jsonl").read_bytes() == new.read_bytes()
    for s in ("rp_a", "rp_b", "rp_c"):
        a_lines = (tmp_path / "arm" / f"{s}.jsonl").read_text().splitlines()
        c_lines = (tmp_path / "ctl" / f"{s}.jsonl").read_text().splitlines()
        assert c_lines[:len(a_lines)] == a_lines
    with pytest.raises(subprocess.CalledProcessError):     # new data may not fill the budget
        ct(parent, "a,b,c", 5, tmp_path / "over", new=[new])


# ---------------------------------------------------------------- rep-b
def test_rep_b_caps_jsg_at_3k_stratified_and_copies_the_rest(tmp_path):
    ds2 = tmp_path / "ds2"
    write(ds2 / "synth.jsonl", [case(f"s:{i}", "synth") for i in range(10)])
    write(ds2 / "jsg_emotion.jsonl", [case(f"jsg:emo:{i}", "jsg_emotion", k=3, y=i % 3) for i in range(6000)])
    write(ds2 / "jsg_sst5.jsonl", [case(f"jsg:sst:{i}", "jsg_sst5", k=2, y=int(i % 4 == 0)) for i in range(4000)])
    write(ds2 / "jsg_anli.jsonl", [case(f"jsg:anli:r{1 + i % 3}:{i}", "jsg_anli", k=3, y=(i // 3) % 3) for i in range(9000)])
    BS.main(["rep-b", "--root", str(tmp_path / "R"), "--ds2", str(ds2)])
    d = tmp_path / "R" / BS.REP_B
    assert (d / "synth.jsonl").read_bytes() == (ds2 / "synth.jsonl").read_bytes()
    emo = [json.loads(l) for l in open(d / "jsg_emotion.jsonl")]
    assert len(emo) == 3000
    assert Counter(r["gold"]["q0"].index(1.0) for r in emo) == {0: 1000, 1: 1000, 2: 1000}
    sst = [json.loads(l) for l in open(d / "jsg_sst5.jsonl")]
    assert Counter(r["gold"]["q0"].index(1.0) for r in sst) == {0: 2250, 1: 750}
    anli = [json.loads(l) for l in open(d / "jsg_anli.jsonl")]
    assert Counter(r["case_id"].split(":")[2] for r in anli) == {"r1": 999, "r2": 999, "r3": 999}   # 333 per (round, label)
    first = (d / "jsg_anli.jsonl").read_bytes()
    BS.main(["rep-b", "--root", str(tmp_path / "R2"), "--ds2", str(ds2)])
    assert (tmp_path / "R2" / BS.REP_B / "jsg_anli.jsonl").read_bytes() == first


# ---------------------------------------------------------------- rl5-vis
def test_rl5_vis_keeps_text_replay_and_a_fifth_vision(tmp_path):
    root = tmp_path / "R"
    src = root / "vis-ct-1"
    write(src / "rp_a.jsonl", [case(f"a:{i}", "a") for i in range(800)])
    write(src / "vis_x.jsonl", [case(f"vx:{i}", "vis_x") for i in range(1500)])
    write(src / "vis_y.jsonl", [case(f"vy:{i}", "vis_y", nq=2) for i in range(500)])
    hrr = write(tmp_path / "rl_hrr.jsonl", [case("h:0", "rl_hrr")])
    BS.main(["rl5-vis", "--root", str(root), "--rl-hrr", str(hrr)])
    out = root / "rl5-vis"
    assert (out / "rp_a.jsonl").is_symlink() and (out / "rl_hrr.jsonl").is_symlink()
    rep = json.loads((out / "rl5_vis.report.json").read_text())
    assert rep["text_replay_q"] == 800 and rep["vision_frac_kept"] == 0.08
    assert abs(rep["vision_share"] - 0.2) < 0.03
    src_ids = {json.loads(l)["case_id"] for f in src.glob("vis_*.jsonl") for l in open(f)}
    kept = [json.loads(l) for f in out.glob("vis_*.jsonl") for l in open(f)]
    assert {r["case_id"] for r in kept} <= src_ids                      # case ids unchanged
    assert all(int(hashlib.sha256(("rl5vis:" + r["case_id"]).encode()).hexdigest(), 16) % 10000 < 800 for r in kept)


# ---------------------------------------------------------------- rl2-b2 and calasym-stage
def test_rl2_splits_take_only_the_holdout_and_leave_it(tmp_path):
    p = tmp_path / "corpus"
    write(p / "a.jsonl", [case(f"a:{i}", "a") for i in range(3000)])
    write(p / "b.jsonl", [case(f"b:{i}", "b") for i in range(2000)])
    BS.main(["rl2-b2", "--root", str(tmp_path / "R"), "--ds2", str(p), "--sources", "a,b"])
    out = tmp_path / "R" / "rl2-b2-src"
    n_hold = sum(held(f"{s}:{i}") for s, n in (("a", 3000), ("b", 2000)) for i in range(n))
    rows = {s: [json.loads(l) for l in open(out / f"{s}.jsonl")] for s in ("rl_slice", "rl_calp", "rl_dev")}
    assert sum(map(len, rows.values())) == n_hold
    for s, frac in (("rl_slice", .5), ("rl_calp", .3), ("rl_dev", .2)):
        assert abs(len(rows[s]) / n_hold - frac) < .07
        for r in rows[s]:
            assert r["source"] == s and not held(r["case_id"])
            name, stem, orig = r["case_id"].rsplit("~", 1)[0].split("|")
            assert name == s and held(orig) and orig.startswith(stem + ":")


def test_calasym_stage_renames_vision_rows_and_hits_the_fraction(tmp_path):
    root = tmp_path / "R"
    rl = root / "rl2-b2-src"
    write(rl / "rl_slice.jsonl", [case(f"sl:{i}", "rl_slice") for i in range(1400)])
    write(rl / "rl_calp.jsonl", [case(f"cp:{i}", "rl_calp") for i in range(30)])
    write(rl / "rl_dev.jsonl", [case(f"dv:{i}", "rl_dev") for i in range(20)])
    cal = root / "calA-ce"
    write(cal / "rp_rp_cov_x.jsonl", [case(f"t:{i}", "rp_rp_cov_x") for i in range(500)])
    write(cal / "dd_lies_near.jsonl", [case(f"d:{i}", "dd_lies_near") for i in range(100)])
    write(cal / "rp_rp_vis_ai2d.jsonl", [case(f"v1:{i}", "rp_rp_vis_ai2d") for i in range(1000)])
    write(cal / "rp_vis3_grid.jsonl", [case(f"v3:{i}", "rp_vis3_grid", nq=3) for i in range(400)])
    BS.main(["calasym-stage", "--root", str(root)])
    out = root / "calasym-stage"
    for s in ("rl_slice", "rl_calp", "rl_dev"):
        assert (out / f"{s}.jsonl").read_bytes() == (rl / f"{s}.jsonl").read_bytes()
    man = json.loads((out / "manifest.json").read_text())
    target = round(.3 / .7 * 1400)
    assert man["slice_q"] == 1400 and target <= man["vis_q"] <= target + 3
    assert abs(man["vis_frac"] - .30) < .002
    by_id = {json.loads(l)["case_id"]: json.loads(l) for f in cal.glob("*vis*.jsonl") for l in open(f)}
    vis = [json.loads(l) for l in open(out / "rl_slice_vis.jsonl")]
    assert sum(len(r["questions"]) for r in vis) == man["vis_q"]
    for r in vis:
        o = by_id[r["case_id"]]                                          # case_id unchanged
        assert r["source"] == "rl_slice_vis" and r["vis_origin"] == o["source"]
        assert {k: v for k, v in r.items() if k not in ("source", "vis_origin")} == \
               {k: v for k, v in o.items() if k != "source"}
    # each vision source contributes in proportion to its questions (1000 q vs 1200 q)
    assert abs(man["per_source_q"]["rp_rp_vis_ai2d"] / man["per_source_q"]["rp_vis3_grid"] - 1000 / 1200) < .02


# ---------------------------------------------------------------- depth-dial generators
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]


def solve_lies(state, q):
    lines = state.split("\n")
    val = {}
    if "honest" in lines[0]:
        m = re.search(r"(\w+) is (honest|dishonest)\.$", lines[0]); val[m[1]] = m[2] == "honest"
        for l in lines[1:]:
            m = re.fullmatch(r"(\w+) states that (\w+) is (honest|dishonest)\.", l)
            val[m[1]] = (m[3] == "honest") == val[m[2]]
        who = re.fullmatch(r"Is (\w+) honest\?", q)[1]
    elif "lamp" in lines[0].lower():
        m = re.search(r"Lamp 1 is (on|off)\.$", lines[0]); val["1"] = m[1] == "on"
        for l in lines[1:]:
            m = re.fullmatch(r"Lamp (\d+) is wired to lamp (\d+): it is (in the same state as|in the opposite state to) "
                             r"lamp \d+\.", l)
            val[m[1]] = val[m[2]] if m[3].startswith("in the same") else not val[m[2]]
        who = re.fullmatch(r"Is lamp (\d+) on\?", q)[1]
    else:
        m = re.search(r"(note \w) is (accurate|inaccurate)\.$", lines[0].lower()); val[m[1]] = m[2] == "accurate"
        for l in lines[1:]:
            m = re.fullmatch(r"(note \w) says that (note \w) is (accurate|inaccurate)\.", l.lower())
            val[m[1]] = (m[3] == "accurate") == val[m[2]]
        who = re.fullmatch(r"is (note \w) accurate\?", q.lower())[1]
    return len(lines) - 1, "yes" if val[who] else "no"


def solve_swap(state, q):
    lines = state.split("\n")
    head = lines[0].split(": ", 1)[1].rstrip(".")
    cur = {}
    for part in head.split("; "):
        h, it = re.fullmatch(r"(.+?) (?:has|contains) the (.+)", part).groups()
        cur[h] = it
    for l in lines[1:]:
        hs = [h for h in cur if re.search(r"(?<!\w)" + re.escape(h) + r"(?!\w)", l)]
        assert len(hs) == 2, (l, hs)
        cur[hs[0]], cur[hs[1]] = cur[hs[1]], cur[hs[0]]
    who = re.fullmatch(r"At the end, which item does (.+) (?:have|contain)\?", q)[1]
    return len(lines) - 1, "the " + cur[who]


def solve_order(state, q):
    lines = state.split("\n")
    objs = lines[0].split(": ", 1)[1].rstrip(".").split(", ")
    N = len(objs)
    cons = []
    for l in lines[1:]:
        x = next(o for o in objs if l.lower().startswith(o.lower() + " "))
        rest = l[len(x) + 1:-1]
        m = re.fullmatch(r"(?:is|finished) (\w+) from (the left|the right|last)", rest) or \
            re.fullmatch(r"finished (\w+)", rest)
        if m and m[1] in ORD:
            i = ORD.index(m[1])
            j = N - 1 - i if m.lastindex == 2 and m[2] in ("the right", "last") else i
            cons.append(lambda p, x=x, j=j: p.index(x) == j)
            continue
        for pat, f in ((r"(?:is immediately left of|finished directly before) (.+)", lambda a, b: a + 1 == b),
                       (r"(?:is left of|finished before) (.+)", lambda a, b: a < b),
                       (r"(?:is right of|finished after) (.+)", lambda a, b: a > b)):
            m = re.fullmatch(pat, rest)
            if m:
                y = next(o for o in objs if o == m[1])
                cons.append(lambda p, x=x, y=y, f=f: f(p.index(x), p.index(y)))
                break
        else:
            raise AssertionError(l)
    sols = [p for p in itertools.permutations(objs) if all(c(p) for c in cons)]
    assert len(sols) == 1, (state, len(sols))
    k = ORD.index(re.fullmatch(r"(?:Which item is|Who finished) (\w+)(?: from the left)?\?", q)[1])
    return N, sols[0][k]


@pytest.fixture(scope="module")
def items(tmp_path_factory):
    p = tmp_path_factory.mktemp("dd") / "gen_test.jsonl"
    DD.write(DD.make("test", 6, 5), p)
    return p


def test_depthdial_items_are_valid_cases(items):
    cs = load_cases(str(items))
    assert len(cs) == 6 * sum(len(v) for v in DD.CELLS.values())
    for c in cs:
        (q,) = c.questions
        assert q.mode == "choice" and len(q.options) == len(set(q.options))
        assert c.gold[q.key].count(1.0) == 1
    assert len({c.case_id for c in cs}) == len(cs)


@pytest.mark.parametrize("gen", sorted(DD.CELLS))
def test_depthdial_labels_are_correct_at_every_depth(items, gen):
    solver = {"lies": solve_lies, "swap": solve_swap, "order": solve_order}[gen]
    seen = set()
    for l in open(items):
        r = json.loads(l)
        m = r["meta"]
        if m["gen"] != gen:
            continue
        q = r["questions"][0]
        depth, ans = solver(r["state"], q["instructions"])
        assert depth == m["depth"] and len(q["options"]) == m["k"]
        assert q["options"].index(ans) == m["y"] == r["gold"]["answer"].index(1.0)
        seen.add((m["depth"], m["k"]))
    assert seen == set(DD.CELLS[gen])


def test_depthdial_is_deterministic():
    assert DD.make("measure", 3, 1) == DD.make("measure", 3, 1)
    assert DD.make("measure", 3, 1) != DD.make("measure", 3, 2)
    # n only cuts each cell's stream: a smaller fold is a per-cell prefix of a larger one
    assert DD.make("measure", 3, 1) == [it for it in DD.make("measure", 5, 1) if int(it["case_id"].split("_")[5]) < 3]


# ---------------------------------------------------------------- bands and soft targets
def test_bands():
    assert BD.band_of(.95, 2, 200) == "solve"
    assert BD.band_of(.56, 2, 200) == "beyond"          # .5 + 2 SE = .571
    assert BD.band_of(.60, 2, 200) == "near"
    assert BD.band_of(.26, 5, 200) == "mid"             # .2 + 2 SE = .257
    assert BD.band_of(.145, 5, 200) == "beyond"


def test_the_shipped_vis_ct_3_table_is_consistent():
    cs = BD.load_cells(ROOT / "data" / "depthdial_cells_vis-ct-3.json")
    assert set(cs) == {f"{g}|{d}|{k}" for g, v in DD.CELLS.items() for d, k in v}
    for c in cs.values():
        assert c["band"] == BD.band_of(c["acc"], c["k"], c["n"])


def test_soft_target_rules():
    assert BD.target("solve", .95, 3, 1) == [0.0, 1.0, 0.0]
    assert BD.target("beyond", .40, 4, 2) == [.25, .25, .25, .25]
    g = BD.target("near", .7, 3, 0)
    assert g == [.7, .15, .15]
    g = BD.target("mid", .1, 5, 4)                      # below chance: clamped to 1/k
    assert all(abs(x - .2) < 1e-6 for x in g)
    for band, acc, k, y in (("near", .61, 3, 2), ("mid", .37, 7, 5), ("beyond", .3, 5, 0)):
        assert abs(sum(BD.target(band, acc, k, y)) - 1) < 1e-6


def test_build_writes_targets_by_band_and_keeps_folds_disjoint(tmp_path):
    folds = tmp_path / "folds"
    for fold, n, seed in (("measure", 4, 1), ("dev", 3, 1), ("probe", 4, 1)):
        DD.write(DD.make(fold, n, seed), folds / f"gen_{fold}.jsonl")
    bands = {("lies", 1): ("solve", .95), ("lies", 2): ("near", .7), ("swap", 3): ("mid", .4), ("order", 5): ("mid", .1)}
    cs = {}
    for g, v in DD.CELLS.items():
        for d, k in v:
            band, acc = bands.get((g, d), ("beyond", 1 / k))
            cs[f"{g}|{d}|{k}"] = {"gen": g, "depth": d, "k": k, "n": 200, "acc": acc, "conf": .9,
                                  "chance": round(1 / k, 4), "auroc": .5, "band": band}
    man = BD.build(cs, "toy", folds, tmp_path / "dd", n_train=5)
    ev = {json.loads(l)["state"] + "||" + json.loads(l)["questions"][0]["instructions"]
          for f in folds.glob("*.jsonl") for l in open(f)}
    n = 0
    for f in (tmp_path / "dd").glob("dd_*.jsonl"):
        load_cases(str(f))                                            # valid Cases
        for l in open(f):
            r = json.loads(l); m = r["meta"]; g = r["gold"]["answer"]; k, y = m["k"], m["y"]
            n += 1
            assert r["state"] + "||" + r["questions"][0]["instructions"] not in ev
            assert r["source"] == f"dd_{m['gen']}_{m['band']}" == f.stem
            assert m["band"] == cs[f"{m['gen']}|{m['depth']}|{k}"]["band"]
            if m["band"] == "solve":
                assert g == [1.0 if j == y else 0.0 for j in range(k)] and m["target"] == "hard"
            elif m["band"] == "beyond":
                assert all(math.isclose(x, 1 / k, abs_tol=1e-5) for x in g)
            else:
                acc = max(m["parent_acc"], 1 / k)
                assert math.isclose(g[y], acc, abs_tol=1e-5)
                assert all(math.isclose(x, (1 - acc) / (k - 1), abs_tol=1e-5) for j, x in enumerate(g) if j != y)
    assert n == sum(man["sources"].values()) and n <= 5 * sum(len(v) for v in DD.CELLS.values())
    probes = {p.name: [json.loads(l) for l in open(p)] for p in (tmp_path / "dd" / "probes").glob("*.jsonl")}
    assert all(r["meta"]["band"] == "beyond" for r in probes["dd_probe_atchance.jsonl"])
    assert {r["meta"]["depth"] for r in probes["dd_probe_near.jsonl"]} == {2}
    assert all(r["gold"]["answer"].count(1.0) == 1 for L in probes.values() for r in L)   # true gold


# ---------------------------------------------------------------- hygiene
NEW = ["ct_build_corpus.py", "build_stage_corpora.py", "build_rl2_calpool.py", "depthdial.py",
       "build_depthdial.py", "depthdial_measure.py"]


@pytest.mark.parametrize("name", NEW)
def test_cli_starts(name):
    r = subprocess.run([sys.executable, str(SCRIPTS / name), "--help"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "usage:" in r.stdout, r.stderr[-800:]
