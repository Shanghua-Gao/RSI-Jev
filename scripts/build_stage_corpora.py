"""The corpus of every training stage of v4.0-VL, one subcommand per stage.

v4.0-VL (checkpoint asym-calA) is a chain of stages, each continuing from the previous
checkpoint. Each subcommand below writes that stage's corpus directory as
<root>/<name>/, the directory the trainer read (corpus_dir, named as in the published
stage specs); a stage's parent corpus is found under the same --root by that name. External inputs are arguments. CPU only, standard library
only, nothing is downloaded.

  stage             subcommand     writes <root>/...        parent / inputs
  v4-rep-b-s17ck    rep-b          v4-rep-b-s17ck           the v3.0 SFT corpus (--ds2)
  vis-ct-1          vis-ct-1       vis-ct-1                 rep-b corpus + vision_v1 (--vision-v1)
  rl5-listwise-vis  rl5-vis        rl5-vis                  vis-ct-1 corpus + v3.0 rl_hrr.jsonl (--rl-hrr)
  vis-ct-3          vis-ct-3       vis-ct-3                 rl5-vis + vision_v2 / vision_v3 (--vision-v2/-v3)
  calA-ce           calA-ce        calA-ce                  vis-ct-3 + depth-dial files (--dd)
  (rl2-b2)          rl2-b2         rl2-b2-src               the v3.0 SFT corpus (--ds2)
  asym-calA         calasym-stage  calasym-stage            rl2-b2-src + calA-ce

    python scripts/build_stage_corpora.py rep-b    --root R --ds2 V3_SFT_CORPUS
    python scripts/build_stage_corpora.py vis-ct-1 --root R --vision-v1 VISION_V1 [--control]
    python scripts/build_stage_corpora.py rl5-vis  --root R --rl-hrr RL2_SOURCES/rl_hrr.jsonl
    python scripts/build_stage_corpora.py vis-ct-3 --root R --vision-v2 VISION_V2 --vision-v3 VISION_V3 [--control]
    python scripts/build_depthdial.py folds --folds R/depthdial/folds
    python scripts/build_depthdial.py build --folds R/depthdial/folds \\
        --cells data/depthdial_cells_vis-ct-3.json --parent vis-ct-3 --out R/depthdial/vis-ct-3 --n-train 300
    python scripts/build_stage_corpora.py calA-ce  --root R --dd R/depthdial/vis-ct-3 [--controls]
    python scripts/build_stage_corpora.py rl2-b2   --root R --ds2 V3_SFT_CORPUS
    python scripts/build_stage_corpora.py calasym-stage --root R

V3_SFT_CORPUS is the v3.0 SFT corpus (data/v3.0_corpus_manifest.json, `v3-stack-ds2-xmlp`:
scripts/build_ds2_addons.py). RL2_SOURCES/rl_hrr.jsonl is v3.0's RL file (same manifest,
`rl_files`; scripts/build_rl2_sources.py --hrr). VISION_V1/V2/V3 are the image corpora
(scripts/build_vision_*.py); images are looked up by case_id, which no stage here changes.

What each stage holds (seed 17 everywhere unless stated; batch 16):

  rep-b          (built as "v3-ds2-rep-b-jsg3k") the v3.0 SFT files, each jsg_* source capped at 3,000 cases, stratified by
                 gold label (and by ANLI round, 1,000 per round), random.Random(4).
  vis-ct-1       ct_build_corpus: 4,000 steps = 64,000 q = all vision_v1 vis_*.jsonl
                 (33,906 q) + 30,094 q replay of the rep-b corpus (V3_SOURCES order).
                 --control also writes vis-ct-1-ctl (replay only).
  rl5-vis        vis-ct-1's rp_* files (symlinked) + a hash subsample of its vis_* rows,
                 sha256("rl5vis:" + case_id) % 10000 < frac * 10000, frac = min(1,
                 text_q / 4 / vis_q) (vision = 20% of replay; 7,410 q = 19.8%), + rl_hrr.jsonl
                 (symlinked). The RL stage trains CE on the replay rows, the listwise
                 objective on rl_hrr.
  vis-ct-3       ct_build_corpus: 3,000 steps = 48,000 q = all vision_v3 vis3_*.jsonl +
                 four vision_v2 sources (iconqa, visual7w, textvqa, ocrvqa), each a
                 random.Random("r3:<source>") shuffle cut at the first case reaching
                 --kept-q questions (1,750) = 17,352 new q, + 30,648 q replay of rl5-vis's
                 rp_* and vis_* files (sorted stems; rl_hrr is not replayed).
  calA-ce        ct_build_corpus: 3,000 steps = 48,000 q = the 8 dd_*.jsonl depth-dial
                 files (8,492 q, scripts/build_depthdial.py) + replay of the vis-ct-3 corpus
                 (its manifest's `sources`). --controls also writes calC-ctl (replay only)
                 and calB-rl (calC-ctl + rl_dd_* = the same depth-dial items with one-hot
                 TRUE gold, source prefixed rl_), the two arms calA-ce was compared with.
  rl2-b2         build_rl2_calpool.py --splits rl_slice:0.5,rl_calp:0.3,rl_dev:0.2 on the
                 v3.0 SFT corpus: 13,834 / 7,789 / 5,259 q.
  calasym-stage  rl2-b2's rl_slice / rl_calp / rl_dev (bytes copied) + rl_slice_vis.jsonl:
                 calA-ce's vision rows (every file whose name contains "vis"),
                 random.Random(17) shuffle per file in sorted order, each file cut once it
                 reaches its question share of target = round(.30 / .70 * slice_q); the rows
                 keep their case_id (images are found by it), get source "rl_slice_vis" and
                 a "vis_origin" field. 5,951 q = 30.1%. The RL stage runs with replay off
                 and trains only on sources starting with rl_slice, so this is how image
                 rows enter it. asym-calA and its matched CE arm ce-calA read this corpus.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The v3.0 SFT corpus's sources in its training spec's order (and v4-rep-b-s17ck's):
# ct_build_corpus's interleave breaks ties by this order.
V3_SOURCES = ("cov_ambigqa,cov_condaqa,cov_cuad,cov_finqa,cov_gsm8k,cov_hotpotqa,cov_massive_multi,cov_musique,"
              "cov_open_jev,cov_privacyqa,cov_tabfact,cov_tasksource_jev,mc_replay_extra,mc_replay,st_aegis2,st_boolq,"
              "st_civil_comments,st_helpsteer2,st_kev_devtools_v1,st_kev_documents_v1,st_kev_hard_v1,st_massive-de-DE,"
              "st_massive-en-US,st_multinli,st_nimble_train,st_paws,st_procedural_train,st_squad2,st_vitaminc,synth,"
              "td_train,jsg_emotion,jsg_sst5,jsg_anli,ds2_tasksource_jev,ds2_open_jev")
REP_B = "v4-rep-b-s17ck"            # the corpus was called v3-ds2-rep-b-jsg3k when it was built
R3_KEPT = ["iconqa", "visual7w", "textvqa", "ocrvqa"]


def nq_file(f) -> int:
    return sum(len(json.loads(l)["questions"]) for l in open(f) if l.strip())


def ct_build(parent: Path, sources: str, steps: int, out: Path, new: list[str]) -> dict:
    """scripts/ct_build_corpus.py, skipped when the corpus is already there."""
    if not (out / "manifest.json").exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(HERE / "ct_build_corpus.py"), "--parent-corpus", str(parent),
                        "--parent-sources", sources, "--steps", str(steps), "--out", str(out),
                        *(["--new", *new] if new else [])], check=True)
    return json.load(open(out / "manifest.json"))


# ---------------------------------------------------------------- v4-rep-b-s17ck
def rep_b(a) -> None:
    """ds2 with each jsg source capped at 3k, stratified by gold label (and ANLI round, 1k per round)."""
    ds2, d = Path(a.ds2), Path(a.root) / REP_B
    d.mkdir(parents=True, exist_ok=True)
    skip = {"jsg_emotion", "jsg_sst5", "jsg_anli"}
    for f in sorted(ds2.glob("*.jsonl")):
        if f.stem not in skip and not (d / f.name).exists():
            shutil.copyfile(f, d / f.name)

    def rows(p):
        return [json.loads(l) for l in open(p) if l.strip()]

    def argmax(g):
        return max(range(len(g)), key=g.__getitem__)
    rep = {}
    for src in ("jsg_emotion", "jsg_sst5", "jsg_anli"):
        rs = rows(ds2 / f"{src}.jsonl"); rng = random.Random(4)
        strata = collections.defaultdict(list)
        for r in rs:
            rnd = r["case_id"].split(":")[2] if src == "jsg_anli" else "all"
            strata[(rnd, argmax(r["gold"][r["questions"][0]["key"]]))].append(r)
        byr = collections.Counter(k[0] for k in strata for _ in strata[k])
        per_r = {k: 3000 // len(byr) for k in byr}
        keep = []
        for (rnd, lab), lst in sorted(strata.items()):
            n = round(per_r[rnd] * len(lst) / byr[rnd]); rng.shuffle(lst); keep += lst[:n]
        rng.shuffle(keep); keep = keep[:3000]
        with open(d / f"{src}.jsonl", "w") as fh:
            for r in keep:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        rep[src] = {"from": len(rs), "kept": len(keep),
                    "rounds": dict(collections.Counter(r["case_id"].split(":")[2] for r in keep)) if src == "jsg_anli" else None}
    print(json.dumps(rep, indent=1))


# ---------------------------------------------------------------- vis-ct-1
def vis_ct_1(a) -> None:
    root = Path(a.root)
    parent = Path(a.parent) if a.parent else root / REP_B
    newf = sorted(str(p) for p in Path(a.vision_v1).glob("vis_*.jsonl"))
    assert newf, f"no vis_*.jsonl in {a.vision_v1}"
    todo = [("vis-ct-1", newf)] + ([("vis-ct-1-ctl", [])] if a.control else [])
    for name, new in todo:
        m = ct_build(parent, a.parent_sources, 4000, root / name, new)
        print(name, 4000, "q:", m["new_questions"], "new +", m["replay_questions"], "replay")


# ---------------------------------------------------------------- rl5-listwise-vis
def rl5_vis(a) -> None:
    """vis-ct-1's own text replay (rp_*, symlinked) + a hash subsample of its vision rows (vis_*,
    same case ids) sized to ~20% of all replay QUESTIONS, + rl_hrr (the RL-side rows)."""
    root = Path(a.root)
    src = (Path(a.src) if a.src else root / "vis-ct-1").resolve()
    out = root / "rl5-vis"; hrr = Path(a.rl_hrr).resolve()
    out.mkdir(parents=True, exist_ok=True)
    nq = nq_file
    nt = 0
    for f in sorted(src.glob("rp_*.jsonl")):
        (out / f.name).unlink(missing_ok=True); os.symlink(f, out / f.name); nt += nq(f)
    vis = sorted(src.glob("vis_*.jsonl")); nv_all = sum(nq(f) for f in vis)
    target = nt / 4                                   # vision = 20% of (text + vision)
    frac = min(1.0, target / nv_all)
    nv = 0
    for f in vis:
        rows = [l for l in open(f) if l.strip()]
        keep = [l for l in rows if int(hashlib.sha256(("rl5vis:" + json.loads(l)["case_id"]).encode()).hexdigest(), 16) % 10000 < frac * 10000]
        (out / f.name).write_text("".join(keep)); nv += sum(len(json.loads(l)["questions"]) for l in keep)
    (out / "rl_hrr.jsonl").unlink(missing_ok=True); os.symlink(hrr, out / "rl_hrr.jsonl")
    rep = {"text_replay_q": nt, "vision_replay_q": nv, "vision_share": round(nv / (nt + nv), 4), "vision_frac_kept": round(frac, 4)}
    (out / "rl5_vis.report.json").write_text(json.dumps(rep, indent=1)); print(rep)
    print("sources:", ",".join(sorted(p.stem for p in out.glob("*.jsonl"))))


# ---------------------------------------------------------------- vis-ct-3
def vis_ct_3(a) -> None:
    root = Path(a.root)
    par = Path(a.parent) if a.parent else root / "rl5-vis"
    psrc = ",".join(sorted(f.stem for f in par.glob("*.jsonl") if f.stem.startswith(("rp_", "vis_"))))
    new = root / "vis-ct-3.new"; new.mkdir(parents=True, exist_ok=True)
    for f in Path(a.vision_v3).glob("vis3_*.jsonl"):
        shutil.copy(f, new / f.name)
    for s in R3_KEPT:
        rows = [l for l in open(Path(a.vision_v2) / f"vis2_{s}.jsonl") if l.strip()]
        random.Random(f"r3:{s}").shuffle(rows)
        keep, q = [], 0
        for l in rows:
            if q >= a.kept_q:
                break
            keep.append(l); q += len(json.loads(l)["questions"])
        (new / f"vis2_{s}.jsonl").write_text("".join(keep))
    nf = sorted(str(p) for p in new.glob("*.jsonl"))
    for name, extra in (("vis-ct-3", nf),) + ((("vis-ct-3-ctl", []),) if a.control else ()):
        m = ct_build(par, psrc, 3000, root / name, extra)
        print(name, 3000, "q:", m["new_questions"], "new +", m["replay_questions"], "replay")


# ---------------------------------------------------------------- calA-ce (+ calB-rl / calC-ctl)
def cal_a_ce(a) -> None:
    root = Path(a.root)
    par = Path(a.parent) if a.parent else root / "vis-ct-3"
    ps = json.load(open(par / "manifest.json"))["sources"]
    new = sorted(str(p) for p in Path(a.dd).glob("dd_*.jsonl"))
    assert new, f"no dd_*.jsonl in {a.dd}"
    ct_build(par, ps, 3000, root / "calA-ce", new)
    names = ["calA-ce"]
    if a.controls:
        ct_build(par, ps, 3000, root / "calC-ctl", [])
        b = root / "calB-rl"
        shutil.rmtree(b, ignore_errors=True); shutil.copytree(root / "calC-ctl", b)
        for f in new:
            with open(b / f"rl_{Path(f).name}", "w") as fh:
                for l in open(f):
                    r = json.loads(l); y = r["meta"]["y"]; k = len(r["questions"][0]["options"])
                    r["source"] = "rl_" + r["source"]; r["gold"] = {"answer": [1.0 if j == y else 0.0 for j in range(k)]}
                    fh.write(json.dumps(r) + "\n")
        names += ["calB-rl", "calC-ctl"]
    for d in names:
        n = {f.stem: nq_file(f) for f in (root / d).glob("*.jsonl")}
        print(d, "total", sum(n.values()), "dd", {k: v for k, v in n.items() if "dd_" in k})


# ---------------------------------------------------------------- rl2-b2
def rl2_b2(a) -> None:
    subprocess.run([sys.executable, str(HERE / "build_rl2_calpool.py"), "--corpus-dir", a.ds2,
                    "--sources", a.sources, "--out", str(Path(a.root) / "rl2-b2-src"),
                    "--splits", "rl_slice:0.5,rl_calp:0.3,rl_dev:0.2"], check=True)


# ---------------------------------------------------------------- asym-calA
def calasym_stage(a) -> None:
    """rl2-b2 rl_slice/rl_calp/rl_dev (bytes copied) + ~30% vision rows from calA-ce's own corpus."""
    root = Path(a.root)
    rl = Path(a.rl2_b2) if a.rl2_b2 else root / "rl2-b2-src"
    cal = Path(a.cal_a_ce) if a.cal_a_ce else root / "calA-ce"
    out = root / "calasym-stage"
    os.makedirs(out, exist_ok=False)
    nq = lambda r: len(r["questions"])   # noqa: E731
    for s in ("rl_slice", "rl_calp", "rl_dev"):
        shutil.copyfile(rl / f"{s}.jsonl", out / f"{s}.jsonl")
    slice_q = sum(nq(json.loads(l)) for l in open(rl / "rl_slice.jsonl") if l.strip())
    target = round(a.frac / (1 - a.frac) * slice_q)
    vis_files = sorted(f for f in glob.glob(str(cal / "*.jsonl")) if "vis" in os.path.basename(f))
    by = defaultdict(list)
    for f in vis_files:
        for l in open(f):
            if l.strip():
                r = json.loads(l); by[os.path.basename(f)[:-6]].append(r)
    tot_q = sum(nq(r) for v in by.values() for r in v)
    rng = random.Random(a.seed)
    rows_out, per = [], {}
    for k in sorted(by):                               # proportional to each source's question share
        rows = by[k][:]; rng.shuffle(rows)
        want = target * sum(map(nq, rows)) / tot_q; got = 0
        for r in rows:
            if got >= want:
                break
            rows_out.append({**r, "source": "rl_slice_vis", "vis_origin": r["source"]}); got += nq(r)
        per[k] = got
    with open(out / "rl_slice_vis.jsonl", "w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")
    vq = sum(per.values())
    man = dict(slice_q=slice_q, vis_q=vq, vis_cases=len(rows_out), vis_frac=round(vq / (vq + slice_q), 4), seed=a.seed,
               per_source_q=per, vis_from="calA-ce", rl_from="rl2-b2",
               note="vision rows are calA-ce's own trained vision replay (no test/probe items); source renamed so rl2 replay=false trains on them")
    json.dump(man, open(out / "manifest.json", "w"), indent=1)
    print(json.dumps(man))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def sp(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--root", required=True, help="stage corpora root: writes <root>/<stage>/")
        p.set_defaults(fn=fn)
        return p
    p = sp("rep-b", rep_b, "v4-rep-b-s17ck: v3.0 SFT corpus with jsg_* capped at 3k")
    p.add_argument("--ds2", required=True, help="the v3.0 SFT corpus dir")
    p = sp("vis-ct-1", vis_ct_1, "images round 1")
    p.add_argument("--vision-v1", required=True, help="dir of vision_v1 vis_*.jsonl")
    p.add_argument("--parent", default="", help=f"default <root>/{REP_B}")
    p.add_argument("--parent-sources", default=V3_SOURCES)
    p.add_argument("--control", action="store_true", help="also write vis-ct-1-ctl")
    p = sp("rl5-vis", rl5_vis, "rl5-listwise-vis")
    p.add_argument("--rl-hrr", required=True, help="v3.0's rl_hrr.jsonl")
    p.add_argument("--src", default="", help="default <root>/vis-ct-1")
    p = sp("vis-ct-3", vis_ct_3, "images round 3")
    p.add_argument("--vision-v2", required=True)
    p.add_argument("--vision-v3", required=True)
    p.add_argument("--kept-q", type=int, default=1750)
    p.add_argument("--parent", default="", help="default <root>/rl5-vis")
    p.add_argument("--control", action="store_true", help="also write vis-ct-3-ctl")
    p = sp("calA-ce", cal_a_ce, "depth-dial calibration round")
    p.add_argument("--dd", required=True, help="dir of dd_*.jsonl (build_depthdial.py build --out)")
    p.add_argument("--parent", default="", help="default <root>/vis-ct-3")
    p.add_argument("--controls", action="store_true", help="also write calC-ctl and calB-rl")
    p = sp("rl2-b2", rl2_b2, "rl_slice / rl_calp / rl_dev from the v3.0 SFT corpus's holdout")
    p.add_argument("--ds2", required=True, help="the v3.0 SFT corpus dir")
    p.add_argument("--sources", default=V3_SOURCES)
    p = sp("calasym-stage", calasym_stage, "asym-calA's RL stage corpus")
    p.add_argument("--rl2-b2", default="", help="default <root>/rl2-b2-src")
    p.add_argument("--cal-a-ce", default="", help="default <root>/calA-ce")
    p.add_argument("--frac", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=17)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
