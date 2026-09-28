"""data-scale-2: the add-on files that turn a base corpus into the v3.0 SFT corpus.

The arm corpus = every *.jsonl of the base corpus (copied unchanged) + five add-on
files, written to <out>/data-scale-2/:

  jsg_emotion          all of jsg_emotion.jsonl
  jsg_sst5             all of jsg_sst5.jsonl
  jsg_anli             jsg_anli.jsonl capped at 20,000 cases, split evenly over the
                       ANLI rounds (case_id field 3), seeded shuffle per round
  ds2_tasksource_jev   30,000 cases of the mined tasksource pool (seeded shuffle)
  ds2_open_jev         10,000 cases of the mined open_jev pool (seeded shuffle)

No add-on includes a probe item: a case is dropped if its case_id is listed in
<probes>/probe_case_ids.txt or its normalised state matches the state of any
<probes>/fr1p_*.jsonl case.

The mined pool (--mine) is the scored candidate pool: pool/mine1_*.jsonl (cases),
scores/shard_*.jsonl (one row per scored question: case_id, source, correct, top,
gold_tie) and admit.json (the sources whose confident errors are admitted). A case
is eligible when it is scored, not a tie, and not a confident error (a wrong
question with top >= 0.9) from a source outside admit.json.

    python scripts/build_ds2_addons.py --base-corpus BASE --jsg JSG_DIR --mine MINE_DIR \\
        --probes PROBE_DIR --out STAGE [--arms dstar,ds2]

`dstar` stages the base corpus alone as <out>/dstar/. Writes
<out>/build_report.<arms>.json. CPU only, standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

report = {}


def norm(s):
    return hashlib.sha256(re.sub(r"\s+", " ", s).strip().lower().encode()).hexdigest()


def read(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def nq(cases):
    return sum(len(c["questions"]) for c in cases)


def write(p, cases, source):
    with open(p, "w") as fh:
        for c in cases:
            c = dict(c); c["source"] = source
            fh.write(json.dumps(c) + "\n")


def load_probes(probes: Path):
    probe_ids = {l.strip() for l in open(probes / "probe_case_ids.txt") if l.strip()}
    probe_sha = set()
    for f in probes.glob("fr1p_*.jsonl"):
        if f.name.endswith(".meta.jsonl"):
            continue
        for c in read(f):
            if c["state"].strip():
                probe_sha.add(norm(c["state"]))
    return probe_ids, probe_sha


def clean(cases, tag, probe_ids, probe_sha):
    keep = [c for c in cases if c["case_id"] not in probe_ids and norm(c["state"]) not in probe_sha]
    report.setdefault("probe_drops", {})[tag] = len(cases) - len(keep)
    return keep


def arm(exp, addons, base: Path, stage: Path):
    """addons: list of (file_stem, cases)."""
    d = stage / exp
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in sorted(base.glob("*.jsonl")):
        shutil.copy2(f, d / f.name)
    add = {}
    for stem, cases in addons:
        write(d / f"{stem}.jsonl", cases, stem)
        add[stem] = {"cases": len(cases), "questions": nq(cases)}
    report[exp] = {"added": add, "added_questions": sum(v["questions"] for v in add.values())}
    print(exp, report[exp], flush=True)


def mine_pool(mine: Path, probe_ids, probe_sha):
    """The mined pool and its eligible case ids (sorted), with each case's source."""
    pool = {}
    for f in sorted((mine / "pool").glob("mine1_*.jsonl")):
        for c in read(f):
            pool[c["case_id"]] = c
    by = defaultdict(list)
    for f in sorted((mine / "scores").glob("shard_*.jsonl")):
        if f.name.endswith(".unfit.json"):
            continue
        for r in read(f):
            by[r["case_id"]].append(r)
    status, src_of = {}, {}
    for cid, qs in by.items():
        src_of[cid] = qs[0]["source"]
        qs2 = [q for q in qs if not q["gold_tie"]]
        if not qs2:
            status[cid] = "tie"
        elif all(q["correct"] for q in qs2):
            status[cid] = "correct"
        elif any((not q["correct"]) and q["top"] >= 0.9 for q in qs2):
            status[cid] = "conf_wrong"
        else:
            status[cid] = "wrong"
    admit = set(json.loads((mine / "admit.json").read_text()))
    # eligible = scored, not a tie, not an un-admitted confident error, not a probe item
    eligible = [c for c in sorted(status) if status[c] != "tie"
                and not (status[c] == "conf_wrong" and src_of[c] not in admit)]
    eligible = [c for c in eligible if c not in probe_ids and norm(pool[c]["state"]) not in probe_sha]
    report["mine_pool_eligible"] = dict(Counter(src_of[c] for c in eligible))
    return pool, eligible, src_of


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-corpus", required=True, type=Path,
                    help="base corpus directory (*.jsonl), copied unchanged into each arm")
    ap.add_argument("--jsg", type=Path, help="directory with jsg_emotion/jsg_sst5/jsg_anli.jsonl (ds2)")
    ap.add_argument("--mine", type=Path, help="mined pool directory: pool/, scores/, admit.json (ds2)")
    ap.add_argument("--probes", required=True, type=Path,
                    help="probe directory: probe_case_ids.txt and fr1p_*.jsonl")
    ap.add_argument("--out", required=True, type=Path, help="stage directory; arms go to <out>/<arm>/")
    ap.add_argument("--arms", default="dstar,ds2", help="comma list of: dstar, ds2")
    a = ap.parse_args()
    which = [w for w in a.arms.split(",") if w]
    bad = set(which) - {"dstar", "ds2"}
    if bad:
        ap.error(f"unknown arms {sorted(bad)}")
    if "ds2" in which and (a.jsg is None or a.mine is None):
        ap.error("ds2 needs --jsg and --mine")
    base, stage = a.base_corpus, a.out

    probe_ids, probe_sha = load_probes(a.probes)
    if a.mine is not None:
        pool, eligible, src_of = mine_pool(a.mine, probe_ids, probe_sha)

    # the base corpus alone
    if "dstar" in which:
        arm("dstar", [], base, stage)

    # data-scale-2
    if "ds2" in which:
        J = a.jsg
        emo = clean(read(J / "jsg_emotion.jsonl"), "jsg_emotion", probe_ids, probe_sha)
        sst = clean(read(J / "jsg_sst5.jsonl"), "jsg_sst5", probe_ids, probe_sha)
        anli = clean(read(J / "jsg_anli.jsonl"), "jsg_anli", probe_ids, probe_sha)
        rng = random.Random("ds2:anli")
        rounds = defaultdict(list)
        for c in anli:
            rounds[c["case_id"].split(":")[2]].append(c)
        for v in rounds.values():
            rng.shuffle(v)
        per = 20000 // len(rounds)
        anli_cap = [c for r in sorted(rounds) for c in rounds[r][:per]]
        report["ds2_anli_rounds"] = {r: min(per, len(v)) for r, v in rounds.items()}
        rng = random.Random("ds2:pool")
        ts = [c for c in eligible if src_of[c] == "mine1_tasksource"]; rng.shuffle(ts)
        oj = [c for c in eligible if src_of[c] == "mine1_open_jev"]; rng.shuffle(oj)
        arm("data-scale-2", [("jsg_emotion", emo), ("jsg_sst5", sst), ("jsg_anli", anli_cap),
                             ("ds2_tasksource_jev", [pool[c] for c in ts[:30000]]),
                             ("ds2_open_jev", [pool[c] for c in oj[:10000]])], base, stage)

    stage.mkdir(parents=True, exist_ok=True)
    (stage / f"build_report.{'_'.join(which)}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
