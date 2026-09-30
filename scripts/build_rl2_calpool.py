"""RL-stage pools cut from a parent's own in-distribution holdout.

    python scripts/build_rl2_calpool.py --corpus-dir PARENT_CORPUS --sources a,b,... \\
        --max-questions 12000 --out DIR                       # rl_cal.jsonl
    python scripts/build_rl2_calpool.py --corpus-dir PARENT_CORPUS --sources a,b,... \\
        --splits rl_slice:0.5,rl_calp:0.3,rl_dev:0.2 --out DIR  # rl2-b2 (asym-calA's RL rows)

The pool is the PARENT's own 1-in-10 in-distribution holdout (the trainer's rule:
sha256(case_id) % 10 == 0), which the parent never trained on and no arm ever trains
on through the replay stream.

rl_cal.jsonl  a fixed hash subsample (sha256("rl2cal:" + id)) capped at --max-questions.
              Case ids become "rl_cal|<file stem>|<orig id>~<salt>".
--splits      the WHOLE holdout split by u = sha256("rl2split:" + id) % 10000 / 10000
              into the named sources (cumulative fractions, in the order given); case ids
              become "<name>|<file stem>|<orig id>~<salt>". Writes <name>.jsonl per split
              and rl2_splits.report.json.

The salt is chosen so the copy is NOT in the arm's own hash holdout; the source stem is
kept in the id for cal-4b groups and strata. Only state, questions and gold are kept.

rl2-b2 (the rl_slice / rl_calp / rl_dev rows of the calasym-stage corpus) is the --splits
command above on the v3.0 SFT corpus (data/v3.0_corpus_manifest.json, `v3-stack-ds2-xmlp`)
with --sources in its training order (scripts/build_stage_corpora.py V3_SOURCES):
13,834 / 7,789 / 5,259 questions.

Caveat, report-only: in arms that train on these rows, the in_distribution target (the
same states under their original ids) is no longer out-of-sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def held(cid: str) -> bool:
    return int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 10 == 0


def unheld(base: str) -> str:
    for k in range(100):
        cid = f"{base}~{k}"
        if not held(cid):
            return cid
    raise RuntimeError(base)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--max-questions", type=int, default=12000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="",
                    help="e.g. rl_slice:0.5,rl_calp:0.3,rl_dev:0.2 -- split the WHOLE holdout "
                         "by sha256('rl2split:'+id) into these sources (ignores --max-questions)")
    a = ap.parse_args()
    if a.splits:
        return split_main(a)
    rows = []
    for stem in a.sources.split(","):
        for line in open(Path(a.corpus_dir) / f"{stem}.jsonl"):
            if not line.strip():
                continue
            c = json.loads(line)
            if held(c["case_id"]):
                rows.append((stem, c))
    total_q = sum(len(c["questions"]) for _, c in rows)
    rows.sort(key=lambda sc: hashlib.sha256(("rl2cal:" + sc[1]["case_id"]).encode()).hexdigest())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_q, by = 0, {}
    with open(out / "rl_cal.jsonl", "w") as fh:
        for stem, c in rows:
            if n_q + len(c["questions"]) > a.max_questions:
                continue
            n_q += len(c["questions"])
            by[stem] = by.get(stem, 0) + len(c["questions"])
            fh.write(json.dumps({**{k: c[k] for k in ("state", "questions", "gold")},
                                 "case_id": unheld(f"rl_cal|{stem}|{c['case_id']}"),
                                 "source": "rl_cal"}) + "\n")
    rep = {"holdout_cases": len(rows), "holdout_questions": total_q, "kept_questions": n_q,
           "by_source": dict(sorted(by.items())),
           "sha256": hashlib.sha256((out / "rl_cal.jsonl").read_bytes()).hexdigest()}
    (out / "rl_cal.report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


def split_main(a):
    parts = [(n, float(f)) for n, f in (x.split(":") for x in a.splits.split(","))]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fhs = {n: open(out / f"{n}.jsonl", "w") for n, _ in parts}
    rep = {n: {"cases": 0, "questions": 0, "modes": {}} for n, _ in parts}
    for stem in a.sources.split(","):
        for line in open(Path(a.corpus_dir) / f"{stem}.jsonl"):
            if not line.strip():
                continue
            c = json.loads(line)
            if not held(c["case_id"]):
                continue
            u = int(hashlib.sha256(("rl2split:" + c["case_id"]).encode()).hexdigest(), 16) % 10000 / 10000
            acc = 0.0
            for n, f in parts:
                acc += f
                if u < acc:
                    break
            fhs[n].write(json.dumps({**{k: c[k] for k in ("state", "questions", "gold")},
                                     "case_id": unheld(f"{n}|{stem}|{c['case_id']}"),
                                     "source": n}) + "\n")
            rep[n]["cases"] += 1
            rep[n]["questions"] += len(c["questions"])
            for q in c["questions"]:
                rep[n]["modes"][q["mode"]] = rep[n]["modes"].get(q["mode"], 0) + 1
    for n, fh in fhs.items():
        fh.close()
        rep[n]["sha256"] = hashlib.sha256((out / f"{n}.jsonl").read_bytes()).hexdigest()
    (out / "rl2_splits.report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
