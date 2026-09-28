"""Build the RL-stage source files for the rl2 arms.

    python scripts/build_rl2_sources.py --hrr fr3_rerank_mem_8k.jsonl \
        --hrr-meta fr3_rerank_mem_8k.meta.jsonl --out DIR
    python scripts/build_rl2_sources.py --cnc choice_noul_consistency.jsonl \
        --cnc-meta choice_noul_consistency.meta.jsonl --pkt page_boundary.jsonl --out DIR

v3.0 used only `--hrr` (the 8k hippo_rr corpus from scripts/build_hippo_rr.py and
its meta file after the `offsets` stage). Every input is optional; each one given
writes one file into --out, plus rl2_sources.json with row counts and sha256s.

rl_hrr.jsonl  hippo_rr (MS MARCO / TopiOCQA / FiQA TRAIN) split into ONE state per
              candidate: "Query: q / Candidate passage: c", one noul. Group = the
              query (case_id "<group>#<j>..."), exactly one relevant row per group
              (the corpus's gold slot); every other candidate is a BM25 hard negative.
              The wording differs on purpose from hippo's per-candidate request
              (instruction, header, criteria), so a per-candidate hippo read stays
              format-neutral.
rl_cnc.jsonl  choice/noul consistency cases (TRAIN sources), one case per decision
              with its three forms. Each key becomes "<key>@@<flag option>": the
              option that means "flagged" in that form (choice: meta flag_option;
              noul: "true" for a positive claim, "false" for a negative one). The
              gold is kept only for a logged diagnostic; the consistency and RLCD
              losses never read it.
rl_pkt.jsonl  page_boundary TRAIN packets, source renamed (see build_pkt).

All files salt case ids so that no case falls in run_arm_lib's 1-in-10 hash
holdout: a group must not be split between training and the in-distribution
target, and the matched controls see exactly the same rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

INSTR = "This candidate passage contains information that helps answer the query."
CRIT = {"false": "The passage does not help answer the query.",
        "true": "The passage helps answer the query."}


def unheld(base: str) -> str:
    """base + the first salt whose sha256 % 10 != 0 (never in the in-distribution holdout)."""
    for k in range(100):
        cid = f"{base}~{k}"
        if int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 10 != 0:
            return cid
    raise RuntimeError(base)


def split_candidates(state: str):
    head, _, rest = state.partition("\n\n[1] ")
    parts = re.split(r"\n\n\[\d+\] ", "[1] " + rest)
    parts[0] = parts[0][4:]
    query = head.split("\n\n")[0]
    assert query.startswith("Query: "), query[:80]
    return query[len("Query: "):], parts


def build_hrr(path, meta_path, out):
    meta = {m["case_id"]: m for m in map(json.loads, open(meta_path))}
    n_g = n_r = 0
    with open(out, "w") as fh:
        for line in open(path):
            c = json.loads(line)
            m = meta[c["case_id"]]
            query, cands = split_candidates(c["state"])
            if len(cands) != m["n_candidates"]:
                raise ValueError(f"{c['case_id']}: {len(cands)} parsed vs {m['n_candidates']}")
            gslot = m["gold_slot"]                      # 1-based
            g = c["case_id"].replace("#", "_")
            for j, cand in enumerate(cands, 1):
                rel = j == gslot
                fh.write(json.dumps({
                    "case_id": unheld(f"rl_hrr:{g}#{j}"), "source": "rl_hrr",
                    "state": f"Query: {query}\n\nCandidate passage:\n\n{cand.strip()}",
                    "questions": [{"key": "rel", "mode": "noul", "instructions": INSTR,
                                   "options": ["false", "true"], "criteria": CRIT}],
                    "gold": {"rel": [0.0, 1.0] if rel else [1.0, 0.0]}}) + "\n")
                n_r += 1
            n_g += 1
    return {"groups": n_g, "rows": n_r}


def flag_of(q, form):
    f = form["form"]
    if q["mode"] == "choice":
        return form["flag_option"]
    pol = form.get("polarity")
    if pol not in ("pos", "neg"):
        raise ValueError(f"noul form without polarity: {form}")
    return "true" if pol == "pos" else "false"


def build_cnc(path, meta_path, out):
    meta = {m["case_id"]: m for m in map(json.loads, open(meta_path))}
    n = 0
    with open(out, "w") as fh:
        for line in open(path):
            c = json.loads(line)
            forms = meta[c["case_id"]]["forms"]
            qs, gold = [], {}
            for q in c["questions"]:
                fl = flag_of(q, forms[q["key"]])
                if fl not in q["options"]:
                    raise ValueError(f"{c['case_id']}/{q['key']}: flag {fl} not in {q['options']}")
                k = f"{q['key']}@@{fl}"
                qs.append({**q, "key": k})
                gold[k] = c["gold"][q["key"]]
            fh.write(json.dumps({"case_id": unheld(f"rl_cnc:{c['case_id'].replace('#', '_')}"),
                                 "source": "rl_cnc", "state": c["state"],
                                 "questions": qs, "gold": gold}) + "\n")
            n += 1
    return {"cases": n}


def build_pkt(path, out):
    """rl_pkt: page_boundary TRAIN packets, source renamed, ids salted out of the
    hash holdout. A packet = one case; its boundary_<n> nouls are
    the segmentation decisions, category_<n> are kept (trained by CE in both arms)."""
    n = 0
    with open(out, "w") as fh:
        for line in open(path):
            c = json.loads(line)
            if not any(k.startswith("boundary_") for k in c["gold"]):
                continue
            fh.write(json.dumps({**{k: c[k] for k in ("state", "questions", "gold")},
                                 "case_id": unheld(f"rl_pkt|{c['case_id'].replace('#', '_')}"),
                                 "source": "rl_pkt"}) + "\n")
            n += 1
    return {"cases": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hrr"); ap.add_argument("--hrr-meta")
    ap.add_argument("--cnc"); ap.add_argument("--cnc-meta")
    ap.add_argument("--pkt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = {}
    if a.hrr:
        rep["rl_hrr"] = build_hrr(a.hrr, a.hrr_meta, out / "rl_hrr.jsonl")
    if a.cnc:
        rep["rl_cnc"] = build_cnc(a.cnc, a.cnc_meta, out / "rl_cnc.jsonl")
    if a.pkt:
        rep["rl_pkt"] = build_pkt(a.pkt, out / "rl_pkt.jsonl")
    for f in sorted(out.glob("rl_*.jsonl")):
        rep.setdefault(f.stem, {})["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    (out / "rl2_sources.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
