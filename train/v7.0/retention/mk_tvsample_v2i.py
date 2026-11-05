#!/usr/bin/env python3
"""v2i: copy of mk_tvsample.py + one change: rows carrying an image field are
skipped (trace_gen is text-only; an image question without its image is not a valid CoT item), counted as skipped_image_rows.
v2m: question sample for the retention trace pool on the arm's OWN TRAIN questions (2026-10-25: regenerate the
long-trace pool on the new corpus; K1 checks first). Same population rule as the screen_v1 pool (trace_gen.py on
sd2b_ctlc, 47,185 choice questions -> mk_long_pool.py 2400 longest boxed): choice questions with 2..8 options of TRAIN cases
(sha256(case_id) % 10 != 0) of the spec's sources, never a drop-list or K1 (case_id, key) pair. Seed-0 uniform sample of N questions.
Writes <out>/<source>.jsonl (each row keeps only its sampled questions; state/gold untouched) + <out>/receipt.json.
usage: mk_tvsample.py <spec.json> <corpus_dir> <drop.json> <k1.json> <N> <out_dir>"""
import hashlib, json, os, random, sys
spec, U, dropf, k1f, N, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
NOPT = 8   # trace_gen.py L = A..H
src = json.load(open(spec))["sources"].split(",")
drop = {tuple(p) for p in json.load(open(dropf))["drop"]}
k1 = {tuple(p) for p in json.load(open(k1f))["drop"]}; bad = drop | k1
holdout = lambda c: int(hashlib.sha256(c.encode()).hexdigest(), 16) % 10 == 0
pool = []   # (source, line_no, key)
n_ho = n_drop = n_k1 = n_img = 0
hasimg = lambda r: any(v for k, v in r.items() if "image" in k.lower() or k.lower() in ("img", "imgs", "video"))
for s in src:
    for i, l in enumerate(open(f"{U}/{s}.jsonl")):
        if not l.strip(): continue
        r = json.loads(l)
        if holdout(r["case_id"]): n_ho += 1; continue
        if hasimg(r): n_img += 1; continue
        for q in r["questions"]:
            p = (r["case_id"], q["key"])
            if p in drop: n_drop += 1; continue
            if p in k1: n_k1 += 1; continue
            if q.get("mode", "choice") == "choice" and 2 <= len(q["options"]) <= NOPT:
                pool.append((s, i, q["key"]))
random.Random(0).shuffle(pool)
sel = sorted(pool[:N])
want = {}
for s, i, k in sel: want.setdefault(s, {}).setdefault(i, set()).add(k)
os.makedirs(out, exist_ok=False)
nq = 0
for s, rows in want.items():
    with open(f"{out}/{s}.jsonl", "w") as f:
        for i, l in enumerate(open(f"{U}/{s}.jsonl")):
            if i in rows:
                r = json.loads(l); r["questions"] = [q for q in r["questions"] if q["key"] in rows[i]]
                assert not holdout(r["case_id"]) and not any((r["case_id"], q["key"]) in bad for q in r["questions"])
                nq += len(r["questions"]); f.write(json.dumps(r, ensure_ascii=False) + "\n")
rc = {"spec": spec, "spec_sha": hashlib.sha256(open(spec, "rb").read()).hexdigest()[:12], "corpus": U, "drop": dropf, "k1": k1f,
      "eligible_questions": len(pool), "sampled": nq, "sources": len(want), "seed": 0, "skipped_holdout_rows": n_ho,
      "skipped_drop_pairs": n_drop, "skipped_k1_pairs": n_k1, "skipped_image_rows": n_img, "max_options": NOPT}
json.dump(rc, open(f"{out}/receipt.json", "w"), indent=1)
print(json.dumps(rc))
