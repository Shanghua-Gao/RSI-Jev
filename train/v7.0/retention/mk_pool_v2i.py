#!/usr/bin/env python3
"""mk_pool_v2i = mk_pool_v2m.py (md5 b63d7e16) with the Instruct rules added:
traces with <think>/</think> or a chat special token (tvrep.py SPECIAL regex) are DROPPED and counted (not stripped), and the receipt gets a
report-only repetition stat (share of selected traces whose most common non-empty line occurs >= 5 times). Everything else unchanged.
Original doc:
v2m retention pool = the two screen_v1 steps on this corpus, unchanged in rule:
 1) build_rtpool_x.py: base4b greedy CoT traces (trace_gen.py on v2m/tvsample200k) -> {"case_id","key","src","nobox","ok",
    "text": COT prompt + trace}; drop holdout (%10) and truncated traces
 2) mk_long_pool.py: the N traces with the longest answer part, boxed only, sorted (-trace chars, case_id, key)
N = file rows; fit keeps retention_pool of them (screen_v1: file 2400 -> fit 2048; here the same ratio, 2026-10-25 scaled pool).
Extra checks (K1 first): no holdout row, no drop-list / K1 (case_id, key) pair, every case_id a TRAIN case of the arm's corpus sources.
usage: mk_pool_v2m.py <tvsample dir> <traces.jsonl> <spec.json> <k1.json> <N> <out_all.jsonl> <out_long.jsonl>"""
import collections, hashlib, json, os, re, sys
SPECIAL = re.compile(r"<think>|</think>|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|[a-z_]+\|>")
sys.path.insert(0, os.environ.get("NV7COT_DIR", os.path.dirname(os.path.abspath(__file__))))
from nv7_cot import COT, L  # noqa: E402
TR, TRC, SPEC, K1, N, OUT, LONG = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6], sys.argv[7]
ho = lambda c: int(hashlib.sha256(c.encode()).hexdigest(), 16) % 10 == 0
# 1) = build_rtpool_x.py (paths only)
want, n = {}, {"traces": 0, "holdout": 0, "trunc": 0, "special": 0, "kept": 0, "nobox_kept": 0}
for ln in open(TRC):
    t = json.loads(ln); n["traces"] += 1
    if t["holdout"]: n["holdout"] += 1; continue
    if t["trunc"]: n["trunc"] += 1; continue
    if SPECIAL.search(t["text"]): n["special"] += 1; continue
    want[(t["case_id"], t["key"])] = t
with open(OUT, "w") as fo:
    for fn in sorted(os.listdir(TR)):
        if not fn.endswith(".jsonl"): continue
        for ln in open(os.path.join(TR, fn)):
            r = json.loads(ln)
            for q in r["questions"]:
                t = want.pop((r["case_id"], q["key"]), None)
                if t is None: continue
                opts = [q["criteria"][o] for o in q["options"]]
                body = r["state"].strip() + "\n" + q["instructions"].strip() + "\n" + "\n".join(f"{L[i]}. {x}" for i, x in enumerate(opts))
                fo.write(json.dumps({"case_id": r["case_id"], "key": q["key"], "src": t["src"], "nobox": t["nobox"], "ok": t["ok"],
                                     "text": COT.format(q=body) + t["text"]}, ensure_ascii=False) + "\n")
                n["kept"] += 1; n["nobox_kept"] += t["nobox"]
n["unmatched"] = len(want)
n["sha"] = hashlib.sha256(open(OUT, "rb").read()).hexdigest()[:12]
json.dump(n, open(OUT + ".receipt.json", "w")); print("all", json.dumps(n))
# 2) = mk_long_pool.py (N from argv)
P = [json.loads(l) for l in open(OUT)]
P = [x for x in P if not x["nobox"] and x.get("text")]
P.sort(key=lambda x: (-len(x["text"].split("Answer:", 1)[-1]), x["case_id"], x["key"]))
sel = P[:N]
assert len(sel) == N, f"only {len(sel)} boxed traces < N {N}"
# extra checks
spec = json.load(open(SPEC)); U = spec["corpus_dir"]; drop = {tuple(p) for p in json.load(open(spec["fit_extra"]["train_drop_file"]))["drop"]}
k1 = {tuple(p) for p in json.load(open(K1))["drop"]}
train = set()
for s in spec["sources"].split(","):
    for l in open(f"{U}/{s}.jsonl"):
        if l.strip():
            c = json.loads(l)["case_id"]
            if not ho(c): train.add(c)
assert not any(ho(x["case_id"]) for x in sel), "holdout row"
assert not any((x["case_id"], x["key"]) in drop for x in sel), "drop-list pair"
assert not any((x["case_id"], x["key"]) in k1 for x in sel), "K1 pair"
assert all(x["case_id"] in train for x in sel), "case_id not a TRAIN case of the arm's corpus"
with open(LONG, "w") as f:
    for x in sel: f.write(json.dumps(x, ensure_ascii=False) + "\n")
tr = [len(x["text"].split("Answer:", 1)[-1]) for x in sel]
rf = spec["fit_extra"]["retention_pool"]
rc = {"rows": len(sel), "sha": hashlib.sha256(open(LONG, "rb").read()).hexdigest()[:12], "from": OUT, "from_sha": n["sha"],
      "boxed_candidates": len(P), "trace_chars": [min(tr), sorted(tr)[len(tr) // 2], max(tr)], "ok": round(sum(x["ok"] for x in sel) / len(sel), 3),
      "srcs": len({x["src"] for x in sel}), "retention_pool": rf, "revisits_per_row": round(spec["steps"] * spec["fit_extra"]["retention_rows"] / rf, 2),
      "screen_v1_revisits": round(3511 * 4 / 2048, 2), "special_dropped": n["special"],
      "rep5_share": round(sum(max(collections.Counter(l for l in x["text"].split("Answer:", 1)[-1].splitlines() if l.strip()).values(), default=0) >= 5 for x in sel) / len(sel), 4), "checks": "holdout 0, drop 0, K1 0, all TRAIN cases, special-token traces dropped"}
json.dump(rc, open(LONG + ".receipt.json", "w"), indent=1); print("long", json.dumps(rc))
