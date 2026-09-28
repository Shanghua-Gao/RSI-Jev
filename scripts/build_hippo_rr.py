"""hippo_rr: a rerank corpus in hippo's wire format, from MS MARCO / TopiOCQA / FiQA TRAIN.

1,000 states x 3 noul questions (1 gold + 2 hard negatives) = 3,000 q. Each state is
the exact wire format of hippo's batched Jev reranker:
  "Query: <q>\n\nNumbered candidate memories from an AI coding agent's project store:\n\n[i] <c>"
with 20-40 candidates (100-1,200 chars, cut at 1,200 + "..."), noul key c<i>,
"Probability that candidate i (numbered in the state above) helps answer the query.",
criteria true Yes / false No. Gold slot uniform over 1..n.

Sources, TRAIN splits only (never LongMemEval, never a test split):
  msmarco   MS MARCO v1.1 train: gold = an is_selected passage; candidates = the query's own
            10 Bing passages + BM25 passages from a 40k-query train pool. Asked negatives =
            the query's own top-ranked non-selected Bing passages.
  topiocqa  TopiOCQA train (conversational QA, gold evidence = Gold_passage): standalone
            turns only (turn 1, or a topic-switch turn naming its topic). Negatives = BM25
            over all TopiOCQA gold passages of OTHER topics.
  fiqa      BEIR FiQA train qrels: gold = a qrels positive; negatives = BM25 ranks 15-150
            not in qrels.
Two variants of the SAME states/questions/gold slots:
  8k: full candidates, state <= 7,900 tokens (Qwen3.5 tokenizer), gold both within and
      beyond 2,048 tokens.
  2k: every candidate cut to a sentence-aligned window (the gold window keeps the
      answer/rationale when it can be located) so the state fits 1,900 tokens.
Decontamination: the four-rule check of scripts/rerank_common.py against every file in
the --decontam-dir directories, the --decontam-parquet files (MMLU-Pro) and the
--probe-dir probes; LongMemEval / hippo eval references (candidate and query 8-gram
containment >= .3 vs the LongMemEval oracle + hippo's retrieved memories; exact query);
MS MARCO queries already used by an earlier rerank corpus (--exclude-rerank-corpus).

--raw layout (Hugging Face exports):
  microsoft__ms_marco/v1.1/train-00000-of-00001.parquet
  McGill-NLP__TopiOCQA/train.parquet
  BeIR__fiqa/{corpus,queries}/*-00000-of-00001.parquet, BeIR__fiqa-qrels/train.tsv
--lme-dir holds longmemeval_oracle.json and hippo's base.jsonl / ce.jsonl eval dumps
(each row: question, retrieved_memories[].content). They are used only to exclude.

Stages:
  build    -> <out>/corpus/fr3_rerank_mem_{8k,2k}.jsonl (+ .meta.jsonl),
              <out>/decontam_dropped.jsonl, <out>/README.json
  offsets  rewrite the meta files with token offsets measured by the 2B tokenizer
              (gold_tok_offset, gold_end_tok, state_tokens, n_candidates, prompt_tokens,
              gold_within_2k) and add hippo_rr_{8k,2k}.jsonl aliases.
              scripts/build_rl2_sources.py reads n_candidates, so run this before it.

    PYTHONHASHSEED=0 python scripts/build_hippo_rr.py --stage build --raw RAW --lme-dir LME \
        --decontam-dir BENCH1 [--decontam-dir BENCH2 ...] [--decontam-parquet 'GLOB'] \
        --probe-dir PROBES [--exclude-rerank-corpus rerank.jsonl] --out OUT
    python scripts/build_hippo_rr.py --stage offsets --out OUT

CPU only; the build needs about 96 GB of memory (the Decontam index).
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import json
import os
import random
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerank_common import (BM25, Decontam, Fitter, beir, benchmark_files, case,  # noqa: E402
                           noul_q, sha256, validate, write_jsonl)

# Set by main() from command-line arguments; there are no defaults.
#   RAW   upstream TRAIN splits (layout above)
#   OUT   output directory
RAW = OUT = None

HEAD = "Numbered candidate memories from an AI coding agent's project store:"
QT = "Probability that candidate {i} (numbered in the state above) helps answer the query."
N_STATES = {"msmarco": 400, "topiocqa": 350, "fiqa": 250}
MAX8, MAX2 = 7900, 1900
SRC = "fr3_rerank_mem"
rng = random.Random(20260928)
fit = None             # Fitter, built in main()
drops = collections.Counter()

_W = re.compile(r"\w+")
def grams(t, n=8):
    w = _W.findall(str(t).lower()); return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
def ws(t): return re.sub(r"\s+", " ", str(t)).strip()
def cut(t, n=1200): return t[:n] + ("..." if len(t) > n else "")


# ---------------- LongMemEval / hippo references (EVAL ONLY: used to exclude, never as data)
lme_g, lme_q = set(), set()
used_mm = set()


def load_exclusions(E, rerank_corpora):
    for q in json.load(open(f"{E}/longmemeval_oracle.json")):
        lme_q.add(ws(q["question"]).lower())
        for p in [q["question"], str(q["answer"])] + [t.get("content", "") for s in q.get("haystack_sessions", []) for t in s]:
            lme_g.update(grams(p))
    for a in ("base", "ce"):
        for l in open(f"{E}/{a}.jsonl"):
            r = json.loads(l); lme_q.add(ws(r["question"]).lower())
            for m in r["retrieved_memories"]:
                lme_g.update(grams(m["content"]))
    for f in rerank_corpora:
        for l in open(f):
            cid = json.loads(l)["case_id"]
            if ":msmarco:" in cid: used_mm.add(cid.split(":")[3])


def lme_hit(t):
    g = grams(t); return bool(g) and len(g & lme_g) / len(g) >= 0.3


def ok_cand(t):
    if len(t) < 100: drops["cand_short"] += 1; return False
    if lme_hit(t): drops["cand_lme"] += 1; return False
    return True


def sents(t):
    """sentence start offsets"""
    return [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", t)]


def window(t, n, keep=None, r=None):
    """sentence-aligned window of <= n chars; keep = (start,end) span to cover if possible."""
    if len(t) <= n: return t
    st = [s for s in sents(t) if s < len(t) - 40]
    if keep is not None:
        cov = [s for s in st if s <= keep[0] and keep[1] <= s + n]
        s0 = r.choice(cov) if cov else max([s for s in st if s <= keep[0]] or [0])
    else:
        s0 = 0 if r.random() < 0.5 else r.choice(st)
    return cut(t[s0:], n)


def render(query, cands):
    lines = [f"[{i + 1}] {c}" for i, c in enumerate(cands)]
    return f"Query: {query}\n\n{HEAD}\n\n" + "\n\n".join(lines), lines


def make(cid, sub, query, items, gold_i, neg_is, ans):
    """items: list of full texts (already whitespace-normalised) in display order; gold_i/neg_is index into items."""
    asked = [gold_i] + neg_is
    order = asked[:]; rng.shuffle(order)
    qs = [noul_q(f"c{i + 1}", QT.format(i=i + 1), "Yes", "No") for i in order]
    gold = {f"c{i + 1}": [0.0, 1.0] if i == gold_i else [1.0, 0.0] for i in order}
    out = {}
    # 8k variant: full candidates; a state over 7,900 tokens is dropped
    full = [cut(t) for t in items]
    st, lines = render(query, full)
    r8 = case(cid, SRC, st, qs, gold)
    n8 = fit.case_len(r8)
    if n8 > MAX8:
        drops["state_over_8k"] += 1; return None
    pre = st[:st.index(lines[gold_i])]
    m = dict(sub=sub, n_cand=len(items), gold_slot=gold_i + 1, gold_tok_offset=fit.n(pre), tokens=n8)
    m["gold_within_2k"] = m["gold_tok_offset"] + fit.n(lines[gold_i]) <= 2048
    r8["meta"] = m; out["8k"] = r8
    # 2k variant: uniform char cap per candidate, sentence-aligned windows
    lo, hi, best = 60, 1200, None
    while lo <= hi:
        c = (lo + hi) // 2
        rr = random.Random(cid)
        ws_ = [window(t, c, keep=(ans if (i == gold_i and ans) else None), r=rr) for i, t in enumerate(items)]
        r2 = case(cid, SRC, render(query, ws_)[0], qs, gold)
        n2 = fit.case_len(r2)
        if n2 <= MAX2: best, lo = (r2, n2, c), c + 1
        else: hi = c - 1
    if best is None:
        drops["no_2k_fit"] += 1; return None
    r2, n2, c = best
    r2["meta"] = {**m, "tokens": n2, "char_cap": c, "gold_tok_offset": None, "gold_within_2k": True}
    out["2k"] = r2
    return out


def locate(text, a):
    a = ws(a)
    if len(a) < 3: return None
    i = text.lower().find(a.lower()[:120])
    return (i, i + min(len(a), 120)) if i >= 0 else None


def build(sub, gen, n_want):
    got = []
    for cid, query, items, gold_i, neg_is, ans in gen:
        if len(got) >= n_want: break
        if ws(query).lower() in lme_q or lme_hit(query): drops["query_lme"] += 1; continue
        r = make(cid, sub, query, items, gold_i, neg_is, ans)
        if r: got.append(r)
    print(sub, len(got), flush=True)
    return got


def place(gold, negs_ranked, fill, n):
    """negs_ranked: hard negatives (best first, >= 2); fill: more negatives. Returns items, gold_i, asked neg indices."""
    others = (negs_ranked + [f for f in fill if f not in negs_ranked])[:n - 1]
    if len(others) < 19: return None
    rest = others[:]; rng.shuffle(rest)
    g = rng.randrange(len(rest) + 1)
    items = rest[:g] + [gold] + rest[g:]
    neg_is = [items.index(x) for x in negs_ranked[:2]]
    return items, g, neg_is


# ---------------- MS MARCO v1.1 train
def msmarco():
    d = pd.read_parquet(f"{RAW}/microsoft__ms_marco/v1.1/train-00000-of-00001.parquet")
    order = list(range(len(d))); rng.shuffle(order)
    pool = sorted({ws(t) for i in order[:40000] for t in d.iloc[i]["passages"]["passage_text"]})
    pool = [t for t in pool if 100 <= len(t)]
    bm = BM25(pool); print("msmarco pool", len(pool), flush=True)
    for i in order[40000:]:
        row = d.iloc[i]; qid = str(row["query_id"])
        if qid in used_mm: drops["mm_used_by_fac"] += 1; continue
        texts = [ws(t) for t in row["passages"]["passage_text"]]; sel = [int(x) for x in row["passages"]["is_selected"]]
        pos = [t for t, s in zip(texts, sel) if s and ok_cand(t)]
        if not pos: continue
        gold = rng.choice(pos)
        own_neg = [t for t, s in zip(texts, sel) if not s and t not in pos and ok_cand(t)]
        hits = [pool[j] for j in bm.top(row["query"], 80)]
        ext = [t for t in hits if t not in texts and ok_cand(t)]
        allneg = own_neg + ext
        if len(allneg) < 19: continue
        hard = (own_neg + ext)[:2]      # the query's own non-selected Bing passages, in Bing rank order, are the hardest
        n = rng.randint(20, 40)
        p = place(gold, hard, [t for t in allneg if t not in hard], n)
        if not p: continue
        items, g, negs = p
        ans = next((locate(gold, a) for a in row["answers"] if locate(gold, a)), None)
        yield f"{SRC}:msmarco:train:{qid}", row["query"], items, g, negs, ans


# ---------------- TopiOCQA train
def topiocqa():
    d = pd.read_parquet(f"{RAW}/McGill-NLP__TopiOCQA/train.parquet")
    gp = [x if isinstance(x, dict) else ast.literal_eval(x) for x in d["Gold_passage"]]
    pas = {}
    for g in gp:
        pas[g["id"]] = (g["title"].split("[SEP]")[0].strip(), ws(g["title"].replace(" [SEP] ", ": ") + ". " + g["text"]))
    ids = list(pas); bm = BM25([pas[i][1] for i in ids]); print("topiocqa passages", len(ids), flush=True)
    prev_topic = {}
    rows = []
    for k, r in enumerate(d.itertuples()):
        conv, turn = r.Conversation_no, int(r.Turn_no)
        switch = prev_topic.get(conv) not in (None, r.Topic)
        prev_topic[conv] = r.Topic
        if r.Answer.strip().upper() == "UNANSWERABLE": continue
        tw = [w for w in _W.findall(r.Topic.lower()) if len(w) >= 4]
        if turn == 1 or (switch and any(w in r.Question.lower() for w in tw)):
            rows.append((k, r))
    rng.shuffle(rows)
    for k, r in rows:
        g = gp[k]; title, gold = pas[g["id"]]
        if not ok_cand(gold): continue
        hits = [ids[j] for j in bm.top(r.Question, 150)]
        negs = [pas[h][1] for h in hits if pas[h][0] != title and ok_cand(pas[h][1])]
        negs = list(dict.fromkeys(negs))
        if len(negs) < 19: continue
        n = rng.randint(20, 40)
        p = place(gold, negs[:2], negs[2:], n)
        if not p: continue
        items, gi, ni = p
        yield f"{SRC}:topiocqa:train:{r.Conversation_no}:{r.Turn_no}", r.Question, items, gi, ni, locate(gold, r.Rationale)


# ---------------- BEIR FiQA train
def fiqa():
    text, qtext, rel = beir(RAW, "fiqa", "train")
    ids = list(text); bm = BM25([text[i] for i in ids])
    qids = [q for q in rel if q in qtext]; rng.shuffle(qids)
    for qid in qids:
        pos = [ws(text[x]) for x, s in rel[qid].items() if s > 0 and x in text]
        pos = [t for t in pos if ok_cand(t)]
        if not pos: continue
        gold = rng.choice(pos)
        hits = [ids[i] for i in bm.top(qtext[qid], 150)]
        negs = [ws(text[h]) for h in hits[15:] if h not in rel[qid]]
        negs = list(dict.fromkeys(t for t in negs if ok_cand(t)))
        if len(negs) < 19: continue
        n = rng.randint(20, 40)
        p = place(gold, negs[:2], negs[2:], n)
        if not p: continue
        items, g, ni = p
        yield f"{SRC}:fiqa:train:{qid}", qtext[qid], items, g, ni, None


def build_all(decontam_dirs, decontam_parquet, probe_dir):
    built = []
    for sub, gen in (("msmarco", msmarco), ("topiocqa", topiocqa), ("fiqa", fiqa)):
        built += build(sub, gen(), N_STATES[sub] + 60)      # headroom for decontam drops
    probes = sorted(p for p in glob.glob(f"{probe_dir}/fr1p_*.jsonl"))
    dc = Decontam(benchmark_files(decontam_dirs, decontam_parquet, extra=probes))
    kept, dropped, per = [], [], collections.Counter()
    for b in built:
        h = set()
        for v in ("8k", "2k"):
            h |= set(dc.check(b[v]["state"], [q["instructions"] for q in b[v]["questions"]]))
        if h:
            dropped.append({"case_id": b["8k"]["case_id"], "hits": sorted(h)[:5]})
            for bn, _, k in h: per[f"{bn}:{k}"] += 1
        else:
            kept.append(b)
    by = collections.defaultdict(list)
    for b in kept: by[b["8k"]["meta"]["sub"]].append(b)
    final = [b for sub, n in N_STATES.items() for b in by[sub][:n]]
    rng.shuffle(final)
    rep = {"built_states": len(built), "decontam_dropped": len(dropped), "drop_reasons": dict(per), "pre_drops": dict(drops)}
    for v in ("8k", "2k"):
        rows = [b[v] for b in final]
        body = [{k: r[k] for k in ("case_id", "source", "state", "questions", "gold")} for r in rows]
        meta = [{"case_id": r["case_id"], **r["meta"]} for r in rows]
        p = f"{OUT}/corpus/{SRC}_{v}.jsonl"
        write_jsonl(p, body); write_jsonl(p.replace(".jsonl", ".meta.jsonl"), meta)
        toks = sorted(m["tokens"] for m in meta)
        nt = [r["gold"][q["key"]][1] for r in body for q in r["questions"]]
        rep[v] = {"file": p, "sha256": sha256(p), "contract_valid": validate(p), "states": len(body), "questions": len(nt),
                  "noul_true_rate": round(sum(nt) / len(nt), 4),
                  "by_source": dict(collections.Counter(m["sub"] for m in meta)),
                  "tokens": {"median": toks[len(toks) // 2], "p90": toks[int(.9 * len(toks))], "max": toks[-1]},
                  "n_cand_hist": dict(sorted(collections.Counter((m["n_cand"] // 5) * 5 for m in meta).items())),
                  "gold_slot_decile_hist": dict(sorted(collections.Counter(min(9, int(10 * (m["gold_slot"] - 1) / m["n_cand"])) for m in meta).items()))}
        if v == "8k":
            rep[v]["gold_within_2k"] = sum(m["gold_within_2k"] for m in meta)
            rep[v]["gold_tok_offset_hist_1k"] = dict(sorted(collections.Counter(m["gold_tok_offset"] // 1000 for m in meta).items()))
        else:
            cc = sorted(m["char_cap"] for m in meta); rep[v]["char_cap"] = {"median": cc[len(cc) // 2], "min": cc[0], "max": cc[-1]}
    write_jsonl(f"{OUT}/decontam_dropped.jsonl", dropped)
    rep.update({
        "built_by": "scripts/build_hippo_rr.py",
        "purpose": "hippo_rr: in-format relevance nouls over 20-40 long candidates",
        "sources": {"msmarco": "microsoft/ms_marco v1.1 train", "topiocqa": "McGill-NLP/TopiOCQA train parquet",
                    "fiqa": "BeIR/fiqa corpus + train qrels"},
        "splits": "TRAIN only; LongMemEval / hippo eval used only as an exclusion list",
        "format": "hippo batched Jev reranker wire format; 3 asked nouls per state (1 gold + 2 hard negatives), true rate 1/3 by design",
        "decontam": "four-rule Decontam vs " + ", ".join(sorted({os.path.basename(os.path.dirname(f)) for f in dc.files})) +
                    " + probes; LongMemEval oracle + hippo base/ce memories 8-gram containment >= .3 per candidate/query + exact query; earlier rerank corpus msmarco qids excluded"})
    json.dump(rep, open(f"{OUT}/README.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k in ("built_states", "decontam_dropped", "drop_reasons", "pre_drops", "8k", "2k")}, indent=1))


def add_offsets(tokenizer="Qwen/Qwen3.5-2B-Base"):
    """Token offsets of the gold candidate, measured with the 2B tokenizer, written into
    the meta files in place; n_candidates / prompt_tokens alias n_cand / tokens."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    n = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
    H = f"{OUT}/corpus"
    for v in ("8k", "2k"):
        rows = [json.loads(l) for l in open(f"{H}/{SRC}_{v}.jsonl")]
        meta = [json.loads(l) for l in open(f"{H}/{SRC}_{v}.meta.jsonl")]
        for r, m in zip(rows, meta):
            assert r["case_id"] == m["case_id"]
            s = r["state"]; g = m["gold_slot"]; i = s.index(f"\n\n[{g}] ")
            m["gold_tok_offset"] = n(s[:i + 2]); m["gold_end_tok"] = n(s[:s.find(f"\n\n[{g + 1}] ") if f"\n\n[{g + 1}] " in s else len(s)])
            m["state_tokens"] = n(s); m["n_candidates"] = m["n_cand"]; m["prompt_tokens"] = m["tokens"]
            m["gold_within_2k"] = m["gold_end_tok"] <= 2048
        with open(f"{H}/{SRC}_{v}.meta.jsonl", "w") as f:
            for m in meta: f.write(json.dumps(m) + "\n")
        print(v, "gold_within_2k", sum(m["gold_within_2k"] for m in meta), "max state tok", max(m["state_tokens"] for m in meta), "max prompt", max(m["prompt_tokens"] for m in meta))
        os.path.lexists(f"{H}/hippo_rr_{v}.jsonl") or os.symlink(f"{SRC}_{v}.jsonl", f"{H}/hippo_rr_{v}.jsonl")


def main() -> None:
    global RAW, OUT, fit
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["build", "offsets", "all"], default="all")
    ap.add_argument("--out", required=True, help="output directory (corpus/, README.json, ...)")
    ap.add_argument("--raw", help="upstream TRAIN splits (layout in the module docstring)")
    ap.add_argument("--lme-dir", help="longmemeval_oracle.json + hippo base.jsonl / ce.jsonl (exclusion only)")
    ap.add_argument("--decontam-dir", action="append", default=[],
                    help="benchmark directory whose *.jsonl are decontaminated against (repeatable, order kept)")
    ap.add_argument("--decontam-parquet", action="append", default=[],
                    help="glob of benchmark parquet files, e.g. MMLU-Pro test + validation (repeatable)")
    ap.add_argument("--probe-dir", help="directory of fr1p_*.jsonl probe files, also decontaminated against")
    ap.add_argument("--exclude-rerank-corpus", action="append", default=[],
                    help="earlier rerank corpus whose MS MARCO query ids are excluded (repeatable)")
    ap.add_argument("--fit-tokenizer", default="Qwen/Qwen3.5-0.8B-Base", help="tokenizer for the length caps")
    ap.add_argument("--offset-tokenizer", default="Qwen/Qwen3.5-2B-Base", help="tokenizer for the offsets stage")
    a = ap.parse_args()
    OUT = a.out.rstrip("/")
    if a.stage in ("build", "all"):
        missing = [f for f in ("raw", "lme_dir", "probe_dir") if getattr(a, f) is None]
        if missing or not a.decontam_dir:
            ap.error("the build stage needs --raw, --lme-dir, --probe-dir and at least one --decontam-dir")
        RAW = a.raw.rstrip("/")
        fit = Fitter(a.fit_tokenizer)
        load_exclusions(a.lme_dir.rstrip("/"), a.exclude_rerank_corpus)
        build_all(a.decontam_dir, a.decontam_parquet, a.probe_dir.rstrip("/"))
    if a.stage in ("offsets", "all"):
        add_offsets(a.offset_tokenizer)


if __name__ == "__main__":
    main()
