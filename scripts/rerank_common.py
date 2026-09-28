"""Helpers for scripts/build_hippo_rr.py: case records, token fitting, decontamination
and a BM25 index. The logic is unchanged from the data build that produced v3.0's
hippo_rr corpus; only the file locations are parameters instead of constants.

- case() / noul_q(): one rsijev-contract case dict and one noul question.
- Fitter: token length of the longest question render, with the Qwen3.5 tokenizer.
- Decontam: the four-rule overlap check (exact_state, exact_text, contain_eval,
  contain_train), run in the forward direction: a TRAINING item is dropped if it hits
  any benchmark item. benchmark_files() lists the benchmark files to index.
- BM25 / beir(): a sparse BM25 index, and a loader for BEIR corpora + qrels.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

_TOK = re.compile(r"[a-z0-9]+")
_JSON_KEY = re.compile(r'"[A-Za-z_][A-Za-z_ 0-9]{0,30}"\s*:')
_LINE_LABEL = re.compile(r"(?m)^\s*[A-Za-z_][A-Za-z_ 0-9]{0,24}:\s")


def norm(s): return " ".join(_TOK.findall(str(s).lower()))
def canon(s): return norm(_LINE_LABEL.sub(" ", _JSON_KEY.sub(" ", str(s))))


def shingles(t):
    w = _TOK.findall(str(t).lower()[:60000])
    return {" ".join(w[i:i + 8]) for i in range(len(w) - 7)} if len(w) >= 8 else set()


NOUL = ("false", "true")


def noul_q(key, instructions, true_desc="yes", false_desc="no"):
    return {"key": key, "mode": "noul", "instructions": instructions, "options": list(NOUL),
            "criteria": {"false": false_desc, "true": true_desc}}


def case(case_id, source, state, questions, gold, **meta):
    r = {"case_id": case_id, "source": source, "state": state, "questions": questions, "gold": gold}
    if meta:
        r["meta"] = meta
    return r


class Fitter:
    """Token length of state + one question render (instructions, options with criteria, cue)."""
    def __init__(self, tokenizer="Qwen/Qwen3.5-0.8B-Base"):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tokenizer)

    def n(self, text): return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def case_len(self, r):
        worst = 0
        st = self.n(r["state"])
        for q in r["questions"]:
            body = q["instructions"] + "\nOptions:\n" + "\n".join(f"{o}: {q['criteria'].get(o, o)}" for o in q["options"])
            worst = max(worst, st + self.n(body) + 8)
        return worst


def validate(path):
    """Number of cases that load under the rsijev contract (raises on an invalid file)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rsijev.contract import load_cases
    return len(load_cases(path))


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".tmp", path)


def sha256(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------ decontamination
def benchmark_files(dirs=(), parquet_globs=(), extra=()):
    """Every *.jsonl of each benchmark directory (in the order given, files sorted,
    skipping names containing "decontam"), then the parquet files matched by each glob
    (MMLU-Pro test + validation for v3.0), then `extra` (the probe files)."""
    fs = [p for d in dirs
          for p in sorted(glob.glob(f"{d}/*.jsonl")) if "decontam" not in os.path.basename(p)]
    for g in parquet_globs:
        fs += sorted(glob.glob(g))
    return fs + list(extra)


def _rows(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        for r in df.itertuples():
            opts = getattr(r, "options", None)
            yield f"{os.path.basename(path)}:{r.Index}", "", [str(r.question) + " " + " ".join(map(str, opts if opts is not None else []))]
        return
    for i, l in enumerate(open(path, errors="replace")):
        try:
            r = json.loads(l)
        except Exception:
            continue
        st = r.get("state", "")
        st = st if isinstance(st, str) else json.dumps(st)
        qs = r.get("questions") or []
        if isinstance(qs, dict):
            qs = list(qs.values())
        yield r.get("case_id", str(i)), st, [str(q.get("instructions", "")) for q in qs if isinstance(q, dict)]


class Decontam:
    """Index benchmark items; check() a training item against the four rules."""
    def __init__(self, files):
        self.files = files
        self.EV, self.S_KEY, self.T_KEY, self.SH, self.IDX = [], defaultdict(list), defaultdict(list), [], defaultdict(list)
        for f in files:
            name = os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f)
            for cid, st, qs in _rows(f):
                i = len(self.EV)
                self.EV.append((name, cid))
                cs = canon(st)
                if len(cs.split()) >= 4:
                    self.S_KEY[cs].append(i)
                if not st.strip():
                    for q in qs:
                        if len(canon(q).split()) >= 4:
                            self.T_KEY[canon(q)].append(i)
                s = shingles(st if st.strip() else " ".join(qs))
                self.SH.append(len(s))
                for g in s:
                    self.IDX[g].append(i)

    def check(self, state, qtexts):
        hits = set()
        for i in self.S_KEY.get(canon(state), ()):
            hits.add((i, "exact_state"))
        for t in [state] + list(qtexts):
            for i in self.T_KEY.get(canon(t), ()):
                hits.add((i, "exact_text"))
        for text in ([state] if state.strip() else []) + [" ".join(qtexts)]:
            s = shingles(text)
            if not s:
                continue
            cnt = Counter(i for g in s if g in self.IDX for i in self.IDX[g])
            for i, c in cnt.items():
                if self.SH[i] and c / self.SH[i] >= 0.5:
                    hits.add((i, "contain_eval"))
                if len(s) >= 20 and c / len(s) >= 0.5:
                    hits.add((i, "contain_train"))
        return [(self.EV[i][0], self.EV[i][1], k) for i, k in hits]


# ------------------------------------------------------------------ retrieval
class BM25:
    def __init__(self, texts):
        from sklearn.feature_extraction.text import CountVectorizer
        self.cv = CountVectorizer(stop_words="english", min_df=2, max_features=300000, dtype=np.float32)
        X = self.cv.fit_transform(texts).tocsc()
        N = X.shape[0]
        df = np.diff(X.indptr)
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(X.sum(1)).ravel(); avg = dl.mean()
        X = X.tocoo()
        k1, b = 1.2, 0.75
        data = X.data * (k1 + 1) / (X.data + k1 * (1 - b + b * dl[X.row] / avg))
        from scipy.sparse import csr_matrix
        self.W = csr_matrix((data, (X.row, X.col)), shape=X.shape).tocsc()

    def top(self, q, n):
        v = self.cv.transform([q])
        cols = v.indices
        if len(cols) == 0:
            return []
        s = (self.W[:, cols] @ self.idf[cols]).ravel()
        idx = np.argpartition(-s, min(n, len(s) - 1))[:n]
        return list(idx[np.argsort(-s[idx])])


def beir(raw, name, split):
    """BEIR `name` from `raw`: <raw>/BeIR__<name>/{corpus,queries}/*-00000-of-00001.parquet
    and <raw>/BeIR__<name>-qrels/<split>.tsv. Returns (doc text, query text, qrels)."""
    d = f"{raw}/BeIR__{name}"
    corpus = pd.read_parquet(f"{d}/corpus/corpus-00000-of-00001.parquet")
    queries = pd.read_parquet(f"{d}/queries/queries-00000-of-00001.parquet")
    qrels = pd.read_csv(f"{raw}/BeIR__{name}-qrels/{split}.tsv", sep="\t", dtype={"query-id": str, "corpus-id": str})
    corpus = corpus.rename(columns={"_id": "did"}); queries = queries.rename(columns={"_id": "did"})
    text = {str(r.did): ((r.title + ". ") if r.title else "") + r.text for r in corpus.itertuples()}
    qtext = {str(r.did): r.text for r in queries.itertuples()}
    rel = defaultdict(dict)
    for r in qrels.itertuples():
        rel[r[1]][r[2]] = int(r[3])
    return text, qtext, rel
