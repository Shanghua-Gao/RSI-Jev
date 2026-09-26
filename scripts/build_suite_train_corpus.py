"""suite-train-1: TRAIN splits of the eval-suite benchmarks, converted into rsijev Cases
with the SAME conventions the suite uses for their test splits (tool-5 targets_suite).

Runs on the cluster (kev env: python 3.12 + datasets). Stages:

  fetch    upstream TRAIN splits of nimble_public's subsets -> <work>/raw/<name>/train.jsonl
           (HF parquet export with the original field names, as tool-5's build_nimble_public.fetch;
           MASSIVE / VitaminC / MultiNLI come from the archives tool-5 already downloaded)
  kev      kev hard-v1 regenerated with kev's own builder + manifest seeds (verified: the
           regenerated test/development partitions must reproduce the pinned sha256s);
           kev documents-v1 candidate partitions regenerated with kev's own builder
  convert  -> <work>/cases/<source>.jsonl (every usable train row, before any cap)
  filter   drop every training case that overlaps ANY suite test item (tool-5 eval dumps, all 15
           benchmarks, raw = before decontamination): exact state (tool-5 canon() and the arm
           runner's whitespace/lower sha), exact (state, instruction), and 8-gram containment
           >= 0.5 of the eval STATE inside the training row's state (instruction templates excluded). Then nested seeded caps:
           <work>/cap3000/, <work>/cap1000/ (the 1k set is a subset of the 3k set).

    python build_suite_train_corpus.py --work W --stage fetch|kev|convert|filter
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

# Roots. All four are set by main() from command-line arguments and have no
# default: a default here would be one author's filesystem, and the scripts that
# shipped with v1.0 had exactly that bug.
#   TOOL5   the frozen suite definitions (targets_suite.py + decontaminated test splits)
#   T5RAW   the raw upstream archives the nimble sources are extracted from
#   KEV     a checkout of the Kev benchmark repo, which builds its own evals
#   NIMBLE  a checkout of the Nimble repo, for its train split and loaders
TOOL5 = T5RAW = KEV = NIMBLE = None


def _set_roots(suite_dir: Path, kev_repo: "Path | None", nimble_repo: "Path | None") -> None:
    global TOOL5, T5RAW, KEV, NIMBLE
    TOOL5 = Path(suite_dir)
    T5RAW = TOOL5 / "nimble_public/raw"
    KEV = Path(kev_repo) if kev_repo else None
    NIMBLE = Path(nimble_repo) if nimble_repo else None
SEED = "suite-train-1"

# nimble_public subset family -> (nimble module, subsets, how to get the TRAIN split)
def upstream() -> dict:
    """Built after _set_roots, because two entries name archives under T5RAW."""
    return {
        "massive": ("massive", ("en-US", "de-DE"), ("tar", T5RAW / "massive/amazon-massive-dataset-1.1.tar.gz")),
        "boolq": ("boolq", ("",), ("hf", "google/boolq", None, "train")),
        "squad2": ("squad2", ("",), ("hf", "rajpurkar/squad_v2", None, "train")),
        "paws": ("paws", ("",), ("hf", "google-research-datasets/paws", "labeled_final", "train")),
        "multinli": ("multinli", ("",), ("zip", T5RAW / "multinli_1.0.zip", "multinli_1.0_train.jsonl")),
        "civil_comments": ("civil_comments", ("",), ("hf", "google/civil_comments", None, "train")),
        "aegis2": ("aegis2", ("",), ("hf", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None, "train")),
        "helpsteer2": ("helpsteer2", ("",), ("hf", "nvidia/HelpSteer2", None, "train")),
        "vitaminc": ("vitaminc", ("",), ("zip", T5RAW / "vitaminc.zip", "train.jsonl")),
    }


CIVIL_SAMPLE = 60000          # civil_comments train has 1.8M rows; a seeded row sample is converted
NIMBLE_LIMIT = 4000           # records selected (whole families) per upstream subset before filtering


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ fetch
def stage_fetch(work: Path) -> None:
    raw = work / "raw"
    for name, (_, _, how) in upstream().items():
        dst = raw / name / "train.jsonl"
        if dst.exists() or how[0] == "tar":
            print("have", name, flush=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        if how[0] == "hf":
            from datasets import load_dataset
            _, hf_id, cfg, split = how
            ds = load_dataset(hf_id, cfg, split=split) if cfg else load_dataset(hf_id, split=split)
            idx = range(len(ds))
            if name == "civil_comments":
                idx = sorted(random.Random(f"{SEED}:civil").sample(range(len(ds)), CIVIL_SAMPLE))
            with open(tmp, "w", encoding="utf-8") as fh:
                for i in idx:
                    r = dict(ds[i])
                    if name == "civil_comments":
                        r = {"row_index": i, **r}      # the upstream row index is the id
                    fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        else:
            _, z, member = how
            with zipfile.ZipFile(z) as zf:
                m = [n for n in zf.namelist() if n.endswith("/" + member) or n == member]
                m = [n for n in m if "__MACOSX" not in n]
                assert len(m) == 1, (z, member, m)
                tmp.write_bytes(zf.read(m[0]))
        tmp.rename(dst)
        print("fetched", name, sum(1 for _ in open(dst)), flush=True)


# ------------------------------------------------------------------ kev
def stage_kev(work: Path) -> None:
    py = sys.executable
    hard = work / "kev_hard_v1"
    if not (hard / "train.jsonl").exists():
        subprocess.run([py, "scripts/build_hard_v1.py", "--out", str(hard)], cwd=KEV, check=True)
    m = json.loads((KEV / "evals/hard-v1/manifest.json").read_text())
    got = {p: sha256(hard / p) for p in ("train.jsonl", "development.jsonl", "test.jsonl")}
    want = {p: f.get("sha256") for p, f in m.get("files", {}).items()}
    pinned_test = sha256(KEV / "evals/hard-v1/test.jsonl")
    rep = {"regenerated": got, "manifest": want, "pinned_test_file": pinned_test,
           "test_reproduced": got["test.jsonl"] == pinned_test,
           "dev_reproduced": got["development.jsonl"] == sha256(KEV / "evals/hard-v1/development.jsonl"),
           "train_matches_manifest": got["train.jsonl"] == want.get("train.jsonl")}
    print("hard-v1", json.dumps(rep), flush=True)
    (work / "kev_hard_v1_check.json").write_text(json.dumps(rep, indent=1))

    docs = work / "kev_documents_v1_cand"
    if not (docs / "train.jsonl").exists():
        subprocess.run([py, "scripts/build_documents_v1.py", "--out", str(docs)], cwd=KEV, check=True)
    frozen = {}
    for part in ("train", "development", "test"):
        p = KEV / f"evals/documents-v1/{part}.jsonl"
        if p.exists():
            frozen[part] = {json.loads(l)["_meta"]["id"] for l in open(p) if l.strip()}
    cand = {part: {json.loads(l)["_meta"]["id"] for l in open(docs / f"{part}.jsonl") if l.strip()}
            for part in ("train", "development", "test")}
    drep = {"candidates": {k: len(v) for k, v in cand.items()},
            "frozen": {k: len(v) for k, v in frozen.items()},
            "frozen_test_in_cand_test": len(frozen["test"] & cand["test"]),
            "frozen_dev_in_cand_dev": len(frozen["development"] & cand["development"]),
            "frozen_test_in_cand_train": len(frozen["test"] & cand["train"]),
            "frozen_dev_in_cand_train": len(frozen["development"] & cand["train"])}
    print("documents-v1", json.dumps(drep), flush=True)
    (work / "kev_documents_v1_check.json").write_text(json.dumps(drep, indent=1))


# ------------------------------------------------------------------ convert
def _suite_mod():
    sys.path.insert(0, str(TOOL5 / "code"))
    from rsijev import targets_suite as ts          # tool-5's conventions, not re-derived
    from rsijev.contract import Case, dump_cases
    return ts, Case, dump_cases


def kev_cases(path: Path, source: str, ts, Case) -> list:
    out = []
    for i, line in enumerate(open(path)):
        if not line.strip():
            continue
        r = json.loads(line)
        meta = r.get("_meta") or {}
        qs, gold = [], {}
        for key, q in r["questions"].items():
            res = ts.jev_question(key, q, label=q.get("label"), probs=q.get("probabilities") or q.get("target"))
            if res is None:
                continue
            qs.append(res[0]); gold[key] = res[1]
        if qs:
            out.append(Case(case_id=f"{source}:{i:05d}:{meta.get('id', '')}", source=source,
                            state=ts._state_str(r["state"]), questions=tuple(qs), gold=gold))
    return out


def stage_convert(work: Path) -> None:
    ts, Case, dump_cases = _suite_mod()
    sys.path.insert(0, str(NIMBLE))
    from nimble.datasets import public_benchmarks as pb
    from nimble.datasets.public_records import read_jsonl
    sources = pb.load_sources()
    out = work / "cases"
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    for name, (mod, subsets, how) in upstream().items():
        spec = sources[mod]
        for sub in subsets:
            if mod == "massive":
                spec.PARTITION = "train"            # the module reads the test partition by default
                rows = spec.rows(how[1], sub)
            elif mod in ("civil_comments",):
                rows = read_jsonl(work / "raw" / name / "train.jsonl")   # row_index already set
            else:
                rows = spec.rows(work / "raw" / name / "train.jsonl", sub)
            # a few upstream train files repeat an id (MultiNLI train pairIDs): keep the first row per id
            seen_ids, uniq, n_dup = set(), [], 0
            for raw in rows:
                rec = spec.record(raw, sub)
                if rec is not None:
                    if rec["id"] in seen_ids:
                        n_dup += 1
                        continue
                    seen_ids.add(rec["id"])
                uniq.append(raw)
            rows = uniq
            tag = f"{name}{('-' + sub) if sub else ''}"
            od = work / "nimble_train" / tag
            man = pb.build(mod, rows, od, seed=20260925, limit=NIMBLE_LIMIT // len(subsets), subset=sub)
            src = f"st_{tag}"
            cases = []
            for line in open(od / "all.jsonl", encoding="utf-8"):
                if not line.strip():
                    continue
                c = ts._nimble_record(json.loads(line), src, soft=True)
                if c is not None:
                    cases.append(c)
            n = dump_cases(cases, str(out / f"{src}.jsonl"))
            report[tag] = {**{k: man[k] for k in ("converted", "count", "families", "skipped", "types", "labels")},
                           "cases": n, "duplicate_ids_dropped": n_dup}
            print(tag, n, flush=True)

    # nimble's own train split (family-disjoint from its eval.jsonl), hard labels like load_nimble_holdout
    cases = []
    for line in open(NIMBLE / "data/train.jsonl"):
        if line.strip():
            c = ts._nimble_record(json.loads(line), "st_nimble_train", soft=False)
            if c is not None:
                cases.append(c)
    report["nimble_train"] = {"cases": dump_cases(cases, str(out / "st_nimble_train.jsonl"))}

    report["kev_devtools_v1"] = {"cases": dump_cases(
        kev_cases(KEV / "evals/devtools-v1/train.jsonl", "st_kev_devtools_v1", ts, Case),
        str(out / "st_kev_devtools_v1.jsonl"))}
    if (work / "kev_hard_v1/train.jsonl").exists():
        report["kev_hard_v1"] = {"cases": dump_cases(
            kev_cases(work / "kev_hard_v1/train.jsonl", "st_kev_hard_v1", ts, Case),
            str(out / "st_kev_hard_v1.jsonl"))}
    if (work / "kev_documents_v1_cand/train.jsonl").exists():
        report["kev_documents_v1"] = {"cases": dump_cases(
            kev_cases(work / "kev_documents_v1_cand/train.jsonl", "st_kev_documents_v1", ts, Case),
            str(out / "st_kev_documents_v1.jsonl"))}
    (work / "convert_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v.get("cases") for k, v in report.items()}), flush=True)


# ------------------------------------------------------------------ filter + cap
_TOK = re.compile(r"[a-z0-9]+")
_JSON_KEY = re.compile(r'"[A-Za-z_][A-Za-z_ 0-9]{0,30}"\s*:')
_LINE_LABEL = re.compile(r"(?m)^\s*[A-Za-z_][A-Za-z_ 0-9]{0,24}:\s")
K, CONTAIN, MIN_STATE_TOKENS = 8, 0.5, 4


def norm(s): return " ".join(_TOK.findall(str(s).lower()))
def canon(s): return norm(_LINE_LABEL.sub(" ", _JSON_KEY.sub(" ", str(s))))
def runner_norm(s): return hashlib.sha256(re.sub(r"\s+", " ", s).strip().lower().encode()).hexdigest()


def shingles(text):
    w = _TOK.findall(text.lower()[:40000])
    return {" ".join(w[i:i + K]) for i in range(len(w) - K + 1)} if len(w) >= K else set()


def stage_filter(work: Path, caps: list[int]) -> None:
    eval_dirs = [TOOL5 / "eval"]
    docs, sizes = [], []
    state_key, runner_key, item_key = defaultdict(list), defaultdict(list), defaultdict(list)
    index = defaultdict(list)
    seen_files = set()
    for d in eval_dirs:
        for p in sorted(d.glob("*.jsonl")):
            for line in open(p):
                r = json.loads(line)
                tag = (p.stem, r["case_id"])
                if tag in seen_files:
                    continue
                seen_files.add(tag)
                i = len(docs)
                docs.append(tag)
                st = r["state"] if r["state"].strip() else r["questions"][0]["instructions"]
                ns = canon(st)
                if len(ns.split()) >= MIN_STATE_TOKENS:
                    state_key[ns].append(i)
                runner_key[runner_norm(r["state"])].append(i)
                for q in r["questions"]:
                    item_key[canon(r["state"]) + " || " + norm(q["instructions"])].append(i)
                # 8-gram containment on the STATE text: the question instructions and criteria are a
                # fixed per-subset template shared by construction between a benchmark's train and test
                # conversions, so including them makes every short-state row "contain" every eval doc.
                # A state-less (MMLU-style) item is represented by its question text instead.
                sh = shingles(st)
                sizes.append(len(sh))
                for h in sh:
                    index[h].append(i)
    print(f"eval docs {len(docs)} from {len({b for b, _ in docs})} benchmarks", flush=True)

    clean_dir = work / "clean"
    clean_dir.mkdir(exist_ok=True)
    report = {}
    kept_by_src = {}
    for p in sorted((work / "cases").glob("st_*.jsonl")):
        src = p.stem
        kept, drops = [], Counter()
        hit_bench = defaultdict(Counter)
        for line in open(p):
            r = json.loads(line)
            hits = set()
            ns = canon(r["state"])
            for d in state_key.get(ns, ()) if ns else ():
                hits.add((d, "exact_state"))
            for d in runner_key.get(runner_norm(r["state"]), ()):
                hits.add((d, "exact_state_runner"))
            for q in r["questions"]:
                for d in item_key.get(ns + " || " + norm(q["instructions"]), ()):
                    hits.add((d, "exact_item"))
            text = r["state"] if r["state"].strip() else "\n".join(
                q["instructions"] + "\n" + "\n".join(str(v) for v in q["criteria"].values()) for q in r["questions"])
            cnt = Counter()
            for h in shingles(text):
                for d in index.get(h, ()):
                    cnt[d] += 1
            for d, c in cnt.items():
                if sizes[d] and c / sizes[d] >= CONTAIN:
                    hits.add((d, "near_dup_8gram"))
            if hits:
                for d, kind in hits:
                    hit_bench[docs[d][0]][kind] += 1
                drops["dropped_cases"] += 1
                continue
            kept.append(line)
        (clean_dir / p.name).write_text("".join(kept))
        kept_by_src[src] = kept
        report[src] = {"in": len(kept) + drops["dropped_cases"], "kept": len(kept),
                       "dropped": drops["dropped_cases"],
                       "hits_by_benchmark": {b: dict(c) for b, c in hit_bench.items()}}
        print(src, json.dumps(report[src]), flush=True)

    for cap in caps:
        od = work / f"cap{cap}"
        od.mkdir(exist_ok=True)
        creport = {}
        for src, lines in kept_by_src.items():
            c = cap // 2 if src.startswith("st_massive-") else cap     # two locales share one cap
            ls = list(lines); random.Random(f"{SEED}:{src}").shuffle(ls)
            pick = ls[:c]
            (od / f"{src}.jsonl").write_text("".join(pick))
            nq = sum(len(json.loads(l)["questions"]) for l in pick)
            creport[src] = {"cases": len(pick), "questions": nq}
        creport["_total"] = {"cases": sum(v["cases"] for v in creport.values()),
                             "questions": sum(v["questions"] for v in creport.values())}
        report[f"cap{cap}"] = creport
        print(f"cap{cap}", json.dumps(creport), flush=True)
    (work / "filter_report.json").write_text(json.dumps(report, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--suite-dir", required=True,
                    help="the frozen suite definitions (targets_suite.py + test splits)")
    ap.add_argument("--kev-repo", default=None,
                    help="a checkout of the Kev benchmark repo; the kev stage shells out to "
                         "its own builders")
    ap.add_argument("--nimble-repo", default=None,
                    help="a checkout of the Nimble repo, for its train split and loaders")
    ap.add_argument("--stage", required=True, choices=["fetch", "kev", "convert", "filter"])
    ap.add_argument("--caps", default="3000,1000")
    a = ap.parse_args()
    _set_roots(a.suite_dir, a.kev_repo, a.nimble_repo)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    {"fetch": stage_fetch, "kev": stage_kev, "convert": stage_convert}.get(a.stage, lambda w: None)(work)
    if a.stage == "filter":
        stage_filter(work, [int(c) for c in a.caps.split(",")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
