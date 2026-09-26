"""Multi-benchmark evaluation suite: a PROPOSED extension of rsijev/targets.py.

The twelve-benchmark suite: what each benchmark is, where its test split comes
from (pinned by commit) and its licence. This module is not protected and
NOT wired into the evaluator. The orchestrator integrates it into the frozen tree.
Until then it must not be imported by any arm.

Every loader returns `list[Case]` built the way `rsijev.targets` builds them:
- `Question(key, mode, instructions, options, criteria)`: one per decision;
- `Case(case_id, source, state, questions, gold)`: gold is a distribution over
  the question's own options, in the question's own option order.

The conventions below copy `targets.load_typed_decisions` and
`targets.load_mmlu_pro_1k` exactly:
- noul: options are ("false", "true"). The criteria are the item's own dict when
  it has one, otherwise the targets.py default ("The statement is false." /
  "The statement is true.").
- choice with a criteria dict: options are the dict keys, in dict order.
- score with a criteria list: options are "0".."K-1", and the criteria are the
  list entries.
- An MMLU-style multiple choice with no state: state="", options are the
  letters A.., and each letter's criterion is its answer text.
- Soft gold, where the benchmark has it, is renormalised exactly as in
  load_typed_decisions. Hard gold is one-hot (`_one_hot`, as in
  load_mmlu_pro_1k). A noul hard label is (0, 1) for true and (1, 0) for false.

One case the targets.py loaders never meet: a choice option whose description
is null (kev, for plain label spaces like emotion names). Its criterion becomes
the humanised key ("very_negative" -> "very negative"). That is the pool_v2
training convention, so train and eval render these options identically.

Every file-backed benchmark is pinned. A loader checks the sha256 of the file it
reads and refuses a mismatch; HF datasets are loaded at a pinned revision.
TRAIN splits are recorded in `SUITE[name].train` (they may be trained on later;
the test split never).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .contract import Case, Question
from .targets import load_mmlu_pro_1k, load_typed_decisions


# jaredpalmer/kev at commit 569ea449b1033c5aa8f149ec5e20d0c7a296d24a (Apache-2.0),
# cloned by tool-2 into pool_v2/_refs. Override with KEV_ROOT.
class _Unset:
    """A root that was never configured, which explains itself when something uses it.

    Every split in this suite is fetched from a pinned upstream repository or a local
    checkout of one. Those locations differ per machine, so they come from the
    environment -- and they have no default, because a default would be one author's
    filesystem. Importing this module always works; only a loader that actually needs
    a missing root fails, and it says which variable to set.
    """

    def __init__(self, env: str, what: str):
        self._env, self._what = env, what

    def _fail(self):
        raise RuntimeError(
            f"{self._env} is not set. It must point at {self._what}. "
            f"See rsijev/README.md, 'Scoring the suite'.")

    def __truediv__(self, other):
        self._fail()

    def __fspath__(self):
        self._fail()

    def __repr__(self):
        return f"<unset {self._env}>"


def _root(env: str, what: str):
    v = os.environ.get(env)
    return Path(v) if v else _Unset(env, what)


KEV_ROOT = _root("KEV_ROOT", "a checkout of the Kev benchmark repo")
KEV_COMMIT = "569ea449b1033c5aa8f149ec5e20d0c7a296d24a"

NOUL_DEFAULT = {"false": "The statement is false.", "true": "The statement is true."}


def _one_hot(n: int, i: int) -> tuple[float, ...]:
    return tuple(1.0 if k == i else 0.0 for k in range(n))


def _nice(key: str) -> str:
    return str(key).replace("_", " ").strip()


def _state_str(state) -> str:
    """kev states are str, dict or list. Non-strings are rendered as indented JSON,
    the same rendering pool_v2 used for the same sources in training."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checked(path: Path, sha256: str) -> Path:
    got = _sha256(path)
    if got != sha256:
        raise RuntimeError(f"{path}: sha256 {got} != pinned {sha256}; the benchmark file changed")
    return path


# ---------------------------------------------------------------- question builder
def jev_question(key: str, q: dict, *, label=None, probs: dict | None = None
                 ) -> tuple[Question, tuple[float, ...]] | None:
    """A Jev / System-One question dict (type, instructions, criteria) -> Question + gold.

    Mirrors targets.load_typed_decisions. `probs` (soft gold keyed by option) wins
    over `label` (a hard label: an option key, a bool for noul, or a level index
    for score). Returns None when the item cannot be scored, exactly where
    load_typed_decisions would `continue`.
    """
    mode = q["type"]
    raw = q.get("criteria")
    if mode == "noul":
        options = ("false", "true")
        if isinstance(raw, dict) and all(raw.get(o) for o in options):
            criteria = {o: str(raw[o]) for o in options}
        else:
            criteria = dict(NOUL_DEFAULT)
    elif isinstance(raw, dict):
        options = tuple(str(k) for k in raw.keys())
        criteria = {str(k): (str(v) if v not in (None, "") else _nice(k)) for k, v in raw.items()}
    elif isinstance(raw, list):                      # score rubrics are ordered lists
        options = tuple(str(i) for i in range(len(raw)))
        criteria = {str(i): str(d) for i, d in enumerate(raw)}
    else:
        return None
    if len(options) < 2 or mode not in ("choice", "noul", "score"):
        return None

    if probs is not None:
        vec = [float(probs.get(o, 0.0)) for o in options]
    elif mode == "noul":
        if isinstance(label, bool):
            truth = label
        elif str(label).strip().lower() in ("true", "yes", "1"):
            truth = True
        elif str(label).strip().lower() in ("false", "no", "0"):
            truth = False
        else:
            return None
        vec = list(_one_hot(2, 1 if truth else 0))
    elif isinstance(label, int) and not isinstance(label, bool) and isinstance(raw, list):
        if not 0 <= label < len(options):
            return None
        vec = list(_one_hot(len(options), label))
    elif str(label) in options:
        vec = list(_one_hot(len(options), options.index(str(label))))
    else:
        return None
    total = sum(vec)
    if total <= 0:
        return None
    vec = [v / total for v in vec]
    instructions = q["instructions"]
    if isinstance(instructions, dict):             # kev allows {"en": ..., ...}
        instructions = " ".join(str(v) for v in instructions.values() if v)
    return (Question(key=key, mode=mode, instructions=str(instructions),
                     options=options, criteria=criteria), tuple(vec))


def _cases_from_kev_file(path: Path, sha256: str, source: str, *,
                         only_src: Callable[[str], bool] | None = None) -> list[Case]:
    cases: list[Case] = []
    with open(_checked(path, sha256)) as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            r = json.loads(line)
            meta = r.get("_meta") or {}
            questions, gold = [], {}
            for key, q in r["questions"].items():
                if only_src is not None and not only_src(str(q.get("src", ""))):
                    continue
                res = jev_question(key, q, label=q.get("label"),
                                   probs=q.get("probabilities") or q.get("target"))
                if res is None:
                    continue
                questions.append(res[0])
                gold[key] = res[1]
            if questions:
                # the line number keeps ids unique (devtools-v1 repeats one _meta id)
                cid = f"{i:05d}:{meta.get('id', '')}"
                cases.append(Case(case_id=f"{source}:{cid}", source=source,
                                  state=_state_str(r["state"]),
                                  questions=tuple(questions), gold=gold))
    return cases


# ---------------------------------------------------------------- JevBench
# fstandhartinger/jevbench @ 26eb72d4 (MIT). Public tiers only: 109 hard items are sealed.
JEVBENCH_ROOT = _root("JEVBENCH_ROOT", "a checkout of the JevBench repo")
JEVBENCH_COMMIT = "26eb72d4e0e60d8ace0adfc77a384063442561cd"
JEVBENCH_FILES = {  # tier -> sha256 (also recorded in the repo's datasets/manifest.json)
    "easy": "231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b",
    "original": "5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180",
    "hard": "89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb",
}


def load_jevbench_public(tiers=("easy", "original", "hard")) -> list[Case]:
    """JevBench public items: easy 48, original ("standard") 72, hard 111. Gold is the
    authored `expected` label (LLM-authored, cross-model reviewed before any system ran).
    Case.source carries the tier, so the hard tier can be read on its own."""
    cases: list[Case] = []
    for tier in tiers:
        path = _checked(JEVBENCH_ROOT / f"datasets/public/{tier}.jsonl", JEVBENCH_FILES[tier])
        for line in open(path):
            r = json.loads(line)
            q = r["question"]
            exp = r["expected"]
            if q["type"] == "noul":
                exp = {"yes": True, "no": False}.get(str(exp).lower(), exp)
            res = jev_question("decision", q, label=exp)
            if res is None:
                raise ValueError(f"jevbench {r['id']}: cannot build gold from expected={r['expected']!r}")
            cases.append(Case(case_id=f"jevbench:{r['id']}", source=f"jevbench_{tier}",
                              state=_state_str(r["state"]), questions=(res[0],),
                              gold={"decision": res[1]}))
    return cases


# ---------------------------------------------------------------- Nimble
# bespokelabsai/nimble @ 62076b4f. data/eval.jsonl is committed (324 model-checked
# contrastive items); the 13 human-labelled public subsets are rebuilt byte-for-byte
# by scripts/build_nimble_public.py and pinned here by Nimble's committed dataset_sha256.
NIMBLE_ROOT = _root("NIMBLE_ROOT", "a checkout of the Nimble repo")
NIMBLE_COMMIT = "62076b4f2d365b5879dafcf7f6dd072a1fe76df7"
NIMBLE_PUBLIC_DIR = _root("NIMBLE_PUBLIC_DIR", "the built nimble_public panel "
                          "(scripts/build_suite_train_corpus.py --stage fetch writes it)")
NIMBLE_PUBLIC = {  # subset -> committed dataset_sha256 (docs/assets/public-benchmarks/subsets)
    "vitaminc-dev": None, "massive-en-US": None, "massive-de-DE": None, "boolq": None, "squad2": None,
    "paws": None, "multinli": None, "civil_comments": None, "aegis2": None, "helpsteer2": None,
    "summeval-relevance": None, "summeval-consistency": None, "pubmedqa": None,
}


def _nimble_record(r: dict, source: str, *, soft: bool) -> Case | None:
    qs = r["input"]["questions"]
    ref = r["reference"]
    questions, gold = [], {}
    for key, q in qs.items():
        target = ref["target"]
        probs = None
        dist = ref.get("distribution")
        if soft and isinstance(dist, dict):
            probs = {str(k).lower() if q["type"] == "noul" else str(k): float(v) for k, v in dist.items()}
        res = jev_question(key, q, label=target, probs=probs)
        if res is None:
            return None
        qq, g = res
        if probs is not None:
            # a human distribution whose argmax ties away from the scored label would
            # score the label the benchmark does not use; fall back to the label then
            hard = jev_question(key, q, label=target)[1]
            if max(range(len(g)), key=g.__getitem__) != max(range(len(hard)), key=hard.__getitem__):
                g = hard
        questions.append(qq)
        gold[key] = g
    return Case(case_id=f"{source}:{r['id']}", source=source,
                state=_state_str(r["input"]["state"]), questions=tuple(questions), gold=gold)


def load_nimble_holdout() -> list[Case]:
    """Nimble's frozen held-out eval: 324 contrastive (base / counterfactual) items,
    labels from an audited rule over model-verified facts (not human-reviewed)."""
    path = _checked(NIMBLE_ROOT / "data/eval.jsonl",
                    "8e9e48b8de5206593912ae01ddc95bd77e40ad2ecf4c9292c1711290eca0d896")
    out = []
    for line in open(path):
        c = _nimble_record(json.loads(line), "nimble_holdout", soft=False)
        if c is None:
            raise ValueError("nimble holdout: unscorable record")
        out.append(c)
    return out


def load_nimble_public(subsets=None, *, soft: bool = True) -> list[Case]:
    """Nimble's public suite: 13 human-labelled subsets, 3,880 records, each measured
    on Jev 1.13.0 by Nimble. Case.source = nimble_<subset> (report the macro over
    subsets, as Nimble does). multinli (5 annotator votes) and civil_comments (rater
    fraction) carry human distributions, used as soft gold like typed-decisions'."""
    committed_dir = NIMBLE_ROOT / "docs/assets/public-benchmarks/subsets"
    out = []
    for name in (subsets or NIMBLE_PUBLIC):
        want = json.loads((committed_dir / f"{name}-manifest.json").read_text())["dataset_sha256"]
        built = json.loads((NIMBLE_PUBLIC_DIR / name / "manifest.json").read_text())
        if built["dataset_sha256"] != want:
            raise RuntimeError(f"nimble {name}: rebuilt sha {built['dataset_sha256']} != committed {want}")
        for line in open(NIMBLE_PUBLIC_DIR / name / "all.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            c = _nimble_record(json.loads(line), f"nimble_{name}", soft=soft)
            if c is None:
                raise ValueError(f"nimble {name}: unscorable record")
            out.append(c)
    return out


# ---------------------------------------------------------------- Jev-Style panel
# chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-v2 @ e7d734bf reports an 11-task
# real-label panel (evaluation/test.metrics.json, n=300 per task, rte 277) but does
# not publish its item ids. Reconstructed here from the same upstream TEST/eval
# splits at the revisions in its data_manifest.json, seed 20260923 (its seed),
# phrased with rsijev.data's own templates where one exists.
JEV_STYLE_SEED = 20260923
JEV_STYLE_N = 300
HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/7299f6f657089ce06a0f98e7e81f8d0f5b7741ce/heuristics_evaluation_set.txt"
HANS_SHA256 = "c55b62feef9913070e88f38938dc2492018c945ac81f70139346472494124e79"
NLI3 = ("entailment", "neutral", "contradiction")
NLI3_CRIT = {"entailment": "The hypothesis follows from the premise.",
             "neutral": "The hypothesis is neither supported nor contradicted.",
             "contradiction": "The hypothesis contradicts the premise."}


def _sample(n_total: int, n: int, seed_tag: str) -> list[int]:
    rng = random.Random(f"{JEV_STYLE_SEED}:{seed_tag}")
    return sorted(rng.sample(range(n_total), min(n, n_total)))


def _single(source: str, i, state: str, mode: str, instructions: str, options, criteria, gold_idx: int) -> Case:
    q = Question(key="label", mode=mode, instructions=instructions, options=tuple(options),
                 criteria=dict(criteria))
    return Case(case_id=f"{source}:{i}", source=source, state=state, questions=(q,),
                gold={"label": _one_hot(len(options), gold_idx)})


def load_jev_style_panel(tasks=None) -> list[Case]:
    import dataclasses
    import hashlib as _h
    import urllib.request
    from datasets import load_dataset
    from .data import SOURCES, recast

    def hf(name, cfg, split, rev):
        return load_dataset(name, cfg, split=split, revision=rev) if cfg else load_dataset(name, split=split, revision=rev)

    def via_recast(key, name, cfg, split, rev, src_key, n=JEV_STYLE_N):
        ds = hf(name, cfg, split, rev)
        idx = _sample(len(ds), n, key)
        src = dataclasses.replace(SOURCES[src_key], name=f"jevstyle_{key}", split=split)
        return recast(src, [ds[i] for i in idx])

    def nli2(key, rows, f1, f2, is_true, instr, crit_false, crit_true):
        out = []
        for i, r in rows:
            state = f"premise: {r[f1]}\n\nhypothesis: {r[f2]}"
            out.append(_single(f"jevstyle_{key}", i, state, "noul", instr, ("false", "true"),
                               {"false": crit_false, "true": crit_true}, 1 if is_true(r) else 0))
        return out

    builders = {
        "ag_news": lambda: via_recast("ag_news", "fancyzhx/ag_news", None, "test", "eb185aade064a813bc0b7f42de02595523103ca4", "ag_news"),
        "boolq": lambda: via_recast("boolq", "google/boolq", None, "validation", "35b264d03638db9f4ce671b711558bf7ff0f80d5", "boolq"),
        "emotion": lambda: via_recast("emotion", "dair-ai/emotion", "split", "test", "cab853a1dbdf4c42c2b3ef2173804746df8825fe", "emotion"),
        "imdb": lambda: via_recast("imdb", "stanfordnlp/imdb", None, "test", "e6281661ce1c48d982bc483cf8a173c1bbeb5d31", "imdb"),
        "sst5": lambda: via_recast("sst5", "SetFit/sst5", None, "test", "e51bdcd8cd3a30da231967c1a249ba59361279a3", "sst5"),
        "mnli": lambda: via_recast("mnli", "nyu-mll/glue", "mnli", "validation_matched", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "mnli"),
    }

    def anli():
        out = []
        for rnd in ("r1", "r2", "r3"):
            ds = hf("facebook/anli", None, f"test_{rnd}", "8e4813d81f46d313dac7892e1c28076917cfcdf9")
            for i in _sample(len(ds), JEV_STYLE_N // 3, f"anli_{rnd}"):
                r = ds[i]
                out.append(_single("jevstyle_anli", f"{rnd}:{r['uid']}",
                                   f"premise: {r['premise']}\n\nhypothesis: {r['hypothesis']}", "choice",
                                   "What is the relationship between premise and hypothesis?",
                                   NLI3, NLI3_CRIT, int(r["label"])))
        return out

    def sst2():
        ds = hf("nyu-mll/glue", "sst2", "validation", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c")
        return [_single("jevstyle_sst2", ds[i]["idx"], f"text: {ds[i]['sentence']}", "noul",
                        "This text is positive.", ("false", "true"),
                        {"false": "The writer's sentiment is negative.", "true": "The writer's sentiment is positive."},
                        int(ds[i]["label"])) for i in _sample(len(ds), JEV_STYLE_N, "sst2")]

    def rte():
        ds = hf("nyu-mll/glue", "rte", "validation", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c")
        rows = [(ds[i]["idx"], {"p": ds[i]["sentence1"], "h": ds[i]["sentence2"], "y": ds[i]["label"]})
                for i in _sample(len(ds), JEV_STYLE_N, "rte")]
        return nli2("rte", rows, "p", "h", lambda r: r["y"] == 0, "The premise entails the hypothesis.",
                    "The hypothesis does not follow from the premise.", "The hypothesis follows from the premise.")

    def enron():
        ds = hf("SetFit/enron_spam", None, "test", "1916f66c89d52221ae33eb57d44498b4f3a5df22")
        return [_single("jevstyle_enron_spam", ds[i]["message_id"] if "message_id" in ds.column_names else i,
                        f"text: {ds[i]['text']}", "noul", "This email is spam.", ("false", "true"),
                        {"false": "A legitimate (ham) email.", "true": "Unsolicited bulk or scam email (spam)."},
                        int(ds[i]["label"])) for i in _sample(len(ds), JEV_STYLE_N, "enron_spam")]

    def hans():
        cache = LAB / "data/suite_refs/hans_heuristics_evaluation_set.txt"
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(HANS_URL, cache)
        _checked(cache, HANS_SHA256)
        lines = open(cache).read().splitlines()
        head = lines[0].split("\t")
        rows = [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]
        picked = [(rows[i]["pairID"], rows[i]) for i in _sample(len(rows), JEV_STYLE_N, "hans")]
        return nli2("hans", picked, "sentence1", "sentence2", lambda r: r["gold_label"] == "entailment",
                    "The premise entails the hypothesis.",
                    "The hypothesis does not follow from the premise.", "The hypothesis follows from the premise.")

    builders.update(anli=anli, sst2=sst2, rte=rte, enron_spam=enron, hans=hans)
    out: list[Case] = []
    for t in (tasks or builders):
        out += builders[t]()
    # recast's ids hash the state; make them unique within the panel
    seen, uniq = {}, []
    for c in out:
        k = seen.get(c.case_id, 0)
        seen[c.case_id] = k + 1
        uniq.append(c if k == 0 else Case(f"{c.case_id}#{k}", c.source, c.state, c.questions, c.gold))
    return uniq


# ---------------------------------------------------------------- tasksource procedural (test)
TSP_REPO, TSP_REV = "tasksource/procedural-typed-decisions", "b4f55bea135283127b2945c8e6fc7e68e9c56a4a"
TSP_CONFIGS = ("arithmetic", "entity_belief_tracking", "event_state_reconstruction", "evidence_sufficiency",
               "multi_view_adjudication", "needle_retrieval", "partial_observation_calibration",
               "policy_applicability", "record_aggregation", "state_perturbation", "table_lookup")


def load_procedural_test(per_config: int = 40, seed: int = 20260925) -> list[Case]:
    """tasksource/procedural-typed-decisions TEST split (1,000 per config), a seeded
    sample per config. Exact programmatic gold; partial_observation_calibration is
    soft by construction. pool_v2 holds the TRAIN split of all 11 configs (tsp_*)."""
    from datasets import load_dataset
    out = []
    for cfg in TSP_CONFIGS:
        ds = load_dataset(TSP_REPO, cfg, split="test", revision=TSP_REV)
        for i in sorted(random.Random(f"{seed}:{cfg}").sample(range(len(ds)), min(per_config, len(ds)))):
            r = ds[i]
            qs_raw, ans = json.loads(r["questions"]), json.loads(r["answers"])
            questions, gold = [], {}
            for key, q in qs_raw.items():
                a = ans.get(key)
                if a is None:
                    continue
                if q.get("type") == "noul":
                    p = float(a["noul"])
                    probs = {"false": 1.0 - p, "true": p}
                else:
                    probs = a.get("probabilities")
                    if q.get("type") == "score" and isinstance(q.get("criteria"), list) and isinstance(probs, dict):
                        legend = list((a.get("legend") or {}).keys())
                        probs = ({str(j): float(probs.get(k, 0.0)) for j, k in enumerate(legend)}
                                 if legend else probs)
                res = jev_question(key, q, probs=probs)
                if res is None:
                    continue
                questions.append(res[0])
                gold[key] = res[1]
            if questions:
                out.append(Case(case_id=f"procedural_test:{r['id']}", source=f"procedural_{cfg}",
                                state=_state_str(r["state"]), questions=tuple(questions), gold=gold))
    return out


# ---------------------------------------------------------------- Open-Jev v1.1 (ood)
OJ_REPO, OJ_REV = "ZefanCai/Open-Jev-v1.1", "10ad6888333fa97f8c948192797bad3de3040802"


def load_open_jev_ood(per_family: int = 40, seed: int = 20260925, split: str = "ood") -> list[Case]:
    """Open-Jev v1.1 `ood` split (84,267 rows), a seeded sample per source family.
    workflow-controls-v1 (written for typed-decisions' own four workflows) is excluded.
    Soft targets as published (`target` is a distribution over `options`)."""
    import gzip
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(OJ_REPO, f"raw/community-hard-mix-v2-redistributable/{split}.jsonl.gz",
                        repo_type="dataset", revision=OJ_REV)
    by_fam: dict[str, list[dict]] = {}
    for line in gzip.open(p, "rt"):
        r = json.loads(line)
        if r["source"].startswith("workflow-controls-v1"):
            continue
        by_fam.setdefault(r["source"].split("/")[0], []).append(r)
    out = []
    for fam in sorted(by_fam):
        rows = by_fam[fam]
        for i in sorted(random.Random(f"{seed}:{fam}").sample(range(len(rows)), min(per_family, len(rows)))):
            r = rows[i]
            kind, opts, target = r["kind"], [str(o) for o in r["options"]], [float(x) for x in r["target"]]
            if kind == "noul":
                q = {"type": "noul", "instructions": r["question"]}
                yes = opts.index("yes") if "yes" in opts else 1
                probs = {"true": target[yes], "false": 1.0 - target[yes]}
            elif kind == "score":
                q = {"type": "score", "instructions": r["question"], "criteria": opts}
                probs = {str(j): t for j, t in enumerate(target)}
            else:
                short = all(len(o) <= 48 and "\n" not in o for o in opts) and len(set(opts)) == len(opts)
                keys = opts if short else [chr(65 + j) if len(opts) <= 26 else f"option_{j + 1}" for j in range(len(opts))]
                q = {"type": "choice", "instructions": r["question"], "criteria": dict(zip(keys, opts))}
                probs = dict(zip(keys, target))
            res = jev_question("decision", q, probs=probs)
            if res is None:
                continue
            out.append(Case(case_id=f"open_jev_{split}:{r['id']}", source=f"open_jev_{fam}",
                            state=_state_str(r["state"]), questions=(res[0],), gold={"decision": res[1]}))
    return out


# ---------------------------------------------------------------- tasksource-jev (test)
TSJ_REPO, TSJ_REV = "tasksource/tasksource-jev-typed-decisions", "119a1645a029c271f7d768a56dca9e7fa8c7d06c"


def load_tasksource_jev_test(n: int = 1000, seed: int = 20260925) -> list[Case]:
    """tasksource/tasksource-jev-typed-decisions TEST (15,000 rows over ~600 sources),
    a seeded uniform sample. procedural-typed-decisions rows are left out (their own
    benchmark). Gold as published (hard or soft)."""
    import ast
    from huggingface_hub import hf_hub_download
    import pandas as pd
    df = pd.read_parquet(hf_hub_download(TSJ_REPO, "data/test-00000-of-00001.parquet",
                                         repo_type="dataset", revision=TSJ_REV))
    df = df[~df["source"].str.startswith("procedural-typed-decisions")].reset_index(drop=True)
    idx = sorted(random.Random(seed).sample(range(len(df)), min(n, len(df))))

    def arr(x):
        if isinstance(x, str):
            x = x.strip()
            try:
                return list(json.loads(x))
            except json.JSONDecodeError:
                return [float(v) for v in x.strip("[]").split()] if not x.startswith("['") else list(ast.literal_eval(x.replace("' '", "', '")))
        return list(x)

    out = []
    for i in idx:
        r = df.iloc[i]
        kind, opts, target = r["kind"], [str(o) for o in arr(r["options"])], [float(t) for t in arr(r["target"])]
        if kind == "noul":
            p = target[-1] if len(target) in (1, 2) else None
            if p is None:
                continue
            q = {"type": "noul", "instructions": r["question"]}
            probs = {"true": p, "false": 1.0 - p}
        elif kind == "score":
            q = {"type": "score", "instructions": r["question"], "criteria": opts}
            probs = {str(j): t for j, t in enumerate(target)}
        else:
            short = all(len(o) <= 48 and "\n" not in o for o in opts) and len(set(opts)) == len(opts)
            keys = opts if short else [chr(65 + j) if len(opts) <= 26 else f"option_{j + 1}" for j in range(len(opts))]
            q = {"type": "choice", "instructions": r["question"], "criteria": dict(zip(keys, opts))}
            probs = dict(zip(keys, target))
        if kind != "noul" and len(target) != len(opts):
            continue
        # the encoder truncates the STATE, never the options; an item whose options alone
        # overflow 2,048 tokens is refused by rsijev.encode. ~4 chars/token, with margin.
        if sum(len(o) for o in opts) + len(str(r["question"])) > 5000:
            continue
        res = jev_question("decision", q, probs=probs)
        if res is None:
            continue
        out.append(Case(case_id=f"tasksource_jev_test:{r['id']}", source=f"tasksource_{r['source']}",
                        state=str(r["state"]), questions=(res[0],), gold={"decision": res[1]}))
    return out


# ---------------------------------------------------------------- the benchmarks
@dataclass(frozen=True)
class Bench:
    name: str
    loader: Callable[[], list[Case]]
    gold: str                 # "soft teacher" | "hard human" | "hard programmatic" | ...
    modes: tuple[str, ...]
    licence: str
    origin: str               # where the TEST split comes from, pinned
    train: str                # where a TRAIN split is, or "none"
    measures: str
    notes: str = ""
    external: dict = field(default_factory=dict)   # published reference numbers


def load_kev_transfer_v4() -> list[Case]:
    """Kev's locked out-of-domain transfer suite, test partition (764 questions).
    Held-out public sets (MMLU, emotion, tweet_eval/offensive, QNLI, PAWS, SciQ)
    plus programmatic contrastive (authorization, deadline) and composition items."""
    return _cases_from_kev_file(KEV_ROOT / "evals/v4/transfer-v4/test.jsonl",
                                "c30a91274f9b483aac9e4f02ada5dea953b3f1a2829456e0bc07806c4e73b517",
                                "kev_transfer_v4")


def load_kev_hard_v1() -> list[Case]:
    """Kev hard-v1, locked test partition: 700 programmatic states, 1,088 questions
    over 7 families (tradeoff, probability, temporal_numeric, multi_hop, judge,
    long_policy, ambiguous)."""
    return _cases_from_kev_file(KEV_ROOT / "evals/hard-v1/test.jsonl",
                                "246ee92234d33f8e3f10ac6e1542651531b128fd7cc66f3a9964589b77dfa22b",
                                "kev_hard_v1")


def load_kev_documents_v1() -> list[Case]:
    """Kev documents-v1, locked test partition: 574 CFPB complaints, 936 choice
    questions (product + issue), native labels verified by a 3-judge panel."""
    return _cases_from_kev_file(KEV_ROOT / "evals/documents-v1/test.jsonl",
                                "742d04a1abc2207bd3e4f24b281f483cdc69238e4da747cbbc3774e836f2e557",
                                "kev_documents_v1")


def load_kev_devtools_v1() -> list[Case]:
    """Kev devtools-v1 test: 900 states, 1,073 questions (commit type/message
    match, code-review need, flaky test, Aegis safety, when2call, prompt injection)."""
    return _cases_from_kev_file(KEV_ROOT / "evals/devtools-v1/test.jsonl",
                                "2fb7c2a239fef923bcb62c0eee9d6884a608ce30e5e74a4e6b9082787679767f",
                                "kev_devtools_v1")


def load_semif_external() -> list[Case]:
    """SemIf's external decision items as vendored by kev (evidence interpretation,
    rule application, candidate selection): 252 choice questions."""
    return _cases_from_kev_file(KEV_ROOT / "evals/external/semif-v1/development.jsonl",
                                "b9e0c7939a25a04a7ef7bc12334b5fa45b2027d04904d32021295b5539a97303",
                                "semif_external")


def load_scienthoon_ood() -> list[Case]:
    """scienthoon/jev-ood-calibration (val), as vendored by kev: 291 support
    tickets x 3 questions (queue choice, priority score, angry noul)."""
    return _cases_from_kev_file(KEV_ROOT / "evals/external/scienthoon-v1/development.jsonl",
                                "4a6511651209736b490097320d2b2a91480e068c6713bc285f5daa5fcca5fcf5",
                                "scienthoon_ood")


SUITE: dict[str, Bench] = {
    "typed_decisions_test": Bench(
        "typed_decisions_test", lambda: load_typed_decisions("test"),
        "soft teacher (mean of 3 samples of a ~4B teacher at T=0.7)", ("choice", "noul", "score"),
        "see LocalLLaMA/typed-decisions card", "LocalLLaMA/typed-decisions:all/test",
        "LocalLLaMA/typed-decisions:all/train (1,200 cases)",
        "agreement with the benchmark teacher on 4 workflows",
        external={"jev_1.13": 0.727, "majority": 0.5185, "latent_factor_ceiling": 0.704}),
    "mmlu_pro_1k": Bench(
        "mmlu_pro_1k", load_mmlu_pro_1k, "hard human", ("choice",), "MIT",
        "TIGER-Lab/MMLU-Pro:test, random.Random(42).sample 1000",
        "TIGER-Lab/MMLU-Pro:validation (70 rows, few-shot only)",
        "general knowledge (guard)", external={"jev_1.13": 0.829, "openjev": 0.588}),
    "nimble_public": Bench(
        "nimble_public", load_nimble_public, "hard human (multinli, civil_comments: soft human)",
        ("choice", "noul", "score"), "per subset: CC BY-SA 3.0/4.0, CC BY 4.0, CC0, MIT, PAWS free use",
        f"bespokelabsai/nimble@{NIMBLE_COMMIT[:8]} docs/assets/public-benchmarks (13 subsets rebuilt by sha)",
        "upstream train splits: MASSIVE train, BoolQ train, SQuAD2 train, PAWS train, MultiNLI train, "
        "Civil Comments train, Aegis2 train, HelpSteer2 train, VitaminC train; SummEval/PubMedQA-labeled: none",
        "human-labelled Jev-style primitives: routing, RAG answerability, paraphrase, NLI, moderation, "
        "guardrails, rubric rating, summary quality, medical QA, contrastive verification",
        external={"jev_1.13_macro": None, "note": "per-subset Jev 1.13.0 / Nimble-9B accuracies in docs/PUBLIC_BENCHMARKS.md"}),
    "jev_style_panel": Bench(
        "jev_style_panel", load_jev_style_panel, "hard human", ("choice", "noul", "score"),
        "per upstream (AG News, ANLI CC BY-NC 4.0, BoolQ CC BY-SA 3.0, emotion, Enron, HANS MIT, IMDB, GLUE)",
        "11 upstream test/eval splits at chaoliangUNSW data_manifest revisions, seed 20260923, n<=300 each",
        "upstream train splits (all 11 have one; 7 are already in pool_v2)",
        "classic real-label classification panel (in-distribution for pool_v2-trained arms)",
        external={"jev_style_2b_v2_macro_11": 0.812}),
    "kev_transfer_v4": Bench(
        "kev_transfer_v4", load_kev_transfer_v4, "hard human + hard programmatic",
        ("choice", "noul", "score"), "Apache-2.0 (kev); upstream per source",
        f"jaredpalmer/kev@{KEV_COMMIT[:8]} evals/v4/transfer-v4/test.jsonl",
        "none (eval-only suite; kev's trainable sources are decision-v*)",
        "transfer to held-out public label spaces (MMLU, emotion, offensive, QNLI, PAWS, SciQ) + contrastive rules",
        external={"kev_0.8b_locked": 0.684, "kev_4b_locked": 0.837, "kev_9b_locked": 0.852}),
    "kev_hard_v1": Bench(
        "kev_hard_v1", load_kev_hard_v1, "hard programmatic", ("choice", "noul", "score"),
        "Apache-2.0 (kev, generated)", f"jaredpalmer/kev@{KEV_COMMIT[:8]} evals/hard-v1/test.jsonl",
        "kev evals/hard-v1/train.jsonl: 6,000 records, NOT in git (regenerable: kev generator + manifest seeds)",
        "one-pass multi-step decisions: tradeoffs, probability, dates/numbers, multi-hop, judging, long policies, ambiguity"),
    "kev_documents_v1": Bench(
        "kev_documents_v1", load_kev_documents_v1, "hard human (native CFPB label, 3-judge verified)",
        ("choice",), "public domain (US CFPB)",
        f"jaredpalmer/kev@{KEV_COMMIT[:8]} evals/documents-v1/test.jsonl",
        "kev evals/documents-v1/train.jsonl: 5,219 records, NOT in git (source: davidheineman/consumer-finance-complaints-large)",
        "long real documents (consumer complaints) -> product / issue"),
    "kev_devtools_v1": Bench(
        "kev_devtools_v1", load_kev_devtools_v1, "hard human", ("choice", "noul"),
        "CC-BY-4.0 / per-source (kev devtools_v1_licences.json)",
        f"jaredpalmer/kev@{KEV_COMMIT[:8]} evals/devtools-v1/test.jsonl",
        "kev evals/devtools-v1/train.jsonl (5,320; in git; in pool_v2 as kev_devtools_*)",
        "software-engineering decisions"),
    "jevbench_public": Bench(
        "jevbench_public", load_jevbench_public, "hard, LLM-authored + cross-model reviewed",
        ("choice", "noul", "score"), "MIT", f"fstandhartinger/jevbench@{JEVBENCH_COMMIT[:8]} datasets/public/*.jsonl",
        "none (eval-only; 109 hard + 308 fresh items sealed)",
        "Jev-class decisions; the hard tier (111) is the discriminating one",
        external={"jev_1.13_hard": 0.730, "decider_2b_v10_hard": 0.459, "decider_4b_v2_hard": 0.676}),
    "nimble_holdout": Bench(
        "nimble_holdout", load_nimble_holdout, "hard, audited rule over model-verified facts",
        ("choice", "noul", "score"), "no LICENSE file in repo (model is Apache-2.0)",
        f"bespokelabsai/nimble@{NIMBLE_COMMIT[:8]} data/eval.jsonl",
        "bespokelabsai/nimble data/train.jsonl (2,676, source-family disjoint)",
        "contrastive: flip one fact, the answer must flip",
        external={"jev_1.13": 0.932, "nimble_9b": 0.901, "qwen3.5_9b_base": 0.664}),
    "procedural_test": Bench(
        "procedural_test", load_procedural_test, "exact programmatic (one config soft)",
        ("choice", "noul", "score"), "Apache-2.0", f"{TSP_REPO}@{TSP_REV[:8]} */test, 40 per config",
        f"{TSP_REPO} */train (20,000 per config; in pool_v2 as tsp_*)",
        "procedural state tracking, arithmetic, lookup, aggregation"),
    "open_jev_ood": Bench(
        "open_jev_ood", load_open_jev_ood, "mixed (rule / stochastic / human), soft",
        ("choice", "noul", "score"), "CC0-1.0 / CC-BY-4.0 (LICENSE-DATA.md)",
        f"{OJ_REPO}@{OJ_REV[:8]} raw/.../ood.jsonl.gz, 40 per family, workflow-controls excluded",
        f"{OJ_REPO} train (147,139; in pool_v2 as oj_*)", "synthetic controls, held-out seeds"),
    "tasksource_jev_test": Bench(
        "tasksource_jev_test", load_tasksource_jev_test, "human / programmatic, hard or soft",
        ("choice", "noul", "score"), "other (unstated)", f"{TSJ_REPO}@{TSJ_REV[:8]} test, 1,000 sampled",
        f"{TSJ_REPO} train (1,000,000)", "broad regrouping of ~600 public task families"),
    "semif_external": Bench(
        "semif_external", load_semif_external, "hard, authored + perturbations", ("choice",),
        "MIT (TheoLeeCJ/SemIf, repo now private)", f"kev@{KEV_COMMIT[:8]} evals/external/semif-v1/development.jsonl",
        "none", "evidence interpretation and rule application", external={"qwen3.5_4b_direct": 0.813}),
    "scienthoon_ood": Bench(
        "scienthoon_ood", load_scienthoon_ood, "hard, template rules (priority is not inferable from text)",
        ("choice", "noul", "score"), "MIT (scienthoon/jev-ood-calibration, repo now private)",
        f"kev@{KEV_COMMIT[:8]} evals/external/scienthoon-v1/development.jsonl", "none",
        "support-ticket triage", external={"jev_live_queue": 0.890, "jev_live_angry": 0.917, "jev_live_priority": 0.447}),
}


# Recommended suite (tool-5): weights sum to 1. The metric per benchmark is pooled
# top-1; nimble_public and jev_style_panel report the MACRO over their subsets
# (Case.source), as Nimble and chaoliangUNSW do. typed_decisions_test and
# mmlu_pro_1k are read in canonical and reversed order; everything else canonical only.
RECOMMENDED: dict[str, float] = {
    "typed_decisions_test": 0.20, "nimble_public": 0.15, "mmlu_pro_1k": 0.10, "jev_style_panel": 0.10,
    "kev_transfer_v4": 0.08, "kev_hard_v1": 0.08, "jevbench_public": 0.06, "kev_documents_v1": 0.05,
    "kev_devtools_v1": 0.05, "nimble_holdout": 0.05, "procedural_test": 0.04, "open_jev_ood": 0.04,
}
MACRO_OVER_SOURCE = {"nimble_public", "jev_style_panel"}
# Eval cases that overlap training data we hold (tool-5 overlap report); dropped at load time.
DECONTAM = _root("SUITE_DECONTAM", "the decontamination report naming eval cases that "
                 "overlap training data, dropped at load time")


def load_suite(names=None, *, decontam: bool = True) -> dict[str, list[Case]]:
    """Load benchmarks (default: the recommended suite), minus the eval cases the
    tool-5 overlap check found in training data."""
    names = list(RECOMMENDED) if names is None else list(names)
    drop = json.loads(DECONTAM.read_text())["drop"] if decontam else {}
    out = {}
    for n in names:
        bad = set(drop.get(n, ()))
        out[n] = [c for c in SUITE[n].loader() if c.case_id not in bad]
    return out
