"""vision_v1: the first image TRAIN corpus of v4.0-VL (arm vis-ct-1), from public TRAIN splits only.

    python scripts/build_vision_v1.py --root OUT/vision_v1 [--scale 1.0] [--only a,b] [--dry N]

--dry N: at most N questions per source (dry run into a scratch root).
Writes <root>/vis_<source>.jsonl, <root>/images/..., <root>/hashes_<source>.jsonl,
<root>/train_image_hashes.jsonl (all image hashes, for decontamination) and <root>/manifest.json.
Never reads a test/validation split. InfographicVQA, FigureQA, POPE/COCO-val, MMBench, RealWorldQA and
HallusionBench are eval-only sources (eval_vision_v1, scripts/build_eval_vision.py) and are never read here.
v4.0-VL: 21,588 cases / 33,906 questions, default quotas, --scale 1.0.

Build order of the v4.0-VL image data (each step reads the previous one's output):
  1. build_vision_v1.py  --root V1
  2. build_eval_vision.py --build EB           (held-out candidates; report-only)
  3. decontam_vision.py eval --build EB --train V1 --out EVAL
  4. build_vision_v2.py  --root V2 --v1 V1 --eval-build EB
  5. decontam_vision.py v2 --root V2 --v1 V1 --eval-build EB --eval EVAL
  6. build_vision_v3.py  --root V3 --font-dir DIR
  7. decontam_vision.py v3 --root V3 --eval-build EB [--demo DIR]

Sources. Every image is downloaded from the Hugging Face Hub at a pinned revision
(vision_common.REVISIONS); nothing is shipped. The licence is the ORIGINAL dataset's: the_cauldron
only repackages them ("each sub-dataset is governed by its own licence"; its prompts are CC BY 4.0).
Check each licence before any use beyond research.

  source         HF repo @ revision                              split / subset       licence (original dataset)
  docvqa         HuggingFaceM4/the_cauldron @ 847a98a7...        docvqa train         DocVQA (RRC) terms; images from the UCSF
                                                                                      Industry Documents Library -- unclear
  chartqa        HuggingFaceM4/the_cauldron                      chartqa train        GPL-3.0 (vis-nlp/ChartQA repo); charts from
                                                                                      Statista, Pew, OWID, OECD -- unclear
  vqav2          HuggingFaceM4/the_cauldron                      vqav2 train          VQA annotations CC BY 4.0; COCO images under
                                                                                      Flickr terms of use
  nlvr2          HuggingFaceM4/the_cauldron                      nlvr2 train          NLVR2 annotations CC BY 4.0; web images, no
                                                                                      rights granted -- unclear
  hateful_memes  HuggingFaceM4/the_cauldron                      hateful_memes train  Hateful Memes licence (Facebook), research
                                                                                      only, non-commercial
  screen2words   HuggingFaceM4/the_cauldron                      screen2words train   Screen2Words CC BY 4.0; RICO screenshots
  ai2d           HuggingFaceM4/the_cauldron                      ai2d train           AI2D (AllenAI) CC BY-SA 4.0
  scienceqa      HuggingFaceM4/the_cauldron                      scienceqa train      ScienceQA CC BY-NC-SA 4.0
  tqa            HuggingFaceM4/the_cauldron                      tqa train            TQA (AllenAI, CK-12 material) CC BY-NC 3.0
  tallyqa        HuggingFaceM4/the_cauldron                      tallyqa train        TallyQA Apache-2.0; COCO + Visual Genome images
  aokvqa         HuggingFaceM4/A-OKVQA @ d1b0efa3...             data/train           A-OKVQA Apache-2.0 (annotations); COCO images
  rvlcdip        dvgodoy/rvl_cdip_mini @ 74de67da...             data/train           RVL-CDIP / IIT-CDIP (Legacy Tobacco Documents
                                                                                      Library) terms -- unclear
"""
from __future__ import annotations

import argparse, collections, datetime, json, os, random, re, sys, time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_common as C  # noqa: E402

CAUL = "HuggingFaceM4/the_cauldron"
# questions per source (target); --scale multiplies
QUOTA = {
    "docvqa": 4000, "chartqa": 3500, "screen2words": 2500, "rvlcdip": 2400,
    "ai2d": 2500, "scienceqa": 2500, "tqa": 1500,
    "aokvqa": 3000, "vqav2": 3500, "nlvr2": 3000, "tallyqa": 2500, "hateful_memes": 3000,
}
MAXQ = 4  # questions per case (one image / image pair per case)


def caul_files(sub):
    fs = C.files(CAUL, sub + "/train")
    assert fs and all("/train" in f for f in fs), fs
    return fs


def caul_rows(sub, shard_order=None):
    fs = caul_files(sub)
    for i in (shard_order or range(len(fs))):
        for j, r in enumerate(C.rows(CAUL, fs[i])):
            yield f"{i}-{j}", r


class YNBalance:
    """Keep yes/no labels near 50/50 by skipping the majority when it leads by > slack."""

    def __init__(self, slack=20):
        self.n = [0, 0]; self.slack = slack

    def ok(self, y: int) -> bool:
        return self.n[y] <= self.n[1 - y] + self.slack

    def add(self, y: int):
        self.n[y] += 1


def finish(src, cases, meta):
    meta["questions"] = sum(len(c["questions"]) for c in cases)
    meta["cases"] = len(cases)
    return cases, meta


# ---------------------------------------------------------------- free-form VQA (docvqa, chartqa, vqav2)
def build_freeform(sub, src, quota, root, rng, hashes, png, shard_order):
    pool = C.AnswerPool()
    fs = caul_files(sub)
    # pass 1: text only, the answer pool (same source, train split)
    for i in shard_order:
        p = C.download(CAUL, fs[i])
        for r in pq.read_table(p, columns=["texts"]).to_pylist():
            for t in r["texts"]:
                pool.add(C.strip_prompt_suffix(t["user"]), C.clean_ans(t["assistant"]))
    ynb = YNBalance()
    cases, nq = [], 0
    for rid, r in caul_rows(sub, shard_order):
        if nq >= quota:
            break
        if len(r["images"]) != 1:
            continue
        qs = []
        for t in r["texts"]:
            if len(qs) >= MAXQ:
                break
            q = C.strip_prompt_suffix(t["user"]); a = C.clean_ans(t["assistant"])
            if not q or not a or len(a) > 120:
                continue
            yn = C.is_yn(a)
            if yn is not None:
                if not ynb.ok(int(yn)):
                    continue
            out = C.recast_freeform(f"q{len(qs)}", q, a, pool, rng, want_noul=rng.random() < 0.35)
            if out is None:
                continue
            if out[0].mode == "noul":
                y = int(out[1][1] == 1.0)
                if not ynb.ok(y):
                    continue
                ynb.add(y)
            qs.append(out)
        if not qs:
            continue
        img = C.save_image(root, src, rid, C.img_bytes(r["images"][0]), png, hashes)
        cases.append(C.make_case(f"vis_{src}:{rid}", f"vis_{src}", C.IMG, [img], qs))
        nq += len(qs)
    return finish(src, cases, {"repo": CAUL, "subset": sub, "split": "train",
                               "shards": [fs[i] for i in shard_order], "recast": "free-form -> choice(4, real-answer distractors) / noul",
                               "noul_yes_no": ynb.n})


# ---------------------------------------------------------------- letter MC (ai2d, scienceqa, tqa)
def build_letter_mc(sub, src, quota, root, rng, hashes, png):
    cases, nq = [], 0
    fs = caul_files(sub)
    for rid, r in caul_rows(sub):
        if nq >= quota:
            break
        if len(r["images"]) != 1:
            continue
        ctxs, qs = set(), []
        for t in r["texts"]:
            if len(qs) >= MAXQ:
                break
            p = C.parse_letter_mc(t["user"]); g = C.letter_answer(t["assistant"])
            if p is None or g is None or g >= len(p[2]) or len(p[2]) < 2:
                continue
            ctx, q, ch = p
            ch = [c.rstrip(".").strip() if src == "tqa" else c for c in ch]
            if len(set(x.lower() for x in ch)) != len(ch):
                continue
            ctxs.add(ctx)
            qs.append(C.choice_q(f"q{len(qs)}", q, ch, g, rng))
        if not qs or len(ctxs) > 1:
            continue
        ctx = ctxs.pop()
        state = C.IMG + (f"\n\n{ctx}" if ctx else "")
        img = C.save_image(root, src, rid, C.img_bytes(r["images"][0]), png, hashes)
        cases.append(C.make_case(f"vis_{src}:{rid}", f"vis_{src}", state, [img], qs))
        nq += len(qs)
    return finish(src, cases, {"repo": CAUL, "subset": sub, "split": "train", "shards": fs,
                               "recast": "letter MC, options reshuffled (uniform gold position)"})


# ---------------------------------------------------------------- screen2words: caption choice
def build_screen2words(quota, root, rng, hashes):
    sub, src = "screen2words", "screen2words"
    fs = caul_files(sub)
    shard = [0]
    p = C.download(CAUL, fs[0])
    caps = [C.clean_ans(t["assistant"]) for r in pq.read_table(p, columns=["texts"]).to_pylist() for t in r["texts"]]
    cases, nq = [], 0
    for rid, r in caul_rows(sub, shard):
        if nq >= quota:
            break
        gold = C.clean_ans(r["texts"][0]["assistant"])
        ds = []
        for _ in range(100):
            c = rng.choice(caps)
            if c.lower() != gold.lower() and c not in ds:
                ds.append(c)
            if len(ds) == 3:
                break
        q = C.choice_q("q0", "Which description best matches this app screen?", [gold] + ds, 0, rng)
        img = C.save_image(root, src, rid, C.img_bytes(r["images"][0]), True, hashes)
        cases.append(C.make_case(f"vis_{src}:{rid}", f"vis_{src}", C.IMG, [img], [q]))
        nq += 1
    return finish(src, cases, {"repo": CAUL, "subset": sub, "split": "train", "shards": [fs[0]],
                               "recast": "caption -> choice(4), distractors = captions of other screens"})


# ---------------------------------------------------------------- nlvr2: paired-image noul
def build_nlvr2(quota, root, rng, hashes):
    sub, src = "nlvr2", "nlvr2"
    fs = caul_files(sub)
    ynb = YNBalance()
    cases, nq = [], 0
    for rid, r in caul_rows(sub, [0, 1]):
        if nq >= quota:
            break
        if len(r["images"]) != 2:
            continue
        qs = []
        for t in r["texts"]:
            m = re.search(r'"(.+)"', t["user"], re.S); y = C.is_yn(t["assistant"])
            if not m or y is None or not ynb.ok(int(y)) or len(qs) >= MAXQ:
                continue
            ynb.add(int(y))
            qs.append((C.noul_q(f"q{len(qs)}", f'Is this statement about the two images true? "{m.group(1).strip()}"'),
                       C.one_hot(2, int(y))))
        if not qs:
            continue
        a = C.save_image(root, src, rid + "_L", C.img_bytes(r["images"][0]), False, hashes)
        b = C.save_image(root, src, rid + "_R", C.img_bytes(r["images"][1]), False, hashes)
        cases.append(C.make_case(f"vis_{src}:{rid}", f"vis_{src}", f"Left image: {C.IMG}\nRight image: {C.IMG}", [a, b], qs))
        nq += len(qs)
    return finish(src, cases, {"repo": CAUL, "subset": sub, "split": "train", "shards": fs[:2],
                               "recast": "yes/no claim -> noul", "noul_yes_no": ynb.n})


# ---------------------------------------------------------------- tallyqa: score 0..9
def build_tallyqa(quota, root, rng, hashes):
    sub, src = "tallyqa", "tallyqa"
    fs = caul_files(sub)
    lv = tuple(str(i) for i in range(10))
    crit = {k: (f"{k} or more" if k == "9" else k) for k in lv}
    cap = collections.Counter()
    cases, nq = [], 0
    for rid, r in caul_rows(sub, [0, 1]):
        if nq >= quota:
            break
        if len(r["images"]) != 1:
            continue
        qs = []
        for t in r["texts"]:
            a = C.as_num(C.clean_ans(t["assistant"]))
            if a is None or a != int(a) or a < 0 or len(qs) >= MAXQ:
                continue
            k = min(int(a), 9)
            if cap[k] >= 0.22 * quota:     # flatten the count prior (1-2 dominate)
                continue
            cap[k] += 1
            qs.append((C.Question(f"q{len(qs)}", "score", C.strip_prompt_suffix(t["user"]), lv, crit),
                       C.one_hot(10, k)))
        if not qs:
            continue
        img = C.save_image(root, src, rid, C.img_bytes(r["images"][0]), False, hashes)
        cases.append(C.make_case(f"vis_{src}:{rid}", f"vis_{src}", C.IMG, [img], qs))
        nq += len(qs)
    return finish(src, cases, {"repo": CAUL, "subset": sub, "split": "train", "shards": fs[:2],
                               "recast": "count -> score levels 0..9 (9 = 9 or more), per-level cap 22%",
                               "level_counts": dict(sorted(cap.items()))})


# ---------------------------------------------------------------- hateful memes: noul
def build_hateful(quota, root, rng, hashes):
    sub, src = "hateful_memes", "hateful_memes"
    fs = caul_files(sub)
    ynb = YNBalance(slack=5)
    cases = []
    for rid, r in caul_rows(sub):
        if len(cases) >= quota:
            break
        t = r["texts"][0]; y = C.is_yn(t["assistant"])
        if y is None or not ynb.ok(int(y)) or len(r["images"]) != 1:
            continue
        ynb.add(int(y))
        q = (C.noul_q("q0", C.strip_prompt_suffix(t["user"]) or "Is this meme hateful?"), C.one_hot(2, int(y)))
        img = C.save_image(root, src, rid, C.img_bytes(r["images"][0]), False, hashes)
        cases.append(C.make_case(f"vis_{src}:{rid}", f"vis_{src}", C.IMG, [img], [q]))
    return finish(src, cases, {"repo": CAUL, "subset": sub, "split": "train", "shards": fs,
                               "recast": "hateful label -> noul, balanced", "noul_yes_no": ynb.n})


# ---------------------------------------------------------------- A-OKVQA (original repo, train split)
def build_aokvqa(quota, root, rng, hashes):
    repo, src = "HuggingFaceM4/A-OKVQA", "aokvqa"
    fs = [f for f in C.files(repo, "data/train")]
    cases = []
    for f in fs:
        for j, r in enumerate(C.rows(repo, f)):
            if len(cases) >= quota:
                break
            ch = r.get("choices"); g = r.get("correct_choice_idx")
            if not ch or g is None:
                continue
            q = C.choice_q("q0", r["question"], list(ch), int(g), rng)
            img = C.save_image(root, src, str(r.get("question_id", f"{f[-12:]}-{j}")), C.img_bytes(r["image"]), False, hashes)
            cases.append(C.make_case(f"vis_{src}:{r.get('question_id', j)}", f"vis_{src}", C.IMG, [img], [q]))
        if len(cases) >= quota:
            break
    return finish(src, cases, {"repo": repo, "split": "train", "shards": fs, "recast": "MC choices, reshuffled"})


# ---------------------------------------------------------------- RVL-CDIP (mini, train split): 16-way
def build_rvl(quota, root, rng, hashes):
    repo, src = "dvgodoy/rvl_cdip_mini", "rvlcdip"
    fs = C.files(repo, "data/train")
    allr = [r for f in fs for r in C.rows(repo, f)]
    names = {}
    for r in allr:
        names[int(r["label"])] = r["category"]
    classes = [names[i] for i in sorted(names)]
    per = collections.Counter()
    rng.shuffle(allr)
    cap = max(1, quota // len(classes))
    crit = {c: c.replace("_", " ") for c in classes}
    cases = []
    for j, r in enumerate(allr):
        if per[r["category"]] >= cap:
            continue
        per[r["category"]] += 1
        q = (C.Question("q0", "choice", "What type of document is this?", tuple(classes), crit),
             C.one_hot(len(classes), classes.index(r["category"])))
        img = C.save_image(root, src, f"{j}", C.img_bytes(r["image"]), False, hashes)
        cases.append(C.make_case(f"vis_{src}:{j}", f"vis_{src}", C.IMG, [img], [q]))
    return finish(src, cases, {"repo": repo, "split": "train", "shards": fs, "classes": classes,
                               "recast": "16-way document type choice, stratified"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--dry", type=int, default=0)
    C.add_hf_args(ap)
    a = ap.parse_args()
    C.apply_hf_args(a)
    root = Path(a.root); root.mkdir(parents=True, exist_ok=True)
    only = [s for s in a.only.split(",") if s]
    man_p = root / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"sources": {}}
    for src, q0 in QUOTA.items():
        if only and src not in only:
            continue
        quota = a.dry or int(q0 * a.scale)
        rng = random.Random(f"vision_v1:{src}")
        hashes: list = []
        t0 = time.time()
        try:
            if src in ("docvqa", "chartqa", "vqav2"):
                order = {"docvqa": [2, 3, 1, 10, 12], "chartqa": [0, 1], "vqav2": [0, 1]}[src]
                cases, meta = build_freeform(src, src, quota, root, rng, hashes, src == "chartqa", order)
            elif src in ("ai2d", "scienceqa", "tqa"):
                cases, meta = build_letter_mc(src, src, quota, root, rng, hashes, png=True)
            else:
                cases, meta = {"screen2words": build_screen2words, "nlvr2": build_nlvr2, "tallyqa": build_tallyqa,
                               "hateful_memes": build_hateful, "aokvqa": build_aokvqa, "rvlcdip": build_rvl}[src](
                    quota, root, rng, hashes)
        except Exception as e:  # a source failing must not kill the others
            import traceback; traceback.print_exc()
            man["sources"][src] = {"error": f"{type(e).__name__}: {e}"}
            man_p.write_text(json.dumps(man, indent=1))
            continue
        sha = C.write_jsonl(root / f"vis_{src}.jsonl", cases)
        with open(root / f"hashes_{src}.jsonl", "w") as fh:
            for h in hashes:
                fh.write(json.dumps(h) + "\n")
        modes = collections.Counter(q["mode"] for c in cases for q in c["questions"])
        gold_pos = collections.Counter(q["mode"] + ":" + str(c["gold"][q["key"]].index(1.0))
                                       for c in cases for q in c["questions"])
        meta.update({"file": f"vis_{src}.jsonl", "sha256": sha, "revision": C.rev(meta["repo"]),
                     "images": len(hashes), "modes": dict(modes), "gold_index_counts": dict(sorted(gold_pos.items())),
                     "seconds": round(time.time() - t0)})
        man["sources"][src] = meta
        print(f"[{src}] cases={meta['cases']} q={meta['questions']} images={len(hashes)} modes={dict(modes)} "
              f"{meta['seconds']}s", flush=True)
        man_p.write_text(json.dumps(man, indent=1))
    ok = {k: v for k, v in man["sources"].items() if "error" not in v}
    man.update({"version": "vision_v1", "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "built_by": "scripts/build_vision_v1.py",
                "splits": "public TRAIN splits only",
                "format": "rsijev contract JSONL + images[] (paths relative to this dir); state holds one '<image>' marker per image",
                "eval_only_sources_excluded": ["InfographicVQA", "FigureQA", "POPE/COCO-val2014", "MMBench", "RealWorldQA", "HallusionBench"],
                "total_cases": sum(v["cases"] for v in ok.values()),
                "total_questions": sum(v["questions"] for v in ok.values())})
    man_p.write_text(json.dumps(man, indent=1))
    # one combined hash list for decontamination
    with open(root / "train_image_hashes.jsonl", "w") as out:
        for f in sorted(root.glob("hashes_*.jsonl")):
            out.write(f.read_text())
    print("TOTAL", man["total_cases"], "cases", man["total_questions"], "questions")


if __name__ == "__main__":
    main()
