"""vision_v2: the second-round image TRAIN corpus of v4.0-VL (arm vis-ct-2), public TRAIN splits only,
plus per-source probes.

    python scripts/build_vision_v2.py --root OUT/vision_v2 --v1 OUT/vision_v1 --eval-build OUT/eval_vision_v1_build \
        [--only a,b] [--dry N]
    python scripts/decontam_vision.py v2 --root OUT/vision_v2 --v1 OUT/vision_v1 \
        --eval-build OUT/eval_vision_v1_build --eval OUT/eval_vision_v1

Per source: builds QUOTA + PROBE questions; the LAST cases (disjoint images) totalling >= PROBE questions go to
<root>/probes_raw/probe_<src>.jsonl, the rest to <root>/vis2_<src>.jsonl. `decontam_vision.py v2` then decontaminates
both and freezes the probes. Every noul portion is balanced to 50/50 (exactly, by construction or by trimming).
Never reads a test/validation split. Eval-only sources of eval_vision_v1 (POPE/COCO-val2014, MMBench, RealWorldQA,
HallusionBench, InfographicVQA) are never read; COCO TRAIN images that coincide with val2014 are removed by
the decontam step (image sha/phash against every eval_vision_v1 candidate image). At build time an image is also
skipped when its phash is within 6 bits of a vision_v1 train image or an eval candidate (`banned`), so --v1 and
--eval-build must be built first.
v4.0-VL: 60,115 train questions (the textvqa file was rebuilt once with `--only textvqa` after the nan/inf answer
filter was added; this is the final code). Of this corpus, vis-ct-3 kept iconqa, visual7w, textvqa and ocrvqa.

Sources (downloaded at the pinned revisions in vision_common.REVISIONS; nothing is shipped; the licence is the
ORIGINAL dataset's, the_cauldron only repackages them and licenses its prompts CC BY 4.0):

  source         HF repo @ revision                              split / subset      licence (original dataset)
  coco_presence  detection-datasets/coco @ cf0b2233...           data/train shards   COCO annotations CC BY 4.0; images under
                                                                 20-22               Flickr terms of use
  vsr            HuggingFaceM4/the_cauldron @ 847a98a7...        vsr train           VSR Apache-2.0 (repo) / CC BY 4.0 (data);
                                                                                     COCO images
  visual7w       HuggingFaceM4/the_cauldron                      visual7w train      Visual7W: unclear; COCO images
  iconqa         HuggingFaceM4/the_cauldron                      iconqa train        IconQA CC BY-NC-SA 4.0
  st_vqa         HuggingFaceM4/the_cauldron                      st_vqa train        ST-VQA: unclear (images from COCO-Text, VG,
                                                                                     VizWiz, ICDAR, ImageNet, IIIT-STR)
  textvqa        HuggingFaceM4/the_cauldron                      textvqa train       TextVQA annotations CC BY 4.0; OpenImages
                                                                                     images (per-image licences, mostly CC BY 2.0)
  ocrvqa         HuggingFaceM4/the_cauldron                      ocrvqa train        OCR-VQA licence: research / education /
                                                                                     non-commercial; book covers (Amazon)
  cocoqa         HuggingFaceM4/the_cauldron                      cocoqa train        COCO-QA: unclear; COCO images
"""
from __future__ import annotations

import argparse, collections, datetime, json, os, random, re, sys, time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_common as C  # noqa: E402
from build_vision_v1 import CAUL, caul_files, caul_rows  # noqa: E402

QUOTA = {"coco_presence": 15000, "vsr": 5000, "visual7w": 8000, "iconqa": 6000, "st_vqa": 6000,
         "textvqa": 11000, "ocrvqa": 6000, "cocoqa": 6000}
PROBE = 300
MAXQ = 4
P = "vis2_"
_BAN = None
BAN_FILES: list[Path] = []   # set by main(): <v1>/train_image_hashes.jsonl, <eval-build>/cand_hashes.jsonl


def banned(raw) -> bool:
    """Pre-filter at build time: the image is a (near) duplicate of a vision_v1 TRAIN image or of any eval_vision_v1
    candidate (phash <= 6 on the decoded original; decontam_v2.py re-checks the saved image at <= 4)."""
    global _BAN
    import numpy as np
    if _BAN is None:
        ph = []
        assert BAN_FILES, "set BAN_FILES (--v1, --eval-build) before building"
        for f in BAN_FILES:
            ph += [int(json.loads(l)["phash"], 16) for l in open(f)]
        _BAN = np.array(ph, dtype=np.uint64)
    im = raw if not isinstance(raw, (bytes, bytearray)) else C.decode(raw)
    h = np.uint64(C.phash(im.convert("RGB")))
    return int(np.bitwise_count(np.bitwise_xor(_BAN, h)).min()) <= 6


class YN:
    """Exact yes/no balance: admit a label only while it does not lead by more than `slack`."""

    def __init__(self, slack=2):
        self.n = [0, 0]; self.slack = slack

    def ok(self, y):
        return self.n[y] < self.n[1 - y] + self.slack

    def add(self, y):
        self.n[y] += 1


def finish(cases, meta, src):
    meta["questions"] = sum(len(c["questions"]) for c in cases); meta["cases"] = len(cases)
    return cases, meta


def trim_noul(cases):
    """Drop noul questions of the majority label (from the end) until yes == no exactly. Cases left empty are dropped."""
    lab = [(i, q["key"], int(c["gold"][q["key"]][1] == 1.0)) for i, c in enumerate(cases)
           for q in c["questions"] if q["mode"] == "noul"]
    n = collections.Counter(y for _, _, y in lab)
    extra = n[1] - n[0]
    maj = 1 if extra > 0 else 0
    drop = set()
    for i, k, y in reversed(lab):
        if len(drop) >= abs(extra):
            break
        if y == maj:
            drop.add((i, k))
    out = []
    for i, c in enumerate(cases):
        qs = [q for q in c["questions"] if (i, q["key"]) not in drop]
        if not qs:
            continue
        if len(qs) != len(c["questions"]):
            c = dict(c); c["questions"] = qs; c["gold"] = {q["key"]: c["gold"][q["key"]] for q in qs}
        out.append(c)
    return out


# ---------------------------------------------------------------- COCO presence (POPE-style, TRAIN annotations)
COCO = "detection-datasets/coco"
COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
          "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
          "scissors", "teddy bear", "hair drier", "toothbrush"]
TEMPL = ["Is there a {} in the image?", "Is there a {} in this image?", "Does the image contain a {}?",
         "Can you see a {} in the image?"]


def coco_names(fpath):
    """Category names from the parquet's HF features metadata when present, else the standard 80-class order."""
    try:
        md = pq.read_schema(fpath).metadata or {}
        info = json.loads(md.get(b"huggingface", b"{}"))
        feats = info["info"]["features"]["objects"]["feature"]["category"]
        names = feats.get("names")
        if names and len(names) == 80:
            return names
    except Exception:
        pass
    return COCO80


def build_coco_presence(src, quota, root, rng, hashes):
    fs = C.files(COCO, "data/train")
    assert fs and all("train" in f for f in fs), fs
    shards = fs[20:23]          # COCO shard 0.. overlaps vision_v1 (A-OKVQA/VQAv2 images) heavily
    paths = [C.download(COCO, f) for f in shards]
    names = coco_names(paths[0])
    # pass 1: frequency + co-occurrence over the shards used (annotations only)
    freq = collections.Counter(); co = collections.defaultdict(collections.Counter)
    for p in paths:
        for r in pq.read_table(p, columns=["objects"]).to_pylist():
            cats = set(r["objects"]["category"])
            for a in cats:
                freq[a] += 1
                for b in cats:
                    if a != b:
                        co[a][b] += 1
    popular = [c for c, _ in freq.most_common()]
    types = ["random", "popular", "adversarial"]
    per_type = {t: [0, 0] for t in types}           # [no, yes]
    cases, nq, img_i = [], 0, 0
    for p, f in zip(paths, shards):
        pf = pq.ParquetFile(p)
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g).to_pylist():
                if nq >= quota:
                    break
                W, H = r["width"], r["height"]
                ob = r["objects"]
                big = collections.defaultdict(float)
                for c, a in zip(ob["category"], ob["area"]):
                    big[c] = max(big[c], a / max(1, W * H))
                present = set(ob["category"])
                pos = [c for c in present if big[c] >= 0.01]     # visible enough to ask about
                if len(pos) < 2:
                    continue
                absent = [c for c in range(80) if c not in present]
                t = types[img_i % 3]
                if t == "random":
                    negs = rng.sample(absent, 2)
                elif t == "popular":
                    negs = [c for c in popular if c not in present][:2]
                else:
                    sc = collections.Counter()
                    for c in present:
                        for b, n in co[c].items():
                            if b not in present:
                                sc[b] += n
                    negs = [c for c, _ in sc.most_common(2)]
                    if len(negs) < 2:
                        continue
                if banned(C.img_bytes(r["image"])):
                    continue
                poss = rng.sample(pos, 2)
                qs = [(c, 1) for c in poss] + [(c, 0) for c in negs]
                rng.shuffle(qs)
                built = []
                for k, (c, y) in enumerate(qs):
                    built.append((C.noul_q(f"q{k}", rng.choice(TEMPL).format(names[c])), C.one_hot(2, y)))
                    per_type[t][y] += 1
                rid = str(r["image_id"])
                img = C.save_image(root, src, rid, C.img_bytes(r["image"]), False, hashes)
                cases.append(C.make_case(f"{P}{src}:{rid}", f"{P}{src}", C.IMG, [img],
                                         built))
                cases[-1]["neg_type"] = t
                nq += 4; img_i += 1
            if nq >= quota:
                break
        if nq >= quota:
            break
    return finish(cases, {"repo": COCO, "split": "train", "shards": shards,
                          "recast": "COCO TRAIN annotations -> POPE-style presence noul: per image 2 present (area >= 1%) + 2 absent of one "
                                    "negative type (random / popular / adversarial = co-occurring), types cycled per image",
                          "per_negative_type_no_yes": per_type, "category_names": "hf features" if names is not COCO80 else "standard COCO-80"}, src)


# ---------------------------------------------------------------- VSR (claims -> noul)
def build_vsr(src, quota, root, rng, hashes):
    sub = "vsr"; fs = caul_files(sub)
    yn = YN(slack=2); cases, nq = [], 0
    for rid, r in caul_rows(sub):
        if nq >= quota:
            break
        if len(r["images"]) != 1:
            continue
        qs = []
        for t in r["texts"]:
            m = re.search(r'"(.+?)"', t["user"], re.S); y = C.is_yn(t["assistant"])
            if not m or y is None or len(qs) >= MAXQ or not yn.ok(int(y)):
                continue
            yn.add(int(y))
            qs.append((C.noul_q(f"q{len(qs)}", f'Is this statement about the image true? "{m.group(1).strip()}"'),
                       C.one_hot(2, int(y))))
        if not qs:
            continue
        if banned(C.img_bytes(r["images"][0])):
            continue
        img = C.save_image(root, src, rid, C.img_bytes(r["images"][0]), False, hashes)
        cases.append(C.make_case(f"{P}{src}:{rid}", f"{P}{src}", C.IMG, [img], qs))
        nq += len(qs)
    return finish(cases, {"repo": CAUL, "subset": sub, "split": "train", "shards": fs, "recast": "spatial claim -> noul"}, src)


# ---------------------------------------------------------------- letter MC (visual7w, iconqa)
def build_letter_mc(src, quota, root, rng, hashes):
    sub = src; fs = caul_files(sub)
    cases, nq = [], 0
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
            ch = [c.rstrip(".").strip() for c in ch]
            if len(set(x.lower() for x in ch)) != len(ch) or any(not x for x in ch):
                continue
            ctxs.add(ctx)
            qs.append(C.choice_q(f"q{len(qs)}", q, ch, g, rng))
        if not qs or len(ctxs) > 1:
            continue
        ctx = ctxs.pop()
        if banned(C.img_bytes(r["images"][0])):
            continue
        img = C.save_image(root, src, rid, C.img_bytes(r["images"][0]), src == "iconqa", hashes)
        cases.append(C.make_case(f"{P}{src}:{rid}", f"{P}{src}", C.IMG + (f"\n\n{ctx}" if ctx else ""), [img], qs))
        nq += len(qs)
    return finish(cases, {"repo": CAUL, "subset": sub, "split": "train", "shards": fs,
                          "recast": "letter MC, options reshuffled (uniform gold position)"}, src)


# ---------------------------------------------------------------- free-form (st_vqa, textvqa, ocrvqa, cocoqa)
SHARDS = {"st_vqa": [0, 1], "textvqa": [0, 1, 2, 3, 4, 5, 6], "ocrvqa": [0], "cocoqa": [0, 1, 2]}


def build_freeform(src, quota, root, rng, hashes):
    sub = src; fs = caul_files(sub); order = SHARDS[src]
    pool = C.AnswerPool()
    for i in order:
        p = C.download(CAUL, fs[i])
        for r in pq.read_table(p, columns=["texts"]).to_pylist():
            for t in r["texts"]:
                pool.add(C.strip_prompt_suffix(t["user"]), C.clean_ans(t["assistant"]))
    yn = YN(slack=2); cases, nq, seen_q = [], 0, set()
    for rid, r in caul_rows(sub, order):
        if nq >= quota:
            break
        if len(r["images"]) != 1:
            continue
        qs = []
        for t in r["texts"]:
            if len(qs) >= MAXQ:
                break
            q = C.strip_prompt_suffix(t["user"]); a = C.clean_ans(t["assistant"])
            if not q or not a or len(a) > 120 or q.lower() in seen_q:
                continue
            v = C.as_num(a)
            if v is not None and (v != v or abs(v) == float("inf")):      # "nan", "inf" answers
                continue
            y0 = C.is_yn(a)
            if y0 is not None and not yn.ok(int(y0)):
                continue
            out = C.recast_freeform(f"q{len(qs)}", q, a, pool, rng, want_noul=rng.random() < 0.3)
            if out is None:
                continue
            if out[0].mode == "noul":
                y = int(out[1][1] == 1.0)
                if not yn.ok(y):
                    continue
                yn.add(y)
            seen_q.add(q.lower())
            qs.append(out)
        seen_q.clear()
        if not qs:
            continue
        if banned(C.img_bytes(r["images"][0])):
            continue
        img = C.save_image(root, src, rid, C.img_bytes(r["images"][0]), src in ("ocrvqa",), hashes)
        cases.append(C.make_case(f"{P}{src}:{rid}", f"{P}{src}", C.IMG, [img], qs))
        nq += len(qs)
    return finish(cases, {"repo": CAUL, "subset": sub, "split": "train", "shards": [fs[i] for i in order],
                          "recast": "free-form -> choice(4, real same-source distractors) / noul (30%, balanced)"}, src)


def split_probe(cases, n_probe):
    """Last cases (disjoint images) totalling >= n_probe questions -> probe."""
    probe, k = [], 0
    while cases and k < n_probe:
        c = cases.pop(); probe.append(c); k += len(c["questions"])
    return cases, probe[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--only", default=""); ap.add_argument("--dry", type=int, default=0)
    ap.add_argument("--v1", required=True, help="the vision_v1 root (reads train_image_hashes.jsonl)")
    ap.add_argument("--eval-build", required=True, help="the eval_vision_v1 build dir (reads cand_hashes.jsonl)")
    C.add_hf_args(ap)
    a = ap.parse_args()
    C.apply_hf_args(a)
    BAN_FILES[:] = [Path(a.v1) / "train_image_hashes.jsonl", Path(a.eval_build) / "cand_hashes.jsonl"]
    root = Path(a.root); root.mkdir(parents=True, exist_ok=True)
    only = [s for s in a.only.split(",") if s]
    man_p = root / "build_manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {"sources": {}}
    for src, q0 in QUOTA.items():
        if only and src not in only:
            continue
        n_probe = min(PROBE, a.dry) if a.dry else PROBE
        quota = (a.dry or q0) + n_probe
        rng = random.Random(f"vision_v2:{src}")
        hashes: list = []
        t0 = time.time()
        try:
            if src == "coco_presence":
                cases, meta = build_coco_presence(src, quota, root, rng, hashes)
            elif src == "vsr":
                cases, meta = build_vsr(src, quota, root, rng, hashes)
            elif src in ("visual7w", "iconqa"):
                cases, meta = build_letter_mc(src, quota, root, rng, hashes)
            else:
                cases, meta = build_freeform(src, quota, root, rng, hashes)
        except Exception as e:
            import traceback; traceback.print_exc()
            man["sources"][src] = {"error": f"{type(e).__name__}: {e}"}
            man_p.write_text(json.dumps(man, indent=1)); continue
        cases, probe = split_probe(cases, n_probe)
        if src != "coco_presence":          # coco is balanced per image by construction
            cases, probe = trim_noul(cases), trim_noul(probe)
        C.write_jsonl(root / f"{P}{src}.jsonl", cases)
        C.write_jsonl(root / "probes_raw" / f"probe_{src}.jsonl", probe)
        with open(root / f"hashes_{src}.jsonl", "w") as fh:
            for h in hashes:
                fh.write(json.dumps(h) + "\n")
        meta.update({"revision": C.rev(meta["repo"]), "images": len(hashes), "seconds": round(time.time() - t0),
                     "questions": sum(len(c["questions"]) for c in cases), "cases": len(cases),
                     "probe_questions": sum(len(c["questions"]) for c in probe), "probe_cases": len(probe)})
        man["sources"][src] = meta
        print(f"[{src}] cases={meta['cases']} q={meta['questions']} probe_q={meta['probe_questions']} images={len(hashes)} "
              f"{meta['seconds']}s", flush=True)
        man_p.write_text(json.dumps(man, indent=1))
    with open(root / "all_image_hashes.jsonl", "w") as out:
        for f in sorted(root.glob("hashes_*.jsonl")):
            out.write(f.read_text())
    man["built_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    man_p.write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
