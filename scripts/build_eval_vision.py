"""eval_vision_v1 step 1: convert VALIDATION/DEV/TEST splits of eval-only image sources to vision cases
(candidates). One question per case. Seeded samples. Report-only held-out set of v4.0-VL; the train-side
decontamination (scripts/decontam_vision.py) reads its image hashes, so it is built before vision_v2/v3.

    python scripts/build_eval_vision.py --build OUT/eval_vision_v1_build [--dry N]
    python scripts/decontam_vision.py eval --build OUT/eval_vision_v1_build --train OUT/vision_v1 --out OUT/eval_vision_v1

Writes BUILD/cand/efvis_<bench>.jsonl, BUILD/images/..., BUILD/cand_hashes.jsonl, BUILD/fetch.json,
BUILD/convert_report.json. None of these sources is read by the train builders.
v4.0-VL (after decontam): mmbench-dev 992, realworldqa 591, pope 853, hallusion 937, infovqa-val 800 = 4,173 q.

Sources (downloaded at the pinned revisions in vision_common.REVISIONS, as recorded in the v4.0-VL fetch.json):

  bench        HF repo @ revision                           split                   licence
  mmbench      lmms-lab/MMBench @ 56ba1af8...               en/dev                  MMBench CC BY 4.0 (OpenCompass); images
                                                                                    collected from other datasets -- unclear
  realworldqa  xai-org/RealworldQA @ 17e7f75e...            data/test               CC BY-ND 4.0 (dataset card)
  pope         lmms-lab/POPE @ 4db12766...                  data/test               POPE (MIT code); COCO val2014 images,
                                                                                    Flickr terms -- unclear
  hallusion    lmms-lab/HallusionBench @ cd417161...        data/image              BSD-3-Clause (tianyi-lab repo); images
                                                                                    from many sources -- unclear
  infovqa      lmms-lab/DocVQA @ 539088ef...                InfographicVQA/valid.   InfographicVQA (RRC) terms; the mirror's
                                                                                    card says apache-2.0 -- unclear
"""
from __future__ import annotations

import argparse, collections, json, os, random, re, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_common as C  # noqa: E402

N = {"mmbench": 1000, "realworldqa": 10**9, "pope": 900, "hallusion": 10**9, "infovqa": 800}
# MMBench dev rows whose 'source' is one of our TRAIN datasets are excluded by name (image decontam below
# catches the rest).
TRAIN_NAMES = ("scienceqa", "ai2d", "tqa", "vqa", "okvqa", "aokvqa", "nlvr", "chartqa", "docvqa", "tally",
               "screen2words", "hateful", "rvl")


def mmbench(b, n, rng, hashes, fetch):
    repo = "lmms-lab/MMBench"; f = "en/dev-00000-of-00001.parquet"
    fetch[f"{repo}/{f}"] = {"rev": C.rev(repo), "split": "dev"}
    rows = list(C.rows(repo, f))
    rng.shuffle(rows)
    out, excl = [], collections.Counter()
    for r in rows:
        if len(out) >= n:
            break
        src = str(r.get("source") or "").lower()
        if any(t in src for t in TRAIN_NAMES):
            excl[src] += 1
            continue
        opts = [(L, r[L]) for L in "ABCD" if r.get(L) not in (None, "", "nan") and str(r.get(L)) != "nan"]
        if len(opts) < 2 or r["answer"] not in [L for L, _ in opts]:
            continue
        texts = [str(t) for _, t in opts]
        g = [L for L, _ in opts].index(r["answer"])
        q, gold = C.choice_q("answer", r["question"], texts, g, rng)
        hint = r.get("hint")
        state = C.IMG + (f"\n\n{hint}" if hint and str(hint) != "nan" else "")
        img = C.save_image(b, "efvis_mmbench", str(r["index"]), C.img_bytes(r["image"]), False, hashes)
        out.append(C.make_case(f"efvis_mmbench:{r['index']}", "efvis_mmbench", state, [img], [(q, gold)]))
    print("mmbench excluded by source name:", dict(excl))
    return out


def realworldqa(b, n, rng, hashes, fetch):
    repo = "xai-org/RealworldQA"
    out, skipped = [], collections.Counter()
    for f in C.files(repo, "data/test"):
        fetch[f"{repo}/{f}"] = {"rev": C.rev(repo), "split": "test"}
        for j, r in enumerate(C.rows(repo, f)):
            if len(out) >= n:
                break
            text = r["question"]; ans = r["answer"].strip().rstrip(".")
            body = re.split(r"\nPlease answer directly", text)[0].strip()
            lines = body.split("\n")
            opt = [(m.group(1), m.group(2).strip()) for l in lines if (m := re.match(r"^([A-F])\.\s*(.+)$", l.strip()))]
            cid = f"efvis_realworldqa:{f[-24:-8]}:{j}"
            if opt and len(opt) >= 2:
                letters = [L for L, _ in opt]
                if ans not in letters:
                    skipped["bad_letter"] += 1; continue
                qtext = "\n".join(l for l in lines if not re.match(r"^([A-F])\.\s", l.strip())).strip()
                q, gold = C.choice_q("answer", qtext, [t for _, t in opt], letters.index(ans), rng)
            elif C.is_yn(ans) is not None:
                q, gold = C.noul_q("answer", body), C.one_hot(2, int(C.is_yn(ans)))
            else:
                skipped["free_form"] += 1; continue
            img = C.save_image(b, "efvis_realworldqa", f"{f[-24:-8]}-{j}", C.img_bytes(r["image"]), False, hashes)
            out.append(C.make_case(cid, "efvis_realworldqa", C.IMG, [img], [(q, gold)]))
    print("realworldqa skipped:", dict(skipped))
    return out


def pope(b, n, rng, hashes, fetch):
    repo = "lmms-lab/POPE"
    rows = []
    for f in C.files(repo, "data/test"):
        fetch[f"{repo}/{f}"] = {"rev": C.rev(repo), "split": "test"}
        rows += list(C.rows(repo, f))
    rng.shuffle(rows)
    per = collections.Counter(); out = []
    for r in rows:
        y = C.is_yn(r["answer"]); key = (r["category"], y)
        if y is None or per[key] >= n // 6:
            continue
        per[key] += 1
        img = C.save_image(b, "efvis_pope", f"{r['category']}-{r['question_id']}", C.img_bytes(r["image"]), False, hashes)
        out.append(C.make_case(f"efvis_pope:{r['category']}:{r['question_id']}", "efvis_pope", C.IMG, [img],
                               [(C.noul_q("answer", r["question"]), C.one_hot(2, int(y)))]))
    return out


def hallusion(b, n, rng, hashes, fetch):
    repo = "lmms-lab/HallusionBench"; f = "data/image-00000-of-00001.parquet"
    fetch[f"{repo}/{f}"] = {"rev": C.rev(repo), "split": "image (the benchmark)"}
    out = []
    for j, r in enumerate(C.rows(repo, f)):
        if str(r.get("visual_input")) == "0" or r.get("image") is None or r["gt_answer"] not in ("0", "1"):
            continue
        img = C.save_image(b, "efvis_hallusion", f"{j}", C.img_bytes(r["image"]), True, hashes)
        out.append(C.make_case(f"efvis_hallusion:{r['category']}:{r['subcategory']}:{r['set_id']}:{r['figure_id']}:{r['question_id']}",
                               "efvis_hallusion", C.IMG, [img],
                               [(C.noul_q("answer", r["question"]), C.one_hot(2, int(r["gt_answer"])))]))
    return out


def infovqa(b, n, rng, hashes, fetch):
    repo = "lmms-lab/DocVQA"
    fs = C.files(repo, "InfographicVQA/validation")
    rows = []
    for f in fs:
        fetch[f"{repo}/{f}"] = {"rev": C.rev(repo), "split": "validation"}
        rows += [r for r in C.rows(repo, f)]
    pool = C.AnswerPool()
    for r in rows:
        if r["answers"]:
            pool.add(r["question"], C.clean_ans(r["answers"][0]))
    rng.shuffle(rows)
    out, per_img, ynb = [], collections.Counter(), [0, 0]
    for r in rows:
        if len(out) >= n:
            break
        if not r["answers"] or per_img[r["image_url"]] >= 2:
            continue
        a = C.clean_ans(r["answers"][0])
        if len(a) > 80:
            continue
        res = C.recast_freeform("answer", r["question"], a, pool, rng, want_noul=len(out) % 2 == 1)
        if res is None:
            continue
        per_img[r["image_url"]] += 1
        img = C.save_image(b, "efvis_infovqa", str(r["questionId"]), C.img_bytes(r["image"]), False, hashes)
        out.append(C.make_case(f"efvis_infovqa:{r['questionId']}", "efvis_infovqa", C.IMG, [img], [res]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--dry", type=int, default=0)
    C.add_hf_args(ap)
    a = ap.parse_args()
    C.apply_hf_args(a)
    b = Path(a.build); (b / "cand").mkdir(parents=True, exist_ok=True)
    fetch, rep = {}, {}
    allh = []
    for name, fn in [("mmbench", mmbench), ("realworldqa", realworldqa), ("pope", pope),
                     ("hallusion", hallusion), ("infovqa", infovqa)]:
        rng = random.Random(f"efvis:{name}")
        hashes = []
        try:
            cases = fn(b, a.dry or N[name], rng, hashes, fetch)
        except Exception as e:
            import traceback; traceback.print_exc()
            rep[name] = {"error": f"{type(e).__name__}: {e}"}; continue
        C.write_jsonl(b / "cand" / f"efvis_{name}.jsonl", cases)
        for h in hashes:
            h["bench"] = name
        allh += hashes
        modes = collections.Counter(q["mode"] for c in cases for q in c["questions"])
        gidx = collections.Counter(str(c["gold"]["answer"].index(1.0)) for c in cases)
        rep[name] = {"cases": len(cases), "modes": dict(modes), "gold_index": dict(sorted(gidx.items()))}
        print(name, rep[name], flush=True)
    with open(b / "cand_hashes.jsonl", "w") as fh:
        for h in allh:
            fh.write(json.dumps(h) + "\n")
    (b / "fetch.json").write_text(json.dumps(fetch, indent=1))
    (b / "convert_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
