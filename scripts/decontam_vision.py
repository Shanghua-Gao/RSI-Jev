"""Decontamination and freezing of the v4.0-VL image data. Three stages, one per corpus, each exactly
as it ran for the release:

    # eval_vision_v1: drop eval candidates that overlap the vision_v1 TRAIN corpus, freeze the set
    python scripts/decontam_vision.py eval --build OUT/eval_vision_v1_build --train OUT/vision_v1 \\
        --out OUT/eval_vision_v1 [--suite TEXT_SUITE_DIR]
    # vision_v2: drop train/probe cases that overlap eval_vision_v1 or vision_v1, re-balance, freeze probes
    python scripts/decontam_vision.py v2 --root OUT/vision_v2 --v1 OUT/vision_v1 \\
        --eval-build OUT/eval_vision_v1_build --eval OUT/eval_vision_v1
    # vision_v3: drop generated cases that match an eval candidate (or a release-demo image), freeze probes
    python scripts/decontam_vision.py v3 --root OUT/vision_v3 --eval-build OUT/eval_vision_v1_build [--demo DIR]

The rules of each stage are in DOC_EVAL, DOC_V2 and DOC_V3 below (the text is copied into each manifest).
Image near-duplicates use the 64-bit DCT phash of vision_common.phash.

Optional inputs that the v4.0-VL build had and the public repo does not ship:
  --suite (eval)  the internal text suite whose exact states were checked (a sanity check). It dropped 0 cases
                  in the v4.0-VL build (drops were img_phash only: hallusion 14, mmbench 8, pope 47), so
                  omitting it reproduces the same eval set. Without it the suite_state rule is skipped.
  --demo (v3)     the release demo's PNGs. In the v4.0-VL build they dropped 0 cases, so omitting it
                  reproduces the same vision_v3 files.
"""
from __future__ import annotations

import argparse, collections, datetime, hashlib, json, os, re, shutil, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PH_MAX = 4
MIN_WORDS = 12


def norm(t: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", t.replace("<image>", " ").lower()))


def grams(t: str, n=8):
    w = t.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


# =================================================================== eval_vision_v1 (decontam.py)
DOC_EVAL = """eval_vision_v1 step 2 -- decontaminate the candidates against the vision TRAIN corpus
and freeze the set.

Drop rules (an eval case is dropped on any):
  img_exact     sha256 of an eval image == a training image
  img_phash     64-bit DCT phash Hamming distance <= 4 to a training image
  text_exact    normalised question text (+ non-marker state text) == a training question/state text,
                AND the text is informative (>= 12 words) -- short generic VQA questions ("Is there a person
                in the image?") are recorded, not dropped, unless the image also matches
  contain_eval  >= 50% of the eval text's 8-grams occur in ONE training text (eval text >= 12 words)
  suite_state   eval state == a state of the text suite (eval_suite_v2) -- sanity check
"""


def case_text(c):
    q = c["questions"][0]
    opts = " ".join(q["criteria"].get(o, "") for o in q["options"]) if q["mode"] == "choice" else ""
    return norm(c["state"] + " " + q["instructions"]), norm(q["instructions"] + " " + opts)


def run_eval(a):
    B, T, O = Path(a.build), Path(a.train), Path(a.out)

    # ---- training side
    th = [json.loads(l) for l in open(T / "train_image_hashes.jsonl")]
    t_sha = {h["sha256"] for h in th}
    t_ph = np.array([int(h["phash"], 16) for h in th], dtype=np.uint64)
    t_text, t_gram_index = set(), collections.defaultdict(set)
    t_files = sorted(T.glob("vis_*.jsonl"))
    tid = 0
    for f in t_files:
        for l in open(f):
            c = json.loads(l)
            texts = [norm(c["state"])] + [norm(q["instructions"]) for q in c["questions"]]
            for t in texts:
                if not t:
                    continue
                t_text.add(t)
                if len(t.split()) >= MIN_WORDS:
                    for g in grams(t):
                        t_gram_index[g].add(tid)
                    tid += 1
    suite_states = set()
    if a.suite:
        for f in sorted(Path(a.suite).glob("*.jsonl")):
            for l in open(f):
                suite_states.add(json.loads(l)["state"])

    # ---- eval side
    eh = {}
    for l in open(B / "cand_hashes.jsonl"):
        h = json.loads(l); eh[h["path"]] = h
    dropped, bench, all_img = [], {}, {}
    O.mkdir(parents=True, exist_ok=True)
    for f in sorted((B / "cand").glob("efvis_*.jsonl")):
        name = f.stem
        cands = [json.loads(l) for l in open(f)]
        keep, reasons, notes = [], collections.Counter(), collections.Counter()
        for c in cands:
            why = []
            for p in c["images"]:
                h = eh[p]
                if h["sha256"] in t_sha:
                    why.append("img_exact")
                d = np.bitwise_count(np.bitwise_xor(t_ph, np.uint64(int(h["phash"], 16)))) if len(t_ph) else np.array([64])
                if d.min() <= PH_MAX:
                    why.append(f"img_phash:{int(d.min())}")
            full, qo = case_text(c)
            for t in {full, qo, norm(c["questions"][0]["instructions"])}:
                if t and t in t_text:
                    if len(t.split()) >= MIN_WORDS:
                        why.append("text_exact")
                    else:
                        notes["short_text_exact_kept"] += 1
            if len(full.split()) >= MIN_WORDS:
                gs = grams(full)
                cnt = collections.Counter(i for g in gs for i in t_gram_index.get(g, ()))
                if cnt and max(cnt.values()) >= 0.5 * len(gs):
                    why.append("contain_eval")
            if c["state"] in suite_states:
                why.append("suite_state")
            if why:
                for w in set(x.split(":")[0] for x in why):
                    reasons[w] += 1
                dropped.append({"benchmark": name, "case_id": c["case_id"], "reasons": why})
                continue
            keep.append(c)
        # copy kept images into the frozen tree
        for c in keep:
            for p in c["images"]:
                dst = O / p
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(B / p, dst)
                all_img[p] = eh[p]["sha256"]
        fo = O / f"{name}.jsonl"
        if fo.exists():
            os.chmod(fo, 0o644)
        with open(fo, "w") as fh:
            for c in keep:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        modes = collections.Counter(c["questions"][0]["mode"] for c in keep)
        bench[name] = {"file": fo.name, "sha256": hashlib.sha256(fo.read_bytes()).hexdigest(),
                       "cases": len(keep), "questions": len(keep), "candidates": len(cands),
                       "dropped_decontam": len(cands) - len(keep), "drop_reasons": dict(reasons),
                       "notes": dict(notes), "modes": dict(modes), "images": len({p for c in keep for p in c["images"]})}
        print(name, bench[name], flush=True)
    dp = O / "decontam_dropped.jsonl"
    if dp.exists():
        os.chmod(dp, 0o644)
    with open(dp, "w") as fh:
        for d in dropped:
            fh.write(json.dumps(d) + "\n")
    ip = O / "images_sha256.json"
    if ip.exists():
        os.chmod(ip, 0o644)
    ip.write_text(json.dumps(dict(sorted(all_img.items())), indent=0))
    fetch = json.loads((B / "fetch.json").read_text())
    man = {
        "version": "eval_vision_v1",
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "built_from": "public VALIDATION/DEV/TEST splits of eval-only vision sources (none is a training source of "
                      "vision_v1); built by scripts/build_eval_vision.py + scripts/decontam_vision.py eval",
        "status": "FROZEN held-out, REPORT-ONLY: read only at release candidates; never for selection or KEEP decisions",
        "format": "rsijev contract JSONL + images[] (paths relative to this dir); state holds one '<image>' marker per image; one question per case",
        "weights": {k: round(1 / len(bench), 6) for k in sorted(bench)},
        "weights_note": "equal per benchmark",
        "sources": fetch,
        "decontamination": {
            "rule": DOC_EVAL.split("Drop rules")[1].strip(),
            "train_corpus": str(T), "train_images_hashed": len(th), "train_files_scanned": [p.name for p in t_files],
            "text_suite_checked": str(a.suite) if a.suite else None, "details": "decontam_dropped.jsonl"},
        "images_sha256_file": {"file": "images_sha256.json", "sha256": hashlib.sha256(ip.read_bytes()).hexdigest(),
                               "images": len(all_img)},
        "benchmarks": bench,
        "total_questions": sum(v["questions"] for v in bench.values()),
    }
    mp = O / "manifest.json"
    if mp.exists():
        os.chmod(mp, 0o644)
    mp.write_text(json.dumps(man, indent=1))
    for p in [*O.glob("efvis_*.jsonl"), ip, mp, dp]:
        os.chmod(p, 0o444)
    for p in (O / "images").rglob("*"):
        if p.is_file():
            os.chmod(p, 0o444)
    src_of = {"efvis_mmbench": "lmms-lab/MMBench en DEV, seeded 1000, rows whose source is a training dataset excluded (choice A-D)",
              "efvis_realworldqa": "xai-org/RealworldQA TEST (the only split): letter MC -> choice, yes/no -> noul, free-form numeric dropped",
              "efvis_pope": "lmms-lab/POPE TEST (COCO val2014): adversarial/popular/random x yes/no, 150 each (noul)",
              "efvis_hallusion": "lmms-lab/HallusionBench image split (the benchmark): yes/no -> noul (charts, tables, maps, OCR, illusions)",
              "efvis_infovqa": "InfographicVQA VALIDATION (lmms-lab/DocVQA InfographicVQA config): free-form -> alternating choice(4)/noul, distractors = other validation answers, <= 2 q per infographic"}
    rows = "\n".join(f"| {k} | {src_of.get(k, '')} | {v['cases']} (of {v['candidates']}) | {', '.join(v['modes'])} | {man['weights'][k]} |"
                     for k, v in sorted(bench.items()))
    suite_line = ", plus exact states of the text suite" if a.suite else ""
    readme = f"""# eval_vision_v1: frozen held-out VISION set (built {man['built_at'][:10]})

**Status: FROZEN, REPORT-ONLY.** Read it only at release candidates; never for selection, KEEP decisions or repair targeting.
Item files, images, images_sha256.json and manifest.json are read-only (0444); manifest.json holds the sha256 of every item file
and of images_sha256.json (sha256 of every image).
Format: rsijev contract JSONL (case_id, source, state, questions, gold) + `images` (paths relative to this dir); the state holds
one `<image>` marker per image. One question per case. Built by scripts/build_eval_vision.py and scripts/decontam_vision.py.

| benchmark | source (split) | cases | modes | weight |
|---|---|---|---|---|
{rows}

Total: {man['total_questions']} questions. None of these sources is a training source of vision_v1 (InfographicVQA,
FigureQA, POPE, MMBench, RealWorldQA, HallusionBench are excluded from training by construction).

## Decontamination (decontam_dropped.jsonl holds the dropped ids and reasons)
Against all {len(th)} training images and all question/state texts of vision_v1{suite_line}:
{man['decontamination']['rule']}
"""
    rp = O / "README.md"
    if rp.exists():
        os.chmod(rp, 0o644)
    rp.write_text(readme)
    print("TOTAL", man["total_questions"])


# =================================================================== vision_v2 (decontam_v2.py)
DOC_V2 = """Decontaminate vision_v2 (train + probes) and freeze the probes.

A case (one image) is dropped on any of:
  eval_img_exact / eval_img_phash  image sha256 == / DCT-phash Hamming <= 4 to ANY eval_vision_v1 candidate image
                                   (<eval-build>/cand_hashes.jsonl, a superset of the frozen set)
  v1_img_exact / v1_img_phash      same against every vision_v1 TRAIN image (no duplicate of a vision_v1 item)
  eval_text_exact                  normalised question text == an eval question text, text >= 12 words
  eval_contain                     >= 50% of the case text's 8-grams inside ONE eval text, or >= 50% of an eval text's
                                   8-grams inside the case text (texts >= 12 words)
  probe_vs_train (probes only)     probe image sha / phash <= 4 to ANY vision_v2 train image (all sources)
After drops, noul yes/no is re-balanced exactly per file (majority-label questions trimmed from the end).
Writes vis2_<src>.jsonl (in place), train_image_hashes.jsonl, manifest.json, decontam_dropped.jsonl,
probes/probe_<src>.jsonl + probes/images/... + probes/manifest.json (0444).
"""


def load_hashes(p):
    hs = [json.loads(l) for l in open(p)]
    return {h["sha256"] for h in hs}, np.array([int(h["phash"], 16) for h in hs], dtype=np.uint64), hs


def ph_min(arr, ph):
    if not len(arr):
        return 64
    return int(np.bitwise_count(np.bitwise_xor(arr, np.uint64(int(ph, 16)))).min())


def run_v2(a):
    from build_vision_v2 import trim_noul, QUOTA
    R = Path(a.root)
    e_sha, e_ph, _ = load_hashes(Path(a.eval_build) / "cand_hashes.jsonl")
    v_sha, v_ph, _ = load_hashes(Path(a.v1) / "train_image_hashes.jsonl")
    # eval texts
    e_text, e_idx, e_texts = set(), collections.defaultdict(set), []
    for f in sorted(Path(a.eval).glob("efvis_*.jsonl")):
        for l in open(f):
            c = json.loads(l)
            for q in c["questions"]:
                opts = " ".join(q["criteria"].get(o, "") for o in q["options"]) if q["mode"] == "choice" else ""
                t = norm(c["state"] + " " + q["instructions"] + " " + opts)
                e_text.add(norm(q["instructions"]))
                if len(t.split()) >= MIN_WORDS:
                    gs = grams(t)
                    for g in gs:
                        e_idx[g].add(len(e_texts))
                    e_texts.append(len(gs))
    own = {}
    for l in open(R / "all_image_hashes.jsonl"):
        h = json.loads(l); own[h["path"]] = h
    srcs = [s for s in QUOTA if (R / f"vis2_{s}.jsonl").exists()]
    # all v2 train images (for probe disjointness)
    tr_paths = {p for s in srcs for l in open(R / f"vis2_{s}.jsonl") for p in json.loads(l)["images"]}
    tr_sha = {own[p]["sha256"] for p in tr_paths}
    tr_ph = np.array([int(own[p]["phash"], 16) for p in tr_paths], dtype=np.uint64)

    def why_drop(c, probe):
        why = []
        for p in c["images"]:
            h = own[p]
            if h["sha256"] in e_sha: why.append("eval_img_exact")
            d = ph_min(e_ph, h["phash"])
            if d <= PH_MAX: why.append(f"eval_img_phash:{d}")
            if h["sha256"] in v_sha: why.append("v1_img_exact")
            d = ph_min(v_ph, h["phash"])
            if d <= PH_MAX: why.append(f"v1_img_phash:{d}")
            if probe:
                if h["sha256"] in tr_sha: why.append("probe_vs_train_exact")
                d = ph_min(tr_ph, h["phash"])
                if d <= PH_MAX: why.append(f"probe_vs_train_phash:{d}")
        for q in c["questions"]:
            qi = norm(q["instructions"])
            if len(qi.split()) >= MIN_WORDS and qi in e_text: why.append("eval_text_exact")
            opts = " ".join(q["criteria"].get(o, "") for o in q["options"]) if q["mode"] == "choice" else ""
            t = norm(c["state"] + " " + q["instructions"] + " " + opts)
            if len(t.split()) >= MIN_WORDS:
                gs = grams(t)
                cnt = collections.Counter(i for g in gs for i in e_idx.get(g, ()))
                if cnt and (max(cnt.values()) >= 0.5 * len(gs) or any(n >= 0.5 * e_texts[i] for i, n in cnt.items())):
                    why.append("eval_contain")
        return why

    dropped, man_src, bench = [], {}, {}
    build_man = json.loads((R / "build_manifest.json").read_text())
    PR = R / "probes"; (PR / "images").mkdir(parents=True, exist_ok=True)
    kept_hashes = []
    for s in srcs:
        for kind in ("train", "probe"):
            f = R / f"vis2_{s}.jsonl" if kind == "train" else R / "probes_raw" / f"probe_{s}.jsonl"
            cases = [json.loads(l) for l in open(f)]
            keep, reasons = [], collections.Counter()
            for c in cases:
                w = why_drop(c, kind == "probe")
                if w:
                    for x in set(y.split(":")[0] for y in w): reasons[x] += 1
                    dropped.append({"file": f.name, "kind": kind, "case_id": c["case_id"], "reasons": w})
                else:
                    keep.append(c)
            n_before = sum(len(c["questions"]) for c in keep)
            keep = trim_noul(keep)
            nq = sum(len(c["questions"]) for c in keep)
            yn = collections.Counter(int(c["gold"][q["key"]][1] == 1.0) for c in keep for q in c["questions"] if q["mode"] == "noul")
            modes = collections.Counter(q["mode"] for c in keep for q in c["questions"])
            info = {"cases": len(keep), "questions": nq, "dropped_cases": len(cases) - len(keep),
                    "drop_reasons": dict(reasons), "trimmed_for_balance": n_before - nq, "modes": dict(modes),
                    "noul_no_yes": [yn[0], yn[1]], "noul_true_rate": round(yn[1] / max(1, yn[0] + yn[1]), 4)}
            if s == "coco_presence":
                pt = collections.defaultdict(lambda: [0, 0])
                for c in keep:
                    for q in c["questions"]:
                        pt[c.get("neg_type", "?")][int(c["gold"][q["key"]][1] == 1.0)] += 1
                info["per_negative_type_no_yes"] = dict(pt)
            if kind == "train":
                for c in keep:
                    c.pop("neg_type", None)
                    for p in c["images"]:
                        kept_hashes.append(own[p])
                out = R / f"vis2_{s}.jsonl"
                with open(out, "w") as fh:
                    for c in keep: fh.write(json.dumps(c, ensure_ascii=False) + "\n")
                info.update(file=out.name, sha256=hashlib.sha256(out.read_bytes()).hexdigest())
                man_src[s] = {**{k: v for k, v in build_man["sources"][s].items()
                                 if k not in ("questions", "cases", "probe_questions", "probe_cases")}, **info}
            else:
                for c in keep:
                    c.pop("neg_type", None)
                    newp = []
                    for p in c["images"]:
                        dst = PR / p
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if not dst.exists():
                            shutil.move(str(R / p), dst) if (R / p).exists() else None
                        newp.append(p)
                    c["images"] = newp
                out = PR / f"probe_{s}.jsonl"
                if out.exists(): os.chmod(out, 0o644)
                with open(out, "w") as fh:
                    for c in keep: fh.write(json.dumps(c, ensure_ascii=False) + "\n")
                bench[f"probe_{s}"] = {"file": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), **info}
            print(s, kind, info, flush=True)
    with open(R / "decontam_dropped.jsonl", "w") as fh:
        for d in dropped: fh.write(json.dumps(d) + "\n")
    with open(R / "train_image_hashes.jsonl", "w") as fh:
        for h in kept_hashes: fh.write(json.dumps(h) + "\n")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rule = DOC_V2.split("A case (one image) is dropped on any of:")[1].split("Writes")[0].strip()
    yn_all = [sum(v["noul_no_yes"][0] for v in man_src.values()), sum(v["noul_no_yes"][1] for v in man_src.values())]
    man = {"version": "vision_v2", "built_at": now, "built_by": "scripts/build_vision_v2.py, scripts/decontam_vision.py v2",
           "splits": "public TRAIN splits only", "purpose": "vis-ct-2 new sources (each file = one tagged corpus source)",
           "format": "rsijev contract JSONL + images[] (paths relative to this dir); one '<image>' marker per image",
           "decontamination": {"rule": rule, "eval": a.eval, "eval_candidates": a.eval_build, "v1": a.v1,
                               "details": "decontam_dropped.jsonl"},
           "sources": man_src, "noul_no_yes_total": yn_all,
           "total_cases": sum(v["cases"] for v in man_src.values()),
           "total_questions": sum(v["questions"] for v in man_src.values())}
    (R / "manifest.json").write_text(json.dumps(man, indent=1))
    pm = {"version": "vision_v2_probes", "built_at": now,
          "status": "FROZEN per-source probes for the vis-ct-2 keep/scale/drop rule: held out from vision_v2 TRAIN (disjoint "
                    "images, phash-checked against every v2 train image); diagnostic, not a report set",
          "format": "same as eval_vision_v1 (images relative to this dir)", "weights": {k: round(1 / len(bench), 6) for k in bench},
          "benchmarks": bench, "total_questions": sum(v["questions"] for v in bench.values())}
    mp = PR / "manifest.json"
    if mp.exists(): os.chmod(mp, 0o644)
    mp.write_text(json.dumps(pm, indent=1))
    for p in [*PR.glob("probe_*.jsonl"), mp]:
        os.chmod(p, 0o444)
    for p in (PR / "images").rglob("*"):
        if p.is_file(): os.chmod(p, 0o444)
    print("TOTAL train q", man["total_questions"], "noul no/yes", yn_all, "probe q", pm["total_questions"])


# =================================================================== vision_v3 (decontam_v3.py)
DOC_V3 = """vision_v3: decontaminate the generated train + probe cases, re-balance yes/no, freeze the probes.

A case is dropped when any of its images is an exact (sha256) or near (DCT phash Hamming <= 6)
match of a release-demo image (--demo, optional) or of any eval_vision_v1 candidate image
(<eval-build>/cand_hashes.jsonl). Probe cases are also dropped on an exact image match with a train
image. After drops, noul is re-balanced exactly per file.
"""
PH_V3 = 6
FAMILIES = {
    "ui_state": ("A: stacked form, light; B: labels-left form with dark header bar", "C: narrow mobile card form, dark/light, '(required)' text, placeholders, pill buttons, Cantarell/URW fonts"),
    "change_detect": ("A: website (header, side menu, hero, button, footer); B: mobile app screen (search, profile card, list, FAB, tab bar)", "C: dark analytics dashboard (chips, KPI cards, chart panel, export button)"),
    "line_rule": ("A: thermal receipt, monospace; B: expense report table with category column", "C: e-commerce order confirmation cards with per-unit prices"),
    "grid_games": ("A: plain board, positional cell names; B: board with A-C/1-3 coordinates", "C: dark tiled board with text glyphs, 'row r, column c' names"),
    "severity": ("A: modal dialog with icon; B: notification stack (most severe of 1-3)", "C: terminal log panel with [INFO]/[WARN]/[ERROR]/[FATAL] tags"),
    "receipt_math": ("A: grocery receipt, qty @ unit lines, $; B: invoice table, $", "C: restaurant bill cards in EUR with German labels (je, Rechnung)"),
}


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def run_v3(a):
    import vision_common as C
    from build_vision_v3 import GENS, balance_noul
    from PIL import Image
    root = Path(a.root); praw = root / "probes_raw"; pout = root / "probes"
    ref = []   # (kind, sha, phash int)
    if a.demo:
        for p in sorted(Path(a.demo).glob("*.png")):
            im = Image.open(p).convert("RGB")
            ref.append(("demo:" + p.name, hashlib.sha256(p.read_bytes()).hexdigest(), C.phash(im)))
    for l in open(Path(a.eval_build) / "cand_hashes.jsonl"):
        h = json.loads(l)
        ref.append(("eval:" + h["path"], h["sha256"], int(h["phash"], 16)))
    ref_sha = {s: k for k, s, _ in ref}
    ref_ph = [(k, p) for k, _, p in ref]

    def load_h(p):
        return {h["path"]: h for h in map(json.loads, open(p))} if p.exists() else {}

    th, ph = load_h(root / "gen_hashes.jsonl"), load_h(praw / "gen_hashes.jsonl")
    train_sha = {h["sha256"] for h in th.values()}
    demo_desc = a.demo if a.demo else "none given"
    dropped, man = [], {"version": "vision_v3", "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "built_by": "scripts/build_vision_v3.py, scripts/decontam_vision.py v3",
                        "labels": "computed by the generators' own code (solver / arithmetic / rule check); no model labels",
                        "decontamination": {"rule": f"image sha256 == or DCT-phash Hamming <= {PH_V3} to any release-demo image "
                                                    f"({demo_desc}) or eval_vision_v1 candidate image; probe image sha == a train image",
                                            "demo_images": sum(1 for k, _, _ in ref if k.startswith("demo:")),
                                            "eval_images": sum(1 for k, _, _ in ref if k.startswith("eval:"))},
                        "generators": {}}
    pman = {"version": "vision_v3_probes", "built_at": man["built_at"],
            "status": "FROZEN held-out probes: template family C only (training uses A/B); report-only",
            "benchmarks": {}}
    pout.mkdir(exist_ok=True)

    def why(c, hs, probe):
        for p in c["images"]:
            h = hs[p]
            if h["sha256"] in ref_sha:
                return "exact:" + ref_sha[h["sha256"]]
            v = int(h["phash"], 16)
            for k, q in ref_ph:
                if hamming(v, q) <= PH_V3:
                    return "phash:" + k
            if probe and h["sha256"] in train_sha:
                return "probe_vs_train_exact"
        return None

    def stats(cases):
        modes = collections.Counter(q["mode"] for c in cases for q in c["questions"])
        yn = [0, 0]; hist = collections.Counter(); gpos = collections.Counter()
        for c in cases:
            for q in c["questions"]:
                g = c["gold"][q["key"]]
                if q["mode"] == "noul":
                    yn[int(g[1] == 1.0)] += 1
                elif q["mode"] == "score":
                    hist[str(g.index(1.0))] += 1
                else:
                    gpos[str(g.index(1.0))] += 1
        return {"cases": len(cases), "questions": sum(modes.values()), "modes": dict(modes), "noul_no_yes": yn,
                "score_hist": dict(sorted(hist.items())), "choice_gold_position": dict(sorted(gpos.items()))}

    for g in GENS:
        for probe in (False, True):
            src = (praw / f"probe_{g}.jsonl") if probe else (root / f"vis3_{g}.jsonl")
            if not src.exists():
                continue
            cases = [json.loads(l) for l in open(src)]
            keep = []
            for c in cases:
                r = why(c, ph if probe else th, probe)
                if r:
                    dropped.append({"case_id": c["case_id"], "probe": probe, "reason": r})
                else:
                    keep.append(c)
            keep, yn = balance_noul(keep)
            if probe:
                for c in keep:
                    for p in c["images"]:
                        dst = pout / p; dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(praw / p, dst)
                fn = pout / f"probe_{g}.jsonl"
                if fn.exists():
                    os.chmod(fn, 0o644)
                sha = C.write_jsonl(fn, keep)
                os.chmod(fn, 0o444)
                pman["benchmarks"][f"probe_{g}"] = {"file": fn.name, "sha256": sha, **stats(keep),
                                                    "candidates": len(cases), "dropped_decontam": len(cases) - len(keep),
                                                    "template_family": FAMILIES[g][1]}
            else:
                sha = C.write_jsonl(src, keep)
                man["generators"][g] = {"file": src.name, "sha256": sha, "source": f"vis3_{g}", **stats(keep),
                                        "candidates": len(cases), "dropped_decontam": len(cases) - len(keep),
                                        "train_template_families": FAMILIES[g][0], "probe_template_family": FAMILIES[g][1]}
            print(("probe " if probe else "train ") + g, stats(keep), "dropped", len(cases) - len(keep), flush=True)
    tot = [0, 0]
    for v in man["generators"].values():
        tot[0] += v["noul_no_yes"][0]; tot[1] += v["noul_no_yes"][1]
    man["total_questions"] = sum(v["questions"] for v in man["generators"].values())
    man["total_cases"] = sum(v["cases"] for v in man["generators"].values())
    man["noul_no_yes_total"] = tot
    man["probe_questions"] = sum(v["questions"] for v in pman["benchmarks"].values())
    with open(root / "decontam_dropped.jsonl", "w") as fh:
        for d_ in dropped:
            fh.write(json.dumps(d_) + "\n")
    (root / "manifest.json").write_text(json.dumps(man, indent=1))
    pm = pout / "manifest.json"
    if pm.exists():
        os.chmod(pm, 0o644)
    pm.write_text(json.dumps(pman, indent=1)); os.chmod(pm, 0o444)
    print("TOTAL train q", man["total_questions"], "noul no/yes", tot, "probe q", man["probe_questions"],
          "dropped", len(dropped), collections.Counter(d_["reason"].split(":")[0] for d_ in dropped))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("eval", help="freeze eval_vision_v1 against vision_v1")
    e.add_argument("--build", required=True, help="eval_vision_v1 build dir (build_eval_vision.py --build)")
    e.add_argument("--train", required=True, help="vision_v1 root")
    e.add_argument("--out", required=True, help="the frozen eval_vision_v1 dir")
    e.add_argument("--suite", default=None, help="optional text-suite dir (*.jsonl) whose exact states are dropped")
    v2 = sub.add_parser("v2", help="decontaminate vision_v2 and freeze its probes")
    v2.add_argument("--root", required=True)
    v2.add_argument("--eval-build", required=True)
    v2.add_argument("--eval", required=True)
    v2.add_argument("--v1", required=True)
    v3 = sub.add_parser("v3", help="decontaminate vision_v3 and freeze its probes")
    v3.add_argument("--root", required=True)
    v3.add_argument("--eval-build", required=True)
    v3.add_argument("--demo", default=None, help="optional dir of release-demo PNGs to decontaminate against")
    a = ap.parse_args(argv)
    {"eval": run_eval, "v2": run_v2, "v3": run_v3}[a.stage](a)


if __name__ == "__main__":
    main()
