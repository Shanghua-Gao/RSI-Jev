"""Rebuild Laya Vision's 34 validation sets locally, then draw a fixed stratified sample.

Used by laya.py (`laya.py prep`); not run on its own. Each builder replays the
corresponding prep job in laya-vision's modal_app.py (prepare_cauldron_subset,
prepare_score_dataset, prepare_eval_dataset) with the same seeds, the same laya record
functions and the same RNG call order, so the validation records (ids, question text,
options, labels) are the ones Laya scored. The three official VQA sets take their ids and
labels from Laya's committed per-question rows (results/raw) and fetch those rows from the
Hub. Only the images of sampled records are written (same resize and quality as the prep
job).

Writes <out>/<name>/{val_full.jsonl, sample.jsonl, images/, check.json}. check.json compares
the rebuilt validation count with the one Laya's own full evaluation recorded
(eval-results/autoresearch-full-long-sep24-b64.json); `match: false` means a source changed.

Revisions: the_cauldron is read at 847a98a7 and the four rubric-scored sets at the
revisions the published run used (SCORE_REVISIONS). The other sources are read as
laya-vision's own prep reads them, at their current revision; check.json is the guard.
"""
import collections, gzip, io, json, os, random, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

LV = None           # the laya-vision checkout; set by setup()
OUT = None
FULL_EVAL = None
CAULDRON_REV = '847a98a779b1652d65111daf20c972dfcd333605'
RAW_NAME = 'results/raw/smolvlm-autoresearch-full-long-sep24-b64-best.vqa-val.predictions.jsonl.gz'
SCORE_REVISIONS = {   # HfApi().dataset_info(...).sha when the published samples were built
    'ava': 'ee09ff2b4151b87c61cfb5165bb46c0ebe39b679',
    'crisismmd': '10a5626ba112ab9c50369c473a2062bcf205d708',
    'richhf': '722f9e0117b58c23f93fe4b0b14df488f742db2f',
    'vlfeedback': '137dcea93e7667decc117c54b60f7401c6903310',
}

CAULDRON = ("ai2d", "aokvqa", "iconqa", "intergps", "scienceqa", "tqa", "visual7w", "raven", "figureqa",
            "hateful_memes", "nlvr2", "vsr", "vqarad", "clevr", "dvqa", "mapqa", "ocrvqa", "vqav2", "chartqa")
ALL = (["aokvqa", "scienceqa", "vqav2_yesno"] + ["cauldron_" + s for s in CAULDRON]
       + ["score_" + s for s in ("vlfeedback", "ava", "richhf", "crisismmd")]
       + ["eval_" + s for s in ("koniq", "evalmuse", "cifar10h", "ferplus", "vizwiz", "pope_random", "pope_popular",
                                "pope_adversarial")])


def setup(out, lv=None):
    """Point the builders at a laya-vision checkout (the pinned one by default) and an output dir."""
    global LV, OUT, FULL_EVAL
    LV = str(lv or C.upstream('laya-vision'))
    if LV not in sys.path:
        sys.path.insert(0, LV)
    OUT = str(out)
    FULL_EVAL = json.load(open(os.path.join(LV, 'eval-results/autoresearch-full-long-sep24-b64.json')))['datasets']


def pil_from(x):
    from PIL import Image
    if isinstance(x, dict):
        return Image.open(io.BytesIO(x["bytes"])) if x.get("bytes") else Image.open(x["path"])
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x))
    return x


# ------------------------------------------------------------------------------------------------ cauldron
def build_cauldron(subset, max_rows=10000, max_texts=4, val_pct=5.0, seed=0):
    """prepare_cauldron_subset, val side only. Returns [(record, [image objects])]."""
    from datasets import load_dataset
    from laya.cauldron import cauldron_records
    rng = random.Random(seed)
    ds = load_dataset("HuggingFaceM4/the_cauldron", subset, split="train", streaming=True, revision=CAULDRON_REV).decode(False)
    n_rows = n_seen = 0
    counts = {"train": collections.Counter(), "val": collections.Counter()}
    out = []
    for i, row in enumerate(ds):
        if max_rows and n_rows >= max_rows:
            break
        n_seen += 1
        n_img = len(row["images"]) if isinstance(row["images"], list) else 1
        paths = ["images/%s-%d-%d.jpg" % (subset, i, j) for j in range(n_img)]
        recs = cauldron_records(row["texts"], paths, "%s-%d" % (subset, i), max_texts=max_texts, rng=rng)
        if not recs:
            continue
        split = "val" if rng.random() < val_pct / 100 else "train"
        for rec in recs:
            counts[split][rec["question"]["type"]] += 1
        if split == "val":
            images = row["images"] if isinstance(row["images"], list) else [row["images"]]
            for rec in recs:
                out.append((rec, list(zip(paths, images))))
        n_rows += 1
    meta = {"rows": n_rows, "rows_seen": n_seen, "records": {k: dict(v) for k, v in counts.items()}}
    return out, meta


def save_cauldron_images(base, pairs, max_side=1024):
    for path, im in pairs:
        p = os.path.join(base, path)
        if os.path.exists(p):
            continue
        im = pil_from(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        im.save(p, quality=90)


# ------------------------------------------------------------------------------------------------ eval sets
def save_eval_image(image, base, key, max_side=1024):
    """modal_app._save_eval_image."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    if isinstance(image, tuple):
        rel = "images/%s%s" % (safe, image[1])
        with open(os.path.join(base, rel), "wb") as f:
            f.write(image[0])
        return rel
    im = pil_from(image).convert("RGB")
    if max(im.size) <= 64:
        rel = "images/%s.png" % safe
        im.save(os.path.join(base, rel))
    else:
        rel = "images/%s.jpg" % safe
        im.thumbnail((max_side, max_side))
        im.save(os.path.join(base, rel), quality=92)
    return rel


def eval_source(name, rng, max_rows=10000, val_pct=10.0, seed=0):
    """modal_app._eval_source with lazy images: yields (split, key, image_or_thunk, records)."""
    import csv
    import requests
    from datasets import load_dataset
    from laya import evalsets as E
    repo = E.SOURCES[name]
    if name == "koniq":
        yield from koniq_source(rng)
    elif name == "evalmuse":
        yield from evalmuse_source(rng, max_rows, val_pct, seed)
    elif name == "cifar10h":
        for i, row in enumerate(load_dataset(repo, split="train", streaming=True).decode(False)):
            rec = E.cifar10h_record(row, i, rng)
            if rec:
                yield "val", rec["id"], row["image"], [rec]
    elif name == "ferplus":
        text = requests.get(E.FERPLUS_VOTES_URL, timeout=120).text
        votes = list(csv.DictReader(io.StringIO(text)))
        for usage, hf_split in (("Training", "train"), ("PublicTest", "valid"), ("PrivateTest", "test")):
            ds = load_dataset(repo, split=hf_split)
            sub = [(i, v) for i, v in enumerate(votes) if v["Usage"] == usage]
            if len(sub) != len(ds):
                raise RuntimeError("FER+ %s: %d vote rows but %d images" % (usage, len(sub), len(ds)))
            labels = ds["label"]
            agree = E.ferplus_agreement((v, labels[k]) for k, (_, v) in enumerate(sub))
            if agree < 0.45:
                raise RuntimeError("FER+ join off (%.2f)" % agree)
            for k, (i, v) in enumerate(sub):
                got = E.ferplus_record(v, i, rng)
                if got:
                    yield got[0], got[1]["id"], (lambda ds=ds, k=k: ds[k]["image"]), [got[1]]
    elif name == "vizwiz":
        for row in load_dataset(repo, split="val", streaming=True).decode(False):
            rec = E.vizwiz_record(row, rng)
            if rec:
                yield "val", rec["id"], row["image"], [rec]
    elif name.startswith("pope_"):
        for row in load_dataset(repo, "Full", split=name.split("_", 1)[1], streaming=True).decode(False):
            rec = E.pope_record(row)
            if rec:
                yield "val", "pope-" + str(row["image_source"]), row["image"], [rec]
    else:
        raise ValueError(name)


def koniq_source(rng):
    """modal_app._koniq_source; images kept as bytes (verbatim)."""
    import csv, tarfile, requests
    from huggingface_hub import hf_hub_url
    from laya.evalsets import SOURCES, koniq_record
    url = hf_hub_url(SOURCES["koniq"], "koniq10k.tgz", repo_type="dataset")
    rows = None
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for m in tar:
                if m.name.endswith("koniq10k_distributions_sets.csv"):
                    rows = {row["image_name"]: row for row in csv.DictReader(io.StringIO(tar.extractfile(m).read().decode("utf-8")))}
                    left = set(rows)
                elif "/512x384/" in m.name and m.name.endswith(".jpg"):
                    fn = m.name.rsplit("/", 1)[1]
                    got = koniq_record(rows[fn], rng) if fn in rows else None
                    if got is None:
                        continue
                    split, rec = got
                    yield split, rec["id"], (tar.extractfile(m).read(), ".jpg"), [rec]
                    left.discard(fn)
                    if not left:
                        return


def evalmuse_source(rng, max_rows, val_pct, seed):
    """modal_app._evalmuse_source, val only, images fetched lazily by range requests."""
    import requests
    from huggingface_hub import hf_hub_download, hf_hub_url
    from PIL import Image
    from laya.evalsets import SOURCES, MultiPartZip, evalmuse_record, stable_split
    repo = SOURCES["evalmuse"]
    with open(hf_hub_download(repo, "train_list.json", repo_type="dataset")) as f:
        rows = json.load(f)
    todo = {"train": [], "val": []}
    for row in rows:
        rec = evalmuse_record(row, rng)
        if rec is not None:
            todo[stable_split(str(row["prompt_id"]), val_pct, seed)].append((row["img_path"], rec))
    parts = ["images.zip.part-a%s" % c for c in "abcdef"]
    urls = [hf_hub_url(repo, p, repo_type="dataset") for p in parts]
    sess = requests.Session()
    sizes = [int(sess.head(u, allow_redirects=True, timeout=60).headers["Content-Length"]) for u in urls]

    def fetch(i, start, end):
        for attempt in range(5):
            try:
                r = sess.get(urls[i], headers={"Range": "bytes=%d-%d" % (start, end - 1)}, timeout=120)
                r.raise_for_status()
                if len(r.content) == end - start:
                    return r.content
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        raise RuntimeError("range read failed")

    archive = MultiPartZip(sizes, fetch)
    zf = archive.open()
    names = {n.split("dataset/images/", 1)[-1]: n for n in zf.namelist()}
    for img_path, rec in todo["val"]:
        yield "val", rec["id"], (lambda p=img_path: Image.open(io.BytesIO(zf.read(names[p])))), [rec]


def build_eval(name):
    rng = random.Random(0)
    out = []
    for split, key, image, rows in eval_source(name, rng):
        if split != "val":
            continue
        for rec in rows:
            out.append((rec, key, image))
    return out, {"records": {"val": len(out)}}


# ------------------------------------------------------------------------------------------------ score sets
def score_source(name, split, rng, max_texts, max_chars, revision=None):
    """modal_app._score_source with images left undecoded."""
    from datasets import load_dataset
    from laya.rubric import SOURCES, ava_record, crisismmd_record, richhf_records, vlfeedback_records
    repo = SOURCES[name]
    if name == "vlfeedback":
        if split == "val":
            return
        for i, row in enumerate(load_dataset(repo, split="train", streaming=True, revision=revision)):
            rid = "vlf-%s" % (row.get("id") or i)
            yield rid, row["image"], vlfeedback_records(row, rid, rng, max_texts=max_texts, max_chars=max_chars)
    elif name == "ava":
        ds = load_dataset(repo, split="validation" if split == "val" else "train", streaming=True, revision=revision)
        for i, row in enumerate(ds.decode(False)):
            rid = "ava-%s" % (row.get("image_id") or i)
            rec = ava_record(row, rid, rng)
            yield rid, row.get("image"), [rec] if rec else []
    elif name == "richhf":
        ds = load_dataset(repo, split="validation" if split == "val" else "train", streaming=True, revision=revision)
        for i, row in enumerate(ds.decode(False)):
            rid = "richhf-%d" % i
            yield rid, row.get("image"), richhf_records(row, rid, rng, max_texts=max_texts)
    elif name == "crisismmd":
        for i, row in enumerate(load_dataset(repo, "damage", split="dev" if split == "val" else "train", revision=revision)):
            rid = "crisis-%s" % (row.get("image_id") or i)
            rec = crisismmd_record(row, rid, rng)
            yield rid, row.get("image") or row["image_path"], [rec] if rec else []


def build_score(name, max_rows=0, max_texts=2, val_pct=5.0, max_val=1000, max_chars=1200, seed=0):
    rng = random.Random(seed)
    rev = SCORE_REVISIONS[name]
    out = []
    n_rows = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for rid, image, rows in score_source(name, split, rng, max_texts, max_chars, rev):
            if not rows:
                continue
            if split == "train" and max_rows and n_rows["train"] >= max_rows:
                break
            if split == "val" and max_val and n_rows["val"] >= max_val:
                break
            target = split
            if split == "train" and name in ("vlfeedback",):
                target = "val" if rng.random() < val_pct / 100 else "train"
                if target == "val" and max_val and n_rows["val"] >= max_val:
                    target = "train"
            if target == "val":
                for rec in rows:
                    out.append((rec, rid, image))
            n_rows[target] += 1
            if name == "vlfeedback" and n_rows["val"] >= max_val:
                break  # later rows can only go to train; nothing more for val
    return out, {"rows": n_rows, "records": {"val": len(out)}, "revision": rev}


def save_score_image(base, rid, image, name, max_side=1024):
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from laya.rubric import SOURCES
    path = "images/%s.jpg" % rid
    p = os.path.join(base, path)
    if not os.path.exists(p):
        if isinstance(image, str):
            image = Image.open(hf_hub_download(SOURCES[name], image, repo_type="dataset"))
        im = pil_from(image).convert("RGB")
        im.thumbnail((max_side, max_side))
        im.save(p, quality=90)
    return path


# ------------------------------------------------------------------------------------------------ official VQA sets
def committed_rows():
    return [json.loads(l) for l in gzip.open(os.path.join(LV, RAW_NAME), "rt")]


def build_vqa(name, sample_ids):
    """Fetch the sampled committed rows from the Hub: [(record, PIL image)]."""
    from datasets import load_dataset
    want = set(sample_ids)
    found = {}
    if name == "aokvqa":
        for row in load_dataset("HuggingFaceM4/A-OKVQA", split="validation"):
            if row["question_id"] in want:
                found[row["question_id"]] = ({"id": row["question_id"], "state_text": None,
                                              "question": {"type": "choice", "instructions": row["question"], "criteria": list(row["choices"])},
                                              "label": int(row["correct_choice_idx"])}, row["image"])
    elif name == "scienceqa":
        ds = load_dataset("derek-thomas/ScienceQA", split="validation")
        for sid in want:
            row = ds[int(sid.split("-")[1])]
            found[sid] = ({"id": sid, "state_text": row["hint"] or None,
                           "question": {"type": "choice", "instructions": row["question"], "criteria": list(row["choices"])},
                           "label": int(row["answer"])}, row["image"])
    elif name == "vqav2_yesno":
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        files = sorted(f for f in fs.ls("datasets/lmms-lab-encoder/VQAv2/data", detail=False) if "/validation-" in f)
        wanted_int = {int(x): x for x in want}
        for f in files:
            pf = pq.ParquetFile(fs.open(f))
            for rg in range(pf.num_row_groups):
                qids = pf.read_row_group(rg, columns=["question_id"]).column(0).to_pylist()
                hit = [k for k, q in enumerate(qids) if int(q) in wanted_int]
                if not hit:
                    continue
                t = pf.read_row_group(rg).to_pylist()
                for k in hit:
                    row = t[k]
                    sid = wanted_int[int(row["question_id"])]
                    found[sid] = ({"id": sid, "state_text": None,
                                   "question": {"type": "noul", "instructions": row["question"], "criteria": None},
                                   "label": None, "answer_type": row.get("answer_type")}, row["image"])
            if len(found) == len(want):
                break
    return found


# ------------------------------------------------------------------------------------------------ main
def sample_idx(name, n, per_set, seed):
    k = min(per_set, n)
    return sorted(random.Random("laya-sample:%d:%s" % (seed, name)).sample(range(n), k))


def expected(name):
    v = FULL_EVAL["val_calibrated"].get(name)
    return v["n"] if v else None


def build(names=ALL, per_set=200, seed=0):
    for name in names:
        base = os.path.join(OUT, name)
        if os.path.exists(os.path.join(base, "check.json")):
            print("skip", name, flush=True)
            continue
        os.makedirs(os.path.join(base, "images"), exist_ok=True)
        t0 = time.time()
        if name in ("aokvqa", "scienceqa", "vqav2_yesno"):
            rows = [r for r in committed_rows() if r["dataset"] == name]
            idx = sample_idx(name, len(rows), per_set, seed)
            samp = [rows[i] for i in idx]
            found = build_vqa(name, [r["id"] for r in samp])
            recs = []
            for r in samp:
                rec, img = found[r["id"]]
                rec["label"] = r["label"]  # Laya's label (VQAv2: majority of yes/no votes in its prep)
                if img is not None:
                    path = "images/%s.jpg" % "".join(c if c.isalnum() or c in "-_." else "_" for c in r["id"])
                    im = pil_from(img).convert("RGB"); im.thumbnail((512, 512)); im.save(os.path.join(base, path), quality=90)
                    rec["image"] = path
                rec["index"] = r["index"]
                recs.append(rec)
            full_n, meta = len(rows), {"records": {"val": len(rows)}, "source": "committed rows (ids, labels)"}
        elif name.startswith("cauldron_"):
            pairs, meta = build_cauldron(name[len("cauldron_"):])
            full_n = len(pairs)
            with open(os.path.join(base, "val_full.jsonl"), "w") as f:
                for rec, _ in pairs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            idx = sample_idx(name, full_n, per_set, seed)
            recs = []
            for i in idx:
                rec, imgs = pairs[i]
                save_cauldron_images(base, imgs)
                rec["index"] = i
                recs.append(rec)
        elif name.startswith("eval_"):
            triples, meta = build_eval(name[len("eval_"):])
            full_n = len(triples)
            with open(os.path.join(base, "val_full.jsonl"), "w") as f:
                for rec, _, _ in triples:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            idx = sample_idx(name, full_n, per_set, seed)
            recs = []
            for i in idx:
                rec, key, image = triples[i]
                if callable(image):
                    image = image()
                rec["image"] = save_eval_image(image, base, key)
                rec["index"] = i
                recs.append(rec)
        elif name.startswith("score_"):
            src = name[len("score_"):]
            triples, meta = build_score(src)
            full_n = len(triples)
            with open(os.path.join(base, "val_full.jsonl"), "w") as f:
                for rec, _, _ in triples:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            idx = sample_idx(name, full_n, per_set, seed)
            recs = []
            for i in idx:
                rec, rid, image = triples[i]
                rec["image"] = save_score_image(base, rid, image, src)
                rec["index"] = i
                recs.append(rec)
        with open(os.path.join(base, "sample.jsonl"), "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        exp = expected(name)
        chk = {"name": name, "rebuilt_val_n": full_n, "laya_eval_n": exp, "match": full_n == exp, "sample_n": len(recs),
               "meta": meta, "laya_meta": FULL_EVAL.get("dataset_meta", {}).get(name), "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(chk, open(os.path.join(base, "check.json"), "w"), indent=1, default=str)
        print(name, "rebuilt", full_n, "laya", exp, "MATCH" if full_n == exp else "MISMATCH", "sample", len(recs),
              "%.1f min" % chk["minutes"], flush=True)

