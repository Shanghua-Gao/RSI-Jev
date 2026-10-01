"""Shared helpers of the image-corpus builders (image IO, hashing, case construction, recasts).

Used by build_vision_v1.py, build_vision_v2.py, build_vision_v3.py, build_eval_vision.py and
decontam_vision.py. A vision case = one rsijev contract JSONL row + "images": [paths relative to
the corpus root]. The state holds the literal marker "<image>" once per image, in order.

Hugging Face revisions are pinned (REVISIONS below) to the commits the v4.0-VL build read.
`--hf-revision REPO=SHA` (repeatable) overrides one; `--unpinned` resolves the current HEAD of
every repo instead, which is what the original build did (every pinned sha below was the HEAD on
the build date, 2026-09-27/28; none of these repos has been committed to since 2024-12).
"""
from __future__ import annotations

import hashlib, io, json, random, re, sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rsijev.contract import Case, Question  # noqa: E402  (validates every case)

MAX_SIDE = 1600
NOUL = {"false": "No.", "true": "Yes."}
LETTERS = "ABCDEFGHIJKLMNOP"
IMG = "<image>"

# ---------------------------------------------------------------- HF access
REVISIONS = {
    # training sources
    "HuggingFaceM4/the_cauldron": "847a98a779b1652d65111daf20c972dfcd333605",
    "HuggingFaceM4/A-OKVQA": "d1b0efa3a436e9101dfbde3752db7607da696c35",
    "dvgodoy/rvl_cdip_mini": "74de67da4e027265262a22d2550fd27072156d44",
    "detection-datasets/coco": "cf0b22332314a937e9dc8a1957b21725430bb41d",
    # eval-only sources (recorded in eval_vision_v1's fetch.json)
    "lmms-lab/MMBench": "56ba1af8954932c4804bd3f522e05ed96e63b654",
    "xai-org/RealworldQA": "17e7f75e092e47169732462ea3cdfebe911105dd",
    "lmms-lab/POPE": "4db1276663dfa5eb8ad16a52d24c31a09e470896",
    "lmms-lab/HallusionBench": "cd417161857aefb23d878d42cf1bb53aa9dd646f",
    "lmms-lab/DocVQA": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
}
_REV: dict[str, str] = {}
_API = None
_PINNED = True


def add_hf_args(ap):
    ap.add_argument("--hf-revision", action="append", default=[], metavar="REPO=SHA",
                    help="override the pinned revision of one HF dataset repo (repeatable)")
    ap.add_argument("--unpinned", action="store_true",
                    help="resolve the current HEAD of every repo instead of the pinned revisions")


def apply_hf_args(a):
    global _PINNED
    _PINNED = not a.unpinned
    for kv in a.hf_revision:
        repo, _, sha = kv.partition("=")
        if not sha:
            raise SystemExit(f"--hf-revision wants REPO=SHA, got {kv!r}")
        REVISIONS[repo] = sha


def api():
    global _API
    if _API is None:
        from huggingface_hub import HfApi
        _API = HfApi()
    return _API


def rev(repo: str) -> str:
    if repo not in _REV:
        _REV[repo] = REVISIONS[repo] if (_PINNED and repo in REVISIONS) else api().dataset_info(repo).sha
    return _REV[repo]


def download(repo: str, fname: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, repo_type="dataset", revision=rev(repo))


def files(repo: str, prefix: str) -> list[str]:
    return sorted(s.rfilename for s in api().dataset_info(repo, revision=rev(repo)).siblings
                  if s.rfilename.startswith(prefix) and s.rfilename.endswith(".parquet"))


def rows(repo: str, fname: str, limit: int | None = None):
    """Iterate parquet rows (dicts) row group by row group."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(download(repo, fname))
    n = 0
    for g in range(pf.num_row_groups):
        for r in pf.read_row_group(g).to_pylist():
            yield r
            n += 1
            if limit is not None and n >= limit:
                return


# ---------------------------------------------------------------- images
def _dct_mat(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]; i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


_D32 = _dct_mat(32)


def phash(im: Image.Image) -> int:
    a = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = (_D32 @ a @ _D32.T)[:8, :8].flatten()
    med = np.median(d[1:])
    bits = d > med
    return int("".join("1" if b else "0" for b in bits), 2)


def dhash(im: Image.Image) -> int:
    a = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int32)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def decode(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    im.load()
    return im


def save_image(root: Path, src: str, iid: str, raw: bytes | Image.Image, png: bool,
               hashes: list | None = None) -> str:
    """Save (cap longest side 1600) and return the path relative to root. Records hashes."""
    im = raw if isinstance(raw, Image.Image) else decode(raw)
    if im.mode not in ("RGB", "L"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        try:
            bg.paste(im, mask=im.convert("RGBA").split()[-1])
        except Exception:
            bg = im.convert("RGB")
        im = bg
    im = im.convert("RGB")
    w, h = im.size
    s = MAX_SIDE / max(w, h)
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    rel = f"images/{src}/{re.sub(r'[^A-Za-z0-9_.-]', '_', iid)}.{'png' if png else 'jpg'}"
    out = root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    im.save(buf, format="PNG" if png else "JPEG", **({"optimize": True} if png else {"quality": 90}))
    data = buf.getvalue()
    out.write_bytes(data)
    if hashes is not None:
        hashes.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(),
                       "phash": f"{phash(im):016x}", "dhash": f"{dhash(im):016x}"})
    return rel


def img_bytes(x) -> bytes:
    """HF image struct {bytes,path} -> bytes."""
    if isinstance(x, dict):
        return x["bytes"]
    return x


# ---------------------------------------------------------------- cases
def noul_q(key: str, instr: str, crit=None) -> Question:
    return Question(key, "noul", instr, ("false", "true"), crit or NOUL)


def choice_q(key: str, instr: str, texts: list[str], gold_i: int, rng: random.Random,
             shuffle: bool = True) -> tuple[Question, tuple]:
    """Letter-keyed choice; options shuffled so gold position is uniform."""
    order = list(range(len(texts)))
    if shuffle:
        rng.shuffle(order)
    letters = tuple(LETTERS[: len(texts)])
    crit = {letters[j]: texts[i] for j, i in enumerate(order)}
    g = order.index(gold_i)
    return Question(key, "choice", instr, letters, crit), tuple(1.0 if j == g else 0.0 for j in range(len(texts)))


def one_hot(n: int, i: int) -> tuple:
    return tuple(1.0 if k == i else 0.0 for k in range(n))


def make_case(cid: str, src: str, state: str, images: list[str], qs: list) -> dict:
    """qs: list of (Question, gold tuple). Validates via rsijev.contract.Case."""
    assert state.count(IMG) == len(images), (cid, state, images)
    c = Case(cid, src, state, tuple(q for q, _ in qs), {q.key: g for q, g in qs})
    return {"case_id": c.case_id, "source": c.source, "state": c.state,
            "questions": [{"key": q.key, "mode": q.mode, "instructions": q.instructions,
                           "options": list(q.options), "criteria": q.criteria} for q in c.questions],
            "gold": {k: list(v) for k, v in c.gold.items()}, "images": images}


def write_jsonl(path: Path, recs: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- free-form recast
_num_re = re.compile(r"^-?[\d,]*\.?\d+%?$")


def clean_ans(a: str) -> str:
    a = a.strip()
    if a.endswith(".") and not _num_re.match(a):
        a = a[:-1].strip()
    elif a.endswith(".") and a[:-1] and _num_re.match(a[:-1]):
        a = a[:-1]
    return a


def as_num(a: str):
    t = a.replace(",", "").rstrip("%").strip()
    try:
        return float(t)
    except ValueError:
        return None


def is_yn(a: str):
    t = a.strip().lower().rstrip(".")
    return True if t == "yes" else False if t == "no" else None


def qtype(q: str, n: int = 3) -> str:
    return " ".join(re.findall(r"[a-z]+", q.lower())[:n])


class AnswerPool:
    """Real answers of one source, for distractors: numbers by magnitude, text by question type."""

    def __init__(self):
        self.nums: list[str] = []
        self.by_type: dict[str, list[str]] = {}
        self.text: list[str] = []

    def add(self, q: str, a: str):
        if is_yn(a) is not None:
            return
        if as_num(a) is not None:
            self.nums.append(a)
        else:
            self.text.append(a)
            for n in (3, 2):
                self.by_type.setdefault(f"{n}:{qtype(q, n)}", []).append(a)

    def distractors(self, q: str, gold: str, k: int, rng: random.Random) -> list[str] | None:
        seen = {gold.lower()}
        out: list[str] = []

        def take(cands, tries=200):
            for _ in range(tries):
                if len(out) >= k or not cands:
                    return
                c = rng.choice(cands)
                if c.lower() not in seen and len(c) <= 120:
                    seen.add(c.lower()); out.append(c)

        g = as_num(gold)
        if g is not None:
            near = [x for x in self.nums if (v := as_num(x)) is not None and v != g and
                    (abs(v - g) <= max(1.0, 2.0 * abs(g)))]
            same_fmt = [x for x in near if ("%" in x) == ("%" in gold) and ("." in x) == ("." in gold)]
            take(same_fmt or near)
            # fallback: perturb the gold (keeps the same format)
            step = [1, 2, 3, -1, -2, 5, 10, -5]
            for d in step:
                if len(out) >= k:
                    break
                if "." in gold.replace("%", ""):
                    dec = len(gold.replace("%", "").split(".")[1])
                    v = round(g + d * (10 ** -dec) * (1 if abs(g) < 10 else 10), dec)
                    s = f"{v:.{dec}f}"
                else:
                    v = int(g) + d * (1 if abs(g) < 20 else max(1, int(abs(g) * 0.1)))
                    s = f"{v:,}" if "," in gold else str(v)
                if "%" in gold:
                    s += "%"
                if s.lower() not in seen:
                    seen.add(s.lower()); out.append(s)
        else:
            # prefer same-shape answers (digits or not, similar length) of the same question type
            hd = any(ch.isdigit() for ch in gold); L = max(1, len(gold))

            def shaped(xs):
                return [x for x in xs if any(ch.isdigit() for ch in x) == hd and 0.5 <= len(x) / L <= 2.0]

            for n in (3, 2):
                pool = self.by_type.get(f"{n}:{qtype(q, n)}", [])
                if len(set(pool)) > k + 2:
                    take(shaped(pool))
                    take(pool)
                    break
            take(shaped(self.text))
            take(self.text)
        return out if len(out) >= k else None


def recast_freeform(key: str, q: str, gold: str, pool: AnswerPool, rng: random.Random,
                    want_noul: bool, n_opts: int = 4):
    """-> (Question, gold) or None. Yes/no answers become a direct noul question."""
    yn = is_yn(gold)
    if yn is not None:
        return noul_q(key, q), one_hot(2, 1 if yn else 0)
    ds = pool.distractors(q, gold, n_opts - 1, rng)
    if ds is None:
        return None
    if want_noul:
        truth = rng.random() < 0.5
        x = gold if truth else ds[0]
        return noul_q(key, f'Question: {q}\nIs "{x}" the correct answer?'), one_hot(2, 1 if truth else 0)
    return choice_q(key, q, [gold] + ds, 0, rng)


def strip_prompt_suffix(u: str) -> str:
    """Cauldron adds a style hint on the last line ('Short answer required.' etc)."""
    lines = u.strip().split("\n")
    if len(lines) > 1:
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_letter_mc(u: str):
    """Cauldron 'Question: ...\\nChoices:\\nA. x\\nB. y\\nAnswer with the letter.' ->
    (context, question, [choices])."""
    m = re.search(r"Question:(.*?)\nChoices:\n(.*?)\nAnswer with the letter", u, re.S)
    if not m:
        return None
    ctx = u[: m.start()].strip()
    qs = m.group(1).strip()
    ch = []
    for line in m.group(2).split("\n"):
        mm = re.match(r"^([A-Z])\.\s?(.*)$", line.strip())
        if not mm:
            return None
        ch.append(mm.group(2).strip())
    return ctx, qs, ch


def letter_answer(a: str):
    m = re.search(r"Answer:\s*([A-Z])\b", a)
    return None if not m else ord(m.group(1)) - 65
