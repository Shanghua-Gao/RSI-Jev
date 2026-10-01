"""The image-corpus builders of v4.0-VL (scripts/vision_common.py, build_vision_v1/v2/v3.py,
build_eval_vision.py, decontam_vision.py). CPU only, tiny synthetic fixtures, no network.

    python -m pytest tests/test_vision_builders.py -q
"""
from __future__ import annotations

import ast
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("numpy")
import numpy as np                                                   # noqa: E402

if not hasattr(np, "bitwise_count"):
    pytest.skip("numpy >= 2.0 needed (np.bitwise_count)", allow_module_level=True)

from PIL import Image, ImageDraw                                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import vision_common as C                                            # noqa: E402
from rsijev.contract import Case, Question                           # noqa: E402

FILES = ["vision_common.py", "build_vision_v1.py", "build_vision_v2.py", "build_vision_v3.py",
         "build_eval_vision.py", "decontam_vision.py"]


# ------------------------------------------------------------------ hygiene
@pytest.mark.parametrize("name", FILES)
def test_no_absolute_path_literal(name):
    tree = ast.parse((ROOT / "scripts" / name).read_text())
    import re
    absolute = re.compile(r"^/(?:[\w.-]+/){1,}")
    bad = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str) and absolute.match(n.value)]
    assert not bad, bad


@pytest.mark.parametrize("name", FILES[1:])
def test_cli_starts(name):
    if name in ("build_vision_v1.py", "build_vision_v2.py"):
        pytest.importorskip("pyarrow")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "usage:" in r.stdout


def test_revisions_are_pinned():
    assert C.rev("HuggingFaceM4/the_cauldron").startswith("847a98a7")
    assert all(len(v) == 40 for v in C.REVISIONS.values())


# ------------------------------------------------------------------ recasts
def _check(qg):
    q, g = qg
    Case("x", "s", "<image>", (q,), {q.key: g})     # validates
    return q, g


def test_clean_ans_and_parsers():
    assert C.clean_ans(" Paris. ") == "Paris"
    assert C.clean_ans("3.5.") == "3.5"
    assert C.as_num("1,200") == 1200.0 and C.as_num("45%") == 45.0 and C.as_num("abc") is None
    assert C.is_yn("Yes.") is True and C.is_yn("no") is False and C.is_yn("maybe") is None
    u = "Some context\nQuestion: Which is red?\nChoices:\nA. apple\nB. sky\nAnswer with the letter."
    assert C.parse_letter_mc(u) == ("Some context", "Which is red?", ["apple", "sky"])
    assert C.letter_answer("Answer: B") == 1
    assert C.strip_prompt_suffix("What is it?\nShort answer required.") == "What is it?"


def _pool():
    p = C.AnswerPool()
    for i, a in enumerate(["red", "blue", "green", "yellow", "purple", "orange", "black", "white"]):
        p.add("What color is the car?", a)
    for n in ["3", "4", "7", "12", "15"]:
        p.add("How many dogs?", n)
    return p


def test_freeform_to_four_way_choice():
    rng = random.Random(0)
    q, g = _check(C.recast_freeform("q0", "What color is the car?", "red", _pool(), rng, want_noul=False))
    assert q.mode == "choice" and q.options == ("A", "B", "C", "D")
    assert sum(g) == 1.0 and q.criteria[q.options[g.index(1.0)]] == "red"
    assert len(set(q.criteria.values())) == 4


def test_freeform_numeric_distractors_keep_format():
    q, g = _check(C.recast_freeform("q0", "How many dogs?", "5", _pool(), random.Random(1), want_noul=False))
    assert all(C.as_num(v) is not None for v in q.criteria.values())
    assert q.criteria[q.options[g.index(1.0)]] == "5"


def test_freeform_to_noul_and_yes_no():
    q, g = _check(C.recast_freeform("q0", "Is it raining?", "Yes", _pool(), random.Random(0), want_noul=False))
    assert q.mode == "noul" and g == (0.0, 1.0)
    seen = set()
    for s in range(20):
        q, g = _check(C.recast_freeform("q0", "What color is the car?", "red", _pool(), random.Random(s), want_noul=True))
        assert q.mode == "noul" and 'the correct answer?' in q.instructions
        seen.add(g)
    assert seen == {(0.0, 1.0), (1.0, 0.0)}


def test_choice_gold_position_is_uniformish():
    rng = random.Random(0)
    pos = [C.choice_q("q", "?", ["a", "b", "c", "d"], 0, rng)[1].index(1.0) for _ in range(400)]
    assert all(60 < pos.count(i) < 140 for i in range(4))


def test_yes_no_balancers():
    pytest.importorskip("pyarrow")
    import build_vision_v1 as V1
    import build_vision_v2 as V2
    b = V1.YNBalance(slack=2)
    admitted = []
    for y in [1] * 10 + [0] * 3:
        if b.ok(y):
            b.add(y); admitted.append(y)
    assert b.n == [3, 3]                 # the majority may lead by at most `slack`, then waits
    # trim_noul: exact balance, cases left empty are dropped
    cases = []
    for i in range(6):
        y = 1 if i < 4 else 0
        cases.append(C.make_case(f"c{i}", "s", "<image>", ["images/x.png"], [(C.noul_q("q0", "?"), C.one_hot(2, y))]))
    out = V2.trim_noul(cases)
    ys = [c["gold"]["q0"][1] for c in out]
    assert ys.count(1.0) == ys.count(0.0) == 2
    # probe split takes the LAST cases
    tr, pr = V2.split_probe(list(cases), 2)
    assert [c["case_id"] for c in pr] == ["c4", "c5"] and len(tr) == 4


# ------------------------------------------------------------------ images
def _img(seed, size=(64, 48)):
    r = random.Random(seed)
    im = Image.new("RGB", size, tuple(r.randrange(256) for _ in range(3)))
    d = ImageDraw.Draw(im)
    for _ in range(6):
        x0, y0 = r.randrange(size[0] - 10), r.randrange(size[1] - 10)
        d.rectangle((x0, y0, x0 + r.randrange(5, 30), y0 + r.randrange(5, 20)), fill=tuple(r.randrange(256) for _ in range(3)))
    return im


def test_save_image_caps_and_hashes(tmp_path):
    hashes = []
    rel = C.save_image(tmp_path, "src", "a/b:1", _img(0, (3200, 100)), True, hashes)
    assert rel == "images/src/a_b_1.png"
    im = Image.open(tmp_path / rel)
    assert max(im.size) == 1600
    assert set(hashes[0]) == {"path", "sha256", "phash", "dhash"} and len(hashes[0]["phash"]) == 16


# ------------------------------------------------------------------ vision_v3 generators
@pytest.fixture(scope="module")
def v3():
    import build_vision_v3 as G
    G.set_fonts([])          # PIL default font; labels do not depend on the font files
    return G


@pytest.mark.parametrize("gen", ["ui_state", "change_detect", "line_rule", "grid_games", "severity", "receipt_math"])
def test_generator_builds_valid_cases(v3, gen, tmp_path):
    rng = random.Random(f"test:{gen}")
    out, hashes, cnt = v3.build(gen, ["A", "B"], 12, tmp_path, rng, "t")
    assert out and cnt[0] == cnt[1]                      # yes/no balanced exactly
    for c in out:
        assert c["source"] == f"vis3_{gen}" and c["family"] in ("A", "B")
        assert c["state"].count("<image>") == len(c["images"]) >= 1
        for p in c["images"]:
            assert (tmp_path / p).is_file()
        qs = tuple(Question(q["key"], q["mode"], q["instructions"], tuple(q["options"]), q["criteria"])
                   for q in c["questions"])
        Case(c["case_id"], c["source"], c["state"], qs, {k: tuple(v) for k, v in c["gold"].items()})
    assert {h["path"] for h in hashes} == {p for c in out for p in c["images"]}


def test_grid_game_labels_come_from_the_solver(v3):
    assert v3.winner("XXXOO....") == "X"
    assert v3.winner("XOXOXOOXO") is None
    assert v3.win_moves("XX.OO....", "X") == [2]
    assert v3.win_moves("XX.OO....", "O") == [5]
    rng = random.Random(5)
    for k in range(30):
        ims, state, qs = v3.gen_grid_games(rng, "A", k)
        qs = [q for q in qs if q]
        won = [g for q, g in qs if q.key == "won"][0]
        assert (won == (0.0, 1.0)) == any(q.key == "who" for q, _ in qs)


def test_receipt_math_label(v3):
    rng = random.Random(11)
    for k in range(20):
        ims, state, qs = v3.gen_receipt_math(rng, "B", k)
        d = {q.key: (q, g) for q, g in (x for x in qs if x)}
        q, g = d["true_total"]
        gold = q.criteria[q.options[g.index(1.0)]]
        assert gold.startswith("$") and len(set(q.criteria.values())) == 4


def test_generation_is_deterministic(v3, tmp_path):
    a = v3.build("severity", ["A", "B"], 8, tmp_path / "a", random.Random("x"), "t")[0]
    b = v3.build("severity", ["A", "B"], 8, tmp_path / "b", random.Random("x"), "t")[0]
    assert a == b


# ------------------------------------------------------------------ decontamination
def _write_case_images(root, src, n, png=True, hashes=None, seed0=0):
    cases = []
    for i in range(n):
        rel = C.save_image(root, src, f"t{i}_0", _img(seed0 + i), png, hashes)
        q = C.noul_q("q0", f"Is this image number {i}?")
        cases.append(C.make_case(f"vis3_{src}:t{i}", f"vis3_{src}", "<image>", [rel], [(q, C.one_hot(2, i % 2))]))
    return cases


def test_decontam_v3_drops_exact_and_near_duplicates(v3, tmp_path):
    import decontam_vision as D
    root, eb = tmp_path / "v3", tmp_path / "eb"
    th, eh = [], []
    cases = _write_case_images(root, "severity", 6, hashes=th)
    C.write_jsonl(root / "vis3_severity.jsonl", cases)
    with open(root / "gen_hashes.jsonl", "w") as fh:
        for h in th:
            fh.write(json.dumps(h) + "\n")
    # eval candidates: an exact copy of image 0, a slightly brightened copy of image 1, an unrelated image
    exact = Image.open(root / cases[0]["images"][0]).convert("RGB")
    eb.mkdir()
    (eb / "images").mkdir()
    near = Image.eval(Image.open(root / cases[1]["images"][0]).convert("RGB"), lambda v: min(255, v + 3))
    C.save_image(eb, "efvis_x", "e0", exact, True, eh)
    C.save_image(eb, "efvis_x", "e1", near, True, eh)
    C.save_image(eb, "efvis_x", "e2", _img(999), True, eh)
    with open(eb / "cand_hashes.jsonl", "w") as fh:
        for h in eh:
            fh.write(json.dumps(h) + "\n")
    D.main(["v3", "--root", str(root), "--eval-build", str(eb)])
    drops = [json.loads(l) for l in open(root / "decontam_dropped.jsonl")]
    by_id = {d["case_id"]: d["reason"] for d in drops}
    assert by_id["vis3_severity:t0"].startswith("exact:eval:")
    assert by_id["vis3_severity:t1"].startswith(("phash:eval:", "exact:eval:"))
    kept = [json.loads(l) for l in open(root / "vis3_severity.jsonl")]
    assert "vis3_severity:t0" not in {c["case_id"] for c in kept}
    ys = [c["gold"]["q0"][1] for c in kept]
    assert ys.count(1.0) == ys.count(0.0)
    man = json.loads((root / "manifest.json").read_text())
    assert man["generators"]["severity"]["dropped_decontam"] >= 2


def test_decontam_eval_drops_train_duplicate_image(tmp_path):
    import decontam_vision as D
    tr, eb, out = tmp_path / "v1", tmp_path / "eb", tmp_path / "ev"
    th, eh = [], []
    tcases = _write_case_images(tr, "docvqa", 3, png=False, hashes=th)
    for c in tcases:
        c["case_id"] = c["case_id"].replace("vis3_", "vis_"); c["source"] = "vis_docvqa"
    C.write_jsonl(tr / "vis_docvqa.jsonl", tcases)
    with open(tr / "train_image_hashes.jsonl", "w") as fh:
        for h in th:
            fh.write(json.dumps(h) + "\n")
    ecases = []
    for i, im in enumerate([Image.open(tr / tcases[0]["images"][0]).convert("RGB"), _img(555), _img(556)]):
        rel = C.save_image(eb, "efvis_pope", f"e{i}", im, False, eh)
        ecases.append(C.make_case(f"efvis_pope:{i}", "efvis_pope", "<image>", [rel],
                                  [(C.noul_q("answer", "Is there a dog in the image?"), C.one_hot(2, i % 2))]))
    C.write_jsonl(eb / "cand" / "efvis_pope.jsonl", ecases)
    with open(eb / "cand_hashes.jsonl", "w") as fh:
        for h in eh:
            fh.write(json.dumps(h) + "\n")
    (eb / "fetch.json").write_text("{}")
    D.main(["eval", "--build", str(eb), "--train", str(tr), "--out", str(out)])
    kept = [json.loads(l)["case_id"] for l in open(out / "efvis_pope.jsonl")]
    assert kept == ["efvis_pope:1", "efvis_pope:2"]
    man = json.loads((out / "manifest.json").read_text())
    assert man["benchmarks"]["efvis_pope"]["dropped_decontam"] == 1
    assert "img_phash" in man["benchmarks"]["efvis_pope"]["drop_reasons"]


# ------------------------------------------------------------------ v1 / v2 builders on a fake cauldron shard
def _png_bytes(seed):
    import io
    buf = io.BytesIO(); _img(seed).save(buf, format="PNG"); return buf.getvalue()


@pytest.fixture()
def fake_hub(tmp_path, monkeypatch):
    """Two the_cauldron-shaped parquet shards per subset, served by patched vision_common HF helpers."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "brown", "pink"]
    shards = {}
    for sub in ("docvqa", "ai2d", "textvqa"):
        for s in range(2):
            rows = []
            for i in range(12):
                k = s * 12 + i
                if sub == "ai2d":
                    texts = [{"user": f"Look at the diagram.\nQuestion: Which part is {k}?\nChoices:\nA. leaf\nB. root\nC. stem\n"
                                      f"Answer with the letter.", "assistant": f"Answer: {'ABC'[k % 3]}"}]
                else:
                    texts = [{"user": f"What color is object {k}?\nShort answer required.", "assistant": colors[k % 10] + "."},
                             {"user": f"Is object {k} large?\nAnswer yes or no.", "assistant": "Yes." if k % 3 else "No."},
                             {"user": f"How many items in box {k}?\nShort answer.", "assistant": str(k % 7 + 1)}]
                rows.append({"images": [{"bytes": _png_bytes(k), "path": None}], "texts": texts})
            p = tmp_path / "hub" / sub / f"train-{s:05d}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(rows), p)
            shards[f"{sub}/train-{s:05d}.parquet"] = str(p)
    monkeypatch.setattr(C, "files", lambda repo, prefix: sorted(f for f in shards if f.startswith(prefix)))
    monkeypatch.setattr(C, "download", lambda repo, f: shards[f])

    def rows(repo, f, limit=None):
        for r in pq.read_table(shards[f]).to_pylist():
            yield r
    monkeypatch.setattr(C, "rows", rows)
    return tmp_path


def _valid(c):
    qs = tuple(Question(q["key"], q["mode"], q["instructions"], tuple(q["options"]), q["criteria"]) for q in c["questions"])
    Case(c["case_id"], c["source"], c["state"], qs, {k: tuple(v) for k, v in c["gold"].items()})
    assert c["state"].count("<image>") == len(c["images"])


def test_v1_freeform_and_letter_mc(fake_hub):
    import build_vision_v1 as V1
    root = fake_hub / "v1"
    h = []
    cases, meta = V1.build_freeform("docvqa", "docvqa", 20, root, random.Random("vision_v1:docvqa"), h, False, [0, 1])
    assert cases and meta["questions"] >= 20 and meta["split"] == "train"
    modes = {q["mode"] for c in cases for q in c["questions"]}
    assert modes <= {"choice", "noul"} and "choice" in modes
    for c in cases:
        _valid(c)
        assert (root / c["images"][0]).is_file()
    cases, meta = V1.build_letter_mc("ai2d", "ai2d", 10, root, random.Random("vision_v1:ai2d"), h, png=True)
    assert cases and all(c["state"].startswith("<image>\n\nLook at the diagram.") for c in cases)
    for c in cases:
        _valid(c)
        q = c["questions"][0]
        assert q["mode"] == "choice" and sorted(q["criteria"].values()) == ["leaf", "root", "stem"]


def test_v2_freeform_skips_banned_images(fake_hub, monkeypatch):
    import build_vision_v2 as V2
    # ban the image of the first row: its phash is in the "vision_v1" hash file
    import io
    ban = fake_hub / "ban.jsonl"
    ban.write_text(json.dumps({"phash": f"{C.phash(Image.open(io.BytesIO(_png_bytes(0))).convert('RGB')):016x}"}) + "\n")
    monkeypatch.setattr(V2, "BAN_FILES", [ban])
    monkeypatch.setattr(V2, "_BAN", None)
    monkeypatch.setitem(V2.SHARDS, "textvqa", [0, 1])
    cases, meta = V2.build_freeform("textvqa", 30, fake_hub / "v2", random.Random("vision_v2:textvqa"), [])
    ids = [c["case_id"] for c in cases]
    assert ids and "vis2_textvqa:0-0" not in ids and ids[0].startswith("vis2_textvqa:")
    for c in cases:
        _valid(c)
    tr, pr = V2.split_probe(cases, 5)
    tr, pr = V2.trim_noul(tr), V2.trim_noul(pr)
    for part in (tr, pr):
        ys = [c["gold"][q["key"]][1] for c in part for q in c["questions"] if q["mode"] == "noul"]
        assert ys.count(1.0) == ys.count(0.0)
