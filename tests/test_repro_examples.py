"""The re-run scripts for the v4.0-VL demos and benchmarks build well-formed requests.

No weights, no GPU, no network: every runner is driven through `Decider.from_scorer`
with a stub scorer (the server's own request validation and answer code), or checked
against the server's request schema. The real-weights run is a separate GPU job.

    python -m pytest tests/test_repro_examples.py -q
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image                                                  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from serve.app import SystemOneRequest                                 # noqa: E402
from serve.decider import Decider                                      # noqa: E402
from serve.images import image_limits                                  # noqa: E402

EX = ROOT / "examples"
VB = ROOT / "scripts" / "vision_benches"
LIMITS = image_limits({"image_token_budget": 1024, "min_tokens_per_image": 64})
calls: list[tuple] = []


def load(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stub_scorer(state, questions, images=None):
    """Non-uniform, deterministic, and it records what reached the model."""
    calls.append((state, questions, images))
    out = []
    for q in questions:
        w = [1.0 + j for j in range(len(q.options))]
        out.append([x / sum(w) for x in w])
    return out, 10


def stub_decider() -> Decider:
    calls.clear()
    return Decider.from_scorer(stub_scorer, name="stub", images=LIMITS)


class StubClient:
    """The benchmark scripts' Client, over the stub Decider."""
    name = "stub"

    def __init__(self):
        self.d = stub_decider()

    def ask(self, state, questions, images=None):
        return self.d.decide(state, questions, images=images or None), 1.0


def png(path: Path, size=(40, 30), color=(200, 30, 30)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


# ---------------------------------------------------------------------------- the gallery
gallery = load("gallery_run", EX / "gallery" / "run.py")


def test_gallery_ships_every_image_with_a_credit():
    items = gallery.load_items()
    assert len(items) == 143 and sum(len(i["questions"]) for i in items) == 227
    credits = (EX / "gallery" / "CREDITS.md").read_text()
    used = {p for i in items for p in i["images"]}
    assert len(used) == 89
    for p in used:
        assert (EX / "gallery" / p).is_file(), p
        if "/photos/" in p:
            assert f"| {Path(p).name} |" in credits, p
    shipped = {str(p.relative_to(EX / "gallery")) for p in (EX / "gallery" / "images").rglob("*") if p.is_file()}
    assert shipped == used                       # nothing shipped that no question uses
    assert "limit30" not in json.dumps(items)


def test_gallery_questions_are_valid_requests():
    for it in gallery.load_items():
        for q in it["questions"]:
            SystemOneRequest.model_validate({"model": "m", "state": it["state"], "questions": {q["key"]: q["spec"]}})
            assert not q.get("ref") or all(isinstance(r, str) for r in q["ref"])


def test_gallery_runner_scores_every_question_through_decider():
    client = SimpleNamespace(ask=lambda s, q, im: (stub_decider().decide(s, q, images=im), 1.0))
    items = gallery.load_items()
    rows = gallery.run(items, client, log=lambda s: None)
    assert len(rows) == 227 and all(r["pick"] for r in rows)
    lines = gallery.summary(rows)
    assert lines[-2].startswith("all") and "/227" in lines[-2].replace(" ", "")


def test_gallery_visa_items_reach_the_model_downscaled():
    it = next(i for i in gallery.load_items() if i["category"] == "visa")
    ims = gallery.item_images(it)
    assert all(max(im.size) <= it["max_side"] for im in ims)
    d = stub_decider()
    q = it["questions"][0]
    d.decide(it["state"], {q["key"]: q["spec"]}, images=ims)
    state, questions, images = calls[-1]
    assert len(images) == 1 and images[0].size == ims[0].size
    assert [o for o in questions[0].options] == ["A", "B"]


def test_gallery_http_body_is_what_decider_takes():
    c = gallery.Client.__new__(gallery.Client)
    it = gallery.load_items()[0]
    q = it["questions"][0]
    body = c.body(it["state"], {q["key"]: q["spec"]}, gallery.item_images(it))
    SystemOneRequest.model_validate(body)
    assert body["images"][0].startswith("data:image/jpeg;base64,")


def test_gallery_generator_reproduces_the_shipped_pngs(tmp_path):
    mk = load("gallery_make_images", EX / "gallery" / "make_images.py")
    if not Path(mk.FONT).exists():
        pytest.skip("DejaVu Sans not installed")
    names = mk.draw_all(tmp_path)
    assert len(names) == 30
    from PIL import ImageChops
    for n in names:
        a = Image.open(tmp_path / n).convert("RGB")
        b = Image.open(EX / "gallery" / "images" / "generated" / n).convert("RGB")
        assert ImageChops.difference(a, b).getbbox() is None, n


# ---------------------------------------------------------------------------- chat vs RSI-Jev
rsi_client = load("chat_rsi_client", EX / "chat_vs_rsi" / "rsi_client.py")
summarize = load("chat_summarize", EX / "chat_vs_rsi" / "summarize.py")


def test_chat_questions_are_gallery_questions():
    gal = {i["id"]: i for i in gallery.load_items()}
    qs = rsi_client.load_questions()
    assert len(qs) == 30
    for q in qs:
        g = gal[q["id"].rsplit("/", 1)[0]]
        assert [str(Path("../gallery") / p) for p in g["images"]] == q["images"]
        assert next(x for x in g["questions"] if x["key"] == q["key"])["spec"] == q["spec"]


def test_chat_rsi_requests_validate_and_score():
    d = stub_decider()
    for q in rsi_client.load_questions():
        body = rsi_client.request_body(q)
        req = SystemOneRequest.model_validate(body)
        ans = d.decide(req.state, body["questions"], images=body["images"])
        assert rsi_client.pick(ans[q["key"]]) in (["true", "false"] if q["spec"]["type"] == "noul"
                                                   else list(q["spec"]["criteria"]))


def test_chat_summary_and_lenient_reading():
    qs = rsi_client.load_questions()[:3]
    rsi = {"model": "m", "results": [dict(id=q["id"], pick=q["ref"][0], correct=True, valid=True,
                                          confidence_available=True, ms_p50=60.0) for q in qs]}
    q0 = qs[0]
    ans = q0["ref"][0] if q0["spec"]["type"] != "noul" else {"true": "yes", "false": "no"}[q0["ref"][0]]
    chat = {"model": "c", "results": [
        dict(id=q0["id"], text=json.dumps({"q": "echo", "answer": ans}), pick=None, valid=False, correct=False,
             confidence_available=False, ms_p50=500.0, output_tokens=12)] +
        [dict(id=q["id"], text="{}", pick=None, valid=False, correct=False, confidence_available=False,
              ms_p50=400.0, output_tokens=3) for q in qs[1:]]}
    out = summarize.summarize(qs, rsi, chat)
    assert out["paired"]["only_rsi_correct"] == 3 and out["chat"]["lenient_post_hoc"]["n_correct"] == 1
    assert "| valid output | 3/3 | 0/3 |" in summarize.markdown(out)


# ---------------------------------------------------------------------------- Breakout adapter
adapter = load("breakout_adapter", EX / "breakout" / "adapter.py")


def _judge_body(image_url):
    crit = {str(i): f"Column {i}." for i in range(1, 6)}
    return {"image": image_url, "mode": "shared",
            "questions": {"action": {"type": "choice", "scoring": "label",
                                     "instructions": "Which numbered column contains the white ball?",
                                     "criteria": crit}}}


def test_breakout_adapter_speaks_both_apis(tmp_path):
    from fastapi.testclient import TestClient
    from serve.images import to_data_url
    d = stub_decider()
    sent = []

    def post(body):
        SystemOneRequest.model_validate(body)       # what rsi-jev serve would accept
        sent.append(body)
        return d.request(body["state"], body["questions"], images=body["images"])

    app = adapter.create_app("http://upstream.invalid", None, post=post)
    c = TestClient(app)
    r = c.post("/v1/judge", json=_judge_body(to_data_url(png(tmp_path / "board.png"))))
    assert r.status_code == 200, r.text
    a = r.json()["answers"]["action"]
    assert a["choice"] in a["probabilities"] and set(a["probabilities"]) == {"1", "2", "3", "4", "5"}
    assert r.json()["metrics"]["elapsed_ms"] >= 0
    assert "scoring" not in sent[-1]["questions"]["action"] and sent[-1]["state"] == "<image>"
    assert len(calls[-1][2]) == 1                  # the screenshot reached the model
    bad = c.post("/v1/judge", json=_judge_body("not-a-data-url"))
    assert bad.status_code == 422


def test_breakout_adapter_maps_noul_and_score():
    body = adapter.to_systemone({"image": "data:image/png;base64,AAAA", "questions": {
        "n": {"type": "noul", "instructions": "Is the ball moving up?"},
        "s": {"type": "score", "instructions": "How close is the ball?", "criteria": ["far", "near"]}}})
    SystemOneRequest.model_validate(body)
    out = adapter.to_judge({"n": {"type": "noul", "noul": 0.8},
                            "s": {"type": "score", "score": 0.3, "probabilities": {"0": 0.7, "1": 0.3}}}, 5.0, "m")
    assert out["answers"]["n"]["probabilities"] == {"yes": 0.8, "no": pytest.approx(0.2)}
    assert out["answers"]["s"]["choice"] == "0"


# ---------------------------------------------------------------------------- benchmark scripts
common = load("common", VB / "common.py")
mc = load("vb_mc", VB / "mc.py")
visa = load("vb_visa", VB / "visa.py")
heldout = load("vb_heldout", VB / "heldout.py")
laya = load("vb_laya", VB / "laya.py")


def test_mc_item_becomes_one_letter_keyed_choice():
    q = {"id": "blink-1", "task": "Counting", "state": "2 images are attached, in order (first to last).",
         "question": "Which image has more dogs?", "options": ["the first", "the second"], "gold": 1,
         "images": [Image.new("RGB", (32, 24), "white"), Image.new("RGB", (24, 32), "black")]}
    state, questions, images = mc.item_request(q)
    SystemOneRequest.model_validate(common.request_body(state, questions, images))
    assert questions["answer"]["criteria"] == {"A": "the first", "B": "the second"}
    rec = mc.score_item(StubClient(), q)
    assert rec["pred"] == 1 and rec["correct"] and len(rec["probs"]) == 2
    assert len(calls[-1][2]) == 2 and calls[-1][0] == q["state"]
    five = dict(q, images=q["images"] * 3)
    assert "excluded" in mc.score_item(StubClient(), five)


def test_visa_prompt_becomes_one_request_per_order(tmp_path):
    prompts = {o: SimpleNamespace(state="Production-line inspection photo of candles.",
                                  question="Is everything in the photo good, or is at least one part defective?",
                                  options=("all good", "defective") if o == 0 else ("defective", "all good"),
                                  defective_index=1 - o) for o in (0, 1)}
    fake = SimpleNamespace(prompt=lambda obj, config, order: prompts[order])
    png(tmp_path / "x.JPG", size=(2000, 1000))
    item = SimpleNamespace(image="x.JPG", obj="candle", label="normal", types=[])
    recs = visa.score_image(StubClient(), fake, item, tmp_path)
    assert [r["order"] for r in recs] == [0, 1]
    assert max(calls[-1][2][0].size) == visa.MAX_SIDE      # scaled to fit 1,536 px
    # the stub prefers the last option: defective in order 0, all good in order 1
    assert recs[0]["logodds_defective"] > 0 > recs[1]["logodds_defective"]


def test_heldout_case_and_blank_control(tmp_path):
    (tmp_path / "images").mkdir()
    png(tmp_path / "images" / "a.png", size=(50, 20))
    case = {"case_id": "efvis_pope:random:1", "source": "efvis_pope", "state": "<image>",
            "images": ["images/a.png"],
            "questions": [{"key": "answer", "mode": "noul", "instructions": "Is there a dog in the image?",
                           "options": ["false", "true"], "criteria": {"false": "No", "true": "Yes"}}],
            "gold": {"answer": [0.0, 1.0]}}
    rows = heldout.score_case(StubClient(), case, tmp_path, blank=False)
    assert rows[0]["correct"] and calls[-1][2][0].getpixel((0, 0)) == (200, 30, 30)
    rows_b = heldout.score_case(StubClient(), case, tmp_path, blank=True)
    im = calls[-1][2][0]
    assert im.size == (50, 20) and im.getpixel((0, 0)) == heldout.GREY and rows_b[0]["blank"]
    choice = {"key": "answer", "mode": "choice", "instructions": "Which?", "options": ["B", "A"],
              "criteria": {"A": "x", "B": "y"}}
    assert list(heldout.wire_question(choice)["criteria"]) == ["B", "A"]      # option order kept


def test_heldout_analysis(tmp_path):
    out = tmp_path / "h.jsonl"
    rows = [{"bench": "efvis_pope", "blank": b, "probs": [0.2, 0.8], "correct": c, "key": f"{i}{b}"}
            for i, (b, c) in enumerate([(False, True), (False, False), (True, False), (True, True)])]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lines, res = heldout.analyze(out)
    assert res["efvis_pope"]["top1"] == 0.5 and res["efvis_pope"]["blank_top1"] == 0.5
    assert res["efvis_pope"]["ece"] == pytest.approx(0.3)


def test_laya_record_encodings(tmp_path):
    png(tmp_path / "i.jpg")
    recs = [{"id": "1", "index": 0, "label": 0, "image": "i.jpg", "state_text": None,
             "question": {"type": "choice", "instructions": "What colour?", "criteria": ["red", "blue"]}},
            {"id": "2", "index": 1, "label": 1, "image": "i.jpg", "state_text": "Hint: look.",
             "question": {"type": "noul", "instructions": "Is it red?", "criteria": None}},
            {"id": "3", "index": 2, "label": 2, "image": "i.jpg", "state_text": None,
             "question": {"type": "score", "instructions": "How good?", "criteria": ["bad", "ok", "good"]}}]
    client = StubClient()
    for r in recs:
        state, questions, images = laya.record_request(r, tmp_path)
        SystemOneRequest.model_validate(common.request_body(state, questions, images))
        client.ask(state, questions, images)
        q = calls[-1][1][0]
        if r["question"]["type"] == "noul":
            assert q.mode == "noul" and q.criteria == {"false": "", "true": ""}   # no descriptions
        elif r["question"]["type"] == "score":
            assert q.mode == "choice" and q.options == ("0", "1", "2") and q.criteria["2"] == "good"
        else:
            assert q.options == ("red", "blue") and q.criteria == {"red": "", "blue": ""}


def test_laya_compare_pairs_items(tmp_path):
    ours, ref = tmp_path / "o.jsonl", tmp_path / "l.jsonl"
    o, r = [], []
    for i in range(20):
        ds = "cauldron_vsr" if i < 10 else "cauldron_ai2d"
        o.append({"dataset": ds, "id": str(i), "qtype": "choice", "label": 0, "probs": [0.9, 0.1] if i % 2 else [0.4, 0.6]})
        r.append({"dataset": ds, "id": str(i), "qtype": "choice", "label": 0, "probs_calibrated": [0.3, 0.7]})
    ours.write_text("".join(json.dumps(x) + "\n" for x in o))
    ref.write_text("".join(json.dumps(x) + "\n" for x in r))
    lines, res = laya.compare(ours, ref)
    allp, clean = res["pooled"]["all types: all sets"], res["pooled"]["all types: clean sets"]
    assert allp["n"] == 20 and allp["ours"] == 0.5 and allp["laya"] == 0.0
    assert clean["n"] == 10                         # ai2d is a set v4.0-VL trained on
    assert allp["diff"][3:5] == (10, 0)             # McNemar discordant counts


# An absolute filesystem path (a machine's own directory layout) or a Windows drive path.
# URL paths such as /v1/systemone do not match: they start at none of these roots.
ABS_PATH = re.compile(r"(?<![\w.~:/-])/(?:home|Users|root|mnt|media|scratch|data|opt|srv|var|tmp|net|n|gpfs|lustre|work|private)/[\w.-]"
                      r"|\b[A-Za-z]:\\[\w.-]")


def test_bench_scripts_name_no_absolute_paths():
    """Everything a reader runs must work from a clone: paths are relative to the repo."""
    for p in [*VB.glob("*.py"), *VB.glob("*.md"), *EX.rglob("*.py"), *EX.rglob("*.md"), *EX.rglob("*.mjs"),
              *EX.rglob("*.json"), *EX.rglob("*.sh")]:
        m = ABS_PATH.search(p.read_text())
        assert m is None, f"{p.relative_to(ROOT)} has an absolute path: {m.group(0)!r}"
