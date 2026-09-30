"""Image requests: the wire schema, its 422s, and an image request end to end.

Against a stub scorer, so no weights and no GPU: what reaches the scorer, what a
bad image gets back, what /v1/limits says, and that Decider and HTTP agree. The
model side (the vision tower's features shared across a request's questions,
batched M-RoPE rows) is pinned on a tiny random model in test_vision_model.py,
and against the real release in test_vision_parity.py (slow).

    python -m pytest tests/test_images.py -q
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PIL = pytest.importorskip("PIL")
from PIL import Image                                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve.app import create_app                                   # noqa: E402
from serve.decider import Decider                                  # noqa: E402
from serve.images import (MAX_IMAGE_PIXELS, image_limits, to_data_url)  # noqa: E402

MODEL = "rsi-jev-v4.0-qwen3.5-2b"
LIMITS = image_limits({"image_token_budget": 1024, "min_tokens_per_image": 64})
calls: list[tuple] = []


def stub_scorer(state, questions, images=None):
    """Probabilities that depend on the images, so a lost image shows."""
    calls.append((state, questions, images))
    bright = 0.0 if not images else sum(im.getpixel((0, 0))[0] for im in images) / (255 * len(images))
    out = []
    for q in questions:
        n = len(q.options)
        raw = [1.0 + (bright if i == n - 1 else 0.0) for i in range(n)]
        out.append([v / sum(raw) for v in raw])
    return out, 7 + 100 * len(images or [])


def png(color=(200, 10, 10), size=(64, 48), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def url(data: bytes, mime="image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


NOUL = {"type": "noul", "instructions": "Is the receipt paid?"}
CHOICE = {"type": "choice", "instructions": "Which colour?",
          "criteria": {"red": "Mostly red", "blue": "Mostly blue"}}


def body(images=None, state="<image> A receipt.", **questions):
    b = {"model": MODEL, "state": state, "questions": questions or {"paid": NOUL}}
    if images is not None:
        b["images"] = images
    return b


@pytest.fixture
def client():
    calls.clear()
    return TestClient(create_app(stub_scorer, served_model_name=MODEL, images=LIMITS))


def test_an_image_request_reaches_the_scorer_as_pil_images(client):
    r = client.post("/v1/systemone", json=body([url(png())], paid=NOUL, colour=CHOICE))
    assert r.status_code == 200, r.text
    state, questions, images = calls[-1]
    assert state == "<image> A receipt."                 # the marker is left for the encoder
    assert [q.key for q in questions] == ["paid", "colour"]
    assert len(images) == 1 and images[0].size == (64, 48) and images[0].mode == "RGB"
    data = r.json()
    assert data["answers"]["colour"]["type"] == "choice"
    assert data["usage"] == {"input_tokens": 107, "output_tokens": 2}


def test_jpeg_and_webp_are_taken(client):
    for fmt, mime in (("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
        r = client.post("/v1/systemone", json=body([url(png(fmt=fmt), mime)]))
        assert r.status_code == 200, (fmt, r.text)


def test_no_images_is_the_text_path(client):
    """Omitted, null and [] all call the scorer exactly as a text request does."""
    for images in (None, [], "null"):
        b = body(state="A receipt.")
        if images == "null":
            b["images"] = None
        elif images is not None:
            b["images"] = images
        r = client.post("/v1/systemone", json=b)
        assert r.status_code == 200, r.text
        assert calls[-1][2] is None


def test_up_to_four_images(client):
    four = [url(png((i * 60, 0, 0))) for i in range(4)]
    r = client.post("/v1/systemone", json=body(four, state="a <image> b <image> c <image> d <image>"))
    assert r.status_code == 200, r.text
    assert len(calls[-1][2]) == 4
    r = client.post("/v1/systemone", json=body(four + [url(png())], state="five"))
    assert r.status_code == 422


def test_images_follow_their_markers_in_order(client):
    a, b = url(png((255, 0, 0))), url(png((0, 0, 255)))
    client.post("/v1/systemone", json=body([a, b], state="Left: <image> Right: <image>"))
    assert [im.getpixel((0, 0)) for im in calls[-1][2]] == [(255, 0, 0), (0, 0, 255)]


@pytest.mark.parametrize("images,state,needle", [
    (["https://example.com/a.png"], "<image>", "only data URLs"),
    (["http://example.com/a.png"], "<image>", "only data URLs"),
    (["not a url"], "<image>", "not a base64 data URL"),
    (["data:image/gif;base64,R0lGODlhAQABAAAAACw="], "<image>", "unsupported type image/gif"),
    (["data:image/png;base64,@@@@"], "<image>", "invalid base64"),
    (["data:image/png;base64,"], "<image>", "empty image"),
    (["data:image/png;base64," + base64.b64encode(b"hello world").decode()], "<image>",
     "could not decode"),
    ([None], "<image>", None),                                # schema: a string is required
    ([url(png()), url(png())], "one <image> only", "1 <image> markers for 2 images"),
    ([url(png())], "<image> and <image>", "2 <image> markers for 1 images"),
    ([url(png())], "<image> <|image_pad|>", "reserved token <|image_pad|>"),
])
def test_bad_images_are_a_clear_422(client, images, state, needle):
    calls.clear()
    r = client.post("/v1/systemone", json=body(images, state=state))
    assert r.status_code == 422, r.text
    if needle:
        assert needle in r.json()["error"]["message"], r.text
    assert not calls                                   # rejected before the model


def test_a_gif_disguised_as_png_is_refused(client):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="GIF")
    r = client.post("/v1/systemone", json=body([url(buf.getvalue())]))
    assert r.status_code == 422 and "unsupported format GIF" in r.json()["error"]["message"]


def test_a_huge_or_thin_image_is_refused(client, monkeypatch):
    import serve.images as si
    monkeypatch.setattr(si, "MAX_IMAGE_PIXELS", 64 * 48 - 1)
    r = client.post("/v1/systemone", json=body([url(png())]))
    assert r.status_code == 422 and "pixels" in r.json()["error"]["message"]
    monkeypatch.setattr(si, "MAX_IMAGE_PIXELS", MAX_IMAGE_PIXELS)
    r = client.post("/v1/systemone", json=body([url(png(size=(1000, 4)))]))
    assert r.status_code == 422 and "aspect ratio" in r.json()["error"]["message"]


def test_an_oversized_payload_is_refused_before_decoding(client, monkeypatch):
    import serve.images as si
    monkeypatch.setattr(si, "MAX_IMAGE_BYTES", 100)
    r = client.post("/v1/systemone", json=body([url(png(size=(300, 300)))]))
    assert r.status_code == 422 and "MiB" in r.json()["error"]["message"]


def test_a_text_model_refuses_images():
    c = TestClient(create_app(stub_scorer, served_model_name=MODEL))
    r = c.post("/v1/systemone", json=body([url(png())]))
    assert r.status_code == 422 and "text-only" in r.json()["error"]["message"]
    assert c.get("/v1/limits").json()["images"] == {"supported": False}
    # ... and still answers text
    assert c.post("/v1/systemone", json=body(state="A receipt.")).status_code == 200


def test_limits_report_image_support(client):
    lim = client.get("/v1/limits").json()["images"]
    assert lim["supported"] is True
    assert lim["max_images"] == 4
    assert lim["image_token_budget"] == 1024
    assert lim["max_tokens_per_image"] == {"1": 1024, "2": 512, "3": 341, "4": 256}
    assert lim["state_marker"] == "<image>"
    assert lim["prefix_cache"] is False and lim["document_cache"] is False
    assert set(lim["formats"]) == {"png", "jpeg", "webp"}


def test_the_schema_documents_images(client):
    schema = client.get("/openapi.json").json()
    props = schema["components"]["schemas"]["SystemOneRequest"]["properties"]
    assert "images" in props
    assert "images" not in schema["components"]["schemas"]["SystemOneRequest"].get("required", [])


def test_decide_with_images_equals_http(tmp_path, client):
    d = Decider.from_scorer(stub_scorer, name=MODEL, images=LIMITS)
    raw = png((90, 0, 0))
    p = tmp_path / "a.png"
    p.write_bytes(raw)
    q = {"paid": NOUL, "colour": CHOICE}
    http = client.post("/v1/systemone", json=body([url(raw)], **q)).json()["answers"]
    for im in (url(raw), str(p), p, raw, Image.open(io.BytesIO(raw))):
        assert d.decide("<image> A receipt.", q, images=[im]) == http


def test_decider_on_a_text_model_refuses_images():
    from serve.wire import RequestError
    d = Decider.from_scorer(stub_scorer, name=MODEL)
    with pytest.raises(RequestError, match="text-only"):
        d.decide("<image>", {"paid": NOUL}, images=[png()])


def test_to_data_url_is_lossless_for_pil_images():
    im = Image.frombytes("RGB", (5, 3), bytes(range(45)))
    back = Image.open(io.BytesIO(base64.b64decode(to_data_url(im).split(",", 1)[1])))
    assert back.convert("RGB").tobytes() == im.tobytes()


def test_exif_rotation_is_applied(client):
    im = Image.new("RGB", (40, 20), (255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = 6                                      # rotate 90 CW on display
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    r = client.post("/v1/systemone", json=body([url(buf.getvalue(), "image/jpeg")]))
    assert r.status_code == 200
    assert calls[-1][2][0].size == (20, 40)


def test_the_model_worker_plans_images_on_their_own_path(monkeypatch):
    """The server's worker (serve/batcher.py) plans an image request with the image
    encoder, runs it with the image scorer, and never pools it with text rows."""
    import serve.infer as infer
    from serve.batcher import GpuWorker, ModelRunner
    from serve.wire import parse_questions

    seen = {}

    def fake_plan(tok, prep, state, images, questions, enc):
        seen["planned"] = (state, len(images))
        return {"path": "image", "encoded": [{"input_ids": [1, 2, 3]}] * len(questions),
                "options": [len(q.options) for q in questions]}

    def fake_run(model, tok, plan, **kw):
        seen["ran"] = plan["path"]
        from rsijev.contract import Prediction
        return [Prediction((0.25, 0.75)) for _ in plan["options"]], 3

    monkeypatch.setattr(infer, "plan_image_request", fake_plan)
    monkeypatch.setattr(infer, "score_image_planned", fake_run)
    runner = ModelRunner(object(), None, None, spec_max_options=8, device="cpu",
                         prep=object(), venc=object())
    w = GpuWorker(runner, window_ms=5)
    try:
        qs = parse_questions({"q": {"type": "noul", "instructions": "Red?"}})
        img = Image.new("RGB", (8, 8))
        plan = w.plan("<image> x", qs, [img])
        assert not runner.poolable(plan)
        probs, tokens = w.enqueue(plan).result(timeout=10)
    finally:
        w.close()
    assert seen == {"planned": ("<image> x", 1), "ran": "image"}
    assert probs == [[0.25, 0.75]] and tokens == 3


def test_the_model_worker_of_a_text_model_refuses_images():
    from serve.batcher import ModelRunner
    from serve.wire import RequestError, parse_questions
    runner = ModelRunner(object(), None, None, spec_max_options=8, device="cpu",
                         name="rsi-jev-v3.0-qwen3.5-2b")
    qs = parse_questions({"q": {"type": "noul", "instructions": "Red?"}})
    with pytest.raises(RequestError, match="text-only"):
        runner.plan("<image> x", qs, [Image.new("RGB", (8, 8))])
