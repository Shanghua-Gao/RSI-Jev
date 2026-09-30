"""Images in the playground: `rsi-jev demo`, the page's image input and the image
examples. No weights, no GPU: the scorers here are stand-ins.

    python -m pytest tests/test_demo_images.py -q
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient                          # noqa: E402

from serve.demo import create_demo_app                             # noqa: E402
from serve.images import image_limits, parse_images, to_data_url   # noqa: E402

PAGE = (ROOT / "serve" / "ui.html").read_text()
EXAMPLES = json.loads((ROOT / "serve" / "examples.json").read_text())
IMAGE_EXAMPLES = [e for e in EXAMPLES if e.get("images")]
VISION = image_limits({"image_token_budget": 1024, "min_tokens_per_image": 64})


def _scorer(seen):
    def scorer(state, questions, images=None):
        seen.append(len(images or []))
        return [[1 / len(q.options)] * len(q.options) for q in questions], 10
    return scorer


def _client(*models):
    return TestClient(create_demo_app(models, hardware="test", dtype="fp32"))


def _png():
    from PIL import Image
    return to_data_url(Image.new("RGB", (40, 30), (200, 10, 10)))


def test_image_examples_are_ours_small_and_valid():
    assert 2 <= len(IMAGE_EXAMPLES) <= 3
    for e in IMAGE_EXAMPLES:
        assert json.loads(e["state"]) and "\n" in e["state"]
        for u in e["images"]:
            assert len(base64.b64decode(u.split(",", 1)[1])) < 150 * 1024, e["id"]
        parse_images(e["images"], e["state"])          # the server's own checks
        assert e["state"].count("<image>") == len(e["images"])


def test_limits_say_whether_any_model_takes_images():
    assert _client(("vl", _scorer([]), VISION)).get("/v1/limits").json()["images"]["supported"]
    text = _client(("text", _scorer([]), {"supported": False}))
    assert text.get("/v1/limits").json()["images"] == {"supported": False}


def test_an_image_example_runs_with_its_teacher_and_skips_text_models():
    seen = []
    c = _client(("vl", _scorer(seen), VISION), ("text", _scorer(seen), {"supported": False}))
    e = IMAGE_EXAMPLES[0]
    r = c.post("/demo/compare", json={"state": e["state"], "questions": e["questions"],
                                      "images": e["images"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert list(d["answers"]) == ["vl"] and d["skipped"] == {"text": "text-only: does not take images"}
    assert d["gold"] == e["gold"] and seen == [1]
    # The same state with another picture is not the example, so no teacher.
    r = c.post("/demo/compare", json={"state": e["state"], "questions": e["questions"],
                                      "images": [_png()]})
    assert r.json()["gold"] is None


def test_image_limits_are_enforced():
    c = _client(("vl", _scorer([]), VISION))
    q = {"a": {"type": "noul", "instructions": "Red?"}}
    r = c.post("/demo/compare", json={"state": "x", "questions": q, "images": [_png()] * 5})
    assert r.status_code == 422 and "at most 4" in r.json()["error"]
    r = c.post("/demo/compare", json={"state": "<image> <image>", "questions": q,
                                      "images": [_png()]})
    assert r.status_code == 422 and "markers" in r.json()["error"]
    only_text = _client(("text", _scorer([]), {"supported": False}))
    r = only_text.post("/demo/compare", json={"state": "x", "questions": q, "images": [_png()]})
    assert r.status_code == 422


def test_the_page_takes_images_only_when_the_model_does():
    assert 'fetch("v1/limits")' in PAGE
    assert '$("#imgs").hidden = !VISION.supported' in PAGE
    assert "!(e.images && e.images.length)" in PAGE, "image examples hidden for text models"
    assert 'id="imgs" hidden' in PAGE, "hidden until /v1/limits says otherwise"
    for how in ('type="file"', '"drop"', 'addEventListener("paste"'):
        assert how in PAGE, f"no image input by {how}"
    assert "IMAGES.length >= maxImages()" in PAGE, "the 1-4 limit is enforced on the page"
    assert "Insert ${MARKER} at the cursor" in PAGE
    assert "body.images = IMAGES.map(i => i.url)" in PAGE
    assert '$("#ex").addEventListener("change"' in PAGE, "the example menu must load examples"


def test_env_reports_the_vision_extra():
    from serve.runtime import describe_vision
    assert "installed" in describe_vision({"pillow": "12", "torchvision": "0.2"})
    line = describe_vision({"pillow": None, "torchvision": "0.2"})
    assert "pillow" in line and 'pip install "rsi-jev[vision]"' in line
