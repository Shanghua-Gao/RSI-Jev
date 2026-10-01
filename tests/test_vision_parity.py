"""On real weights: an image request over HTTP answers what the offline encoder does.

The offline path is the one the vision releases were gated with: one question at
a time through `encode_vision_question` + `vision_collate` + the model. The served
path runs the image processor and the vision tower once per request and batches
the questions. Both run here, in one process, on the same loaded release.

Skips unless given a release with images and a set of image items:

    RSIJEV_VISION_CKPT=path/to/v4.0-release \\
    RSIJEV_VISION_ITEMS=path/to/items \\
    python -m pytest tests/test_vision_parity.py -q -m slow

`items` holds `cases.jsonl` (contract rows with `images`), `images.json`
({case_id: [image paths]}), and optionally `rt_ref.json`, a release build's own
round-trip reference ({"vision": [{case_id, key, probs}]}). With it, the served
answers are also compared with that reference: every argmax, and probabilities
within RSIJEV_VISION_REF_TOL (default 1e-3; a reference computed on another GPU
generation differs by more, see serve/README.md).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CKPT = os.environ.get("RSIJEV_VISION_CKPT")
ITEMS = os.environ.get("RSIJEV_VISION_ITEMS")
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (CKPT and ITEMS), reason="set RSIJEV_VISION_CKPT and "
                                 "RSIJEV_VISION_ITEMS to run on real weights")]


def _wire(q):
    if q["mode"] == "noul":
        return {"type": "noul", "instructions": q["instructions"],
                "criteria": {"true": q["criteria"]["true"], "false": q["criteria"]["false"]}}
    if q["mode"] == "choice":
        return {"type": "choice", "instructions": q["instructions"],
                "criteria": {o: q["criteria"].get(o) for o in q["options"]}}
    return {"type": "score", "instructions": q["instructions"],
            "criteria": [q["criteria"][o] for o in q["options"]]}


def _probs(ans, q):
    if ans["type"] == "noul":
        return [1 - ans["noul"], ans["noul"]]
    return [ans["probabilities"][o] for o in q["options"]]


def _argmax(v):
    return max(range(len(v)), key=v.__getitem__)


@pytest.fixture(scope="module")
def setup():
    import torch
    from fastapi.testclient import TestClient
    from serve.app import create_app
    from serve.decider import Decider
    d = Decider(CKPT, dtype="fp32")
    assert d.image_limits["supported"], d.image_limits
    root = Path(ITEMS)
    cases = [json.loads(line) for line in open(root / "cases.jsonl") if line.strip()]
    images = json.loads((root / "images.json").read_text())
    ref = json.loads((root / "rt_ref.json").read_text()) if (root / "rt_ref.json").exists() else None
    client = TestClient(create_app(d._scorer, served_model_name=d.name, images=d.image_limits))
    yield d, cases, images, ref, client
    del d
    torch.cuda.empty_cache()


def test_http_equals_the_offline_encoder(setup):
    import torch
    from PIL import Image
    from rsijev.contract import Question
    from rsijev.encode import unpermute_logits
    from rsijev.vision import encode_vision_question, vision_collate
    from serve.images import to_data_url
    d, cases, images, _, client = setup
    s = d.served
    worst = 0.0
    for c in cases:
        qs = {q["key"]: _wire(q) for q in c["questions"]}
        r = client.post("/v1/systemone", json={"model": d.name, "state": c["state"], "questions": qs,
                                               "images": [to_data_url(p) for p in images[c["case_id"]]]})
        assert r.status_code == 200, r.text
        pil = [Image.open(p).convert("RGB") for p in images[c["case_id"]]]
        for q in c["questions"]:
            qq = Question(q["key"], q["mode"], q["instructions"], tuple(q["options"]), q["criteria"])
            bt = vision_collate(s.tok, [encode_vision_question(s.tok, s.prep, c["state"], pil, qq, s.venc)],
                                s.meta["spec"]["max_options"], device=s.device)
            with torch.no_grad():
                z = unpermute_logits(s.model(**bt).float(), bt["option_perm"], bt["option_mask"])
            off = torch.softmax(z, -1)[0, :len(q["options"])].tolist()
            got = _probs(r.json()["answers"][q["key"]], q)
            assert _argmax(got) == _argmax(off), (c["case_id"], q["key"], got, off)
            worst = max(worst, max(abs(a - b) for a, b in zip(got, off)))
    assert worst < 1e-3, worst


def test_http_equals_the_release_reference(setup):
    from serve.images import to_data_url
    d, cases, images, ref, client = setup
    if not ref or not ref.get("vision"):
        pytest.skip("no rt_ref.json with a vision part")
    tol = float(os.environ.get("RSIJEV_VISION_REF_TOL", "1e-3"))
    by_id = {c["case_id"]: c for c in cases}
    worst = 0.0
    for r in ref["vision"]:
        c = by_id[r["case_id"]]
        q = next(q for q in c["questions"] if q["key"] == r["key"])
        resp = client.post("/v1/systemone", json={
            "model": d.name, "state": c["state"], "questions": {q["key"]: _wire(q)},
            "images": [to_data_url(p) for p in images[c["case_id"]]]})
        got = _probs(resp.json()["answers"][q["key"]], q)
        assert _argmax(got) == _argmax(r["probs"]), (r["case_id"], got, r["probs"])
        worst = max(worst, max(abs(a - b) for a, b in zip(got, r["probs"])))
    assert worst < tol, worst
