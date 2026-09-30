"""The vision tower and the image processor load a pinned base-model revision.

Both come from the base repo (Qwen/Qwen3.5-2B-Base), not from the release, so an
unpinned load follows whatever the repo's main branch is today. These check, with
from_pretrained and hf_hub_download mocked, that the revision reaches every call:
the release's own `vision.revision` when meta.json has one, else the pinned hash.

    python -m pytest tests/test_vision_revision.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev import vision                                          # noqa: E402

BASE = "Qwen/Qwen3.5-2B-Base"
PINNED = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"


def test_the_revision_comes_from_meta_else_the_pin():
    assert vision.PINNED_REVISIONS[BASE] == PINNED
    assert vision.vision_revision({"budget": 1024}, BASE) == PINNED
    assert vision.vision_revision(None, BASE) == PINNED
    assert vision.vision_revision({"revision": "abc123"}, BASE) == "abc123"
    assert vision.vision_revision({}, "some/other-model") is None


class _Proc:
    patch_size, merge_size = 16, 2


@pytest.mark.parametrize("given,expected", [(None, PINNED), ("abc123", "abc123")])
def test_image_prep_passes_the_revision(monkeypatch, given, expected):
    import transformers
    calls = []

    def fake(model_id, **kw):
        calls.append((model_id, kw.get("revision")))
        return _Proc()
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", staticmethod(fake))
    prep = vision.ImagePrep(BASE, vision.VisionConfig(), revision=given)
    prep._proc(256)                                     # the budgeted processor too
    assert calls == [(BASE, expected), (BASE, expected)]


@pytest.mark.parametrize("given,expected", [(None, PINNED), ("abc123", "abc123")])
def test_load_visual_passes_the_revision(monkeypatch, tmp_path, given, expected):
    import huggingface_hub
    import transformers
    seen = {"config": None, "files": []}

    class Stop(Exception):
        pass

    def fake_config(model_id, **kw):
        seen["config"] = kw.get("revision")
        raise Stop                                      # nothing past the config is needed

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(fake_config))
    with pytest.raises(Stop):
        vision.load_visual(BASE, revision=given)
    assert seen["config"] == expected

    # And the weight files: stop at the index download and check its revision.
    from transformers.models.qwen3_5 import modeling_qwen3_5
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        staticmethod(lambda m, **kw: type("C", (), {"vision_config": type("V", (), {})()})()))
    monkeypatch.setattr(modeling_qwen3_5, "Qwen3_5VisionModel", lambda vc: None)

    def fake_download(repo, filename, **kw):
        seen["files"].append((filename, kw.get("revision")))
        raise Stop
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    with pytest.raises(Stop):
        vision.load_visual(BASE, revision=given)
    assert seen["files"] == [("model.safetensors.index.json", expected)]
