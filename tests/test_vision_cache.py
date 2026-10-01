"""The vision-output cache (serve.infer.VisionCache): keyed on decoded pixels and the
token budget, bounded, and a hit hands back the very tensor the miss computed."""
import torch
from PIL import Image

from serve.infer import VisionCache, _image_feats, default_vision_cache


def _img(color, size=(40, 30)):
    return Image.new("RGB", size, color)


def test_key_follows_pixels_budget_and_order():
    a, b = _img((1, 2, 3)), _img((1, 2, 4))
    k = VisionCache.key
    assert k([a], 1024) == k([_img((1, 2, 3))], 1024)        # same pixels, new object
    assert k([a], 1024) != k([b], 1024)
    assert k([a], 1024) != k([a], 512)                       # another resolution
    assert k([a, b], 512) != k([b, a], 512)                  # order matters
    assert k([a], 1024, "m|r1") != k([a], 1024, "m|r2")      # another processor


def test_hit_returns_the_stored_tensor_and_bounds_hold():
    vc = VisionCache(max_entries=2, max_bytes=10**9)
    f = torch.randn(4, 8)
    vc.put("a", f, torch.tensor([[1, 4, 4]]), [4])
    got = vc.get("a")
    assert got[0] is f and got[2] == [4]
    vc.put("b", torch.zeros(1), torch.zeros(1, 3), [1])
    vc.put("c", torch.zeros(1), torch.zeros(1, 3), [1])
    assert vc.get("a") is None and vc.stats["evicted"] == 1
    small = VisionCache(max_entries=8, max_bytes=8)
    small.put("x", torch.zeros(100), torch.zeros(1, 3), [1])  # too big: not kept
    assert small.get("x") is None


def test_image_feats_uses_the_plan_hit_and_stores_a_miss():
    class M:
        calls = 0

        def image_embeds(self, pv, grid):
            M.calls += 1
            return pv * 2
    vc = VisionCache()
    plan = {"image_feats": None, "pixel_values": torch.ones(3), "grid": torch.ones(1, 3),
            "ntok": [3], "vision_cache": vc, "vision_key": "k"}
    out = _image_feats(M(), plan, "cpu")
    assert M.calls == 1 and vc.get("k")[0] is out
    hit = {"image_feats": out}
    assert _image_feats(M(), hit, "cpu") is out and M.calls == 1


def test_off_unless_asked(monkeypatch):
    class Holder:
        pass
    monkeypatch.delenv("RSIJEV_VISION_CACHE", raising=False)
    assert default_vision_cache(Holder()) is None
    monkeypatch.setenv("RSIJEV_VISION_CACHE", "1")
    h = Holder()
    assert default_vision_cache(h) is default_vision_cache(h) is not None
