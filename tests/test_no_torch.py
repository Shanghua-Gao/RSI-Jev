"""The Mac install has no torch and no transformers: the MLX path must work without them."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from test_mlx_parity import release, tok  # noqa: F401  (fixtures)

ROOT = Path(__file__).resolve().parents[1]
CPU = {**os.environ, "RSIJEV_MLX_DEVICE": "cpu"}       # MLX: these tests run on CPU, as the MLX parity tests do
BLOCK = """
import sys
class _NoHeavy:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "torchvision", "transformers"):
            raise ImportError(name + " is not installed in this test")
sys.meta_path.insert(0, _NoHeavy())
"""


def run_blocked(body: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCK + textwrap.dedent(body)], cwd=ROOT, env=CPU,
                          capture_output=True, text=True, timeout=600)


def test_the_block_blocks():
    r = run_blocked("import torch")
    assert r.returncode != 0 and "not installed in this test" in r.stderr


def test_decide_imports_nothing_heavy():
    r = run_blocked("""
        import serve.decide_cli
        print("ok")
    """)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


@pytest.mark.parametrize("text", [
    "plain words", "á é क्ष", "before<|image_pad|>after", "<tts_text_bos_single>", "x\n\ny",
    "中文，标点。", "don't  stop\t\r\n", "\U0001F469‍\U0001F4BB",
])
def test_frozen_tokenizer_equals_transformers(tmp_path, text):
    transformers = pytest.importorskip("transformers")
    from rsijev.frozen_tok import FrozenTokenizer, freeze
    from rsijev.vision import PINNED_REVISIONS
    base = "Qwen/Qwen3.5-4B-Base"
    try:
        at = transformers.AutoTokenizer.from_pretrained(base, revision=PINNED_REVISIONS[base])
    except Exception as e:
        pytest.skip(f"Qwen3.5-4B tokenizer unavailable: {e}")
    at.save_pretrained(tmp_path)
    freeze(tmp_path)
    ft = FrozenTokenizer(tmp_path)
    ids = at(text, add_special_tokens=False)["input_ids"]
    assert ft(text)["input_ids"] == ids
    assert ft([text, text])["input_ids"] == [ids, ids]
    assert ft.decode(ids) == at.decode(ids)
    assert (ft.pad_token_id, ft.eos_token_id) == (at.pad_token_id, at.eos_token_id)


def test_mlx_answers_without_torch_equal_the_transformers_tokenizer(release, tmp_path):  # noqa: F811
    pytest.importorskip("mlx.core")
    _, out16, _ = release
    assert (out16 / "tokenizer.frozen.json").is_file()
    old = tmp_path / "old-package"
    old.mkdir()
    for f in out16.iterdir():
        if f.name != "tokenizer.frozen.json":
            (old / f.name).symlink_to(f)
    body = """
        import json
        from rsijev.contract import Question
        from rsijev.mlx.serving import load_for_serving, make_scorer
        from serve.app import create_app
        s = load_for_serving(%r, dtype="fp32")
        qs = [Question("a", "noul", "Is it raining?", ("false", "true"), {"false": "No.", "true": "Yes."}),
              Question("b", "choice", "Which one?", ("x", "y"), {"x": "First.", "y": "Second."})]
        out = make_scorer(s)("It is raining \\u00e9 in the city.\\n\\nReally.", qs)
        print("RESULT" + json.dumps(out[0]))
    """
    new_run = run_blocked(body % str(out16))
    assert new_run.returncode == 0, new_run.stderr[-2000:]
    old_run = subprocess.run([sys.executable, "-c", textwrap.dedent(body % str(old))], cwd=ROOT, env=CPU,
                             capture_output=True, text=True, timeout=600)
    assert old_run.returncode == 0, old_run.stderr[-2000:]
    pick = lambda r: [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT")][-1]
    assert pick(new_run) == pick(old_run)


def test_mlx_loads_images_without_torch(release):  # noqa: F811
    pytest.importorskip("mlx.core")
    pytest.importorskip("PIL")
    _, out16, _ = release
    r = run_blocked(f"""
        from PIL import Image
        from rsijev.mlx.vision import ImagePrep, NativeProcessor
        prep = ImagePrep({str(out16)!r}, budget=64)
        assert prep._cls is NativeProcessor
        pv, grid, tokens = prep([Image.new("RGB", (64, 96), (9, 80, 200))])
        print("ok", pv.shape, grid.tolist(), tokens)
    """)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]


def test_native_image_processor_equals_transformers(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("transformers")
    pytest.importorskip("mlx.core")
    from rsijev.mlx.vision import NativeProcessor, _processor_class
    cfg = {"size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16, "temporal_patch_size": 2,
           "merge_size": 2, "image_mean": [0.5] * 3, "image_std": [0.5] * 3,
           "image_processor_type": "Qwen2VLImageProcessorFast"}
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(cfg))
    rng = np.random.default_rng(3)
    images = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
              for h, w in ((64, 64), (33, 500), (700, 41), (480, 640), (1, 90), (1023, 767))]
    pil = _processor_class("pil")
    for tokens in (64, 1024):
        u2 = 32 * 32
        size = {"shortest_edge": min(64, tokens) * u2, "longest_edge": tokens * u2}
        for im in images:
            a = pil.from_pretrained(str(tmp_path), size=dict(size))([im], return_tensors="np")
            b = NativeProcessor.from_pretrained(str(tmp_path), size=dict(size))([im], return_tensors="np")
            assert np.array_equal(np.asarray(a["image_grid_thw"]), b["image_grid_thw"])
            x, y = np.asarray(a["pixel_values"], dtype=np.float32), b["pixel_values"]
            assert x.shape == y.shape and x.dtype == y.dtype
            assert np.array_equal(x, y), float(np.abs(x - y).max())
