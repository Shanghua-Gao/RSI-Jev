"""Releases without an early exit, a recorded cap or a truncation policy (v1.0 through
v4.0-VL) are served exactly as before the early-exit work, and fixed-exit releases without
aux exits exactly as before adaptive exit.

The same tiny random Qwen3.5 (text + vision, no exit) is run through the serving
paths by this tree and by the tree it was cut from (`git archive BASE_REV`), each
in its own process: plain, read once, document cache, image states plain and read
once, and the encoder on a state over the cap. Every probability and every encoded
id must be bit-for-bit the same. The same for an exit-4 model against 6362d36 (the exit
port, before adaptive exit). Skips without git history.

    python -m pytest tests/test_old_releases_unchanged.py -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("torchvision")

ROOT = Path(__file__).resolve().parent.parent
BASE_REV = "0af7fe0"            # the serving code of v4.0-VL
EXIT_REV = "6362d36"            # the early-exit port, before adaptive exit

SCRIPT = r'''
import json, sys
root, out, exit_layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path[:0] = [root]
from serve.runtime import keep_fused_kernels_off
keep_fused_kernels_off("cpu")
import torch
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel, Qwen3_5VisionModel
from rsijev.arch import ArchConfig, DecisionModel
from rsijev.contract import Question
from rsijev.encode import EncodeConfig, encode_question
from rsijev.vision import IMAGE_PAD, ImagePrep, VisionConfig, VisionDecisionModel
from serve import infer
BASE = "Qwen/Qwen3.5-2B-Base"
tok = AutoTokenizer.from_pretrained(BASE)
prep = ImagePrep(BASE, VisionConfig(image_token_budget=64, min_tokens_per_image=16))
torch.manual_seed(0)
tc = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                       num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_num_key_heads=2,
                       linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                       layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2,
                       rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                        "mrope_section": [2, 1, 1], "mrope_interleaved": True})
tc._attn_implementation = "sdpa"
tower = Qwen3_5TextModel(tc).eval()
vc = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                         patch_size=16, spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
                         num_position_embeddings=64)
vc._attn_implementation = "sdpa"
torch.manual_seed(1)
visual = Qwen3_5VisionModel(vc).eval()
torch.manual_seed(5)
arch = ArchConfig(max_options=8, freeze_base=True, readout="option_xattn", xattn_combine="mlp",
                  xattn_mlp_hidden=16, readout_layer=-1,
                  **({"exit_layer": exit_layer} if exit_layer else {}))
vm = VisionDecisionModel(tower, 64, arch, visual=visual, image_token_id=tok.convert_tokens_to_ids(IMAGE_PAD)).eval()
enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
venc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical", max_length=2048 + 64)
QS = [Question("a", "noul", "Is the square red?", ("false", "true"), {"false": "No.", "true": "Yes."}),
      Question("b", "choice", "Which colour?", ("red", "green", "blue"), {"red": "Red.", "green": "Green.", "blue": "Blue."}),
      Question("c", "score", "How bright?", ("0", "1", "2", "3"), {"0": "Dark", "1": "Dim", "2": "Bright", "3": "Glaring"})]
state = "The report says revenue grew 12% while costs fell. " * 12
a = Image.new("RGB", (96, 64), (220, 20, 20)); b = Image.new("RGB", (64, 64), (20, 20, 220))
kw = dict(max_options=8, device="cpu")
res = {}
P = lambda r: [list(p.probs) for p in r[0]] + [r[1]]
res["plain"] = P(infer.score_questions(vm, tok, state, QS, enc, **kw))
res["cached"] = P(infer.score_questions_cached(vm, tok, state, QS, enc, min_saved_tokens=0, doc_cache=False, **kw))
c = infer.DocCache(max_entries=8, max_bytes=1 << 30, holdback=8)
res["doc"] = [P(infer.score_questions_cached(vm, tok, s, QS, enc, doc_cache=c, **kw))
              for s in (state, state, state + "Margins held.")]
img_state = "Left: <image>\nRight: <image>\nA note."
res["img_plain"] = P(infer.score_image_questions(vm, tok, prep, img_state, [a, b], QS, venc, **kw))
res["img_cached"] = P(infer.score_image_questions_cached(vm, tok, prep, img_state, [a, b], QS, venc,
                                                         min_saved_tokens=0, doc_cache=False, **kw))
c = infer.DocCache(max_entries=8, max_bytes=1 << 30, holdback=8)
res["img_doc"] = [P(infer.score_image_questions_cached(vm, tok, prep, img_state, [a, b], QS, venc, doc_cache=c, **kw))
                  for _ in range(2)]
long_state = "Query: where?\n\n" + " ".join(f"Sentence {i}." for i in range(1500))
keys = ("input_ids", "option_index", "option_span", "decision_index", "options", "option_perm", "mode")
res["encode_over_cap"] = [{k: encode_question(tok, long_state, q, enc)[k] for k in keys} for q in QS]
rows, _ = infer.encode_questions(tok, state, QS, enc)
res["fast_rows"] = [{k: r[k] for k in keys} for r in rows]
res["plan_path"] = infer.plan_request(tok, state, QS, enc, min_saved_tokens=0, doc_cache=False)["path"]
json.dump(res, open(out, "w"))
'''


def _run(root: Path, exit_layer: int = 0) -> dict:
    out = Path(tempfile.mkdtemp()) / "out.json"
    script = Path(tempfile.mkdtemp()) / "run.py"
    script.write_text(SCRIPT)
    p = subprocess.run([sys.executable, str(script), str(root), str(out), str(exit_layer)], capture_output=True,
                       text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd=str(root))
    if p.returncode != 0 and ("OSError" in p.stderr or "offline" in p.stderr.lower()):
        pytest.skip(f"Qwen3.5 tokenizer / image processor unavailable: {p.stderr[-300:]}")
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(out.read_text())


@pytest.mark.parametrize("rev,exit_layer", [(BASE_REV, 0), (EXIT_REV, 4)],
                         ids=["no_exit_vs_v4.0-VL", "fixed_exit_vs_exit_port"])
def test_serving_is_bitwise_unchanged(rev, exit_layer):
    try:
        blob = subprocess.run(["git", "-C", str(ROOT), "archive", rev, "rsijev", "serve"],
                              check=True, capture_output=True).stdout
    except Exception:                                                   # noqa: BLE001
        pytest.skip(f"git history with {rev} unavailable")
    base = Path(tempfile.mkdtemp())
    tarfile.open(fileobj=BytesIO(blob)).extractall(base, filter="data")
    old, new = _run(base, exit_layer), _run(ROOT, exit_layer)
    assert set(old) == set(new)
    for k in old:
        assert new[k] == old[k], k
