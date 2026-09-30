"""The worked examples in `serve/README.md` must be what the models actually return.

An invented response is not caught by any cheap check, because a careful author
makes it self-consistent: the one this replaced had a `confidence` that matched
its own probabilities and a `score` that was their exact expectation. It was
still wrong -- it showed `urgency` as "Routine" where the model says "Urgent".
Only the real weights can tell, so that part is marked `slow`.

    python -m pytest tests/test_documented_example.py -q            # shapes only
    python -m pytest tests/test_documented_example.py -q -m slow    # against weights

The README shows one request and each release's answer to it, each answer under a
`<!-- response: vX.Y -->` marker. v1.0 is fetched from the Hub. v3.0 is read from
RSIJEV_V3_CKPT (a checkpoint directory) or from the Hugging Face cache, and is
skipped when neither has it, so the quick sweep never starts a 5 GB download.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from serve.wire import parse_questions, to_answer           # noqa: E402

DOC = ROOT / "serve" / "README.md"
REPOS = {"v1.0": "shgao/rsi-jev-v1.0-qwen3.5-2b", "v3.0": "shgao/rsi-jev-v3.0-qwen3.5-2b"}
TOLERANCE = 0.01            # the doc rounds to three places; kernels are not deterministic


def documented(version: str = "v3.0") -> tuple[dict, dict]:
    """The request and `version`'s response as `serve/README.md` prints them."""
    text = DOC.read_text()
    request = json.loads(re.search(r"-d '(\{.*?\})'\n```", text, re.S).group(1))
    marker = re.escape(f"<!-- response: {version} -->")
    response = json.loads(re.search(marker + r"\n```json\n(\{.*?\})\n```", text, re.S).group(1))
    return request, response


def test_the_documented_request_is_a_legal_request():
    request, _ = documented()
    assert set(request) == {"model", "state", "questions"}
    questions = parse_questions(request["questions"])
    assert [q.mode for q in questions] == ["noul", "choice", "score"], \
        "the example is there to show all three modes"


@pytest.mark.parametrize("version", sorted(REPOS))
def test_the_documented_response_has_the_shape_its_types_require(version):
    """noul carries only `noul`; choice adds probabilities; score adds a legend."""
    request, response = documented(version)
    questions = {q.key: q for q in parse_questions(request["questions"])}
    for key, answer in response["answers"].items():
        q = questions[key]
        expected = to_answer(q, [1.0 / len(q.options)] * len(q.options))
        assert set(answer) == set(expected), f"{key}: fields differ from the contract"
        if q.mode == "score":
            assert 0 <= answer["score"] <= len(q.options) - 1, \
                f"{key}: score must be a zero-based rubric index, not a fraction"
            assert set(answer["legend"]) == set(answer["probabilities"])
    assert response["usage"]["output_tokens"] == len(questions), \
        "one readout per question; nothing is generated"


def _checkpoint(version: str):
    from huggingface_hub import snapshot_download
    if version == "v3.0":
        local = os.environ.get("RSIJEV_V3_CKPT")
        if local:
            return local
        try:
            return snapshot_download(REPOS[version], local_files_only=True)
        except Exception:
            pytest.skip("v3.0 is not in the Hugging Face cache; set RSIJEV_V3_CKPT to a "
                        "checkpoint directory, or run `rsi-jev download v3.0-2b`")
    return snapshot_download(REPOS[version])


@pytest.mark.slow
@pytest.mark.parametrize("version", sorted(REPOS))
def test_the_documented_numbers_are_the_ones_the_model_returns(version):
    import torch
    from serve.infer import score_questions_cached
    from serve.release import load_release
    from serve.wire import state_to_text

    path = _checkpoint(version)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model, tok, enc, meta = load_release(path, device, infer_dtype=dtype)

    request, response = documented(version)
    questions = parse_questions(request["questions"])
    preds, _ = score_questions_cached(
        model, tok, state_to_text(request["state"]), questions, enc,
        device=device, batch_size=8, max_options=meta["spec"]["max_options"])

    wrong = []
    for q, pred in zip(questions, preds):
        got, said = to_answer(q, list(pred.probs)), response["answers"][q.key]
        for field, value in said.items():
            if isinstance(value, (int, float)) and abs(value - got[field]) > TOLERANCE:
                wrong.append(f"{q.key}.{field}: documented {value}, model returns {got[field]:.3f}")
            elif isinstance(value, dict):
                wrong += [f"{q.key}.{field}[{k}]: documented {v}, model returns {got[field][k]:.3f}"
                          for k, v in value.items()
                          if isinstance(v, (int, float)) and abs(v - got[field][k]) > TOLERANCE]
    assert not wrong, (f"serve/README.md documents {version} answers the model does not give:\n  "
                       + "\n  ".join(wrong))
