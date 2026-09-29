"""The vLLM backend against the PyTorch serve path, on a small fixture.

The fixture (tests/fixtures/vllm_parity_v3.0.*) holds the eight serve/examples.json
requests plus one 1,093-token state with four questions (long enough that vLLM
reads a cached block of it), answered by the PyTorch bf16 serve path
(`score_questions_cached`, what scripts/serve.py runs), and the tower's final
normed hidden state at each question's decision token in bf16 and in fp32.

Two things are checked on the vLLM path:

* the hidden state vLLM returns (`token_embed`, no activation) is the tensor
  `readout_layer: -1` reads -- the final, normed state -- and is as close to the
  fp32 tower as the PyTorch bf16 tower is;
* every answer picks the same option as the PyTorch bf16 path, with
  probabilities within bf16 kernel noise.

Needs vLLM, a GPU and the v3.0 checkpoint (RSIJEV_V3_CKPT); skipped otherwise.

    RSIJEV_V3_CKPT=path/to/rsi-jev-v3.0-qwen3.5-2b python -m pytest tests/test_vllm_parity.py -q

Regenerate the fixture with the PyTorch path (no vLLM needed):

    python tests/test_vllm_parity.py --ckpt path/to/rsi-jev-v3.0-qwen3.5-2b
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "scripts"))

FIXTURE = ROOT / "tests" / "fixtures" / "vllm_parity_v3.0.json"
ROWS = FIXTURE.with_suffix(".safetensors")
LONG_TICKET_COPIES = 13


def _requests():
    """[(id, state_text, questions)] -- the fixture's inputs, rebuilt from the repo."""
    from bench import QUESTIONS, TICKET
    from serve.wire import parse_questions, state_to_text
    out = []
    for e in json.loads((ROOT / "serve" / "examples.json").read_text()):
        out.append((e["id"], state_to_text(e["state"]), parse_questions(e["questions"])))
    out.append(("long_ticket", "\n\n".join([TICKET] * LONG_TICKET_COPIES),
                parse_questions(QUESTIONS)))
    return out


def _decision_rows(model, tok, enc, requests, device):
    """Final normed hidden state at each question's decision token, from the
    PyTorch tower (HF `last_hidden_state`)."""
    from rsijev.encode import encode_question
    rows = []
    with torch.no_grad():
        for _, state, qs in requests:
            for q in qs:
                e = encode_question(tok, state, q, enc)
                out = model.tower(input_ids=torch.tensor([e["input_ids"]], device=device))
                rows.append(out.last_hidden_state[0, e["decision_index"]].float().cpu())
    return torch.stack(rows)


def make_fixture(ckpt: str, device: str = "cuda") -> None:
    from safetensors.torch import save_file

    from load_release import load_release
    from serve.infer import score_questions_cached
    requests = _requests()
    model, tok, enc, meta = load_release(ckpt, device, infer_dtype=torch.bfloat16)
    spec = meta["spec"]
    answers = []
    for rid, state, qs in requests:
        preds, _ = score_questions_cached(model, tok, state, qs, enc, device=device,
                                          max_options=spec["max_options"])
        answers.append({"id": rid, "keys": [q.key for q in qs],
                        "probs": [[round(x, 6) for x in p.probs] for p in preds]})
    bf16 = _decision_rows(model, tok, enc, requests, device)
    del model
    torch.cuda.empty_cache()
    model, tok, enc, meta = load_release(ckpt, device, infer_dtype=None)
    fp32 = _decision_rows(model, tok, enc, requests, device)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps({"checkpoint": Path(ckpt).resolve().name,
                                   "path": "PyTorch bf16 serve path (score_questions_cached)",
                                   "answers": answers}, indent=1) + "\n")
    save_file({"decision_bf16": bf16.to(torch.bfloat16), "decision_fp32": fp32}, str(ROWS))
    print(f"wrote {FIXTURE} and {ROWS}")


@pytest.fixture(scope="module")
def vllm_model():
    pytest.importorskip("vllm")
    ckpt = os.environ.get("RSIJEV_V3_CKPT")
    if not ckpt or not Path(ckpt, "tower.safetensors").exists():
        pytest.skip("set RSIJEV_V3_CKPT to the v3.0 checkpoint")
    # No torch.cuda call before the engine starts: touching CUDA here makes the
    # forked engine process fail to initialise it.
    from serve.vllm_backend import load_vllm_release
    model, tok, enc, meta = load_vllm_release(ckpt, "cuda",
                                              model_dir=os.environ.get("RSIJEV_VLLM_MODEL_DIR"))
    yield model, tok, enc, meta
    model.engine.shutdown()


@pytest.mark.slow
def test_hidden_state_is_final_normed(vllm_model):
    from safetensors.torch import load_file

    from rsijev.encode import encode_question
    model, tok, enc, _ = vllm_model
    ref = load_file(str(ROWS))
    seqs, idx = [], []
    for _, state, qs in _requests():
        for q in qs:
            e = encode_question(tok, state, q, enc)
            seqs.append(e["input_ids"])
            idx.append(e["decision_index"])
    got = model.engine.hidden_rows(seqs, idx)
    vl = torch.stack([rows[i - off].float() for (off, rows), i in zip(got, idx)])
    fp32, bf16 = ref["decision_fp32"], ref["decision_bf16"].float()

    def rel(a, b):
        return ((a - b).norm(dim=-1) / b.norm(dim=-1)).max().item()
    err_vllm, err_torch = rel(vl, fp32), rel(bf16, fp32)
    print(f"\nmax relative error vs fp32 tower: vLLM bf16 {err_vllm:.4f}, "
          f"PyTorch bf16 {err_torch:.4f}")
    # Not a different layer: a mid-stack or un-normed state is nowhere near this.
    assert err_vllm < 0.05
    assert err_vllm < 2 * err_torch + 0.01


@pytest.mark.slow
def test_answers_match_torch_bf16(vllm_model):
    from serve.infer import score_questions
    model, tok, enc, meta = vllm_model
    want = {a["id"]: a for a in json.loads(FIXTURE.read_text())["answers"]}
    worst, n = 0.0, 0
    for rid, state, qs in _requests():
        preds, _ = score_questions(model, tok, state, qs, enc, device="cuda",
                                   batch_size=len(qs), max_options=meta["spec"]["max_options"])
        for q, p, w in zip(qs, preds, want[rid]["probs"]):
            assert max(range(len(w)), key=w.__getitem__) == \
                max(range(len(p.probs)), key=p.probs.__getitem__), f"{rid}/{q.key}"
            worst = max(worst, max(abs(x - y) for x, y in zip(p.probs, w)))
            n += 1
    print(f"\n{n} answers, worst |dprob| vs PyTorch bf16 {worst:.4f}")
    assert worst < 0.05


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    make_fixture(ap.parse_args().ckpt)
