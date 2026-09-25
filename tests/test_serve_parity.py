"""The serving forward pass must equal the evaluated one, exactly.

`serve.infer.score_questions` exists because `rsijev.evaluate.predict` requires
`Case` objects and a `Case` requires gold, which a served request does not have.
Two code paths that are supposed to agree will drift unless something checks, so
this loads the real weights and asserts they agree bit for bit.

CPU, one forward pass, no training:

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_serve_parity.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.arch import ArchConfig, DecisionModel          # noqa: E402
from rsijev.contract import Case                           # noqa: E402
from rsijev.encode import EncodeConfig                     # noqa: E402
from rsijev.evaluate import predict                        # noqa: E402
from rsijev.train import seed_everything                   # noqa: E402
from serve.infer import score_questions                    # noqa: E402
from serve.wire import parse_questions, state_to_text      # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B-Base"
STATE = {"ticket": "RV-1", "body": "Charged twice, please refund."}
QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
    "team": {"type": "choice", "instructions": "Which department?",
             "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
    "urgency": {"type": "score", "instructions": "How urgent?",
                "criteria": ["Routine", "Urgent", "Emergency"]},
}


@pytest.mark.slow
def test_matches_evaluate_predict():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    hidden = (getattr(lm.config, "text_config", None) or lm.config).hidden_size
    arch = ArchConfig(readout="option_xattn", readout_layer=-1, max_options=80,
                      freeze_base=True, option_pool="mean", residual=False)
    seed_everything(17)
    model = DecisionModel(lm.model, hidden, arch)
    model.scorer.to(torch.float32)
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")

    questions = parse_questions(QUESTIONS)
    state = state_to_text(STATE)

    served, tokens = score_questions(model, tok, state, questions, enc,
                                     max_options=80, device="cpu", batch_size=16)

    # The same questions through the evaluated path. Gold is required by `Case`
    # and is never read by `predict`; it is uniform here and used by nothing.
    case = Case(case_id="parity", source="test", state=state, questions=tuple(questions),
                gold={q.key: tuple([1 / len(q.options)] * len(q.options)) for q in questions})
    evaluated = predict(model, tok, [case], enc, max_options=80, device="cpu", batch_size=16)

    assert len(served) == len(evaluated) == len(questions)
    for (_, q, want), got in zip(evaluated, served):
        assert got.probs == want.probs, f"{q.key}: serving path diverged from predict()"
    assert tokens > 0
