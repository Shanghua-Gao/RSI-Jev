"""End-to-end check: a real checkpoint answering a real Jev request.

Covers load_release -> serve.wire -> serve.infer -> serve.wire answers, which is
everything except the HTTP layer (that is covered without a GPU by
tests/test_serve.py). Deliberately imports no web framework, so it runs in the
training environment.

    python scripts/serve_smoke.py --ckpt DIR [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Root before scripts/: scripts/serve.py shadows the serve/ package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUEST = {
    "model": "jev-latest",
    "state": [
        {"role": "system", "content": "You are a support assistant."},
        {"role": "user", "content": "I was charged twice for the same order. "
                                    "Please refund the duplicate today."},
    ],
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
        "department": {"type": "choice", "instructions": "Which department should handle this?",
                       "criteria": {"billing": "Payments and refunds",
                                    "technical": "Software bugs and outages",
                                    "shipping": "Delivery and logistics"}},
        "urgency": {"type": "score", "instructions": "How urgent is the request?",
                    "criteria": ["Routine", "Urgent", "Emergency"]},
        "wide": {"type": "choice", "instructions": "Pick the matching bucket.",
                 "criteria": {f"bucket_{i}": f"Bucket number {i}" for i in range(64)}},
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import torch
    from load_release import load_release
    from serve.infer import score_questions
    from serve.wire import parse_questions, state_to_text, to_answer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, enc, meta = load_release(a.ckpt, device)
    questions = parse_questions(REQUEST["questions"])
    state = state_to_text(REQUEST["state"])
    preds, tokens = score_questions(model, tok, state, questions, enc, device=device,
                                    max_options=max(meta["spec"]["max_options"],
                                                    max(len(q.options) for q in questions)))
    answers = {q.key: to_answer(q, list(p.probs)) for q, p in zip(questions, preds)}

    # The contract the reference's own smoke test checks.
    for key, ans in answers.items():
        if ans["type"] == "noul":
            assert set(ans) == {"type", "noul"}, f"{key}: noul carries extra fields"
            assert 0.0 <= ans["noul"] <= 1.0
        else:
            probs = ans["probabilities"]
            assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-6), f"{key}: not normalized"
            assert all(0.0 <= v <= 1.0 and math.isfinite(v) for v in probs.values())
            assert 0.0 <= ans["confidence"] <= 1.0
        if ans["type"] == "score":
            assert set(ans["legend"]) == set(ans["probabilities"])
            assert 0.0 <= ans["score"] <= len(ans["legend"]) - 1
        if ans["type"] == "choice":
            assert ans["choice"] in probs
    assert len(answers["wide"]["probabilities"]) == 64

    body = {"model": REQUEST["model"], "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": len(questions)}}
    print(json.dumps(body, indent=2)[:2000])
    print(f"\nOK: {len(answers)} answers, contract checks passed, device={device}")
    if a.json:
        Path(a.json).write_text(json.dumps({"ckpt": Path(a.ckpt).name, "response": body}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
