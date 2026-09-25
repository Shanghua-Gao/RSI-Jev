"""The Jev wire contract: request/response mapping, with no torch and no server.

This module is the compatibility surface. It turns a `POST /v1/systemone` body
into this project's `Question` contract and turns a probability distribution
back into a Jev answer. Everything here is pure Python so the contract can be
tested without a GPU or an HTTP stack.

What is copied exactly from the reference (`reference/openjev-sglang`, which
implements https://docs.typesafe.ai/api):

  * the request shape `{state, model, questions}`, unknown fields rejected;
  * the three question types and their `criteria` shapes, including noul's
    `"true"`/`"false"` keys and their `"Yes"`/`"No"` defaults;
  * the answer shapes -- noul carries ONLY `noul`, choice carries `choice`,
    `probabilities` and `confidence`, score adds `legend` and reports the
    expected zero-based rubric index;
  * `confidence` as 1 - H(p)/ln(K), clamped to [0, 1];
  * the limits 1-64 questions and 2-64 options.

What deliberately differs, because the wire contract is the compatibility
surface and the prompt is not (the reference says so itself: "The providers can
use different internal prompts and inference procedures despite receiving
equivalent payloads."):

  * **The prompt.** The reference renders a chat template and reads the logprobs
    of single-token labels A/B/C. This model is a BASE model with a trained
    readout, and it is served with exactly the encoder it was trained with.
  * **Option keys are visible to this model.** The reference hides them and
    guarantees that renaming a key cannot change the answer. This model was
    trained on blocks rendered `- key: description`, so the key is part of the
    input and renaming it can move the answer. Reported by `GET /v1/limits` as
    `option_keys_visible_to_model: true`.
  * **`usage.output_tokens`.** The reference generates one token per question
    plus a warm-up, and reports N+1. This path generates nothing at all, so it
    reports one readout per question and no warm-up.
"""
from __future__ import annotations

import json
from typing import Any

from rsijev.contract import Prediction, Question

MAX_QUESTIONS = 64
MAX_ANSWERS = 64                       # options per choice question / levels per score
NOUL_OPTIONS = ("false", "true")       # this project's contract order; the wire is key-based
NOUL_DEFAULTS = {"true": "Yes", "false": "No"}
CHAT_ROLES = {"system", "user", "assistant", "tool"}


class RequestError(ValueError):
    """A request the API rejects. `status` is the HTTP code to return."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def serialize(value: Any) -> str:
    """Text for the model. Strings pass through; everything else is compact JSON.

    Compact separators match the reference's `orjson.dumps`, and compact JSON is
    also the form the training states took, so structured state is rendered in
    distribution rather than in an invented transcript format.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def state_to_text(state: Any) -> str:
    """Render `state` for the encoder.

    A chat transcript is validated exactly as the reference validates it -- the
    same roles, text-only content, the same messages envelope -- so a request
    that the reference rejects is rejected here too. It is then serialized as
    JSON rather than run through a chat template, because a base model has no
    chat template and was never trained on one.
    """
    candidate = state
    if isinstance(state, dict) and set(state) == {"messages"}:
        candidate = state["messages"]
    if (isinstance(candidate, list) and candidate
            and all(isinstance(item, dict) and "role" in item for item in candidate)):
        for item in candidate:
            if item["role"] not in CHAT_ROLES:
                raise RequestError("Chat state has an unsupported message role")
            content = item.get("content")
            if isinstance(content, list):
                if not all(isinstance(part, dict) and part.get("type") == "text"
                           and isinstance(part.get("text"), str) for part in content):
                    raise RequestError("Jev state supports text content only")
            elif content is not None and not isinstance(content, str):
                raise RequestError("Chat content must be text, text parts, or null")
        return serialize(candidate)
    return serialize(state)


def to_question(key: str, spec: dict[str, Any]) -> Question:
    """One wire question -> this project's `Question`.

    Raises RequestError (422) for anything the reference would also reject.
    """
    if not isinstance(spec, dict):
        raise RequestError(f"questions.{key}: must be an object")
    qtype = spec.get("type")
    instructions = spec.get("instructions")
    if instructions is None:
        raise RequestError(f"questions.{key}: instructions is required")
    text = serialize(instructions)
    criteria = spec.get("criteria")

    if qtype == "noul":
        # Wire keys are "true"/"false" with "Yes"/"No" defaults. The contract
        # fixes the option ORDER as ("false", "true"); p(true) is read by key,
        # never by position, so the two orders cannot drift apart.
        given = criteria or {}
        if not isinstance(given, dict):
            raise RequestError(f"questions.{key}: noul criteria must be an object")
        unknown = set(given) - {"true", "false"}
        if unknown:
            raise RequestError(f"questions.{key}: unknown noul criteria {sorted(unknown)}")
        described = {k: serialize(given.get(k, NOUL_DEFAULTS[k])) for k in ("false", "true")}
        return Question(key=key, mode="noul", instructions=text,
                        options=NOUL_OPTIONS, criteria=described)

    if qtype == "choice":
        if not isinstance(criteria, dict):
            raise RequestError(f"questions.{key}: choice criteria must be an object")
        if not 2 <= len(criteria) <= MAX_ANSWERS:
            raise RequestError(
                f"questions.{key}: choice needs 2-{MAX_ANSWERS} options, got {len(criteria)}")
        options = tuple(criteria)                       # insertion order is the option order
        # A null description means "the key is its own meaning"; the encoder
        # already falls back to the bare key when a description is empty.
        described = {k: ("" if v is None else serialize(v)) for k, v in criteria.items()}
        return Question(key=key, mode="choice", instructions=text,
                        options=options, criteria=described)

    if qtype == "score":
        if not isinstance(criteria, list):
            raise RequestError(f"questions.{key}: score criteria must be an array")
        if not 2 <= len(criteria) <= MAX_ANSWERS:
            raise RequestError(
                f"questions.{key}: score needs 2-{MAX_ANSWERS} levels, got {len(criteria)}")
        options = tuple(str(i) for i in range(len(criteria)))
        described = {str(i): serialize(v) for i, v in enumerate(criteria)}
        return Question(key=key, mode="score", instructions=text,
                        options=options, criteria=described)

    raise RequestError(f"questions.{key}: unknown question type {qtype!r}")


def parse_questions(questions: Any) -> list[Question]:
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a non-empty object")
    if len(questions) > MAX_QUESTIONS:
        raise RequestError(f"questions: at most {MAX_QUESTIONS}, got {len(questions)}")
    return [to_question(k, v) for k, v in questions.items()]


def confidence(probs: list[float]) -> float:
    """(K * p_max - 1) / (K - 1): how far the peak is above chance, 0 at uniform
    and 1 when one option takes everything.

    This is the published definition -- TypeSafe documents it for three options as
    "(3 x largest probability - 1) / 2" -- and it is PEAK-based, not entropy-based.
    We shipped 1 - H(p)/ln(K) first, back when the statistic was undocumented. The
    two agree exactly at uniform and at one-hot and nowhere else, which is why the
    tests here did not catch it: they only pinned those two points. For four
    options with the peak at 0.50 the entropy form gives 0.10 against the correct
    0.33, so anything thresholding on confidence saw the wrong number.

    Only choice and score answers carry this; noul answers do not, which matches
    the reference and is enforced in to_answer.
    """
    k = len(probs)
    if k < 2:                       # the contract forbids it; do not divide by zero
        return 1.0
    return min(1.0, max(0.0, (k * max(probs) - 1) / (k - 1)))


def to_answer(q: Question, probs: list[float]) -> dict[str, Any]:
    """One distribution -> one Jev answer.

    The scalar readouts come from `Prediction`, where the contract defines them
    once, so this wrapper cannot quietly use a different rule than the evaluator.
    """
    pred = Prediction(tuple(probs))
    p = list(pred.probs)                                 # normalized by the contract
    if q.mode == "noul":
        return {"type": "noul", "noul": pred.noul()}
    distribution = dict(zip(q.options, p))
    if q.mode == "score":
        return {"type": "score",
                "score": pred.score(),
                "legend": dict(q.criteria),
                "probabilities": distribution,
                "confidence": confidence(p)}
    return {"type": "choice",
            "choice": pred.choice(q.options),
            "probabilities": distribution,
            "confidence": confidence(p)}


def limits() -> dict[str, Any]:
    return {"max_answers_per_question": MAX_ANSWERS,
            "max_questions": MAX_QUESTIONS,
            "option_keys_visible_to_model": True}
