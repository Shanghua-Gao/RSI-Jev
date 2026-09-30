"""The HTTP API's answers, in-process.

    from rsijev import Decider
    d = Decider("shgao/rsi-jev-v3.0-qwen3.5-2b")
    d.decide("I was charged twice. Please refund.",
             {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}})

`decide(state, questions)` takes the `state` and `questions` of a `POST
/v1/systemone` body and returns its `answers`. It is not a second implementation:
the checkpoint is loaded by `serve.server.load_for_serving`, the request is
validated by the server's own schema, and `serve.app.answer_request` runs it, which
is exactly what the route calls. `tests/test_decider.py` asserts the two agree.
"""
from __future__ import annotations

from typing import Any

from serve.app import SystemOneRequest, answer_request


class Decider:
    """A released checkpoint, ready to answer typed questions.

    `model` is a checkpoint directory, a Hugging Face repo id or an alias such as
    `v3.0-2b`. Device and precision are picked as the server picks them (CUDA,
    then Apple Silicon, then CPU; a bf16 tower on CUDA, fp32 elsewhere; the scorer
    always fp32). `profile` is the server's `--profile`: "agent" or "server". It
    sets the same RSIJEV_* defaults in this process's environment, and variables
    already set win.
    """

    def __init__(self, model: str, *, device: str | None = None, dtype: str | None = None,
                 batch_size: int = 16, profile: str | None = None,
                 revision: str | None = None):
        from serve.runtime import apply_profile
        from serve.server import load_for_serving, make_scorer
        apply_profile(profile)
        self.served = load_for_serving(model, device=device, dtype=dtype, revision=revision)
        self.name = self.served.name
        self.version = self.served.version
        self.device = self.served.device
        self.dtype = self.served.dtype_name
        self.calibration = self.served.meta.get("calibration", "none")
        self._scorer = make_scorer(self.served, batch_size)

    @classmethod
    def from_scorer(cls, scorer, *, name: str = "rsi-jev") -> "Decider":
        """A Decider over any scorer with the create_app signature. For tests."""
        d = cls.__new__(cls)
        d.served, d.name, d.version = None, name, None
        d.device = d.dtype = None
        d.calibration = "none"
        d._scorer = scorer
        return d

    def request(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """The full response body: {"model", "answers", "usage"}."""
        req = SystemOneRequest.model_validate(
            {"state": state, "model": self.name, "questions": questions})
        answers, usage, _, _ = answer_request(self._scorer, req)
        return {"model": self.name, "answers": answers, "usage": usage}

    def decide(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """The `answers` object /v1/systemone returns for this state and questions."""
        return self.request(state, questions)["answers"]

    def __repr__(self) -> str:
        return (f"Decider({self.name!r}, device={self.device!r}, dtype={self.dtype!r}, "
                f"calibration={self.calibration!r})")
