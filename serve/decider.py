"""The HTTP API's answers, in-process.

    from rsijev import Decider
    d = Decider("shgao/rsi-jev-v3.0-qwen3.5-2b")
    d.decide("I was charged twice. Please refund.",
             {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}})
    d.decide("<image> Is this receipt paid?", {...}, images=["receipt.png"])   # v4.0-VL on

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
    `v3.0-2b`. `backend="mlx"` runs an MLX checkpoint (Apple Silicon, scripts/convert_mlx.py)
    with the same answers and fields; `dtype` there is "bf16" (default) or "fp32" compute. Device and precision are picked as the server picks them (CUDA,
    then Apple Silicon, then CPU; a bf16 tower on CUDA, fp32 elsewhere; the scorer
    always fp32). `profile` is the server's `--profile`: "agent" or "server". It
    sets the same RSIJEV_* defaults in this process's environment, and variables
    already set win.
    """

    def __init__(self, model: str, *, device: str | None = None, dtype: str | None = None,
                 batch_size: int = 16, profile: str | None = None,
                 revision: str | None = None, backend: str = "torch"):
        from serve.runtime import apply_profile
        apply_profile(profile)
        if backend == "mlx":
            # Apple Silicon: `model` is an MLX checkpoint (scripts/convert_mlx.py); no torch
            from rsijev.mlx.serving import load_for_serving, make_scorer
            self.served = load_for_serving(model, revision=revision, dtype=dtype)
        elif backend == "torch":
            from serve.server import load_for_serving, make_scorer
            self.served = load_for_serving(model, device=device, dtype=dtype, revision=revision)
        else:
            raise ValueError(f"backend must be 'torch' or 'mlx', got {backend!r}")
        self.backend = backend
        self.name = self.served.name
        self.version = self.served.version
        self.device = self.served.device
        self.dtype = self.served.dtype_name
        self.calibration = self.served.meta.get("calibration", "none")
        self._scorer = make_scorer(self.served, batch_size)

    @classmethod
    def from_scorer(cls, scorer, *, name: str = "rsi-jev",
                    images: dict[str, Any] | None = None) -> "Decider":
        """A Decider over any scorer with the create_app signature. For tests.
        `images` is the image-limits dict to report (None: text-only)."""
        d = cls.__new__(cls)
        d.served, d.name, d.version = None, name, None
        d.device = d.dtype = None
        d.calibration = "none"
        d._scorer = scorer
        d._image_limits = images or {"supported": False}
        return d

    @property
    def image_limits(self) -> dict[str, Any]:
        """What /v1/limits reports under `images` for this checkpoint."""
        if self.served is None:
            return getattr(self, "_image_limits", {"supported": False})
        return self.served.image_limits

    def request(self, state: Any, questions: dict[str, Any],
                images: list | None = None) -> dict[str, Any]:
        """The full response body: {"model", "answers", "usage"}, plus "truncated"
        for a model served with the long-context encoder, as the HTTP API gives it.

        `images` are what the HTTP request's `images` carries: data URLs. For
        convenience a PIL image, a file path or raw bytes are also taken; each is
        turned into a data URL first, so it goes through the server's own checks."""
        body = {"state": state, "model": self.name, "questions": questions}
        if images:
            from serve.images import to_data_url
            from serve.wire import RequestError
            if not self.image_limits.get("supported"):
                raise RequestError(self.image_limits.get("reason") or
                                   f"{self.name} is a text-only model; it does not take images")
            body["images"] = [to_data_url(im) for im in images]
        req = SystemOneRequest.model_validate(body)
        report: dict = {}
        answers, usage, _, _ = answer_request(self._scorer, req, report=report)
        body = {"model": self.name, "answers": answers, "usage": usage}
        if report.get("truncated") is not None:
            body["truncated"] = report["truncated"]
        return body

    def decide(self, state: Any, questions: dict[str, Any],
               images: list | None = None) -> dict[str, Any]:
        """The `answers` object /v1/systemone returns for this state, questions and
        (optionally) images."""
        return self.request(state, questions, images)["answers"]

    def __repr__(self) -> str:
        return (f"Decider({self.name!r}, device={self.device!r}, dtype={self.dtype!r}, "
                f"calibration={self.calibration!r})")
