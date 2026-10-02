"""The Jev-compatible HTTP API.

Routes and shapes follow `reference/openjev-sglang`, which implements
https://docs.typesafe.ai/api. The app takes a `scorer` callable so the wire
contract can be tested without a GPU; `serve/server.py` supplies the real one.
"""
from __future__ import annotations

import asyncio
import math
import secrets
import time
from itertools import chain
from typing import Annotated, Any, Callable, Literal, Sequence
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from rsijev.contract import Question
from rsijev.encode import InputTooLong
from serve.batcher import CallRunner, GpuWorker
from serve.images import MAX_IMAGES, parse_images
from serve.wire import (MAX_ANSWERS, MAX_QUESTIONS, RequestError, limits,
                        parse_questions, state_to_text, to_answer)

try:                                     # the `http` extra; the stdlib encoder otherwise
    import orjson
except ImportError:                      # pragma: no cover - depends on the install
    orjson = None


class FastJSONResponse(JSONResponse):
    """JSONResponse, encoded by orjson when it is installed (5-10x faster than the
    stdlib encoder on an answer body). The bytes differ only in how a float is
    spelled (1e-05 against 1e-5); every value parses back to the same double."""

    def render(self, content: Any) -> bytes:
        if orjson is None:
            return super().render(content)
        return orjson.dumps(content)

# A scorer answers every question about one state and reports the prompt tokens
# it encoded: (state_text, questions) -> (list[list[float]] probabilities, tokens).
# A request with images calls it as (state_text, questions, images), images being
# validated PIL images; a scorer that takes none is never called that way when
# create_app is told the model has no image support.
Scorer = Callable[..., tuple[list[list[float]], int]]

JsonValue = Any


class StrictModel(BaseModel):
    # Unknown fields are rejected and types are not coerced, as in the reference.
    model_config = ConfigDict(extra="forbid", strict=True)


Content = str | dict[str, JsonValue] | list[JsonValue]


class NoulCriteria(StrictModel):
    # The wire keys really are "true"/"false"; there is no populate_by_name.
    yes: str = Field(default="Yes", alias="true")
    no: str = Field(default="No", alias="false")


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: Content
    criteria: NoulCriteria = Field(default_factory=NoulCriteria)


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Content
    criteria: dict[str, str | None] = Field(min_length=2, max_length=MAX_ANSWERS)


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Content
    criteria: list[str] = Field(min_length=2, max_length=MAX_ANSWERS)


QuestionModel = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion,
                          Field(discriminator="type")]


class SystemOneRequest(StrictModel):
    state: Content
    model: str = Field(min_length=1)
    questions: dict[str, QuestionModel] = Field(min_length=1, max_length=MAX_QUESTIONS)
    # An extension to the Jev request, in the shape imajev's Jev-style payloads use:
    # base64 data URLs, referenced from the state by `<image>` markers
    # (serve/images.py). Omitted, null or empty, the request is a text request.
    images: list[str] | None = Field(default=None, max_length=MAX_IMAGES)


def _wire_questions(req: SystemOneRequest) -> list[Question]:
    """Back to plain dicts, so `serve.wire` stays the single mapping rule."""
    out = []
    for key, q in req.questions.items():
        spec: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
        if isinstance(q, NoulQuestion):
            spec["criteria"] = {"true": q.criteria.yes, "false": q.criteria.no}
        else:
            spec["criteria"] = q.criteria
        out.append(spec)
    return parse_questions(dict(zip(req.questions, out)))


def prepare(req: SystemOneRequest) -> tuple[list[Question], str, list]:
    """A validated request -> (questions, state text, images). Raises RequestError
    (422). Images are decoded and checked here, before the request queues for the
    model, so a bad image never waits for it; a text request gets []."""
    questions, state = _wire_questions(req), state_to_text(req.state)
    return questions, state, parse_images(req.images, state) if req.images else []


def finish(questions: list[Question], probs, prompt_tokens: int):
    """Probabilities -> (answers, usage)."""
    # A non-finite probability used to fail inside the stdlib JSON encoder
    # (allow_nan=False) as a 500; orjson would write null instead. Keep the 500.
    if not all(map(math.isfinite, chain.from_iterable(probs))):
        raise ValueError("the model returned a non-finite probability")
    answers = {q.key: to_answer(q, p) for q, p in zip(questions, probs)}
    # This path generates no tokens: one readout per question, and no warm-up
    # token. The reference reports N+1 because it decodes one token per question
    # plus a prefix warm-up.
    usage = {"input_tokens": prompt_tokens, "output_tokens": len(questions)}
    return answers, usage


def answer_request(scorer: Scorer, req: SystemOneRequest, *, lock=None):
    """One validated request -> (answers, usage, prepare_ms, infer_ms).

    `serve.decider.Decider` calls this, and the route runs the same `prepare` and
    `finish` around the same scorer, so the Python API and the HTTP API cannot give
    different answers to the same request."""
    t0 = time.perf_counter()
    questions, state, images = prepare(req)
    # A text request calls the scorer exactly as before images existed.
    args = (state, questions, images) if images else (state, questions)
    prepared_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    if lock is None:
        probs, prompt_tokens = scorer(*args)
    else:
        with lock:
            probs, prompt_tokens = scorer(*args)
    infer_ms = (time.perf_counter() - t1) * 1000

    answers, usage = finish(questions, probs, prompt_tokens)
    return answers, usage, prepared_ms, infer_ms


def create_app(scorer: Scorer, *, served_model_name: str, alias: str = "jev-latest",
               api_key: str | None = None, version: str = "v2.1",
               calibration: str = "none", accept_models: Sequence[str] = (),
               images: dict[str, Any] | None = None) -> FastAPI:
    # Apps built on Jev often pin a Jev version ("jev-1.13.0") in their requests.
    # `accept_models` lets a deployment answer those names without code changes in
    # the app. The response still names THIS model: echoing a borrowed name would
    # tell the client it was answered by a model it was not.
    # `images` is what /v1/limits reports about image input (serve.images.image_limits);
    # None means a text-only model, and a request carrying images gets a 422.
    images = images or {"supported": False}
    own = {alias, served_model_name}
    borrowed = set(accept_models) - own
    app = FastAPI(title="RSI-Jev", version=version,
                  description="A Jev-compatible typed-decision API served by a "
                              "trained decision model. See GET /v1/limits for the "
                              "ways this deployment differs from the reference.")
    # One GPU, one thread driving it. The route is async: it parses and validates on
    # the event loop, plans (tokenizes) in the threadpool, and queues the plan for
    # the worker thread, which runs requests one at a time in arrival order -- or,
    # given a worker with a batch window, several at once (serve/batcher.py).
    worker = scorer if isinstance(scorer, GpuWorker) else GpuWorker(CallRunner(scorer))
    plan_off_loop = not isinstance(worker.runner, CallRunner)

    def error(message: str, status: int) -> JSONResponse:
        headers = {"Retry-After": "1"} if status in {429, 503, 529} else {}
        return JSONResponse({"error": {"message": message}}, status_code=status,
                            headers=headers)

    @app.exception_handler(InputTooLong)
    async def _too_long(_request: Request, exc: InputTooLong) -> JSONResponse:
        return error(f"{exc}; the input is not truncated, shorten it or raise "
                     "RSIJEV_MAX_INPUT_TOKENS", 422)

    @app.exception_handler(RequestError)
    async def _request_error(_request: Request, exc: RequestError) -> JSONResponse:
        return error(str(exc), exc.status)

    @app.middleware("http")
    async def _auth(request: Request, call_next):
        # Auth is off unless a key is configured. The health routes stay open so a
        # load balancer can probe them, exactly as the reference does.
        if api_key and request.url.path not in {"/health", "/health/live"}:
            presented = request.headers.get("authorization", "")
            if not secrets.compare_digest(presented, f"Bearer {api_key}"):
                return JSONResponse({"error": {"message": "Missing or invalid API key"}},
                                    status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)

    @app.post("/v1/systemone", tags=["System One"])
    async def systemone(req: SystemOneRequest) -> FastJSONResponse:
        if req.model not in own | borrowed:
            raise RequestError(f"Unknown model: {req.model}")
        if req.images and not images.get("supported"):
            raise RequestError(images.get("reason") or
                               f"{served_model_name} is a text-only model; it does not take images")
        t0 = time.perf_counter()
        # Decoding up to four 20 MiB images is too slow for the event loop.
        questions, state, pics = (await run_in_threadpool(prepare, req) if req.images
                                  else prepare(req))
        prepared_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        if plan_off_loop:
            plan = await run_in_threadpool(worker.plan, state, questions, pics)
        else:
            plan = worker.plan(state, questions, pics)
        probs, prompt_tokens = await asyncio.wrap_future(worker.enqueue(plan))
        infer_ms = (time.perf_counter() - t1) * 1000
        answers, usage = finish(questions, probs, prompt_tokens)
        body = {"model": served_model_name if req.model in borrowed else req.model,
                "answers": answers, "usage": usage}
        return FastJSONResponse(body, headers={
            "x-typesafe-request-id": uuid4().hex,
            "x-rsijev-model": served_model_name,
            "x-rsijev-version": version,
            "Server-Timing": f"prepare;dur={prepared_ms:.2f}, infer;dur={infer_ms:.2f}",
        })

    @app.get("/v1/models", tags=["Models"])
    def models() -> dict[str, Any]:
        names = list(dict.fromkeys([alias, served_model_name]))
        return {"object": "list",
                "data": [{"id": n, "object": "model", "owned_by": "rsi-jev"} for n in names],
                "models": [{"name": n, "description": f"RSI-Jev {version} typed decisions"}
                           for n in names]}

    @app.get("/v1/limits", tags=["Limits"])
    def limits_route() -> dict[str, Any]:
        # `calibration` is part of what a client needs to know: from v2.0 a checkpoint
        # may ship a fitted calibration, and then the probabilities in an answer are
        # rescaled. The chosen option is the same either way, the numbers are not.
        return {**limits(), "served_model_name": served_model_name, "version": version,
                "calibration": calibration, "accepted_model_names": sorted(borrowed),
                "images": images}

    @app.get("/health", tags=["Health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/live", tags=["Health"])
    def live() -> dict[str, str]:
        return {"status": "ok"}

    return app
