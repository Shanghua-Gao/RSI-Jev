"""The Jev-compatible HTTP API.

Routes and shapes follow `reference/openjev-sglang`, which implements
https://docs.typesafe.ai/api. The app takes a `scorer` callable so the wire
contract can be tested without a GPU; `scripts/serve.py` supplies the real one.
"""
from __future__ import annotations

import secrets
import threading
import time
from typing import Annotated, Any, Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from rsijev.contract import Question
from serve.wire import (MAX_ANSWERS, MAX_QUESTIONS, RequestError, limits,
                        parse_questions, state_to_text, to_answer)

# A scorer answers every question about one state and reports the prompt tokens
# it encoded: (state_text, questions) -> (list[list[float]] probabilities, tokens).
Scorer = Callable[[str, list[Question]], tuple[list[list[float]], int]]

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


def create_app(scorer: Scorer, *, served_model_name: str, alias: str = "jev-latest",
               api_key: str | None = None, version: str = "v1.0") -> FastAPI:
    app = FastAPI(title="RSI-Jev", version=version,
                  description="A Jev-compatible typed-decision API served by a "
                              "trained decision model. See GET /v1/limits for the "
                              "ways this deployment differs from the reference.")
    # One GPU, one pass at a time. Endpoints are sync so Starlette runs them in a
    # threadpool; the lock keeps concurrent requests from interleaving on the model.
    gpu = threading.Lock()

    def error(message: str, status: int) -> JSONResponse:
        headers = {"Retry-After": "1"} if status in {429, 503, 529} else {}
        return JSONResponse({"error": {"message": message}}, status_code=status,
                            headers=headers)

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
    def systemone(req: SystemOneRequest) -> JSONResponse:
        if req.model not in {alias, served_model_name}:
            raise RequestError(f"Unknown model: {req.model}")
        t0 = time.perf_counter()
        questions = _wire_questions(req)
        state = state_to_text(req.state)
        prepared_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        with gpu:
            probs, prompt_tokens = scorer(state, questions)
        infer_ms = (time.perf_counter() - t1) * 1000

        answers = {q.key: to_answer(q, p) for q, p in zip(questions, probs)}
        body = {"model": req.model,
                "answers": answers,
                # This path generates no tokens: one readout per question, and no
                # warm-up token. The reference reports N+1 because it decodes one
                # token per question plus a prefix warm-up.
                "usage": {"input_tokens": prompt_tokens, "output_tokens": len(questions)}}
        return JSONResponse(body, headers={
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
        return {**limits(), "served_model_name": served_model_name, "version": version}

    @app.get("/health", tags=["Health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/live", tags=["Health"])
    def live() -> dict[str, str]:
        return {"status": "ok"}

    return app
