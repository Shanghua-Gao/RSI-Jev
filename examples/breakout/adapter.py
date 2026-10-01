"""jev-visual's `/v1/judge` API, answered by RSI-Jev v4.0-VL through `rsi-jev serve`.

hr98w/jev-visual's browser demos post a screenshot and one typed question to the
same-origin `/v1/judge`. This adapter serves their unmodified demo pages and turns each
`/v1/judge` call into one `POST /v1/systemone` to a running `rsi-jev serve`, then
answers in the shape their pages read (`answers.<key>.choice`, `.probabilities`,
`metrics.elapsed_ms`). It changes nothing in the game and adds no game state: the model
sees only the screenshot, their fixed instructions and their option texts.

    rsi-jev serve v4.0-vl-2b --port 8000
    python examples/breakout/adapter.py --game path/to/jev-visual --upstream http://127.0.0.1:8000 --port 8788
    # then open http://127.0.0.1:8788/demo/breakout/
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request


def to_systemone(body: dict[str, Any], model: str = "jev-latest") -> dict[str, Any]:
    """A jev-visual judge request -> a /v1/systemone request body.

    Their question carries `scoring` ("label"), which the Jev wire format does not
    have; it is dropped. A choice whose criteria is a list becomes keys with no
    description. The state defaults to the image alone."""
    image = body.get("image")
    if not isinstance(image, str) or not image.startswith("data:image"):
        raise ValueError("image must be a base64 data URL")
    questions = {}
    for key, q in body["questions"].items():
        typ = q.get("type", "choice")
        spec: dict[str, Any] = {"type": typ, "instructions": q["instructions"]}
        crit = q.get("criteria")
        if typ == "choice":
            spec["criteria"] = crit if isinstance(crit, dict) else {o: None for o in crit}
        elif typ == "score":
            spec["criteria"] = list(crit)
        elif typ == "noul":
            if crit:
                spec["criteria"] = crit
        else:
            raise ValueError(f"questions.{key}: unknown type {typ!r}")
        questions[key] = spec
    return {"model": model, "state": body.get("state") or "<image>", "questions": questions, "images": [image]}


def to_judge(answers: dict[str, Any], elapsed_ms: float, model: str) -> dict[str, Any]:
    """/v1/systemone answers -> the response their pages read."""
    out = {}
    for key, a in answers.items():
        if a["type"] == "noul":
            p = a["noul"]
            out[key] = {"type": "noul", "noul": p, "probabilities": {"yes": p, "no": 1 - p}}
        elif a["type"] == "score":
            pr = a["probabilities"]
            out[key] = {"type": "score", "score": a["score"], "probabilities": pr, "choice": max(pr, key=pr.get)}
        else:
            pr = a["probabilities"]
            out[key] = {"type": "choice", "probabilities": pr, "choice": a["choice"],
                        "concentration": max(pr.values())}
    return {"model": model, "answers": out,
            "probability_semantics": "RSI-Jev option probabilities, as served (release calibration applied)",
            "metrics": {"elapsed_ms": elapsed_ms, "generated_tokens": 0,
                        "decisions_per_second": len(out) * 1000 / max(elapsed_ms, 1e-6)}}


def create_app(upstream: str, game_dir: str | Path | None = None, post=None) -> FastAPI:
    """`post(body) -> response dict` defaults to an HTTP call to `upstream`; tests pass a stub."""
    upstream = upstream.rstrip("/")

    def http_post(body):
        req = urllib.request.Request(upstream + "/v1/systemone", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except urllib.error.HTTPError as e:            # a 422 from the server keeps its message
            raise HTTPException(e.code, e.read().decode(errors="replace")) from None

    post = post or http_post
    lock = threading.Lock()                            # one decision at a time, like their own server
    app = FastAPI(title="jev-visual adapter for rsi-jev serve")
    log: list[dict] = []

    @app.get("/health")
    def health():
        return {"ready": True, "upstream": upstream}

    @app.post("/v1/judge")
    async def judge(req: Request):
        try:
            body = to_systemone(await req.json())
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(422, str(e)) from None
        from starlette.concurrency import run_in_threadpool

        def call():
            with lock:
                t = time.perf_counter()
                out = post(body)
                return out, (time.perf_counter() - t) * 1000
        out, ms = await run_in_threadpool(call)
        log.append({"t": time.time(), "ms": ms})
        return to_judge(out["answers"], ms, out.get("model", "rsi-jev"))

    @app.get("/latency")
    def latency():
        return log

    if game_dir is not None:
        from fastapi.staticfiles import StaticFiles
        demo = Path(game_dir) / "demo"
        if not (demo / "breakout" / "index.html").exists():
            raise SystemExit(f"{demo}/breakout/index.html not found: pass the jev-visual checkout with --game")
        app.mount("/demo", StaticFiles(directory=str(demo), html=True), name="demo")
    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--game", required=True, help="a checkout of hr98w/jev-visual")
    ap.add_argument("--upstream", default="http://127.0.0.1:8000", help="URL of `rsi-jev serve`")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8788)
    a = ap.parse_args(argv)
    import uvicorn
    uvicorn.run(create_app(a.upstream, a.game), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
