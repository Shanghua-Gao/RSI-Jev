"""`rsi-jev demo`: the playground page (serve/ui.html) on the served path.

    rsi-jev demo v4.0-vl-2b                      # http://127.0.0.1:8000
    rsi-jev demo v4.0-vl-2b v3.0-2b --host 0.0.0.0

Each model is loaded exactly as `rsi-jev serve` loads it and answered by the
server's own scorer (`serve.server.make_scorer`), so the page shows what the API
returns, images included. `GET /v1/limits` says whether any loaded model takes
images; the page shows its image input only then. A request with images is
answered by the models that take them, and the rest are listed as skipped.

scripts/demo_web.py serves the same page for the text releases side by side; it
has no /v1/limits, so the page hides the image input there.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

HERE = Path(__file__).resolve().parent


def _same_document(a: str, b: str) -> bool:
    """True when two documents are the same, comparing structure where both are JSON."""
    if a.strip() == b.strip():
        return True
    try:
        return json.loads(a) == json.loads(b)
    except (ValueError, TypeError):
        return False


def create_demo_app(models: Sequence[tuple[str, Callable, dict]], *, hardware: str,
                    dtype: str, examples: list[dict] | None = None, page: str | None = None):
    """The page and its three routes over `models`: (label, scorer, image limits).

    `scorer` has the create_app signature: (state, questions[, images]) ->
    (probabilities, prompt tokens)."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from serve.images import parse_images
    from serve.wire import RequestError, limits, parse_questions, state_to_text, to_answer

    examples = examples if examples is not None else json.loads((HERE / "examples.json").read_text())
    page = page if page is not None else (HERE / "ui.html").read_text()
    by_label = {label: (scorer, lim or {"supported": False}) for label, scorer, lim in models}
    vision = next((lim for _, lim in by_label.values() if lim.get("supported")),
                  {"supported": False})
    lock = threading.Lock()              # one device, one forward pass at a time
    app = FastAPI(title="RSI-Jev demo", docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(page)

    @app.get("/demo/config", include_in_schema=False)
    def config() -> dict:
        return {"hardware": hardware, "dtype": dtype, "versions": list(by_label),
                "examples": examples,
                "images": {k: bool(lim.get("supported")) for k, (_, lim) in by_label.items()}}

    @app.get("/v1/limits", include_in_schema=False)
    def limits_route() -> dict[str, Any]:
        return {**limits(), "images": vision}

    @app.post("/demo/compare", include_in_schema=False)
    def compare(body: dict) -> JSONResponse:
        state_text = (body.get("state") or "").strip()
        if not state_text:
            return JSONResponse({"error": "Give it a document to read."}, 400)
        chosen = [v for v in by_label if v in (body.get("versions") or list(by_label))]
        if not chosen:
            return JSONResponse({"error": "Pick at least one version."}, 400)
        urls = body.get("images") or []
        try:
            questions = parse_questions(body.get("questions") or {})
            state = state_to_text(state_text)
            images = parse_images(urls, state) if urls else []
        except RequestError as e:
            return JSONResponse({"error": str(e)}, e.status)
        skipped = {}
        if images:
            skipped = {v: "text-only: does not take images" for v in chosen
                       if not by_label[v][1].get("supported")}
            chosen = [v for v in chosen if v not in skipped]
            if not chosen:
                return JSONResponse({"error": "None of the chosen versions takes images."}, 422)
        # A reference answer exists only for an unedited example: same document, same images.
        gold = next((e["gold"] for e in examples
                     if _same_document(e["state"], state_text) and (e.get("images") or []) == urls),
                    None)
        answers, picks, ms, tokens = {}, {}, {}, 0
        for label in chosen:
            scorer = by_label[label][0]
            args = (state, questions, images) if images else (state, questions)
            with lock:
                t = time.perf_counter()
                try:
                    probs, tokens = scorer(*args)
                except RequestError as e:
                    return JSONResponse({"error": str(e)}, e.status)
                ms[label] = round((time.perf_counter() - t) * 1000, 1)
            answers[label] = [to_answer(q, p) for q, p in zip(questions, probs)]
            picks[label] = [q.options[max(range(len(p)), key=p.__getitem__)]
                            for q, p in zip(questions, probs)]
        return JSONResponse({
            "questions": [{"key": q.key, "mode": q.mode, "options": list(q.options),
                           "instructions": q.instructions,
                           "criteria": dict(q.criteria) if q.criteria else None}
                          for q in questions],
            "answers": answers, "picks": picks, "skipped": skipped,
            "gold": gold, "ms": ms, "tokens": tokens,
        })

    return app


def add_demo_args(ap) -> None:
    ap.add_argument("models", nargs="*", default=["v4.0-vl-2b"],
                    help="checkpoint directories, Hugging Face repo ids or aliases "
                         "(default: v4.0-vl-2b)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None, help="default: cuda, else mps, else cpu")
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp32"])
    ap.add_argument("--profile", default=None, choices=["agent", "server"])


def run(a) -> int:
    import torch
    import uvicorn
    from serve.runtime import apply_profile
    from serve.server import load_for_serving, make_scorer, warm_up
    apply_profile(a.profile)
    models = []
    for ref in a.models:
        t = time.perf_counter()
        s = load_for_serving(ref, device=a.device, dtype=a.dtype)
        scorer = make_scorer(s)
        warm_up(scorer, images=s.prep is not None)
        lim = s.image_limits
        print(f"  {s.name:<40} loaded in {time.perf_counter() - t:4.1f}s · images "
              f"{'yes' if lim.get('supported') else 'no'}"
              + (f" ({lim['reason']})" if lim.get("reason") else ""), flush=True)
        models.append((s.name, scorer, lim))
        device, dtype = s.device, s.dtype_name
    hardware = (f"{torch.cuda.get_device_name(0)}" if device == "cuda"
                else "Apple Silicon (MPS)" if device == "mps" else "CPU")
    app = create_demo_app(models, hardware=hardware, dtype=dtype)
    print(f"  open http://{a.host}:{a.port}\n", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0
