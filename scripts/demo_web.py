"""The demo: one question, every version, side by side — on your own machine.

    python scripts/demo_web.py                    # http://127.0.0.1:8000
    python scripts/demo_web.py --host 0.0.0.0     # reach it from another machine
    python scripts/demo_web.py --versions 2b      # load just one, if memory is tight

Built for an **NVIDIA DGX Spark**: the GB10's unified memory holds every released
version at once, so switching between them costs nothing and comparing them is one
forward pass each. It also runs on Apple Silicon and on plain CPU.

The document and the questions are both editable, and it re-answers as you type,
because after the document is read once each extra question costs about 12 ms.
Nothing is generated: one forward pass returns a probability for every option.

Answering goes through the project's own `serve.wire` and `serve.infer` -- the same
contract and the same forward pass as the API server in `scripts/serve.py` -- so this
shows what the model does rather than a demo-specific reimplementation.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

# Root before scripts/: scripts/serve.py shadows the serve/ package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent

# Newest last. Adding a release is one line.
VERSIONS = [
    ("v1.0 · 0.8B", "0.8b", "shgao/rsi-jev-v1.0-qwen3.5-0.8b"),
    ("v1.0 · 2B", "2b", "shgao/rsi-jev-v1.0-qwen3.5-2b"),
]


def pick_device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def describe(device: str) -> str:
    import torch

    if device == "cuda":
        return f"{torch.cuda.get_device_name(0)} · {torch.cuda.mem_get_info()[1] / 2**30:.0f} GiB"
    return "Apple Silicon (MPS)" if device == "mps" else "CPU"


def _same_document(a: str, b: str) -> bool:
    """True when two documents are the same, comparing structure where both are JSON."""
    if a.strip() == b.strip():
        return True
    try:
        return json.loads(a) == json.loads(b)
    except (ValueError, TypeError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"],
                    help="default: cuda, else mps, else cpu")
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp32"],
                    help="tower precision (default bf16 on CUDA, fp32 elsewhere); "
                         "the scorer is always fp32")
    ap.add_argument("--versions", nargs="*", default=None,
                    help=f"which to load, from {[k for _, k, _ in VERSIONS]}; default all")
    a = ap.parse_args()

    import torch
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from huggingface_hub import snapshot_download
    from load_release import load_release
    from serve.infer import score_questions_cached
    from serve.wire import RequestError, parse_questions, state_to_text, to_answer

    device = pick_device(a.device)
    # bf16 only on CUDA, where it measured 3.4-6.2x faster for at most 0.003 pooled
    # top-1. MPS and CPU stay fp32: less tested there, and little to gain.
    dtype_name = a.dtype or ("bf16" if device == "cuda" else "fp32")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]

    wanted = [v for v in VERSIONS if a.versions is None or v[1] in a.versions]
    if not wanted:
        raise SystemExit(f"--versions must name some of {[k for _, k, _ in VERSIONS]}")

    print(f"\n  RSI-Jev demo · {describe(device)} · {dtype_name}", flush=True)
    loaded = {}
    for label, _key, repo in wanted:
        t = time.perf_counter()
        loaded[label] = load_release(snapshot_download(repo), device, infer_dtype=dtype)
        print(f"    {label:<14} loaded in {time.perf_counter() - t:4.1f}s", flush=True)

    examples = json.loads((ROOT / "serve" / "examples.json").read_text())
    page = (ROOT / "serve" / "ui.html").read_text()
    gpu = threading.Lock()              # one device, one forward pass at a time

    def answer(label, state, questions):
        model, tok, enc, meta = loaded[label]
        cap = max(meta["spec"]["max_options"], max(len(q.options) for q in questions))
        with gpu:
            t = time.perf_counter()
            preds, tokens = score_questions_cached(model, tok, state, questions, enc,
                                                   device=device, batch_size=8,
                                                   max_options=cap)
            if device == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000
        return preds, tokens, ms

    # Warm the kernels before the page can be opened: the first pass otherwise costs
    # about a second against ~0.1 s for every one after, and a page that re-answers on
    # each keystroke would open with a visible stall.
    warm = parse_questions(examples[0]["questions"])
    for label in loaded:
        for _ in range(2):
            answer(label, state_to_text(examples[0]["state"]), warm)
    print("    warm", flush=True)

    app = FastAPI(title="RSI-Jev demo", docs_url=None, redoc_url=None)

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(page)

    @app.get("/demo/config", include_in_schema=False)
    def config() -> dict:
        return {"hardware": describe(device), "dtype": dtype_name,
                "versions": list(loaded), "examples": examples}

    @app.post("/demo/compare", include_in_schema=False)
    def compare(body: dict) -> JSONResponse:
        state_text = (body.get("state") or "").strip()
        if not state_text:
            return JSONResponse({"error": "Give it a document to read."}, 400)
        chosen = [v for v in loaded if v in (body.get("versions") or list(loaded))]
        if not chosen:
            return JSONResponse({"error": "Pick at least one version."}, 400)
        try:
            questions = parse_questions(body.get("questions") or {})
        except RequestError as e:
            return JSONResponse({"error": str(e)}, e.status)

        # A reference answer exists only for an unedited example. Showing one for a
        # document somebody typed would mean inventing a right answer.
        #
        # Matched on the parsed document, not on the text: the page rebuilds the JSON
        # from its form fields, and Python and JavaScript do not serialise identically
        # -- a float of 32.0 comes back as "32" -- so comparing strings would drop the
        # reference column the moment an example round-tripped through the form.
        gold = next((e["gold"] for e in examples if _same_document(e["state"], state_text)),
                    None)
        state = state_to_text(state_text)
        answers, picks, ms, tokens = {}, {}, {}, 0
        for label in chosen:
            preds, tokens, took = answer(label, state, questions)
            answers[label] = [to_answer(q, list(p.probs))
                              for q, p in zip(questions, preds)]
            picks[label] = [q.options[max(range(len(p.probs)), key=p.probs.__getitem__)]
                            for q, p in zip(questions, preds)]
            ms[label] = round(took, 1)
        return JSONResponse({
            "questions": [{"key": q.key, "mode": q.mode, "options": list(q.options),
                           "instructions": q.instructions,
                           "criteria": dict(q.criteria) if q.criteria else None}
                          for q in questions],
            "answers": answers, "picks": picks,
            "gold": gold, "ms": ms, "tokens": tokens,
        })

    print(f"  open http://{a.host}:{a.port}\n", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
