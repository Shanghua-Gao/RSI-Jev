"""Load a release for serving, and run the HTTP server.

    rsi-jev serve shgao/rsi-jev-v3.0-qwen3.5-2b --port 8000
    python scripts/serve.py --ckpt path/to/ckpt --host 0.0.0.0 --port 8000

The checkpoint is a directory, a Hugging Face repo id or an alias such as
`v3.0-2b` (serve/release.py). The served model name defaults to the checkpoint's
own name; `jev-latest` is accepted as an alias, as in the reference. Set --api-key
(or RSIJEV_API_KEY) to require a bearer token on everything except the health
routes.

`load_for_serving` and `make_scorer` are shared with `serve.decider.Decider`, so
the Python API runs exactly what the server runs.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from serve.release import checkpoint_name, load_release, release_version, resolve_ckpt
from serve.runtime import (PROFILES, apply_profile, best_device, default_dtype_name,
                           startup_lines)


@dataclass
class Served:
    """A loaded release and everything the server reports about it."""
    model: Any
    tok: Any
    enc: Any
    meta: dict
    device: str
    dtype_name: str
    path: Path
    name: str
    version: str | None
    applied: list[str] = field(default_factory=list)


def load_for_serving(ref, *, device: str | None = None, dtype: str | None = None,
                     revision: str | None = None) -> Served:
    """Resolve `ref`, load it the way the server does, and apply the opt-in speed
    paths the environment asks for (RSIJEV_COMPILE; the document cache needs
    nothing here)."""
    import torch
    from serve.accel import apply_env
    from serve.runtime import keep_fused_kernels_off
    path = resolve_ckpt(ref, revision=revision)
    device = device or best_device(torch)
    keep_fused_kernels_off(device)
    dtype_name = dtype or default_dtype_name(device)
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    model, tok, enc, meta = load_release(path, device, infer_dtype=torch_dtype)
    applied = apply_env(model)
    name = checkpoint_name(ref, path)
    return Served(model, tok, enc, meta, device, dtype_name, path, name,
                  release_version(name), applied)


def make_scorer(s: Served, batch_size: int = 16):
    """(state_text, questions) -> (probabilities, prompt tokens), as create_app wants."""
    from serve.infer import score_questions_cached
    spec = s.meta["spec"]

    def scorer(state: str, questions):
        preds, tokens = score_questions_cached(s.model, s.tok, state, questions, s.enc,
                                               device=s.device, batch_size=batch_size,
                                               max_options=max(spec["max_options"],
                                                               max(len(q.options) for q in questions)))
        return [list(p.probs) for p in preds], tokens

    return scorer


def warm_up(scorer) -> float:
    """Run two throwaway requests so kernel JIT (fla's Triton kernels) and most of
    torch.compile happen before the first real request.

    One short question (the state read per question), and four questions on a
    long state (read once). A request of a shape not seen yet can still compile
    under --profile server. Returns seconds spent."""
    from serve.wire import parse_questions
    t = time.perf_counter()
    q = {"a": {"type": "noul", "instructions": "Is this a test?"},
         "b": {"type": "choice", "instructions": "Which?", "criteria": {"x": "X", "y": "Y"}},
         "c": {"type": "score", "instructions": "How much?", "criteria": ["Low", "High"]},
         "d": {"type": "noul", "instructions": "Is it long?"}}
    scorer("Warm-up.", parse_questions({"a": q["a"]}))
    scorer("A warm-up document. " * 120, parse_questions(q))
    return time.perf_counter() - t


def _env_float(name: str) -> float | None:
    v = os.environ.get(name, "").strip()
    return float(v) if v else None


def add_serve_args(ap: argparse.ArgumentParser, *, positional: bool) -> None:
    if positional:
        ap.add_argument("model", nargs="?", default=None,
                        help="checkpoint directory, Hugging Face repo id "
                             "(shgao/rsi-jev-v3.0-qwen3.5-2b) or alias (v3.0-2b)")
    ap.add_argument("--ckpt", default=None, help="same as MODEL" if positional else
                    "checkpoint directory, Hugging Face repo id or alias (v3.0-2b)")
    ap.add_argument("--revision", default=None, help="Hugging Face revision to download")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--profile", default=None, choices=sorted(PROFILES),
                    help="agent: keep document reads across requests (RSIJEV_DOC_CACHE=1). "
                         "server: torch.compile, ~40 s at startup (RSIJEV_COMPILE=1). "
                         "Environment variables that are already set win.")
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--alias", default="jev-latest")
    ap.add_argument("--accept-model", action="append", default=[], metavar="NAME",
                    help="also answer requests naming this model (repeatable), e.g. "
                         "jev-1.13.0 for an app that pins a Jev version; responses still "
                         "carry this server's own model name")
    ap.add_argument("--api-key", default=os.environ.get("RSIJEV_API_KEY"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp32"],
                    help="tower precision (default bf16 on CUDA, fp32 elsewhere). "
                         "The scorer is always fp32. Evaluation always uses fp32.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--batch-window-ms", type=float,
                    default=_env_float("RSIJEV_BATCH_WINDOW_MS"),
                    help="micro-batching: run the questions of requests that arrive "
                         "within this many ms of each other in one forward pass "
                         "(0: only those already waiting). Off by default: it moves "
                         "probabilities slightly. Also RSIJEV_BATCH_WINDOW_MS.")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the throwaway requests run before serving")
    ap.add_argument("--version", default=None,
                    help="the release being served, as GET /v1/limits reports it. "
                         "Read off the checkpoint's own name when it carries one.")


def serve(a: argparse.Namespace) -> int:
    ref = a.ckpt or getattr(a, "model", None)
    if not ref:
        raise SystemExit("give a checkpoint: a directory, a Hugging Face repo id such as "
                         "shgao/rsi-jev-v3.0-qwen3.5-2b, or an alias such as v3.0-2b")
    for k, v in apply_profile(a.profile).items():
        print(f"profile {a.profile}: {k}={v}", flush=True)

    import torch
    import uvicorn
    from serve.app import create_app

    s = load_for_serving(ref, device=a.device, dtype=a.dtype, revision=a.revision)
    for applied in s.applied:
        print(f"speed path: {applied}", flush=True)
    from serve.batcher import model_worker
    from serve.infer import speed_options
    if a.batch_window_ms is None:
        a.batch_window_ms = _env_float("RSIJEV_BATCH_WINDOW_MS")   # set by --profile
    scorer = model_worker(s, batch_size=a.batch_size, window_ms=a.batch_window_ms)
    if not a.no_warmup:
        # The first call JIT-compiles fla's Triton kernels (~19 s measured on a GB10
        # after a fresh install; cached afterwards) and, with --profile server,
        # torch.compile's graphs. Do it before accepting requests, not in one.
        print(f"warm-up: {warm_up(scorer):.1f} s", flush=True)
    name = a.served_model_name or s.name
    # /v1/limits reports which release is answering, so take it from the checkpoint
    # rather than from whatever this tree was cut for: serving a v1.0 checkpoint out of
    # a v2.1 checkout would otherwise announce v2.1, and a client tuning a threshold
    # per release would be told the wrong one.
    served_version = a.version or s.version
    app = create_app(scorer, served_model_name=name, alias=a.alias, api_key=a.api_key,
                     accept_models=a.accept_model,
                     calibration=s.meta.get("calibration", "none"),
                     **({"version": served_version} if served_version else {}))
    for line in startup_lines(torch=torch, device=s.device, dtype_name=s.dtype_name,
                              model=s.model, profile=a.profile, applied=s.applied):
        print(line, flush=True)
    opt = speed_options()
    print("micro-batching " + ("off" if a.batch_window_ms is None else
                               f"on, window {a.batch_window_ms:g} ms")
          + f"; rows sorted by length {'on' if opt['sort'] else 'off'}"
          + f"; option slots trimmed {'on' if opt['trim_options'] else 'off'}", flush=True)
    print(f"serving {ref} as {name!r} (alias {a.alias!r}) on {a.host}:{a.port}; "
          f"base {s.meta['base_model']}, calibration {s.meta.get('calibration', 'none')}",
          flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Serve an RSI-Jev release behind the "
                                             "Jev-compatible API.")
    add_serve_args(ap, positional=True)
    return serve(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
