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
import dataclasses
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
    # Image support (v4.0-VL on): the image processor and the encoder config for image
    # requests, or None for a text-only checkpoint. `vision_error` says why a
    # checkpoint trained with images is served text-only here (a missing extra).
    prep: Any = None
    venc: Any = None
    vision_error: str | None = None

    @property
    def image_limits(self) -> dict:
        from serve.images import image_limits
        out = image_limits(self.meta.get("vision") if self.prep is not None else None)
        if self.vision_error:
            out["reason"] = self.vision_error
        return out


def load_for_serving(ref, *, device: str | None = None, dtype: str | None = None,
                     revision: str | None = None, max_length: int | None = None,
                     truncate: str | None = None, fixed_exit: bool | None = None,
                     adaptive: str | None = None) -> Served:
    """Resolve `ref`, load it the way the server does, and apply the opt-in speed
    paths the environment asks for (RSIJEV_COMPILE; the document cache needs
    nothing here).

    `max_length` and `truncate` override the input cap and the over-cap policy the
    checkpoint's meta.json gives (serve.release.load_release; RSIJEV_MAX_LENGTH and
    RSIJEV_TRUNCATE do the same). Left unset, every release is served as it was
    trained. `adaptive` (auto / on / off, RSIJEV_ADAPTIVE) picks which requests of a
    release with aux exits use adaptive exit; `fixed_exit` (RSIJEV_FIXED_EXIT) is an alias
    for off (serve.release.adaptive_mode)."""
    import torch
    from serve.accel import apply_env
    from serve.runtime import keep_fused_kernels_off
    path = resolve_ckpt(ref, revision=revision)
    device = device or best_device(torch)
    keep_fused_kernels_off(device)
    dtype_name = dtype or default_dtype_name(device)
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    model, tok, enc, meta = load_release(path, device, infer_dtype=torch_dtype,
                                         max_length=max_length, truncate=truncate,
                                         fixed_exit=fixed_exit, adaptive=adaptive)
    # Serve whole inputs: unless a cap or a cut policy was asked for explicitly
    # (--max-length / --truncate, RSIJEV_MAX_LENGTH / RSIJEV_TRUNCATE), nothing is cut
    # and a question past RSIJEV_MAX_INPUT_TOKENS gets a 422. load_release itself keeps
    # the training cap and its cut, which the benchmark scripts reproduce numbers with.
    explicit = (max_length is not None or truncate is not None
                or os.environ.get("RSIJEV_MAX_LENGTH", "").strip()
                or os.environ.get("RSIJEV_TRUNCATE", "").strip())
    if not explicit:
        from serve.wire import MAX_INPUT_TOKENS
        enc = dataclasses.replace(enc, max_length=MAX_INPUT_TOKENS, truncate="none")
        meta["serving"] = {**(meta.get("serving") or {}),
                           "max_length": MAX_INPUT_TOKENS, "truncate": "none"}
    applied = apply_env(model)
    name = checkpoint_name(ref, path)
    s = Served(model, tok, enc, meta, device, dtype_name, path, name,
               release_version(name), applied)
    if meta.get("vision"):
        from rsijev.vision import ImagePrep, VisionConfig
        v = meta["vision"]
        try:
            import PIL           # noqa: F401  the image processor needs both
            import torchvision   # noqa: F401
        except ImportError as e:
            s.vision_error = (f"image requests need Pillow and torchvision ({e}); "
                              f'install them with: pip install "rsi-jev[vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev"')
        else:
            # The vision tower's rotary embedding as one fused, bit-exact kernel
            # (serve/kernels.py); RSIJEV_VISION_ROPE=0 turns it off.
            from serve.kernels import install
            install()
            src = meta.get("weights_source") or meta["base_model"]
            s.prep = ImagePrep(src, VisionConfig(
                image_token_budget=v["image_token_budget"],
                min_tokens_per_image=v["min_tokens_per_image"]),
                revision=None if src != meta["base_model"] else v.get("revision"))
        # The budget is added to the text length, so no state is cut shorter
        # because it came with an image.
        s.venc = dataclasses.replace(enc, max_length=enc.max_length + v["image_token_budget"],
                                     option_order="canonical")
    return s


def _env_int(name: str) -> int | None:
    v = os.environ.get(name, "").strip()
    return int(v) if v else None


def make_scorer(s: Served, batch_size: int = 16):
    """(state_text, questions[, images]) -> (probabilities, prompt tokens), as
    create_app wants. `images` are validated PIL images (serve/images.py)."""
    from serve.images import state_too_long, text_only_error
    from serve.infer import (plan_image_request, plan_request, score_image_planned,
                             score_planned)
    spec = s.meta["spec"]

    def scorer(state: str, questions, images=None, effort=None, threshold=None):
        """score_questions_cached / score_image_questions_cached, split into plan and
        run so the plan's reports reach the response. A third element ({"truncated",
        "depth"}) is returned only for a model served with the long-context encoder or
        one with aux exits."""
        from serve.effort import canonical, resolve
        from serve.effort import threshold as check_threshold
        from serve.infer import record_confidence
        from serve.wire import RequestError
        try:
            effort, threshold = resolve(s.model, canonical(effort), check_threshold(threshold),
                                        bool(images))
        except ValueError as e:
            raise RequestError(str(e)) from None
        mo = max(spec["max_options"], max(len(q.options) for q in questions))
        if images:
            if s.prep is None:
                raise text_only_error(s.name, s.vision_error)
            # The state and its image tokens are read once; see score_image_questions_cached.
            try:
                plan = plan_image_request(s.tok, s.prep, state, images, questions, s.venc)
            except ValueError as e:
                err = state_too_long(e, s.venc.max_length, s.venc)
                if err is None:
                    raise
                raise err from None
            if effort is not None:
                plan["effort"] = effort         # "high": the aux heads read text only
            preds, tokens = score_image_planned(s.model, s.tok, plan, max_options=mo,
                                                device=s.device, batch_size=batch_size)
            record_confidence(s.model, plan, preds)
        else:
            s.model.eval()
            plan = plan_request(s.tok, state, questions, s.enc)
            if effort is not None:
                plan["effort"] = effort
            if threshold is not None:
                plan["threshold"] = threshold
            preds, tokens = score_planned(s.model, s.tok, plan, max_options=mo,
                                          device=s.device, batch_size=batch_size)
        out = [list(p.probs) for p in preds], tokens
        extras = {k: plan[k] for k in ("truncated", "depth", "effort", "confidence")
                  if plan.get(k) is not None}
        return (*out, extras) if extras else out

    return scorer


def warm_up(scorer, images: bool = False) -> float:
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
    if images:                  # the vision tower's kernels, before the first image request
        from PIL import Image
        scorer("<image> Warm-up.", parse_questions({"a": q["a"]}),
               [Image.new("RGB", (448, 448), (128, 128, 128))])
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
    ap.add_argument("--max-length", type=int, default=_env_int("RSIJEV_MAX_LENGTH"),
                    help="input cap in tokens (text; images add their budget), with the cut "
                         "--truncate picks. Default: RSIJEV_MAX_INPUT_TOKENS (32,768) and "
                         "nothing cut, a longer question gets a 422. Also RSIJEV_MAX_LENGTH.")
    ap.add_argument("--truncate", default=os.environ.get("RSIJEV_TRUNCATE") or None,
                    choices=["left", "middle"],
                    help="a state over the cap: 'left' cuts its start (every release up to "
                         "v4.0-VL); 'middle' keeps its first line, head and tail around a "
                         "marker, and responses then carry a `truncated` field. Default: "
                         "nothing is cut. Also RSIJEV_TRUNCATE.")
    ap.add_argument("--adaptive", default=os.environ.get("RSIJEV_ADAPTIVE", "").strip().lower() or None,
                    choices=["auto", "on", "off"],
                    help="adaptive exit, for a release with aux exits and a tuned tau: "
                         "'auto' for multi-question requests only (single questions take "
                         "the fixed exit), 'on' for every request, 'off' never. Image "
                         "requests and micro-batched rows always take the fixed exit. "
                         "Default: meta.json adaptive.serving, else auto. Also RSIJEV_ADAPTIVE.")
    ap.add_argument("--fixed-exit", action="store_true",
                    default=os.environ.get("RSIJEV_FIXED_EXIT", "").strip().lower()
                    not in ("", "0", "false", "no", "off"),
                    help="the same as --adaptive off. Also RSIJEV_FIXED_EXIT=1.")
    ap.add_argument("--effort", default=None,
                    help="default effort for a multi-exit release, used when a request gives "
                         "none: low (shallowest aux exit), medium (deepest aux exit), high "
                         "(main exit, all layers), auto (the release's confidence cascade, "
                         "every text request). A request's own \"effort\" wins. Unset: "
                         "serving as before (--adaptive). Image requests always run at full "
                         "depth. Also RSIJEV_EFFORT.")
    ap.add_argument("--version", default=None,
                    help="the release being served, as GET /v1/limits reports it. "
                         "Read off the checkpoint's own name when it carries one.")
    ap.add_argument("--backend", default=os.environ.get("RSIJEV_BACKEND", "").strip().lower() or "torch",
                    choices=["torch", "mlx"],
                    help="torch (default): the PyTorch path. mlx: Apple Silicon, an MLX checkpoint "
                         "made by scripts/convert_mlx.py (8-bit or bf16); same API and response "
                         "fields, no torch needed. Also RSIJEV_BACKEND.")


def serve(a: argparse.Namespace) -> int:
    ref = a.ckpt or getattr(a, "model", None)
    if not ref:
        raise SystemExit("give a checkpoint: a directory, a Hugging Face repo id such as "
                         "shgao/rsi-jev-v3.0-qwen3.5-2b, or an alias such as v3.0-2b")
    for k, v in apply_profile(a.profile).items():
        print(f"profile {a.profile}: {k}={v}", flush=True)
    if getattr(a, "backend", "torch") == "mlx":
        return serve_mlx(a, ref)

    import torch
    import uvicorn
    from serve.app import create_app

    s = load_for_serving(ref, device=a.device, dtype=a.dtype, revision=a.revision,
                         max_length=getattr(a, "max_length", None),
                         truncate=getattr(a, "truncate", None),
                         fixed_exit=getattr(a, "fixed_exit", None) or None,
                         adaptive=getattr(a, "adaptive", None))
    from serve.effort import default_effort
    effort = default_effort(getattr(a, "effort", None))
    explicit_mode = (getattr(a, "adaptive", None) is not None) or bool(getattr(a, "fixed_exit", False))
    if effort is not None and explicit_mode:
        raise SystemExit("--effort / RSIJEV_EFFORT sets the default depth itself; drop "
                         "--adaptive / --fixed-exit (and RSIJEV_ADAPTIVE / RSIJEV_FIXED_EXIT)")
    for applied in s.applied:
        print(f"speed path: {applied}", flush=True)
    from serve.batcher import model_worker
    from serve.infer import speed_options
    if a.batch_window_ms is None:
        a.batch_window_ms = _env_float("RSIJEV_BATCH_WINDOW_MS")   # set by --profile
    scorer = model_worker(s, batch_size=a.batch_size, window_ms=a.batch_window_ms,
                          default_effort=effort)
    if not a.no_warmup:
        # The first call JIT-compiles fla's Triton kernels (~19 s measured on a GB10
        # after a fresh install; cached afterwards) and, with --profile server,
        # torch.compile's graphs. Do it before accepting requests, not in one.
        print(f"warm-up: {warm_up(scorer, images=s.prep is not None):.1f} s", flush=True)
    name = a.served_model_name or s.name
    # /v1/limits reports which release is answering, so take it from the checkpoint
    # rather than from whatever this tree was cut for: serving a v1.0 checkpoint out of
    # a v2.1 checkout would otherwise announce v2.1, and a client tuning a threshold
    # per release would be told the wrong one.
    served_version = a.version or s.version
    app = create_app(scorer, served_model_name=name, alias=a.alias, api_key=a.api_key,
                     accept_models=a.accept_model, images=s.image_limits,
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
    il = s.image_limits
    if il.get("supported"):
        print(f"images: up to {il['max_images']} per request, {il['image_token_budget']} image "
              f"tokens per question; image requests skip the prefix and document caches",
              flush=True)
    elif il.get("reason"):
        print(f"images: off ({il['reason']})", flush=True)
    sv = s.meta.get("serving") or {}
    if sv.get("adaptive"):
        ad = sv["adaptive"]
        which = {"auto": "multi-question requests; one question takes the fixed exit",
                 "on": "every text request"}.get(ad.get("mode"), ad.get("mode"))
        print(f"adaptive exit {ad.get('mode')}: exits {ad['exits']}, tau {ad['tau']} "
              f"({which}; usage.depth reports the layers run)", flush=True)
    elif sv.get("adaptive_off"):
        print(f"adaptive exit off: {sv['adaptive_off']}", flush=True)
    if getattr(s.model, "effort_base", None) is not None:
        print(f"effort: default {effort or 'unset (as above)'}; requests may ask for low, "
              f"medium, high or auto (usage.effort reports it); images run at full depth",
              flush=True)
    print(f"input cap {s.enc.max_length} tokens; "
          + {"middle": "over-cap states cut in the middle (responses report `truncated`)",
             "left": "over-cap states cut from the start",
             "none": "nothing is cut, a longer question gets a 422"}[s.enc.truncate], flush=True)
    print(f"serving {ref} as {name!r} (alias {a.alias!r}) on {a.host}:{a.port}; "
          f"base {s.meta['base_model']}, calibration {s.meta.get('calibration', 'none')}",
          flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


def serve_mlx(a: argparse.Namespace, ref) -> int:
    """`serve` on the MLX backend (rsijev/mlx/serving.py): the same app, routes and response
    fields; one request at a time (no micro-batching), no document cache."""
    import uvicorn
    from serve.app import create_app
    from serve.effort import default_effort
    from rsijev.mlx.serving import load_for_serving as load_mlx, model_worker as mlx_worker
    if a.batch_window_ms is not None:
        print("micro-batching is not available on the MLX backend; --batch-window-ms ignored", flush=True)
    s = load_mlx(ref, revision=a.revision, max_length=getattr(a, "max_length", None),
                 truncate=getattr(a, "truncate", None),
                 fixed_exit=getattr(a, "fixed_exit", None) or None,
                 adaptive=getattr(a, "adaptive", None), dtype=a.dtype)
    effort = default_effort(getattr(a, "effort", None))
    explicit_mode = (getattr(a, "adaptive", None) is not None) or bool(getattr(a, "fixed_exit", False))
    if effort is not None and explicit_mode:
        raise SystemExit("--effort / RSIJEV_EFFORT sets the default depth itself; drop "
                         "--adaptive / --fixed-exit (and RSIJEV_ADAPTIVE / RSIJEV_FIXED_EXIT)")
    scorer = mlx_worker(s, batch_size=a.batch_size, default_effort=effort)
    if not a.no_warmup:
        print(f"warm-up: {warm_up(scorer, images=s.prep is not None):.1f} s", flush=True)
    name = a.served_model_name or s.name
    served_version = a.version or s.version
    app = create_app(scorer, served_model_name=name, alias=a.alias, api_key=a.api_key,
                     accept_models=a.accept_model, images=s.image_limits,
                     calibration=s.meta.get("calibration", "none"),
                     **({"version": served_version} if served_version else {}))
    import mlx.core as mx
    rec = s.model.mlx_record.get("tower") or {}
    print(f"runtime: mlx {mx.__version__}, {s.device}, tower {s.dtype_name}"
          + (f" (affine g{rec.get('group_size')})" if rec.get("group_size") else "")
          + ", heads fp32", flush=True)
    sv = s.meta.get("serving") or {}
    if sv.get("adaptive"):
        ad = sv["adaptive"]
        print(f"adaptive exit {ad.get('mode')}: exits {ad['exits']}, tau {ad['tau']} "
              f"(usage.depth reports the layers run)", flush=True)
    elif sv.get("adaptive_off"):
        print(f"adaptive exit off: {sv['adaptive_off']}", flush=True)
    if getattr(s.model, "effort_base", None) is not None:
        print(f"effort: default {effort or 'unset (as above)'}; requests may ask for low, "
              f"medium, high or auto (usage.effort reports it); images run at full depth", flush=True)
    il = s.image_limits
    if il.get("supported"):
        print(f"images: up to {il['max_images']} per request, {il['image_token_budget']} image "
              f"tokens per question", flush=True)
    elif il.get("reason"):
        print(f"images: off ({il['reason']})", flush=True)
    print(f"serving {ref} on MLX as {name!r} (alias {a.alias!r}) on {a.host}:{a.port}; "
          f"calibration {s.meta.get('calibration', 'none')}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Serve an RSI-Jev release behind the "
                                             "Jev-compatible API.")
    add_serve_args(ap, positional=True)
    return serve(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
