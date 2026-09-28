"""Serve a released checkpoint behind the Jev-compatible API.

    python scripts/serve.py --ckpt path/to/ckpt --host 0.0.0.0 --port 8000

`--ckpt` is a release checkpoint (see scripts/load_release.py). The served model
name defaults to the checkpoint's own id; `jev-latest` is accepted as an alias,
as in the reference. Set --api-key (or RSIJEV_API_KEY) to require a bearer token
on everything except the health routes.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# The root must precede scripts/: THIS FILE is what shadows the serve/ package,
# so with scripts/ first "from serve.app import ..." below resolves to itself.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _best_device(torch) -> str:
    """CUDA, else Apple Silicon, else CPU. Everything runs fp32, so there is no
    dtype decision to get wrong on MPS."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
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
    ap.add_argument("--version", default=None,
                    help="the release being served, as GET /v1/limits reports it. "
                         "Read off the checkpoint's own name when it carries one.")
    a = ap.parse_args()

    import torch
    import uvicorn
    from load_release import load_release
    from serve.app import create_app
    from serve.infer import score_questions_cached

    device = a.device or _best_device(torch)
    # bf16 on CUDA: measured 3.4x (0.8B) to 6.2x (2B) faster, and over the full
    # 2,000-decision test set it moves pooled top-1 by at most 0.003 against a
    # per-seed sd of 0.011. Serving takes the speed; evaluation stays fp32, so a
    # published number is never a bf16 number.
    dtype_name = a.dtype or ("bf16" if device == "cuda" else "fp32")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    model, tok, enc, meta = load_release(a.ckpt, device, infer_dtype=dtype)
    name = a.served_model_name or Path(a.ckpt).resolve().name
    spec = meta["spec"]

    def scorer(state: str, questions):
        preds, tokens = score_questions_cached(model, tok, state, questions, enc,
                                               device=device, batch_size=a.batch_size,
                                               max_options=max(spec["max_options"],
                                                               max(len(q.options) for q in questions)))
        return [list(p.probs) for p in preds], tokens

    # /v1/limits reports which release is answering, so take it from the checkpoint
    # rather than from whatever this tree was cut for: serving a v1.0 checkpoint out of
    # a v2.1 checkout would otherwise announce v2.1, and a client tuning a threshold
    # per release would be told the wrong one.
    found = re.search(r"v\d+\.\d+", Path(a.ckpt).resolve().name)
    served_version = a.version or (found.group(0) if found else None)
    app = create_app(scorer, served_model_name=name, alias=a.alias, api_key=a.api_key,
                     accept_models=a.accept_model,
                     calibration=meta.get("calibration", "none"),
                     **({"version": served_version} if served_version else {}))
    print(f"serving {a.ckpt} as {name!r} (alias {a.alias!r}) on {a.host}:{a.port}; "
          f"base {meta['base_model']}, kernel {meta['linear_attn_kernel']}, "
          f"tower {dtype_name} on {device}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
