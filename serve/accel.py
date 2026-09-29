"""Opt-in speed path for serving, off by default.

    RSIJEV_COMPILE=1   torch.compile of the tower's layers and of the scorer

Not used by evaluation, and it does not change what `score_questions` does; it
changes the model object it is handed. serve/README.md has what it was measured
to cost in agreement and calibration.
"""
from __future__ import annotations

import os

import torch
from torch import nn


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


def enable_compile(model, *, mode: str | None = None, dynamic: bool = True,
                   scope: str | None = None) -> int:
    """torch.compile the tower in pieces, and the scorer.

    scope "layers" (default): each whole decoder layer, fla calls included;
    whatever dynamo cannot trace becomes a graph break inside the layer. This is
    the configuration that was gated (serve/README.md).
    scope "blocks": only each layer's MLP and full-attention block, with the
    DeltaNet mixers eager. Measured barely faster than eager on a GB10.

    `reduce-overhead` (CUDA graphs) is refused: it made this box slower, and it
    needs static shapes the serving path does not have. Returns the number of
    compiled modules."""
    if mode == "reduce-overhead":
        raise ValueError("reduce-overhead is not supported here; use default or "
                         "max-autotune-no-cudagraphs")
    import torch._dynamo
    from rsijev.arch import _decoder_layers
    scope = scope or os.environ.get("RSIJEV_COMPILE_SCOPE") or "layers"
    # One compiled code object serves every layer instance; the default limit of 8
    # is used up by the layer count and the with/without-cache variants alone.
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 256)
    torch._dynamo.config.accumulated_recompile_limit = max(
        torch._dynamo.config.accumulated_recompile_limit, 4096)
    kw = {"dynamic": dynamic, **({"mode": mode} if mode else {})}
    n = 0
    for layer in _decoder_layers(model.tower):
        if scope == "layers":
            layer.compile(**kw); n += 1
            continue
        if hasattr(layer, "mlp"):
            layer.mlp.compile(**kw); n += 1
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            attn.compile(**kw); n += 1
    model.scorer.compile(**kw); n += 1
    model._rsijev_numerics = f"compiled-{scope}-{mode or 'default'}"
    return n


def apply_env(model) -> list[str]:
    """Apply what RSIJEV_COMPILE asks for; returns what was applied."""
    done = []
    if _flag("RSIJEV_COMPILE"):
        mode = os.environ.get("RSIJEV_COMPILE_MODE") or None
        done.append(f"compile ({enable_compile(model, mode=mode)} modules, {mode or 'default'})")
    return done
