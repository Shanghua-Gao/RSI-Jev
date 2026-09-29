"""Opt-in speed paths for serving, each off by default.

    RSIJEV_FP8=1       FP8 (dynamic activation + weight) on the tower's Linear layers
    RSIJEV_COMPILE=1   torch.compile of the tower's layers and of the scorer

Neither is used by evaluation, and neither changes what `score_questions` does;
they change the model object it is handed. serve/README.md has what each was
measured to cost in agreement and calibration, and whether it earned a place.
"""
from __future__ import annotations

import os

import torch
from torch import nn


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


def _fp8_eligible(module: nn.Module, fqn: str, min_features: int = 128) -> bool:
    """Large Linear layers only. Embeddings are not Linear; the scorer and its
    calibration head live outside the tower; the DeltaNet recurrence and its
    convolution are fla kernels, not Linear; the tiny per-head gate projections
    (in_proj_a / in_proj_b, 16 outputs) are too small for FP8 GEMMs to help and
    too few to matter."""
    return (isinstance(module, nn.Linear)
            and module.in_features % 16 == 0 and module.out_features % 16 == 0
            and min(module.in_features, module.out_features) >= min_features)


def enable_fp8(model, *, granularity: str = "row") -> int:
    """Quantize the tower's Linear layers to FP8 in place, via torchao's dynamic
    activation + weight scheme. The tower must already be bf16; the scorer stays
    fp32. Returns the number of layers converted."""
    tower = model.tower
    if next(tower.parameters()).dtype != torch.bfloat16:
        raise ValueError("FP8 expects a bf16 tower (load_release(..., infer_dtype=torch.bfloat16))")
    from torchao.quantization import (Float8DynamicActivationFloat8WeightConfig, PerRow,
                                      PerTensor, quantize_)
    names = [n for n, m in tower.named_modules() if _fp8_eligible(m, n)]
    g = PerRow() if granularity == "row" else PerTensor()
    # The DeltaNet mixers zero padded positions before their projections, and a
    # row of zeros has amax 0: a zero scale, 0/0, and NaN that the recurrence and
    # attention then spread to real tokens. A tiny floor on amax fixes that and
    # leaves every non-zero row's scale exactly as it was.
    quantize_(tower, Float8DynamicActivationFloat8WeightConfig(
        granularity=g, activation_value_lb=1e-12), filter_fn=_fp8_eligible)
    model.__dict__.pop("_rsijev_fingerprint", None)   # the weights changed
    model._rsijev_numerics = f"fp8-{granularity}"
    return len(names)


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
    """Apply whatever RSIJEV_FP8 / RSIJEV_COMPILE ask for; returns what was applied."""
    done = []
    if _flag("RSIJEV_FP8"):
        done.append(f"fp8 ({enable_fp8(model)} layers)")
    if _flag("RSIJEV_COMPILE"):
        mode = os.environ.get("RSIJEV_COMPILE_MODE") or None
        done.append(f"compile ({enable_compile(model, mode=mode)} modules, {mode or 'default'})")
    return done
