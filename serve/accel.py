"""Opt-in speed paths for serving, off by default.

    RSIJEV_COMPILE=1   torch.compile of the tower's layers and of the scorer
    RSIJEV_FLASHQLA=1  FlashQLA's TileLang kernel for the DeltaNet layers' chunked forward

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


def _import_flash_qla():
    """Import flash_qla (tested: git 846bb99, tilelang 0.1.12), working around three
    things that stop it on an sm_121 GB10. It runs the sm_120 kernels there:
    - its launch table knows compute version 12.0 but not 12.1, and it raises at import;
    - the sm_120 build lacks `aggregate_card_state`, which the forward imports by name
      even when context parallelism is off;
    - the sm_120 forward kernel at block_DV=128 needs 110 KB of shared memory per block,
      over the 99 KB this card allows. At 64 it fits.
    """
    import importlib.util
    if importlib.util.find_spec("flash_qla") is None:
        raise ImportError("RSIJEV_FLASHQLA needs flash_qla (github.com/QwenLM/FlashQLA)")
    import tilelang.contrib.nvcc as nvcc
    real = nvcc.get_target_compute_version
    seen = real()
    if seen == "12.1":
        nvcc.get_target_compute_version = lambda *a, **k: "12.0"
    try:
        import flash_qla
        from flash_qla.ops.gated_delta_rule import chunk as fq_chunk
    finally:
        nvcc.get_target_compute_version = real
    if not hasattr(fq_chunk, "aggregate_card_state"):
        fq_chunk.aggregate_card_state = None
    if seen in ("12.0", "12.1"):
        from flash_qla.ops.gated_delta_rule.chunk.blackwell_sm120 import fused_fwd
        kern = fused_fwd.tilelang_fused_chunk_gdr_fwd
        if not getattr(kern, "_rsijev_dv", False):
            smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
            dv = int(os.environ.get("RSIJEV_FLASHQLA_BLOCK_DV", 128 if smem >= 112640 else 64))

            def narrow(*args, **kw):
                return kern(*args, **{**kw, "block_DV": min(kw.get("block_DV", 128), dv)})
            narrow._rsijev_dv = True
            fused_fwd.tilelang_fused_chunk_gdr_fwd = narrow
    return flash_qla


def enable_flashqla(model) -> int:
    """Route the DeltaNet layers' chunked forward to FlashQLA.

    Swaps `chunk_gated_delta_rule` on each linear-attention module of the tower,
    so the rest of the layer (projections, conv, gate, norm) is untouched. Calls
    FlashQLA doesn't cover go to the kernel the layer had: fp32 inputs, head
    dims other than 128, and anything that needs a gradient. Context parallelism
    is off: the sequences here are short and it is not implemented for sm_120.
    Returns the number of layers switched; raises ImportError without flash_qla."""
    fq = _import_flash_qla().chunk_gated_delta_rule
    n = 0
    for mod in model.tower.modules():
        orig = getattr(mod, "chunk_gated_delta_rule", None)
        if orig is None or getattr(orig, "_rsijev_flashqla", False):
            continue

        def run(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                use_qk_l2norm_in_kernel=False, cu_seqlens=None, _orig=orig, **kw):
            if (q.dtype not in (torch.bfloat16, torch.float16) or q.shape[-1] != 128
                    or v.shape[-1] != 128 or kw or (torch.is_grad_enabled() and any(
                        t is not None and t.requires_grad for t in (q, k, v, g, beta)))):
                return _orig(q, k, v, g=g, beta=beta, scale=scale, initial_state=initial_state,
                             output_final_state=output_final_state,
                             use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                             cu_seqlens=cu_seqlens, **kw)
            return fq(q, k, v, g=g, beta=beta, scale=scale, initial_state=initial_state,
                      output_final_state=output_final_state,
                      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens,
                      auto_cp=False, enable_fwd_cp_cache=False)
        run._rsijev_flashqla = True
        mod.chunk_gated_delta_rule = run
        n += 1
    if n:
        prev = getattr(model, "_rsijev_numerics", "eager")
        model._rsijev_numerics = f"{prev}+flashqla"
    return n


def apply_env(model) -> list[str]:
    """Apply what RSIJEV_FLASHQLA and RSIJEV_COMPILE ask for; returns what was applied."""
    done = []
    if _flag("RSIJEV_FLASHQLA"):
        done.append(f"flashqla ({enable_flashqla(model)} DeltaNet layers)")
    if _flag("RSIJEV_COMPILE"):
        mode = os.environ.get("RSIJEV_COMPILE_MODE") or None
        done.append(f"compile ({enable_compile(model, mode=mode)} modules, {mode or 'default'})")
    return done
