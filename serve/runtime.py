"""What this process will actually run: device, precision, kernels, presets.

Everything here reports the running machine, never the training one. A checkpoint's
meta.json records the kernel stack it was TRAINED with (`linear_attn_kernel`), which
says nothing about whether the fused kernels are present where it is served.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
import os
import sys
from typing import Mapping, MutableMapping

FAST_INSTALL = 'pip install "rsi-jev[fast]"'

# A profile only sets defaults. A variable already in the environment wins, so
# `RSIJEV_DOC_CACHE=0 rsi-jev serve ... --profile agent` still runs without it.
PROFILES: dict[str, dict[str, str]] = {
    "agent": {"RSIJEV_DOC_CACHE": "1"},     # same or growing state asked about again
    "server": {"RSIJEV_COMPILE": "1"},      # long-running: pays ~40 s of compile once
}


def profile_env(profile: str | None, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The variables `profile` would set that `environ` does not already set."""
    if profile in (None, "", "default"):
        return {}
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    environ = os.environ if environ is None else environ
    return {k: v for k, v in PROFILES[profile].items() if k not in environ}


def apply_profile(profile: str | None,
                  environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Set the profile's defaults in `environ` (os.environ by default). Returns what
    was set."""
    environ = os.environ if environ is None else environ
    added = profile_env(profile, environ)
    environ.update(added)
    return added


def best_device(torch) -> str:
    """CUDA, else Apple Silicon, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def default_dtype_name(device: str) -> str:
    # bf16 on CUDA: measured 3.4x (0.8B) to 6.2x (2B) faster, and over the full
    # 2,000-decision test set it moves pooled top-1 by at most 0.003 against a
    # per-seed sd of 0.011. Serving takes the speed; evaluation stays fp32, so a
    # published number is never a bf16 number.
    return "bf16" if str(device).startswith("cuda") else "fp32"


_FUSED = ("fla", "causal_conv1d")


def keep_fused_kernels_off(device: str) -> list[str]:
    """Stop transformers binding the CUDA-only kernels for a model that runs elsewhere.

    transformers 5.17 picks the DeltaNet implementation when the Qwen3.5 modeling
    module is first imported, from whether `fla` imports, whatever device the
    model later runs on. With `[fast]` installed, a CPU run then calls fla's Triton
    kernels on CPU tensors and fails ("0 active drivers"). Hiding the packages
    from that import gives the torch reference, which is what runs on CPU and MPS
    anyway. Does nothing on CUDA, after the modeling module is already imported,
    or on transformers older than 5.17 (whose own check needs CUDA, and which
    would trip over the hidden module). Returns the packages it hid."""
    if str(device).startswith("cuda") or \
            "transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules:
        return []
    try:
        major, minor = (int(x) for x in
                        importlib.metadata.version("transformers").split(".")[:2])
    except Exception:
        return []
    if (major, minor) < (5, 17):
        return []
    hidden = []
    for name in _FUSED:
        if sys.modules.get(name, 0) is not None and importlib.util.find_spec(name) is not None:
            sys.modules[name] = None
            hidden.append(name)
    return hidden


def _dist_version(*names: str) -> str | None:
    for n in names:
        try:
            return importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:                     # a broken install is not an installed kernel
        return False


def _impl_module(fn, depth: int = 4) -> str | None:
    """The module of the function a transformers kernel hook finally calls.

    transformers <= 5.16 stores the chosen function on the layer, so its module is
    the answer. 5.17 wraps the torch reference and keeps the chosen function in the
    wrapper's closure as `implementation`; that is looked for through up to `depth`
    layers of `__wrapped__`. Returns None when it cannot tell."""
    outer = getattr(fn, "__module__", None)
    for _ in range(depth):
        if fn is None:
            break
        try:
            inner = inspect.getclosurevars(fn).nonlocals.get("implementation")
        except (TypeError, ValueError):
            inner = None
        if callable(inner):
            return getattr(inner, "__module__", None)
        fn = getattr(fn, "__wrapped__", None)
    return outer


def _model_kernels(model) -> dict[str, str | None]:
    """Which implementation the loaded DeltaNet layers call, read off the model
    and the modeling module it came from.

    Returns {"chunk_gated_delta_rule": module or None, "causal_conv1d_fn": ...}.
    None means this transformers version hides it; the caller falls back to
    importability."""
    found: dict[str, str | None] = {}
    if model is None:
        return found
    for mod in model.modules():
        if not hasattr(mod, "in_proj_qkv") and not hasattr(mod, "chunk_gated_delta_rule"):
            continue
        glb = sys.modules.get(type(mod).__module__)
        for name in ("chunk_gated_delta_rule", "causal_conv1d_fn"):
            fn = getattr(mod, name, None)
            if fn is None and hasattr(mod, name):
                found[name] = "torch"         # set to None on the layer: no kernel
                continue
            if fn is None and glb is not None:
                # transformers 5.17 calls module-level hooks, the delta rule under
                # the name of its torch reference.
                fn = getattr(glb, name, None) or getattr(glb, f"torch_{name}", None)
            found[name] = _impl_module(fn) if fn is not None else None
        break
    return found


def kernel_report(device: str, model=None) -> dict[str, dict]:
    """Whether fla and causal_conv1d are installed, importable, and used.

    Both are CUDA-only (Triton / CUDA extensions): on any other device the
    DeltaNet layers run the torch reference whatever is installed."""
    cuda = str(device).startswith("cuda")
    seen = _model_kernels(model)
    out = {}
    for key, dist, module, hook, prefix in (
            ("fla", ("flash-linear-attention", "fla-core"), "fla.ops.gated_delta_rule",
             "chunk_gated_delta_rule", "fla"),
            ("causal_conv1d", ("causal-conv1d", "causal_conv1d"), "causal_conv1d",
             "causal_conv1d_fn", "causal_conv1d")):
        version = _dist_version(*dist)
        importable = (version is not None or importlib.util.find_spec(module.split(".")[0])
                      is not None) and _importable(module)
        used = importable and cuda
        where = seen.get(hook)
        if where is not None:                 # the model says what it calls; trust it
            used = where.split(".")[0] == prefix
        out[key] = {"installed": version is not None or importable, "version": version,
                    "importable": importable, "used": used}
    return out


def describe_kernels(report: dict[str, dict], device: str) -> list[str]:
    """The kernel lines of the startup log, plus the install hint when it matters."""
    def one(name, r, optional=""):
        v = f" {r['version']}" if r["version"] else ""
        if r["used"]:
            return f"{name}{v} active"
        if r["importable"] or (r["installed"] and not str(device).startswith("cuda")):
            return f"{name}{v} installed, not used on {device}"
        if r["installed"]:
            return f"{name}{v} installed but fails to import"
        return f"{name} not installed{optional}"
    lines = ["kernels: " + one("fla", report["fla"]) + "; "
             + one("causal_conv1d", report["causal_conv1d"], " (optional)")]
    if str(device).startswith("cuda") and not report["fla"]["used"]:
        lines.append("fla is not active, so the DeltaNet layers run the slower torch "
                     f"path. Install it: {FAST_INSTALL} (from a clone: pip install -e \".[fast]\")")
    return lines


VISION_INSTALL = 'pip install "rsi-jev[vision]"'


def vision_report() -> dict[str, str | None]:
    """The `[vision]` extra: the installed version of Pillow and torchvision, or None.

    Image requests to a release trained with images (v4.0 on) need both: the image
    processor imports torchvision, and requests are decoded with Pillow."""
    return {name: (_dist_version(dist) or "installed") if _importable(module) else None
            for name, dist, module in (("pillow", "pillow", "PIL"),
                                       ("torchvision", "torchvision", "torchvision"))}


def describe_vision(report: dict[str, str | None]) -> str:
    missing = [k for k, v in report.items() if v is None]
    if not missing:
        return "images: [vision] extra installed (" + ", ".join(
            f"{k} {v}" for k, v in report.items()) + ")"
    return (f"images: [vision] extra missing ({', '.join(missing)} not installed); image "
            f"requests to v4.0 releases need it: {VISION_INSTALL}")


def startup_lines(*, torch, device: str, dtype_name: str, model=None,
                  profile: str | None = None, applied: list[str] = ()) -> list[str]:
    """What the server prints before it answers anything."""
    from serve.infer import _flag, default_min_saved_tokens
    if str(device).startswith("cuda"):
        idx = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(idx)
        where = f"{device} ({torch.cuda.get_device_name(idx)}, sm_{major}{minor})"
    else:
        where = device
    lines = [f"runtime: torch {torch.__version__}, {where}, tower {dtype_name}, scorer fp32"]
    lines += describe_kernels(kernel_report(device, model), device)
    lines.append(describe_vision(vision_report()))
    doc_cache = "on" if _flag("RSIJEV_DOC_CACHE") else "off"
    compile_ = ("on" if any(a.startswith("compile") for a in applied)
                else "off")
    lines.append(f"document read once per request when (questions - 1) x state tokens >= "
                 f"{default_min_saved_tokens()}; document cache {doc_cache}; compile {compile_}"
                 + (f"; profile {profile}" if profile else ""))
    return lines
