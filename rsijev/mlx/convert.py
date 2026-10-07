"""Release package -> MLX checkpoint directory.

    python scripts/convert_mlx.py PKG OUT --bits 8 --group 64 [--embed-bits 8]

PKG is a self-contained release (config.json next to meta.json: v5.0-VL on). OUT gets:

  tower.safetensors      the text tower. Every decoder-layer linear (gated-DeltaNet
                         in_proj_qkv/z/b/a and out_proj; attention q/k/v/o; MLP gate/up/down)
                         in MLX's affine group format when --bits < 16: `<name>.weight` uint32
                         packed codes, `<name>.scales` and `<name>.biases` bf16, exactly what
                         `mx.quantize(w, group_size, bits)` returns for the bf16 weight
                         (rsijev/mlx/quant.py). `embed_tokens` too with --embed-bits. Norms,
                         conv1d, A_log, dt_bias stay as shipped (bf16). --bits 16 copies the
                         tower unchanged (the bf16 MLX build).
  visual.safetensors     the vision tower, bf16, unchanged
  scorer.safetensors, aux_scorers.safetensors, calibration.safetensors, calibration.json,
  meta.json, config.json, tokenizer and image-processor files
                         copied unchanged
  mlx.json               what was quantized and how, and the source package's name

Decoder layers an early-exit release never runs (index >= spec.arch_extra.exit_layer) are
dropped. Nothing here needs MLX: the codes are computed in PyTorch, bit for bit what MLX
computes (tests/test_mlx_convert.py checks them against mx.quantize when MLX is installed).

`overrides` ({tower key: (codes uint8, scales, biases)}) packs another quantizer's result
(GPTQ, DWQ) in place of round-to-nearest for those matrices; the format is the same.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

FORMAT = "rsijev-mlx"
FORMAT_VERSION = 1

# tower linears quantized at --bits (the 248 matrices of the quantization study at 4B)
LINEAR_NAMES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_LINEAR = re.compile(r"^layers\.\d+\.(linear_attn|self_attn|mlp)\.(" + "|".join(LINEAR_NAMES) + r")\.weight$")

COPY = ("meta.json", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
        "preprocessor_config.json", "video_preprocessor_config.json",
        "scorer.safetensors", "aux_scorers.safetensors", "calibration.safetensors",
        "calibration.json", "visual.safetensors")


def is_quantized_linear(key: str) -> bool:
    return bool(_LINEAR.match(key))


def _layer_index(key: str) -> int | None:
    m = re.match(r"^layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(pkg: str | Path, out: str | Path, *, bits: int = 8, group: int = 64,
            embed_bits: int | None = None, overrides: dict | None = None,
            log=print) -> dict:
    """Write the MLX checkpoint for `pkg` into `out`; returns the mlx.json record."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from .quant import pack, quantize

    pkg, out = Path(pkg), Path(out)
    meta = json.loads((pkg / "meta.json").read_text())
    if not (pkg / "config.json").exists():
        raise SystemExit(f"{pkg} is not a self-contained release (no config.json); the MLX "
                         f"converter needs one (v5.0-VL on)")
    if bits not in (2, 3, 4, 5, 6, 8, 16):
        raise SystemExit(f"--bits must be one of 2, 3, 4, 5, 6, 8 or 16, got {bits}")
    if embed_bits not in (None, 2, 3, 4, 5, 6, 8):
        raise SystemExit(f"--embed-bits must be one of 2, 3, 4, 5, 6 or 8, got {embed_bits}")
    exit_layer = ((meta.get("spec") or {}).get("arch_extra") or {}).get("exit_layer")
    out.mkdir(parents=True, exist_ok=True)
    overrides = dict(overrides or {})
    t0 = time.time()

    tensors: dict[str, torch.Tensor] = {}
    n_q = n_kept = n_dropped = 0
    skipped: list[str] = []
    bytes_in = bytes_out = 0
    with safe_open(str(pkg / "tower.safetensors"), framework="pt") as fh:
        for key in fh.keys():
            li = _layer_index(key)
            if exit_layer and li is not None and li >= int(exit_layer):
                n_dropped += 1
                continue
            w = fh.get_tensor(key)
            bytes_in += w.numel() * w.element_size()
            b = None
            if key in overrides or (bits < 16 and is_quantized_linear(key)):
                b = bits
            elif key == "embed_tokens.weight" and embed_bits:
                b = embed_bits
            if b is not None and w.shape[-1] % group:
                skipped.append(key)       # as mlx-lm does: a width the group does not divide stays dense
                b = None
            if b is None:
                tensors[key] = w.contiguous()
                bytes_out += w.numel() * w.element_size()
                n_kept += 1
                continue
            if w.dtype != torch.bfloat16:
                w = w.to(torch.bfloat16)        # MLX quantizes the bf16 weight it serves
            if key in overrides:
                q, s, z = overrides.pop(key)
            else:
                q, s, z = quantize(w, b, group)
            name = key[: -len(".weight")]
            tensors[f"{name}.weight"] = torch.from_numpy(pack(q.numpy(), b).view("int32")).view(torch.uint32)
            tensors[f"{name}.scales"] = s.to(torch.bfloat16).contiguous()
            tensors[f"{name}.biases"] = z.to(torch.bfloat16).contiguous()
            bytes_out += sum(tensors[f"{name}.{p}"].numel() * tensors[f"{name}.{p}"].element_size()
                             for p in ("weight", "scales", "biases"))
            n_q += 1
    if overrides:
        raise SystemExit(f"overrides name tensors the tower does not have: {sorted(overrides)[:5]}")
    save_file(tensors, str(out / "tower.safetensors"), metadata={"format": "mlx"})
    del tensors

    copied = []
    for name in COPY:
        src = pkg / name
        if src.exists():
            shutil.copy2(src, out / name)
            copied.append(name)

    record = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "source": pkg.resolve().name,
        "source_tower_sha256": _tower_sha(pkg),
        "tower": {"bits": bits, "group_size": group if bits < 16 else None, "mode": "affine",
                  "quantized": f"{n_q} decoder-layer linears" if bits < 16 else "none (bf16)",
                  "scales_dtype": "bfloat16"},
        "embed_tokens": {"bits": embed_bits, "group_size": group if embed_bits else None},
        "vision": {"bits": 16},
        "heads": "fp32 as shipped (scorer, aux_scorers, calibration)",
        "not_quantized": skipped,
        "exit_layer": exit_layer, "layers_dropped": n_dropped,
        "tower_bytes": {"source": bytes_in, "mlx": bytes_out},
        "files": copied + ["tower.safetensors", "mlx.json"],
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "mlx.json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"converted {pkg.name}: {n_q} matrices at {bits}-bit g{group}"
        + (f", embeddings {embed_bits}-bit" if embed_bits else "")
        + f"; tower {bytes_in / 1e9:.2f} -> {bytes_out / 1e9:.2f} GB; {time.time() - t0:.0f} s")
    return record


def _tower_sha(pkg: Path) -> str | None:
    """The tower's sha256 from the package's SHA256SUMS (no rehash of 8 GB), else None."""
    sums = pkg / "SHA256SUMS"
    if not sums.exists():
        return None
    for line in sums.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*").endswith("tower.safetensors"):
            return parts[0]
    return None


def checkpoint_bytes(path: str | Path) -> dict:
    """On-disk bytes per weight file of an MLX checkpoint (for the docs' size table)."""
    path = Path(path)
    out = {p.name: p.stat().st_size for p in sorted(path.glob("*.safetensors"))}
    out["total"] = sum(out.values())
    return out
