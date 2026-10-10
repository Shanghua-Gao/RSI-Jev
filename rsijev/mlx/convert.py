"""Release package -> MLX checkpoint directory.

    python scripts/convert_mlx.py PKG OUT --bits 8 --group 64 [--embed-bits 8] [--heads-dtype bf16]
    python scripts/convert_mlx.py PKG OUT --codes gptq4g32.safetensors [--embed-bits 8]

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

`codes` (a safetensors file with `<name>.q` uint8 codes, `<name>.s` scales and `<name>.b`
biases for every tower linear, e.g. a GPTQ result) is packed as is: nothing is re-quantized,
bits and group size are read from the file, and the scales and biases are stored in bf16, the
dtype MLX dequantizes in. `overrides` ({tower key: (codes, scales, biases)}) does the same for
single matrices.

--embed-bits quantizes embed_tokens by round-to-nearest at --embed-group (default 64), which
may differ from the tower's group. --heads-dtype bf16 stores the scorer and exit-head weights in
bf16 (they are still computed in fp32); calibration.safetensors stays fp32.

The output also gets SHA256SUMS (every file) and a size line in mlx.json.
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


HEAD_FILES = ("scorer.safetensors", "aux_scorers.safetensors")


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


def load_codes(path: str | Path) -> tuple[dict, int, int]:
    """A precomputed quantization (`<name>.q` / `.s` / `.b` per linear) -> (overrides keyed by
    the tower's `<name>.weight`, bits, group). Bits = the narrowest width holding every code."""
    from safetensors.torch import load_file
    sd = load_file(str(path))
    names = sorted({k.rsplit(".", 1)[0] for k in sd})
    over, groups, qmax = {}, set(), 0
    for n in names:
        try:
            q, s, b = sd[n + ".q"], sd[n + ".s"], sd[n + ".b"]
        except KeyError as e:
            raise SystemExit(f"{path}: {n} lacks {e}") from None
        if s.shape != b.shape or q.shape[:-1] != s.shape[:-1] or q.shape[-1] % s.shape[-1]:
            raise SystemExit(f"{path}: {n} codes {tuple(q.shape)} / scales {tuple(s.shape)} / biases "
                             f"{tuple(b.shape)} do not fit together")
        groups.add(q.shape[-1] // s.shape[-1])
        qmax = max(qmax, int(q.max()))
        over[n + ".weight"] = (q, s, b)
    if len(groups) != 1:
        raise SystemExit(f"{path}: mixed group sizes {sorted(groups)}")
    bits = next(b for b in (2, 3, 4, 5, 6, 8) if qmax < (1 << b))
    return over, bits, groups.pop()


def convert(pkg: str | Path, out: str | Path, *, bits: int = 8, group: int = 64,
            embed_bits: int | None = None, embed_group: int = 64, codes: str | Path | None = None,
            heads_dtype: str = "fp32", overrides: dict | None = None, log=print) -> dict:
    """Write the MLX checkpoint for `pkg` into `out`; returns the mlx.json record."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from .quant import pack, quantize

    pkg, out = Path(pkg), Path(out)
    codes_sha = None
    if codes is not None:
        over, bits, group = load_codes(codes)
        codes_sha = _sha256(Path(codes))
        log(f"codes {Path(codes).name}: {len(over)} matrices, {bits}-bit g{group} (packed as is)")
        overrides = {**over, **(overrides or {})}
    if heads_dtype not in ("fp32", "bf16"):
        raise SystemExit(f"--heads-dtype must be fp32 or bf16, got {heads_dtype}")
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
    from_codes = codes is not None
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
                overrides.pop(key, None)          # codes for a layer the release never runs
                continue
            w = fh.get_tensor(key)
            bytes_in += w.numel() * w.element_size()
            b, g = None, group
            if from_codes and is_quantized_linear(key) and key not in overrides:
                raise SystemExit(f"--codes has no entry for {key}: every tower linear must come "
                                 f"from the codes file (nothing is re-quantized)")
            if key in overrides or (bits < 16 and is_quantized_linear(key)):
                b = bits
            elif key == "embed_tokens.weight" and embed_bits:
                b, g = embed_bits, embed_group
            if b is not None and w.shape[-1] % g:
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
                if tuple(q.shape) != tuple(w.shape) or q.shape[-1] // s.shape[-1] != g:
                    raise SystemExit(f"{key}: codes {tuple(q.shape)} g{q.shape[-1] // s.shape[-1]} "
                                     f"vs weight {tuple(w.shape)} g{g}")
                q = q.to(torch.uint8)
            else:
                q, s, z = quantize(w, b, g)
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
        if not src.exists():
            continue
        if heads_dtype == "bf16" and name in HEAD_FILES:
            from safetensors.torch import load_file
            sd = load_file(str(src))
            meta_st = safe_open(str(src), framework="pt").metadata()
            save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous()
                       for k, v in sd.items()}, str(out / name), metadata=meta_st)
        else:
            shutil.copy2(src, out / name)
        copied.append(name)

    record = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "source": pkg.resolve().name,
        "source_tower_sha256": _tower_sha(pkg),
        "tower": {"bits": bits, "group_size": group if bits < 16 else None, "mode": "affine",
                  "quantized": f"{n_q - (1 if embed_bits else 0)} decoder-layer linears" if bits < 16 else "none (bf16)",
                  "method": ("precomputed codes (packed as is)" if from_codes else "round-to-nearest (mx.quantize)")
                  if bits < 16 else None,
                  "codes_sha256": codes_sha, "scales_dtype": "bfloat16"},
        "embed_tokens": {"bits": embed_bits, "group_size": embed_group if embed_bits else None,
                         "method": "round-to-nearest (mx.quantize)" if embed_bits else None},
        "vision": {"bits": 16},
        "heads": ("fp32 as shipped (scorer, aux_scorers, calibration)" if heads_dtype == "fp32" else
                  "scorer and aux_scorers weights bf16 (computed in fp32); calibration fp32 as shipped"),
        "not_quantized": skipped,
        "exit_layer": exit_layer, "layers_dropped": n_dropped,
        "tower_bytes": {"source": bytes_in, "mlx": bytes_out},
        "files": copied + ["tower.safetensors", "mlx.json", "SHA256SUMS"],
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sizes = checkpoint_bytes(out)
    record["size"] = {"weights_bytes": sizes["total"],
                      "line": f"{sizes['total'] / 1e9:.2f} GB of weights ("
                              + ", ".join(f"{k} {v / 1e9:.2f}" for k, v in sizes.items() if k != "total") + ")"}
    (out / "mlx.json").write_text(json.dumps(record, indent=1) + "\n")
    write_sha256sums(out)
    log(f"converted {pkg.name}: {n_q} matrices at {bits}-bit g{group}"
        + (f", embeddings {embed_bits}-bit" if embed_bits else "")
        + f"; tower {bytes_in / 1e9:.2f} -> {bytes_out / 1e9:.2f} GB; {time.time() - t0:.0f} s")
    return record


def write_sha256sums(out: str | Path) -> Path:
    """SHA256SUMS over every file in `out` (sorted, `sha256sum -c` format)."""
    out = Path(out)
    lines = [f"{_sha256(p)}  {p.name}" for p in sorted(out.iterdir())
             if p.is_file() and p.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return out / "SHA256SUMS"


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
