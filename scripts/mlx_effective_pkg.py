"""The PyTorch release package that holds exactly an MLX checkpoint's weights.

    python scripts/mlx_effective_pkg.py MLXDIR OUT

Every quantized tower tensor is dequantized the way MLX dequantizes it ((q * scale) + bias in
bf16; rsijev/mlx/quant.py, bit-exact with mx.dequantize) and written back as a bf16 weight,
so the PyTorch path (serve, the Decision Index gate, scripts/mlx_parity.py torch) scores the
MLX build's numerics with only the matmul accumulation order differing. Heads, calibration,
vision tower, config, tokenizer and meta are linked from MLXDIR (bf16 heads load into the
fp32 scorer exactly). No MLX needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from rsijev.mlx.quant import dequantize, unpack

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("mlx_dir")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    src, out = Path(a.mlx_dir), Path(a.out)
    rec = json.loads((src / "mlx.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    td, n = {}, 0
    with safe_open(str(src / "tower.safetensors"), "pt") as fh:
        keys = set(fh.keys())
        for k in sorted(keys):
            if k.endswith((".scales", ".biases")):
                continue
            name = k[: -len(".weight")] if k.endswith(".weight") else None
            if name and name + ".scales" in keys:
                s, b = fh.get_tensor(name + ".scales"), fh.get_tensor(name + ".biases")
                words = fh.get_tensor(k).view(torch.int32).numpy().view(np.uint32)
                part = rec["embed_tokens"] if k == "embed_tokens.weight" else rec["tower"]
                bits, g = part["bits"], part["group_size"]
                n_in = s.shape[-1] * g
                q = torch.from_numpy(unpack(words, bits, n_in))
                td[k] = dequantize(q, s, b, g, torch.bfloat16).contiguous()
                n += 1
            else:
                td[k] = fh.get_tensor(k)
    save_file(td, str(out / "tower.safetensors"))
    for f in src.iterdir():
        if f.name in ("tower.safetensors", "SHA256SUMS", "mlx.json") or not f.is_file():
            continue
        d = out / f.name
        if d.is_symlink() or d.exists():
            d.unlink()
        os.symlink(f.resolve(), d)
    (out / "EFFECTIVE.json").write_text(json.dumps({"from": str(src.resolve()), "dequantized": n,
                                                    "tower": rec["tower"], "embed_tokens": rec["embed_tokens"],
                                                    "heads": rec.get("heads")}, indent=1) + "\n")
    print(f"wrote {out}: {n} tensors dequantized, {len(td)} tower tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
