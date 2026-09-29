"""Convert a release checkpoint into one self-contained MLX directory.

    python -m rsijev_mlx.convert --ckpt shgao/rsi-jev-v3.0-qwen3.5-2b --out rsijev-v3.0-mlx
    python -m rsijev_mlx.convert --ckpt DIR --out OUT --dtype bfloat16   # half the size

A release ships the tower without its embedding (it comes from the public base
model), in PyTorch layout. The converted directory holds the whole tower in
mlx-lm layout, the scorer and calibration unchanged (they stay fp32 at any
--dtype), meta.json and the tokenizer, so `rsijev_mlx.load(OUT)` needs nothing else.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402

from rsijev_mlx.model import CONVERTED, DTYPES, release_tower_weights, resolve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    a = ap.parse_args()
    ckpt, out = resolve(a.ckpt), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ckpt / "meta.json").read_text())
    weights, text_cfg = release_tower_weights(ckpt, meta)
    dt = DTYPES[a.dtype]
    weights = {k: (v if k.endswith("A_log") else v.astype(dt)) for k, v in weights.items()}
    mx.save_safetensors(str(out / CONVERTED), weights)
    (out / "mlx_config.json").write_text(json.dumps(text_cfg, indent=1) + "\n")
    for f in ("scorer.safetensors", "calibration.safetensors", "calibration.json", "meta.json"):
        if (ckpt / f).exists():
            shutil.copy2(ckpt / f, out / f)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(meta["base_model"]).save_pretrained(str(out))
    print(f"wrote {out} ({a.dtype} tower)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
