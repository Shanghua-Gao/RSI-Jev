"""Convert a release package into an MLX checkpoint (Apple Silicon): python scripts/convert_mlx.py PKG OUT --bits 8 --group 64

PKG is a release directory, a Hugging Face repo id or an alias (v6.1-vl-4b); OUT the new
directory. --bits 8 --group 64 is the build the quantization study found free (Decision
Index +0.01 vs bf16, argmax agreement at the bf16 noise floor); --bits 16 keeps the tower in
bf16. --codes packs a precomputed quantization (GPTQ codes) without re-quantizing.
--embed-bits 8 also quantizes the embedding table (0.6 GB less, agreement at the noise
floor alone, not Decision-Index-gated in combination). The vision tower and the decision
heads are copied unchanged. See rsijev/mlx/convert.py for the format.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("pkg", help="release directory, Hugging Face repo id or alias")
    ap.add_argument("out", help="output directory")
    ap.add_argument("--bits", type=int, default=8, help="tower linears: 2-8, or 16 for bf16 (default 8)")
    ap.add_argument("--group", type=int, default=64, help="quantization group size (default 64)")
    ap.add_argument("--embed-bits", type=int, default=None,
                    help="also quantize embed_tokens at this many bits (default: keep bf16)")
    ap.add_argument("--embed-group", type=int, default=64, help="embed_tokens group size (default 64)")
    ap.add_argument("--codes", default=None,
                    help="precomputed codes for every tower linear (<name>.q/.s/.b safetensors, e.g. "
                         "GPTQ): packed as is, bits and group read from the file")
    ap.add_argument("--heads-dtype", choices=("fp32", "bf16"), default="fp32",
                    help="store the scorer and exit-head weights in bf16 (computed in fp32 either way)")
    ap.add_argument("--revision", default=None, help="Hugging Face revision to download")
    a = ap.parse_args(argv)
    from serve.release import resolve_ckpt
    from rsijev.mlx.convert import checkpoint_bytes, convert
    pkg = resolve_ckpt(a.pkg, revision=a.revision)
    rec = convert(pkg, a.out, bits=a.bits, group=a.group, embed_bits=a.embed_bits,
                  embed_group=a.embed_group, codes=a.codes, heads_dtype=a.heads_dtype)
    sizes = checkpoint_bytes(a.out)
    for k, v in sizes.items():
        print(f"  {k:28s} {v / 1e9:6.2f} GB")
    print("size:", rec["size"]["line"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
