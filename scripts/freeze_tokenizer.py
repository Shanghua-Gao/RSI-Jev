"""Add tokenizer.frozen.json to an MLX package and refresh its SHA256SUMS.

    python scripts/freeze_tokenizer.py PACKAGE [PACKAGE ...]

Run with the transformers version the release was trained with (see pyproject.toml).
"""
from __future__ import annotations

import sys
from pathlib import Path

from rsijev.frozen_tok import freeze
from rsijev.mlx.convert import write_sha256sums

if __name__ == "__main__":
    for p in map(Path, sys.argv[1:]):
        print(freeze(p))
        write_sha256sums(p)
