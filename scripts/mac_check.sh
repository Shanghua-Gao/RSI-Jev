#!/usr/bin/env bash
# Check the RSI-Jev MLX builds on this Mac (Apple silicon): answers vs the shipped references,
# median latency per effort and peak memory.
#
#   scripts/mac_check.sh                       # both builds, downloaded from Hugging Face
#   scripts/mac_check.sh ~/models/rsi-jev-v6.1-vl-4b-mlx-8bit   # a local package
#
# Creates .venv-mac-check in the repo (Python 3.10+), installs the MLX extra, runs
# scripts/mac_check.py on each build and saves mac_check_<name>.json next to this repo's root.
# Send those files back. Each build takes a few minutes (200 questions x 4 efforts).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "This check is for Apple silicon (macOS arm64); found $(uname -s) $(uname -m)." >&2
  exit 1
fi
PY=${PYTHON:-python3}
if [ ! -x .venv-mac-check/bin/python ]; then
  "$PY" -m venv .venv-mac-check
  .venv-mac-check/bin/pip install -q --upgrade pip
fi
.venv-mac-check/bin/pip install -q -e ".[mlx,vision]"
MODELS=("$@")
[ ${#MODELS[@]} -gt 0 ] || MODELS=(shgao/rsi-jev-v6.1-vl-4b-mlx-8bit shgao/rsi-jev-v6.1-vl-4b-mlx-4bit)
.venv-mac-check/bin/python -c "import mlx.core as mx; print('mlx', mx.__version__, mx.default_device())"
sysctl -n machdep.cpu.brand_string hw.memsize 2>/dev/null | paste -sd' ' - || true
for M in "${MODELS[@]}"; do
  N=$(basename "${M%/}")
  echo "== $N"
  .venv-mac-check/bin/python scripts/mac_check.py "$M" --json "mac_check_${N}.json"
done
