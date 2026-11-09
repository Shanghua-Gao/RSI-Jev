#!/usr/bin/env bash
# Check an RSI-Jev MLX build on this Mac (Apple silicon): latency of one short question (median
# and mean, per effort), peak memory, and answers vs the shipped references.
#
#   scripts/mac_check.sh ~/models/rsi-jev-v7.0-vl-3b-mlx   # a local package
#   scripts/mac_check.sh                                         # v7.0-VL 3B 8-bit from Hugging Face
#
# Keep the package's directory name: it picks the reference in scripts/mac_check_ref/.
# Needs Python 3.10 or later. It uses the first python3.10-3.13 it finds (or $PYTHON); if there is
# none (the macOS system Python is 3.9), it downloads a private standalone Python 3.12 into
# .mac-check-python in the repo, 17 MB, no admin rights, nothing outside this directory,
# checked by SHA-256. Then it creates .venv-mac-check, installs the MLX extra (torch and
# transformers come with it, about 1 GB), runs scripts/mac_check.py on each build and saves
# mac_check_<name>.json next to this repo's root.
# Send those files back. Each build takes a few minutes (200 questions x 4 efforts).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "This check is for Apple silicon (macOS arm64); found $(uname -s) $(uname -m)." >&2
  exit 1
fi
version_ok() { "$1" -c 'import sys; sys.exit(not (3, 10) <= sys.version_info[:2] < (3, 14))' 2>/dev/null; }
find_python() {
  for c in "${PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3 .mac-check-python/python/bin/python3; do
    if [ -n "$c" ] && command -v "$c" >/dev/null && version_ok "$c"; then command -v "$c"; return 0; fi
  done
  return 1
}
if [ ! -x .venv-mac-check/bin/python ]; then
  PY=$(find_python) || {
    TAG=20251217; FILE=cpython-3.12.12+$TAG-aarch64-apple-darwin-install_only.tar.gz
    SHA=127ccc1cbc164592c4460c92ce8b4b61da31b0eaa4dc28946f76ef2b9d274d76
    echo "No Python 3.10-3.13 found; downloading a private Python 3.12 into .mac-check-python"
    mkdir -p .mac-check-python
    curl -fL --retry 3 -o .mac-check-python/python.tar.gz \
      "https://github.com/astral-sh/python-build-standalone/releases/download/$TAG/${FILE//+/%2B}"
    echo "$SHA  .mac-check-python/python.tar.gz" | shasum -a 256 -c -
    tar -xzf .mac-check-python/python.tar.gz -C .mac-check-python
    rm .mac-check-python/python.tar.gz
    PY=.mac-check-python/python/bin/python3
    version_ok "$PY" || { echo "the downloaded Python does not run" >&2; exit 1; }
  }
  "$PY" -m venv .venv-mac-check
  .venv-mac-check/bin/pip install -q --upgrade pip
fi
.venv-mac-check/bin/pip install -q -e ".[mlx,vision]" "mlx==0.32.3"   # the version the builds were checked with
if [ $# -gt 0 ]; then MODELS=("$@"); else MODELS=(shgao/rsi-jev-v7.0-vl-3b-mlx); fi
.venv-mac-check/bin/python -c "import mlx.core as mx; print('mlx', mx.__version__, mx.default_device())"
sysctl -n machdep.cpu.brand_string hw.memsize 2>/dev/null | paste -sd' ' - || true
for M in "${MODELS[@]}"; do
  N=$(basename "${M%/}")
  echo "== $N"
  .venv-mac-check/bin/python scripts/mac_check.py "$M" --json "mac_check_${N}.json"
done
