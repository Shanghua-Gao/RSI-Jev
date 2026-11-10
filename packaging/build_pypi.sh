#!/usr/bin/env bash
# Build what goes to PyPI: this package under the name decideit.
# The repository stays rsi-jev, so `pip install "rsi-jev[...] @ git+..."` in the published
# model cards keeps working: pip refuses a git install whose name has changed.
#
#   packaging/build_pypi.sh [OUT_DIR]        # builds the committed tree
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-$ROOT/dist/pypi}" && cd "${1:-$ROOT/dist/pypi}" && pwd)
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

git -C "$ROOT" archive HEAD | tar -x -C "$WORK"
sed -i.orig 's/^name = "rsi-jev"$/name = "decideit"/' "$WORK/pyproject.toml"
grep -q '^name = "decideit"$' "$WORK/pyproject.toml"
(cd "$WORK" && uv build --out-dir "$OUT" >/dev/null)
ls -l "$OUT"
