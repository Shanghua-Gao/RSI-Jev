#!/usr/bin/env bash
# Assemble a self-contained Space directory from this checkout. Nothing is uploaded.
#   bash space/assemble.sh /tmp/rsi-jev-space
set -euo pipefail
out=${1:?usage: space/assemble.sh OUT_DIR}
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$out/scripts" "$out/serve"
cp "$root/space/app.py" "$root/space/requirements.txt" "$root/space/README.md" "$out/"
cp -r "$root/rsijev" "$out/"
cp "$root/serve/__init__.py" "$root/serve/wire.py" "$root/serve/infer.py" \
   "$root/serve/examples.json" "$root/serve/README.md" "$out/serve/"
cp "$root/scripts/load_release.py" "$out/scripts/"
find "$out" -name __pycache__ -type d -prune -exec rm -rf {} +
echo "assembled $out"
