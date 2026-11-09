#!/usr/bin/env bash
# Build the self-contained Apple Silicon bundle: its own Python, MLX and this package, no torch.
# Runs on any machine with uv (nothing here executes macOS code).
#
#   packaging/build_mac_bundle.sh [OUT_DIR]
#
# Writes rsi-jev-<version>-macos-arm64.tar.gz and its .sha256.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-$ROOT/dist}" && cd "${1:-$ROOT/dist}" && pwd)
PY_RELEASE=20261026
PY_VERSION=3.12.16
PY_SHA256=353d5f1b4fd2715e7053f12f068435d5bbad16b91e057006b5d01e0925709a7e
PY_NAME=cpython-$PY_VERSION+$PY_RELEASE-aarch64-apple-darwin-install_only_stripped.tar.gz
export MACOSX_DEPLOYMENT_TARGET=14.0       # MLX ships no wheels for older macOS

WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)
NAME=rsi-jev-$VERSION-macos-arm64
APP=$WORK/$NAME

curl -fsSL --proto '=https' -o "$WORK/python.tar.gz" \
    "https://github.com/astral-sh/python-build-standalone/releases/download/$PY_RELEASE/${PY_NAME//+/%2B}"
echo "$PY_SHA256  $WORK/python.tar.gz" | sha256sum -c - >/dev/null
mkdir -p "$APP"
tar -xzf "$WORK/python.tar.gz" -C "$APP"

(cd "$ROOT" && uv build --wheel --out-dir "$WORK/wheel" >/dev/null)
SITE=$APP/python/lib/python3.12/site-packages
uv pip install --quiet --python-platform aarch64-apple-darwin --python-version 3.12 --only-binary :all: \
    --target "$SITE" "$WORK"/wheel/*.whl

# What nobody runs: pip, its console scripts for the build machine, tests, docs, Tcl/Tk, idle
rm -rf "$SITE"/pip "$SITE"/pip-* "$SITE"/bin "$APP"/python/lib/python3.12/test \
       "$APP"/python/lib/python3.12/idlelib "$APP"/python/lib/python3.12/tkinter \
       "$APP"/python/lib/tcl* "$APP"/python/lib/tk* "$APP"/python/lib/itcl* "$APP"/python/lib/thread* \
       "$APP"/python/share
find "$APP" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$SITE" -type d \( -name tests -o -name test \) -prune -exec rm -rf {} +

mkdir -p "$APP/bin"
wrap() {
    cat > "$APP/bin/$1" <<WRAPPER
#!/bin/sh
exec "\$(dirname "\$(readlink -f "\$0")")/../python/bin/python3" -I -m $2 "\$@"
WRAPPER
    chmod +x "$APP/bin/$1"
}
wrap decide serve.decide_cli
wrap rsi-jev serve.cli
wrap agentguard agentguard.agentguard

if find "$SITE" -maxdepth 1 \( -iname 'torch*' -o -iname 'transformers*' \) | grep -q .; then
    echo "torch or transformers got into the bundle" >&2; exit 1
fi
test -f "$SITE/serve/decide_cli.py" && test -f "$SITE/agentguard/agentguard.py" && test -d "$SITE/mlx"

tar -C "$WORK" -czf "$OUT/$NAME.tar.gz" "$NAME"
(cd "$OUT" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
du -sh "$APP" "$OUT/$NAME.tar.gz"
