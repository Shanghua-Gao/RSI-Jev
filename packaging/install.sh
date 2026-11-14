#!/bin/sh
# Install RSI-Jev on an Apple Silicon Mac or on Linux (x86_64, aarch64). Nothing else needs to be installed first.
#   curl -fsSL https://github.com/Shanghua-Gao/RSI-Jev/releases/latest/download/install.sh | sh
# Remove it with: rm -rf ~/.rsi-jev ~/.local/bin/decide ~/.local/bin/decideit ~/.local/bin/agentguard ~/.local/bin/rsi-jev
set -eu

version=7.0.0

main() {
    case "$(uname -s)" in
        Darwin) mac ;;
        Linux) linux ;;
        *) die "this installer is for macOS and Linux; elsewhere see the README" ;;
    esac
}

mac() {
    base=${RSIJEV_DOWNLOAD:-https://github.com/Shanghua-Gao/RSI-Jev/releases/download/v$version}
    name=rsi-jev-$version-macos-arm64
    app=${RSIJEV_HOME:-$HOME/.rsi-jev}/app
    bin=${RSIJEV_BIN:-$HOME/.local/bin}

    [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = 1 ] || die "this needs an Apple Silicon Mac"
    major=$(sw_vers -productVersion | cut -d. -f1)
    [ "$major" -ge 14 ] || die "this needs macOS 14 or newer"

    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    echo "Downloading RSI-Jev (about 90 MB)..."
    curl -fL --progress-bar --proto '=https' -o "$tmp/$name.tar.gz" "$base/$name.tar.gz"
    curl -fsSL --proto '=https' -o "$tmp/$name.tar.gz.sha256" "$base/$name.tar.gz.sha256"
    (cd "$tmp" && shasum -a 256 -c "$name.tar.gz.sha256" >/dev/null) || die "the download is corrupt; try again"

    tar -xzf "$tmp/$name.tar.gz" -C "$tmp"
    mkdir -p "$(dirname "$app")" "$bin"
    rm -rf "$app.old"
    [ ! -e "$app" ] || mv "$app" "$app.old"
    mv "$tmp/$name" "$app"
    rm -rf "$app.old"
    # So the first decide does not spend seconds compiling.
    "$app/python/bin/python3" -I -m compileall -q "$app/python/lib/python3.12" >/dev/null 2>&1 || true
    for c in decide decideit agentguard rsi-jev; do ln -sf "$app/bin/$c" "$bin/$c"; done
    on_path "$HOME/.zprofile"
    echo
    echo "Done. The model (about 1.7 GB) downloads the first time you use it:"
    echo "  echo 'The build failed on line 3.' | decide 'Did the build pass?' && echo yes || echo no"
}

on_path() {
    case ":$PATH:" in
        *":$bin:"*) ;;
        *)
            profile=$1
            line="export PATH=\"$bin:\$PATH\""
            grep -qsF "$line" "$profile" || printf '\n%s\n' "$line" >> "$profile"
            echo "Added $bin to PATH in $profile; open a new terminal to use it."
            ;;
    esac
}

# Linux: a private Python (fetched by uv) with RSI-Jev and PyTorch in ~/.rsi-jev/app. With an NVIDIA
# driver PyTorch is the matching CUDA build (about 3 GB more), otherwise the CPU build.
linux() {
    uv_version=0.12.2
    case "$(uname -m)" in
        x86_64) triple=x86_64-unknown-linux-gnu; uv_sha=d66e96b5f1ca3b99806eee283a8125d33a0bd669e6e6d9bc4ab7ffda63c41bf4 ;;
        aarch64|arm64) triple=aarch64-unknown-linux-gnu; uv_sha=19b7f1f66895261fbaa07f8ea91da0f86337ad4e47efa594e87641c1718ffc52 ;;
        *) die "this installer supports x86_64 and aarch64 Linux; elsewhere: pip install, see the README" ;;
    esac
    home=${RSIJEV_HOME:-$HOME/.rsi-jev}
    app=$home/app
    bin=${RSIJEV_BIN:-$HOME/.local/bin}
    source=${RSIJEV_SOURCE:-https://github.com/Shanghua-Gao/RSI-Jev/archive/refs/tags/v$version.tar.gz}
    command -v curl >/dev/null || die "curl is needed"

    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    echo "Downloading the installer's tools..."
    curl -fsSL --proto '=https' -o "$tmp/uv.tar.gz" \
        "https://github.com/astral-sh/uv/releases/download/$uv_version/uv-$triple.tar.gz"
    echo "$uv_sha  $tmp/uv.tar.gz" | sha256sum -c - >/dev/null || die "the download is corrupt; try again"
    tar -xzf "$tmp/uv.tar.gz" -C "$tmp"
    uv=$tmp/uv-$triple/uv

    extras=vision
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        extras=vision,fast
        echo "NVIDIA GPU found: installing the CUDA build of PyTorch (about 4 GB)..."
    else
        echo "No NVIDIA GPU found: installing the CPU build of PyTorch (about 1 GB)..."
    fi
    mkdir -p "$home" "$bin"
    rm -rf "$app.old"
    [ ! -e "$app" ] || mv "$app" "$app.old"
    if ! install_linux "$uv" "$extras" "$source"; then
        rm -rf "$app"
        [ ! -e "$app.old" ] || mv "$app.old" "$app"
        die "the install failed; nothing was changed"
    fi
    rm -rf "$app.old"
    for c in decide decideit agentguard rsi-jev; do ln -sf "$app/bin/$c" "$bin/$c"; done
    on_path "$HOME/.profile"
    echo
    echo "Done. The model (about 6.8 GB) downloads the first time you use it:"
    echo "  echo 'The build failed on line 3.' | decide 'Did the build pass?' && echo yes || echo no"
}

install_linux() {
    # Built where it stays: a venv records its own path.
    UV_PYTHON_INSTALL_DIR="$app/python" UV_NO_CONFIG=1 "$1" venv --quiet --python 3.12 --managed-python "$app/venv" || return 1
    UV_NO_CONFIG=1 "$1" pip install --quiet --no-cache --python "$app/venv/bin/python" --torch-backend auto \
        "rsi-jev[$2] @ $3" || return 1
    mkdir -p "$app/bin"
    for pair in decide:serve.decide_cli decideit:serve.decide_cli rsi-jev:serve.cli agentguard:agentguard.agentguard; do
        c=${pair%%:*}
        cat > "$app/bin/$c" <<WRAPPER
#!/bin/sh
RSIJEV_AUTOSTART=1 exec "$app/venv/bin/python" -I -m ${pair#*:} "\$@"
WRAPPER
        chmod +x "$app/bin/$c"
    done
    "$app/bin/decide" --help >/dev/null
}

die() { echo "rsi-jev: $1" >&2; exit 1; }

main "$@"
