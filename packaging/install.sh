#!/bin/sh
# Install RSI-Jev on an Apple Silicon Mac. Nothing else needs to be installed first.
#   curl -fsSL https://github.com/Shanghua-Gao/RSI-Jev/releases/latest/download/install.sh | sh
# Remove it with: rm -rf ~/.rsi-jev ~/.local/bin/decide ~/.local/bin/decideit ~/.local/bin/agentguard ~/.local/bin/rsi-jev
set -eu

main() {
    version=7.0.0
    base=${RSIJEV_DOWNLOAD:-https://github.com/Shanghua-Gao/RSI-Jev/releases/download/v$version}
    name=rsi-jev-$version-macos-arm64
    app=${RSIJEV_HOME:-$HOME/.rsi-jev}/app
    bin=${RSIJEV_BIN:-$HOME/.local/bin}

    [ "$(uname -s)" = Darwin ] || die "this installer is for Macs; on Linux see the README"
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
    for c in decide decideit agentguard rsi-jev; do ln -sf "$app/bin/$c" "$bin/$c"; done

    case ":$PATH:" in
        *":$bin:"*) ;;
        *)
            profile=$HOME/.zprofile
            line="export PATH=\"$bin:\$PATH\""
            grep -qsF "$line" "$profile" || printf '\n%s\n' "$line" >> "$profile"
            echo "Added $bin to PATH in $profile; open a new terminal to use it."
            ;;
    esac
    echo
    echo "Done. The model (about 1.7 GB) downloads the first time you use it:"
    echo "  echo 'The build failed on line 3.' | decide 'Did the build pass?' && echo yes || echo no"
}

die() { echo "rsi-jev: $1" >&2; exit 1; }

main "$@"
