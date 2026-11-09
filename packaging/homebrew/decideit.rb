# The Homebrew formula. Copy to the tap, with the url and sha256 of the release's bundle:
#   brew install shanghua-gao/tap/decideit
class Decideit < Formula
  desc "Local judgment model for your shell: decide, agentguard"
  homepage "https://github.com/Shanghua-Gao/RSI-Jev"
  url "https://github.com/Shanghua-Gao/RSI-Jev/releases/download/v7.0.0/rsi-jev-7.0.0-macos-arm64.tar.gz"
  version "7.0.0"
  sha256 "@SHA256@"
  license "MIT"

  depends_on arch: :arm64
  depends_on macos: :sonoma

  # The bundle carries its own signed Python and MLX: leave its binaries as they are.
  skip_clean "libexec"

  def install
    libexec.install Dir["*"]
    # so the first decide does not spend seconds compiling; a file that will not compile is harmless
    quiet_system libexec/"python/bin/python3", "-I", "-m", "compileall", "-q", libexec/"python/lib/python3.12"
    %w[decide decideit agentguard rsi-jev].each { |c| bin.install_symlink libexec/"bin"/c }
  end

  def caveats
    "The model (about 1.7 GB) downloads the first time you run decide."
  end

  test do
    assert_match "usage: decide", shell_output("#{bin}/decide --help")
  end
end
