"""Serve a released checkpoint behind the Jev-compatible API.

    python scripts/serve.py --ckpt path/to/ckpt --host 0.0.0.0 --port 8000
    python scripts/serve.py --ckpt shgao/rsi-jev-v3.0-qwen3.5-2b --profile agent

The same as `rsi-jev serve` once the package is installed; the code is
serve/server.py. `--ckpt` is a checkpoint directory, a Hugging Face repo id or an
alias such as v3.0-2b. Set --api-key (or RSIJEV_API_KEY) to require a bearer
token on everything except the health routes.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The root must precede scripts/: THIS FILE is what shadows the serve/ package,
# so with scripts/ first "from serve.server import ..." below resolves to itself.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
