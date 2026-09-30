"""How fast is it on YOUR machine?

    python scripts/bench.py                      # v3.0-2B, downloads on first run
    python scripts/bench.py --model v1.0-0.8b

The same as `rsi-jev bench` once the package is installed; the code is
serve/bench.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The repository root must precede scripts/: scripts/serve.py would otherwise
# shadow the serve/ PACKAGE, and "from serve.bench import ..." fails with
# "serve is not a package".
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve.bench import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
