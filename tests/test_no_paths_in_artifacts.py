"""A file that ships inside a checkpoint must not say where it was trained.

v1.0's `verify.json` shipped with the full training path in it: the filesystem,
the group, the directory layout, and an internal version number that did not match
the public one. It got there because the writer recorded the `--ckpt` argument
verbatim. The fixture below is synthetic on purpose -- pinning the real one would
republish it.

    python -m pytest tests/test_no_paths_in_artifacts.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from load_release import artifact_name                      # noqa: E402

LEAKY = "/mnt/shared/GROUPS/some_lab/someone/project/release/v9.9/rsi-jev-2b"


def test_a_checkpoint_is_recorded_by_name_only():
    assert artifact_name(LEAKY) == "rsi-jev-2b"
    assert "/" not in artifact_name(LEAKY)
    assert artifact_name(Path(LEAKY)) == "rsi-jev-2b"


def test_the_writers_use_it():
    """Both scripts that emit a shipped JSON record must go through it."""
    bad = []
    for rel, field in (("scripts/load_release.py", "ckpt"),
                       ("scripts/serve_smoke.py", "ckpt")):
        src = (ROOT / rel).read_text()
        for line in src.splitlines():
            if f'"{field}":' in line and ("str(" in line):
                bad.append(f"{rel}: {line.strip()}")
    assert not bad, (f"these write a raw path into a shipped record: {bad}. "
                     f"Use artifact_name() so only the checkpoint's name goes out.")
