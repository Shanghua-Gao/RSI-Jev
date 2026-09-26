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

from check_release_records import leaks, offending          # noqa: E402
from load_release import artifact_name                      # noqa: E402

LEAKY = "/mnt/shared/GROUPS/some_lab/someone/project/release/v9.9/rsi-jev-2b"


def test_a_checkpoint_is_recorded_by_name_only():
    assert artifact_name(LEAKY) == "rsi-jev-2b"
    assert "/" not in artifact_name(LEAKY)
    assert artifact_name(Path(LEAKY)) == "rsi-jev-2b"


def test_the_writers_use_it():
    """No script may put a stringified path into a shipped record.

    Widened from two named scripts to all of them: the leak came back in v2.0
    because the record was cut by a writer nobody had thought to check.
    """
    bad = []
    for rel in sorted((ROOT / "scripts").glob("*.py")):
        for line in rel.read_text().splitlines():
            if '"ckpt":' in line and "str(" in line:
                bad.append(f"scripts/{rel.name}: {line.strip()}")
    assert not bad, (f"these write a raw path into a shipped record: {bad}. "
                     f"Use artifact_name() so only the checkpoint's name goes out.")


# ---------------------------------------------------------------------------
# The checker that reads what is actually staged, since a test here cannot see a
# record cut by another copy of the writer.
# ---------------------------------------------------------------------------

def test_the_checker_catches_the_leak_that_already_happened_twice():
    assert offending(LEAKY) == "absolute path"
    found = leaks({"ckpt": LEAKY, "meta": {"spec": {"corpus": LEAKY + "/corpus"}}})
    assert len(found) == 2
    assert "ckpt" in found[0] and "meta.spec.corpus" in found[1]


def test_the_checker_passes_a_record_that_only_names_things():
    """The fields a real record legitimately holds, including the two that carry a
    single slash -- flag either and the check would be turned off within a day."""
    clean = {"ckpt": "rsi-jev-v2.0-qwen3.5-2b",
             "meta": {"base_model": "Qwen/Qwen3.5-2B-Base",
                      "linear_attn_kernel": "fla-0.5.2/torch-2.7.1+cu128",
                      "spec": {"sources": "synth,td_train,mc_replay", "readout": "option_xattn"},
                      "calibration": "oof_head_scorefloor"},
             "agreement": "1.0000"}
    assert leaks(clean) == []


def test_the_checker_looks_inside_lists():
    assert leaks({"records": [{"path": LEAKY}]}) != []
