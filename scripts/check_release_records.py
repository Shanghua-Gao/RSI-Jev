"""Refuse to publish a checkpoint whose records say where it was trained.

    python scripts/check_release_records.py --ckpt DIR        # exit 1 if anything leaks

v1.0's `verify.json` went out carrying a lab filesystem path, the group name and an
internal version number. `load_release.py` grew `artifact_name()` to stop that -- but
the fix only protects records written BY THIS TREE, and v2.0's were first cut by an
older copy of the writer, which reproduced the same leak. A test in this repo cannot
see that copy. This script can: point it at the staged directory before upload, and it
reads what is actually there.

The rules are deliberately about shape, not about any particular string, so nothing
internal has to be named here to be caught:

  * a value that is an absolute path (`/...`, `~/...`, `C:\\...`)
  * a value with three or more `/` separators, which no legitimate field has --
    `base_model` and `linear_attn_kernel` each hold exactly one
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ABSOLUTE = re.compile(r"^(?:/|~/|[A-Za-z]:[\\/])")
DEEP = re.compile(r"(?:[^/]*/){3,}")


def offending(value: str) -> str | None:
    if ABSOLUTE.match(value):
        return "absolute path"
    if DEEP.search(value):
        return "path-like: three or more '/' separators"
    return None


def walk(obj, trail: str = ""):
    """Every string leaf, with a dotted trail naming where it sits."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{trail}.{k}" if trail else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, f"{trail}[{i}]")
    elif isinstance(obj, str):
        yield trail, obj


def leaks(record) -> list[str]:
    """Every (field, reason, value) in one parsed record, as readable lines."""
    out = []
    for trail, value in walk(record):
        why = offending(value)
        if why:
            out.append(f"{trail}: {why}: {value}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="a staged checkpoint directory, checked before upload")
    a = ap.parse_args()
    files = sorted(a.ckpt.glob("*.json"))
    if not files:
        print(f"  no .json records in {a.ckpt} -- nothing to check, which is suspicious")
        return 1
    bad = 0
    for f in files:
        try:
            found = leaks(json.loads(f.read_text()))
        except ValueError as e:
            print(f"  {f.name}: not readable JSON: {e}")
            bad += 1
            continue
        if found:
            bad += 1
            print(f"  {f.name}: {len(found)} leak(s)")
            for line in found:
                print(f"      {line}")
        else:
            print(f"  {f.name}: clean")
    if bad:
        print(f"\n  {bad} of {len(files)} records would publish a path. Rewrite them with "
              f"the checkpoint's NAME (see artifact_name in scripts/load_release.py).")
        return 1
    print(f"\n  all {len(files)} records clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
