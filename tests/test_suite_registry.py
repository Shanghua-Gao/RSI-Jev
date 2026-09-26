"""The suite is twelve public benchmarks, each pinned, each named with its licence.

The point of shipping `rsijev/targets_suite.py` is that a reader can rebuild the suite
from the same upstream sources rather than take our word for the headline. That only
holds if every entry says where its split comes from, pins it, and names no path on the
machine we ran it on.

    python -m pytest tests/test_suite_registry.py -q
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsijev import targets_suite as ts                             # noqa: E402

PINNED = re.compile(r"@[0-9a-f]{6,}|:[a-z_]+/?[a-z_]*|revisions|sha|random\.Random")


def test_the_recommended_suite_is_twelve_and_sums_to_one():
    assert len(ts.RECOMMENDED) == 12
    assert round(sum(ts.RECOMMENDED.values()), 6) == 1.0


@pytest.mark.parametrize("name", sorted(ts.SUITE))
def test_every_benchmark_names_its_licence_and_origin(name):
    b = ts.SUITE[name]
    assert b.licence and b.licence.strip(), f"{name} has no licence"
    assert b.origin and b.origin.strip(), f"{name} has no origin"


@pytest.mark.parametrize("name", sorted(ts.RECOMMENDED))
def test_every_scored_benchmark_pins_its_split(name):
    """A commit, a revision or a seeded sample. 'the test split' is not reproducible."""
    assert PINNED.search(ts.SUITE[name].origin), \
        f"{name} origin is not pinned: {ts.SUITE[name].origin!r}"


def test_the_registry_names_no_filesystem():
    tree = ast.parse((ROOT / "rsijev" / "targets_suite.py").read_text())
    bad = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str)
           and re.match(r"^/(?:[\w.-]+/){2,}", n.value)]
    assert not bad, f"targets_suite.py names a filesystem: {bad}"


def test_an_unset_root_explains_itself():
    """Importing always works; a loader that needs a missing root says which one."""
    for root in (ts.KEV_ROOT, ts.NIMBLE_ROOT, ts.JEVBENCH_ROOT):
        if isinstance(root, ts._Unset):
            with pytest.raises(RuntimeError, match="is not set. It must point at"):
                root / "x"


def test_the_suite_script_lists_the_registry_without_a_checkpoint():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "suite.py"), "--list"],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-1500:]
    for name in ts.RECOMMENDED:
        assert name in r.stdout, f"--list omits {name}"


def test_the_suite_script_requires_a_checkpoint_to_score():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "suite.py")],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode != 0 and "--ckpt" in r.stderr


def test_the_two_reversed_benchmarks_are_the_documented_ones():
    """BENCHMARKS.md says only these two are scored in both option orders."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("s", ROOT / "scripts" / "suite.py")
    assert spec and spec.loader
    src = (ROOT / "scripts" / "suite.py").read_text()
    assert 'BOTH_ORDERS = {"typed_decisions_test", "mmlu_pro_1k"}' in src
