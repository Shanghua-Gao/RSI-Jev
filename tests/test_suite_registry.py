"""The suites are public benchmarks, each pinned, each named with its licence.

v2 is the twelve that v2.0 and v2.1 report; v3, the default, is the fifteen that v3.0
and v4.0-VL report.

The point of shipping `rsijev/targets_suite.py` is that a reader can rebuild the suite
from the same upstream sources rather than take our word for the headline. That only
holds if every entry says where its split comes from, pins it, and names no path on the
machine we ran it on.

    python -m pytest tests/test_suite_registry.py -q
"""
from __future__ import annotations

import ast
import builtins
import re
import symtable
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsijev import targets_suite as ts                             # noqa: E402

PINNED = re.compile(r"@[0-9a-f]{6,}|:[a-z_]+/?[a-z_]*|revisions|sha|random\.Random")


def test_the_v2_suite_is_twelve_and_sums_to_one():
    assert len(ts.SUITES["v2"]) == 12
    assert round(sum(ts.SUITES["v2"].values()), 6) == 1.0


def test_the_v3_suite_is_fifteen_and_sums_to_one():
    assert len(ts.SUITES["v3"]) == 15
    assert round(sum(ts.SUITES["v3"].values()), 6) == 1.0
    assert set(ts.SUITES["v2"]) < set(ts.SUITES["v3"])


def test_the_default_suite_is_the_one_the_current_releases_report():
    assert ts.DEFAULT_SUITE == "v3"
    assert ts.RECOMMENDED is ts.SUITES["v3"]


@pytest.mark.parametrize("suite", sorted(ts.SUITES))
def test_every_suite_benchmark_is_in_the_registry(suite):
    assert set(ts.SUITES[suite]) <= set(ts.SUITE)


def test_the_v3_weights_reproduce_the_v3_card():
    """v3.0's record, section 4.2: its per-benchmark top-1 and its suite mean 0.756.
    On the twelve-benchmark weights the same numbers give 0.751, which is the gap
    issue #13 reported."""
    v3 = {"typed_decisions_test": 0.791, "nimble_public": 0.801, "mmlu_pro_1k": 0.364,
          "jev_style_panel": 0.835, "kev_transfer_v4": 0.792, "kev_hard_v1": 0.760,
          "jevbench_public": 0.700, "tasksource_jev_test": 0.701, "semif_external": 0.921,
          "scienthoon_ood": 0.705, "nimble_holdout": 0.778, "kev_documents_v1": 0.869,
          "kev_devtools_v1": 0.715, "procedural_test": 0.871, "open_jev_ood": 0.838}
    assert round(sum(w * v3[k] for k, w in ts.SUITES["v3"].items()), 3) == 0.756
    assert round(sum(w * v3[k] for k, w in ts.SUITES["v2"].items()), 3) == 0.751


def _undefined_globals(path: Path, namespace: dict) -> list[tuple[str, int, str]]:
    """Names a function reads as globals that the module never defines.

    Python only finds these when the line runs, so a loader that is not exercised by
    a test ships broken (issue #14: `LAB` in the HANS loader)."""
    known = set(namespace) | set(dir(builtins))
    bad = []

    def walk(table):
        for child in table.get_children():
            for sym in child.get_symbols():
                if sym.is_global() and sym.is_referenced() and sym.get_name() not in known:
                    bad.append((child.get_name(), child.get_lineno(), sym.get_name()))
            walk(child)

    walk(symtable.symtable(path.read_text(), str(path), "exec"))
    return bad


def test_no_suite_loader_reads_an_undefined_global():
    import importlib
    mods = {importlib.import_module(b.loader.__module__) for b in ts.SUITE.values()}
    for mod in mods:
        bad = _undefined_globals(Path(mod.__file__), vars(mod))
        assert not bad, f"{mod.__name__} reads undefined globals: {bad}"


@pytest.mark.parametrize("name", sorted(ts.SUITE))
def test_every_benchmark_names_its_licence_and_origin(name):
    b = ts.SUITE[name]
    assert b.licence and b.licence.strip(), f"{name} has no licence"
    assert b.origin and b.origin.strip(), f"{name} has no origin"


@pytest.mark.parametrize("name", sorted(set().union(*ts.SUITES.values())))
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
    assert "suite v3: 15 benchmarks" in r.stdout
    for name in ts.SUITES["v3"]:
        assert name in r.stdout, f"--list omits {name}"


def test_the_suite_script_lists_the_v2_suite_on_request():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "suite.py"), "--list", "--suite", "v2"],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "suite v2: 12 benchmarks" in r.stdout
    assert "scienthoon_ood" not in r.stdout


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


def test_the_hans_loader_caches_under_the_user_cache(tmp_path, monkeypatch):
    """Issue #14: the HANS task named a lab-only `LAB` and crashed on a clean checkout.
    Download and checksum stubbed: the file lands under $RSIJEV_BENCH_CACHE and the
    task builds from it."""
    import types
    import urllib.request
    rows = ["gold_label\tsentence1\tsentence2\tpairID"]
    rows += [f"{'entailment' if i % 2 else 'non-entailment'}\tThe doctor saw the lawyer {i}."
             f"\tThe lawyer saw the doctor {i}.\tex{i}" for i in range(20)]
    fetched = []

    def fake_retrieve(url, dest):
        assert url == ts.HANS_URL
        fetched.append(Path(dest))
        Path(dest).write_text("\n".join(rows) + "\n")

    monkeypatch.setenv("RSIJEV_BENCH_CACHE", str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_retrieve)
    monkeypatch.setattr(ts, "_checked", lambda path, sha: path)
    # the panel imports `datasets` for its other tasks; HANS does not use it
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=None))

    cases = ts.load_jev_style_panel(tasks=["hans"])
    assert len(cases) == 20 and {c.source for c in cases} == {"jevstyle_hans"}
    cache = tmp_path / "suite_refs" / "hans_heuristics_evaluation_set.txt"
    assert cache.exists() and not cache.with_name(cache.name + ".part").exists()
    assert len(fetched) == 1 and fetched[0].parent == cache.parent
    gold = {c.case_id: c.gold["label"] for c in cases}
    assert gold["jevstyle_hans:ex1"] == (0.0, 1.0) and gold["jevstyle_hans:ex0"] == (1.0, 0.0)

    ts.load_jev_style_panel(tasks=["hans"])     # a second call reads the cache
    assert len(fetched) == 1
