"""The four scripts that build v2.0's corpus import, start, and name no filesystem.

They were ported out of the search loop, where they were allowed to default a path
to the machine they ran on. v1.0 shipped a record carrying exactly such a path, so
the rule here is that a root is an argument or it does not exist.

    python -m pytest tests/test_corpus_builders.py -q
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILDERS = ["build_specialist_corpus.py", "build_specialist_replay_corpus.py",
            "build_suite_train_corpus.py", "corpus_gate.py",
            "build_synth_corpus.py", "build_corpus.py"]
# An absolute POSIX path of two or more segments, in a string literal.
ABSOLUTE = re.compile(r"^/(?:[\w.-]+/){1,}")


@pytest.mark.parametrize("name", BUILDERS)
def test_it_parses(name):
    ast.parse((ROOT / "scripts" / name).read_text())


@pytest.mark.parametrize("name", BUILDERS)
def test_its_cli_starts(name):
    """--help exercises argument construction, which is where a missing root shows up."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"{name} --help failed:\n{r.stderr[-1500:]}"
    assert "usage:" in r.stdout


@pytest.mark.parametrize("name", BUILDERS)
def test_it_hardcodes_no_path(name):
    """No string literal may be an absolute path. A root is passed in or absent."""
    tree = ast.parse((ROOT / "scripts" / name).read_text())
    bad = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str) and ABSOLUTE.match(n.value)]
    assert not bad, f"{name} names a filesystem: {bad}"


@pytest.mark.parametrize("name", BUILDERS)
def test_every_import_it_makes_is_one_this_repo_ships(name):
    """A script ported out of the search loop must not import the loop.

    Checked structurally rather than against a list of module names: every
    top-level import has to be the standard library, a declared dependency, or a
    module that exists here. A deny-list would both miss new modules and print
    the private tree's filenames into a public test.
    """
    src = (ROOT / "scripts" / name).read_text()
    declared = set()
    for req in ("requirements.txt", "requirements-repro.txt"):
        for line in (ROOT / req).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                declared.add(re.split(r"[=<>\[]", line)[0].strip().replace("-", "_").lower())
    here = {p.stem for p in (ROOT / "scripts").glob("*.py")}
    here |= {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    here |= {p.stem for p in (ROOT / "rsijev").glob("*.py")}
    allowed = set(sys.stdlib_module_names) | declared | here | {
        "torch", "transformers", "datasets", "safetensors", "huggingface_hub",
        "numpy", "pytest", "fastapi", "starlette", "uvicorn", "pydantic", "fla",
        # Loaded from a checkout the user points at, not installed: the suite builder
        # reads Nimble's train split through Nimble's own loader so the conversion
        # matches the one its authors use.
        "nimble"}
    missing = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [] if node.level else [(node.module or "").split(".")[0]]
        else:
            continue
        missing += [m for m in mods if m and m.lower() not in allowed and m not in allowed]
    assert not missing, (f"{name} imports {sorted(set(missing))}, which this repo does not "
                         f"ship and does not declare as a dependency")


def test_the_suite_builder_requires_its_roots():
    """--suite-dir has no default: the raw upstream archives live wherever the user put
    them, and guessing is how v1.0 shipped a lab path inside a checkpoint."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_suite_train_corpus.py"),
                        "--work", "/tmp/x", "--stage", "fetch"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "--suite-dir" in r.stderr


def test_the_gate_ships_its_reference():
    """corpus_gate compares a corpus against published reference statistics; without
    the reference file the gate silently has nothing to compare against."""
    ref = ROOT / "scripts" / "corpus_reference_stats.json"
    assert ref.exists(), "scripts/corpus_reference_stats.json is missing"
    import json
    d = json.loads(ref.read_text())
    for k in ("questions", "mode_share", "noul_true_rate", "choice_first_option_rate"):
        assert k in d, f"reference statistics have no {k}"
