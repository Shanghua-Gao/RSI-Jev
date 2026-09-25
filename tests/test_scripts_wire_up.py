"""Every shipped script must actually call the functions it imports.

`scripts/bench.py` and `scripts/calibration.py` were published calling
`load_release(..., infer_dtype=...)` against a public loader that had no such
parameter: the feature existed only in the internal tree, so both scripts died
on their first line of real work. Nothing caught it, because nothing here runs a
script without a GPU and the unit tests never touch `scripts/`.

This binds each script's calls against the real signatures, with no weights, no
GPU and no network -- the cheapest check that would have caught it.

    python -m pytest tests/test_scripts_wire_up.py -q
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))          # root first: scripts/serve.py shadows serve/

from load_release import load_release                  # noqa: E402
from serve.infer import score_questions, score_questions_cached  # noqa: E402

CHECKED = {"load_release": load_release,
           "score_questions": score_questions,
           "score_questions_cached": score_questions_cached}
SCRIPTS = sorted(p for p in (ROOT / "scripts").glob("*.py"))


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in CHECKED:
            yield node


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_call_matches_the_real_signature(script):
    tree = ast.parse(script.read_text())
    seen = 0
    for call in _calls(tree):
        sig = inspect.signature(CHECKED[call.func.id])
        args = [inspect.Parameter.empty] * len(call.args)
        kwargs = {kw.arg: inspect.Parameter.empty for kw in call.keywords if kw.arg}
        if any(kw.arg is None for kw in call.keywords):
            continue                     # **kwargs: nothing static to check
        try:
            sig.bind(*args, **kwargs)
        except TypeError as e:
            pytest.fail(f"{script.name}:{call.lineno} calls "
                        f"{call.func.id}{sig} and does not fit: {e}")
        seen += 1
    if script.name in {"bench.py", "calibration.py", "routing.py", "serve.py"}:
        assert seen, f"{script.name} is supposed to load a release and does not"
