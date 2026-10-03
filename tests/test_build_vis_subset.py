"""scripts/build_vis_subset.py: capped, seeded, whole-row samples per vision source."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_vis_subset.py"
sys.path.insert(0, str(ROOT / "scripts"))
import build_vis_subset as bvs  # noqa: E402


def _roots(d: Path, n_rows: int = 40, soft: str | None = None) -> None:
    for root, srcs in bvs.SOURCES.items():
        (d / root).mkdir(parents=True, exist_ok=True)
        for s in srcs:
            rows = []
            for i in range(n_rows):
                g = [0.5, 0.5] if s == soft else [1.0, 0.0]
                rows.append({"case_id": f"{s}:{i}", "images": [f"{i}.png"],
                             "questions": [{"key": "q"}] * (1 + i % 2), "gold": {"q": g}})
            (d / root / f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _run(d: Path, out: Path, *args: str) -> dict:
    subprocess.run([sys.executable, str(SCRIPT), "--root", str(d), "--out", str(out), *args],
                   check=True, capture_output=True)
    return json.loads((out / "manifest.json").read_text())


def test_fixed_cap_and_seeded_prefixes(tmp_path):
    _roots(tmp_path / "data")
    lo = _run(tmp_path / "data", tmp_path / "lo", "--cap", "10")
    hi = _run(tmp_path / "data", tmp_path / "hi", "--cap", "20", "--weight", "vis5_abstain=3")
    for s, v in lo["files"].items():
        lim = 10 * bvs.WEIGHT.get(s, 1.0)                         # built-in weights (vis2_iconqa x2)
        assert lim <= v["questions"] <= lim + 1                    # whole rows: may pass the cap by one row
        a = (tmp_path / "lo" / f"{s}.jsonl").read_text().splitlines()
        b = (tmp_path / "hi" / f"{s}.jsonl").read_text().splitlines()
        assert b[: len(a)] == a                                   # a lower cap is a prefix of a higher one
    assert hi["files"]["vis5_abstain"]["questions"] >= 55          # weight 3 x cap 20 (60 q available)


def test_soft_gold_refused_except_sokoban(tmp_path):
    _roots(tmp_path / "ok", soft="vis6_sokoban")
    _run(tmp_path / "ok", tmp_path / "o1", "--cap", "5")
    _roots(tmp_path / "bad", soft="vis_ai2d")
    with pytest.raises(subprocess.CalledProcessError):
        _run(tmp_path / "bad", tmp_path / "o2", "--cap", "5")
