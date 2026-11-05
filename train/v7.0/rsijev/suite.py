"""The frozen multi-benchmark eval suite (run owner, 2026-09-25: "good performance
on all datasets"), DI-disjoint since eval_suite_v3 / eval_final_v3 (runbook A77,
KNOWN_BUGS KB-25, owner-approved evaluator rebuild 2026-10-08).

typed-decisions test and the knowledge guard kg_v1 come from targets.py. The other
benchmarks are read from a frozen, decontaminated dump: every file is sha256-pinned
in manifest.json, eval cases whose text also appears in any training corpus we hold
were dropped when the dump was frozen, and (v3) every row that is also a Decision
Index kit row (kit03 + kit 0.2.1; results/audit-di-guard-overlap.exclude.json plus
the v3 receipt a77_overlap_check.json) was dropped. mmlu_pro_1k is gone from the
suite: all of its rows are DI MMLU-Pro test rows. Nothing here downloads anything,
so a worker scores exactly what was frozen.

Suite targets are scored in canonical option order only, and without a
zero-shot control; typed-decisions keeps both orders and the control
(run_arm_lib.FULL_EVAL_TARGETS).

eval_final_v3 (FINAL_DIR, load_final_set) is the frozen held-out set (runbook A15)
minus its DI rows: efv2_legalbench, efv2_conditionalqa and the 327 non-DI rows of
efv2_bfcl (efv2_bbh and efv2_musr were 100% DI rows and are gone). Its targets are
named "final.<benchmark>", scored like suite targets, and kept out of the suite
weights. It IS used as a guard and release bar ("final_clean"), which is why it
must stay DI-disjoint (A77).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .contract import Case, load_cases

SUITE_DIR = Path(os.environ.get(
    "RSIJEV_EVAL_SUITE",
    "<data_root>/eval_suite_v3"))
FINAL_DIR = Path(os.environ.get(
    "RSIJEV_EVAL_FINAL",
    "<data_root>/eval_final_v3"))
# Target-name prefix of the final set; as_cluster keeps it out of the suite weights (final_clean is pooled separately).
FINAL_PREFIX = "final."


def manifest(root: Path = SUITE_DIR) -> dict:
    return json.loads((root / "manifest.json").read_text())


def weights(root: Path = SUITE_DIR) -> dict[str, float]:
    """Suite weights, keyed by target name as it appears in the items files."""
    return dict(manifest(root)["weights"])


def _load_verified(root: Path, what: str) -> dict[str, list[Case]]:
    out: dict[str, list[Case]] = {}
    for name, b in manifest(root)["benchmarks"].items():
        p = root / b["file"]
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != b["sha256"]:
            raise RuntimeError(f"{what} file {p} changed: sha256 {got} != {b['sha256']}")
        out[name] = load_cases(str(p))
    return out


def load_eval_suite(root: Path = SUITE_DIR) -> dict[str, list[Case]]:
    """The non-typed-decisions benchmarks (13 in eval_suite_v3), sha256-verified."""
    return _load_verified(root, "eval suite")


def load_final_set(root: Path = FINAL_DIR) -> dict[str, list[Case]]:
    """eval_final_v3, sha256-verified, keyed by benchmark name as in its manifest
    (efv2_bfcl, efv2_conditionalqa, efv2_legalbench)."""
    return _load_verified(root, "eval final set")
