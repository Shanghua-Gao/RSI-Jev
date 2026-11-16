"""What the MLX server keeps between requests (#53).

  * freed buffers stay under RSIJEV_MLX_CACHE_GB, however the request lengths vary;
  * live memory does not grow over many requests of varying length: nothing is leaked;
  * /v1/limits reports active, cached and peak memory with the budget;
  * a bad budget is refused.

    python -m pytest tests/test_mlx_cache_budget.py -q
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from test_mlx_parity import release, tok  # noqa: E402,F401  (fixtures)

QS = {"state": {"type": "choice", "instructions": "What is the terminal doing?",
                "criteria": {"working": "still running", "waiting": "waits for the user",
                             "done": "finished", "failed": "failed"}},
      "ok": {"type": "noul", "instructions": "Did it succeed?"}}


def _states():
    words = "build test deploy error waiting prompt finished traceback output line".split()
    for n in (3, 40, 7, 120, 15, 300, 2, 90, 33, 200, 5, 60) * 3:
        yield " ".join(words[i % len(words)] for i in range(n))


def test_freed_buffers_stay_under_the_cache_budget_and_nothing_leaks(release, monkeypatch):  # noqa: F811
    from rsijev.mlx.serving import memory_stats
    from serve.decider import Decider
    monkeypatch.setenv("RSIJEV_MLX_CACHE_GB", "0.002")
    monkeypatch.setenv("RSIJEV_MLX_DEVICE", "cpu")
    d = Decider(str(release[1]), backend="mlx", dtype="fp32")
    states = list(_states())
    d.decide(states[0], QS)
    first = memory_stats()["active_gb"]
    for s in states:
        d.decide(s, QS)
        m = memory_stats()
        assert m["cache_gb"] <= 0.002 + 1e-3, m
    assert memory_stats()["active_gb"] <= first + 0.002, (first, memory_stats())
    assert memory_stats()["cache_limit_gb"] == 0.002
    monkeypatch.delenv("RSIJEV_MLX_CACHE_GB")
    from rsijev.mlx.serving import configure_memory
    configure_memory()


def test_limits_report_memory(monkeypatch):
    from fastapi.testclient import TestClient

    from rsijev.mlx.serving import configure_memory, memory_stats
    from serve.app import create_app
    monkeypatch.setenv("RSIJEV_MLX_CACHE_GB", "0.5")
    monkeypatch.setenv("RSIJEV_MLX_MEMORY_GB", "64")
    before = mx.set_memory_limit(1 << 40)      # MLX's own limit, put back below
    configure_memory()
    app = create_app(lambda *a, **k: ([], 0), served_model_name="m", memory=memory_stats)
    m = TestClient(app).get("/v1/limits").json()["memory"]
    assert m["backend"] == "mlx" and m["cache_limit_gb"] == 0.5 and m["memory_limit_gb"] == 64
    assert {"active_gb", "cache_gb", "peak_gb"} <= set(m)
    monkeypatch.delenv("RSIJEV_MLX_MEMORY_GB")
    monkeypatch.delenv("RSIJEV_MLX_CACHE_GB")
    configure_memory()
    mx.set_memory_limit(before)


def test_a_bad_budget_is_refused(monkeypatch):
    from rsijev.mlx.serving import configure_memory
    monkeypatch.setenv("RSIJEV_MLX_CACHE_GB", "-1")
    with pytest.raises(ValueError, match="RSIJEV_MLX_CACHE_GB"):
        configure_memory()
