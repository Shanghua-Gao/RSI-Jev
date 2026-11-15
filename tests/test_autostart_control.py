"""serve/autostart.py's control flow, with no model: which daemon runs, how long a caller waits.

  * a daemon serving another model (or another backend) is replaced, never silently used,
    and a request names the model it wants, so a daemon serving another one refuses it (#49);
  * --start-timeout is one budget for the starter lock, a first download and the start (#50);
  * one daemon per home for its whole life, loading included: a starter that finds one
    loading waits for it instead of starting a second (#51).

    python -m pytest tests/test_autostart_control.py -q
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from serve import autostart as a
from serve import daemon

WANT = "shgao/rsi-jev-v7.0-vl-3b-mlx-4bit"
OLD = "shgao/rsi-jev-v6.1-vl-4b-mlx-4bit"


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "h"
    h.mkdir()
    monkeypatch.setenv("RSIJEV_HOME", str(h))
    monkeypatch.setenv("RSIJEV_DECIDE_MODEL", WANT)
    monkeypatch.setenv("RSIJEV_BACKEND", "mlx")
    return h


class Life:
    """Holds daemon.lock the way a loading or serving daemon does."""

    def __init__(self, h: Path, record: dict | None):
        self.f = open(h / "daemon.lock", "a")
        fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if record is not None:
            (h / "daemon.json").write_text(json.dumps({"pid": os.getpid(), "vision": False,
                                                       "backend": "mlx", **record}))

    def release(self):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


# ------------------------------------------------------------------- #49 which model answers
def test_a_live_daemon_serving_another_model_is_replaced(home):
    life = Life(home, {"model": OLD})
    stopped = []

    def stop():
        stopped.append(True)
        life.release()

    with patch.object(a, "alive", return_value=True), patch.object(a, "stop", side_effect=stop), \
         patch.object(a, "_fetch"), patch.object(a, "_spawn") as spawn, patch.object(a, "_wait"):
        a.ensure()
    assert stopped and spawn.call_count == 1


def test_a_live_daemon_on_another_backend_is_replaced(home):
    life = Life(home, {"model": WANT, "backend": "torch"})     # noqa: F841  (held while it lives)
    with patch.object(a, "alive", return_value=True):
        assert a.running() and not a._ready(False)


def test_the_right_live_daemon_is_used(home):
    life = Life(home, {"model": WANT})     # noqa: F841  (held while it lives)
    with patch.object(a, "alive", return_value=True), patch.object(a, "_spawn") as spawn, \
         patch.object(a, "_fetch") as fetch:
        a.ensure()
    assert spawn.call_count == 0 and fetch.call_count == 0


def test_a_request_names_the_model_and_a_wrong_daemon_refuses_it(home):
    sent = []

    def request(method, route, body, timeout):
        sent.append(json.loads(body))
        return 400, json.dumps({"error": {"message": f"Unknown model: {WANT}"}}).encode()

    with patch.object(a, "_request", side_effect=request):
        with pytest.raises(a._OtherModel):
            a.post({"model": "jev-latest", "state": "s", "questions": {}}, timeout=1)
    assert sent[0]["model"] == WANT


def test_ask_restarts_once_when_the_daemon_turns_out_to_serve_another_model(home):
    answers = iter([a._OtherModel("no"), {"answers": {}}])

    def post(body, timeout):
        x = next(answers)
        if isinstance(x, Exception):
            raise x
        return x

    with patch.object(a, "ensure") as ensure, patch.object(a, "post", side_effect=post):
        assert a.ask({}, timeout=1) == {"answers": {}}
    assert ensure.call_count == 2


# ------------------------------------------------------------------- #50 one start budget
def test_start_timeout_bounds_the_wait_for_another_starter(home):
    held = threading.Event()

    def hold():
        with open(home / "lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            held.set()
            time.sleep(1.0)

    t = threading.Thread(target=hold)
    t.start()
    held.wait()
    t0 = time.monotonic()
    with patch.object(a, "_ready", return_value=False), patch.object(a, "post", return_value={}):
        with pytest.raises(a.Unavailable, match="start-timeout"):
            a.ask({}, timeout=60, start_timeout=0.1)
    assert time.monotonic() - t0 < 0.5
    t.join()


def test_start_timeout_bounds_a_download(home):
    def run(*args, timeout=None, **kw):
        assert timeout is not None and timeout <= 0.2
        raise subprocess.TimeoutExpired(args[0], timeout)

    with patch.object(a, "_fetched", return_value=False), patch.object(a.subprocess, "run", side_effect=run):
        with pytest.raises(a.Unavailable, match="download"):
            a._fetch(False, time.monotonic() + 0.2)


def test_without_start_timeout_a_download_has_no_limit(home):
    seen = {}

    def ensure(vision, *, deadline, fetch_deadline):
        seen["fetch"] = fetch_deadline
        seen["start"] = deadline - time.monotonic()

    with patch.object(a, "ensure", side_effect=ensure), patch.object(a, "post", return_value={}):
        a.ask({}, timeout=5)
    assert seen["fetch"] is None and seen["start"] > a.START_TIMEOUT - 5


def test_retries_share_one_start_budget(home):
    deadlines = []

    def ensure(vision, *, deadline, fetch_deadline):
        deadlines.append(deadline)

    with patch.object(a, "ensure", side_effect=ensure), \
         patch.object(a, "post", side_effect=[ConnectionError("gone"), {}]):
        a.ask({}, timeout=5, start_timeout=30)
    assert len(deadlines) == 2 and deadlines[0] == deadlines[1]


def test_decide_takes_start_timeout():
    from serve.decide_cli import build_parser
    assert build_parser().parse_args(["q", "--start-timeout", "2.5"]).start_timeout == 2.5
    with pytest.raises(SystemExit):
        build_parser().parse_args(["q", "--start-timeout", "0"])


# ------------------------------------------------------------------- #51 one daemon
LOADING = """
import fcntl, json, os, sys, time
h = sys.argv[1]
f = open(os.path.join(h, "daemon.lock"), "a")
fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
json.dump({"pid": os.getpid(), "vision": False, "model": sys.argv[2], "backend": "mlx",
           "state": "loading"}, open(os.path.join(h, "daemon.json"), "w"))
open(os.path.join(h, "loading"), "w").close()
time.sleep(float(sys.argv[3]))
"""


def _loading_child(h: Path, seconds: float = 30.0) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-c", LOADING, str(h), WANT, str(seconds)])
    end = time.monotonic() + 10
    while not (h / "loading").exists() and time.monotonic() < end:
        time.sleep(0.02)
    return p


def test_kick_then_ensure_starts_one_daemon(home):
    spawned = []

    def spawn(vision):
        spawned.append(_loading_child(home))
        return spawned[-1]

    became_alive = {}

    def alive():                     # the first daemon answers half a second after its start
        if spawned and "t" not in became_alive:
            became_alive["t"] = time.monotonic() + 0.5
        return bool(spawned) and time.monotonic() >= became_alive["t"]

    try:
        with patch.object(a, "_fetched", return_value=True), patch.object(a, "_spawn", side_effect=spawn), \
             patch.object(a, "alive", side_effect=alive):
            a.kick()
            a.kick()                 # a second kick while the first is loading
            a.ensure(timeout=10)
        assert len(spawned) == 1
    finally:
        for p in spawned:
            p.kill()
            p.wait()


def test_a_second_daemon_exits_at_once_when_one_is_loading(home):
    p = _loading_child(home)
    try:
        t0 = time.monotonic()
        assert daemon.run(False) == 0      # never reads model.json, never loads
        assert time.monotonic() - t0 < 1.0
        assert json.loads((home / "daemon.json").read_text())["pid"] == p.pid   # left alone
    finally:
        p.kill()
        p.wait()


def test_stop_reaches_a_daemon_that_is_still_loading(home):
    p = _loading_child(home)
    try:
        assert a.running()
        a.stop()
        assert not a.running() and p.wait(timeout=10) is not None
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()


def test_a_starter_waits_for_a_loading_daemon_of_the_right_model(home):
    p = _loading_child(home, seconds=5.0)
    calls = {"n": 0}

    def alive():
        calls["n"] += 1
        return calls["n"] > 5

    try:
        with patch.object(a, "_spawn") as spawn, patch.object(a, "alive", side_effect=alive):
            a.ensure(timeout=10)
        assert spawn.call_count == 0
    finally:
        p.kill()
        p.wait()


def test_a_starter_stops_a_loading_daemon_of_another_model(home):
    p = subprocess.Popen([sys.executable, "-c", LOADING, str(home), OLD, "30"])
    end = time.monotonic() + 10
    while not (home / "loading").exists() and time.monotonic() < end:
        time.sleep(0.02)
    try:
        with patch.object(a, "_fetch"), patch.object(a, "_spawn") as spawn, patch.object(a, "_wait"):
            a.ensure(timeout=10)
        assert p.wait(timeout=10) is not None and spawn.call_count == 1
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
