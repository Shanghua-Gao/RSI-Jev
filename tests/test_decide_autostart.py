"""`decide` with no server running: the model starts itself, is shared, and goes away."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from test_mlx_parity import release, tok  # noqa: E402,F401  (fixtures)

ROOT = Path(__file__).resolve().parent.parent
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


@pytest.fixture
def home(release):  # noqa: F811
    h = Path(tempfile.mkdtemp(prefix="rj", dir="/tmp"))     # a socket path has a short limit
    env = {**os.environ, "RSIJEV_AUTOSTART": "1", "RSIJEV_HOME": str(h), "RSIJEV_DECIDE_MODEL": str(release[1]),
           "RSIJEV_MLX_DEVICE": "cpu",
           "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
    env.pop("DECIDE_URL", None)
    yield h, env
    try:
        pid = json.loads((h / "daemon.json").read_text())["pid"]
        os.kill(pid, 15)
        time.sleep(1)
    except (OSError, ValueError):
        pass
    shutil.rmtree(h, ignore_errors=True)


def decide(env, *args, stdin="some state", cwd=ROOT):
    return subprocess.run([sys.executable, "-m", "serve.decide_cli", "q?", "--json", *args], input=stdin,
                          env=env, cwd=cwd, capture_output=True, text=True, timeout=300)


def pid(h):
    return json.loads((h / "daemon.json").read_text())["pid"]


def test_starts_answers_and_keeps_stdout_clean(home):
    h, env = home
    r = decide(env)
    assert r.returncode in (0, 1), r.stderr
    assert 0.0 <= json.loads(r.stdout)["p"] <= 1.0
    first = pid(h)
    assert decide(env).returncode in (0, 1)
    assert pid(h) == first, "the second question must reuse the running model"


def test_many_at_once_start_one_model(home):
    h, env = home
    ps = [subprocess.Popen([sys.executable, "-m", "serve.decide_cli", "q?", "--json"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT)
          for _ in range(4)]
    out = [p.communicate("state", timeout=300) for p in ps]
    assert [p.returncode in (0, 1) for p in ps] == [True] * 4, out
    assert (h / "daemon.log").read_text().count("on MLX as") == 1


def test_goes_away_when_idle(home):
    h, env = home
    env = {**env, "RSIJEV_IDLE_MINUTES": "0.02"}
    assert decide(env).returncode in (0, 1)
    end = time.monotonic() + 30
    while time.monotonic() < end and (h / "model.sock").exists():
        time.sleep(0.2)
    assert not (h / "model.sock").exists() and not (h / "daemon.json").exists()


def test_a_picture_restarts_a_text_model_with_images(home, tmp_path):
    import base64
    h, env = home
    assert decide(env).returncode in (0, 1)
    text_pid = pid(h)
    assert json.loads((h / "daemon.json").read_text())["vision"] is False
    img = tmp_path / "a.png"
    img.write_bytes(base64.b64decode(PNG))
    r = decide(env, "--image", str(img), stdin="")
    assert r.returncode in (0, 1), r.stderr
    assert pid(h) != text_pid and json.loads((h / "daemon.json").read_text())["vision"] is True


def test_a_named_server_is_not_replaced(home):
    h, env = home
    r = decide(env, "--url", "http://127.0.0.1:1")
    assert r.returncode == 2 and not (h / "daemon.json").exists()


def test_a_broken_model_is_exit_2(home):
    h, env = home
    env = {**env, "RSIJEV_DECIDE_MODEL": str(h / "does-not-exist-anywhere/nor-here/at-all")}
    r = decide(env)
    assert r.returncode == 2 and "error" in json.loads(r.stdout)


def test_the_hook_never_waits_for_a_cold_model(home):
    h, env = home
    assert decide(env).returncode in (0, 1)
    subprocess.run([sys.executable, "-c", f"import os,json;os.kill({pid(h)},15)"], check=True)
    end = time.monotonic() + 15
    while time.monotonic() < end and (h / "model.sock").exists():
        time.sleep(0.1)
    call = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls"}, "cwd": "/tmp"})
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, str(ROOT / "apps/agentguard/agentguard.py"), "hook"], input=call,
                       env=env, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "allow" not in r.stdout, r.stdout     # no answer is never an allow
    assert time.monotonic() - t0 < 5
    end = time.monotonic() + 120
    while time.monotonic() < end and not (h / "daemon.json").exists():
        time.sleep(0.2)
    assert (h / "daemon.json").exists(), "the hook starts the model for the next call"
