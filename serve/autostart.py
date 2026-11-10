"""Start the local model when `decide` needs it, so nobody has to run a server.

The standard library only: `decide` imports this. The model itself runs in serve/daemon.py,
a separate process that listens on a socket in ~/.rsi-jev and exits when it has been idle
(RSIJEV_IDLE_MINUTES, default 10), which hands its memory back.
"""
from __future__ import annotations

import contextlib
import http.client
import importlib.util
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "shgao/rsi-jev-v7.0-vl-3b-mlx-4bit"
START_TIMEOUT = 300.0        # a busy Mac can take minutes to page 2 GB in
STOP_TIMEOUT = 15.0


class Unavailable(Exception):
    """The local model could not answer."""


def home() -> Path:
    h = Path(os.environ.get("RSIJEV_HOME") or "~/.rsi-jev").expanduser()
    h.mkdir(mode=0o700, parents=True, exist_ok=True)
    return h


def model_ref() -> str:
    return os.environ.get("RSIJEV_DECIDE_MODEL") or DEFAULT_MODEL


def available() -> bool:
    """Whether this machine runs the model through this path (Apple Silicon with MLX)."""
    if os.environ.get("RSIJEV_AUTOSTART") == "1":
        return True
    return (sys.platform == "darwin" and platform.machine() == "arm64"
            and importlib.util.find_spec("mlx") is not None)


class _Unix(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


def _request(method: str, route: str, body: bytes | None, timeout: float) -> tuple[int, bytes]:
    c = _Unix(str(home() / "model.sock"), timeout)
    try:
        c.request(method, route, body=body, headers={"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


def alive() -> bool:
    try:
        return _request("GET", "/health/live", None, 2.0)[0] == 200
    except (OSError, http.client.HTTPException):
        return False


def _read(name: str) -> dict:
    try:
        d = json.loads((home() / name).read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


@contextlib.contextmanager
def _lock():
    """One starter at a time: pipelines run several `decide` at once."""
    import fcntl
    with open(home() / "lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _fetched(vision: bool) -> bool:
    m = _read("model.json")
    return (m.get("ref") == model_ref() and Path(str(m.get("path"))).is_dir()
            and (bool(m.get("vision")) or not vision))


def _python() -> list[str]:
    """This interpreter, isolated from the user's Python settings if we are."""
    return [sys.executable] + (["-I"] if sys.flags.isolated else [])


def _say(text: str) -> None:
    if sys.stderr is not None and sys.stderr.isatty():
        print(f"decide: {text}", file=sys.stderr, flush=True)


def _fetch(vision: bool) -> None:
    if _fetched(vision):
        return
    print("decide: downloading the model, once "
          + ("(about 0.2 GB more for images)" if _fetched(False) else "(about 1.7 GB)"),
          file=sys.stderr, flush=True)
    cmd = _python() + ["-m", "serve.daemon", "fetch"] + (["--vision"] if vision else [])
    # stdout belongs to the answer
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=sys.stderr.fileno(), check=False)
    if r.returncode != 0 or not _fetched(vision):
        raise Unavailable("the model could not be downloaded; check the network and try again")


def _log_tail(n: int = 400) -> str:
    try:
        return (home() / "daemon.log").read_bytes()[-n:].decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _spawn(vision: bool) -> subprocess.Popen:
    with open(home() / "daemon.log", "ab") as log:
        return subprocess.Popen(_python() + ["-m", "serve.daemon", "run"] + (["--vision"] if vision else []),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                cwd=home(), start_new_session=True)


def _wait(proc: subprocess.Popen, timeout: float) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if alive():
            return
        if proc.poll() is not None:
            raise Unavailable(f"the model stopped while loading: {_log_tail()}")
        time.sleep(0.1)
    raise Unavailable(f"the model did not start in {timeout:.0f} s; see {home() / 'daemon.log'}")


def stop() -> None:
    pid = _read("daemon.json").get("pid")
    if not isinstance(pid, int) or not alive():
        return
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)
    end = time.monotonic() + STOP_TIMEOUT
    while time.monotonic() < end and alive():
        time.sleep(0.05)


def _ready(vision: bool) -> bool:
    return alive() and (not vision or bool(_read("daemon.json").get("vision")))


def ensure(vision: bool = False, *, timeout: float = START_TIMEOUT) -> None:
    if _ready(vision):
        return
    with _lock():
        if _ready(vision):
            return
        stop()                       # text only, and images were asked for
        _fetch(vision)
        _say("starting the model")
        _wait(_spawn(vision), timeout)


def kick() -> None:
    """Start the model in the background if it is on disk. Never waits, never downloads."""
    if alive() or not _fetched(False):
        return
    import fcntl
    with open(home() / "lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return                   # someone is already starting it
        if not alive():
            _spawn(False)


def post(body: dict, *, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    status, raw = _request("POST", "/v1/systemone", data, timeout)
    try:
        answer = json.loads(raw)
    except ValueError as e:
        raise Unavailable(f"the model failed ({status}): {_log_tail()}") from e
    if status != 200:
        error = answer.get("error") if isinstance(answer, dict) else None
        detail = error.get("message") if isinstance(error, dict) else answer
        raise Unavailable(f"the model said {status}: {str(detail)[:300]}")
    return answer


def ask(body: dict, *, vision: bool = False, timeout: float = 60.0) -> dict:
    """The model's response to `body`, starting (and first downloading) it when needed."""
    for retry in (True, False):
        ensure(vision)
        try:
            return post(body, timeout=timeout)
        except (ConnectionError, FileNotFoundError, http.client.RemoteDisconnected) as e:
            if not retry:                # it went idle between the check and the question
                raise Unavailable(f"the model went away: {e}") from e
        except (OSError, http.client.HTTPException) as e:
            raise Unavailable(f"the model did not answer: {e}") from e
    raise Unavailable("the model did not answer")
