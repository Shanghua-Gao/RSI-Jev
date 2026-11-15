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
TORCH_MODEL = "shgao/rsi-jev-v7.0-vl-3b"     # elsewhere (the Linux installer sets RSIJEV_AUTOSTART=1)
START_TIMEOUT = 300.0        # a busy Mac can take minutes to page 2 GB in
STOP_TIMEOUT = 15.0


class Unavailable(Exception):
    """The local model could not answer."""


class _OtherModel(Unavailable):
    """The running daemon serves another model than the one asked for."""


def home() -> Path:
    h = Path(os.environ.get("RSIJEV_HOME") or "~/.rsi-jev").expanduser()
    h.mkdir(mode=0o700, parents=True, exist_ok=True)
    return h


def backend() -> str:
    """The model behind `decide` runs on MLX on Apple Silicon, on PyTorch elsewhere."""
    chosen = os.environ.get("RSIJEV_BACKEND", "").strip().lower()
    if chosen in ("mlx", "torch"):
        return chosen
    mac = sys.platform == "darwin" and platform.machine() == "arm64"
    return "mlx" if mac and importlib.util.find_spec("mlx") is not None else "torch"


def model_ref() -> str:
    return os.environ.get("RSIJEV_DECIDE_MODEL") or (DEFAULT_MODEL if backend() == "mlx" else TORCH_MODEL)


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
def _lock(deadline: float | None = None):
    """One starter at a time: pipelines run several `decide` at once. Waits until `deadline`."""
    import fcntl
    with open(home() / "lock", "w") as f:
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise Unavailable("another decide is starting the model and it is taking longer "
                                      "than --start-timeout") from None
                time.sleep(0.05)
        yield


def running() -> bool:
    """Whether a daemon holds its lifetime lock: loading or serving."""
    import fcntl
    try:
        f = open(home() / "daemon.lock", "a")
    except OSError:
        return False
    with f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def _record() -> dict:
    """What the running daemon serves; nothing when no daemon holds the lifetime lock."""
    return _read("daemon.json") if running() else {}


def _serves(rec: dict, vision: bool) -> bool:
    """Whether a daemon's record is the model this caller asked for, with images if needed."""
    return (rec.get("model") == model_ref() and rec.get("backend", "mlx") == backend()
            and (not vision or bool(rec.get("vision"))))


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    left = deadline - time.monotonic()
    if left <= 0:
        raise Unavailable("the model did not start within --start-timeout")
    return left


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


def _fetch(vision: bool, deadline: float | None = None) -> None:
    if _fetched(vision):
        return
    if backend() == "mlx":
        size = "(about 0.2 GB more for images)" if _fetched(False) else "(about 1.7 GB)"
    else:
        size = "(about 6.8 GB)"
    print(f"decide: downloading the model, once {size}", file=sys.stderr, flush=True)
    cmd = _python() + ["-m", "serve.daemon", "fetch"] + (["--vision"] if vision else [])
    # stdout belongs to the answer
    try:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=sys.stderr.fileno(), check=False,
                           timeout=_remaining(deadline))
    except subprocess.TimeoutExpired:
        raise Unavailable("the model download did not finish within --start-timeout; "
                          "it resumes where it stopped next time") from None
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


def _wait(proc: subprocess.Popen | None, deadline: float) -> None:
    """Until the daemon answers. `proc` is the one this caller started, or None when another
    caller's daemon is loading. A child that finds a daemon already loading exits at once (0);
    then the wait is for that one."""
    while time.monotonic() < deadline:
        if alive():
            return
        if proc is not None and proc.poll() is not None:
            if proc.returncode != 0 or not running():
                raise Unavailable(f"the model stopped while loading: {_log_tail()}")
            proc = None
        elif proc is None and not running():
            raise Unavailable(f"the model stopped while loading: {_log_tail()}")
        time.sleep(0.1)
    raise Unavailable(f"the model did not start in time; see {home() / 'daemon.log'}")


def stop() -> None:
    """Stop the daemon, loading or serving, and wait until it has let go of its lock."""
    if not running():
        return
    pid = _read("daemon.json").get("pid")
    if not isinstance(pid, int):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with contextlib.suppress(OSError):
            os.kill(pid, sig)
        end = time.monotonic() + STOP_TIMEOUT
        while time.monotonic() < end and running():
            time.sleep(0.05)
        if not running():
            return
    raise Unavailable(f"the running model (pid {pid}) did not stop")


def _ready(vision: bool) -> bool:
    return _serves(_record(), vision) and alive()


def ensure(vision: bool = False, *, timeout: float = START_TIMEOUT, deadline: float | None = None,
           fetch_deadline: float | None = None) -> None:
    """The model asked for (model_ref(), backend(), images if `vision`) running and answering.

    One deadline covers waiting for another starter, stopping a daemon that serves something
    else, and the start; `fetch_deadline` bounds a first download (None: no limit)."""
    if deadline is None:
        deadline = time.monotonic() + timeout
    if _ready(vision):
        return
    with _lock(deadline):
        if _ready(vision):
            return
        rec = _record()
        if rec and _serves(rec, vision):
            _say("waiting for the model to start")
            _wait(None, deadline)    # another caller's daemon is loading what we want
            return
        stop()                       # another model, or text only and images were asked for
        _fetch(vision, fetch_deadline)
        _remaining(deadline)
        _say("starting the model")
        _wait(_spawn(vision), deadline)


def kick() -> None:
    """Start the model in the background if it is on disk. Never waits, never downloads,
    never replaces a daemon that is loading or serving."""
    if running() or not _fetched(False):
        return
    import fcntl
    with open(home() / "lock", "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return                   # someone is already starting it
        if not running():
            _spawn(False)            # a second child that loses the lifetime lock exits at once


def post(body: dict, *, timeout: float) -> dict:
    """Ask the running daemon. The request names the model asked for, so a daemon that serves
    another one refuses it instead of answering in its place."""
    body = {**body, "model": model_ref()}
    data = json.dumps(body).encode("utf-8")
    status, raw = _request("POST", "/v1/systemone", data, timeout)
    try:
        answer = json.loads(raw)
    except ValueError as e:
        raise Unavailable(f"the model failed ({status}): {_log_tail()}") from e
    if status != 200:
        error = answer.get("error") if isinstance(answer, dict) else None
        detail = error.get("message") if isinstance(error, dict) else answer
        if str(detail).startswith("Unknown model"):
            raise _OtherModel(f"the running model is not {model_ref()}")
        raise Unavailable(f"the model said {status}: {str(detail)[:300]}")
    return answer


def ask(body: dict, *, vision: bool = False, timeout: float = 60.0,
        start_timeout: float | None = None) -> dict:
    """The model's response to `body`, starting (and first downloading) it when needed.

    `timeout` bounds each answer. `start_timeout` is one budget for everything before it,
    retries included: waiting for another caller, a first download and the start. Without it a
    download has no limit and the start gets START_TIMEOUT."""
    now = time.monotonic()
    deadline = now + (START_TIMEOUT if start_timeout is None else start_timeout)
    fetch_deadline = None if start_timeout is None else deadline
    for retry in (True, False):
        ensure(vision, deadline=deadline, fetch_deadline=fetch_deadline)
        try:
            return post(body, timeout=timeout)
        except _OtherModel:
            if not retry:                # another caller switched it back in between
                raise
        except (ConnectionError, FileNotFoundError, http.client.RemoteDisconnected) as e:
            if not retry:                # it went idle between the check and the question
                raise Unavailable(f"the model went away: {e}") from e
        except (OSError, http.client.HTTPException) as e:
            raise Unavailable(f"the model did not answer: {e}") from e
    raise Unavailable("the model did not answer")
