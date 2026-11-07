"""`decide` against a fake server, as a real subprocess: exit codes are its interface."""
from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")


class Fake:
    """Answers each POST with `reply(body)` -> (status, raw bytes)."""

    def __init__(self):
        self.bodies, self.reply = [], lambda body: (200, b"{}")
        fake = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.bodies.append(body)
                status, raw = fake.reply(body)
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.http.server_port}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()


@pytest.fixture
def fake():
    f = Fake()
    yield f
    f.http.shutdown()


def answers(answer, **usage):
    return 200, json.dumps({"model": "m", "answers": {"q": answer}, "usage": usage}).encode()


def run(*args, stdin="some state", url=None):
    cmd = [sys.executable, "-m", "serve.decide_cli", *args]
    if url:
        cmd += ["--url", url]
    return subprocess.run(cmd, input=stdin.encode(), cwd=ROOT, capture_output=True, timeout=30)


@pytest.mark.parametrize("p,min_p,code", [(0.95, "0.9", 0), (0.9, "0.9", 0), (0.89, "0.9", 1)])
def test_noul_exit_follows_min_p(fake, p, min_p, code):
    fake.reply = lambda b: answers({"type": "noul", "noul": p})
    r = run("q?", "--min-p", min_p, url=fake.url)
    assert r.returncode == code
    assert r.stdout == b"" and r.stderr.decode().strip() == f"{p:.3f}"
    q = fake.bodies[0]["questions"]["q"]
    assert q == {"type": "noul", "instructions": "q?"}
    assert fake.bodies[0]["state"] == "some state"


def test_default_min_p_is_a_half(fake):
    fake.reply = lambda b: answers({"type": "noul", "noul": 0.49})
    assert run("q?", url=fake.url).returncode == 1
    fake.reply = lambda b: answers({"type": "noul", "noul": 0.5})
    assert run("q?", url=fake.url).returncode == 0


def test_choice_routes_on_the_option_probability_not_confidence(fake):
    # confidence 0.99 would pass; the option's own probability does not.
    fake.reply = lambda b: answers({"type": "choice", "choice": "b", "confidence": 0.99,
                                    "probabilities": {"a": 0.4, "b": 0.6}})
    r = run("q?", "--choice", "a,b", "--min-p", "0.8", url=fake.url)
    assert r.returncode == 1 and r.stdout.decode().strip() == "b 0.600"
    assert set(fake.bodies[0]["questions"]["q"]["criteria"]) == {"a", "b"}


def test_json(fake):
    fake.reply = lambda b: answers({"type": "noul", "noul": 0.7}, depth=[20])
    r = run("q?", "--json", url=fake.url)
    d = json.loads(r.stdout)
    assert r.returncode == 0 and d["p"] == 0.7 and d["pass"] is True and d["depth"] == [20]


def test_effort_and_images_are_sent(fake, tmp_path):
    fake.reply = lambda b: answers({"type": "noul", "noul": 0.7})
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    assert run("q?", "--effort", "low", "--image", str(img), stdin="", url=fake.url).returncode == 0
    body = fake.bodies[0]
    assert body["effort"] == "low"
    assert body["images"][0].startswith("data:image/png;base64,")


@pytest.mark.parametrize("reply", [
    lambda b: (500, b'{"error": {"message": "boom"}}'),
    lambda b: (422, b'{"detail": [{"msg": "bad"}]}'),
    lambda b: (503, b"overloaded"),
    lambda b: (200, b"not json"),
    lambda b: (200, b"[]"),
    lambda b: (200, b'{"answers": {}}'),
    lambda b: (200, b'{"answers": {"q": null}}'),
    lambda b: answers({"type": "noul"}),
    lambda b: answers({"type": "noul", "noul": True}),     # a bool is not P(true)
    lambda b: answers({"type": "noul", "noul": "0.9"}),
    lambda b: answers({"type": "noul", "noul": 1.5}),
    lambda b: answers({"type": "noul", "noul": -0.1}),
    lambda b: (200, b'{"answers": {"q": {"noul": NaN}}}'),
])
def test_every_bad_answer_is_exit_2(fake, reply):
    fake.reply = reply
    r = run("q?", "--min-p", "0", url=fake.url)     # min-p 0 would pass anything real
    assert r.returncode == 2 and r.stderr.startswith(b"decide:")


def test_a_choice_outside_the_options_is_exit_2(fake):
    fake.reply = lambda b: answers({"type": "choice", "choice": "c", "probabilities": {"c": 1.0}})
    assert run("q?", "--choice", "a,b", "--min-p", "0", url=fake.url).returncode == 2


def test_unreachable_server_is_exit_2():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    r = run("q?", "--min-p", "0", url=f"http://127.0.0.1:{port}")
    assert r.returncode == 2 and b"cannot reach" in r.stderr


def test_json_error_on_stdout_says_it_did_not_pass():
    r = run("q?", "--json", url="http://127.0.0.1:1")
    assert r.returncode == 2
    assert json.loads(r.stdout)["pass"] is False


@pytest.mark.parametrize("args,stdin", [
    (("q?",), ""),
    (("q?",), "   \n"),
    (("q?", "--choice", "a"), "x"),
    (("q?", "--choice", "a,a"), "x"),
    (("q?", "--choice", "a,,b"), "x"),
    (("q?", "--image", "/does/not/exist.png"), "x"),
    (("q?", "--min-p", "1.5"), "x"),
    (("q?", "--min-p", "nan"), "x"),
    (("q?", "--effort", "extreme"), "x"),
    ((), "x"),
])
def test_bad_use_is_exit_2_and_asks_nothing(fake, args, stdin):
    fake.reply = lambda b: answers({"type": "noul", "noul": 1.0})
    assert run(*args, stdin=stdin, url=fake.url).returncode == 2
    assert fake.bodies == []


def test_more_than_four_images_is_exit_2(fake, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    r = run("q?", *sum((("--image", str(img)) for _ in range(5)), ()), url=fake.url)
    assert r.returncode == 2 and fake.bodies == []


def test_a_file_that_is_not_an_image_is_exit_2(fake, tmp_path):
    bad = tmp_path / "a.png"
    bad.write_bytes(b"hello")
    assert run("q?", "--image", str(bad), url=fake.url).returncode == 2


def test_binary_stdin_does_not_crash(fake):
    fake.reply = lambda b: answers({"type": "noul", "noul": 0.7})
    p = subprocess.run([sys.executable, "-m", "serve.decide_cli", "q?", "--url", fake.url],
                       input=b"\xff\xfe\x00bin", cwd=ROOT, capture_output=True, timeout=30)
    assert p.returncode == 0


def test_client_path_imports_no_torch():
    code = ("import sys, serve.decide_cli as d; d.post; "
            "bad = {'torch', 'transformers', 'numpy', 'fastapi', 'rsijev'} & set(sys.modules); "
            "sys.exit(repr(bad) if bad else 0)")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    assert subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT).returncode == 0


def test_console_script_is_declared():
    assert 'decide = "serve.decide_cli:main"' in (ROOT / "pyproject.toml").read_text()
