"""`decide`: grep for judgment.

    git diff | decide "Does this change touch authentication?" --min-p 0.9 && notify
    cat ticket.txt | decide "Who should handle this?" --choice billing,technical,other
    decide "Is the build green?" --image screenshot.png

Reads the state from stdin and asks one question of a running server. Exit status:
0 the probability reached --min-p, 1 it did not, 2 anything went wrong. No answer is
ever 0: a failed judgment must not read as yes.

Imports only the standard library, and only what it needs: no torch, no transformers.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time

YES, NO, ERROR = 0, 1, 2
DEFAULT_URL = "http://127.0.0.1:8000"
MAX_IMAGES = 4
KEY = "q"
SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


class Failure(Exception):
    """A judgment could not be made. Always exit 2."""


def data_url(path: str) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise Failure(f"cannot read image {path}: {e.strerror or e}") from e
    mime = next((m for sig, m in SIGNATURES if data.startswith(sig)), None)
    if mime is None and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        mime = "image/webp"
    if mime is None:
        raise Failure(f"{path}: not a PNG, JPEG or WebP image")
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def build_request(question: str, state: str, *, options: list[str] | None = None,
                  images: list[str] | None = None, effort: str | None = None) -> dict:
    if options:
        q = {"type": "choice", "instructions": question, "criteria": {o: o for o in options}}
    else:
        q = {"type": "noul", "instructions": question}
    body: dict = {"model": "jev-latest", "state": state, "questions": {KEY: q}}
    if images:
        body["images"] = images
    if effort:
        body["effort"] = effort
    return body


def post(url: str, body: dict, *, timeout: float, api_key: str | None = None) -> dict:
    # Imported here: the argument errors do not pay for it. No proxies: a local server
    # must not be reached through whatever the shell has set.
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone",
                                 data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise Failure(f"server said {e.code}: {_server_message(e)}") from e
    except (urllib.error.URLError, OSError) as e:
        raise Failure(f"cannot reach {url}: {getattr(e, 'reason', e)}") from e
    try:
        return json.loads(raw)
    except ValueError as e:
        raise Failure(f"{url} did not answer with JSON") from e


def _server_message(e) -> str:
    try:
        raw = e.read().decode("utf-8", "replace")
        d = json.loads(raw)
    except Exception:
        return str(e.reason)
    if isinstance(d, dict) and isinstance(d.get("error"), dict):
        return str(d["error"].get("message"))
    return str(d.get("detail") if isinstance(d, dict) else d)[:300]


def _probability(x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) \
            or not 0.0 <= x <= 1.0:
        raise Failure(f"server gave {x!r} where a probability belongs")
    return float(x)


def read_answer(response: dict, options: list[str] | None) -> tuple[float, str | None]:
    """(probability, chosen option) from a response. A noul answer is P(true); a choice
    is scored by the option's own probability, never by `confidence`."""
    try:
        answer = response["answers"][KEY]
    except (KeyError, TypeError) as e:
        raise Failure("the response has no answer to the question") from e
    if not isinstance(answer, dict):
        raise Failure("the response has no answer to the question")
    if not options:
        return _probability(answer.get("noul")), None
    choice, probs = answer.get("choice"), answer.get("probabilities")
    if choice not in options or not isinstance(probs, dict) or choice not in probs:
        raise Failure(f"server chose {choice!r}, which is not one of the options")
    return _probability(probs[choice]), choice


def parse_options(text: str) -> list[str]:
    options = [o.strip() for o in text.split(",")]
    if len(options) < 2 or not all(options) or len(set(options)) != len(options):
        raise Failure("--choice needs at least two different options: a,b,c")
    return options


def read_state(has_images: bool) -> str:
    if sys.stdin is None or sys.stdin.isatty():
        if has_images:
            return ""
        raise Failure("nothing on stdin to judge")
    data = sys.stdin.buffer.read().decode("utf-8", "replace")
    if not data.strip() and not has_images:
        raise Failure("stdin is empty, so there is nothing to judge")
    return data


def ask(a: argparse.Namespace, body: dict, *, vision: bool) -> dict:
    """A named server if there is one, else the local model, started when needed."""
    from serve import autostart
    if a.url or not autostart.available():
        return post(a.url or DEFAULT_URL, body, timeout=a.timeout, api_key=os.environ.get("RSIJEV_API_KEY"))
    try:
        return autostart.ask(body, vision=vision, timeout=a.timeout)
    except autostart.Unavailable as e:
        raise Failure(str(e)) from e


def _min_p(text: str) -> float:
    v = float(text)
    if not 0.0 <= v <= 1.0:          # NaN fails this too
        raise argparse.ArgumentTypeError(f"{text} is not between 0 and 1")
    return v


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="decide", description=(
        "Read a state from stdin, ask the model one question. "
        "Exit 0 if the probability reaches --min-p, 1 if not, 2 on any error."))
    ap.add_argument("question")
    ap.add_argument("--min-p", type=_min_p, default=0.5, help="default 0.5")
    ap.add_argument("--choice", metavar="A,B,C", help="pick one option; prints the winner")
    ap.add_argument("--image", action="append", default=[], metavar="PATH",
                    help=f"repeatable, at most {MAX_IMAGES}")
    ap.add_argument("--effort", choices=("low", "medium", "high", "auto"))
    ap.add_argument("--url", default=os.environ.get("DECIDE_URL"),
                    help="a running server (default $DECIDE_URL); on Apple Silicon the model "
                         f"otherwise starts by itself, elsewhere the default is {DEFAULT_URL}")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds, default 60")
    ap.add_argument("--json", action="store_true", help="one JSON object on stdout")
    return ap


def judge(a: argparse.Namespace) -> dict:
    if len(a.image) > MAX_IMAGES:
        raise Failure(f"{len(a.image)} images: at most {MAX_IMAGES}")
    options = parse_options(a.choice) if a.choice else None
    images = [data_url(p) for p in a.image]
    state = read_state(bool(images))
    t0 = time.perf_counter()
    body = build_request(a.question, state, options=options, images=images, effort=a.effort)
    response = ask(a, body, vision=bool(images))
    ms = (time.perf_counter() - t0) * 1000
    p, choice = read_answer(response, options)
    usage = response.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return {"p": p, "choice": choice, "pass": p >= a.min_p, "min_p": a.min_p,
            "depth": usage.get("depth"), "model": response.get("model"), "ms": round(ms, 1)}


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)     # argparse exits 2 on bad arguments
    try:
        r = judge(a)
        if a.json:
            print(json.dumps(r), flush=True)
        elif r["choice"] is not None:
            print(f"{r['choice']} {r['p']:.3f}", flush=True)
        else:
            print(f"{r['p']:.3f}", file=sys.stderr, flush=True)
        return YES if r["pass"] else NO
    except Failure as e:
        message = str(e)
    except KeyboardInterrupt:
        message = "interrupted"
    except Exception as e:                  # never let a bug look like an answer
        message = f"{type(e).__name__}: {e}"
    print(f"decide: {message}", file=sys.stderr)
    if a.json:
        try:
            print(json.dumps({"error": message, "pass": False}), flush=True)
        except OSError:
            pass
    return ERROR


if __name__ == "__main__":
    sys.exit(main())
