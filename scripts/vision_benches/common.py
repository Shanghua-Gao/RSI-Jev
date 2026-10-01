"""Shared by the benchmark scripts: the model client, the pinned upstream code, small stats.

Every benchmark here is scored through the public API only: `rsijev.Decider` in-process
(the default), or `--server URL` for a running `rsi-jev serve`. Both run the same
request validation and the same answer code, and both return the served
probabilities (bf16 tower on CUDA, release calibration applied).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))                       # from a clone, uninstalled

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_IMAGES = 4                                      # what one request may carry (serve/images.py)

# Third-party harness code, read at the commits the published numbers were made with.
# Their item builders, prompts and statistics are imported from these checkouts, not copied.
UPSTREAM = {
    "jev-omni-eval": ("https://github.com/CondadosAI/jev-omni-eval",
                      "5a8b6d105688d273fcbc1e278385214676749518"),       # Apache-2.0
    "jev-omni-inspection": ("https://github.com/CondadosAI/jev-omni-inspection",
                            "25da8c77527e9303f709ccb24a94ee46ccd19351"),  # Apache-2.0
    "laya-vision": ("https://github.com/r33drichards/laya-vision",
                    "d4075b09d31d5c80fa52c349f89702a2e2abe2e3"),          # Apache-2.0
}


def cache_dir() -> Path:
    return Path(os.environ.get("RSIJEV_BENCH_CACHE", Path.home() / ".cache" / "rsi-jev-benches"))


def upstream(name: str, root: Path | None = None) -> Path:
    """A checkout of `name` at its pinned commit (cloned on first use)."""
    url, commit = UPSTREAM[name]
    d = (root or cache_dir()) / name
    if not (d / ".git").exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", url, str(d)], check=True)
    head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != commit:
        subprocess.run(["git", "-C", str(d), "fetch", "--quiet", "origin"], check=False)
        subprocess.run(["git", "-C", str(d), "checkout", "--quiet", commit], check=True)
    return d


def load_module(name: str, path: Path):
    """Import a file under a private name (their `analyze.py` must not shadow anything)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------- the model
class Client:
    """One typed request -> its answers, through `Decider` or over HTTP."""

    def __init__(self, model: str | None = None, server: str | None = None):
        self.server = server.rstrip("/") if server else None
        if self.server is None:
            from rsijev import Decider
            self.decider = Decider(model or "v4.0-vl-2b")
            self.name = repr(self.decider)
        else:
            lim = json.loads(urllib.request.urlopen(self.server + "/v1/limits").read())
            if not lim.get("images", {}).get("supported"):
                raise SystemExit(f"{self.server}: this model does not take images")
            self.name = self.server

    def ask(self, state, questions: dict, images: list | None = None) -> tuple[dict, float]:
        """(answers, milliseconds for the whole call)."""
        t = time.perf_counter()
        if self.server is None:
            answers = self.decider.decide(state, questions, images=images or None)
        else:
            body = request_body(state, questions, images)
            req = urllib.request.Request(self.server + "/v1/systemone", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            answers = json.loads(urllib.request.urlopen(req, timeout=600).read())["answers"]
        return answers, (time.perf_counter() - t) * 1000


def request_body(state, questions: dict, images: list | None = None) -> dict:
    """The `POST /v1/systemone` body; images (PIL images, paths or bytes) go as data URLs."""
    body = {"model": "jev-latest", "state": state, "questions": questions}
    if images:
        from serve.images import to_data_url
        body["images"] = [to_data_url(im) for im in images]
    return body


def choice_question(instructions: str, options: list[str]) -> dict:
    """A letter-keyed choice: the vision training format and the upstream harnesses' format."""
    return {"type": "choice", "instructions": instructions,
            "criteria": dict(zip(LETTERS[: len(options)], options))}


def answer_probs(answer: dict, options: list[str] | None = None) -> list[float]:
    """The probabilities in option order. A noul answer is [p(false), p(true)]."""
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    pr = answer["probabilities"]
    return [pr[k] for k in (options if options is not None else pr)]


def add_client_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", default="v4.0-vl-2b", help="alias, Hub repo id or checkpoint dir (in-process Decider)")
    ap.add_argument("--server", help="URL of a running `rsi-jev serve`, instead of --model")


# ---------------------------------------------------------------------------- records
def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.open() if line.strip()}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def ece(conf, hit, bins: int) -> float:
    """Top-label ECE over `bins` equal-width confidence bins (a confidence of 1 goes in the top bin)."""
    groups: dict[int, list[int]] = {}
    for i, c in enumerate(conf):
        groups.setdefault(min(int(c * bins), bins - 1), []).append(i)
    n = len(conf)
    return sum(len(ix) / n * abs(sum(hit[i] for i in ix) / len(ix) - sum(conf[i] for i in ix) / len(ix))
               for ix in groups.values()) if n else float("nan")


def log_odds(p_yes: float, p_no: float, floor: float = 1e-12) -> float:
    return math.log(max(p_yes, floor)) - math.log(max(p_no, floor))
