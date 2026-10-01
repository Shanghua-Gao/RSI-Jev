"""Side (a): RSI-Jev v4.0-VL through `rsi-jev serve`, over HTTP. Writes rsi.json.

    rsi-jev serve v4.0-vl-2b --port 8000          # in another terminal
    python examples/chat_vs_rsi/rsi_client.py [http://127.0.0.1:8000] [--out rsi.json]

Per question: 3 warm calls, then 9 timed calls; the result is the p50 of the 9. The whole
HTTP call is timed: JSON, base64 image decode, resize, tokenise and the forward pass.
Leave RSIJEV_DOC_CACHE unset (off by default) so a repeated request is not answered
from a document cache.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))        # from a clone, uninstalled


def load_questions(path: Path = HERE / "questions.json") -> list[dict]:
    return json.loads(path.read_text())


def images_of(q: dict) -> list:
    """The question's images: file paths, or for VisA the photo scaled to fit max_side
    (what the VisA harness and the published run sent)."""
    paths = [HERE / p for p in q["images"]]
    if not q.get("max_side"):
        return [str(p) for p in paths]
    from PIL import Image
    out = []
    for p in paths:
        s = q["max_side"]
        im = Image.open(p)
        im.draft("RGB", (s, s))
        im = im.convert("RGB")
        im.thumbnail((s, s), Image.LANCZOS)
        out.append(im)
    return out


def request_body(q: dict) -> dict:
    from serve.images import to_data_url
    return {"model": "jev-latest", "state": q["state"], "questions": {q["key"]: q["spec"]},
            "images": [to_data_url(im) for im in images_of(q)]}


def pick(answer: dict) -> str:
    if answer["type"] == "noul":
        return "true" if answer["noul"] > 0.5 else "false"
    return answer["choice"]


def call(base: str, body: dict) -> tuple[dict, float]:
    req = urllib.request.Request(base + "/v1/systemone", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.perf_counter()
    out = json.loads(urllib.request.urlopen(req).read())
    return out, (time.perf_counter() - t) * 1000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("server", nargs="?", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=str(HERE / "rsi.json"))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    base = a.server.rstrip("/")
    qs = load_questions()[: a.limit or None]
    limits = json.loads(urllib.request.urlopen(base + "/v1/limits").read())
    if not limits.get("images", {}).get("supported"):
        raise SystemExit("this server's model does not take images")
    res, out = [], None
    for q in qs:
        body = request_body(q)
        for _ in range(3):
            call(base, body)
        ts = []
        for _ in range(9):
            out, ms = call(base, body)
            ts.append(ms)
        ans = out["answers"][q["key"]]
        p = pick(ans)
        res.append(dict(id=q["id"], answer=ans, pick=p, correct=p in q["ref"], valid=True,
                        confidence_available=True, ms_p50=round(statistics.median(ts), 1),
                        ms_all=[round(t, 1) for t in ts], input_tokens=out["usage"]["input_tokens"]))
        print(q["id"], res[-1]["ms_p50"], "ms", p, q["ref"], json.dumps(ans)[:160], flush=True)
    Path(a.out).write_text(json.dumps(dict(limits=limits, model=out["model"] if out else None, results=res), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
