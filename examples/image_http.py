"""The same checks over HTTP, against `rsi-jev serve`.

    pip install "rsi-jev[vision]"
    rsi-jev serve v4.0-vl-2b --port 8000           # in another terminal
    python examples/image_http.py [http://127.0.0.1:8000]

Each request is what this curl sends (the image goes as a base64 data URL):

    IMG=$(base64 -w0 chart.png)
    curl localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
      "model": "jev-latest",
      "state": "Weekly product dashboard: <image>",
      "images": ["data:image/png;base64,'"$IMG"'"],
      "questions": {
        "trend": {"type": "choice", "instructions": "Is this chart'"'"'s trend up, down or flat?",
                  "criteria": {"up": "It rises.", "down": "It falls.", "flat": "It stays level."}}
      }
    }'

The script first asks GET /v1/limits whether the model takes images at all.

Output (v4.0-VL on an HP ZGX Nano, bf16; probabilities shortened):

    images supported: True, up to 4 per request, 1024 tokens per question
    chart       {"trend": {"type": "choice", "choice": "down", "probabilities": {"up": 0.0006, "down": 0.9952, "flat": 0.0042}, "confidence": 0.9928}}   76 ms, 288 input tokens
    screenshot  {"error_dialog": {"type": "noul", "noul": 0.9942}}   84 ms, 346 input tokens
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # from a clone, uninstalled

from serve.demo_images import chart, screenshot                    # noqa: E402
from serve.images import to_data_url                               # noqa: E402

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path).read())


def post(body):
    req = urllib.request.Request(BASE + "/v1/systemone", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.perf_counter()
    out = json.loads(urllib.request.urlopen(req).read())
    return out, (time.perf_counter() - t) * 1000


lim = get("/v1/limits")["images"]
if not lim["supported"]:
    raise SystemExit(f"this server's model does not take images: {lim.get('reason', 'text-only')}")
print(f"images supported: True, up to {lim['max_images']} per request, "
      f"{lim['image_token_budget']} tokens per question")

for name, (image, _), state, questions in [
    ("chart", chart(), "Weekly product dashboard: <image>",
     {"trend": {"type": "choice", "instructions": "Is this chart's trend up, down or flat?",
                "criteria": {"up": "It rises.", "down": "It falls.", "flat": "It stays level."}}}),
    ("screenshot", screenshot(), "A user attached this screenshot to a ticket: <image>",
     {"error_dialog": {"type": "noul",
                       "instructions": "Does this screenshot show an error dialog?"}}),
]:
    out, ms = post({"model": "jev-latest", "state": state, "questions": questions,
                    "images": [to_data_url(image)]})
    print(f"{name:<11} {json.dumps(out['answers'])}   {ms:.0f} ms, "
          f"{out['usage']['input_tokens']} input tokens")
