#!/usr/bin/env python3
"""Shellback runs: score a player over seeds, watch one, or write a GIF.

    play.py score --player model --seeds 20 --url http://127.0.0.1:8000 --out runs/model
    play.py score --player stay  --seeds 20
    play.py watch --url http://127.0.0.1:8000 --seed 3
    play.py gif runs/model/seed-3.jsonl run.gif
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import (ACTIONS, CRAB_W, CRAB_X, GROUND, MAX_TICKS, TICKS_PER_ACTION,  # noqa: E402
                  TICKS_PER_SECOND, H, Shellback, W)

from PIL import Image, ImageDraw                        # noqa: E402

REPLAY_FPS = TICKS_PER_SECOND / TICKS_PER_ACTION
BUDGET_MS = 1000 / REPLAY_FPS
STATE = "A side-view frame from a game. A hermit crab on the left walks to the right. <image>"
QUESTION = {
    "type": "choice",
    "instructions": "What should the crab do right now?",
    "criteria": {
        "jump": "Part of a spiky purple urchin is inside the yellow dotted box",
        "duck": "Part of a pink jellyfish is inside the yellow dotted box",
        "stay": "The yellow dotted box holds only the crab, sand and water",
    },
}
# The model's view: a fixed dotted box over the crab and the stretch just ahead of it (where an
# urchin means "jump now" and a jellyfish "duck now"), and the left half of the frame, enlarged
# twice. The box never moves and shows no game state; it is how the frame tells the player
# where to look. Without it, and at full size, the model times jumps badly: on labelled frames
# it picked a safe move on 45% of the urchin frames where a move could kill, 95% with this view.
ZONE = (CRAB_X, 60, CRAB_X + CRAB_W + 30, GROUND)
VIEW = (0, 40, 240, H)


def model_view(frame: Image.Image) -> Image.Image:
    im = frame.copy()
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = ZONE
    for x in range(x0, x1, 8):
        d.line((x, y0, min(x + 4, x1), y0), fill=(255, 230, 60), width=2)
        d.line((x, y1 - 1, min(x + 4, x1), y1 - 1), fill=(255, 230, 60), width=2)
    for y in range(y0, y1, 8):
        d.line((x0, y, x0, min(y + 4, y1)), fill=(255, 230, 60), width=2)
        d.line((x1, y, x1, min(y + 4, y1)), fill=(255, 230, 60), width=2)
    x0, y0, x1, y1 = VIEW
    return im.crop(VIEW).resize(((x1 - x0) * 2, (y1 - y0) * 2), Image.NEAREST)


def frame_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class Stay:
    def act(self, frame):
        return "stay", {}


class Random:
    def __init__(self, seed):
        self.rng = random.Random(seed + 7919)

    def act(self, frame):
        return self.rng.choice(ACTIONS), {}


class Model:
    """Pixels in (model_view of the frame), the model's top option out. Nothing else about the
    game reaches it, and nothing about the game changes its answer."""

    def __init__(self, url, timeout=60.0):
        import urllib.request
        self.request = urllib.request
        self.url, self.timeout = url.rstrip("/") + "/v1/systemone", timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def act(self, frame):
        body = {"model": "jev-latest", "state": STATE, "images": [frame_url(model_view(frame))],
                "questions": {"move": QUESTION}}
        req = self.request.Request(self.url, data=json.dumps(body).encode(), method="POST",
                                   headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with self.opener.open(req, timeout=self.timeout) as r:
            d = json.loads(r.read())
        ms = (time.perf_counter() - t0) * 1000
        a = d["answers"]["move"]
        return a["choice"], {"probs": a["probabilities"], "ms": ms, "depth": d["usage"].get("depth")}


def overlay(frame: Image.Image, seed: int, score: int, action: str, info: dict) -> Image.Image:
    """The viewer's copy, with the model's probabilities drawn on. The model never sees this."""
    im = frame.copy()
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 22), fill=(6, 30, 44))
    d.text((6, 6), f"seed {seed}  score {score}", fill=(235, 235, 220))
    probs = info.get("probs") or {}
    x = 170
    for name in ACTIONS:
        p = probs.get(name)
        if p is None:
            continue
        d.rectangle((x, 7, x + 50, 15), outline=(120, 150, 160))
        d.rectangle((x, 7, x + 50 * p, 15), fill=(255, 200, 80) if name == action else (120, 150, 160))
        d.text((x + 54, 6), f"{name} {p:.2f}", fill=(235, 235, 220))
        x += 104
    return im


def run_one(seed, player, record=None, on_frame=None):
    g = Shellback(seed)
    log = []
    while not g.over:
        frame = g.render()
        action, info = player.act(frame)
        if record is not None:
            record.write(json.dumps({"tick": g.tick, "action": action, **info}) + "\n")
        if on_frame:
            on_frame(overlay(frame, seed, g.score, action, info))
        if "ms" in info:
            log.append(info["ms"])
        g.step(action)
    return {"seed": seed, "score": g.score, "cleared": g.cleared, "alive": g.alive,
            "capped": g.tick >= MAX_TICKS, "ms": log}


def make_player(name, seed, url):
    if name == "stay":
        return Stay()
    if name == "random":
        return Random(seed)
    return Model(url)


def cmd_score(a):
    results, ms = [], []
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    for seed in range(a.first_seed, a.first_seed + a.seeds):
        record = open(os.path.join(a.out, f"seed-{seed}.jsonl"), "w") if a.out else None
        if record:
            record.write(json.dumps({"seed": seed}) + "\n")
        r = run_one(seed, make_player(a.player, seed, a.url), record)
        if record:
            record.close()
        ms += r.pop("ms")
        results.append(r)
        print(json.dumps(r), flush=True)
    scores = [r["score"] for r in results]
    summary = {"player": a.player, "seeds": len(scores), "mean": statistics.mean(scores),
               "sd": statistics.pstdev(scores), "min": min(scores), "max": max(scores),
               "cleared_mean": statistics.mean(r["cleared"] for r in results), "scores": scores}
    if ms:
        ms.sort()
        summary["latency_ms"] = {"n": len(ms), "median": ms[len(ms) // 2],
                                 "p95": ms[int(len(ms) * 0.95)], "max": ms[-1]}
        summary["realtime_budget_ms"] = BUDGET_MS
    print(json.dumps(summary))
    if a.out:
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(summary, f)
    return 0


def cmd_gif(a):
    """Replays a recorded run exactly: the game is deterministic and the record has every action."""
    with open(a.record) as f:
        head, *rows = [json.loads(line) for line in f]
    g, frames = Shellback(head["seed"]), []
    for r in rows:
        frames.append(overlay(g.render(), head["seed"], g.score, r["action"], r).convert("P", palette=Image.ADAPTIVE))
        g.step(r["action"])
    frames.append(overlay(g.render(), head["seed"], g.score, "", {}).convert("P", palette=Image.ADAPTIVE))
    start = max(0, len(frames) - a.last) if a.last else 0
    frames = frames[start:]
    frames[0].save(a.gif, save_all=True, append_images=frames[1:], loop=0,
                   duration=[math.ceil(1000 / REPLAY_FPS)] * (len(frames) - 1) + [1500])
    print(f"{a.gif}: {len(frames)} frames, final score {g.score}")
    return 0


def cmd_watch(a):
    """A live window. Needs tkinter; the game still waits for each answer."""
    import tkinter as tk
    from PIL import ImageTk
    root = tk.Tk()
    root.title("Shellback")
    label = tk.Label(root)
    label.pack()

    def show(im):
        photo = ImageTk.PhotoImage(im.resize((W * 2, H * 2), Image.NEAREST))
        label.configure(image=photo)
        label.image = photo
        root.update()

    r = run_one(a.seed, make_player(a.player, a.seed, a.url), on_frame=show)
    print(json.dumps({k: v for k, v in r.items() if k != "ms"}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pixelrun")
    sub = ap.add_subparsers(dest="command", required=True)
    url = os.environ.get("PIXELRUN_URL", "http://127.0.0.1:8000")

    p = sub.add_parser("score")
    p.add_argument("--player", choices=("model", "random", "stay"), default="model")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--url", default=url)
    p.add_argument("--out")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("watch")
    p.add_argument("--player", choices=("model", "random", "stay"), default="model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--url", default=url)
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("gif")
    p.add_argument("record")
    p.add_argument("gif")
    p.add_argument("--last", type=int, help="keep only the last N frames")
    p.set_defaults(func=cmd_gif)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
