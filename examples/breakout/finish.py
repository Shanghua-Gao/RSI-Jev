"""Summarise a Breakout check: OUT_DIR/diagnose.json + OUT_DIR/run.json -> OUT_DIR/summary.json.

    python examples/breakout/finish.py breakout-out [--video]

--video also cuts the recorded .webm into breakout.mp4 and a 2x-speed breakout.gif
(needs ffmpeg on PATH, or FFMPEG=/path/to/ffmpeg).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


def lane_check(rows: list[dict]) -> dict:
    out = [dict(id=r["id"], expected=r["expected"], choice=r["output"]["answers"]["action"]["choice"],
                probabilities=r["output"]["answers"]["action"]["probabilities"],
                model_ms=r["output"]["metrics"]["elapsed_ms"]) for r in rows]
    return dict(n=len(out), correct=sum(r["expected"] == r["choice"] for r in out), rows=out)


def game(run: dict) -> dict:
    s = run["summary"]
    ended = {"won": "won: all 24 bricks cleared", "lost": "lost all three balls"}.get(s["phase"])
    if not ended:
        ended = ("stopped by the game's own 200-decision limit (not a win)" if s["decisions"] >= 200
                 else f"stopped at the {run.get('max_s', 240)} s time limit")
    ms = [r["output"]["metrics"]["elapsed_ms"] for r in run["records"] if r.get("output")]
    wall = [r["wallMs"] for r in run["records"] if "wallMs" in r]
    lanes: dict[str, int] = {}
    for r in run["records"]:
        c = (r.get("answer") or {}).get("choice")
        if c:
            lanes[c] = lanes.get(c, 0) + 1
    return dict(summary=s, status=run["status"], ended=ended, errors=run.get("errors", []),
                executed=sum(1 for r in run["records"] if r.get("executed")),
                stale=sum(1 for r in run["records"] if r.get("executed") is False), lane_counts=lanes,
                median_model_ms=statistics.median(ms) if ms else None,
                median_round_trip_ms=statistics.median(wall) if wall else None)


def video(out: Path) -> dict:
    ff = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    vids = sorted(glob.glob(str(out / "video/*.webm")), key=os.path.getmtime)
    if not ff or not vids:
        return {"skipped": "no ffmpeg" if not ff else "no video"}
    v = vids[-1]
    crop = "crop=1240:780:20:180"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", v, "-vf", f"{crop},scale=1240:-2", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", str(out / "breakout.mp4")], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", v, "-vf",
                    f"{crop},setpts=0.5*PTS,fps=8,scale=760:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    str(out / "breakout.gif")], check=True)
    return {"mp4": "breakout.mp4", "gif": "breakout.gif (2x speed, 8 fps)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="breakout-out")
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    res = {}
    if (out / "diagnose.json").exists():
        res["lane_check"] = lane_check(json.loads((out / "diagnose.json").read_text()))
        print(f"lane check: {res['lane_check']['correct']}/{res['lane_check']['n']}")
    if (out / "run.json").exists():
        res["game"] = game(json.loads((out / "run.json").read_text()))
        s = res["game"]["summary"]
        print(f"game: cleared {s['cleared']} of 24 bricks, {s['lives']} lives left, {s['decisions']} decisions; "
              f"{res['game']['ended']}")
    if a.video:
        res["video"] = video(out)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
