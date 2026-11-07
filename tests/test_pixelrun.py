"""Shellback with no model: the game, the baselines, the recording, a fake server."""
from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "apps" / "pixelrun"
sys.path.insert(0, str(APP))
import game          # noqa: E402
import play          # noqa: E402
from PIL import Image  # noqa: E402


def perfect(g):
    """Reads game state, which no player gets: only here to prove every seed is winnable."""
    near = [o for o in g.obstacles if o.right > game.CRAB_X]
    if not near:
        return "stay"
    o = min(near, key=lambda o: o.x)
    d = o.x - (game.CRAB_X + game.CRAB_W)
    if o.kind == "urchin":
        return "jump" if d < 40 else "stay"
    return "duck" if d < 60 else "stay"


def test_same_seed_same_game():
    a, b = game.Shellback(5), game.Shellback(5)
    for action in ["stay", "jump", "stay", "duck"] * 30:
        a.step(action)
        b.step(action)
    assert a.render().tobytes() == b.render().tobytes() and a.score == b.score


def test_seeds_differ():
    assert game.Shellback(1).render().tobytes() != game.Shellback(2).render().tobytes()


@pytest.mark.parametrize("seed", range(10))
def test_every_seed_can_be_finished(seed):
    g = game.Shellback(seed)
    while not g.over:
        g.step(perfect(g))
    assert g.alive and g.tick >= game.MAX_TICKS and g.cleared > 10


def test_the_game_waits_for_the_player():
    g = game.Shellback(0)
    before = g.render().tobytes()
    assert g.render().tobytes() == before and g.tick == 0


def test_urchins_kill_a_crab_that_stays_and_jellies_one_that_jumps():
    for kind, action in (("urchin", "stay"), ("urchin", "duck"), ("jelly", "stay")):
        g = game.Shellback(0)
        g.next_spawn = 10 ** 9
        g.obstacles = [game.Obstacle(kind, game.CRAB_X + game.CRAB_W + 60.0)]
        while not g.over:
            g.step(action)
        assert not g.alive and g.tick < 100, (kind, action)


def test_unknown_action_is_refused():
    with pytest.raises(ValueError):
        game.Shellback(0).step("fly")


def test_the_players_frame_has_no_overlay():
    """What the model gets is game.render() and nothing else."""
    g = game.Shellback(3)
    seen = []

    class Spy:
        def act(self, frame):
            seen.append(frame.tobytes())
            return "stay", {"probs": {"stay": 1.0}}

    play.run_one(3, Spy())
    assert seen[0] == g.render().tobytes()
    hud = play.overlay(g.render(), 3, 0, "stay", {"probs": {"stay": 0.5}})
    assert hud.tobytes() != seen[0]


class FakeModel:
    """Always answers `stay`; records the bodies."""

    def __init__(self):
        fake = self
        self.bodies = []

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                fake.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                raw = json.dumps({"answers": {"move": {
                    "choice": "stay", "probabilities": {"jump": 0.1, "duck": 0.1, "stay": 0.8},
                    "confidence": 0.7}}, "usage": {"depth": [20]}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.http.server_port}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()


def test_model_player_sends_one_image_and_a_choice(tmp_path):
    fake = FakeModel()
    assert play.main(["score", "--player", "model", "--seeds", "2", "--url", fake.url,
                      "--out", str(tmp_path)]) == 0
    fake.http.shutdown()
    body = fake.bodies[0]
    assert set(body) == {"model", "state", "images", "questions"}
    assert len(body["images"]) == 1 and body["state"].count("<image>") == 1
    assert body["questions"]["move"]["type"] == "choice"
    assert set(body["questions"]["move"]["criteria"]) == set(game.ACTIONS)
    png = base64.b64decode(body["images"][0].split(",", 1)[1])
    assert Image.open(io.BytesIO(png)).size == (game.W, game.H)
    # a model that always stays scores what the always-stay player scores
    stay = play.run_one(0, play.Stay())["score"]
    assert json.loads((tmp_path / "summary.json").read_text())["scores"][0] == stay


def test_gif_replays_the_recorded_run(tmp_path):
    fake = FakeModel()
    play.main(["score", "--seeds", "1", "--url", fake.url, "--out", str(tmp_path)])
    fake.http.shutdown()
    gif = tmp_path / "run.gif"
    assert play.main(["gif", str(tmp_path / "seed-0.jsonl"), str(gif)]) == 0
    im = Image.open(gif)
    assert im.is_animated and im.size == (game.W, game.H)


def test_baselines_run_from_the_command_line():
    p = subprocess.run([sys.executable, str(APP / "play.py"), "score", "--player", "random", "--seeds", "3"],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0
    assert json.loads(p.stdout.strip().splitlines()[-1])["seeds"] == 3


def test_a_dead_server_is_an_error_not_a_score():
    with pytest.raises(OSError):
        play.run_one(0, play.Model("http://127.0.0.1:1"))
