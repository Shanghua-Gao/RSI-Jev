# Breakout from pixels

The v4.0-VL card (section 4.3) says the model plays the Breakout demo of
[hr98w/jev-visual](https://github.com/hr98w/jev-visual) (MIT) from screenshots: 10 of 10 on
the game's lane check, and 7 of 24 bricks cleared in a live game. This folder reruns both.

The game is theirs and runs unmodified. Each decision, their page sends a PNG of the board
and one fixed question ("Which numbered column contains the white ball?", five lanes) to a
same-origin `/v1/judge` and moves the paddle to the chosen lane. `adapter.py` answers
`/v1/judge` by forwarding each call to `rsi-jev serve` as a `POST /v1/systemone`. The model
gets only the screenshot, their instructions and their option texts: no coordinates and no
game state.

## Run it

```bash
# 1. the game, at the commit used for the published numbers (we do not vendor it)
git clone https://github.com/hr98w/jev-visual
git -C jev-visual checkout 4382bba455647400951429134ceb012ca155e3fe

# 2. the model
pip install "rsi-jev[vision]"
rsi-jev serve v4.0-vl-2b --port 8000

# 3. the adapter, in another terminal: serves their pages and /v1/judge on :8788
python examples/breakout/adapter.py --game jev-visual --upstream http://127.0.0.1:8000 --port 8788
# play it yourself at http://127.0.0.1:8788/demo/breakout/  ("Start model")

# 4. the two checks, headless (Node 18+)
cd examples/breakout && npm install && npx playwright install chromium && cd -
DEMO_URL=http://127.0.0.1:8788 OUT_DIR=breakout-out node examples/breakout/diagnose.mjs   # lane check
DEMO_URL=http://127.0.0.1:8788 OUT_DIR=breakout-out node examples/breakout/record.mjs     # one live game
python examples/breakout/finish.py breakout-out [--video]                                 # summary.json (+ mp4/gif)
```

## What to expect

| check | published (NVIDIA GB10, bf16, calibration on) |
|---|---|
| lane check: ten fixed scenes drawn by the game's renderer | **10 / 10** |
| live game, Easy speed (110 px/s), unmodified page | **7 of 24 bricks** cleared, 3 lives left, stopped by the game's 200-decision limit (not a win) |

For reference, jev-visual's README reports 8/10 on the lane check for its own local model,
and one Easy run of 9 bricks with 2 lives left, paused at 80 decisions.

The lane check is fixed input, so it should repeat. The live game should not be expected to
repeat brick for brick: the ball keeps moving while a decision is in flight, so the
outcome depends on latency (round trip about 80 ms on the GB10) and on browser timing.

## Files

| file | what it does |
|---|---|
| `adapter.py` | `/v1/judge` to `/v1/systemone`, and their `demo/` pages served on one origin |
| `diagnose.mjs` | the ten lane-check scenes (the same as jev-visual's own `demo/breakout/diagnose.mjs`) |
| `record.mjs` | one live game, recorded; stops at win, loss, 200 decisions or `MAX_S` seconds |
| `finish.py` | `summary.json` from both, and optionally the mp4 and GIF |
| `package.json` | Playwright, for the two Node scripts |

Credit: Breakout game from hr98w/jev-visual (MIT, © 2026 Jev Visual contributors). A
recording of their page carries that credit.
