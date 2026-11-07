# pixelrun: Shellback

```
python apps/pixelrun/play.py watch --seed 3
```

A hermit crab carries a letter along the sea floor. Urchins sit on the sand, so the crab jumps.
Jellyfish drift at shell height, so it ducks. A decision model plays from screenshots alone.

**Status: the model has not played yet.** Only the baselines below have been run.

The model gets one PNG of the frame and a `choice` question: jump, duck or stay. It answers, the
game takes that answer, and the world moves on. There is no physics planner, no labelled state and
no assisted override. Its top option is the move. That is the point of the game.

The probabilities on screen are drawn on a separate copy for you. The model never sees them.

## Run it

Needs Pillow, and a server for the model player. No display is needed except for `watch`.

```bash
python apps/pixelrun/play.py score --player model --seeds 20 --out runs/model
python apps/pixelrun/play.py score --player random --seeds 20
python apps/pixelrun/play.py score --player stay --seeds 20
python apps/pixelrun/play.py gif runs/model/seed-3.jsonl run.gif
```

`PIXELRUN_URL` or `--url` sets the server, default `http://127.0.0.1:8000`.

## How it is scored

The score is decisions survived, up to 750. The player acts once every 4 ticks. Runs are lockstep:
the game waits for each answer, so latency cannot change a score. At the game's 30 ticks a second,
playing live would need an answer within 133 ms. `score` reports measured latency beside that
figure.

Baselines over seeds 0 to 19:

| player | mean score | sd | min | max |
|---|---|---|---|---|
| always stay | 60.9 | 10.3 | 44 | 78 |
| random | 67.2 | 15.9 | 46 | 111 |

A scripted player that reads the game's internal state finishes the 40 seeds tried, so the game
can be won. That player is only a check on the game, and is not a score to compare with.

## Not tested

`watch` opens a tkinter window and has never been run: the machines used so far have no tkinter or
display. It has not been tried on a Mac.

Tests: `pytest tests/test_pixelrun.py` (a fake server, no model).
