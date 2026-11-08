# pixelrun: Shellback

```
python apps/pixelrun/play.py watch --seed 3
```

A hermit crab carries a letter along the sea floor. Urchins sit on the sand, so the crab jumps.
Jellyfish drift at shell height, so it ducks. A decision model plays from screenshots alone.

The model gets one PNG and a `choice` question: jump, duck or stay. It answers, the game takes
that answer, and the world moves on. There is no physics planner, no labelled state and no
assisted override. Its top option is the move.

The PNG is the left half of the frame, enlarged twice, with a fixed yellow dotted box over the
crab and the 30 pixels ahead of it. The question asks what is inside the box. The box never moves
and shows no game state, but its width was set from the game's jump physics, so it carries one
piece of knowledge: where to react. With the plain full frame the model timed jumps badly and
scored about the same as random (69.2 over seeds 0 to 19).

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

Seeds 20 to 39, the v7.0-VL 3B (bf16, PyTorch on one GPU). The view was chosen on seeds 0 to 19,
so these seeds were not used to choose it:

| player | mean score | sd | min | max | runs finished (750) |
|---|---|---|---|---|---|
| model | 384.8 | 230.5 | 77 | 750 | 3 of 20 |
| model, 4-bit weights of the 1.9 GB MLX build | 330.5 | 243.0 | 74 | 750 | 3 of 20 |
| random | 67.5 | 17.6 | 42 | 103 | 0 |
| always stay | 61.1 | 11.8 | 42 | 78 | 0 |

The 4-bit row runs that build's weights in PyTorch on the same GPU; the gap to bf16 is well
inside the spread between seeds. Median answer time 31 ms on that GPU, inside the 133 ms a live
game would allow. On an M2 Max the MLX build takes about 0.9 s a move, so there the game is only
played lockstep, and one wrong move ends a run: its scores on single seeds differ from the GPU's
(seed 20: 242 on the Mac, 520 on the GPU) though it picks the same move on all 413 labelled
frames we checked.

A scripted player that reads the game's internal state finishes the 40 seeds tried, so the game
can be won. That player is only a check on the game, and is not a score to compare with.

## Not tested

`watch` opens a tkinter window and has never been run: the machines used so far have no tkinter or
display. It has not been tried on a Mac.

Tests: `pytest tests/test_pixelrun.py` (a fake server, no model).
