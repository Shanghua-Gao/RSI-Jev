# The demo gallery, re-runnable

The gallery in the v4.0-VL card (section 4.3) is 143 pictures with 227 questions: road
scenes, "is it really there?", counting, charts, screens and factory parts. Every
question has a reference answer, and every miss is kept. This folder has all of it,
images included, and a runner that scores it through the public API.

```bash
pip install "rsi-jev[vision]"
python examples/gallery/run.py                                  # in-process Decider, v4.0-vl-2b from the Hub
python examples/gallery/run.py --compare --out answers.json     # also diff against the release run
```

Or against a server:

```bash
rsi-jev serve v4.0-vl-2b --port 8000
python examples/gallery/run.py --server http://127.0.0.1:8000
```

The runner sends one request per question, batch 1, as the published run did. It prints
every answer and then the score per category.

## What to expect

From the release run on an NVIDIA GB10 (bf16 tower, calibration on, one process, warm):

| category | right | what it asks |
|---|---|---|
| road | 99 / 124 | what a car should do about an object in a given place, and what the object is |
| hallucination | 26 / 26 | is an object really in the photo |
| clevr | 37 / 44 | CLEVR scenes and our own shape scenes: counting, colours, positions |
| charts | 13 / 15 | bar, line, pie and scatter charts, a diagram, tic-tac-toe, receipts against a policy |
| ui | 8 / 12 | checkout forms, alert dialogs, before/after screenshots, a destructive dialog |
| visa | 4 / 6 | VisA factory parts: good or defective |
| **all** | **187 / 227 (82%)** | median 62 ms per call |

`expected.json` has the release run's top answer and its probability for every question.
On other hardware a few close calls can flip; `--compare` lists them. Latency depends on
the GPU.

## Files

| file | what it is |
|---|---|
| `questions.json` | each item: `images` (relative paths), `state`, `questions` (each a wire-format question `spec` with its reference `ref`), source and licence |
| `run.py` | the runner: `Decider` in-process, or `--server` for `rsi-jev serve` |
| `expected.json` | the release run's answers |
| `make_images.py` | draws `images/generated/`; `--check` confirms it reproduces the shipped PNGs |
| `CREDITS.md` | per-file credit and licence for every image |

A question's `spec` is exactly what goes under `questions` in a `POST /v1/systemone` body,
so any item can be sent by hand:

```python
import json
from rsijev import Decider
item = json.load(open("examples/gallery/questions.json"))["items"][0]
q = item["questions"][0]
d = Decider("v4.0-vl-2b")
print(d.decide(item["state"], {q["key"]: q["spec"]},
               images=["examples/gallery/" + p for p in item["images"]]))
```

## Where the images come from

All 89 distinct images ship in `images/`, so no item is left out and the totals above are the
full published ones. 30 are ours (CC0). 15 are Wikimedia Commons photos (public domain, CC0, CC BY
2.0, and six under CC BY-SA, which come with a share-alike condition). 38 are CLEVR scenes
and 6 are VisA photos, both CC BY 4.0. None of the gallery uses POPE/COCO photos:
its "is it there?" questions are asked about the licensed photos instead.
Per-file credits are in [CREDITS.md](CREDITS.md). These images are not under the repository's MIT licence.

The road questions, their option texts and the acceptable actions per object and place
come from the game in [reinhard-z/vision-jev](https://github.com/reinhard-z/vision-jev) (MIT).
