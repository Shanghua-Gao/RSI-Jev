# Image benchmarks of the v4.0-VL card, re-runnable

One script per benchmark in section 4.3 of [`versions/v4.0-vl.md`](../../versions/v4.0-vl.md).
Each downloads its public data, asks every question through the public API, and prints
the card's table with the same prompts and the same statistics. Where another system's
per-question predictions are public, the comparison is paired on the same items.

```bash
pip install "rsi-jev[vision]" pandas pyarrow scipy        # pandas/pyarrow/scipy: the upstream loaders and stats
```

Every script takes `--model` (default `v4.0-vl-2b`, an in-process `Decider`; also a Hub
repo id or a checkpoint directory) or `--server http://host:port` for a running
`rsi-jev serve`. Both give the served probabilities: bf16 tower on CUDA, release
calibration on. Results are appended to a JSONL file as they arrive, so an interrupted
run resumes; `--limit N` makes a smoke run, and `--analyze-only` redoes the tables from
an existing file.

| benchmark | command | what it needs |
|---|---|---|
| MMStar | `python scripts/vision_benches/mmstar.py` | a GPU |
| BLINK (val) | `python scripts/vision_benches/blink.py` | a GPU |
| VisA | `python scripts/vision_benches/visa.py --visa-root DATA/visa --download` | a 1.9 GB download, a GPU |
| Laya Vision's validation sets | `laya.py prep`, `run`, `reference`, `compare` (below) | CPU for prep (many downloads), a GPU |
| the five held-out benchmarks | `python scripts/vision_benches/heldout.py --eval D/eval_vision_v1` | the set built first (below), a GPU |

## Expected numbers

These are the card's numbers. They were computed on an NVIDIA GB10 with the served
model path (bf16 tower, fp32 scorer, calibration on, batch 1). Another GPU or kernel
can flip a few close calls, so expect agreement to within a few tenths of a point, not
to the digit.

**MMStar** (1,498 items; 2 of the 1,500 are dropped by jev-omni-eval's loader because the
keyed answer is a padding "nan" option):

| system | accuracy % [95% CI] | ECE | v4.0-VL minus system [95% CI], McNemar p |
|---|---|---|---|
| v4.0-VL | 62.4 [60.0, 64.8] | 0.108 | |
| A Jev-Omni | 64.3 [62.0, 66.9] | 0.142 | −1.9 [−4.7, +0.7], p = 0.18 |
| B Gemma 4 12B | 63.8 [61.4, 66.2] | 0.279 | −1.3 [−4.1, +1.5], p = 0.38 |

v4.0-VL is wrong at p ≥ 0.99 once.

**BLINK val** (1,901 items, 14 tasks):

| system | accuracy % [95% CI] | ECE | v4.0-VL minus system [95% CI], McNemar p |
|---|---|---|---|
| v4.0-VL | 56.3 [54.2, 58.3] | 0.071 | |
| A Jev-Omni | 60.8 [58.8, 62.8] | 0.093 | −4.5 [−7.0, −1.9], p = 0.00074 |
| B Gemma 4 12B | 61.3 [59.3, 63.4] | 0.226 | −5.0 [−7.5, −2.6], p < 1e-4 |

v4.0-VL is never wrong at p ≥ 0.99. The scripts also print the task-macro accuracy,
temperature-scaled ECE for the other systems, and per-task accuracy with Holm-adjusted p.

**VisA** (2,162 test images, 12 categories, zero-shot, both option orders averaged):

| system | good parts seen | macro image AUROC (95% CI) | v4.0-VL minus system [95% CI] |
|---|---|---|---|
| v4.0-VL | 0 | 86.6 (85.1–88.1) | |
| A0 Jev-Omni | 0 | 81.1 (79.4–82.7) | +5.5 [+3.5, +7.5] |
| B0 Gemma 4 12B | 0 | 82.9 (81.3–84.5) | +3.7 [+1.8, +5.6] |
| PatchCore, 16 good parts, 256 px (published) | 16 | 85.7 (84.2–87.2) | not paired |

**Laya Vision's validation sets** (6,357 questions over 34 sets, against Laya Vision 201M):

| questions | n | v4.0-VL | Laya Vision | difference (95% CI) | ECE v4.0-VL / Laya |
|---|---|---|---|---|---|
| choice and yes/no, clean sets | 2,516 | 0.828 | 0.713 | +0.115 (+0.094, +0.136) | 0.043 / 0.064 |
| all types, all sets | 6,357 | 0.763 | 0.717 | +0.046 (+0.032, +0.060) | 0.044 / 0.053 |
| all types, clean sets | 3,716 | 0.683 | 0.664 | +0.019 (−0.001, +0.038) | 0.061 / 0.087 |

"Clean sets" drops the 14 sets whose source training split v4.0-VL's image data used
(`CLEAN_EXCLUDED` in `laya.py`: the V1 sources plus IconQA, Visual7W and OCR-VQA). It is a
set-level filter only.

**Held-out** (4,173 questions, equal weights):

| benchmark | n | top-1 | ECE | blank |
|---|---|---|---|---|
| MMBench (dev) | 992 | 0.845 | 0.040 | 0.290 |
| RealWorldQA | 591 | 0.707 | 0.085 | 0.371 |
| POPE | 853 | 0.912 | 0.052 | 0.502 |
| HallusionBench | 937 | 0.695 | 0.144 | 0.503 |
| InfographicVQA (val) | 800 | 0.858 | 0.026 | 0.624 |
| **mean** | 4,173 | **0.803** | **0.069** | 0.458 |

## Laya Vision, step by step

```bash
B=scripts/vision_benches/laya.py
python $B prep --data DATA/laya                     # rebuild the 34 sets and draw 200 items each
python $B run --data DATA/laya --out bench-results/laya_v4.jsonl
pip install -e ~/.cache/rsi-jev-benches/laya-vision  # Laya's own code, cloned by prep
python $B reference --data DATA/laya --out bench-results/laya_reference.jsonl
python $B compare --ours bench-results/laya_v4.jsonl --laya bench-results/laya_reference.jsonl --md laya.md
```

`prep` replays Laya's own data preparation and writes a `check.json` per set comparing
the rebuilt validation count with the count Laya's full evaluation recorded; all 34 match
at the pinned revisions. The cauldron and rubric sets are read at pinned revisions. The
others are read as Laya's prep reads them, at their current revision, so `check.json` is
the guard against a source that has since changed.

## The held-out set

The held-out set is built by the repository's own builders, not by a script here:

```bash
python scripts/build_eval_vision.py --build D/eval_build                 # candidates, pinned revisions
python scripts/build_vision_v1.py --root D/vision_v1                      # the training corpus to check against
python scripts/decontam_vision.py eval --build D/eval_build --train D/vision_v1 --out D/eval_vision_v1
python scripts/vision_benches/heldout.py --eval D/eval_vision_v1
```

Decontamination drops candidates whose image (sha256 or perceptual hash) or text matches
the vision_v1 training corpus; it needs that corpus built first. Without it,
`heldout.py --build D/eval_build` scores the candidates: a slightly larger set (MMBench
1,000 and POPE 900, against 992 and 853), so slightly different numbers. "Blank" asks the
same questions with every image replaced by a grey one of the same size.

## Where the data and the comparison predictions come from

Nothing below is shipped in this repository; each script fetches it.

| what | source | licence |
|---|---|---|
| MMStar | `Lin-Chen/MMStar` @ `bc98d668` (Hugging Face) | its dataset card's terms |
| BLINK | `BLINK-Benchmark/BLINK` @ `a3666eb2` (Hugging Face) | its dataset card's terms |
| VisA | `VisA_20220922.tar`, Amazon Science, sha256 checked | CC BY 4.0 |
| item builders, prompts and statistics for MMStar and BLINK; Jev-Omni, Gemma 4 12B and JevDigits per-question predictions | [CondadosAI/jev-omni-eval](https://github.com/CondadosAI/jev-omni-eval) @ `5a8b6d1` (`src/`, `output/results/`) | Apache-2.0 |
| VisA prompt, split, statistics; Jev-Omni and Gemma per-image predictions; PatchCore summary | [CondadosAI/jev-omni-inspection](https://github.com/CondadosAI/jev-omni-inspection) @ `25da8c7` (`src/`, `results/`) | Apache-2.0 |
| Laya's data preparation, loader and scorer | [r33drichards/laya-vision](https://github.com/r33drichards/laya-vision) @ `d4075b0` | Apache-2.0 |
| Laya Vision 201M weights (reference predictions are recomputed from them) | `thaitea/laya-vision-201m` @ `0b6228f7` (Hugging Face) | CC BY-NC-SA 4.0 |
| Laya's validation images and questions | the_cauldron @ `847a98a7`, A-OKVQA, ScienceQA, VQAv2, POPE, VizWiz, CIFAR-10H, FER+, KonIQ, EvalMuse, AVA, RichHF, CrisisMMD, VLFeedback | each source's own terms |
| held-out set | MMBench, RealWorldQA, POPE, HallusionBench, InfographicVQA at the revisions in `scripts/vision_common.py` | see `scripts/build_eval_vision.py`; several are unclear for redistribution |

The third-party repositories are cloned into `~/.cache/rsi-jev-benches/` (set
`RSIJEV_BENCH_CACHE` to move it) and checked out at the commits above. Their code is
imported, not copied, so the prompts and the statistics are theirs. Laya Vision publishes
no per-question predictions for these samples, so `laya.py reference` recomputes them
from its public weights with its own code.

## What these scripts cannot reproduce

The public API returns the served, calibrated probabilities only. The card's
pre-calibration figures (held-out image ECE 0.120 before the confidence head; the
"raw head" ECE of MMStar and BLINK) need the model internals and are not reproduced here.
