# Training, evaluation and export

The code that produced the released models. Paths are relative to the repository
root. To *use* a checkpoint — demo, HTTP API, or loading it in Python — see the
[root README](../README.md); per-version records are in
[`../versions/`](../versions/), and the serving layer has its own guide in
[`../serve/README.md`](../serve/README.md).

Every module here opens by declaring itself **EDITABLE** or **PROTECTED**. That is
the guard rail the autonomous loop runs inside, and it is worth understanding
before reading the code: an experiment may rewrite the editable modules and may
not touch the protected ones, so a change to the model, the data or the recipe can
never quietly become a change to how it is scored. An experiment that needs to
move a protected file is a change to the task, and goes through a human.

| | path | what it decides |
|---|---|---|
| **EDITABLE** | `arch.py` | the model axis: how state and options are encoded and read out |
| **EDITABLE** | `vision.py` | images in the state (v4.0 on): the base model's own vision tower, `<image>` markers, M-RoPE positions |
| **EDITABLE** | `data.py` | the data axis: what gets collected, generated or deleted |
| **EDITABLE** | `train.py` | the training axis: objective, optimiser, schedule, calibration |
| **PROTECTED** | `contract.py` | what a typed decision *is* — one `Case`, one prediction |
| **PROTECTED** | `evaluate.py`, `metrics.py`, `targets.py` | what is scored, how, and on which splits |
| **PROTECTED** | `encode.py` | position and padding bookkeeping (the *layout* is an `arch.py` choice) |
| **mixed** | `fit.py` | the loop: bookkeeping protected, configuration editable |
| **EDITABLE** | `rl2.py` | continued training with a reward instead of a label (v3.0's listwise reranking stage: Plackett-Luce rankings scored by NDCG@k, leave-one-out baseline, KL penalty to the parent). `fit.py` hands over to it when `FitConfig.rl2` is set |

Entry points are in `scripts/`, and each one says what it is on its first line
(`head -1 scripts/*.py`). The three that matter here:

| path | what it does |
|---|---|
| `run_arm_lib.py` | one experiment end to end. `release_train.py` and the internal search loop both call it, so a release is the same code path as an experiment |
| `release_train.py` | train one release checkpoint |
| `load_release.py` | load one; `--verify` re-scores it against the run that produced it |
| `calibrate.py` | fits the confidence head (v2.0 on) on withheld data, and loads it back |

Two corpus builders ship: `build_synth_corpus.py` builds v1.0's corpus, and
`build_corpus.py` builds a corpus from public classification sources that are
deliberately **not** evaluation targets.

**v2.0's and v2.1's corpus takes four more builders**, all of them here:

| script | what it produces |
|---|---|
| `build_specialist_corpus.py` | synth + the typed-decisions train split, oversampled x3 |
| `build_specialist_replay_corpus.py` | adds the general-knowledge replay at a given share of questions |
| `build_suite_train_corpus.py` | the suite benchmarks' train splits, converted and decontaminated against their test splits, capped per source |
| `corpus_gate.py` | refuses a corpus whose label distribution has drifted from `corpus_reference_stats.json` |
| `build_ds2_addons.py` | v3.0's data-scale-2 additions (emotion, SST-5, ANLI, tasksource and Open-Jev train items) on top of a base corpus |
| `build_hippo_rr.py` | v3.0's reranking corpus: MS MARCO / TopiOCQA / FiQA train queries with BM25 hard negatives, in the hippo request shape |
| `build_rl2_sources.py` | the RL stage's source files from that corpus (`rl_hrr.jsonl`: one query group per case) |
| `fit_release_calibration.py` | fits a release's confidence head on the held-out tenth of its parent's corpus |

Each takes every root as an argument. None has a default path, and
`tests/test_corpus_builders.py` fails if one appears.

**Two inputs are still not published, and they are data rather than code.** The replay
pool's `dc_*` files come from an external task registry that a collection tool read once,
and `--suite-dir` is the frozen suite definitions with the decontaminated test splits. The
`kev` and `nimble` stages also shell out to those benchmarks' own repositories, which a
reproducer needs anyway. So the *assembly* of v2.0's corpus is fully reproducible from
here and two of its *inputs* are not yet downloadable; publishing them is a data-release
decision, not a boundary.

## Environment

**To run or serve the models**, `requirements.txt` is enough and deliberately
loose — none of that path needs the pinned stack below, and no inference code
imports flash-linear-attention.

**To reproduce the published numbers**, use `requirements-repro.txt`. Numbers
are only comparable within one kernel stack, and every record carries a
`linear_attn_kernel` stamp saying which produced it. v1.0, v2.0 and v2.1 share one stack:

- Python 3.11, **torch 2.7.1+cu128, Triton 3.3.1, flash-linear-attention 0.5.2**
- transformers 5.17.0, datasets 4.1.1, safetensors

Do not move to Triton 3.4-3.7.0 on Hopper GPUs: flash-linear-attention's gated
delta-rule backward gives wrong gradients there (fla #640), and fla refuses to
run.

## Reproduce a release

The expected numbers are in that release's record under
[`../versions/`](../versions/). One H100 80 GB per run; about 13 minutes of
training at 2B for v1.0 and about 50 for v2.0, which takes 5,774 steps.

**v2.0 and v2.1** additionally fit a calibration head after training, on data
withheld from it (`rsijev/calibrate.py`). `run_arm_lib` splats `fit_extra` into the
`FitConfig`, so the spec carries the whole recipe difference:

```json
{"steps": 5774, "sources": "synth,td_train,mc_replay,st_*",
 "fit_extra": {"cal_method": "oof_head_scorefloor",
               "lower_layers_n": 8, "lower_layers_lr_scale": 0.1}}
```

`cal_method` `none` reproduces the model as trained, uncalibrated; both released
checkpoints use `oof_head_scorefloor`. The fitted values are saved beside the weights
as `calibration.safetensors`, and `load_release.py` applies them when it finds them.
Section 2.1 of [`../versions/v2.0.md`](../versions/v2.0.md) says what the head is and
what it costs.

`lower_layers_n` / `lower_layers_lr_scale` are **v2.1's** one optimiser change: the
tower's lowest *n* decoder layers train in their own parameter group at a fraction of
`lr_base`. v1.0 and v2.0 leave them at the defaults (`0` / `1.0`), which is one group
for the whole tower. Section 2 of [`../versions/v2.1.md`](../versions/v2.1.md) says why
it matters more than any of the data changes tried against the same problem.

### v1.0

```bash
python scripts/build_synth_corpus.py --out corpus/        # teacher soft gold; zero typed-decisions overlap
cat > spec.json <<'J'
{"readout": "option_xattn", "objective": "soft_ce", "steps": 1500, "residual": false,
 "option_order": "shuffled", "sources": "synth", "freeze_base": false, "readout_layer": -1,
 "lr_head": 0.0001, "eval_option_orders": ["canonical", "reversed"]}
J
python scripts/release_train.py --model Qwen/Qwen3.5-2B-Base --seed 17 --spec spec.json \
    --save-dir ckpt/ --out records/ --name v1.0_2b_s17 --corpus corpus/
python scripts/load_release.py --ckpt ckpt/ --verify --record records/v1.0_2b_s17.items.jsonl
```

## Things this code checks, because each one fired once

- **The option scorer runs in fp32 outside autocast.** Under bf16 autocast,
  `nn.Linear` and `MultiheadAttention` compute in bf16 whatever their weight
  dtype. That was the instability behind the 2B attempt before v1.0.
- **A non-finite loss stops training** at the step it happens.
- **A tower-training run checks** that the tower actually moved and that the
  shared resident model did not.
- **RL objectives pass the R0 diagnostic** before training. Their noise and
  log-density are restricted to valid options, because masked options carry
  -inf and produced NaN.

## Tests

```bash
pytest tests/ -m "not slow"                  # wire contract + precision guards, no GPU, <1s
CUDA_VISIBLE_DEVICES="" pytest tests/test_serve_parity.py -q   # real weights on CPU
python scripts/padding_invariance.py --help  # forward-only diagnostic; needs a model
```

`test_serve_parity.py` is the guard that keeps the serving forward pass
bit-identical to the evaluated one; it is marked `slow` because it loads real
weights.

**If you change something, tell us what happened** — this code is the input to an
autonomous research loop, and a well-measured negative is worth more than a small
positive. [Open an experiment report](https://github.com/Shanghua-Gao/RSI-Jev/issues/new?template=experiment.yml);
what the loop then does with it is in [`../CONTRIBUTING.md`](../CONTRIBUTING.md).
