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
| **EDITABLE** | `data.py` | the data axis: what gets collected, generated or deleted |
| **EDITABLE** | `train.py` | the training axis: objective, optimiser, schedule, calibration |
| **PROTECTED** | `contract.py` | what a typed decision *is* — one `Case`, one prediction |
| **PROTECTED** | `evaluate.py`, `metrics.py`, `targets.py` | what is scored, how, and on which splits |
| **PROTECTED** | `encode.py` | position and padding bookkeeping (the *layout* is an `arch.py` choice) |
| **mixed** | `fit.py` | the loop: bookkeeping protected, configuration editable |

Entry points are in `scripts/`, and each one says what it is on its first line
(`head -1 scripts/*.py`). The three that matter here:

| path | what it does |
|---|---|
| `run_arm_lib.py` | one experiment end to end. `release_train.py` and the internal search loop both call it, so a release is the same code path as an experiment |
| `release_train.py` | train one release checkpoint |
| `load_release.py` | load one; `--verify` re-scores it against the run that produced it |

Two corpus builders ship: `build_synth_corpus.py` is what v1.0 trained on,
`build_corpus.py` is a data-axis option that v1.0 did not use.

## Environment

**To run or serve the models**, `requirements.txt` is enough and deliberately
loose — none of that path needs the pinned stack below, and no inference code
imports flash-linear-attention.

**To reproduce the published numbers**, use `requirements-repro.txt`. Numbers
are only comparable within one kernel stack, and every record carries a
`linear_attn_kernel` stamp saying which produced it. The v1.0 stack:

- Python 3.11, **torch 2.7.1+cu128, Triton 3.3.1, flash-linear-attention 0.5.2**
- transformers 5.17.0, datasets 4.1.1, safetensors

Do not move to Triton 3.4-3.7.0 on Hopper GPUs: flash-linear-attention's gated
delta-rule backward gives wrong gradients there (fla #640), and fla refuses to
run.

## Reproduce v1.0

The expected numbers are in [`../versions/v1.0.md`](../versions/v1.0.md). One
H100 80 GB per run; about 13 minutes of training at 2B.

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
