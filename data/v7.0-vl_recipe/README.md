# v7.0-VL recipe

v7.0-VL 4B is one fine-tune of `Qwen/Qwen3.5-4B` (the Instruct model, not `-Base`), 36,149 steps,
one epoch, seed 17, on 4x H200 with DDP. No average and no second stage: the checkpoint the fit
writes is the one the release packages. The method:

- option echo: the option blocks are repeated once and the readout reads the second copy;
- prefix-bidirectional attention: on full-attention layers the prompt and options attend both ways;
- exits: main exit at layer 32, auxiliary exits at 12, 16 and 20 (detached from the tower, total weight 0.9);
- retention KL (0.5, top-64, last 1,024 tokens, 4 rows per step) on long boxed CoT traces that the
  base model wrote for the run's own training questions;
- confidence-rank loss (coefficient 1.0) and a per-row option-permutation KL (2.0) on text rows.

The code that produced the checkpoint is vendored in `train/v7.0/` (see below). The training rows
are not published; the specs keep every setting and the source names, with directories replaced
by placeholders.

| step | spec | what | compute |
|---|---|---|---|
| 0 | `0-trace-sample.json` | 398,101 text choice questions drawn from the training split (seed 0) | CPU |
| 1 | `1-traces.json` | greedy CoT traces from `Qwen/Qwen3.5-4B`, raw prompt, 1,536 new tokens max | 1x H100 (resumable) |
| 2 | `2-retention-pool.json` | the 24,712 longest clean boxed traces; the fit keeps the first 21,087 | CPU |
| 3 | `3-train.json` | the 4-GPU DDP fit | 4x H200, 147,845 s (41.1 h) train time |

## Code (`train/v7.0/`)

| path | role |
|---|---|
| `rsijev/` | model, encoder, fit, eval (the training code dir's package; unused `tree.py`, `calibrate.py`, `adaptive.py` left out) |
| `scripts/run_arm_isolated.py` | runner: integrity and test-overlap checks, base load, then `run_arm_lib.run_arm` |
| `scripts/run_arm_lib.py` | control eval, `rsijev.fit.fit`, post-fit eval, checkpoint save |
| `scripts/worker.py` | not run; kept because the runner checks it is byte-identical to its own tree |
| `ddp_arm.py` | torchrun entry: pins each rank, starts NCCL, sets `fit_extra.ddp`, hands off to the runner |
| `smokefix/sitecustomize.py` | was on `PYTHONPATH`: makes `metrics.decision_score` NaN when a choice base rate is 1 (only small smoke evals hit it) |
| `retention/` | `mk_tvsample_v2i.py`, `trace_gen.py` (imports `nv7_cot.py`), `tvrep.py`, `mk_pool_v2i.py` |
| `SERVE_ENV` | the two input-format switches the checkpoint was trained with |
| `SOURCES.sha256` | each file's sha256 as it ran and as vendored here |

The scrub changed comments, docstrings and path defaults only (`SOURCES.sha256` lists which).
The training code dir as a whole hashed to tree `ece3a3580abe` (sha256 over the sorted
`sha256sum` lines of every file, including the files left out here).

## Environment

Python 3.11, torch 2.7.1+cu128, transformers 5.17.0, flash-linear-attention / fla-core 0.5.2,
safetensors 0.8.0, tokenizers 0.23.2, datasets 4.1.1, numpy 2.4.6, huggingface_hub 1.32.0,
pillow 12.3.0, torchvision 0.22.1+cu128; NCCL 2.27.7 preloaded (`LD_PRELOAD=.../libnccl.so.2`).
Trace generation: vLLM 0.30.0 in its own environment (torch 2.13.0). `trace_gen.py` reads the
model from the local HF cache (`HF_HOME`, one snapshot), so download `Qwen/Qwen3.5-4B` first.

Environment flags of the fit:

```bash
export RSIJEV_OPTION_POOL_OWN_TOKENS=1   # the option readout pools only the option's own tokens
# RSIJEV_OPTION_ECHO and RSIJEV_PREFIX_BIDIR were unset in the job; both default to 1 in this code,
# so the fit ran with option echo and prefix-bidirectional attention on. Serving sets both to 1 (SERVE_ENV).
export PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
```

## Commands

`D` is one data directory. Fill the placeholders in `3-train.json` first (`<corpus_dir>`,
`<save_dir>`, `<step_ckpt_dir>`, `<train_drop_file>`, `<retention_pool>`, `<vision_root>`): step 2
reads the corpus and the drop list through it. The drop list holds the (case_id, question key)
pairs left out of training (945,864: rows over 29,581 tokens, a per-root image-case cap, intake
drops); the K1 list is a method screen's held-back pairs (none of them were sampled).

```bash
T=train/v7.0
# 0. question sample (text rows of TRAIN cases only)
python $T/retention/mk_tvsample_v2i.py D/sample_spec.json D/corpus D/train_drop.json D/k1_drop.json 398101 D/tvsample

# 1. traces: 2,000-item smoke first, stop if fewer than half end in \boxed{}; then the full run (resumable)
python $T/retention/trace_gen.py --name inst4b --base Qwen/Qwen3.5-4B --tag smoke --inputs D/tvsample --out D/tvgen --limit 2000
python $T/retention/tvrep.py D/tvgen/inst4b.smoke.traces.jsonl - D/tvgen/report_smoke.json   # exit 3 = stop
python $T/retention/trace_gen.py --name inst4b --base Qwen/Qwen3.5-4B --tag full --inputs D/tvsample --out D/tvgen
python $T/retention/tvrep.py D/tvgen/inst4b.full.traces.jsonl - D/tvgen/report.json

# 2. pool: 24,712 longest clean boxed traces (fit_extra.retention_text_file)
python3 -I $T/retention/mk_pool_v2i.py D/tvsample D/tvgen/inst4b.full.traces.jsonl data/v7.0-vl_recipe/3-train.json \
    D/k1_drop.json 24712 D/pool/rt_pool_all.jsonl D/pool/rt_pool_long.jsonl

# 3. fit, 4 GPUs
export PYTHONPATH=$PWD/$T/smokefix RSIJEV_OPTION_POOL_OWN_TOKENS=1
export RSIJEV_EVAL_SUITE=D/eval_suite_v3 RSIJEV_EVAL_FINAL=D/eval_final_v3 RSIJEV_KG=D/eval_kg_v1_1 RSIJEV_CONTROL_CACHE=D/cache
A="--runner $T/scripts/run_arm_isolated.py --code-dir $T --spec data/v7.0-vl_recipe/3-train.json \
   --name v7.0-vl-4b --out D/runs --model Qwen/Qwen3.5-4B --corpus D/corpus"
python $T/ddp_arm.py --check --world 4 $A
torchrun --standalone --nproc_per_node 4 $T/ddp_arm.py $A
```

Step 3 batches 64 rows per step, 16 per rank. On steps whose longest row is over 3,518 tokens
each rank runs its 16 rows one at a time (`grad_accum` 64); shorter steps run them in one pass
(`grad_accum_short` 4). Step checkpoints every 9,037 steps allow
a resume. The fit writes `tower.safetensors`, `scorer.safetensors`, `aux_scorers.safetensors` and
`meta.json` (with `spec.option_pool_own_tokens = true`) to `save_dir`.
Final step loss 0.8826.

Before it loads the model, the runner refuses a corpus that shares a state with the
typed-decisions test split, the eval suite, the held-out final set or a kg_v1.1 question, and
then scores a zero-shot control. It therefore needs those eval sets at the `RSIJEV_*` paths
above. The post-fit eval does not change the checkpoint; in the release run it was stopped
after the checkpoint was saved.
