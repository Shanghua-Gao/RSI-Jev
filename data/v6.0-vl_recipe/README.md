# v6.0-VL recipe

Four training stages from the base, then per-exit temperatures and the exit policy, fitted
on the result. Seed 17 for stages 1-3; the head stage ran seeds 0 and 1 and shipped seed 1.

| step | spec | from | what |
|---|---|---|---|
| 1 | `1-trunk-exits.json` | Qwen/Qwen3.5-4B-Base | text, 10,000 steps (16 x 8 accumulation); main head at layer 32, early-exit heads at 12, 16 and 20 reading a detached copy of their layer |
| 2 | `2-exit-head-refit.json` | stage 1 (`trunk/s17`) | the early-exit heads refit on the frozen trunk, 3,000 steps, distillation to the main exit at 0.5 |
| 3 | `3-images.json` | stage 2 (`trunk-refit/s17`) | images + text, 3,000 steps; exit heads carried at weight 0; layers 0-7 frozen |
| 4 | `4-heads.json` | stage 3 (`images/s17`) | all four heads retrained, tower frozen, 600 steps, seed 1 |
| 5 | `5-calibration.json` | stage 4 (`heads-s1`) | one temperature per exit; the package |
| 6 | `6-exit-policy.json` | stage 4 | the exit policy: the cascade 16 -> 20 -> 32 at 0.59, and `auto`'s thresholds 0.95 at 16, 0.59 at 20 |

Specs 1 and 3 are the `release_train` specs as run, with the checkpoint and corpus directories
renamed; 2, 4, 5 and 6 record the settings and results of their scripts. Paths in them are relative to one data directory, `D` below.
Stage 1's spec leaves out three keys of the run, `ckpt_dir`, `ckpt_keep` and `ckpt_minutes`:
crash-resume checkpoints, which do not change the trained weights and are not in this
repository's `fit.py`. The run's code set the encoder through its defaults (32,768 tokens,
middle truncation, own-token option pooling); the specs here set the same through
`max_length`, `truncate` and `option_pool_own_tokens`.

`lr_aux` 0 in stage 1 means the exit heads train at `lr_head` (1e-4). Stage 2's
`--distill -1` takes stage 1's `aux_distill`, 0.5.

```bash
# 1. trunk, main head and detached early exits; corpus D/text-corpus (148 sources; see "Not public")
python scripts/release_train.py --model Qwen/Qwen3.5-4B-Base --seed 17 --root D \
    --spec data/v6.0-vl_recipe/1-trunk-exits.json --corpus D/text-corpus \
    --save-dir D/trunk/s17 --out D/runs --name trunk

# 2. the early-exit heads refit on the frozen trunk; states shared with an evaluation set are left out
python scripts/refit_exit_heads.py --ckpt D/trunk/s17 --corpus D/text-corpus --out D/refit \
    --steps 3000 --lr 1e-4 --distill -1 --exclude-public --exclude D/eval_suite_v2 --exclude D/eval_final_v2 \
    --link-into D/trunk-refit/s17

# 3. images (roots vision_v1/v2/v3, vision_spatial, vision_joint, vision_recast, vision_sokoban under D)
python scripts/release_train.py --model Qwen/Qwen3.5-4B-Base --seed 17 --root D \
    --spec data/v6.0-vl_recipe/3-images.json --corpus D/image-corpus \
    --save-dir D/images/s17 --out D/runs --name images

# 4. all four heads under the own-token readout, tower frozen (seeds 0 and 1; 1 shipped)
for s in 0 1; do
  RSIJEV_OPTION_POOL_OWN_TOKENS=1 python scripts/fit_head.py --ckpt D/images/s17 --corpus D/image-corpus \
      --dev-corpus D/text-corpus --out D/heads-s$s --heads all --sdpa-math --max-length 4096 \
      --skip-prefix vis --steps 600 --bs 8 --lr 1e-5 --seed $s
done

# 5-6. per-exit dumps (one GPU each), then the CPU selection
for s in 0 1; do
  RSIJEV_OPTION_POOL_OWN_TOKENS=1 python scripts/dump_exits.py dev --ckpt D/heads-s$s --corpus D/image-corpus --out D/dumps/s$s
  RSIJEV_OPTION_POOL_OWN_TOKENS=1 python scripts/dump_exits.py eval --ckpt D/heads-s$s --out D/dumps/s$s --public \
      --set D/eval_suite_v2 --set final=D/eval_final_v2
done
python scripts/select_exit_policy.py seed --dump s0=D/dumps/s0 --dump s1=D/dumps/s1 --out D/policy/seed.json
C=D/heads-s1
RSIJEV_OPTION_POOL_OWN_TOKENS=1 python scripts/dump_exits.py cases --ckpt $C --pd D/policy-dev-v1 --out D/dumps/pdtext.pt
RSIJEV_OPTION_POOL_OWN_TOKENS=1 python scripts/dump_exits.py vision --ckpt $C --root D/policy-dev-v1/vision --out D/dumps/pdvis.pt
RSIJEV_OPTION_POOL_OWN_TOKENS=1 python scripts/dump_exits.py vision --ckpt $C --root D/eval_vision_v1 --out D/dumps/visexits.pt
V1="--dev-dump D/dumps/s1/dev_dump.pt --text-dump D/dumps/pdtext.pt --vis-dump D/dumps/pdvis.pt --vis-cases D/policy-dev-v1/vision"
python scripts/select_exit_policy.py policy $V1 --eval-dump D/dumps/s1/eval_dump.pt \
    --eval-vis-dump D/dumps/visexits.pt --out D/policy/cascade.json
for pd in policy-dev-v2 policy-dev-v2-fill policy-dev-sm; do
  RSIJEV_OPTION_POOL_OWN_TOKENS=1 python scripts/dump_exits.py cases --ckpt $C --pd D/$pd --out D/dumps/$pd.pt
done
TH="$V1 --policy D/policy/cascade.json --bench-weights D/di_diag.json --chance KIT/data/index-0.2.json \
    --heldout D/dumps/policy-dev-v2.pt D/policy-dev-v2 --heldout D/dumps/policy-dev-v2-fill.pt D/policy-dev-v2-fill \
    --shift-matched D/dumps/policy-dev-sm.pt D/policy-dev-sm --out D/policy/auto.json"
python scripts/select_exit_policy.py thresholds $TH
python scripts/select_exit_policy.py thresholds $TH --eval --eval-dump D/dumps/s1/eval_dump.pt \
    --eval-vis-dump D/dumps/visexits.pt

# 5. the package: per-exit temperatures, the policy, exit 12 dropped, auto's thresholds
python scripts/package_multiexit.py --ckpt $C --seed-metric D/policy/seed.json --seed s1 \
    --policy D/policy/cascade.json --drop-exit 12 --thresholds D/policy/auto.json --out D/package
```

The main exit's served temperature is the `cal_logT` buffer in `calibration.safetensors`
(0.5 in the shipped file, T = 1.65), fitted on the checkpoint's own DEV split by the `seed`
step. `calibration.json` also records the cascade selection's main-exit fit (`main_logT` 0.80, T = 2.23);
serving does not read it. The early exits are served at the cascade selection's temperatures (T = 1.88 at
16, 1.97 at 20).

## Not public

These inputs are not in this repository and will not be published:

- `text-corpus`, the 148-source corpus of stages 1, 2 and 4. It builds on earlier unpublished
  corpora and generated sources, and includes the Decision Index training pools; some of those
  are non-commercial (ANLI, the MS MARCO and Yelp parts of RAGTruth) or share-alike (HoVer,
  ForecastBench).
- `image-corpus`, stage 3's corpus: its 24,246 replayed text rows, its 2,000 new text rows, and
  four of its seven image roots (`vision_spatial`, `vision_joint`, `vision_recast`,
  `vision_sokoban`, generated by code). The other roots are built from public datasets as for
  v4.0-VL.
- The builders of both corpora.
- The policy development sets (`policy-dev-v1`, `policy-dev-v2`, `policy-dev-v2-fill`,
  `policy-dev-sm`): held-out rows of the corpora above, the checkpoint's own DEV split,
  development splits of the Decision Index training pools, and tasks rendered with the Decision
  Index kit. Also the per-benchmark diagnostic `di_diag.json` that weights the Decision Index
  benchmarks in the threshold search.

So stages 1-4 and the policy selection cannot be rerun from this repository alone. The code
for every stage is here; the evaluation sets the one-time reads use (`eval_suite_v2`,
`eval_final_v2`, `eval_vision_v1`) are needed only to reproduce the reported numbers. Each
stage ran once; none has been rerun to the digit.
