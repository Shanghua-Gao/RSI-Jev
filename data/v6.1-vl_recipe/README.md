# v6.1-VL recipe

v6.1-VL is the average of two checkpoints with the same architecture: v6.0-VL's
(`data/v6.0-vl_recipe/`, stages 1-4) and a second fine-tune of Qwen3.5-4B-Base, member B, trained
on other data. Nothing is trained after the average. One temperature per exit is refit on it,
and the exit policy is chosen again by v6.0-VL's rules.

| step | spec | from | what |
|---|---|---|---|
| 0 | `0-lineage.json` | Qwen/Qwen3.5-4B-Base | both members; member B's stages B0-B2 (text from the base, images, a first domain mix) |
| 1 | `1-member-b-domain-mix.json` | B2 (`b-domain-1/s17`) | member B, stage B3: a second domain mix, 3,000 steps (16 x 8 accumulation); exit heads detached |
| 2 | `2-member-b-heads.json` | B3 (`b-domain-2/s17`) | member B, stage B4: all four heads retrained, tower frozen, 600 steps, seed 1 |
| 3 | `3-soup.json` | v6.0-VL's `heads-s1` and `b-heads-s1` | the uniform average of tower, main head and exit heads |
| 4 | `4-calibration.json` | the average (`soup`) | one temperature per exit, refit; development rows a member trained on dropped first |
| 5 | `5-exit-policy.json` | the average | `auto`'s thresholds (0.85 at 16, 0.50 at 20, confirmed) and the single tau (0.95, the rule's fallback, not confirmed); the package |

Spec 1 is the `release_train` spec as run, with the checkpoint and corpus directories renamed
and the crash-resume keys left out (as for v6.0-VL). The run's own-token option pooling is
recorded by its pre-registration; the job's environment was not re-read. Specs 0 and 2-5
record the settings and results of their scripts. Paths are relative to one data directory,
`D` below.

```bash
# member B, stage B3 (B0-B2: 0-lineage.json)
python scripts/release_train.py --model Qwen/Qwen3.5-4B-Base --seed 17 --root D \
    --spec data/v6.1-vl_recipe/1-member-b-domain-mix.json --corpus D/domain-mix-2 \
    --save-dir D/b-domain-2/s17 --out D/runs --name b-domain-2

# member B, stage B4: all four heads under the own-token readout, tower frozen
RSIJEV_OPTION_POOL_OWN_TOKENS=1 python scripts/fit_head.py --ckpt D/b-domain-2/s17 --corpus D/domain-mix-2 \
    --dev-corpus D/domain-mix-1 --out D/b-heads-s1 --heads all --sdpa-math --max-length 4096 \
    --skip-prefix vis --steps 600 --bs 8 --lr 1e-5 --seed 1

# 3. the average (D/heads-s1 is v6.0-VL's stage-4 checkpoint)
python scripts/soup_checkpoints.py D/soup D/heads-s1 D/b-heads-s1

# 4-5. per-exit dumps of the average, as for v6.0-VL (data/v6.0-vl_recipe/README.md, steps 5-6)
C=D/soup
export RSIJEV_OPTION_POOL_OWN_TOKENS=1
python scripts/dump_exits.py dev --ckpt $C --corpus D/image-corpus --out D/dumps/soup
python scripts/dump_exits.py eval --ckpt $C --out D/dumps/soup --public --set D/eval_suite_v2 --set final=D/eval_final_v2
python scripts/dump_exits.py cases --ckpt $C --pd D/policy-dev-v1 --out D/dumps/pdtext.pt
python scripts/dump_exits.py vision --ckpt $C --root D/policy-dev-v1/vision --out D/dumps/pdvis.pt
python scripts/dump_exits.py vision --ckpt $C --root D/eval_vision_v1 --out D/dumps/visexits.pt
for pd in policy-dev-v2 policy-dev-v2-fill policy-dev-sm; do
  python scripts/dump_exits.py cases --ckpt $C --pd D/$pd --out D/dumps/$pd.pt
done

# development rows that either member trained on, dropped from every dump
python scripts/select_exit_policy.py overlap --dev-dump D/dumps/soup/dev_dump.pt \
    --pd D/policy-dev-v1 --pd D/policy-dev-v2 --pd D/policy-dev-v2-fill --pd D/policy-dev-sm \
    --corpus D/domain-mix-2 --corpus D/domain-mix-1 --corpus D/image-corpus --out D/policy/excl.json
X=D/dumps-x; mkdir -p $X
python scripts/select_exit_policy.py exclude --ids D/policy/excl.json --out D/policy/exclude.json \
    --pair D/dumps/soup/dev_dump.pt $X/dev_dump.pt --pair D/dumps/pdtext.pt $X/pdtext.pt \
    --pair D/dumps/policy-dev-v2.pt $X/policy-dev-v2.pt --pair D/dumps/policy-dev-v2-fill.pt $X/policy-dev-v2-fill.pt \
    --pair D/dumps/policy-dev-sm.pt $X/policy-dev-sm.pt

# temperatures (refit, printed as "T") and auto's thresholds; then the single tau; then one evaluation read
TH="--dev-dump $X/dev_dump.pt --text-dump $X/pdtext.pt --vis-dump D/dumps/pdvis.pt --vis-cases D/policy-dev-v1/vision \
    --bench-weights D/di_diag.json --chance KIT/data/index-0.2.json \
    --heldout $X/policy-dev-v2.pt D/policy-dev-v2 --heldout $X/policy-dev-v2-fill.pt D/policy-dev-v2-fill \
    --shift-matched $X/policy-dev-sm.pt D/policy-dev-sm"
python scripts/select_exit_policy.py thresholds $TH --out D/policy/auto.json
python scripts/select_exit_policy.py thresholds $TH --single --out D/policy/single.json
python scripts/select_exit_policy.py thresholds $TH --eval --eval-dump D/dumps/soup/eval_dump.pt \
    --eval-vis-dump D/dumps/visexits.pt --out D/policy/auto.json

# the package: one temperature per exit from the refit, the single tau, exit 12 dropped, auto's thresholds
python scripts/package_multiexit.py --ckpt $C --temperatures D/policy/auto.json --single-tau D/policy/single.json \
    --drop-exit 12 --thresholds D/policy/auto.json --out D/package
```

The thresholds step without `--policy` refits the temperatures instead of checking them against
a cascade selection; that refit is the one the package carries. The main exit's temperature is
the same number in `calibration.safetensors` (`cal_logT`) and in `calibration.json`
(`main_logT`): 0.02, T = 1.02.

The released package carries these settings (`4-calibration.json`, `5-exit-policy.json`) and,
like v6.0-VL's, is self-contained.

## Not public

These inputs are not in this repository and will not be published, as for v6.0-VL:

- v6.0-VL's corpora (`text-corpus`, `image-corpus`; see `data/v6.0-vl_recipe/README.md`).
- Member B's corpora: `b-text-corpus` (95 sources, stage B0), `domain-mix-1` (158 sources, B2) and
  `domain-mix-2` (171 files, B3), and their builders. They build on earlier unpublished corpora
  and generated sources and include the Decision Index training pools, some of them
  non-commercial or share-alike (listed in the record).
- The policy development sets and `di_diag.json`, as for v6.0-VL.

So member B and the policy selection cannot be rerun from this repository alone. The code for
every step is here, and the average itself can be rebuilt from the two head-stage checkpoints.
Each stage ran once; none has been rerun to the digit.
