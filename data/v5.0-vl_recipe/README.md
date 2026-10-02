# v5.0-VL recipe

Two training stages from the base, then a calibrator fitted on the result. One seed (17).

| step | spec | from | what |
|---|---|---|---|
| 1 | `b4-exit20.json` | Qwen/Qwen3.5-4B-Base | text, 3,000 steps; read at an exit after layer 20 of 32; retention term on the exit model |
| 2 | `vis-v4k.json` | stage 1 (`b4-exit20/s17`) | images + text, 3,000 steps; 21,754 image questions, 2,000 new "unknown" text rows, 24,246 replayed |
| 3 | `calibration.json` | stage 2 | cal-4b refit with own-token option pooling (readout B), seed 0 |

The weights are trained with whole-block option pooling. The release reads options from their
own tokens (`option_pool_own_tokens`), set in the release's meta.json and used for step 3.

Paths in the specs are relative to one data directory, `D` below.

```bash
# 1. stage 1: corpus rt4-tap16-ret-b (68 sources; see "Not yet public")
python scripts/release_train.py --model Qwen/Qwen3.5-4B-Base --seed 17 --root D \
    --spec data/v5.0-vl_recipe/b4-exit20.json --corpus D/rt4-tap16-ret-b \
    --save-dir D/b4-exit20/s17 --out D/runs --name b4-exit20

# 2a. image half of the stage-2 corpus: vision roots under D (vision_v1/v2/v3 as in v4.0-VL,
#     plus vision_spatial, vision_joint, vision_recast, vision_sokoban)
W=""; for s in vis3_change_detect vis3_grid_games vis3_line_rule vis3_receipt_math vis3_severity \
               vis3_ui_state vis4_fp_obstacle vis4_grasp vis5_pair vis5_record vis5_rc_count \
               vis5_rc_colour vis6_sokoban; do W="$W --weight $s=1.32"; done
python scripts/build_vis_subset.py --root D --cap 600 $W --weight vis2_iconqa=1 \
    --weight vis5_abstain=3.96 --out D/vis_new
# 2b. the 2,000 new text rows (1,000 KoBBQ-style + 1,000 multilingual abstention) go in D/text_new
# 2c. the stage-2 corpus: new rows + replay, NLI/scope/abstention/negation replay at the control's dose
python scripts/ct_build_corpus.py --parent-corpus D/corpus_hard \
    --parent-sources "$(cat data/v5.0-vl_recipe/replay_sources.txt)" --steps 3000 --batch 16 \
    --protect v4_nli_fever,st_multinli,jsg_anli,st_vitaminc,v4_scope,pf_p1__abst_ml,v4_kobbq,v4_own_choice,v4_own_noul,v4_negation \
    --new D/vis_new/*.jsonl D/text_new/*.jsonl --out D/vis-v4k
# 2d. stage 2
python scripts/release_train.py --model Qwen/Qwen3.5-4B-Base --seed 17 --root D \
    --spec data/v5.0-vl_recipe/vis-v4k.json --corpus D/vis-v4k \
    --save-dir D/vis-v4k/s17 --out D/runs --name vis-v4k

# 3. the calibrator, read with own-token pooling, DEV from stage 1's corpus, seed 0
RSIJEV_OPTION_POOL_OWN_TOKENS=1 python scripts/fit_release_calibration.py --ckpt D/vis-v4k/s17 \
    --root D --lineage --dev-corpus D/rt4-tap16-ret-b \
    --dev-sources "$(python -c "import json;print(json.load(open('data/v5.0-vl_recipe/b4-exit20.json'))['sources'])")" \
    --fit-seed 0 --out D/vis-v4k/s17
```

## Not yet public

These inputs were built by internal tools that are not in this repository yet, so steps 1 and 2
cannot be rerun from public sources alone:

- `rt4-tap16-ret-b`, stage 1's corpus (68 sources; the v3.0 corpus plus generated and public
  sources, decontaminated against every evaluation set).
- `corpus_hard`, the replay stream of stage 2 (`replay_sources.txt`).
- `vision_spatial`, `vision_joint`, `vision_recast`, `vision_sokoban` (generated image roots).
- `text_new`, the 2,000 new text rows of stage 2.

Step 3's command is the public equivalent of the script that fitted the released calibrator; it
has not been rerun to the digit.
