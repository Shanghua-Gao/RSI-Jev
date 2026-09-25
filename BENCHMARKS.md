# Benchmarks

What each number means and how to reproduce it. The numbers themselves belong to a
release, so they live in that release's record under [`versions/`](versions/) — this
file is the method, and it does not change when a version does.

## What is scored

`LocalLLaMA/typed-decisions`, test split: 400 documents, 2,000 typed decisions.

| metric | what it is |
|---|---|
| **pooled top-1** | the headline: the fraction of all 2,000 decisions whose highest-probability option matches the teacher's modal label. Pooled across the three question types, because a model can look good on one and fail another |
| **confident-item top-1** | the same, restricted to items the benchmark's own teacher was sure about (gold max-probability ≥ 0.67, and ≥ 0.9). 61% of the split is below 0.67 and carries 75% of the errors, so the pooled figure is mostly a measure of how you handle the teacher's own uncertainty |
| **options reversed** | every question scored twice, once in the dataset's option order and once reversed. A model answering by position scores differently on the two; the gap is the robustness check |
| **MMLU-Pro 1k** | a guard metric, not a target: checked so that gains on decisions are not bought by damaging general ability |
| **score-mode Brier** | calibration in one number, on `score` questions. Judged before any rank correlation, because Spearman is invariant to monotone reshaping and once certified a head whose probabilities had got worse |

**Generalist rule.** Training never touches the benchmark's train split. Models are
trained on a separate synthetic corpus, and the separation is verified rather than
asserted: no training document shares a state with either split, and none shares
even a single 12-word phrase.

## How a result becomes a result

- **Three seeds** (17/29/43), judged on the mean. Per-seed sd is 0.011–0.016, so a
  difference of two 3-seed means has an sd of about 0.010 — anything smaller is not
  a finding, and is reported as one at nobody's peril but ours.
- **The released checkpoint is seed 17**, fixed in advance as primary, not the best
  of the three.
- **Numbers only compare within one kernel stack.** Every record carries a
  `linear_attn_kernel` stamp; see [`rsijev/README.md`](rsijev/README.md).
- **A checkpoint is re-scored from disk** before publication and must reproduce its
  training run's per-question predictions exactly.

## Reproducing each table

| command | what it prints |
|---|---|
| `python scripts/bench.py --model 2b` | speed: the fixed cost of reading a document and the marginal cost of one more decision, fitted over 1–32 questions at three document lengths |
| `python scripts/calibration.py --model 2b --cases 400` | all 2,000 decisions binned by the probability the model gave them, against how often that bin was right |
| `python scripts/routing.py --cases 400` | accuracy against coverage when you act only on the top slice by confidence, and whether a small-model-first cascade earns its place |
| `python scripts/load_release.py --ckpt DIR --verify` | re-scores a checkpoint against the run that produced it |

Speed moves with GPU load — up to ±20% between runs — so measure on your own
hardware before depending on a figure. Each release's record names the machine its
tables were taken on.
