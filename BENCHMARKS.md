# Benchmarks

What each number means and how to reproduce it. The numbers themselves belong to a
release, so they live in that release's record under [`versions/`](versions/) — this
file is the method, and it does not change when a version does.

## What is scored

**The suite: one weighted mean over many benchmarks.** v1.0 was judged on one benchmark;
that turned out to be too narrow to tell a real gain from a benchmark-specific one, so
from v2.0 the headline is a weighted mean over twelve. From v3.0 it is over fifteen: v2.1's
three held-out benchmarks joined the suite. Both weightings are in code (`SUITES` in
`rsijev/targets_suite.py`); a number is only comparable with a release record when it is
on that record's suite.

| benchmark | v3 weight (v3.0 on) | v2 weight (v2.0, v2.1) | what it is |
|---|---|---|---|
| typed-decisions | 0.20 | 0.20 | `LocalLLaMA/typed-decisions`, test split: 400 documents, 2,000 typed decisions |
| Nimble public | 0.121 | 0.15 | public classification decisions, ten upstream sources |
| MMLU-Pro 1k | 0.081 | 0.10 | a guard, not a target: general ability, so decision gains are not bought by forgetting |
| Jev-Style panel | 0.081 | 0.10 | decisions in Jev's own house style |
| Kev transfer | 0.065 | 0.08 | decisions from a different generator than the training teacher |
| Kev hard | 0.065 | 0.08 | items selected for being hard |
| JevBench | 0.049 | 0.06 | the public JevBench split |
| tasksource | 0.05 | – | a broad regrouping of about 600 public task families |
| SemIf external | 0.05 | – | evidence interpretation and rule application |
| scienthoon OOD | 0.05 | – | support-ticket triage |
| Kev documents | 0.041 | 0.05 | long-document decisions |
| Kev devtools | 0.041 | 0.05 | developer-tooling decisions |
| Nimble holdout | 0.041 | 0.05 | Nimble items held out of its own public split |
| procedural | 0.033 | 0.04 | multi-step procedural decisions |
| Open-Jev OOD | 0.032 | 0.04 | deliberately out-of-distribution documents |

Every split is frozen and decontaminated against the others before it is used.

| metric | what it is |
|---|---|
| **pooled top-1** | the fraction of decisions whose highest-probability option matches the teacher's modal label. Pooled across the three question types, because a model can look good on one and fail another |
| **suite mean** | the weighted mean of per-benchmark top-1 over the table above, on the release's suite. The headline from v2.0 on |
| **ECE** | expected calibration error: bin decisions by the confidence the model reported, compare each bin's confidence with its accuracy, average the gaps by bin size. Lower is better. A headline from v2.0 on, because a probability you cannot act on is not an answer |
| **confident-item top-1** | pooled top-1 restricted to items the benchmark's own teacher was sure about (gold max-probability ≥ 0.67, and ≥ 0.9). On typed-decisions 61% of the split is below 0.67 and carries 75% of the errors, so the pooled figure is largely a measure of how you handle the teacher's own uncertainty |
| **options reversed** | every question scored twice, once in the dataset's option order and once reversed. A model answering by position scores differently on the two; the gap is the robustness check |
| **score-mode Brier** | distance from the teacher's full distribution on `score` questions. Judged before any rank correlation, because Spearman is invariant to monotone reshaping and once certified a head whose probabilities had got worse |

## What a model may train on

**Test splits, never.** No release trains on any benchmark's test split, and no training
document shares a state with one. This is verified by the arm runner rather than asserted.

**Train splits: it depends on the release, and it changes what its scores mean.**

| release | trains on benchmark train splits? | so its benchmark scores are |
|---|---|---|
| v1.0 | **no** — a separate synthetic corpus only, with no shared state and no shared 12-word phrase with either split | zero-shot |
| v2.0 | **yes** — typed-decisions and nine other suite benchmarks contributed their train splits | not zero-shot: in-domain, for the ten it trained on |
| v2.1 | **yes** — the same ten plus procedural | not zero-shot: in-domain, for eleven of the twelve. MMLU-Pro is the one it did not train on |

Two releases' benchmark numbers are therefore **not** interchangeable. When comparing
across releases, use the held-out set below.

**The held-out set, read once per release.** Three benchmarks — tasksource, SemIf external
and scienthoon OOD — were frozen before this work began and are scored **once**, after a
release model has been chosen, never during the search. No release trains on them. They are
the only place two releases are measured on equal terms, and they are where to look first
when a release claims an improvement. Each release's record reports them in full: v2.0 was
flat or worse than v1.0 on all three, v2.1 is ahead on all three.

**A held-out set is consumable, so each release freezes the next one.** Once a set has been
read, the choice of what to release has been informed by it, and reading it again for a
later release measures something a little less clean each time. Folding a spent set into the
routine evaluation suite is the right thing to do with it — the measurement gets better —
but only alongside freezing a *new* set from sources the search has not touched. Otherwise
the suite slowly absorbs every independent check the project has, nothing is left outside
it, and no later release can be caught fitting its own benchmarks. The set behind v2.0 and
v2.1 is spent; v2.2 is measured against a new one.

## What you can check yourself

Every split in the suite comes from a public upstream repository, pinned by commit or by a
seeded sample. This repo ships the **loader**, not the data: `rsijev/targets_suite.py` holds
each benchmark's origin, licence, question shapes and label conventions, so you fetch the
same bytes we did rather than a copy of ours.

```bash
python scripts/suite.py --list                     # every benchmark, its pin and its licence
python scripts/suite.py --ckpt DIR                 # the fifteen (--suite v3), as v3.0, v4.0-VL, v5.0-VL, v6.0-VL and v6.1-VL report
python scripts/suite.py --ckpt DIR --suite v2      # the twelve, as v2.0 and v2.1 report
python scripts/suite.py --ckpt DIR --only kev_hard_v1
```

The loaders read a few upstream checkouts from the environment — `KEV_ROOT`,
`NIMBLE_ROOT`, `JEVBENCH_ROOT` and the rest. None has a default: a default would be one
author's filesystem, which is how v1.0 shipped a lab path inside a checkpoint. `--list`
names the variable each benchmark needs, and a loader that is missing one stops the run
and says so; no benchmark is skipped silently. With `--only`, the mean is renormalised over
the weight scored, and the script names the benchmarks it left out. Files fetched by URL
(HANS, in the Jev-Style panel) are cached under `~/.cache/rsi-jev-benches`, or
`$RSIJEV_BENCH_CACHE`, and checked against a pinned sha256.

Two are read in canonical and reversed option order; `nimble_public` and `jev_style_panel`
are averaged over their subsets rather than pooled, because that is what their authors
report. Those conventions live in the loader, so a number you get here and a number in a
release record are the same measurement.

**The decontamination report** (`SUITE_DECONTAM`) names eval cases that overlap training
data we hold; they are dropped at load time. `--no-decontam` keeps them, and the two
numbers should be compared before trusting either.

## How a result becomes a result

- **Seeds scale with the effect.** An arm starts at one seed. A gap of about +0.015 or more
  — several times the paired seed sd — is confirmed by **one fresh seed** the arm has never
  run on. A smaller gap near the bar earns up to three seeds, plus that fresh confirmation
  seed. Spending four seeds on a difference you can already see is compute that buys
  nothing; the confirmation seed is what is load-bearing, because it is the only number the
  arm was not selected on.
- v1.0 used three seeds and **no** confirmation step, and a near-miss that looked real twice
  is what taught us to add one. v2.0 was judged under a four-seed rule with four
  confirmation seeds, which is why its record quotes four; the rule above is what a new arm
  is held to.
- **The bar is the suite's own noise**: +0.006 on the suite mean, no benchmark down by more
  than its own seed noise, MMLU-Pro within 0.030. Per-seed sd is 0.011–0.016 on a single
  benchmark, so single-benchmark differences smaller than that are not findings.
- **The released checkpoint is seed 17**, fixed in advance as primary rather than chosen
  for scoring best.
- **Numbers only compare within one kernel stack.** Every record carries a
  `linear_attn_kernel` stamp; see [`rsijev/README.md`](rsijev/README.md).
- **A checkpoint is re-scored from disk** before publication and must reproduce its
  training run's per-question predictions exactly.
- **An arm that clears the bar but fails exactly one guard is not discarded.** It gets a
  diagnosis of the failing metric and a targeted repair, and becomes a dead end only if the
  repair fails too. What survives that is in [`EXPLORE.md`](EXPLORE.md).

## Reproducing each table

| command | what it prints |
|---|---|
| `python scripts/bench.py --model v3.0-2b` | speed: the fixed cost of reading a document and the marginal cost of one more decision, fitted over 1–32 questions at three document lengths |
| `python scripts/calibration.py --model v3.0-2b --cases 400` | all 2,000 decisions binned by the probability the model gave them, against how often that bin was right |
| `python scripts/routing.py --cases 400` | accuracy against coverage when you act only on the top slice by confidence, and whether a small-model-first cascade earns its place. Runs v1.0's two sizes, the only release with two |
| `python scripts/load_release.py --ckpt DIR --verify` | re-scores a checkpoint against the run that produced it |

`--model` takes `v6.1-vl-4b`, `v6.0-vl-4b`, `v5.0-vl-3b`, `v4.0-vl-2b`, `v3.0-2b`, `v2.1-2b`, `v2.0-2b`, `v1.0-2b` or `v1.0-0.8b`.

Speed moves with GPU load — up to ±20% between runs — so measure on your own
hardware before depending on a figure. Each release's record names the machine its
tables were taken on.
