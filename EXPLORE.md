# Exploration

Where the loop has been between releases, and what each direction turned out to
hold. Releases themselves are in [`versions/`](versions/); this is the territory
around them.

The agent transcripts, proposals and queues stay private — publishing them would
let a reader re-run our search instead of checking our result. The arms and their
numbers are here in full.

## jevtr_v1 — the first multi-agent exploration of training (2026-09-23/24)

Three analyst agents, six GPU agents, a monitor and an orchestrator, sent out along
the model, data and training axes from v1.0's recipe. **24 directions explored. None
came back with a keeper — and that is the finding.**

The bar was fixed in advance: better than **+0.020** on the 3-seed mean, all seeds
stable, MMLU-Pro within 0.020, same kernel stack. Champion 0.6622.

<p align="center">
  <img src="assets/loop-line.gif" width="880" alt="Six turns of the cycle climb to v1.0 and the
  champion line rises with them; the twenty-four directions since press against that line
  without crossing it">
</p>

| axis | arms | best | what it was |
|---|---|---|---|
| training | 10 | **+0.0035** | keep-last-k 100 — not confirmed at three more seeds (6-seed mean 0.6626) |
| model | 5 | **+0.0008** | option-marker capacity control |
| data | 8 | **−0.0015** | entropy filter, keep below 0.5 |
| control | 2 | +0.0003 | champion rerun, which is the noise floor |

Four arms came out above the champion and none by enough to mean anything: +0.0035,
+0.0013, +0.0010, +0.0008, against a per-seed sd of 0.011–0.016. One arm failed on
infrastructure — another session deployed into the live evaluator tree — and was
rerun. Three data arms used teacher labels read by a biased first-token readout, so
they do not test the axis they were aimed at.

### What the territory turned out to be like

- **The benchmark's teacher is unsure about most of it.** 61% of test items have a
  gold max-probability below 0.67, and they carry 75% of the errors. On those the
  champion already beats the teacher's own mass, so there is little left to win.
  Matching the teacher where it *is* confident is worth about +0.024.
- **The model fits the wrong teacher well.** 0.850 on held-out synth against 0.662
  on the benchmark. More capacity and more synth fit both transferred worse.
- **The bar was above the resolution.** A difference of two 3-seed means has an sd
  of about 0.010, so a realistic single-axis effect of +0.005 is invisible against a
  +0.020 threshold. The run was not powered to find what it was looking for.
- **What has ever worked here was capability, not fit** — fine-tuning the tower
  (+0.13) and 0.8B to 2B (+0.048). The readout head is interchangeable.

That last point is the run's most useful output, and it is a negative: eight data
arms and ten training arms moved nothing, which says the next release has to change
what the model *can do*, not how it is fitted.

### What comes back with us

- **RL is stable on this base** at last (arm 24): learning rate ×0.2, KL β 0.5 to
  the SFT snapshot, and a hard KL gate. It still needs a reward carrying information
  the SFT target does not.
- **A reasoning teacher is worth distilling.** Zero-shot Qwen3.5-4B with a
  1,024-token thinking budget scores 0.677 on the test split and 0.964 where its own
  labels are confident. It cannot serve — Jev is System One and generates nothing —
  but it can label or reward at training time.
- **Order averaging at inference is null**: +0.001, canonical plus reversed.
- **Tooling that outlived the run:** a confident-item metric, a stricter stability
  rule, a frozen worker tree, an orphan watchdog, and an append-only result log.

### What it cost

The infrastructure failure above is worth naming, because it was ours: a deploy into
the tree the evaluator was reading cost the run three seeds and one arm. Agents run
against frozen snapshots from arm 10 onward for that reason.

## jevtr_v1, continued — from v1.0 to v2.0 (2026-09-24/26)

The same loop with a revised protocol (four seeds per arm, a diagnostic step, and a
portfolio that must include regime-level bets), later with a 12-benchmark evaluation
suite. **About 50 more arms; two lasting wins, both about data.** v2.0 is built from
them.

Three things changed during this stretch, all at the run owner's direction, and they
shape how to read the numbers:

- **The benchmark's train split became allowed.** v1.0 never saw any
  `LocalLLaMA/typed-decisions` data. From here on, training may use the *train* split of
  any benchmark; *test* splits never, checked by the arm runner. v2.0's scores on those
  benchmarks are therefore **not zero-shot**, unlike v1.0's.
- **One benchmark became twelve.** typed-decisions test, Nimble (public and holdout), MMLU-Pro
  1k, the Jev-Style panel, Kev transfer, hard, documents and devtools, JevBench, a
  procedural set and Open-Jev OOD, frozen and decontaminated. The headline became the
  weighted suite mean, and the KEEP bar was recalibrated to the suite's own noise
  (+0.006, confirmed on four fresh seeds).
- **Three more benchmarks were set aside and never looked at** until release, so the
  release numbers carry no selection bias from dozens of KEEP decisions: tasksource,
  SemIf external and scienthoon OOD.

The bar was fixed in advance and recalibrated once, when the suite replaced the single
benchmark: better than **+0.006** on the four-seed suite mean, no benchmark regressed
beyond its own seed noise, MMLU-Pro within 0.030, all seeds stable, same kernel stack,
then repeated on four fresh seeds against a baseline run on those same seeds.

| axis | arms | best | what it was |
|---|---|---|---|
| data | 23 | **+0.072** | the other benchmarks' train splits, 3k cases per source — this became v2.0 |
| calibration | 8 | **ECE −0.041** | an out-of-fold confidence head; the shipped variant gives back 0.010 of that to keep its score answers faithful to the teacher |
| training | 5 | +0.0003 | an order-consistency penalty — inside the noise, like the ten before it |
| tooling | 8 | — | the 12-benchmark suite, a 2.48M-question inventory, the adaptive generator |
| control | 5 | +0.0010 | the champion repeated on fresh seeds, which is the noise floor |

Eight of the 49 were tooling that produced no number of its own, and five were controls.
Of the 36 that were judged, **four were kept**: the benchmark's own train split with
general-knowledge replay, the other benchmarks' train splits, and two forms of the
confidence head. None of the five training arms was kept, and no
reinforcement-learning arm has been kept anywhere in this run.

### What worked

| what | effect | notes |
|---|---|---|
| **Train on a benchmark's own train split** | typed-decisions 0.662 → **0.797** | general-knowledge multiple-choice replay at 15% keeps MMLU-Pro within its guard |
| **The same, for the other benchmarks' train splits** (capped at 3k cases each) | suite mean 0.633 → **0.705**, every seed up, confirmed on fresh seeds | Kev hard +0.30, Nimble holdout +0.19, Kev documents +0.15; MMLU-Pro −0.027 |
| **An out-of-fold confidence head** for calibration | suite ECE 0.105 → **0.055–0.074** | one forward pass, answers unchanged; fixes over- *and* under-confidence, which a single temperature cannot |
| Generated data shaped like the test profile | +0.003 to +0.009 on the suite mean | real gains on out-of-distribution benchmarks, but they overlap each other, and they add nothing on top of the train splits |

### What did not work

| what | result | why, as far as we can tell |
|---|---|---|
| **Distilling a reasoning teacher** (Qwen3.5-4B, thinking), four ways: mixed labels, pure labels, targeted swaps, RL reward | −0.007 to −0.016 | the teacher is better at the task (0.677 zero-shot) but disagrees with the benchmark's own teacher exactly on the ambiguous items, and that is where its labels move the student |
| **RL on the decision itself**: reward = SFT target, reasoning-teacher agreement, exact accuracy + Brier, order consistency, calibration (RLCR) | −0.007 to +0.000; calibration RL moved ECE by only 0.009 | a one-pass decision model with a known answer for every option is a one-step, full-information bandit; RL either reproduces the supervised optimum or, KL-anchored, cannot move far enough. We found no published result showing RL beating SFT on the same labels for this setting |
| **A bandit over the data mix** | −0.006 | small slices got outsized credit, and it optimised its own probes while test fell. It lives on as the policy of the data generator, with those fixes |
| **More public data in bulk** (a 2.48M-question pool, filtered) | 50k: −0.017; 200k: −0.024, MMLU-Pro −0.09 | hundreds of thousands of one-hot classification rows erode general knowledge and inflate confidence |
| **Stacking the generated corpora** | +0.0015 | their gains overlap; MMLU-Pro falls as more is added |
| **More replay to protect MMLU-Pro** in the v2.0 recipe | no better | 15% and 30% replay did not lift MMLU-Pro; a science-weighted 10% helped slightly and is being confirmed |
| A single global temperature for calibration | ECE unchanged | the model was under-confident on typed-decisions and over-confident elsewhere; one number cancels out |

### What comes back with us

- **Data aimed at the model's weaknesses, not bulk data.** An adaptive generator
  (Azure GPT-5.x and self-hosted models, with a gpt-6-class model only for hard cases)
  now profiles the current model on dev probes, writes cases for its weakest cells, and
  checks them for grounding and triviality. On v1.0's base it helped; on v2.0's base
  the train splits already cover what it taught, so its next rounds aim at what v2.0
  still gets wrong: wide label spaces, negation, dates.
- **RL has a job, just not on the decision head**: choosing what to generate next,
  rewarded by the student's measured progress. That head-to-head is the next experiment.
- **Near-misses are repaired, not dropped**: an arm that clears the bar but fails one
  guard gets a diagnosis and a targeted-data repair, instead of being discarded.

### Every arm, with its numbers

**Outcomes:** *kept* entered the champion line · *improved, not kept* beat the champion
by less than the bar, or failed a guard · *rejected* did neither · *noise* landed inside
the noise floor · *control* is a reference run, not a bet · *tool* produced no score of
its own.

Twenty-one arms were judged on typed-decisions pooled top-1, the v1.0 headline. The
suite was built partway through, and the twenty-eight after it were judged on the
weighted suite mean. The two groups are not comparable, so they are listed apart.

**Judged on typed-decisions top-1** · champion 0.6625, then 0.796 from `spec-2` onward

| arm | axis | outcome | typed-decisions | MMLU-Pro | suite ECE | what it was |
|---|---|---|---|---|---|---|
| `tool-1-validate-q35-4b-thinking-traces` | tool | tool | — | — | — | validated a reasoning teacher: 0.677 zero-shot, 0.964 where it is confident |
| `training-10-distill-reasonteacher-sft` | training | rejected | 0.6514 | 0.341 | — | supervised distillation from that teacher, mixed with the synth target |
| `data-14-trace-distill-pure-t3` | data | rejected | 0.6458 | 0.327 | — | the same, on the teacher's labels alone |
| `data-15-trace-surgical-correct` | data | rejected | 0.6482 | 0.331 | — | the reasoning teacher's labels only where it disagrees with the benchmark's teacher (2,775 of 24,320 questions) |
| `training-9-rl-reasonteacher-reward` | training | rejected | 0.6552 | 0.349 | — | the teacher's agreement as an RL reward |
| `tool-2-collect-all-decision-data` | tool | tool | — | — | — | inventoried 2.48M public decision questions, 34 sources, licences recorded |
| `tool-4-generation-pipeline` | tool | tool | — | — | — | a writer/judge pipeline for new cases, with grounding checks |
| `spec-1-synth-plus-td-train` | data | rejected | 0.7794 | 0.332 | — | **first arm allowed the benchmark's train split**; MMLU-Pro paid for it |
| `spec-1-synth-plus-td-train-x3` | data | rejected | 0.7993 | 0.331 | — | the same, each case seen three times |
| `tool-3-filter-decision-pool` | tool | tool | — | — | — | filtered that pool to a usable corpus |
| `spec-2-x3-plus-mc-replay-r15` | data | **kept** | 0.796 | 0.346 | — | **+ 15% general-knowledge replay — the first keeper** |
| `gen-1-pool-v2-50k` | data | rejected | 0.6455 | 0.301 | — | 50k questions from the public pool |
| `spec-2-x3-plus-mc-replay-r30` | data | rejected | 0.7915 | 0.347 | — | 30% replay instead of 15% |
| `gen-2-pool-v2-200k` | data | rejected | 0.6384 | 0.265 | — | 200k from the pool; general knowledge fell hardest here |
| `spec-3-replay-dose-r05` | data | rejected | 0.7951 | 0.335 | — | 5% replay |
| `spec-3-replay-dose-r10` | data | improved, not kept | 0.7964 | 0.330 | — | 10% replay; 15% stayed the best dose |
| `tool-5-eval-suite` | tool | tool | — | — | — | **built the 12-benchmark suite**, frozen and decontaminated |
| `training-13-accbrier-exact-anchor` | training | rejected | 0.7908 | 0.341 | — | exact accuracy + Brier reward, KL-anchored |
| `training-12-orderconsist-specialist-w1` | training | improved, not kept | 0.7963 | 0.340 | — | penalise disagreement between option orders |
| `training-11-lp-bandit-slice-pilot` | training | rejected | 0.79 | 0.333 | — | a bandit re-weighting the existing data mix |
| `gen-data-1-teacher-select-and-10k` | tool | tool | — | — | — | teacher-selected generation, 10k cases |

**Judged on the 12-benchmark suite mean** · champion 0.6327

| arm | axis | outcome | suite mean | MMLU-Pro | suite ECE | what it was |
|---|---|---|---|---|---|---|
| `ctl-suite-best` | control | control | 0.6327 | 0.349 | — | the champion on the new suite — **the bar for everything below** |
| `ctl-suite-ref` | control | control | 0.609 | 0.351 | — | **the v1.0 recipe on the new suite** — the release-to-release reference |
| `tool-6-adaptive-generation-loop` | tool | tool | — | — | — | **the adaptive generator**: profile the model, write for its weak cells |
| `gen-arm-1-best-plus-testprofile` | data | improved, not kept | 0.6361 | 0.343 | — | generated data shaped like the test profile |
| `gen-arm-2-best-plus-refprior` | data | improved, not kept | 0.6376 | 0.344 | — | the same, with a reference-value prior |
| `suite-train-1-cap1k` | data | rejected | 0.6932 | 0.325 | — | **the other benchmarks' train splits**, 1k cases per source |
| `cal-2a-rlcr-tdsynth-pool` | calibration | rejected | 0.6265 | 0.337 | 0.0986 | calibration RL (RLCR reward) on a narrow pool |
| `cal-2b-rlcr-broad-pool` | calibration | rejected | 0.6381 | 0.336 | 0.0963 | the same on a broad pool |
| `cal-1-global-temperature` | calibration | rejected | 0.6353 | 0.349 | 0.1073 | one temperature for the whole model |
| `cal-4-oof-confidence-head` | calibration | **kept** | 0.6362 | 0.357 | 0.0639 | **an out-of-fold confidence head** — the calibration keeper |
| `tool-6-prod-round-1` | tool | tool | — | — | — | the generator's first production round, 5,147 cases, hand-checked |
| `suite-train-1-cap3k` | data | **kept**¹ | 0.7049 | 0.323 | — | **3k cases per source — v2.0's recipe** |
| `cal-4-confirm` | calibration | **kept** | 0.6337 | 0.323 | 0.0603 | the head repeated on four fresh seeds |
| `cal-4b-oof-scorefloor` | calibration | **kept** | 0.6349 | 0.352 | 0.0736 | **the head, forbidden to sharpen score questions** — shipped in v2.0 |
| `stack-1a-both-generated` | data | rejected | 0.6342 | 0.319 | 0.1208 | both generated corpora at once |
| `stack-1b-both-generated-r15x2` | data | rejected | 0.6261 | 0.313 | 0.1343 | the same, with the original data doubled |
| `adapt-r1-best-plus-round1` | data | improved, not kept | 0.6415 | 0.344 | 0.1139 | the generator's round-1 corpus on v1.0's base |
| `adapt-r1-ctl-best-s17-ckpt` | control | control | 0.6372 | 0.352 | 0.0973 | a matched single-seed baseline for that comparison |
| `suite-train-2-cap3k-rep15` | data | rejected | 0.7044 | 0.317 | — | 15% replay, to win MMLU-Pro back |
| `ctl-suite-best-fresh` | control | control | 0.6337 | 0.337 | 0.1049 | the champion on four fresh seeds — **the paired noise floor** |
| `cal-4c-joint-l05` | calibration | rejected | 0.6355 | 0.359 | 0.0683 | top-label and soft-label losses jointly, weight 0.5 |
| `cal-4c-joint-l2` | calibration | rejected | 0.637 | 0.358 | 0.0768 | the same at weight 2 |
| `adapt-r1-repair-R1-kevdoc-anchor` | data | improved, not kept | 0.6449 | 0.340 | 0.1094 | round 1 + the regressed benchmark's train split as an anchor |
| `adapt-r1-repair-RS-cap3k-plus-r002` | data | noise | 0.7038 | 0.302 | 0.0937 | round 1 on top of v2.0's recipe |
| `suite-train-1-cap3k-confirm` | data | **kept** | 0.7029 | 0.309 | — | **v2.0's recipe on four fresh seeds — confirmed** |
| `suite-train-2-cap3k-rep15stem10` | data | improved, not kept | 0.7029 | 0.331 | — | replay weighted toward science and technical questions |
| `suite-train-2-cap3k-rep15stemall10` | data | rejected | 0.703 | 0.315 | — | the same, across all subjects |
| `rc-A-cap3k-cal4b` | control | control | 0.7056 | 0.305 | 0.0546 | **the released v2.0 checkpoint**: v2.0's recipe + the confidence head |

¹ `suite-train-1-cap3k` is logged as rejected and was later kept. It failed on one guard
only — MMLU-Pro −0.026 against a 0.020 limit — and the run owner widened that limit to
0.030 on the grounds that MMLU-Pro is a guard against forgetting, not a target. It was
then confirmed on four fresh seeds and became v2.0. The original rejection is left in the
log, because a bar moved after seeing the result is the kind of thing a reader should be
able to catch us doing.

## jevtr_v1, continued — from v2.0 to v2.1 (2026-09-26/27)

The same loop, same suite, same bar, with one rule changed: **seeds now scale with the
effect** (one seed by default, a gap of about +0.015 or more confirmed by one fresh seed,
smaller gaps up to three plus that confirmation seed) instead of four seeds for everything.
Four-seed arms below were run under the old rule; the two-seed ones under the new.

**Forty-three more arms; one lasting win, and it is not about data.** v2.1 is built from it.
Some rows in the table at the end pair an RL arm with its own matched control, so the 43 arms
fit in 37 rows.

Two questions drove the stretch. The first was v2.0's one real regression: it had given up
general knowledge (MMLU-Pro 0.305 against v1.0's 0.355) to gain on the decision benchmarks,
and the near-miss repair rule says that is to be diagnosed, not accepted. The second was
whether reinforcement learning could beat supervised training anywhere in this setting, with
matched controls this time — the earlier RL arms had been judged against the champion rather
than against the supervised run of the same data, which is the comparison that settles it.

| axis | arms | best | what it was |
|---|---|---|---|
| optimiser | 6 | **+0.022 suite, +0.054 MMLU-Pro** | the tower's bottom 8 of 24 layers at one tenth the learning rate — this became v2.1 |
| data | 6 | +0.008 MMLU-Pro | replay weighted toward science, half recast to ten options; no suite gain |
| reinforcement learning | 20 | +0.002 | five setups, four of them with a matched SFT control; two failed to run, none of the rest cleared the bar |
| scale | 7 | — | 4B would not train; 0.8B gained 0.12 over its own reference and stayed 0.05 below 2B |
| tooling / control | 4 | — | a 13-benchmark suite, and the single sanctioned read of the held-out set |

### What worked

| what | effect | notes |
|---|---|---|
| **The tower's lower third at one tenth the learning rate** | MMLU-Pro 0.305 → **0.359**, suite 0.705 → **0.726** | the whole recovery, at no cost to any benchmark. Confirmed on a fresh seed, where the gain was *larger* |
| Replay weighted toward science, half of it recast to MMLU-Pro's ten-option shape | MMLU-Pro +0.008 | the best any data arm managed, and it is an eighth of what the learning rate managed |
| The procedural benchmark's train split | procedural 0.62 → **0.87** | the same in-domain lever v2.0 was built on, applied to the one decision benchmark v2.0 had got worse at |

### What did not work

| what | result | why, as far as we can tell |
|---|---|---|
| **A retention KL** against the base model's next-token distribution | MMLU-Pro 0.3085, worse than v2.0 | the KL held near 0.02 nats/token, so the *language model* was preserved and the decision readout's general knowledge eroded regardless. What the readout uses is not what next-token prediction protects. +22% training time |
| **Halving the whole tower's learning rate** | +0.004 MMLU-Pro, inside the noise | the lower layers are where it matters; a uniform change dilutes it away |
| **Freezing the tower's lower third** | MMLU-Pro +0.016 on 4/4 seeds, but JevBench −0.026, Kev devtools −0.021, Nimble holdout −0.026 | those layers do have to adapt to the decision task. Slowing them beats stopping them |
| **More replay** (30%), or recasting *all* of it to ten options | no better than 15% and half | the dose and the shape were already right |
| **More train-split data** (6k per source instead of 3k) | +0.002, sign inconsistent, one regression | the dose is saturated past 3k |
| **Eleven extra coverage sources** | suite −0.001, MMLU-Pro 0.2845 | breadth of public classification data erodes general knowledge, as it did at 200k |
| **Two data-selection policies head to head** (a heuristic against EXP3 over sources) | 0.703 against 0.707, inside the noise | neither policy beats taking the mix as given, at this corpus size |
| **Reinforcement learning, five setups, four with a matched SFT control** | RL 0.7063 / ctl 0.7075 · RL 0.7079 / ctl 0.7056 · RL 0.7065 / ctl 0.7082 · RL 0.7093 / ctl 0.7074 | every RL arm landed at or below the supervised run it was matched against, and the two that beat the champion beat it by +0.002. With a known target distribution for every option and one step per decision, the supervised optimum is already the answer; a KL-anchored policy cannot get far from it, and an unanchored one degrades (0.6892 at a higher policy learning rate). Counting every arm whose training signal was a reward rather than the teacher's target distribution, this project has now run **eighteen, and kept none** |
| **A 4B tower** on v2.0's recipe | would not train | at the same batch and schedule; not pursued further |
| The v2.1 treatment **at 0.8B** | 0.6834, rejected | the 0.8B line has its own keeper at 0.6791 (against its 0.5547 reference), still 0.05 below 2B |

### What comes back with us

- **When something is forgotten, look at the learning rate before looking at the data.**
  Four repairs were tried; the three data-and-regularisation ones bought +0.008, +0.004 and
  a trade. Slowing the layers the knowledge sits in bought +0.054 and cost nothing.
- **Preserving a language model is not preserving what a readout reads.** The retention KL
  is the cleanest negative result of the stretch, because it succeeded at its own objective
  and failed at the goal.
- **RL is done here.** Eighteen arms across the project, eleven of them in this stretch, the
  last four against matched supervised controls. It is not that RL is weak; it is that a
  one-step, full-information decision with a known target distribution leaves it nothing to
  find.
- **A held-out set is spent once it is read.** The three benchmarks frozen before v2.0 have
  now decided two releases. Folding them into the routine suite is right, but only alongside
  freezing a new set; otherwise the suite absorbs every independent check the project has.
  See [`BENCHMARKS.md`](BENCHMARKS.md).

### Every arm, with its numbers

**Judged on the 12-benchmark suite mean** · champion 0.7049 (`suite-train-1-cap3k`), the
recipe v2.0 shipped

| arm | axis | outcome | suite mean | MMLU-Pro | what it was |
|---|---|---|---|---|---|
| `mmlu-keep-1a-retention-kl` | training | rejected | 0.7046 | 0.3085 | a KL against the base model's next-token distribution, to stop forgetting |
| `mmlu-keep-1b-st-tower-half` | training | rejected | 0.7067 | 0.3273 | the whole tower at half the learning rate |
| `mmlu-keep-1c-freeze-lower-third` | training | rejected | 0.7030 | 0.3390 | freeze the tower's lower third — recovers knowledge, breaks three benchmarks |
| `suite-train-2-cap3k-rep15stem10-confirm` | data | **kept** | 0.7024 | 0.3245 | science-weighted replay confirmed on fresh seeds |
| `suite-train-2-cap3k-rep30` | data | rejected | 0.7041 | 0.3097 | 30% replay instead of 15% |
| `suite-train-3a-cap6k` | data | rejected | 0.7058 | 0.3085 | 6k train-split cases per source instead of 3k |
| `coverage-1-cap3k-plus-cov` | data | rejected | 0.7036 | 0.2845 | eleven further public classification sources |
| `round2-a-cap3k-plus-heuristic` | data | noise | 0.7030 | 0.3050 | generator round 2, heuristic source selection |
| `round2-b-cap3k-plus-exp3_mixed` | data | noise | 0.7072 | 0.3097 | the same, EXP3 over sources |
| `ctl-2b-cap3k-adamw8bit` | control | control | 0.6953 | 0.2970 | 8-bit AdamW on the tower, for the 4B feasibility question |
| `rl-env-1-cap3k` | control | control | 0.7056 | 0.3050 | the v2.0 checkpoint inside the new RL environment, untrained — the environment's own reference |
| `rl-env-1-sftgold` | training | control | 0.7021 | 0.3080 | supervised on the environment's gold answers |
| `rl-env-1-sftgold-b` | training | control | 0.7074 | 0.3045 | the same, second configuration |
| `rl-env-1-rl` | training | failed | — | — | the first RL run in that environment: would not complete |
| `rl-env-1-rl-b` | training | failed | — | — | the second, likewise |
| `rl-env-1-rl-c` | training | rejected | 0.7098 | 0.3135 | RL in that environment; +0.002 on its control, below the bar |
| `rl-env-2-a-sftrl` / `-ctl` | training | improved, not kept | 0.7063 / 0.7075 | 0.305 / 0.305 | SFT-then-RL against its own SFT control — **below it** |
| `rl-env-2-a-sftrl-confirm` / `-ctl-confirm` | training | improved, not kept | 0.7086 / 0.7068 | 0.309 / 0.303 | the same pair on a fresh seed; +0.002, below the bar |
| `rl-env-2-b-rl` | training | rejected | 0.7076 | 0.3080 | a second reward in the same environment |
| `rl-env-2-c-rl` / `-ctl` | training | improved, not kept | 0.7079 / 0.7056 | 0.309 / 0.305 | a third; +0.002 |
| `rl-env-2-c-rl-confirm` | training | improved, not kept | 0.7066 | 0.307 | and on a fresh seed it did not hold |
| `rl-env-3-sftrl5-gate` / `-sft5` | training | rejected | 0.7093 / 0.7074 | 0.312 / 0.310 | a gated reward against its SFT control |
| `rl-env-4-sftrl5v3-hilr` / `-sft5v3` | training | rejected | 0.6892 / 0.7068 | 0.305 / 0.306 | a higher policy learning rate — clearly worse |
| `rl-judge-1-rl` / `-ctl-sft` | training | rejected | 0.7065 / 0.7082 | 0.3225 / 0.326 | a learned judge as the reward, against its SFT control — below it |
| `scale-4b-cap3k` | model | failed | — | — | v2.0's recipe on a 4B base: would not train |
| `s08b-1b-cap3k` | model | failed | — | — | the same at 0.8B, first attempt |
| `ctl-08b-ref` | control | control | 0.5547 | 0.2565 | the 0.8B reference for the arms below |
| `s08b-1c-rep15stem10` | model | improved, not kept | 0.6767 | 0.2687 | the science-weighted recipe at 0.8B |
| `s08b-1c-rep15stem10-confirm` | model | **kept** | 0.6791 | 0.2717 | confirmed: +0.124 over its reference, 0.05 below 2B |
| `s08b-1d-cal4b-rep15stem10` | training | control | 0.6757 | 0.2580 | the confidence head at 0.8B |
| `s08b-2a-lowerlr01-proc` | model | rejected | 0.6834 | 0.2720 | v2.1's treatment at 0.8B — it does not carry down |
| `cand-2b-1b-freeze4-stem-proc` | training | rejected | 0.7185 | 0.3270 | the new corpus with the bottom 4 layers frozen |
| `cand-2b-1a-lowerlr01-stem-proc` | training | improved, not kept | 0.7258 | 0.3590 | **the new corpus with the bottom 8 layers at lr ×0.1** |
| `cand-2b-1a-lowerlr01-stem-proc-confirm` | training | **kept** | 0.7286 | 0.3600 | confirmed on fresh seed 97, larger there than where it was selected |
| `tool-suite-v2` | tool | tool | — | — | a 13-benchmark suite; retired unapplied, because folding the held-out sets into the routine suite would have spent them |
| `final-report-rcA-vs-v1` | control | control | — | — | the single sanctioned read of the held-out set for v2.0 |
| `rc-B-cand1a-cal4b` | control | control | **0.7291** | **0.3830** | **the released v2.1 checkpoint**: the recipe + v2.0's confidence head, seed 17 |

## jevtr_v1, continued — from v2.1 to v3.0 (2026-09-27/28)

Same loop, one new evaluator: a 15-benchmark suite (v2.1's three held-out benchmarks joined it)
and a new held-out set, eval_final_v2, whose mean is the **final** column below. The suite decided
every keep. Final was read along the way as a consistency check, so it is held out from training,
not sealed from the search. Arms are counted one per run record, a rerun counting once:
**120 arms**, 33 of them short RL pilots with no suite read.

**Two wins built v3.0: more data, and the first RL stage that beat its supervised control.**

| axis | arms | best | what it was |
|---|---|---|---|
| data | 26 | **+0.011 suite** (+0.013 on a fresh seed) | 84k more questions from emotion, NLI, tasksource and Open-Jev: the v3.0 corpus |
| reinforcement learning | 41 | **hippo R@1 +0.052 over SFT on the same rows** (z 3.9) | a listwise NDCG reward for reranking; the RL stage of v3.0 |
| model | 20 | +0.002 suite, twice | an MLP to combine option cross-attention; 0.8B, context-length and scorer-depth sweeps |
| stack / vision | 4 | image eval 0.809 vs 0.796 | the first image arm; and a stack of every kept data component, below the bar |
| control / tooling | 29 | — | matched controls, evaluator smokes, length bucketing (adopted) |

### What worked

| what | effect | notes |
|---|---|---|
| **More coverage**: emotion / SST-5 / ANLI, 30k tasksource, 10k Open-Jev | suite 0.748 → **0.759** | confirmed on a fresh seed; the whole suite gain of this release |
| **Listwise reranking RL**, KL penalty only | hippo per-candidate R@1 **+0.052** over pointwise SFT on the same rows | on v3.0's own parent: 0.192 → **0.308**, suite −0.003 |
| MLP combine for option cross-attention | +0.002 suite, twice | small, method-only, and the held-out set agreed |
| Length bucketing (64) | suite-neutral | cheaper training; every later arm uses it |

### What didn't

- **Every per-item RL objective** — RLCD reconstructions (binary, proper-score, RLCR, bandit),
  decision-utility and confidence-ranking objectives — tied or lost to SFT on the same labels.
  [`docs/rl.md`](docs/rl.md) has the matrix and why.
- **Selecting hard items** (the ones v2.1 gets wrong) did worse on its own targets than random
  items from the same pool.
- **An option-joint head**, **general-knowledge coverage for MMLU-Pro** and **more Open-Jev** moved
  the suite a little and the held-out set the other way.
- **The failure-type data factory**, first two rounds: its probes moved, the suite did not. The
  data portfolio after it (bounded choices, wide label sets, abstention) found components worth
  keeping, but their stack stayed under the +0.006 bar.
- **0.8B:** coverage helped the suite by +0.02 and cost BFCL 0.08; repairs did not recover it.

### Every arm

| arm | axis | outcome | suite mean | final | MMLU-Pro | what it was |
|---|---|---|---|---|---|---|
| `arch-fix-1b-xattn-bilinear` | model | failed | — | — | — | bilinear option cross-attention combine; crashed before scoring |
| `ctl-08b-champ-v2` (+ `ctl-08b-champ-v2-mo160`) | control | control | 0.6782 | — | 0.2520 | 0.8B champion on the v2 suite (reference for s08b-3) |
| `s08b-3-proc-only` (+ `s08b-3-proc-only-mo160`) | model | kept (0.8B line) | 0.6919 / 0.6912 | — | 0.2650 / 0.2600 | 0.8B + procedural train split: +0.014 vs its reference, confirmed s97 |
| `arch-fix-1a-xattn-mlp` (+ `arch-fix-1a-xattn-mlp-mo160`) | model | kept | 0.7333 | — | 0.3640 | xattn_combine=mlp: +0.002, all 5 OOD targets up ; became part of v3.0 |
| `rlcd-1a-laya-soft` (+ `rlcd-1a-laya-soft-mo160`) | RL | rejected | 0.7308 | — | 0.3600 | RLCD reproduction, soft Laya reward (vs rlcd-1c); calibration-only effect |
| `data-scale-1-baseline-v2` (+ `data-scale-1-baseline-v2-mo160`) | control | control | 0.7313 | — | 0.3630 | v2.1 recipe rerun on the 15-benchmark suite (0.7313: the base of the first batch) |
| `arith-1-verif-6k` (+ `arith-1-verif-6k-mo160`) | data | rejected | 0.7315 | — | 0.3440 | 6k verification/arithmetic items: net 0 (scienthoon +.026, mmlu -.019) |
| `rlcd-1c-ctl-ce` (+ `rlcd-1c-ctl-ce-mo160`) | control | control | 0.7281 | — | 0.3580 | CE control for rlcd-1a/1b |
| `rlcd-1b-outcome` (+ `rlcd-1b-outcome-mo160`) | RL | rejected | 0.7294 | — | 0.3630 | RLCD reproduction, outcome reward (vs rlcd-1c) |
| `rl-weak-1-ctl-sft` (+ `rl-weak-1-ctl-sft-mo160`) | control | control | 0.7302 | — | 0.3690 | SFT on the 4.5k weak pool (control for rl-weak-1-rl) |
| `rl-weak-1-rl` (+ `rl-weak-1-rl-mo160`) | RL | rejected | 0.7295 | — | 0.3600 | RL on the weak pool: -0.001 vs its SFT control |
| `data-scale-1-a-cap6k` (+ `data-scale-1-a-cap6k-mo160`) | data | rejected | 0.7323 | — | 0.3470 | train-split caps 3k -> 6k |
| `data-scale-1-b-cap6k-ts30k-oj10k` (+ `data-scale-1-b-cap6k-ts30k-oj10k-mo160`) | data | improved, not kept | 0.7453 | — | 0.3690 | + tasksource 30k + open_jev 10k |
| `arch-fix-1b2-xattn-bilinear-norm` (+ `arch-fix-1b2-xattn-bilinear-norm-mo160`) | model | rejected | 0.7317 | — | 0.3620 | bilinear combine with norm: +0.000, dominated by the mlp combine |
| `data-scale-1-c-plus-cov1k` (+ `data-scale-1-c-plus-cov1k-mo160`) | data | kept | 0.7456 / 0.7485 | — | 0.3560 / 0.3580 | + 10 coverage sources x 1k = D*; the base of v3.0 (suite 0.7456 / 0.7485 on s17/s97) |
| `smoke-v2-eval` | tooling | smoke | 0.4327 | — | 0.1610 | 20-step evaluator smoke |
| `bucket-1-lb64-mo160` | tooling | adopted | 0.7322 | — | 0.3640 | length_bucket=64 (cheaper training, suite-neutral); used by every later arm |
| `ctl-08b-v3` | control | control | 0.6834 | 0.5766 | 0.2720 | 0.8B reference on the new evaluator |
| `s08b-4` | model | rejected (0.8B) | 0.7047 / 0.7089 | 0.5867 / 0.5688 | 0.2550 / 0.2490 | 0.8B + D* sources: suite +.021/+.026, bfcl -.080/-.111 (not a release candidate while BFCL is down) |
| `arch-2-joint` | model | rejected | 0.7508 | 0.6338 | 0.3660 | option-joint head (2-layer transformer over options): +.001 vs stack-1, final -.011 |
| `d-star-ref` | control | control | 0.7476 | 0.6399 | 0.3540 | D* rerun on the new evaluator: the reference for the arms after it |
| `stack-1-xmlp` | model | kept | 0.7498 | 0.6447 | 0.3620 | D* + xattn mlp: +.0022 suite, +.0048 final (2nd measurement) |
| `mmlu-cov-1` | data | rejected | 0.7449 | 0.6295 | 0.3670 | 20k general-knowledge MC recast to 10 options: mmlu +.013, suite -.0027, final -.0104 |
| `mine-1-ctl` | data | subsumed | 0.7510 | 0.6426 | 0.3420 | random tasksource/open_jev adds from the mine_1 pool; subsumed by data-scale-2 |
| `mine-1` | data | rejected | 0.7526 | 0.6362 | 0.3800 | on-policy hard mining (items v2.1 gets wrong): +.0016 vs its control, final -.0064 |
| `fac-r1` | data | rejected | 0.7457 | 0.6410 | 0.3430 | factory round 1 (5 failure types, 30k q): suite -.0019 vs D*; probes in-template |
| `data-scale-2` | data | kept | 0.7586 / 0.7605 | 0.6474 / 0.6442 | 0.3690 / 0.3670 | +84k q (jsg emotion/sst5/anli, tasksource 30k, open_jev 10k): +.0110/+.0129 vs D*, confirmed |
| `long-state-8k` (+ `long-state-8k-r2`, `long-state-8k-r3`) | model | use-case (not kept) | 0.7481 | 0.6490 | 0.3620 | 8k context training on long_state (r2 OOM, r3 on H200): suite +.0005, final +.009 |
| `v3-stack-ds2-xmlp` | stack | kept (v3.0 SFT parent) | 0.7589 | 0.6512 | 0.3730 | data-scale-2 corpus + xattn mlp: suite .7589, final .6512 |
| `s08b-5-rep` | model | rejected (0.8B) | 0.7015 | 0.5707 | 0.2570 | near-miss repair of 0.8B nimble/bfcl: nimble repaired, bfcl not |
| `v3-ds2-rep-a-nojsg` | data | rejected | 0.7545 | 0.6410 | 0.3540 | ds2 minus jsg: jev_style gain lost |
| `v3-ds2-rep-b-jsg3k` | data | preferred variant | 0.7603 | 0.6482 | 0.3680 | ds2 with jsg capped 3k: +.0014 vs stack, ECE -.046 (needed a second seed) |
| `v3-data-scale-3` | data | rejected | 0.7605 | 0.6445 | 0.3640 | +15.9k open_jev: +.0016 vs stack, final -.0067 |
| `fac-r2` | data | held | 0.7564 | 0.6431 | 0.3530 | factory r2 (3 types x 3k): suite -.0025, probes in-template |
| `v4-rep-b-s97` | data | confirmation | 0.7559 | 0.6490 | 0.3550 | rep-b fresh seed 97: .7559 |
| `v4-rep-b-s17ck` | data | checkpoint | 0.7582 | 0.6434 | 0.3530 | rep-b s17 rerun with --save-ckpt: .7582 (parent of the data-portfolio arms) |
| `v4-08b-bfcl-anchor` | model | rejected (0.8B) | 0.6963 | 0.5523 | 0.2320 | 0.8B bfcl repair (devtools x2 anchor) |
| `v4-08b-bfcl-drop` | model | rejected (0.8B) | 0.7011 | 0.5726 | 0.2530 | 0.8B bfcl repair (drop cov_massive_multi) |
| `dp-ct-smoke-load0` | tooling | smoke | 0.7589 | 0.6512 | 0.3730 | continued-training load check: 0 flips, exact |
| `dp-ct-smoke-20` | tooling | smoke | 0.7527 | 0.6402 | 0.3650 | continued-training 20-step smoke |
| `rl2-smoke-select-cal` | tooling | smoke | 0.7271 | 0.6700 | 0.3000 | RL-loop GPU smoke |
| `rl2p-b-k0` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-c-k01` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-b-k01` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-afull` | control | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-c-k0` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-d-l01` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-d-l03` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-a-replay` | control | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-d-l03-k01` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-d-l1` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-e-l01` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-e-l03-k01` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-e-l03` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-e-l1` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-eg-l03` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-eg-l1-k01` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-eg-l1` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-eg-l3` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-eprime` | control | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `dp-r1-gk` | data | rejected | 0.7596 | 0.6410 | 0.3740 | portfolio r1: MMLU-Pro aspect sources (vs dp-r1-ctl) |
| `dp-r1-bnd` | data | kept as component | 0.7577 | 0.6397 | 0.3630 | portfolio r1: bounded choice / tier sources: probe pool +.050 |
| `dp-r1-ctl` | control | control | 0.7601 | 0.6381 | 0.3630 | portfolio r1 replay-only control |
| `dp-r1-kev` | data | rejected | 0.7578 | 0.6506 | 0.3530 | portfolio r1: kev rules/arithmetic |
| `rl2m-a-replay` | control | control | 0.7562 | 0.6522 | 0.3580 | RLCD matrix A: replay only |
| `rl2m-afull` | control | control | 0.7566 | 0.6482 | 0.3670 | RLCD matrix A-full: CE on full labels |
| `rl2m-b-binary` | RL | rejected | 0.7509 | 0.6525 | 0.3510 | RLCD matrix B: binary reward |
| `rl2m-c-proper` | RL | rejected | 0.5861 | 0.5245 | 0.2690 | C: proper-score reward; diverged (suite .586) |
| `dp-r1-ts` | data | rejected | 0.7573 | 0.6418 | 0.3590 | portfolio r1: tasksource noise |
| `cr-r1-noce` (+ `cr2-r1-noce`) | RL | pilot | — | — | — | wave-2 decision-objective pilot (confrank / select), dev metrics only; judged NOT MET |
| `rl2m-d-rlcr` | RL | rejected | 0.7565 | 0.6472 | 0.3730 | D: RLCR |
| `cr-r10` (+ `cr2-r10`) | RL | pilot | — | — | — | wave-2 decision-objective pilot (confrank / select), dev metrics only; judged NOT MET |
| `cr-r1` (+ `cr2-r1`) | RL | pilot | — | — | — | wave-2 decision-objective pilot (confrank / select), dev metrics only; judged NOT MET |
| `rl2m-e-bandit` | RL | rejected | 0.7487 | 0.6546 | 0.3630 | E: bandit RLCD; killed vs E' |
| `cr-r3` (+ `cr2-r3`) | RL | pilot | — | — | — | wave-2 decision-objective pilot (confrank / select), dev metrics only; judged NOT MET |
| `rl2m-eg-bandit-grad` | RL | rejected | 0.7535 | 0.6490 | 0.3620 | E-grad; killed vs E' |
| `rl2m-eprime-bandit-sup` | control | control | 0.7545 | 0.6490 | 0.3720 | E': supervised use of the same bandit outcome |
| `rl2p-x-gauss005` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-x-gauss01` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `rl2p-x-gauss02` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `dp-r2-bnd-noomp` | data | kept as component | 0.7588 | 0.6410 | 0.3820 | portfolio r2: bnd without omp sources |
| `dp-r2-bnd` | data | kept as component | 0.7569 | 0.6431 | 0.3680 | portfolio r2: bnd at protected replay dose |
| `long-ctx-32k` | model | report-only | 0.7553 | 0.6437 | 0.3580 | 32k-context arm |
| `rl2p-x-temp15` | RL | pilot | — | — | — | RLCD matrix pilot (300 steps, dev metrics only) |
| `sel-t98` | RL | pilot | — | — | — | wave-2 decision-objective pilot (confrank / select), dev metrics only; judged NOT MET |
| `dp-r2-ctl` | control | control | 0.7570 | 0.6415 | 0.3520 | portfolio r2 control |
| `sel-t9` | RL | pilot | — | — | — | wave-2 decision-objective pilot (confrank / select), dev metrics only; judged NOT MET |
| `vis-ct-1-ctl` | control | control | 0.7576 | 0.6434 | 0.3760 | replay-only control for vis-ct-1 |
| `vis-ct-1` | vision | kept (vision candidate) | 0.7609 | 0.6480 | 0.3630 | first image arm: image eval .809 vs ctl .796, suite .761 |
| `dp-r3-ctl` | control | control | 0.7590 | 0.6407 | 0.3570 | portfolio r3/r4 control |
| `dp-r3-sci` | data | rejected | 0.7579 | 0.6453 | 0.3600 | sci yes/no sources: probe -.028 |
| `dp-r3-wide` | data | partly kept | 0.7541 | 0.6453 | 0.3620 | 90-130-option label sets: probe +.080 |
| `dp-r3-inj` | data | rejected | 0.7566 | 0.6439 | 0.3650 | prompt-injection sources: probe -.049 |
| `dp-r4-rebal` | data | rejected as source | 0.7559 | 0.6455 | 0.3650 | escalation rebalance: esc -.014, suite -.003 |
| `rl3-pointwise` | control | control | 0.7544 | 0.6528 | 0.3590 | pointwise CE control |
| `rl3-packet` | RL | rejected | 0.7527 | 0.6514 | 0.3530 | packet exact-split RL: loses to CE -.047 |
| `rl3-packet-ce` | control | control | 0.7526 | 0.6509 | 0.3540 | packet CE control |
| `rl3-listwise` | RL | not a win | 0.7537 | 0.6496 | 0.3540 | listwise NDCG RL, KL gate on: +.014 vs pointwise (z 1.3) |
| `s2-cr-noce` | RL | pilot | — | — | — | wave-2 decision-objective pilot (confrank / select), dev metrics only; judged NOT MET |
| `s2-afull` | control | pilot | — | — | — | wave-2 decision-objective pilot (confrank / select), dev metrics only; judged NOT MET |
| `s2-cr-r1` | RL | pilot | — | — | — | wave-2 decision-objective pilot (confrank / select), dev metrics only; judged NOT MET |
| `s2-a-replay` | control | pilot | — | — | — | wave-2 decision-objective pilot (confrank / select), dev metrics only; judged NOT MET |
| `lt-16` | model | report-only | 0.7209 | 0.6210 | 0.3400 | scorer tap at layer 16 (cost sweep) |
| `lt-18` | model | report-only | 0.7200 | 0.6255 | 0.3410 | tap 18 |
| `lt-20` | model | report-only | 0.7194 | 0.6226 | 0.3380 | tap 20 |
| `lt-23` | model | report-only | 0.7221 | 0.6258 | 0.3520 | tap 23 (current) |
| `dp-r4-abst` | data | kept as component | 0.7594 | 0.6410 | 0.3660 | abstention sources: KoBBQ unknown .448 -> .918, disambiguated .803 -> .636 |
| `rl4-packet-ce` | control | control | 0.7547 | 0.6455 | 0.3560 | packet CE control |
| `rl4-pointwise` | control | control | 0.7536 | 0.6541 | 0.3570 | pointwise CE control |
| `rl4-packet` | RL | rejected | 0.7532 | 0.6455 | 0.3620 | packet RL gate off: -.010 vs CE |
| `rl4-listwise` | RL | kept | 0.7546 | 0.6509 | 0.3560 | listwise RL, gate off: hippo pc R@1 .246 vs .194 (+.052, z 3.9) |
| `vis-ct-2-ctl` | control | control | 0.7529 | 0.6389 | 0.3720 | control for vis-ct-2 |
| `vis-ct-2` | vision | rejected | 0.7539 | 0.6485 | 0.3520 | second image portfolio: image eval .8026 vs vis-ct-1 .8088 |
| `r3-16-nr` | model | report-only | 0.7214 | 0.6151 | 0.3420 | tap 16, no LM retention |
| `r3-23-nr` | model | report-only | 0.7221 | 0.6258 | 0.3520 | tap 23, no retention |
| `r3-16-ret` | model | report-only | 0.7201 | 0.6186 | 0.3310 | tap 16 + retention KL .5 |
| `r3-23-ret` | model | report-only | 0.7192 | 0.6268 | 0.3270 | tap 23 + retention KL .5 |
| `dp-r5-stack-ctl` | control | control | 0.7586 | 0.6402 | 0.3690 | control for dp-r5-stack |
| `dp-r5-stack` | stack | no KEEP (product candidate) | 0.7622 | 0.6450 | 0.3770 | stack of every kept dp component: +.0036 vs ctl, below the +.006 bar; KoBBQ gate fails |
| `rl5-listwise-rcC` | RL | kept = v3.0 | 0.7561 | 0.6490 | 0.3640 | rl4 recipe on the v3.0 SFT parent: hippo pc R@1 .192 -> .308 |

## jevtr_v1, continued — from v3.0 to v4.0-VL (2026-09-28/30)

Same loop and evaluator, now with images. Counted as before, one per run record and a rerun
once: **108 arms** ran between v3.0 and v4.0-VL. The 11 below are the ones on the released line:
the stages v4.0-VL was built from, with their controls and rivals, in the order they started.
`vis-ct-1` and `vis-ct-2` are in the v3.0 table above. The other 97 will be written up with the
work they belong to.

- **Images train like text.** Each image round held the suite against its replay-only control
  and moved its own held-out probes: round 3's generators took theirs from 0.613 to 0.854.
- **Stacked continued training made the model overconfident.** Raw held-out ECE rose from
  v3.0's 0.154 to 0.25–0.29 over the image rounds, and the confidence head removed only part of it.
- **Soft targets on questions it cannot answer repaired most of that** (`calA-ce`); the same
  questions with true answers taught BIG-Bench Hard's task families instead (`calB-rl`).
- **An asymmetric confidence penalty calibrated in the weights**: raw held-out ECE 0.200 →
  0.082, against 0.121 for the matched supervised stage. One seed.
  [`docs/rl.md`](docs/rl.md#v40-vl-a-reward-that-prices-confident-mistakes) has the reward and
  where it did not help.

| arm | axis | outcome | suite mean | final | MMLU-Pro | what it was |
|---|---|---|---|---|---|---|
| `rl5-listwise-vis` | RL | kept (parent of round 3) | 0.7601 | 0.6423 | 0.3860 | v3.0's listwise reranking stage on `vis-ct-1`: hippo R@1 .220; image eval .807 (`vis-ct-1` .809) |
| `vis-ct-3` | vision | kept | 0.7577 | 0.6447 | 0.3750 | image round 3: six synthetic generators + four vision_v2 sources, 51% text replay: held-out-template probes .854 vs .613 for its control; image eval .804 |
| `vis-ct-3b` | vision | rejected | 0.7574 | 0.6445 | 0.3650 | `vis-ct-3` with 2,048 image tokens for document and chart sources: probes .856, no resolution effect |
| `vis-ct-3-ctl` | control | control | 0.7537 | 0.6431 | 0.3580 | replay-only control for `vis-ct-3` (image eval .806) |
| `vis-rl-2-pw` | control | control | 0.7532 | 0.6373 | 0.3720 | pointwise CE control for `vis-rl-2` |
| `vis-rl-2` | RL | rejected | 0.7543 | 0.6341 | 0.3710 | listwise RL over 16 images per query: retrieval NDCG@5 .980, identical to its pointwise control |
| `calA-ce` | data | kept | 0.7555 | 0.6469 | 0.3750 | 8,492 generated reasoning questions with soft targets where `vis-ct-3` is at chance: raw final ECE .200 vs .316 for its control, .075 after calibration |
| `calB-rl` | RL | not a release candidate | 0.7551 | 0.6875 | 0.3660 | bandit RL on the same questions with true answers: BBH web_of_lies .52 → .91; without BBH, lower accuracy than its parent and ECE .087 after calibration |
| `calC-ctl` | control | control | 0.7537 | 0.6394 | 0.3630 | replay-only control for `calA-ce` and `calB-rl` |
| `asym-calA` | RL | **kept = v4.0-VL** | 0.7564 | 0.6528 | 0.3850 | asymmetric RL stage on `calA-ce` (a confident mistake costs 4× a timid correct answer), 30% image rows: raw final ECE .200 → .082 |
| `ce-calA` | control | control | 0.7578 | 0.6439 | 0.3750 | the matched supervised stage for `asym-calA`: same data, steps and seed; raw final ECE .121. The release gate job read final .6442, as in the card |

## jevtr_v1, continued — from v4.0-VL to v5.0-VL (2026-09-29/10-02)

Same loop and evaluator, on a new base: Qwen3.5-4B, read at an early layer. Counted as before, one
per run record and a rerun once: **62 arms** ran between v4.0-VL and v5.0-VL. The ones below are
on the released line, in the order they started; suite is the weighted fifteen-benchmark mean
throughout. The rest will be written up with the work they
belong to.

- **A System One model doesn't need the deep layers.** Reading the 4B base at layer 20 of 32
  scores like reading it at 32 on the decision suite and the held-out set; only knowledge-heavy
  MMLU-Pro keeps rising past 20.
- **Images on the cut model work as they did on the 2B**, and cost it the "unknown" answer until
  the data asked for it: KoBBQ unknown-when-ambiguous fell from 0.828 to 0.679, and 2,000 text
  rows whose right answer is "unknown" brought it to 0.891.
- **A readout bug, found by an outside benchmark.** On long numbered option lists the 4B line
  picked the option after the right one, because each option's pooled vector began with tokens
  that had just read the previous option. Pooling only an option's own tokens fixed it without
  retraining (CLINC150 0.383 → 0.753).

| arm | axis | outcome | suite mean | final | MMLU-Pro | what it was |
|---|---|---|---|---|---|---|
| `b4-exit16` | depth | rejected | 0.7439 | 0.6576 | 0.403 | Qwen3.5-4B-Base read at layer 16 of 32, from the base, 3,000 steps; matched 2B control 0.7150 / 0.6381 / 0.330 |
| `b4-exit20` | depth | **kept** (stage 1) | 0.7599 | 0.6881 | 0.422 | the same at layer 20: the best quality per H100 millisecond |
| `b4-exit24` | depth | rejected | 0.7603 | 0.6881 | 0.441 | the same at layer 24: only MMLU-Pro moves |
| `vis-v4-ctl` | control | control | – | 0.6872 | 0.407 | replay-only control for `vis-v4`: KoBBQ unknown-when-ambiguous 0.828 |
| `vis-v4` | vision | rejected | 0.7655 | 0.6881 | – | images on `b4-exit20` (24k image + 24k text questions): image probes up across the board, KoBBQ unknown 0.828 → 0.679 |
| `vis-v4k` | vision | **kept** (stage 2) | 0.7670 | 0.6961 | 0.438 | `vis-v4` plus 2,000 "unknown"-answer text rows and more abstention images: every pre-registered line passed; KoBBQ 0.899 / 0.891 |
| `exit12`, `exit13`, `exit28`, `exit32` | depth | context | 0.6809 · 0.7116 · 0.7584 · 0.7609 | 0.5905 · 0.6343 · 0.6774 · 0.6859 | 0.291 · 0.338 · 0.415 · 0.457 | the depth curve completed: 20 scores like 32 on suite and held-out |
| `vis-v4k` + own-token readout | readout | **kept = v5.0-VL** | 0.7621 | 0.6915 | 0.429 | same weights, each option pooled over its own tokens, calibrator refitted: CLINC150 0.383 → 0.753, short lists unchanged |
