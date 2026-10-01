# Where RL beat SFT, and where it didn't

*Written for release [v3.0](../versions/v3.0.md) · 2026-09-28. The v4.0-VL section was added on 2026-10-01.*

**RL works for this model when the reward says something the labels cannot.** A listwise reward
for reranking — NDCG over many separately scored candidates — beats supervised training on the
same rows (+0.052 R@1, z 3.9), and it is the RL stage in v3.0, where it lifts reranking R@1 by 60%
over v3.0's own supervised parent. Finding it took four waves: on per-item decisions, where a
label already says what the answer is, supervised training matched every reward we tried. This is
what we tried, why those lost, why this one wins, and what we think Jev's RLCD must be doing.

**v4.0-VL adds a second reward of that kind.** It charges a confident mistake four times what it
charges a timid correct answer, a cost that no training target can express. As v4.0-VL's last
stage, it brings held-out ECE before calibration from 0.200 to 0.082 (one seed).
[The v4.0-VL section](#v40-vl-a-reward-that-prices-confident-mistakes) has the numbers, including
where it did not help.

The model throughout is a Qwen3.5-2B tower with an option scorer: one forward pass, one
probability per option, no generated text.

## What we tried

Through v4.0-VL: fourteen RL setups and 63 reward-trained arms. Arms are counted one per run
record. A rerun counts once, pilots count, and controls do not. After v3.0, only arms on a released
line are counted. Each setup was judged against a supervised (SFT) control trained on the same items
for the same steps. **Two setups kept: listwise reranking (v3.0) and an asymmetric confidence
penalty (v4.0-VL).**

| attempt | reward | result against its SFT control |
|---|---|---|
| outcome RL in simulated workflows (4 versions) | episode success in multi-step scenarios | lost; hurt the 15-benchmark suite |
| calibration RL (RLCR-style) | correct + Brier on a sampled answer | lost |
| RL on weak benchmarks | correctness on the model's weakest sources | lost (suite −0.001) |
| first "RLCD" runs | soft agreement with the gold, plus Gaussian logit noise | within noise; below baseline |
| binary correctness RL | 1 if the sampled option is right | probabilities collapsed to 0/1 |
| proper-score RL (Laya-style) | log + spherical score | fine at 300 steps, diverged at 1,500 |
| RLCR | correct − λ·Brier | tied on accuracy, worse raw calibration |
| bandit RLCD | only the chosen option's outcome is revealed | no better than SFT on the same feedback |
| act/defer and confidence-ranking objectives | selective-decision utility; pairwise rank | AURC −0.002 (0.6 SE) |
| **listwise reranking RL** | **NDCG@5 over 16 candidates** | **R@1 +0.052 (z 3.9): kept in v3.0** |
| whole-packet splitting RL | 1 only if every page boundary is right | packet exact −0.047 |
| listwise RL over images (v4.0-VL line) | NDCG@5 over 16 images per query | tied: NDCG@5 0.980 for both |
| bandit RL on generated reasoning questions (v4.0-VL line) | correctness, on questions the model was at chance on | not kept: it learned the benchmark's task families, not calibration |
| **asymmetric confidence penalty** | **c − 4·max(0, p − c)² − max(0, c − p)²** | **held-out ECE before calibration 0.082 vs 0.121: the last stage of v4.0-VL** |

## Why RL loses when the answer is known

The model's output is already a full distribution over a handful of options. With a gold label
in hand, a reward built from that label adds no information: the expected policy-gradient update
equals the gradient of a supervised loss.

- **A proper-score reward has the gold distribution as its optimum**, the same target soft
  cross-entropy already trains toward. Sampling only adds variance.
- **With a KL anchor the RL optimum is closed-form**: p* ∝ p_ref · exp(r/β). When r is the SFT
  target, p* is the SFT policy.
- **A correctness reward on a sampled option pushes toward 0/1.** It rewards picking right, not
  reporting honest odds. That is exactly the collapse we saw.

So on labelled decisions an "RL arm" is a loss-design arm. Loss-level changes here have moved the
suite by at most 0.002. New data has moved it by 0.13.

## Reconstructing Jev's RLCD

TypeSafe describes Jev's training as RLCD, Reinforcement Learning for Calibrated Decisions, and
has not published how it works. We built one controlled matrix covering the public
reconstructions: the same checkpoint, a 13.8k-question slice and 1,500 steps for every arm.
Bandit RLCD (the gold hidden, only the chosen option's outcome revealed) was killed by its
pre-registered test: it had to beat SFT on the same bandit feedback on 2 of 4 calibration
metrics, and it won 1.

Each cell reads raw / after the post-hoc confidence head (cal-4b). AURC: lower is better.

| arm | suite top-1 | ECE | AURC | coverage at 98% precision |
|---|---|---|---|---|
| A: SFT continuation | 0.756 | 0.060 / 0.020 | 0.061 / 0.059 | 0.34 / 0.38 |
| A-full: SFT, full labels | 0.757 | 0.077 / 0.021 | 0.062 / 0.062 | 0.30 / 0.33 |
| B: binary correctness RL | 0.751 | 0.143 / 0.029 | 0.063 / 0.061 | 0.36 / 0.36 |
| C: proper-score RL (Laya-style) | 0.586 | 0.371 / 0.364 | 0.369 / 0.331 | 0 / 0 |
| D: RLCR | 0.757 | 0.129 / 0.029 | 0.061 / 0.061 | 0.38 / 0.37 |
| E: bandit RLCD | 0.749 | 0.140 / 0.027 | 0.063 / 0.062 | 0.35 / 0.37 |
| E′: SFT on the same bandit feedback | 0.755 | 0.073 / 0.023 | 0.062 / 0.061 | 0.33 / 0.35 |

- **Every RL reward made raw calibration worse.** The confidence head brought every arm except C
  back to ECE 0.02–0.03, so RL's calibration story disappears after an inexpensive post-hoc fit.
- **The bandit reward collapses unless its calibration term is differentiated.** In a toy check
  on one state with gold 0.6 / 0.3 / 0.1, it went to 1 / 0 / 0 like binary RL.
- **No objective improved the ordering of confidence.** AURC stayed at the parent's level. Plain
  SFT continuation was best after calibration.
- **Exploration changes did not rescue it.** Gaussian logit noise diverged at σ 0.05 and 0.1 and
  lost accuracy at 0.2; temperature 1.5 collapsed like plain sampling.

## Rewards labels can't express

If RL needs information the labels lack, the honest test is a reward that no per-item label
contains. We tried two, each against SFT on exactly the same rows. The first run had a KL gate
that silently skipped 49–83% of steps, so we reran with the gate off and a KL penalty only. That
rerun is below.

| model | hippo per-candidate R@1 | R@5 | packets split exactly | suite |
|---|---|---|---|---|
| parent | 0.140 | 0.494 | 0.22 | 0.756 |
| **listwise RL (NDCG@5)** | **0.246** | **0.614** | 0.20 | 0.755 |
| pointwise SFT, same rows | 0.194 | 0.574 | 0.29 | 0.754 |
| whole-packet RL | 0.148 | 0.468 | 0.96 | 0.753 |
| packet SFT, same rows | 0.138 | 0.480 | 0.97 | 0.755 |

- **Reranking: RL wins.** Paired on the same 500 hippo-memory questions, listwise RL beats SFT by
  +0.052 R@1 (36 wins, 10 losses, z 3.9) and +0.040 R@5, with the suite unchanged. With the gate
  on, the gap had shrunk to +0.014.
- **Why this one works:** per-item cross-entropy scores each candidate on its own. NDCG rewards
  the order across candidates, which no single label encodes.
- **Document splitting: no win.** Both arms reach 0.96–0.97 exact on the data alone, so the probe
  no longer separates them.
- **Side lesson:** a KL gate tight enough to protect the parent (0.05) disables RL whenever the
  data is a new row type, since learning it moves those rows away from the parent.

**How far this evidence reaches.** That pair is one seed, and it was run on a different parent
from v3.0's. v3.0 applies the same recipe to its own supervised parent without a matched
pointwise control, so for v3.0 itself the comparison is RL against its parent, not against SFT
on the same rows: hippo per-candidate R@1 0.192 → **0.308**, R@5 0.552 → 0.626 (64 questions
won, 6 lost, exact McNemar *p* = 2.4 × 10⁻¹³), at a suite cost of −0.0028 and −0.009 on MMLU-Pro.

**Where it goes next:** hippo's own retrieval order scores 0.484 R@1 on the same benchmark.
v3.0 is the best reranker in this line so far, and closing that gap is the next target.

## v4.0-VL: a reward that prices confident mistakes

Four reward-trained arms ran on the line v4.0-VL was built from. In the order they ran:

| arm | reward | result |
|---|---|---|
| `rl5-listwise-vis` | v3.0's listwise NDCG@5 stage, on the first image model | kept as a stage. No matched control: hippo R@1 0.220, held-out image top-1 0.807 against 0.809 before it |
| `vis-rl-2` | NDCG@5 over 16 images per caption | tied with SFT on the same rows: retrieval NDCG@5 0.980 for both (paired difference 0.000, SE 0.004) |
| `calB-rl` | correctness, on 8,492 generated reasoning questions where the model was at chance | not kept: it learned the reasoning families, not calibration |
| **`asym-calA`** | **correctness minus an asymmetric confidence penalty** | **kept: the last stage of v4.0-VL** |

- **Listwise RL did not raise reranking on the image model.** Over images the reward tied plain
  SFT on the same rows: retrieval NDCG@5 0.980 for both.
- **`calB-rl` is a data result, not RL against SFT.** Its supervised rival, `calA-ce`, trained the
  same questions toward soft targets ("you are at chance here"); `calB-rl` was rewarded with the
  true answers. It learned the task: BIG-Bench Hard's web_of_lies went from 0.52 to 0.91. Those
  generated questions share BIG-Bench Hard's task families, so that is not held out. Without
  BIG-Bench Hard, held-out accuracy was 0.6945 against `calA-ce`'s 0.6983, and ECE after
  calibration 0.087 against 0.075. `calA-ce` became the parent of the last stage.

### The asymmetric stage

The image rounds made the model overconfident. Raw held-out ECE rose from v3.0's 0.154 to
0.25–0.29, and `calA-ce`'s soft targets brought it back to 0.200. The last stage keeps `calA-ce`'s
data and changes only the objective. For the option the model picks, with probability `p`, and
`c` = 1 if it is right and 0 if not:

```
R = c − 4·max(0, p − c)² − 1·max(0, c − p)²
```

- **A wrong answer at `p` costs 4p². A right answer at `p` costs (1 − p)².** Being 0.9 sure and
  wrong costs 3.24; being 0.1 sure and right costs 0.81.
- **Why asymmetric.** A training target says what the probability should be. It cannot say that one
  kind of error costs more than the other, so cross-entropy treats too sure and too unsure alike.
  The reward can price them differently.
- **What it aims at.** If an option is right with probability `q`, the reward is highest at
  `p = q / (q + 4(1 − q))`. That is below `q`, and furthest below where the model is often wrong.
  The confidence head then rescales the level without changing which option wins.

**What it did.** 1,500 steps, 30% of rows with images, one seed. Held-out ECE before any
calibration fell from 0.200 to 0.082. The matched supervised stage, with the same data, steps and
seed and only the objective changed, reached 0.121. After the same calibrator selection for both:

| both after the same calibrator selection | asym stage (v4.0-VL) | matched supervised stage |
|---|---|---|
| held-out, pooled top-1 | **0.6528** | 0.6442 |
| held-out without BIG-Bench Hard, top-1 | **0.703** | 0.694 |
| 15-benchmark suite, top-1 | 0.7564 | **0.7578** |
| held-out ECE | **0.0822** | 0.0871 |
| suite ECE | **0.043** | 0.055 |
| MMLU-Pro 1k | **0.385** | 0.375 |
| images, held-out top-1 | 0.803 | 0.804 |
| held-out without BIG-Bench Hard, AUROC | 0.6935 | 0.7017 |

Better held-out accuracy and calibration, and no detectable ranking difference: the AUROC gap is
−0.008 with a paired bootstrap SE of 0.006.

**Where it did not help.**

- **Ranking.** The penalty moves confidence levels. It did not sort right answers above wrong ones
  any better than the supervised stage.
- **The suite.** The matched supervised stage is slightly higher, 0.7578 against 0.7564. Images
  are level.
- **After calibration the lead is small.** Held-out ECE is 0.0822 against 0.0871 once both have
  their confidence head.
- **One seed**, for the stage and for its comparison.

**Why this one fits the rule.** The reward adds no information: correctness is the same label
cross-entropy uses. It adds a cost, that a confident mistake is worse than a timid correct answer,
and no target distribution can carry that. On the pure-chance check from v3.0, v4.0-VL puts
0.447 on the option it picks, against v3.0's 0.480.

## What moved the model instead

Data did. The 15-benchmark suite went from 0.61 (v1.0) to 0.76, almost entirely through what the
model trains on: benchmark train splits, wider source coverage, and a slower learning rate on the
lower layers. An audit of 680 wrong answers explains why a new loss has little to work with:

| cause of error | share |
|---|---|
| lacks the knowledge | 39% |
| question genuinely ambiguous | 35% |
| prior bias (e.g. escalating to the higher option) | 13% |
| gold label wrong | 11% |
| input truncated | 2% |

None of these is fixed by reweighting the same labels. Coverage, abstention data and rebalanced
label priors are.

The one RL that won here before v3.0 acted on the data, not the model: a bandit choosing the
training mix beat the hand-set mix by 0.004 on 4 of 4 seeds. The research loop now runs the same
idea at a larger scale: it generates data for the case types the model fails, keeps a type if its
held-out probe moves, drops it after rounds that stay flat, and allocates by yield.

## What Jev's RLCD must contain

If RLCD works for Jev, our results say it needs at least one of three ingredients we did not have.

1. **Feedback the labels don't contain.** Real delayed outcomes, or states nobody can label per
   step. Wherever a label or its bandit shadow exists, SFT plus a post-hoc confidence head already
   matched every RL objective we tried.
2. **A separate confidence channel with its own gradient.** A calibration term on the decision
   itself collapsed like binary RL; it only survived when differentiated directly.
3. **An effect on ordering, not only calibration.** Jev's value shows in which answers it is
   confident about (coverage at high precision). None of our objectives moved that.

Where we would still look:

- **Real agent loops**, where success shows up several steps later: tool choice, retry, escalate,
  stop. Our simulated versions were too simple.
- **Consistency across forms that labels don't tie together.** P(q) + P(not q) should be 1. On
  an external probe v2.1 sums to about 0.65 where Jev sums to about 0.98.

One property worth keeping while we try: on pure-chance questions v2.1 keeps its uncertainty
(a fair die: 0.36 on the face it picks, against Jev's 0.83 and a truth of 0.17). Every
continued-training run so far has moved it a little, v3.0's RL stage included (an in-house
proxy moves from 0.42 to 0.48), so every sharpening objective is now checked against it.
