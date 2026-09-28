# Where RL beat SFT, and where it didn't

*Written for release [v3.0](../versions/v3.0.md) · 2026-09-28*

We spent four waves trying to make reinforcement learning improve a one-pass decision model.
**On per-item decisions, plain supervised training on the same data matched or beat it every
time. RL won once:** a listwise reward for reranking, where the reward spans many separately
scored candidates. That one win is the RL stage in v3.0. This is what we tried, why it lost,
where it won, and what we think Jev's RLCD must be doing.

The model throughout is a Qwen3.5-2B tower with an option scorer: one forward pass, one
probability per option, no generated text.

## What we tried

Eleven RL setups, 59 reward-trained arms counting pilots and repeats, each setup judged against
a supervised (SFT) control trained on the same items for the same steps. **One setup kept:
listwise reranking**, the only one to clear its pre-registered bar.

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
| **listwise reranking RL** | **NDCG@5 over 16 candidates** | **R@1 +0.052 (z 3.9): the one win** |
| whole-packet splitting RL | 1 only if every page boundary is right | packet exact −0.047 |

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

**Still far from the goal:** reranking with no model at all scores 0.484 R@1 on the same
benchmark. v3.0 reranks better than any earlier release and still worse than leaving the order
alone.

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
continued-training run so far has eroded it a little, v3.0's RL stage included (an in-house
proxy moves from 0.42 to 0.48), so any sharpening objective has to be checked against it.
