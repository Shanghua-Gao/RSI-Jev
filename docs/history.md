# How the releases were made

<p align="center">
  <img src="../assets/loop-social.gif" width="900" alt="The champion line climbs from v1.0 (0.622) to v2.0 (0.709), v2.1 (0.736) and v3.0 (0.756) on the 15-benchmark suite, one evaluator for every release; grey dots are the experiments tried in between that did not clear it. Below: propose, experiment, learn; at the gate most become published negatives and one becomes a release, which becomes the bar to clear">
</p>

**v6.0-VL 5B**, the current release, **picks its depth per question.** It reads a decision at
layers 16, 20 and 32 of Qwen3.5-4B-Base and answers at the first one that is confident enough.
Classification, intent and retrieval are done by layer 16; judgement tasks need about 20;
knowledge and multi-step reasoning keep improving to 32. On Decision Index 0.2.1 it scores [PLACEHOLDER: full index] as served
([PLACEHOLDER: rank claim]), and four effort levels run from
23 ms to 40 ms per request on one H200. [`versions/v6.0-vl.md`](../versions/v6.0-vl.md) is its record.

<details>
<summary>the six releases behind it</summary>

**v5.0-VL 3B stopped at layer 20.** It runs the first 20 of
Qwen3.5-4B-Base's 32 layers: for a model that decides rather than writes, the loop found that
reading at layer 20 scores like reading at 32 on the decision suite and the held-out set (0.760
vs 0.761, 0.688 vs 0.686); only knowledge-heavy MMLU-Pro keeps rising. On Decision Index 0.2.1
it scores 38.38, the best at 3B and under on the public board and ahead of eleven 4B-class
entries. It reads images as v4.0-VL does, reads each option from that option's own tokens, and
ships as 3.25B parameters with nothing fetched from the base model.
[`versions/v5.0-vl.md`](../versions/v5.0-vl.md) is its record.

**v4.0-VL reads images.** It takes one to four pictures with a request —
a photo, a screenshot, a scanned page — and answers the same typed questions about them. On five
image benchmarks held out from training it answered 80.3% correctly, and 45.8% with the images
blanked out. On text it kept v3.0's level (suite 0.756). It is v3.0's lineage with two rounds
of image training and a new RL stage that charges a confident mistake four times what it charges
a timid correct answer. That stage takes held-out ECE before calibration from 0.200 to 0.082,
one seed ([how it works](rl.md#v40-vl-a-reward-that-prices-confident-mistakes)).
[`versions/v4.0-vl.md`](../versions/v4.0-vl.md) is its record.

**v3.0** is **the first release where reinforcement learning works**. The loop found the reward
RL needs here: one that scores the *order* of many candidates, which no per-item label can
express. Trained with it, v3.0 ranks the right memory first **60% more often** than its own
supervised parent (hippo R@1 0.192 → 0.308, 64 questions won against 6), with the rest of the
suite held level. [`versions/v3.0.md`](../versions/v3.0.md) is its record;
[`docs/rl.md`](rl.md) is how the loop got there; it now covers 63 reward-trained arms.

**v2.1** is the first release better *everywhere it was checked*: ahead of v1.0 on three
benchmarks it never trained on, and it recovers the general knowledge v2.0 had traded away by
training the bottom third of the tower at a tenth of the learning rate.
[`versions/v2.1.md`](../versions/v2.1.md).

**v2.0** trains on the *train* splits of the decision benchmarks it is measured on — never
their test splits — and adds a small confidence head that rescales every answer without
changing which option wins. Much stronger on those benchmarks, better calibrated everywhere,
and no better than v1.0 off them: [`versions/v2.0.md`](../versions/v2.0.md).

**v1.0** is a Qwen3.5 Base tower fine-tuned end to end with a trained cross-attention scorer
on top, fitted to a teacher's full probability distribution over 6,977 synthetic documents
rather than to its labels. It had never seen any benchmark's train split, so its numbers are
zero-shot and remain the reference every later release is compared against.
[`versions/v1.0.md`](../versions/v1.0.md) is its record.

</details>

*ECE below is expected calibration error: bin the decisions by the confidence the model
reported, compare each bin with how often it was actually right, average the gaps. Lower is
better, 0 is perfect.*

Of the 469 arms run since v1.0 (the latest 95 counted by a slightly different rule), these are the ones that got from one release to the next, each
ruling something out. (Arms are now counted one per run record, a rerun counting once; v2.1's
"ninety-two" counted logged experiment ids, a log most v3.0 arms were never written to.)

| where it went | result | what it established |
|---|---|---|
| **v1.0** | typed-decisions **0.662** | the bar to clear, and it had never seen a benchmark's train split |
| train on the benchmark's own train split | typed-decisions 0.7794 | worth +0.12 — and it costs general knowledge, so it was **rejected** |
| the same + 15% general-knowledge replay | typed-decisions 0.796 | the replay pays that cost back. The first keeper |
| the other benchmarks' train splits too, 3k per source | suite 0.7049 | the lever is not specific to one benchmark |
| a per-question confidence head, forbidden to sharpen score questions | suite ECE 0.0736 | confidence that means something, and it never changes which option wins. Shipped |
| **v2.0** | suite **0.7056**, ECE **0.0546** | capability was v1.0's lever; **what it trains on, and calibrating it afterwards, is a second one** — and free at inference. But the gain was on the benchmarks it had trained for |
| the replay weighted toward science, half of it recast to ten options | MMLU-Pro 0.331 | more and better-shaped data barely moves general knowledge: +0.008 |
| **freeze** the tower's lower third instead | MMLU-Pro 0.339 | holding those layers still does recover some of it — and breaks three of the decision benchmarks, because they do need to adapt |
| the bottom 8 layers at **one tenth** the learning rate | MMLU-Pro 0.359 | it was never a data problem. Fitting decisions through the whole tower at one rate was overwriting what the knowledge sat in — frozen layers cannot adapt, slow ones can |
| **v2.1** | suite **0.7291**, MMLU-Pro **0.383** | and it travels: ahead of v1.0 on all three benchmarks it never trained on, where v2.0 was not |
| wider coverage (data-scale-2): +84k questions, emotion/SST-5/ANLI and more tasksource and Open-Jev train items | suite +0.011 / +0.013 on two seeds | coverage is still the lever that moves the suite |
| a two-layer MLP combining the option and cross-attention features | suite 0.7589 on the new 15-benchmark suite | the readout had a little left in it: +0.002 |
| a listwise reranking reward (NDCG@5 over 16 candidates) on top, against a KL anchor | hippo R@1 0.192 → 0.308, suite −0.003 | **the first RL stage kept**: a reward over the *order* of candidates carries what per-item labels cannot |
| **v3.0** | suite **0.756**, ECE **0.066** | **the first release where RL works**: +60% reranking R@1 over its supervised parent, the rest held level |
| image training on the train splits of twelve public datasets, with text replay | held-out image top-1 0.809 against 0.796 for a text-only control | the model reads images through the base model's frozen vision tower |
| generated reasoning questions with soft targets where the model is at chance | held-out ECE after calibration, BBH excluded, 0.096 → 0.075 | the image rounds had made it overconfident; this repairs most of it |
| an RL stage charging a confident mistake 4x a timid correct answer | held-out ECE before calibration 0.200 → 0.082 | against a matched supervised stage: better accuracy and calibration, no detectable ranking difference (one seed) |
| **v4.0-VL** | image top-1 **0.803** held out, suite **0.756**, ECE **0.043** | **reads images**, with text held at v3.0's level |
| the same recipe on Qwen3.5-4B, read at layer 16, 20 or 24 of its 32 | suite 0.744 · 0.760 · 0.760 | a System One model doesn't need the deep layers: quality flattens by layer 20 |
| images on the layer-20 model, plus questions whose right answer is "unknown" | KoBBQ unknown-when-ambiguous 0.679 → 0.891 | images cost the model its "unknown" answer until the data asked for it |
| pool each option over its own tokens, not the separator that follows the previous one | CLINC150 0.383 → 0.753 | a readout bug found by an outside benchmark; the same weights read correctly once each option is pooled over its own tokens |
| retrain only the decision head for that readout, tower frozen, 600 steps | final ECE after calibration 0.080 → 0.042, Decision Index 38.38 | the head had learned the old pooling; a short head-only stage recovers the calibration |
| **v5.0-VL** | MMLU-Pro **0.429**, Decision Index **38.38**, KoBBQ unknown **0.932** | **3B: the first 20 of 32 layers** |
| early-exit heads that read a detached copy of their layer, so no gradient from them reaches the trunk | MMLU-Pro 0.443 at layer 32 | retuning heads on a trunk trained with attached exits had not recovered depth: the trunk had lost it |
| 10,000 steps on a cleaned corpus instead of 40,000 on the old one | held-out +0.026, MMLU-Pro +0.043 | longer training overfits the suite; shorter and cleaner wins held out |
| one temperature per exit and an exit policy fitted on held-out proxies of the test mix | every release bar passes at 20.9 layers on average | policies tuned on in-distribution rows stop too early on hard questions |
| **v6.0-VL** | Decision Index **[PLACEHOLDER: full index]**, MMLU-Pro **0.440** | **5B: answers at layer 16, 20 or 32, whichever is confident first** |

The animation covers the cycle through v1.0 · [`EXPLORE.md`](../EXPLORE.md) has every arm through v3.0
and every arm on the v4.0-VL, v5.0-VL and v6.0-VL lines, with why each failed · [`versions/v1.0.md`](../versions/v1.0.md#10-how-it-got-here) has the trail before
v1.0

## Human in the loop

The loop runs its own experiments and retires its own champions, but it does not decide what is
worth measuring, and it does not notice on its own when a number is technically true and
practically misleading. People do that, and it has changed both the experiments this project
runs and the rules it judges them by. Two of the rules in
[`BENCHMARKS.md`](../BENCHMARKS.md#how-a-result-becomes-a-result) exist because someone pushed
back: the MMLU-Pro guard was widened rather than letting a near-miss be discarded, and the seed
requirement now scales with the effect size instead of spending four seeds on every difference.
v2.1 itself started as a refusal to drop an arm that had failed one guard.

Thank you to the people who have sent that feedback:

- **Shanghua Gao** · [@gasvn](https://github.com/gasvn)
- **Sufian** · [@SufianTA](https://github.com/SufianTA)
