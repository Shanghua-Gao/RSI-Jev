# How the releases were made

<p align="center">
  <img src="../assets/loop-social.gif" width="900" alt="The champion line climbs from v1.0 (0.622) to v2.0 (0.709), v2.1 (0.736) and v3.0 (0.756) on the 15-benchmark suite, one evaluator for every release; grey dots are the experiments tried in between that did not clear it. Below: propose, experiment, learn; at the gate most become published negatives and one becomes a release, which becomes the bar to clear">
</p>

**v6.1-VL**, the current release, ships at 4B and 27B. **At 4B it is v6.0-VL averaged with a
second fine-tune** of the same base, trained on other data. Nothing was trained after the average; the loop refit each
exit's temperature and chose the exit thresholds again. The gain is in the averaged tower. At
layer 32 it leads v6.0-VL on the suite (0.793 vs 0.770) and the held-out set (0.728 vs 0.695),
and it scores 50.98 on the public Decision Index 0.3 (v6.0-VL 46.23). Calibration is worse than v6.0-VL's (final ECE 0.048 vs 0.036).
[`versions/v6.1-vl.md`](../versions/v6.1-vl.md) is its record.

**v6.1-VL 27B** (2026-10-08) brings the same method to Qwen3.8-27B: two LoRA fine-tunes of one
parent, averaged, with answer heads at layers 48, 56 and 64. The average was chosen from 9 full
Decision Index 0.3 reads of candidate trunks. On the release package it scores
65.46 on the Decision Index 0.3 public set at the default effort and 65.64 with `auto`; the official
score is pending. At layer 64 it reads 0.843 on the suite and 0.808 on the held-out set (v6.1-VL
4B: 0.793 and 0.728); final ECE is 0.064 (4B: 0.048). It runs on one 96 GB GPU.
[`versions/v6.1-vl-27b.md`](../versions/v6.1-vl-27b.md) is its record.

<details>
<summary>the seven releases behind it</summary>

**v6.0-VL 4B picked its depth per question.** It read a decision at layers 16, 20 and 32 of
Qwen3.5-4B-Base and answered at the first one that was confident enough. Classification, intent
and retrieval were done by layer 16; judgement tasks needed about 20; knowledge and multi-step
reasoning kept improving to 32. On Decision Index 0.2.1 it scored 46.24 as served, the highest on
the public board of 2026-09-28 among 4B models and anything smaller, and its four effort levels
ran from 23 ms to 40 ms per request on one H200. [`versions/v6.0-vl.md`](../versions/v6.0-vl.md)
is its record.

**v5.0-VL 3B stopped at layer 20.** It ran the first 20 of
Qwen3.5-4B-Base's 32 layers: for a model that decides rather than writes, the loop found that
reading at layer 20 scored like reading at 32 on the decision suite and the held-out set (0.760
vs 0.761, 0.688 vs 0.686); only knowledge-heavy MMLU-Pro kept rising. On Decision Index 0.2.1
it scored 38.38, the best at 3B and under on the public board at the time and ahead of eleven
4B-class entries. It read images as v4.0-VL does, read each option from that option's own tokens,
and shipped as 3.25B parameters with nothing fetched from the base model.
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

Of the 496 arms run since v1.0 (the latest 122 counted by a slightly different rule), these are the ones that got from one release to the next, each
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
| **v6.0-VL** | Decision Index **46.24**, MMLU-Pro **0.440** | **4B: answers at layer 16, 20 or 32, whichever is confident first** |
| v6.0-VL averaged, weight 0.5 each, with a second fine-tune of the same base trained on other data | suite 0.793 and held-out 0.728 at layer 32, against 0.770 and 0.695 | the average beats both members; the gain is in the tower, the heads add nothing |
| one temperature per exit and both thresholds chosen again for the average | `auto` at 0.85 / 0.50 matches layer 32 at 19.5 layers | averaging shrinks the logits: at v6.0-VL's temperatures it was underconfident; calibration ends worse than v6.0-VL's (final ECE 0.048 vs 0.036) |
| **v6.1-VL** | Decision Index 0.3 **50.98**, held-out **0.729** | **4B: v6.0-VL averaged with a second fine-tune** |
| two LoRA fine-tunes of Qwen3.8-27B, averaged weight 0.5 each, chosen from 9 full Decision Index 0.3 reads | Decision Index 0.3 65.59, against 64.83 and 65.30 for the two members | averaging carries over to 27B; uneven weights and a third ingredient read 65.37 to 65.58 |
| short inputs right-padded to a multiple of 64 tokens and replayed as CUDA graphs | short one-question requests in about half the time | at one request a time the 27B is bound by kernel launches, not compute |
| **v6.1-VL 27B** | Decision Index 0.3 public set **65.46**, held-out **0.808** | **27B: the same method on Qwen3.8-27B, answers at layer 48, 56 or 64** |

The animation covers the cycle through v1.0 · [`EXPLORE.md`](../EXPLORE.md) has every arm through v3.0
and every arm on the v4.0-VL, v5.0-VL, v6.0-VL and v6.1-VL lines, with why each failed · [`versions/v1.0.md`](../versions/v1.0.md#10-how-it-got-here) has the trail before
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
