# RSI-Jev

**A recursively self-improving research system that builds
[Jev](https://docs.typesafe.ai/api)-style *System One* models.** AI agents propose the
hypotheses, register their predictions before spending GPU time, run the experiments, and
retire their own champions when the evidence says to.

[![License](https://img.shields.io/badge/code-MIT-black)](LICENSE)
[![Weights](https://img.shields.io/badge/weights-🤗%20Hugging%20Face-black)](https://huggingface.co/shgao)
[![API](https://img.shields.io/badge/API-Jev%20compatible-black)](serve/README.md)
[![Release](https://img.shields.io/badge/release-v3.0-black)](versions/v3.0.md)

Ask one of these models a typed question about a document — yes/no, pick-one-of-*k*,
rate-on-a-rubric — and a single forward pass returns a probability for every option instead of
prose. Nothing is generated, so another decision about a document already read costs about
**10 ms**.

The loop running the research is the **next version of
[AutoScientists](https://github.com/mims-harvard/AutoScientists)**.

*Want to collaborate, or support the work with compute or funding? Reach out to
**[Shanghua Gao](https://shgao.site)**.*

## RSI process

<p align="center">
  <img src="assets/loop-social.gif" width="900" alt="Above: six turns of the cycle climb to v1.0 and the champion line rises with them, then twenty-four directions press against that line without crossing it. Below: one experiment travels propose, experiment, learn; at the gate nearly all become published negatives and one becomes a release, which then becomes the bar to clear">
</p>

**v3.0**, the current release, is the first with a **reinforcement-learning stage that earned
its place**. Across 59 reward-trained arms, RL on per-item decisions never beat plain supervised
training on the same data; the one reward that did scores the *order* of many candidates, which
no per-item label encodes. v3.0 is v2.1's lineage with more training data and that listwise
reranking stage on top. It reranks far better than earlier releases and **still worse than
leaving the order alone**, and it does not yet fix the weaknesses external testing found — its
record says which. [`versions/v3.0.md`](versions/v3.0.md) is that record;
[`docs/rl.md`](docs/rl.md) is what was tried and why only this one won.

<details>
<summary>the three releases behind it</summary>

**v2.1** is the first release better *everywhere it was checked*: ahead of v1.0 on three
benchmarks it never trained on, and it recovers the general knowledge v2.0 had traded away by
training the bottom third of the tower at a tenth of the learning rate.
[`versions/v2.1.md`](versions/v2.1.md).

**v2.0** trains on the *train* splits of the decision benchmarks it is measured on — never
their test splits — and adds a small confidence head that rescales every answer without
changing which option wins. Much stronger on those benchmarks, better calibrated everywhere,
and no better than v1.0 off them: [`versions/v2.0.md`](versions/v2.0.md).

**v1.0** is a Qwen3.5 Base tower fine-tuned end to end with a trained cross-attention scorer
on top, fitted to a teacher's full probability distribution over 6,977 synthetic documents
rather than to its labels. It had never seen any benchmark's train split, so its numbers are
zero-shot and remain the reference every later release is compared against.
[`versions/v1.0.md`](versions/v1.0.md) is its record.

</details>

*ECE below is expected calibration error: bin the decisions by the confidence the model
reported, compare each bin with how often it was actually right, average the gaps. Lower is
better, 0 is perfect.*

**Being explored now: can it explain a decision it has already made?** The answer comes from
one forward pass with nothing generated, so there is no reasoning trace to read. The current
arms ask whether a decision-tuned tower can be given its words back without giving up the
decision, and whether an explanation it produces is the reason or a plausible story.

Ten arms out of the ninety-two run since v1.0 got from one release to the next, each ruling
something out:

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
| a listwise reranking reward (NDCG@5 over 16 candidates) on top, against a KL anchor | hippo R@1 0.192 → 0.308, suite −0.003 | **the first RL stage kept**: a reward over the *order* of candidates carries what per-item labels cannot. Fifty-eight other reward-trained arms did not |
| **v3.0** | suite **0.756**, ECE **0.066** | better at ranking, still below no reranking at all; the weaknesses found outside are the next release's work |

The animation covers the cycle through v1.0 · [`EXPLORE.md`](EXPLORE.md) has every arm and why
each failed · [`versions/v1.0.md`](versions/v1.0.md#10-how-it-got-here) has the trail before
v1.0

## RSI-Jev models

| model | download | 12-benchmark suite | calibration (ECE) | per decision |
|---|---|---|---|---|
| **RSI-Jev-v2.1-2B** | [**⬇ Hugging Face**](https://huggingface.co/shgao/rsi-jev-v2.1-qwen3.5-2b) | **0.7291** | **0.0556** | ~10 ms |
| RSI-Jev-v2.0-2B | [⬇ Hugging Face](https://huggingface.co/shgao/rsi-jev-v2.0-qwen3.5-2b) | 0.7056 | 0.0546 | ~10 ms |
| RSI-Jev-v1.0-2B | [⬇ Hugging Face](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-2b) | 0.609 | 0.0921 | ~10 ms |
| RSI-Jev-v1.0-0.8B | [⬇ Hugging Face](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-0.8b) | – | – | ~10 ms |

| benchmark | **v2.1** | v1.0 | v2.1 ECE |
|---|---|---|---|
| typed-decisions | **0.7905** | 0.659 | 0.063 |
| Nimble public | **0.8068** | 0.677 | 0.017 |
| MMLU-Pro 1k | **0.383** | 0.351 | 0.065 |
| Jev-Style panel | **0.7493** | 0.734 | 0.052 |
| Kev transfer | **0.7546** | 0.665 | 0.068 |
| Kev hard | **0.6976** | 0.393 | 0.040 |
| JevBench | **0.7217** | 0.663 | 0.062 |
| Kev documents | **0.8632** | 0.714 | 0.069 |
| Kev devtools | **0.7151** | 0.579 | 0.079 |
| Nimble holdout | **0.7469** | 0.565 | 0.088 |
| procedural | **0.8728** | 0.647 | 0.032 |
| Open-Jev OOD | **0.6513** | 0.600 | 0.085 |
| **suite mean** | **0.7291** | 0.609 | **0.0556** |

**v2.1 is ahead on all twelve.** It is also the first release whose gains are not only on the
benchmarks it trained for: on three benchmarks frozen before any of this work began, and in no
training corpus, it beats v1.0 by +0.039, +0.036 and +0.060 — v2.0 beat it on none of them.

**The calibration is still the part to act on.** v2.1 says 0.9 or better on 519 of
typed-decisions' 2,000 questions and is right on 98.1% of them; v1.0 was that confident about
only 29. So over a quarter of the workload can be decided automatically, and it costs nothing —
same ~10 ms, and rescaling never changes which option wins.

Eleven of the twelve benchmarks contributed train-split data, so treat those figures as
in-domain and not as zero-shot: pick v1.0 if you need a zero-shot number or the 0.8B size.

Architecture, training recipe and limitations are on each model card.
[`BENCHMARKS.md`](BENCHMARKS.md) is what these numbers mean and how to re-run them;
[`versions/v2.1.md`](versions/v2.1.md) is the full record, every figure with its caveats.

To run them:

```bash
git clone https://github.com/Shanghua-Gao/RSI-Jev && cd RSI-Jev && pip install -r requirements.txt
```

| | | |
|---|---|---|
| **Compare releases** | `python scripts/demo_web.py` | every released version side by side, answers moving as you type |
| **Serve** | `python scripts/serve.py --ckpt DIR` | `POST /v1/systemone`, Jev's own request and answer shapes → [`serve/README.md`](serve/README.md) |
| **Load** | `load_release(path, device="cuda")` | from `scripts/load_release.py`; a published checkpoint carries its own code |
| **Score the suite** | `python scripts/suite.py --ckpt DIR` | the table above; `--list` names each benchmark's pinned source |
| **Retrain** | `python scripts/release_train.py` | one H100, ~50 min per seed at 2B → [`rsijev/README.md`](rsijev/README.md) |
| **Measure** | `bench.py`, `calibration.py`, `routing.py` | these tables, on your hardware → [`BENCHMARKS.md`](BENCHMARKS.md) |

Runs on an **NVIDIA DGX Spark** (where it is developed), any CUDA GPU, Apple Silicon, or plain
CPU — per-machine costs in [`versions/v1.0.md`](versions/v1.0.md#7-what-it-costs-to-run).

## What the next version is

Not decided yet. That is the invitation.

**Say what is wrong with it. Bluntly.** There is no maintainer here to offend — an AI agent
does not get defensive about a bad review, it registers a prediction and spends GPU time on it.
A case where the model is **confidently wrong** is worth more to this project than any
compliment. A variant you tried that **failed** is worth more than one that worked, because it
deletes a branch nobody has to pay for again.

[**The model got it wrong →**](https://github.com/Shanghua-Gao/RSI-Jev/issues/new?template=wrong-answer.yml) · the document, the question, the answer you expected
[**I tried a variant →**](https://github.com/Shanghua-Gao/RSI-Jev/issues/new?template=experiment.yml) · what you changed, per-seed numbers, which kernel stack

Your report becomes a registered prediction with a null floor measured against it, and ships as
a version — pass or fail. [`CONTRIBUTING.md`](CONTRIBUTING.md) is what happens in between.

## Human in the loop

The loop runs its own experiments and retires its own champions, but it does not decide what is
worth measuring, and it does not notice on its own when a number is technically true and
practically misleading. People do that, and it has changed both the experiments this project
runs and the rules it judges them by. Two of the rules in
[`BENCHMARKS.md`](BENCHMARKS.md#how-a-result-becomes-a-result) exist because someone pushed
back: the MMLU-Pro guard was widened rather than letting a near-miss be discarded, and the seed
requirement now scales with the effect size instead of spending four seeds on every difference.
v2.1 itself started as a refusal to drop an arm that had failed one guard.

Thank you to the people who have sent that feedback:

- **Shanghua Gao** · [@gasvn](https://github.com/gasvn)
- **Sufian** · [@SufianTA](https://github.com/SufianTA)

## Read more

| | |
|---|---|
| [`versions/`](versions/) | **one record per release, kept** — how it was built, what it scores, what it costs, its limitations. Currently [`v3.0.md`](versions/v3.0.md), [`v2.1.md`](versions/v2.1.md), [`v2.0.md`](versions/v2.0.md) and [`v1.0.md`](versions/v1.0.md) |
| [`EXPLORE.md`](EXPLORE.md) | **what the loop tried and rejected** between releases |
| [`docs/rl.md`](docs/rl.md) | **where RL beat supervised training, and where it didn't** — 59 reward-trained arms, one kept |
| [`BENCHMARKS.md`](BENCHMARKS.md) | **what each number means** and how to reproduce it |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | **what to send and what happens to it** |
| [`serve/README.md`](serve/README.md) | **the HTTP API** — copied from Jev exactly, except where stated |
| [`rsijev/README.md`](rsijev/README.md) | **the code** — and which files the loop may rewrite |

## Acknowledgements

Thanks to **NVIDIA** for providing the **DGX Spark** used for inference testing.

## License

Code MIT. Checkpoints follow their base model's license (Apache-2.0). Not affiliated with
TypeSafe AI.
