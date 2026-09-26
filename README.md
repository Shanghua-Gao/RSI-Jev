# RSI-Jev

**A recursively self-improving research system that builds
[Jev](https://docs.typesafe.ai/api)-style *System One* models.** AI agents propose the
hypotheses, register their predictions before spending GPU time, run the experiments, and
retire their own champions when the evidence says to.

[![License](https://img.shields.io/badge/code-MIT-black)](LICENSE)
[![Weights](https://img.shields.io/badge/weights-🤗%20Hugging%20Face-black)](https://huggingface.co/shgao)
[![API](https://img.shields.io/badge/API-Jev%20compatible-black)](serve/README.md)
[![Release](https://img.shields.io/badge/release-v2.0-black)](versions/v2.0.md)

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

**v2.0**, the current release, trains on the *train* splits of the decision benchmarks it is
measured on — never their test splits — and adds a small confidence head that rescales every
answer without changing which option wins. It is much stronger on those benchmarks and better
calibrated everywhere. On three benchmarks frozen before the search began it does not beat
v1.0, which is the more interesting half of the result: [`versions/v2.0.md`](versions/v2.0.md).

<details>
<summary><b>v1.0</b>, the release it had to beat</summary>

A Qwen3.5 Base tower fine-tuned end to end with a trained cross-attention scorer on top,
fitted to a teacher's full probability distribution over 6,977 synthetic documents rather than
to its labels. It had never seen any benchmark's train split, so its numbers are zero-shot and
remain the reference every later release is compared against.
[`versions/v1.0.md`](versions/v1.0.md) is its record.

</details>

*ECE below is expected calibration error: bin the decisions by the confidence the model
reported, compare each bin with how often it was actually right, average the gaps. Lower is
better, 0 is perfect.*

**Being explored now: data aimed at the model's own weaknesses.** A generator profiles the
current model, writes cases for what it gets wrong, and checks them before training. RL chooses
what to generate next; RL on the decision itself lost to plain supervised training every time.

Nine arms out of about fifty got from one release to the next, each ruling something out:

| where it went | result | what it established |
|---|---|---|
| **v1.0** | typed-decisions **0.662** | the bar to clear, and it had never seen a benchmark's train split |
| train on the benchmark's own train split | typed-decisions 0.7794 | worth +0.12 — and it costs general knowledge, so it was **rejected** |
| the same + 15% general-knowledge replay | typed-decisions 0.796 | the replay pays that cost back. The first keeper |
| the other benchmarks' train splits too, 1k per source | suite 0.6932 | the lever is not specific to one benchmark |
| 3k per source | suite 0.7049 | more of it is better |
| the same on four fresh seeds | suite 0.7029 | +0.069 on seeds it had never run on. Confirmed |
| one global temperature for confidence | suite ECE 0.1073 | the model is under-confident on typed-decisions and over-confident out of distribution, so one number cancels out — it has to be per question |
| a per-question confidence head | suite ECE 0.0639 | it can, and it never changes which option wins |
| the same, forbidden to sharpen score questions | suite ECE 0.0736 | score answers stay faithful to the teacher's spread, for 0.010 of ECE. Shipped |
| **v2.0** | suite **0.7056**, ECE **0.0546** | the same tower as v1.0, start to finish. Capability was v1.0's lever; **what it trains on, and calibrating it afterwards, is a second one** — and that one is free at inference |

The animation covers the cycle through v1.0 · [`EXPLORE.md`](EXPLORE.md) has all fifty arms and
why each failed · [`versions/v1.0.md`](versions/v1.0.md#10-how-it-got-here) has the trail before
v1.0

## RSI-Jev models

| model | download | 12-benchmark suite | calibration (ECE) | per decision |
|---|---|---|---|---|
| **RSI-Jev-v2.0-2B** | [**⬇ Hugging Face**](https://huggingface.co/shgao/rsi-jev-v2.0-qwen3.5-2b) | **0.705** | **0.0546** | ~10 ms |
| RSI-Jev-v1.0-2B | [⬇ Hugging Face](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-2b) | 0.609 | 0.0921 | ~10 ms |
| RSI-Jev-v1.0-0.8B | [⬇ Hugging Face](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-0.8b) | – | – | ~10 ms |

| benchmark | **v2.0** | v1.0 | v2.0 ECE |
|---|---|---|---|
| typed-decisions | **0.7925** | 0.659 | 0.068 |
| Nimble public | **0.7888** | 0.677 | 0.037 |
| MMLU-Pro 1k | 0.305 | **0.351** | 0.040 |
| Jev-Style panel | **0.7419** | 0.734 | 0.056 |
| Kev transfer | **0.7244** | 0.665 | 0.046 |
| Kev hard | **0.6921** | 0.393 | 0.035 |
| JevBench | **0.7217** | 0.663 | 0.065 |
| Kev documents | **0.8782** | 0.714 | 0.059 |
| Kev devtools | **0.7085** | 0.579 | 0.063 |
| Nimble holdout | **0.7654** | 0.565 | 0.066 |
| procedural | 0.6223 | **0.647** | 0.094 |
| Open-Jev OOD | **0.6237** | 0.600 | 0.057 |
| **suite mean** | **0.7056** | 0.609 | **0.0546** |

**The calibration is the part to care about, and it generalises.** v2.0 says 0.9 or better on
539 of typed-decisions' 2,000 questions and is right on 98.3% of them; v1.0 was that confident
about only 29 of them. So over a quarter of the workload can be acted on automatically, and it
costs nothing — same ~10 ms, and rescaling never changes which option wins. Expected
calibration error improves out of distribution too.

The accuracy gains are large but in-domain: v2.0 wins ten of the twelve, and all ten are
benchmarks whose train split it saw. Pick v1.0 if you need a zero-shot number, general knowledge
intact, or the 0.8B size.

Architecture, training recipe and limitations are on each model card.
[`BENCHMARKS.md`](BENCHMARKS.md) is what these numbers mean and how to re-run them;
[`versions/v2.0.md`](versions/v2.0.md) and [`versions/v1.0.md`](versions/v1.0.md) are the full
records, every figure with its caveats.

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

## Read more

| | |
|---|---|
| [`versions/`](versions/) | **one record per release, kept** — how it was built, what it scores, what it costs, its limitations. Currently [`v2.0.md`](versions/v2.0.md) and [`v1.0.md`](versions/v1.0.md) |
| [`EXPLORE.md`](EXPLORE.md) | **what the loop tried and rejected** between releases |
| [`BENCHMARKS.md`](BENCHMARKS.md) | **what each number means** and how to reproduce it |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | **what to send and what happens to it** |
| [`serve/README.md`](serve/README.md) | **the HTTP API** — copied from Jev exactly, except where stated |
| [`rsijev/README.md`](rsijev/README.md) | **the code** — and which files the loop may rewrite |

## Acknowledgements

Thanks to **NVIDIA** for providing the **DGX Spark** used for inference testing.

## License

Code MIT. Checkpoints follow their base model's license (Apache-2.0). Not affiliated with
TypeSafe AI.
