# RSI-Jev

**A recursively self-improving research system that builds
[Jev](https://docs.typesafe.ai/api)-style *System One* models.** AI agents propose the
hypotheses, register their predictions before spending GPU time, run the experiments, and
retire their own champions when the evidence says to.

[![License](https://img.shields.io/badge/code-MIT-black)](LICENSE)
[![Weights](https://img.shields.io/badge/weights-🤗%20Hugging%20Face-black)](https://huggingface.co/shgao)
[![API](https://img.shields.io/badge/API-Jev%20compatible-black)](serve/README.md)
[![Release](https://img.shields.io/badge/release-v4.0--VL-black)](versions/v4.0-vl.md)

Ask one of these models a typed question about a document or, from v4.0-VL, about an image —
yes/no, pick-one-of-*k*, rate-on-a-rubric — and a single forward pass returns a probability for
every option instead of prose. Nothing is generated, so another decision about a document already read costs about
**10 ms**.

The loop running the research is the **next version of
[AutoScientists](https://github.com/mims-harvard/AutoScientists)**.

*Want to collaborate, or support the work with compute or funding? Reach out to
**[Shanghua Gao](https://shgao.site)**.*

## RSI process

<p align="center">
  <img src="assets/loop-social.gif" width="900" alt="The champion line climbs from v1.0 (0.622) to v2.0 (0.709), v2.1 (0.736) and v3.0 (0.756) on the 15-benchmark suite, one evaluator for every release; grey dots are the experiments tried in between that did not clear it. Below: propose, experiment, learn; at the gate most become published negatives and one becomes a release, which becomes the bar to clear">
</p>

**v4.0-VL**, the current release, **reads images.** Send one to four pictures with a request —
a photo, a screenshot, a scanned page — and ask the same typed questions about them. On five
image benchmarks held out from training it answers 80.3% correctly, and 45.8% with the images
blanked out. On text it keeps v3.0's level (suite 0.756). It is v3.0's lineage with three rounds
of image training and a new RL stage that charges a confident mistake four times what it charges
a timid correct answer. [`versions/v4.0-vl.md`](versions/v4.0-vl.md) is its record.

<details>
<summary>the four releases behind it</summary>

**v3.0** is **the first release where reinforcement learning works**. The loop found the reward
RL needs here: one that scores the *order* of many candidates, which no per-item label can
express. Trained with it, v3.0 ranks the right memory first **60% more often** than its own
supervised parent (hippo R@1 0.192 → 0.308, 64 questions won against 6), with the rest of the
suite held level. [`versions/v3.0.md`](versions/v3.0.md) is its record;
[`docs/rl.md`](docs/rl.md) is how the loop got there, across 59 reward-trained arms.

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

Of the 204 arms run since v1.0, these are the ones that got from one release to the next, each
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
| generated reasoning questions with soft targets where the model is at chance | held-out ECE after calibration 0.121 → 0.075 | the image rounds had made it overconfident; this repairs most of it |
| an RL stage charging a confident mistake 4x a timid correct answer | held-out ECE before calibration 0.200 → 0.082 | against a matched supervised stage: better accuracy and calibration, no detectable ranking difference (one seed) |
| **v4.0-VL** | image top-1 **0.803** held out, suite **0.756**, ECE **0.043** | **reads images**, with text held at v3.0's level |

The animation covers the cycle through v1.0 · [`EXPLORE.md`](EXPLORE.md) has every arm and why
each failed · [`versions/v1.0.md`](versions/v1.0.md#10-how-it-got-here) has the trail before
v1.0

## RSI-Jev models

| model | download | 15-benchmark suite | held-out (eval_final_v2) | calibration (ECE) | per decision |
|---|---|---|---|---|---|
| **RSI-Jev-v4.0-VL-2B** · text and images | [**⬇ Hugging Face**](https://huggingface.co/shgao/rsi-jev-v4.0-vl-qwen3.5-2b) | **0.756** | **0.653** | **0.043** | ~10 ms text, ~0.2–0.5 s with an image |
| RSI-Jev-v3.0-2B | [⬇ Hugging Face](https://huggingface.co/shgao/rsi-jev-v3.0-qwen3.5-2b) | 0.756 | 0.649 | 0.066 | ~10 ms |
| RSI-Jev-v2.1-2B | [⬇ Hugging Face](https://huggingface.co/shgao/rsi-jev-v2.1-qwen3.5-2b) | 0.736 | 0.633 | 0.059 | ~10 ms |
| RSI-Jev-v2.0-2B | [⬇ Hugging Face](https://huggingface.co/shgao/rsi-jev-v2.0-qwen3.5-2b) | – | – | – | ~10 ms |
| RSI-Jev-v1.0-2B | [⬇ Hugging Face](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-2b) | 0.622 | 0.604 | – | ~10 ms |
| RSI-Jev-v1.0-0.8B | [⬇ Hugging Face](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-0.8b) | – | – | – | ~10 ms |

v4.0-VL and v3.0 were scored in one job by the same evaluator, and v3.0, v2.1 and v1.0 in an
earlier one with the same evaluator; v3.0 scores the same in both. v2.0 and the 0.8B were not
re-scored on this suite, and their own cards have their numbers on the earlier twelve-benchmark
one. v1.0 ships no calibration, so it has no calibrated ECE. Only v4.0-VL takes images.

| benchmark | **v4.0-VL** | v3.0 | v4.0-VL ECE |
|---|---|---|---|
| typed-decisions | 0.787 | 0.791 | 0.025 |
| Nimble public | 0.801 | 0.801 | 0.022 |
| MMLU-Pro 1k | 0.385 | 0.364 | 0.078 |
| Jev-Style panel | 0.823 | 0.835 | 0.026 |
| Kev transfer | 0.784 | 0.792 | 0.034 |
| Kev hard | 0.762 | 0.760 | 0.020 |
| JevBench | 0.696 | 0.700 | 0.106 |
| tasksource | 0.707 | 0.701 | 0.033 |
| SemIf external | 0.913 | 0.921 | 0.122 |
| scienthoon OOD | 0.754 | 0.705 | 0.042 |
| Nimble holdout | 0.769 | 0.778 | 0.066 |
| Kev documents | 0.856 | 0.869 | 0.060 |
| Kev devtools | 0.717 | 0.715 | 0.043 |
| procedural | 0.862 | 0.871 | 0.017 |
| Open-Jev OOD | 0.828 | 0.838 | 0.045 |
| **suite mean** | **0.756** | 0.756 | **0.043** |

| images, held out from training | **v4.0-VL** | images blanked |
|---|---|---|
| MMBench (dev) | 0.845 | 0.290 |
| RealWorldQA | 0.707 | 0.371 |
| POPE | 0.912 | 0.502 |
| HallusionBench | 0.695 | 0.503 |
| InfographicVQA (val) | 0.858 | 0.624 |
| **mean** | **0.803** | 0.458 |

Ten of the fifteen text benchmarks contribute train-split data, so none of those figures is
zero-shot; the held-out set is the comparison that is. Pick v1.0 if you need a zero-shot number
or the 0.8B size.

Architecture, training recipe and limitations are on each model card.
[`BENCHMARKS.md`](BENCHMARKS.md) is what these numbers mean and how to re-run them;
[`versions/v4.0-vl.md`](versions/v4.0-vl.md) is the full record, every figure with its caveats.

To run one:

```bash
pip install "rsi-jev[fast,vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev"
rsi-jev serve shgao/rsi-jev-v4.0-vl-qwen3.5-2b --port 8000

curl localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": [{"role": "user", "content": "I was charged twice. Please refund."}],
  "questions": {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}}
}'
```

The first run downloads the checkpoint and its base model into the Hugging Face cache.
`[fast]` adds the fused DeltaNet kernels (CUDA only, 1.25–1.6x on a GB10); the startup log
says whether they are active; `[vision]` (Pillow, torchvision) is what image requests need.
An existing Jev client works unchanged once its base URL points here.

With an image, add an `images` list of data URLs and mark where each one goes in the state:

```bash
IMG=$(base64 -w0 photo.jpg)
curl localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": "Customer photo: <image>\nThe customer says it arrived damaged.",
  "images": ["data:image/jpeg;base64,'"$IMG"'"],
  "questions": {"damaged": {"type": "noul", "instructions": "Does the photo show visible damage?"}}
}'
```

In Python, with no server:

```python
from rsijev import Decider
d = Decider("shgao/rsi-jev-v4.0-vl-qwen3.5-2b")
d.decide("Customer photo: <image>\nThe customer says it arrived damaged.",
         {"damaged": {"type": "noul", "instructions": "Does the photo show visible damage?"}},
         images=["photo.jpg"])     # a path, bytes, a PIL image or a data URL
```

[`docs/inference.md`](docs/inference.md) is the inference guide: which setup to pick, what
each costs on a GB10, and what we tried to make it faster.

To work on the code, clone it instead:

```bash
git clone https://github.com/Shanghua-Gao/RSI-Jev && cd RSI-Jev && pip install -r requirements.txt
```

| | | |
|---|---|---|
| **Compare releases** | `python scripts/demo_web.py` | every released version side by side, answers moving as you type |
| **Serve** | `python scripts/serve.py --ckpt DIR` | `POST /v1/systemone`, Jev's own request and answer shapes → [`serve/README.md`](serve/README.md). `--ckpt` also takes a Hugging Face id |
| **Load** | `load_release(path, device="cuda")` | from `scripts/load_release.py`; a published checkpoint carries its own code |
| **Score the suite** | `python scripts/suite.py --ckpt DIR` | the table above; `--list` names each benchmark's pinned source |
| **Retrain** | `python scripts/release_train.py` | one H100, ~50 min per seed at 2B → [`rsijev/README.md`](rsijev/README.md) |
| **Measure** | `bench.py`, `calibration.py`, `routing.py` | these tables, on your hardware → [`BENCHMARKS.md`](BENCHMARKS.md) |

Runs on an **HP ZGX Nano** (NVIDIA GB10, where it is developed), any CUDA GPU, Apple Silicon, or plain
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
| [`versions/`](versions/) | **one record per release, kept** — how it was built, what it scores, what it costs, its limitations. Currently [`v4.0-vl.md`](versions/v4.0-vl.md), [`v3.0.md`](versions/v3.0.md), [`v2.1.md`](versions/v2.1.md), [`v2.0.md`](versions/v2.0.md) and [`v1.0.md`](versions/v1.0.md) |
| [`EXPLORE.md`](EXPLORE.md) | **what the loop tried and rejected** between releases |
| [`docs/rl.md`](docs/rl.md) | **where RL beat supervised training, and where it didn't** — 59 reward-trained arms, one kept |
| [`docs/speed.md`](docs/speed.md) | **how AutoScientists made inference faster** — read the document once; 1.25–1.63x by default, up to 5.8x in agent loops |
| [`BENCHMARKS.md`](BENCHMARKS.md) | **what each number means** and how to reproduce it |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | **what to send and what happens to it** |
| [`serve/README.md`](serve/README.md) | **the HTTP API** — copied from Jev exactly, except where stated |
| [`docs/inference.md`](docs/inference.md) | **running it** — setups, measured latency, and the speed-ups tried |
| [`rsijev/README.md`](rsijev/README.md) | **the code** — and which files the loop may rewrite |

## Acknowledgements

Thanks to **HP** and **NVIDIA** for providing the **HP ZGX Nano AI Station**, powered by the
**NVIDIA GB10 Grace Blackwell Superchip**.

## License

Code MIT. Checkpoints follow their base model's license (Apache-2.0). Not affiliated with
TypeSafe AI.
