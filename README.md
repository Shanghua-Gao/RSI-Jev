<h1 align="center">RSI-Jev</h1>

<p align="center">
<b>A recursively self-improving research system that builds
<a href="https://docs.typesafe.ai/api">Jev</a>-style <i>System One</i> models.</b>
</p>

<p align="center">
  <b><a href="docs/history.md">How each release was found</a> ·
  <a href="EXPLORE.md">Every experiment, failures included</a> ·
  <a href="docs/rl.md">Where RL helped and didn't</a></b>
</p>

<p align="center">
  <a href="https://huggingface.co/shgao">🤗 Models</a> ·
  <a href="https://shanghua-gao.github.io/RSI-Jev/">Demos</a> ·
  <a href="https://colab.research.google.com/github/Shanghua-Gao/RSI-Jev/blob/main/notebooks/rsi_jev_v4_vl_quickstart.ipynb">Colab</a> ·
  <a href="docs/inference.md">Docs</a> ·
  <a href="versions/v4.0-vl.md">Release notes</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/code-MIT-black" alt="MIT"></a>
  <a href="https://huggingface.co/shgao/rsi-jev-v4.0-vl-qwen3.5-2b"><img src="https://img.shields.io/badge/weights-Apache--2.0-black" alt="Apache-2.0 weights"></a>
  <a href="serve/README.md"><img src="https://img.shields.io/badge/API-Jev%20compatible-black" alt="Jev-compatible API"></a>
</p>

AI agents propose the hypotheses, register their predictions before spending GPU time, run
the experiments, and retire their own champions when the evidence says to. Every release, and
every experiment that failed on the way, is published with its numbers. The loop running the
research is the next version of [AutoScientists](https://github.com/mims-harvard/AutoScientists).

<p align="center">
  <img src="assets/loop-social.gif" width="720" alt="The champion line climbs from v1.0 (0.622) to v2.0 (0.709), v2.1 (0.736) and v3.0 (0.756) on the 15-benchmark suite; grey dots are the experiments that did not clear it">
</p>

**Read the exploration:** of 312 experiments since v1.0, the ones that made a release are in
[**docs/history.md**](docs/history.md); all the others, with why each failed, are in
[**EXPLORE.md**](EXPLORE.md); the reinforcement-learning arms are in [**docs/rl.md**](docs/rl.md).

<details>
<summary><b>The path from v1.0 to v4.0-VL: each experiment that moved a release, and what it ruled out</b></summary>

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
| pool each option over its own tokens, not the separator that follows the previous one | CLINC150 0.383 → 0.753 | a readout bug found by an outside benchmark, fixed without retraining |
| **v5.0-VL** | MMLU-Pro **0.429**, image top-1 **0.830**, KoBBQ unknown **0.932** | **3B: the first 20 of 32 layers** |

</details>

## What it builds

Models that answer a yes/no, pick-one-of-*k* or rate-on-a-rubric question about a document, a
chat or an image. One forward pass returns a calibrated probability for every option; nothing is
generated. A second question about a document already read takes about 10 ms.

<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="site/assets/v4/capsules-dark.webp"><img src="site/assets/v4/capsules-light.webp" width="49%" alt="A tray of gel capsules, one leaking: defective, 68% sure"></picture>
  <picture><source media="(prefers-color-scheme: dark)" srcset="site/assets/v4/keep-file-dark.webp"><img src="site/assets/v4/keep-file-light.webp" width="49%" alt="A delete-file dialog; the user said keep the file: Cancel, 99% sure"></picture>
  <br><sub>Capsules photo from VisA (Zou et al. 2022, CC BY 4.0, resized).</sub>
</p>

### News

- **2026-10-02 · v5.0-VL 3B** cuts the LLM to what a System One model needs: the first 20 of
  Qwen3.5-4B's 32 layers, 3.25B parameters. MMLU-Pro 0.385 → 0.429, held-out images 0.802 →
  0.830, and it says "unknown" when a question has no answer (KoBBQ 0.18 → 0.93).
  [Release notes](versions/v5.0-vl.md)
- **2026-10-01 · v4.0-VL** reads images: 80.3% on five held-out image benchmarks, text held
  at v3.0's level, calibration error 0.066 → 0.043. On VisA defect photos it scores 86.6
  AUROC with no defect examples, against 82.9 for Gemma 4 12B.
  [Release notes](versions/v4.0-vl.md)
- **2026-09-28 · v3.0** is the first release where reinforcement learning helps: +60%
  reranking R@1 over its supervised parent. [Release notes](versions/v3.0.md)

## Quickstart

```bash
pip install "rsi-jev[fast,vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev"
rsi-jev serve v5.0-vl-3b --port 8000
```

```bash
curl localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": "I was charged twice. Please refund.",
  "questions": {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}}
}'
```

Or in Python, with no server:

```python
from rsijev import Decider

d = Decider("v5.0-vl-3b")
d.decide("Customer photo: <image>\nThe customer says it arrived damaged.",
         {"damaged": {"type": "noul", "instructions": "Does the photo show visible damage?"}},
         images=["photo.jpg"])
```

The server speaks Jev's API, so an existing Jev client works once its base URL points here.
Images, limits and every option: [`serve/README.md`](serve/README.md). Choosing a setup and
what each costs: [`docs/inference.md`](docs/inference.md).

## Models

| Model | Input | 15-benchmark suite | Held-out set | ECE | |
|---|---|---|---|---|---|
| **v5.0-VL-3B** | text, images | **0.762** | **0.692** | 0.053 | [🤗](https://huggingface.co/shgao/rsi-jev-v5.0-vl-3b) |
| v4.0-VL-2B | text, images | 0.756 | 0.653 | **0.043** | [🤗](https://huggingface.co/shgao/rsi-jev-v4.0-vl-qwen3.5-2b) |
| v3.0-2B | text | 0.756 | 0.649 | 0.066 | [🤗](https://huggingface.co/shgao/rsi-jev-v3.0-qwen3.5-2b) |
| v2.1-2B | text | 0.736 | 0.633 | 0.059 | [🤗](https://huggingface.co/shgao/rsi-jev-v2.1-qwen3.5-2b) |
| v2.0-2B | text | – | – | – | [🤗](https://huggingface.co/shgao/rsi-jev-v2.0-qwen3.5-2b) |
| v1.0-2B | text | 0.622 | 0.604 | – | [🤗](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-2b) |
| v1.0-0.8B | text | – | – | – | [🤗](https://huggingface.co/shgao/rsi-jev-v1.0-qwen3.5-0.8b) |

All are fine-tuned from Qwen3.5 Base; v5.0-VL runs the first 20 of Qwen3.5-4B-Base's 32 layers. Ten of the fifteen suite benchmarks contribute
train-split data, so the held-out set is the zero-shot comparison; v1.0 never saw a train
split. Per-benchmark scores, image results and caveats are in each release's
[record](versions/). ECE is expected calibration error (lower is better).

## Documentation

| | |
|---|---|
| [Inference guide](docs/inference.md) | setups, measured latency, speed-ups |
| [HTTP API](serve/README.md) | the Jev-compatible server |
| [Benchmarks](BENCHMARKS.md) | what each number means and how to reproduce it |
| [Release records](versions/) | how each release was built, what it scores, its limitations |
| [Reinforcement learning](docs/rl.md) | where RL beat supervised training, and where it didn't |
| [Speed](docs/speed.md) | how inference was made faster |
| [Code](rsijev/README.md) | training and the files the loop may rewrite |

## Contributing

A case where the model is confidently wrong, or a variant you tried that failed, is the most
useful thing you can send: [report a wrong answer](https://github.com/Shanghua-Gao/RSI-Jev/issues/new?template=wrong-answer.yml)
· [report an experiment](https://github.com/Shanghua-Gao/RSI-Jev/issues/new?template=experiment.yml).
Each becomes a registered prediction tested in the next version. See
[`CONTRIBUTING.md`](CONTRIBUTING.md). To collaborate or support the work with compute,
contact [Shanghua Gao](https://shgao.site).

## Acknowledgements

Thanks to **HP** and **NVIDIA** for providing the **HP ZGX Nano AI Station**, powered by the
**NVIDIA GB10 Grace Blackwell Superchip**.

## License

Code: MIT. Weights: Apache-2.0, following the base model; some image training sources are
non-commercial, listed on each model card. Not affiliated with TypeSafe AI. To cite, use
[`CITATION.cff`](CITATION.cff).
