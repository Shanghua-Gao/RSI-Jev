# How deep does a decision need to go?

A System One model reads a document, looks at a few fixed options, and returns a probability for
each one. It never writes. In v5.0-VL we found that a model built this way doesn't need the whole
LLM: cut at layer 20 of 32, it decided as well as the full tower.

v6.0-VL takes the next step. It is still a System One model: one forward pass, no generated
reasoning, a probability for every option. But it has answer heads at layers 16, 20 and 32, and
`effort` sets how many layers that one pass uses, not how many tokens it writes; it writes none:

```bash
rsi-jev serve v6.0-vl-4b --effort medium     # or "effort": "low" | "medium" | "high" | "auto" per request
```

`low`, `medium` and `high` stop at 16, 20 and 32. `auto` answers at the first layer that is
confident enough and goes deeper otherwise. Each layer has its own bar: layer 16 answers only when
its calibrated top-option probability is at least 0.95, layer 20 at 0.59. Every response says which layer answered and how confident it was.

So the obvious question: which decisions actually need the deep layers? We measured it on
Decision Index 0.2.1, a public benchmark of 38 decision tasks.

## The short answer: 20 layers, mostly

| effort | layers | Decision Index | median latency |
|---|---|---:|---:|
| `low` | 16 | 43.3 | 23 ms |
| `medium` | 20 | **45.9** | 27 ms |
| `high` | 32 | **45.9** | 40 ms |
| `auto` | 22 on average | 45.8 | – |
| v5.0-VL | 20 | 37.4 | – |

*16,000-row sample of Decision Index, one H200, bf16; rows that overlap our training data removed. `auto` was read on a different GPU, so it has no latency here; at 22 layers on average it sits between `medium` and `high`.*

This table compares effort levels on the same sample rows. The full benchmark, run as served by default,
scores 46.24: on the 2026-09-28 board, the highest among 4B models and anything smaller (14th of 71 overall).

*These are v6.0-VL's numbers. v6.1-VL, the average of v6.0-VL and a second fine-tune, reads 47.9 /
50.8 / 51.4 / 50.9 (`low` / `medium` / `high` / `auto`) on the same rows. There layer 32 leads
layer 20 by 0.7, so "20 layers, mostly" holds less well for v6.1-VL. The per-task and
per-question analysis below was not repeated on it ([v6.1-VL's record](../versions/v6.1-vl.md)).*

On this benchmark the last 12 layers add nothing on average. Layer 20 matches all 32 at about
two thirds of the latency. Layer 16 gives up 2.6 points for the lowest latency.

## Which tasks need depth

The average hides a clear split by task:

- **Done at layer 16:** intent detection, classification, retrieval, sentiment, function calling.
  BFCL scores .93 at layer 16 and .93 at 32.
- **Needs about 20 layers:** judgement. Natural-language inference, commonsense, claim
  verification, causal questions. HoVer goes from .21 at layer 16 to .32 at 20, BBH from .39 to .48.
- **Keeps improving to 32:** knowledge and multi-step reasoning. GPQA Diamond .14 → .18 → .22,
  MuSR .34 → .39, code execution (CRUXEval) .07 → .13, tool documentation (API-Bank) .60 → .66.

And a few tasks get *worse* with depth: BANKING77 intents, ESCI product search and New Yorker
caption matching all lose a few points from 16 to 32. Deeper is not always better.

## Which questions need depth

Inside a task, the best predictor is the shallow layer's own confidence. We took the 27 Decision
Index benchmarks scored by per-question accuracy (10,232 questions) and checked each question at
layers 16 and 32:

| layer 16's confidence | right at 16 | right at 32 | gain |
|---|---:|---:|---:|
| below 0.50 | 33% | 36% | +3 |
| 0.50–0.59 | 54% | 67% | **+13.5** |
| 0.59–0.70 | 64% | 70% | **+5** |
| 0.70–0.80 | 77% | 77% | +1 |
| 0.80 and above | 90% | 90% | 0 |

When layer 16 is sure, it is right as often as layer 32, so going deeper buys nothing. When it
is torn between two or three options, the deeper layers fix a lot of answers. Two-option
judgement questions and very long option lists (more than 50 options) gain the most; questions
with 5–10 options don't gain at all. Depth also breaks some answers: 4.8% of questions that layer
16 gets right are wrong at layer 32, against 8.7% that depth rescues.

## How we fixed `auto`

Our first `auto` used one bar for both early layers: answer at 0.59. On our own test suite that was
fine; it matched the full model at about 21 layers. On Decision Index it answered 78% of questions
at layer 16 and scored 44.9, a point below both `medium` and `high`.

The table above shows why. Questions where layer 16 was 0.59–0.70 sure still gained 5 points from
going deeper, but they stopped at layer 16. The bar was chosen on data that looks like our suite,
where layer 16 is rarely confidently wrong; Decision Index has more two-option judgement questions
and unfamiliar formats, and there layer 16 is overconfident more often.

So `auto` now has a bar per layer: 0.95 at layer 16 and 0.59 at layer 20. Layer 16 answers only when
it is nearly sure, and the questions it used to take go on to layer 20, which is enough for most of
them. We chose the two bars on our development sets (picked on one half, confirmed on the other,
average depth capped at 24 layers) and then read the test suite once:

| | suite | held-out | MMLU-Pro | calibration error | layers |
|---|---:|---:|---:|---:|---:|
| `high` | 0.770 | 0.695 | 0.443 | 0.033 | 32 |
| `auto`, one bar at 0.59 | 0.769 | 0.696 | 0.444 | 0.032 | 20.9 |
| `auto`, 0.95 at 16 and 0.59 at 20 | **0.771** | 0.696 | **0.444** | **0.024** | 23.3 |

On Decision Index the new `auto` reads 45.8, up from 44.9 and within 0.1 of `medium` and `high`. It averages 22 layers there: 15% of questions stop at layer 16, 62% at 20 and 23% at 32.

`medium` still wins Decision Index on its own, because 20 layers are enough there, but it is not a
better default everywhere: on suite-like questions it falls behind the cascade. No single effort
level wins every workload, which is why it is a setting and not a constant.

## What we'd use

- `low` for routing, intent and retrieval, where layer 16 is already as good as it gets.
- `medium` for classification and judgement: Decision Index quality at about 70% of the latency.
- `high` for knowledge, multi-step reasoning, code and tool documentation.
- `auto` for mixed traffic: its default bars match `high` on our suite at about 23 layers. Read
  `usage.depth` and `usage.confidence` in the response to see where answers came from, and pass
  your own `confidence_threshold` (one number, or one per layer) if your traffic differs.

## How it was made

Like every RSI-Jev release, v6.0-VL was trained, evaluated and documented by a loop of AI
research agents, with every experiment that didn't make it written up next to the ones that did.
The per-layer measurements above came out of the release checks: the same benchmark rows, the same
weights, only the exit changed.

- Model: [shgao/rsi-jev-v6.0-vl-4b](https://huggingface.co/shgao/rsi-jev-v6.0-vl-4b)
- The full per-benchmark and per-question tables: [the release record, section 4.4](../versions/v6.0-vl.md#44-which-tasks-and-questions-need-depth)
- How to set `effort` and `confidence_threshold`: [inference.md](inference.md#effort-levels)

*Apart from the full-run score of 46.24, Decision Index numbers are 16,000-row sample reads of version 0.2.1 with one seed. Per-benchmark
differences under 0.03 are within noise.*
