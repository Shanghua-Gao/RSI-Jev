# Making a decision model fast on a desktop GPU

*RSI-Jev v3.0 · measured on an HP ZGX Nano (NVIDIA GB10)*

RSI-Jev is an open, Jev-style decision model. You give it some text and a few questions, each
with the answers you allow, and it returns a probability for every answer instead of writing a
reply. It runs on your own machine, and it usually sits inside something that is waiting for it:
a support queue, a moderation pipeline, an agent deciding its next step.

So we set out to answer one question: how fast can the released v3.0 model answer on a desk-side
GPU, without changing a single answer? Out of the box it is now **1.3–1.6x faster**. A
long-running server gets **up to 2.2x** with one flag, and an agent that keeps asking about the
same conversation gets **up to 5.8x**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/speed_gb10_dark.svg">
  <img src="assets/speed_gb10_light.svg" alt="Milliseconds per request on an HP ZGX Nano for six workloads, v3.0 as first released against now: 1.5x to 5.8x faster with the profile that fits." width="100%">
</picture>

Most of this comes from one observation about how these models spend their time.

## The document is the expensive part

Take a support ticket of about 1,000 tokens, and eight questions about it: which team should
handle it, is it a refund request, how urgent is it, and so on. Each question with its options
is 20–60 tokens.

The straightforward way to answer is to give the model the ticket plus one question, eight times
over. That reads the ticket eight times. But the model reads left to right, so its work on the
ticket is identical for every question that follows. It can read the ticket once, keep what it
computed, and continue each question from there. The answers come out exactly the same.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/read_once_dark.svg">
  <img src="assets/read_once_light.svg" alt="Before: the ticket is read once per question, about 8,400 tokens. After: the ticket is read once and all eight questions branch off it, about 1,400 tokens." width="100%">
</picture>

On the GB10, the eight questions take **1,347 ms** when the ticket is read eight times and
**253 ms** when it is read once, 5.3x less. With sixteen questions the gap is 8.8x. A request
costs roughly one reading of the document, plus a little for each question. The practical
advice follows directly: **ask all your questions about a document in one request.**

RSI-Jev's server already did this in v3.0. What we found was that it often was not doing it on
this machine, and that agents were paying for the document again on every call.

## A setting tuned on one GPU was switching it off on another

Reading once has a small fixed cost: a second pass, and a copy of the kept state for each
question. For a short document with only two or three questions, that overhead is bigger than
the saving. So the server only reads once when enough work is saved, and the cut-off had been
measured on an A100.

An A100 is fast enough per pass that the saving must be large to pay off. The GB10 is not, and
its break-even point is about four times lower:

| tokens saved by reading once | 280 | 400 | 480 | 1,200 | 2,800 |
|---|---|---|---|---|---|
| speed-up on the GB10 | 0.75x | 1.06x | 1.28x | 1.97x | 3.84x |

With the A100's cut-off, the GB10 was reading documents repeatedly on every request in that
middle range and losing 1.3–2x. The server now chooses the cut-off by GPU. A number that trades
a fixed cost against saved work belongs to the hardware, not to the model.

## Agents ask about the same thing again and again

An agent calls the model at every step with its whole conversation so far, and each call usually
adds only a few lines. Reading once within a request does not help there: every call starts
from scratch.

`--profile agent` keeps what the model computed for a document across requests. A repeated
document skips the reading entirely, and a document that has grown only reads the new part.
Two details keep it exact. Entries are keyed on the exact tokens and the model weights, never on
the raw text. And the last few tokens are always re-read, because adding text can change how
the old ending splits into tokens (a JSON transcript's closing `}]` turning into `},{` is the
usual case).

On a 1,000-token conversation, asking again drops from **149 ms to 26 ms**. A conversation that
grows by about 100 tokens per step drops from 143 ms to **53 ms** per step. Every answer matches
a fresh reading.

## The smaller wins: make sure the fast code runs

Three quarters of the model's layers are a newer kind of attention (Gated DeltaNet), and their
fast implementation ships as a separate package. Without it, the model silently falls back to
a slow reference version, and the only sign is one line in a log. Installing it made everything
**1.25–1.63x** faster, with results identical to the training run's own records. It is now part
of `pip install "rsi-jev[fast]"`, and the server prints at startup which kernels it is using.

For a server that runs for hours, `--profile server` also compiles the model (about 40 s at
startup), for another **1.1–1.3x**.

## What did not help, and why

A single pass has a floor on this machine. The model's weights are 2.75 GB, and the GB10 reads
memory at about 273 GB/s, so one pass cannot take less than about 10 ms; one question measures
22–27 ms. When many questions are answered together, the weights are read once for all of them
and the GPU spends its time computing. Tricks aimed at other bottlenecks had little to work with:

| tried | what happened |
|---|---|
| 4-bit and 8-bit weights | faster only for a single short question (28 → 19 ms with 4-bit), and they changed answers: 4-bit agreed with the full model on only 69–77% of MMLU-Pro questions |
| FP8 | slower without compiling, and changed 2–8% of answers |
| a faster kernel for the DeltaNet layers (FlashQLA) | 2–2.7x faster on its own, but those layers take about 9% of the time: 1–13% overall |
| CUDA graphs | slower: they remove launch overhead, and the GPU was already busy computing |
| vLLM | built for generating text; on this model it keeps shared documents in 544-token blocks, so short documents are re-read for every question. It did handle many concurrent single questions 2.2x better |

We held every change to one rule: **a faster answer must be the same answer.** People set
thresholds on these probabilities, so a speed-up that moves them is a regression. Each change was
compared with the unchanged model on the release's verification sets, on 200 questions from the
training run's own records, and on calibration error. Quantization failed that test, and it
failed first on knowledge-heavy questions.

## Try it

```bash
pip install "rsi-jev[fast] @ git+https://github.com/Shanghua-Gao/RSI-Jev"
rsi-jev serve shgao/rsi-jev-v3.0-qwen3.5-2b                    # default
rsi-jev serve shgao/rsi-jev-v3.0-qwen3.5-2b --profile agent    # agents re-asking about a conversation
rsi-jev serve shgao/rsi-jev-v3.0-qwen3.5-2b --profile server   # long-running servers
```

It speaks Jev's API, so an existing Jev client only needs its base URL changed.
[`inference.md`](inference.md) has the details.

## What comes next

The server side is now close to what this machine allows. Going further means changing the model:
fewer layers per answer, a smaller distilled model, or training with low-bit weights in mind so
that they keep the answers. On datacenter GPUs, CUDA graphs do help: one question takes 12.1 ms
on an H100 in our tests, a path we have not released yet.

*Measured on an HP ZGX Nano AI Station with the NVIDIA GB10 Grace Blackwell Superchip, provided by
HP and NVIDIA. Medians of 20 runs, one configuration per process, bf16 model; raw data in
[`assets/bench_gb10.json`](assets/bench_gb10.json).*
