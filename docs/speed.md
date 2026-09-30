# Where the time goes in a decision model

*RSI-Jev v3.0 · HP ZGX Nano (NVIDIA GB10)*

**In a decision model the document is the cost, and the questions are almost free. So the
biggest speed-up comes from reading each document once, not from faster kernels.** With that
and two smaller changes, the same v3.0 model on the same machine answers 1.25–1.63x faster by
default, up to 2.2x faster with `--profile server`, and up to 5.8x faster in an agent loop.
No answer changed.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/speed_gb10_dark.svg">
  <img src="assets/speed_gb10_light.svg" alt="Milliseconds per request on an HP ZGX Nano for six workloads, v3.0 as first released against now: 1.5x to 5.8x faster with the profile that fits." width="100%">
</picture>

## What we tried

| change | on the GB10 | kept |
|---|---|---|
| **read the document once per request**, with a threshold set per machine | up to 2x on requests the old threshold missed | **yes, default** |
| **fused kernels for the linear-attention layers** | 1.25–1.63x | **yes, `[fast]`** |
| **remember documents across requests** | same state again 148.6 → 25.7 ms; a growing state 142.6 → 52.6 ms | **yes, `--profile agent`** |
| **compile the model** | another 1.1–1.3x | **yes, `--profile server`** |
| FP8, 4-bit and 8-bit weights | faster only for one short question | no: answers changed |
| a faster linear-attention kernel (FlashQLA) | 2–2.7x on the kernel, 1–13% end to end | no: answers moved more than allowed |
| CUDA graphs | slower | no |
| vLLM | 2.2x throughput for many single short questions | experimental: slower for several questions per request |

## The document is the cost

A decision model reads a state, then a question and its options, and scores the options. Ask
eight questions about a 1,000-token ticket and the naive server reads the ticket eight times,
because each question is encoded as its own sequence. The questions themselves are a few dozen
tokens each.

Under a causal mask the document's contribution is the same for every question, so it can be
read once, kept, and every question continued from that point. The answer is identical. On this
model the kept state is hybrid: keys and values for the 6 attention layers, and a recurrent state
for the 18 linear-attention layers. Both continue exactly.

**Take-home:** the cost of a request is roughly *one read of the document plus a small cost
per question*. Design the serving path around that, and ask many questions per request.

## A threshold is a property of the machine, not of the model

Reading once is not free: it adds a second pass and copies the kept state to every question in
the batch. So the server only does it when enough work is saved. That cut-off had been tuned on
an A100, where one pass is cheap, and it was applied everywhere. On the GB10 the break-even point
is four times lower:

| tokens saved by reading once | 280 | 400 | 480 | 1,200 | 2,800 | 15,000 |
|---|---|---|---|---|---|---|
| speed-up on the GB10 | 0.75x | 1.06x | 1.28x | 1.97x | 3.84x | 8.77x |

With the A100 setting the GB10 was giving up 1.3–2x on every request in between. The server now
picks the threshold by GPU.

**Take-home:** any performance setting that trades a fixed cost against saved work has to be
measured on the hardware it runs on.

## Check what is actually running

Three quarters of Qwen3.5's layers are linear attention. Without the fused kernels installed,
transformers runs them through a slow reference implementation, and the only sign is a line in
the log. Installing the kernels gave 1.25–1.63x, with the verification agreeing 1.0000 with the
training record. The server now reports at startup which kernels are in use.

**Take-home:** on new hybrid architectures, the fast path is an optional dependency. Verify it
is active before measuring anything else.

## Agents ask again: remember what you read

An agent asks about the same state step after step, and usually only appends to it. Keeping each
read across requests turns a repeated state into a cache hit and a growing state into a read of
only the new part. Two details make it safe: the cache is keyed on the exact tokens and the
weights, never on text; and the last few tokens are always re-read, because appending text can
change how the old end tokenizes. On a 1,052-token state, asking again drops from 148.6 ms to
**25.7 ms**; a transcript growing by about 100 tokens per step drops from 142.6 to **52.6 ms**
per step, with every answer equal to a fresh read.

**Take-home:** for agents, the fastest document is the one you already read.

## Why the usual tricks did not help here

One pass has a hard floor on this machine: the model's 2.75 GB of bf16 weights take about 10 ms
to read at the GB10's ~273 GB/s. A single question measures 22–27 ms. With many questions the
weights are read once for all of them, and the GPU is busy computing: changing the batch size
moved latency by at most 15%, and CUDA graphs, which save kernel-launch time, made it slower.

That explains the rest of the table. Low-bit weights only speed up the one case that is bound by
reading weights, a single short question (28 → 19 ms with 4-bit). A faster linear-attention
kernel speeds up a part that is about 9% of the time. And vLLM, built for generating text, keeps
shared documents in 544-token blocks on this hybrid model, so it re-reads short documents for
every question.

**Take-home:** know whether you are bound by memory or by compute before choosing a trick.

## A faster answer must be the same answer

A decision model's product is its probabilities, and users set thresholds on them. So every
change was checked against the unchanged model: agreement on the release's verification sets,
the 200-question roundtrip against the training record, and calibration error within ±0.003.
Several fast options failed. 4-bit weights agreed on only 69–77% of MMLU-Pro answers, 8-bit on
97.7%, and FP8 changed 2–8% of answers. Quantization changed knowledge-heavy answers first.

**Take-home:** for decision models, speed-ups that change answers are regressions.

## What would make it faster still

The remaining gains need a different model, not a different server: fewer layers per pass
(early exit, being tested), a smaller distilled tower, or training with low-bit weights in mind
so that they pass the checks above. On datacenter GPUs, CUDA graphs do help: one question drops
to 12.1 ms on an H100 in our tests; that path is not released yet.

Measured on an HP ZGX Nano AI Station (NVIDIA GB10), provided by HP and NVIDIA. The numbers are
medians of 20 runs, one configuration per process; raw data in
[`assets/bench_gb10.json`](assets/bench_gb10.json). [`inference.md`](inference.md) covers
installation, the profiles and which one to use.
