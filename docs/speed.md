# How the agents made inference faster

*For the v3.0 serving update · measured on an HP ZGX Nano (NVIDIA GB10)*

**The same v3.0 model, on the same machine, now answers 1.25–1.63x faster by default,
up to 2.2x faster with `--profile server`, and up to 5.8x faster in agent loops. The answers
did not change.** This is how the agents got there, and what they tried that did not ship.

Everything below was measured on one HP ZGX Nano (NVIDIA GB10) with RSI-Jev-v3.0-2B, a bf16
tower and 4-option choice questions. Each timing is the median of 20 runs after warm-up. The
before-and-after table comes from one benchmark in one session
([`assets/bench_gb10.json`](assets/bench_gb10.json)).

## Where it started and where it ended

| | on release day | now, default | `--profile server` | `--profile agent` |
|---|---|---|---|---|
| 1 question, 80-token document | 33.5 ms | 26.8 ms | **21.9 ms** | 23.1 ms |
| 1 question, 1,052-token document | 148.0 ms | 90.8 ms | 68.5 ms | **25.6 ms**¹ |
| 32 questions, 80-token document | 286.9 ms | 194.5 ms | **166.4 ms** | 191.6 ms |
| 32 questions, 1,052-token document | 440.5 ms | 302.9 ms | 253.9 ms | **240.7 ms** |
| per decision, 32 questions, 80 tokens | 9.0 ms | 6.1 ms | **5.2 ms** | 6.0 ms |
| agent asks again about the same 1,052-token state | 148.6 ms | 92.6 ms | 68.7 ms | **25.7 ms** |
| agent state grows ~100 tokens a step | 142.6 ms | 88.8 ms | 65.7 ms | **52.6 ms** |

¹ A document it has already read, served from the cache.

In every configuration, each chosen option matches a reference that reads the whole document
for every question with no cache. For scale: TypeSafe reports 70–500 ms for its hosted Jev
(vendor-reported). Called from our machine over the internet, it took 164–188 ms on the six
examples on our showcase page.

## How it went

The agents worked in the same way the research loop trains models. They measured first, proposed
one change at a time, and kept a change only if it passed the checks at the end of this page.

**1. The kernels were missing.** Qwen3.5 mixes attention layers with Gated DeltaNet layers.
A plain install ran the DeltaNet layers on transformers' PyTorch reference code, and the only
sign of it was a warning in the log. Installing the fused kernels (fla) gave **1.25–1.63x**.
The release verification still agrees 1.0000 with the training record on all three sets.
`pip install "rsi-jev[fast]"` now installs them, and the server says at startup which kernels
it is actually using.

**2. A threshold tuned on another GPU was switching the cache off.** When a request asks
several questions about one document, the server reads the document once and continues each
question from that read, but only when that saves enough tokens. The cut-off had been tuned on
an A100, where one pass is cheap, and it applied whenever fla was installed. On the GB10 the
break-even point is much lower:

| tokens saved | 280 | 400 | 480 | 1,200 | 2,800 | 15,000 |
|---|---|---|---|---|---|---|
| reading once is | 0.75x | 1.06x | 1.28x | 1.97x | 3.84x | 8.77x |

The A100 threshold gave up **1.3–2x** on every request between 480 and 2,048 saved tokens. The
threshold now depends on the GPU.

**3. Looking at how others do it.** The agents read how other Jev-style projects serve their
models: an explicit prefix cache and CUDA graphs (kev), a merged-LoRA fast path with CUDA graphs
(imajev), one read of the state with every question as a short branch (OneJev), and vLLM or
SGLang with prefix caching (vllm-jev, openjev). Each idea below was tried here.

**4. The GPU was already busy.** Changing the batch size moved latency by at most ±15%, and
CUDA graphs made the GB10 slower. On this machine the time goes into real compute, not into
launching kernels. (On an H100 the same graphs brought one question to 12.1 ms; that path is not
released yet.) So the remaining gains had to come from doing less work.

**5. Doing less work: remember what was read.** Agents ask about the same state again and again,
often with a few lines added. The document cache keeps each read across requests, keyed on the
exact tokens and the weights. A repeated state skips the read, and a state that grew reads only
the new part. Asking again about the same state went from **148.6 to 25.7 ms**, and a state that
grows each step from **142.6 to 52.6 ms**. Every answer matches a fresh read.

**6. Compiling the model.** `torch.compile` of each layer and the scorer gave **1.1–1.3x**
more. It agrees with the uncompiled model on 99.9% / 99.9% / 99.0% of the verification
questions, and calibration moved by 0.0001. It costs about 40 seconds of compiling on the first
requests, so it is opt-in: `--profile server`.

**7. A bug found on the way.** While measuring, the agents found that transformers 5.17 stores
the DeltaNet states in dicts where 5.16 used lists. The cache copied only lists, so a copy shared
state with the original. Requests with more than 16 questions, and every document-cache hit,
continued from the wrong state; on a 32-question request, 3 answers changed. It is fixed, with a
regression test that fails without the fix.

## What did not ship

| tried | on the GB10 | why it did not ship |
|---|---|---|
| **FP8** weights and activations | slower (45 ms against 24 ms for one question) | converting activations to FP8 cost more than the faster matrix multiply saved, and 2–8% of verification answers changed, most on MMLU-Pro |
| **4-bit weights** (W4A16, NVFP4A16) | one short question 28 → 19 ms, no gain elsewhere | MMLU-Pro agreement fell to 0.77 and 0.69 |
| **8-bit weights** (W8A16) | slower than bf16 almost everywhere | MMLU-Pro agreement 0.977, below the bar |
| **FlashQLA** DeltaNet kernel | kernel 2.0–2.7x faster, end to end only 1–13% | that kernel is about 9% of the time; agreement 0.993–0.998, above what a kernel swap may change |
| **vLLM** backend | 2.2x the throughput for many clients asking single short questions; 3x faster offline evaluation | slower for multi-question requests: on this hybrid model it shares a document only in 544-token blocks. Kept as an experimental backend |

Quantization was the most tempting, because one pass on the GB10 is limited by reading the
weights. It only helped the one case that is limited that way, a single short question, and it
changed answers on knowledge-heavy questions first.

## Where the floor is

The tower's weights are about 2.75 GB in bf16. At the GB10's ~273 GB/s, reading them once takes
about 10 ms, and that is the floor for one pass on this machine. One question takes 22–27 ms;
32 questions at once share one read of the weights and cost 5–10 ms each. Going lower means
changing the model itself: early exit (being tested), distilling to a smaller tower, or training
with quantization in mind so that 4-bit weights pass the checks.

## The checks

A speed change ships only if it passes all three, against the unchanged model:

1. **Release verification:** answer agreement on the typed-decisions test set in both option
   orders and on MMLU-Pro 1k. Kernel swaps must agree about 1.000; other changes at least 0.99.
2. **Roundtrip:** 200 questions against the training run's own record, at least 0.99.
3. **Calibration:** error after the shipped calibration within ±0.003 on the benchmark suite.

## Thanks

Thanks to HP and NVIDIA for providing the HP ZGX Nano AI Station, powered by the NVIDIA GB10
Grace Blackwell Superchip. Every measurement here was made on it.

[`inference.md`](inference.md) is how to install and run it, and which setup to pick.
