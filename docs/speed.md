# Serving RSI-Jev fast

*Technical report · RSI-Jev v3.0-2B · HP ZGX Nano (NVIDIA GB10)*

The same v3.0 checkpoint, on the same machine, now answers **1.25–1.63x faster by default**,
**up to 2.2x faster with `--profile server`**, and **up to 5.8x faster in agent loops with
`--profile agent`**. No chosen option changed. This report describes the three changes that
produced the gain, how each is implemented and verified, and the approaches that were measured
and rejected.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/speed_gb10_dark.svg">
  <img src="assets/speed_gb10_light.svg" alt="Milliseconds per request on an HP ZGX Nano for six workloads, v3.0 as first released against now: 1.5x to 5.8x faster with the profile that fits." width="100%">
</picture>

| workload | v3.0 as first released | now, default | `--profile server` | `--profile agent` |
|---|---|---|---|---|
| 1 question, 80-token document | 33.5 ms | 26.8 ms | **21.9 ms** | 23.1 ms |
| 1 question, 1,052-token document | 148.0 ms | 90.8 ms | **68.5 ms** | — |
| 32 questions, 80-token document | 286.9 ms | 194.5 ms | **166.4 ms** | 191.6 ms |
| 32 questions, 1,052-token document | 440.5 ms | 302.9 ms | **253.9 ms** | — |
| per decision, 32 questions, 80 tokens | 9.0 ms | 6.1 ms | **5.2 ms** | 6.0 ms |
| same 1,052-token state asked again | 148.6 ms | 92.6 ms | 68.7 ms | **25.7 ms** |
| state growing ~100 tokens per step | 142.6 ms | 88.8 ms | 65.7 ms | **52.6 ms** |

## 1. Setup

- **Hardware.** HP ZGX Nano AI Station (NVIDIA GB10, provided by HP and NVIDIA): a Blackwell GPU
  (sm_121) and a 20-core Arm CPU sharing 121 GB of unified memory at about 273 GB/s.
- **Software.** torch 2.14.0+cu130, transformers 5.17.0, flash-linear-attention 0.5.2.
- **Model.** RSI-Jev v3.0: a Qwen3.5-2B tower (24 layers, hidden size 2,048; 18 Gated DeltaNet
  layers and 6 full-attention layers, one every fourth layer) in bf16, an option-cross-attention
  scorer and a fitted calibration head in fp32. States are cut to 2,048 tokens.
- **Measurement.** Each configuration runs in its own process, alone on the GPU. Requests are
  4-option choice questions with distinct instructions. Each cell is the median of 20 in-process
  requests after 3 warm-up requests. The raw data is [`assets/bench_gb10.json`](assets/bench_gb10.json).

## 2. How a request is computed

For each question, the encoder builds one sequence: the state, then the question's instructions,
then its options. The tower runs over the sequence; the scorer reads the hidden states over each
option's span and at a decision token after the last option, and returns one logit per option.
The calibration head rescales the logits, and a softmax gives the probabilities.

A request with Q questions about a state of P tokens, with question suffixes of S tokens, costs
Q·(P + S) token-passes if each question is encoded separately. Since the state is a prefix of
every sequence and attention is causal, the state's contribution is identical across questions.
Each optimization below either removes repeated work on the state or makes the remaining work
cheaper.

## 3. Fused Gated DeltaNet kernels

**Mechanism.** Three quarters of the tower's layers are Gated DeltaNet (linear attention with a
gated delta-rule state update). Without flash-linear-attention installed, transformers runs these
layers through its PyTorch reference implementation, which expresses the chunked recurrence as a
sequence of small tensor operations. With it installed, the same recurrence runs as fused Triton
kernels: a chunked kernel for prefill, and a fused recurrent kernel for continuing from a cached
state.

**Result.** 1.25–1.63x across the grid (the default column against the first).

**Correctness.** The release verification, run in fp32 with the kernels active, agrees
1.0000 / 1.0000 / 1.0000 with the training run's per-question record (typed-decisions in both
option orders, MMLU-Pro 1k).

## 4. Reading the state once per request

**Mechanism.** The server encodes the state once, keeps its cache, and continues each question's
suffix from that cache. The cache is hybrid: the full-attention layers hold keys and values, and
the DeltaNet layers hold a convolution state and a recurrent state. Continuing from it is exact
under the causal mask. The suffixes are batched: the cache is replicated across the batch rows,
and position ids continue from the end of the state.

**Cost model.** Separate encoding costs Q·(P + S). Reading once costs P + Q·S plus a fixed cost:
a second sequential pass and the replication of the cache. It pays when the saved work,
(Q − 1)·P tokens, exceeds that fixed cost, and the break-even point depends on how expensive one
tower pass is on the machine. Measured on the GB10 with the fused kernels (bf16, reading once
against separate encoding):

| tokens saved, (Q − 1)·P | 40 | 280 | 400 | 480 | 1,200 | 2,800 | 15,000 |
|---|---|---|---|---|---|---|---|
| speed-up from reading once | 0.68x | 0.75x | 1.06x | 1.28x | 1.97x | 3.84x | 8.77x |

On an A100 one pass is cheap enough that reading once lost below about 2,048 saved tokens, and
that threshold had been applied whenever the fused kernels were installed. The server now picks
the threshold by device: 2,048 on A100- and H100-class GPUs, 480 elsewhere. On the GB10 the A100
threshold had been giving up 1.3–2x on every request between 480 and 2,048 saved tokens.

**Correctness.** Continuing from the cache equals encoding each question separately: every
argmax equal, probabilities within 1.3e-6 in fp32. One detail matters for the replication: from
transformers 5.17 the DeltaNet states are held in dicts rather than lists, and a replica must copy
both, or reordering the replica rewrites the original cache. A regression test checks that
replication never modifies the original.

## 5. A cache of states across requests

**Mechanism.** An agent asks about the same state repeatedly, and often appends to it between
calls. `--profile agent` keeps the cache from section 4 across requests, in an LRU keyed on the
exact state token ids and a fingerprint of the weights, never on raw text.

- **The same state again** skips the state pass entirely.
- **A state that extends a cached one** runs only the new tokens, continuing a copy of the cached
  DeltaNet and attention states, and stores the result under the new key.
- **Token boundaries.** Re-tokenizing a longer text can merge tokens across the old end; a JSON
  transcript whose closing `}]` becomes `},{` is the common case. The last 8 tokens of each state
  are therefore not cached but re-read with each question, and a state whose token ids do not
  extend a cached entry is read in full.
- **Memory.** The cache is bounded by entry count and bytes (`RSIJEV_DOC_CACHE_ENTRIES`,
  default 32; `RSIJEV_DOC_CACHE_MB`, default 2,048). Entries are never modified in place.

**Result.** The same 1,052-token state asked again drops from 148.6 ms to **25.7 ms**; a state
growing by about 100 tokens per step (188 to 1,804 tokens over 34 steps) from 142.6 ms to
**52.6 ms** per step. The first request for a state pays one extra pass (49 ms instead of 31 ms
for a 260-token state), so the cache is opt-in.

**Correctness.** Repeated and extended states are tested against a fresh read: every argmax
equal, |dp| at most 1.6e-6 in fp32 on CPU and 4.6e-4 on the GPU with the fused kernels.

## 6. Compiling the tower

**Mechanism.** `--profile server` applies `torch.compile` to each decoder layer and to the
scorer, with dynamic shapes and a raised recompile limit, so one compiled code object serves all
layers. CUDA-graph modes are refused (section 7).

**Result.** 1.1–1.3x over the default (the `--profile server` column against the default). The
first requests compile for about 40 s, and a new request shape can recompile.

**Correctness.** Agreement with the uncompiled bf16 model 0.9985 / 0.999 / 0.990 on the
verification sets, roundtrip 0.995 (as uncompiled), suite calibration error +0.0001.

## 7. Measured and rejected

| approach | result on the GB10 | reason |
|---|---|---|
| FP8 weights and activations (torchao) | eager: slower (45 against 24 ms, one question), as quantizing activations on the fly costs more than the FP8 matmul saves; compiled: up to 1.25x faster than compiling alone on long documents | 2–8% of verification answers changed, most on MMLU-Pro; roundtrip 0.975–0.985 |
| 4-bit weights, W4A16 and NVFP4A16 (vLLM, Marlin kernels) | one short question 28 → 19 ms; no gain on multi-question or long requests | MMLU-Pro agreement 0.77 and 0.69 |
| 8-bit weights, W8A16 (Marlin) | slower than bf16 except one short question | MMLU-Pro agreement 0.977, below the 0.99 bar |
| FlashQLA DeltaNet kernel | kernel 2.0–2.7x faster than fla; end to end 1–5% (5–13% compiled) | the kernel is about 9% of tower time; agreement 0.993–0.998, above what a kernel replacement may change |
| CUDA graphs | slower than eager | the batched workload is compute-bound on this GPU: batch size 16 → 64 changed latency by at most ±15% |
| vLLM backend (pooling runner, our scorer on top) | 84 against 34–37 requests/s for concurrent single short questions; 3x faster offline scoring | slower for multi-question requests (8 questions, 1,052 tokens: 2.9 against 7–8 requests/s), because on this hybrid model its prefix cache shares only whole 544-token blocks. Passes the checks; kept as an experimental backend |

Weight quantization helps only where one pass is bound by reading weights, which here is a single
short question; elsewhere the workload is compute-bound. It also changed answers on
knowledge-heavy questions first.

## 8. Limits

The tower's non-embedding weights are 2.75 GB in bf16. At about 273 GB/s, reading them once takes
about 10 ms, which bounds a single pass on this machine; one question measures 22–27 ms. Batched
requests share one read of the weights and cost 5–10 ms per decision, where compute dominates.
Going below these floors requires changing the model: fewer layers per pass (early exit),
distillation to a smaller tower, or quantization-aware training so that low-bit weights pass the
checks in section 9.

## 9. Verification

Every change is compared with the unchanged model and ships only if it passes:

1. **Release verification.** Argmax agreement on the typed-decisions test set in both option
   orders and on MMLU-Pro 1k: about 1.000 for a kernel replacement or a cache, at least 0.99 per
   set otherwise.
2. **Roundtrip.** 200 questions against the training run's per-question record: at least 0.99.
3. **Calibration.** Suite calibration error after the shipped calibration within ±0.003.

[`inference.md`](inference.md) covers installation, the profiles and which one to use.
