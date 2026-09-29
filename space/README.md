---
title: RSI-Jev v3.0
emoji: 🧭
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Typed decisions (choice, noul, score) from one forward pass
---

# RSI-Jev v3.0 demo

A state, typed questions (`choice`, `noul`, `score`) in the Jev wire format, and
a probability bar for every option, with the latency of the forward pass. The
examples are the ones in `serve/README.md` and `serve/examples.json`.

It loads [shgao/rsi-jev-v3.0-qwen3.5-2b](https://huggingface.co/shgao/rsi-jev-v3.0-qwen3.5-2b)
with `scripts/load_release.py` and answers through `serve.wire` and `serve.infer`,
the same path as `scripts/serve.py`.

## Run it

From a checkout of the repo:

```bash
pip install -r requirements.txt gradio
python space/app.py                      # http://localhost:7860
```

| variable | default | |
|---|---|---|
| `RSIJEV_CKPT` | `shgao/rsi-jev-v3.0-qwen3.5-2b` | a repo id or a local release directory |
| `RSIJEV_DTYPE` | `bf16` | tower precision; `fp32`, or `auto` for bf16 on GPU and fp32 on CPU |
| `RSIJEV_THREADS` | torch default | CPU threads |
| `PORT`, `HOST` | `7860`, `0.0.0.0` | |

To make a Space directory, `bash space/assemble.sh OUT` copies this app with the
parts of the repo it imports (`rsijev/`, `serve/`, `scripts/load_release.py`).
Nothing is uploaded.

## Latency on CPU

Measured on the NVIDIA GB10's Arm CPU (not a Hugging Face machine), torch 2.13,
no fused linear-attention kernels, the `Customer service` example (333 prompt
tokens for one question, 876 for eight with the shared state encoded once).
Median of three after a warm-up:

| tower | threads | 1 question | 8 questions | peak memory at load |
|---|---|---|---|---|
| bf16 | 2 | 1.6 s | 4.4 s | 8.7 GB |
| fp32 | 2 | 2.8 s | 8.0 s | 16.9 GB |
| fp32 | 8 | 2.5 s | 7.1 s | 16.9 GB |
| bf16 on the GB10 GPU | – | 79 ms | 223 ms | |

More threads barely help: without the fused kernels, the Gated DeltaNet layers
run as a sequential PyTorch loop. On the 125-question MLX parity fixture, bf16 on
CPU picked the same answer as the fp32 reference on all 125 (largest probability
difference 0.014).

So a free CPU Space (2 vCPU, 16 GB) should work in bf16, at a few seconds per
request; fp32 does not fit its memory. The x86 vCPUs of a Space are not the CPU
measured here, so expect the numbers to move.

## If CPU is too slow

1. **ZeroGPU.** `app.py` wraps scoring in `spaces.GPU` when the `spaces` package
   is present, so the same app should run on a ZeroGPU Space (not tried yet).
   Each request waits for a GPU slot; the forward pass itself is tens of
   milliseconds.
2. **A smaller model.** `RSIJEV_CKPT=shgao/rsi-jev-v1.0-qwen3.5-0.8b` loads the
   0.8B release, about 2.5x fewer parameters, with v1.0's lower accuracy and no
   calibration.
