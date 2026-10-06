# Running RSI-Jev

On a desk-side HP ZGX Nano (NVIDIA GB10), RSI-Jev-v3.0-2B answers a one-question request in
22–27 ms and costs 5–10 ms per decision when 32 are asked together. There is no per-call
cost, and the documents never leave the machine.

For comparison, TypeSafe reports 70–500 ms end to end for its hosted Jev (vendor-reported).
Every figure in this guide was measured on the machine the client ran on, so none of them
includes a network hop.

## How much faster

[`speed.md`](speed.md) explains where the time goes and why these changes work.

We ran one benchmark in one session on the GB10. Every configuration used the same
checkpoint, documents and questions, and each ran in its own process, one at a time. Each
cell is the p50 of 20 runs after 3 warm-up runs, in milliseconds, with 4-option choice
questions.

| | A. v3.0 as released | B. now, default | C. `--profile server` | D. `--profile agent` |
|---|---|---|---|---|
| 80-token doc, 1 question | 33.5 | 26.8 (1.25x) | 21.9 (1.53x) | 23.1 (1.45x) |
| 80-token doc, 8 questions | 92.0 | 68.8 (1.34x) | 62.0 (1.49x) | 48.2 (1.91x) |
| 80-token doc, 32 questions | 286.9 | 194.5 (1.48x) | 166.4 (1.72x) | 191.6 (1.50x) |
| 1,052-token doc, 1 question | 148.0 | 90.8 (1.63x) | 68.5 (2.16x) | 25.6 (5.77x) |
| 1,052-token doc, 8 questions | 212.9 | 141.4 (1.51x) | 114.6 (1.86x) | 62.7 (3.40x) |
| 1,052-token doc, 32 questions | 440.5 | 302.9 (1.45x) | 253.9 (1.73x) | 240.7 (1.83x) |
| per decision, 32 questions, 80 / 1,052 tokens | 9.0 / 13.8 | 6.1 / 9.5 | 5.2 / 7.9 | 6.0 / 7.5 |
| agent loop: same state again, 1 / 4 questions, 1,052 tokens | 148.6 / 184.7 | 92.6 / 124.3 | 68.7 / 97.3 | **25.7 / 41.0** |
| agent loop: state growing +101 tokens a step (188 → 1,804), 1 / 4 questions | 142.6 / 171.8 | 88.8 / 119.0 | 65.7 / 94.5 | **52.6 / 65.8** |

**Compared with v3.0 on release day, the same model on the same machine answers 1.25–1.63x
faster by default, 1.5–2.2x faster with `--profile server`, and 1.45–5.8x faster in agent
loops with `--profile agent`.** In every configuration, each chosen option matches a
reference that reads the whole state for every question with no cache, and probabilities
stay within 0.022 of it.

- **A** is the public code at release (`1767f6e`), in a fresh venv from `requirements.txt`
  with no fla. That is what a user got on release day.
- **B** is the current code installed with `rsi-jev[fast]`.
- **C** is B with `RSIJEV_COMPILE=1`, and **D** is B with `RSIJEV_DOC_CACHE=1`. Every cell
  asks about the same document 23 times, so in D all but the first are cache hits: D's
  grid rows are the "same state again" case, not a new document.
- All four ran on torch 2.14.0+cu130 and transformers 5.17.0, with a bf16 tower.
- The raw results are in [`assets/bench_gb10.json`](assets/bench_gb10.json), and the script
  is [`assets/bench_gb10.py`](assets/bench_gb10.py).

One correction came out of this run. Under transformers 5.17, the release-day code answered
questions 17–32 of a request from the wrong state: it read the document once, and the
second batch of 16 continued from a cache that the first batch had already written into.
In A's 1,052-token, 32-question cell, 3 of 32 answers differ from the reference. The
document cache had the same fault on every hit. Both are fixed now, and
`tests/test_prefix_cache.py` covers it.

## Quick start

```bash
pip install "rsi-jev[fast,vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev"
rsi-jev serve shgao/rsi-jev-v6.0-vl-5b --port 8000
curl localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": [{"role": "user", "content": "I was charged twice. Please refund."}],
  "questions": {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}}
}'
# {"model":"jev-latest","answers":{"refund":{"type":"noul","noul":0.9948}},"usage":{"input_tokens":42,"output_tokens":1}}
```

The first run downloads the checkpoint and its base model into the standard
Hugging Face cache. You can name the model three ways:

- a Hugging Face id, such as `shgao/rsi-jev-v5.0-vl-3b`;
- an alias: `v6.0-vl-5b`, `v5.0-vl-3b`, `v4.0-vl-2b`, `v3.0-2b`, `v2.1-2b`, `v2.0-2b`, `v1.0-2b` or `v1.0-0.8b`;
- a local directory.

`[fast]` installs fla (`flash-linear-attention` and `fla-core` 0.5.x). It only does
anything on CUDA, and it is harmless elsewhere. `causal-conv1d` is optional: it needs an
nvcc build and made little difference here.

The server says what it is actually running before it answers anything:

```
warm-up: 1.1 s
runtime: torch 2.14.0+cu130, cuda (NVIDIA GB10, sm_121), tower bf16, scorer fp32
kernels: fla 0.5.2 active; causal_conv1d not installed (optional)
document read once per request when (questions - 1) x state tokens >= 480; document cache off; compile off
serving shgao/rsi-jev-v3.0-qwen3.5-2b as 'rsi-jev-v3.0-qwen3.5-2b' (alias 'jev-latest') on 127.0.0.1:8000; ...
```

If fla is missing on a CUDA machine, the server prints one more line, with the command that
installs it. `rsi-jev env` prints the same report without loading a model.

The same thing from Python, with no server:

```python
from rsijev import Decider
d = Decider("shgao/rsi-jev-v5.0-vl-3b")
d.decide("I was charged twice. Please refund.",
         {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}})
```

`Decider` picks the device and precision the way the server does. It validates the request
with the server's schema and runs the server's request path, so `decide` returns exactly
the `answers` object of `POST /v1/systemone` (`tests/test_easy_infer.py`).

From a clone, without installing, run
`pip install -r requirements.txt pillow torchvision && python scripts/serve.py --ckpt shgao/rsi-jev-v4.0-vl-qwen3.5-2b`.

## Effort levels (v6.0-VL)

What each level buys, by benchmark and by question: [effort-depth.md](effort-depth.md).

A multi-exit release answers at more than one depth. `effort` picks the depth per request
(`"effort"` in the `POST /v1/systemone` body) or as the server default (`--effort`, or
`RSIJEV_EFFORT`); a request's own effort wins. It uses the heads the release ships, no other
weights:

v6.0-VL is a System One model: one forward pass, no generated reasoning, a probability for every option. `effort` sets how many layers that one pass uses (16, 20 or 32), not how many tokens it writes; it writes none.

| effort | depth (v6.0-VL) | Decision Index 0.2.1 | median |
|---|---|---:|---:|
| `low` | the shallowest aux exit: layer 16 | 43.3 | 23 ms |
| `medium` | the deepest aux exit: layer 20 | 45.9 | 27 ms |
| `high` | the main exit, all layers: layer 32 | 45.9 | 40 ms |
| `auto` | the release's confidence cascade (16 → 20 → 32, stop at the first exit whose calibrated top-1 probability reaches that exit's threshold: 0.95 at 16 and 0.59 at 20 by default on v6.0-VL), every text request | 45.8 | – |
| unset | the default below | 45.7 | 40 ms |

Any other value is a 422 that lists these four.

Decision Index figures are reads of the kit's stratified 16,000-request sample (one H200, bf16,
one request at a time), not full runs; v5.0-VL reads 37.4 on the same sample.
`auto` was read on a different GPU; at 22 layers on average it sits between `medium` and `high`.

```bash
rsi-jev serve v6.0-vl-5b --effort medium            # server default
curl -s localhost:8000/v1/systemone -d '{"model":"jev-latest","effort":"low", ...}'
```

With no effort anywhere, serving is what it was before `effort` existed: `--adaptive auto`
(the cascade for multi-question requests, the main exit for a single question), or whatever
`--adaptive` / `meta.json` `adaptive.serving` says. `--effort` sets that default itself, so it
is refused together with `--adaptive` or `--fixed-exit`. Responses to a request that ran with an
effort carry `usage.effort` (the effort it actually ran at) next to `usage.depth` (the layer
each question was answered at).

Image requests run at full depth whatever the effort, because the aux heads read text only;
their `usage.effort` is `high`. A release with a single exit serves `high` (or no effort) as
before and answers `low`, `medium` and `auto` with a 422, as it does an unknown value.

**Confidence.** Every response from a multi-exit release carries `usage.confidence`: for each
question, the calibrated probability of its top option at the exit that answered it (the
value the cascade compares with the threshold), next to `usage.depth`, at every effort and on
image requests (layer 32, the main head's calibration). A single-exit release's responses keep
their shape.

**Threshold.** `"confidence_threshold"` (a number in (0, 1]) replaces the release's tau (0.59 on
v6.0-VL) for one request: a question stops at the first aux exit where its calibrated top-1
probability is at least the threshold, the same threshold at every aux exit. It applies to
effort `auto` and to the default adaptive path (multi-question requests); with `low`,
`medium` or `high`, or outside the range, it is a 422. Lower thresholds stop earlier (any
value below 1/K, the uniform probability, answers every question at the first aux exit, as
`low` does). `1.0` means never stop early: the request runs at full depth and reports
effort `high`; this also covers a single-option question, whose confidence is exactly 1.

With effort `auto` the threshold can also be set per aux exit, as an object:
`"confidence_threshold": {"16": 0.8, "20": 0.6}`. An exit left out keeps effort auto's default
there; `1.0` at an exit means never stop at that exit; an exit the release does not have, or a
value outside (0, 1], is a 422. The object form is refused without effort `auto`. Effort auto's
default is the release's `meta.json` `adaptive.auto_thresholds` (`{"16": ..., "20": ...}`) when
it has one, else the single tau; v6.0-VL ships `{"16": 0.95, "20": 0.59}`.
The unset default (no effort) always uses the single tau and ignores `auto_thresholds`, so the
numbers above hold.

Choosing a threshold on v6.0-VL (dev sets, descriptive, not used for any selection; quality is the
suite-like score, held-out accuracy and Decision-Index-style skill):

| setting | layers | stops at 16 / 20 / 32 | suite-like | held-out | Decision-Index-style |
|---|---:|---|---:|---:|---:|
| `low` (16) | 16.0 | 100 / 0 / 0 % | 0.752 | 0.697 | 0.521 |
| `medium` (20) | 20.0 | 0 / 100 / 0 % | 0.760 | 0.703 | 0.550 |
| `high` (32) | 32.0 | 0 / 0 / 100 % | 0.765 | 0.697 | 0.555 |
| threshold 0.50 | 19.4 | 76 / 4 / 20 % | 0.758 | 0.705 | 0.532 |
| **threshold 0.59** (default) | 20.9 | 65 / 5 / 30 % | 0.765 | 0.705 | 0.535 |
| threshold 0.70 | 22.9 | 52 / 6 / 42 % | 0.766 | 0.698 | 0.542 |
| threshold 0.80 | 24.8 | 40 / 6 / 54 % | 0.765 | 0.697 | 0.544 |
| threshold 0.90 | 26.9 | 29 / 5 / 67 % | 0.765 | 0.697 | 0.551 |
| threshold 0.95 | 28.3 | 21 / 4 / 76 % | 0.765 | 0.697 | 0.555 |

Layers and stop shares are on the suite-like set. From 0.59 up, suite-like and held-out text match
`high`; on inputs unlike the training data a low threshold costs quality, which comes back only from
about 0.9. If inputs may be unusual, use a threshold of 0.9 or more, or `medium`.

### Images

A release trained with images (v4.0-VL on) also answers questions about 1–4 images. It needs
the `vision` extra (Pillow and torchvision), which the quick start installs. Send the images
as data URLs and mark where each one goes in the state with `<image>`:

```python
from rsijev import Decider
d = Decider("shgao/rsi-jev-v5.0-vl-3b")
d.decide("Customer photo: <image>\nThe customer says it arrived damaged.",
         {"damaged": {"type": "noul", "instructions": "Does the photo show visible damage?"}},
         images=["photo.jpg"])          # a path, bytes, a PIL image or a data URL
```

Over HTTP the same request carries `"images": ["data:image/jpeg;base64,..."]` next to
`state`. Full limits and an example: [`serve/README.md`](../serve/README.md#images). Larger worked
examples, each with its expected output: [`examples/gallery/`](../examples/gallery/) (227 questions
through `Decider`) and [`examples/breakout/`](../examples/breakout/) (a game played over HTTP). In short:

- 1–4 PNG, JPEG or WebP images per request, as data URLs; http(s) URLs are not fetched.
- 1,024 image tokens per question, split evenly: one image up to about one megapixel, four up
  to 256 tokens each. Larger images are scaled down.
- Up to 32,768 text tokens on top (`RSIJEV_MAX_INPUT_TOKENS`). Nothing is cut: a longer
  request is refused with a 422.
- The vision tower runs once per request, and the state with its image tokens is read once:
  every question continues from that reading, as it does for text. With `--profile agent`, a
  repeated image state (same image bytes, same text) is not read again at all.

Measured on the GB10 with v4.0-VL through the server's request path (bf16 tower, fla; median of
20 requests after warm-up, no other job on the GPU):

| request | before (each question read the image) | now | same image again, `--profile agent` |
|---|---|---|---|
| one 640 px photo, 1 question | 79 ms | 80 ms | 36 ms |
| one 640 px photo, 4 questions | 172 ms | 110 ms | 50 ms |
| one 1,600 px photo, 1 question | 245 ms | 245 ms | 82 ms |
| one 1,600 px photo, 4 questions | 561 ms | 273 ms | 101 ms |

Reading once changes no answer: on 230 image questions every top answer matches the
per-question path, and the probabilities differ by no more than bf16 rounding already moves
them. An image costs roughly what the same number of text tokens would, and the photo's size
decides that number.

### v5.0-VL 3B: a cut model

v5.0-VL runs the first 20 of Qwen3.5-4B-Base's 32 decoder layers and reads its decision there;
the deeper layers are neither downloaded nor built. The checkpoint is self-contained: tokenizer,
embeddings and vision tower ship with it, so `rsi-jev serve v5.0-vl-3b` downloads 11 GB and
nothing from the base model. Older releases still take their embeddings and vision tower from
the base, as before.

Measured on the GB10 (bf16 tower, fla; p50 over warm requests):

| questions per request | 1 | 8 | 32 |
|---|---|---|---|
| v5.0-VL 3B | 43.7 ms | 124.6 ms | 338.2 ms |

That is about 1.45× v4.0-VL on the GB10, which is compute-bound; on an H100 with CUDA graphs a
4B model cut at layer 20 runs within 0–13% of the 2B.

### Multi-exit checkpoints: adaptive exit

A checkpoint with readout heads at several depths (`spec.arch_extra.aux_exits`, e.g. 12 and
16 next to `exit_layer` 20; the extra heads in `aux_scorers.safetensors`) can stop each
question at the first exit whose calibrated top-1 probability reaches a threshold tuned on
DEV (`meta.json` `adaptive.tau`); the last exit always answers. Each exit has its own
calibration. With scalar temperatures, `calibration.json` holds the main head's and one per
aux exit:

```json
{"cal_mode": "temp",
 "exits": {"12": {"cal_mode": "temp", "logT": 0.83},
           "16": {"cal_mode": "temp", "logT": 0.88}}}
```

(`cal_logT` for the main head in `calibration.safetensors`.) A per-exit cal-4b in
`adaptive_calibration.safetensors` also loads; a checkpoint carries one or the other.
`--adaptive auto` (the default) uses adaptive exit for multi-question requests,
`--adaptive on` for every text request, `--fixed-exit` never; responses report the depth each
question used in `usage.depth`. A question answered at a shallow exit gets exactly what a
model built at that exit would return.

## A local, drop-in Jev replacement

Code written against Jev only needs its base URL changed. We ran each snippet below against
`rsi-jev serve` on the GB10, using `typesafe-sdk` 0.7.2, `@typesafe-ai/sdk` 0.6.0 and
`pydantic-ai-slim` 2.52.0. These are TypeSafe's own client libraries and the Pydantic AI
integration. This project is not affiliated with TypeSafe.

The clients always send an API key. If the server has no `--api-key`, any non-empty string
works.

```python
# pip install typesafe-sdk          (or set TYPESAFE_BASE_URL=http://localhost:8000)
from typesafe_sdk import TypeSafeClient
client = TypeSafeClient(base_url="http://localhost:8000", api_key="local")
r = client.system_one("I was charged twice. Please refund.",
                      {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}})
```

```ts
// npm install @typesafe-ai/sdk     (or set TYPESAFE_BASE_URL)
import { TypeSafeClient } from "@typesafe-ai/sdk";
const client = new TypeSafeClient({ baseURL: "http://localhost:8000", apiKey: "local" });
```

```python
# pip install "pydantic-ai-slim[typesafe]"
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
model = TypeSafeModel("jev-latest", provider=TypeSafeProvider(api_key="local", base_url="http://localhost:8000"))
Agent(model, output_type=bool, instructions="Does the user request a refund?").run_sync("I was charged twice.")
```

| | Jev (reference) | this server |
|---|---|---|
| routes | `POST /v1/systemone`, `GET /v1/models`, `GET /v1/limits`, `GET /health`, `GET /health/live` | the same |
| question types | `noul`, `choice`, `score`, with their `criteria` shapes | the same, and the same answer shapes |
| questions per request | 1–64 | 1–64 |
| options per question | 2–64 | 2–5,120 (`RSIJEV_MAX_ANSWERS`) |
| structured criteria (an object instead of a string) | accepted | accepted, read as compact JSON |
| `model` | `jev-latest` or a version | `jev-latest`, the served name, or any name given with `--accept-model`; the answer names this server |
| input length | — | up to 32,768 tokens per question (`RSIJEV_MAX_INPUT_TOKENS`); longer is a 422, never cut |
| option keys | hidden from the model | part of the prompt, so renaming a key can move the answer (`/v1/limits` reports `option_keys_visible_to_model: true`) |
| prompt | chat template, label logprobs | the encoder the model was trained with; a chat state is serialized as compact JSON |
| `usage.output_tokens` | N + 1 | N, one readout per question |
| `usage.input_tokens` | — | the tokens read; a state read once per request counts once |
| `confidence` | (K·p_max − 1)/(K − 1) | the same |
| calibration | — | `/v1/limits` names the fitted calibration a release ships |
| auth | bearer key | optional, `--api-key` or `RSIJEV_API_KEY` |

[`serve/README.md`](../serve/README.md) has the reasons behind each difference.

## Pick a setup

All latencies are p50 on the GB10, v3.0-2B with a bf16 tower. The first three rows come from
the benchmark above.

| scenario | how | measured |
|---|---|---|
| one request at a time | `rsi-jev serve <id>` | 1 question: 27 ms (80-token doc), 91 ms (1,052 tokens). 32 questions: 195 / 303 ms |
| an agent asking again about the same or a growing state | `--profile agent` | same state again: 26 ms. Growing by 101 tokens a step: 53 ms per step, 1 question |
| a long-running server | `--profile server` | 32 questions on 1,052 tokens: 254 ms. Warm-up at startup: 60 s |
| many concurrent clients, one short question each | `--batch-window-ms 0` | 72 requests/s at 8 and 32 clients, against 32–35 without it (vLLM backend, experimental: 84) |
| questions of very different lengths in one request | `RSIJEV_SORT_ROWS=1 RSIJEV_TRIM_OPTIONS=1` | 32 questions with 2–40 options: 1.7–1.8x. 32 uniform questions: 1.08–1.13x |
| Python, no server | `Decider(...)` | the server's numbers minus HTTP |
| CPU only | `--device cpu` (fp32) | on the GB10's Arm CPU: 1 question 733 ms, 3 questions 1.5 s, 32 questions on 1,052 tokens 16.7 s |
| Apple Silicon | torch on MPS or CPU | not measured; an MLX port is in progress and not released |

Environment variables win over a profile: `RSIJEV_DOC_CACHE=0 rsi-jev serve ... --profile
agent` runs without the document cache. On an H100, CUDA graphs on an unreleased internal
branch measured 12.1 ms for one question. That code is not public.

## On the HP ZGX Nano (NVIDIA GB10)

Thanks to HP and NVIDIA for providing the HP ZGX Nano AI Station, powered by the NVIDIA GB10
Grace Blackwell Superchip. Everything in this guide was developed and measured on it.

### What this gives you

- **Private and offline.** Documents and customer data stay on the desk. After the first
  download it runs with no network (set `HF_HUB_OFFLINE=1`, or point it at a local
  directory). There are no per-call fees and no rate limits.
- **Fast enough to sit inside an app or an agent loop.**
  - One request with one question takes 22–27 ms.
  - Asked together, 32 decisions cost 5–10 ms each: 32 questions on an 80-token document
    take 166 ms with `--profile server`, about 190 decisions per second.
  - An agent asking again about the same state waits 26 ms; one whose state grows each step
    waits 53 ms per step (`--profile agent`).
  - One server answers 34–39 single-question requests per second.
- **Room to spare.** A serving process takes about 7 GB of the 121 GB of unified memory the
  OS sees (8.7 GB with `--profile server`). A larger local LLM or agent can run beside it,
  with RSI-Jev as the fast decision layer.
- **A tested install path on aarch64 and Blackwell.** The commands in this guide were run
  on this machine: the PyPI torch wheel (2.14.0+cu130) and fla 0.5.2, with no source build
  and no container.

### Latency by document length and question count

The four configurations are compared in [How much faster](#how-much-faster). `rsi-jev bench`
separates the fixed cost of reading the document from the marginal cost of each decision,
by fitting total time against question count (1–32). These numbers are from the same
machine and a fresh venv, with the default configuration and fla:

| document | read once | per extra decision | 32 asked together | 32 asked one at a time |
|---|---|---|---|---|
| 80 tokens | 24 ms | 5.5 ms | 200 ms | 826 ms |
| 404 tokens | 47 ms | 5.7 ms | 232 ms | 1,321 ms |
| 1,052 tokens | 91 ms | 7.0 ms | 315 ms | 2,900 ms |

With `--profile server`, reading once costs 21 / 36 / 68 ms and each extra decision
4.5–5.6 ms. Without fla, reading once costs 32 / 69 / 142 ms and each extra decision
8.1–9.6 ms.

### Agent loops

An agent often asks about the same transcript again, or about one that has grown. With
`--profile agent`, the server keeps each document's read across requests, keyed on the
exact token ids and the weights. A repeated state skips the read, and a longer state reads
only its new tail. From the benchmark above:

| | default | `--profile agent` |
|---|---|---|
| same 1,052-token state again, 1 / 4 questions | 92.6 / 124.3 ms | 25.7 / 41.0 ms |
| state growing 101 tokens a step, 1 / 4 questions | 88.8 / 119.0 ms | 52.6 / 65.8 ms |

The first read of a new state costs one extra pass: 49 ms instead of 31 ms for one
question, as measured when the cache was built. The cache holds 32 entries and 2 GB by
default (`RSIJEV_DOC_CACHE_ENTRIES`, `RSIJEV_DOC_CACHE_MB`).

### Concurrency

By default the server runs one request at a time, so throughput stays flat as clients are
added. With `--batch-window-ms 0`, the single questions of requests waiting at the same time
share a forward pass (up to 1,280 padded tokens), and 1-question requests on an 80-token document
go from 32 to 72 req/s at 8 and 32 clients. Long documents and multi-question requests are not
batched, because on the GB10 they gain nothing from it. It is opt-in because batching moves
probabilities slightly: agreement with unbatched serving is 1.000 / 1.000 / 0.990, and ECE
moves +0.0016.
`pip install "rsi-jev[http]"` adds orjson, uvloop and httptools, which the server uses when they
are present.

These figures come from an HTTP benchmark in an earlier session, with closed-loop clients:

| request | 1 client | 8 clients | 32 clients |
|---|---|---|---|
| 1 question, 80-token doc | 34.5 req/s | 34.4 req/s | 34.4 req/s |
| the same, `RSIJEV_COMPILE=1` | 32.5 | 38.9 | 36.8 |
| 8 questions, 1,052-token doc | 6.9 | 7.0 | 7.0 |
| the same, `RSIJEV_COMPILE=1` | 8.2 | 8.4 | 8.4 |

### Memory

We measured system memory in use before and after starting `rsi-jev serve` with v3.0-2B:

| | default | `--profile server` |
|---|---|---|
| after startup and warm-up | +6.3 GB | +7.4 GB |
| after a 32-question request on a 1,052-token document | +7.0 GB | +8.7 GB |

### The fused kernels and the cache threshold

- **fla.** Without fla, the DeltaNet layers run transformers' torch reference. With fla
  0.5.2, the benchmark above measured 1.25–1.63x (B against A). The release verify agrees
  1.0000 with the training record on all three sets.
- **Warm-up.** The first request after a fresh install compiled fla's Triton kernels and
  took 19 s. The server now runs two warm-up requests before it listens; they take 1.1 s
  once the kernels are cached.
- **The cache threshold.** A request reads its document once when that saves enough
  tokens; otherwise it reads the document once per question. Where that pays depends on
  the machine. On an A100 with fla, one tower pass is cheap, and a 98-token document read
  once was 0.55x, a loss. So the threshold there is 2,048 saved tokens. On the GB10 with
  fla, reading once measured 0.75x at 280 saved tokens, 1.06x at 400, 1.28x at 480 and
  1.97x at 1,200. The A100 threshold would have given up 1.3–2x on every request between
  480 and 2,048 saved tokens. So the server uses 480 unless it is on an A100- or
  H100-class GPU with fla. `RSIJEV_MIN_SAVED_TOKENS` overrides it.

### What we learned about the chip

- **sm_121.** The PyPI torch wheel (2.14.0+cu130) and fla's Triton kernels run on it as
  they are. FlashQLA needed three runtime patches: its version map had no 12.1, it had a
  missing import on the sm_120 build, and one kernel needed 110 KB of shared memory per
  block where the GB10 allows 99 KB.
- **Unified memory.** CPU and GPU share one pool. A second model process takes memory from
  the first, and parallel model processes have crashed this machine. We run one model
  process at a time. vLLM needs `gpu_memory_utilization` capped (0.2 here, with a fixed
  8 GiB KV pool); its default would claim most of the machine.
- **Bandwidth sets a floor.** The tower's non-embedding weights are about 2.75 GB in bf16.
  At about 273 GB/s, reading them once takes about 10 ms, which is the floor for one pass.
  One question measures 22–27 ms. In a batch, the weights are read once for all the
  questions, so each decision costs 5–10 ms.
- **Compute-bound at batch.** Changing the batch size moved latency by at most ±15%.
  CUDA graphs (`torch.compile` `reduce-overhead`) made this machine slower, and the server
  refuses that mode.
- **4-bit runs through Marlin.** vLLM reports no native FP4 on this GPU. W4A16, NVFP4A16 and
  W8A16 all ran as weight-only Marlin kernels.

## What we tried to make it faster

A speed option ships only if it passes three checks: release verify agreement, the
roundtrip, and suite ECE within ±0.003 (see [the last section](#checks-every-speed-change-is-held-to)).

| option | speed on the GB10 | exactness evidence | verdict |
|---|---|---|---|
| **fla** fused DeltaNet kernels | 1.25–1.63x (benchmark above) | release verify agreement 1.0000 / 1.0000 / 1.0000 with the training record | **shipped**, `[fast]` |
| **device-aware cache threshold** | up to 2x between 480 and 2,048 saved tokens | reading once is exact under a causal mask: every answer equal, probabilities within 1.3e-6 in fp32 | **shipped**, default |
| **document cache** | same state again: 3.0–3.6x faster than B on 1,052 tokens, 1.2–1.5x on 80; growing state 1.7–1.8x | every argmax equal to a fresh read; \|dp\| 1.6e-6 on CPU, 4.6e-4 on GPU | **shipped**, `--profile agent` |
| **torch.compile** | 1.1–1.3x faster than B (benchmark above); 60 s warm-up | agreement 0.9985 / 0.999 / 0.990 with bf16, roundtrip 0.995, ECE +0.0001 | **shipped**, `--profile server` |
| **bf16 tower** (scorer stays fp32) | 3.4–6.2x against fp32 (measured on an A100 slice) | 0.4–1.6% of single predictions flip; pooled top-1 moves ≤ 0.003 | **shipped**, default on CUDA; evaluation stays fp32 |
| **one-pass tokenization, one GPU worker, orjson** | 32 questions on 1,052 tokens: 309 → 277 ms; tokenization 29 → 2.8 ms | bit-identical on all 12,389 verify, roundtrip and suite answers | **shipped**, default |
| **micro-batching** | 1 short question, 8–32 clients: 32 → 72 req/s | agreement 1.000 / 1.000 / 0.990 with unbatched serving, roundtrip 0.990, ECE +0.0016 | passes; opt-in, `--batch-window-ms` |
| **rows sorted by length, option slots trimmed** | mixed-length 32 questions: 1.7–1.8x; uniform: 1.08–1.13x | agreement 1.000 everywhere, roundtrip 0.990, ECE −0.00004 / ±0 | passes; opt-in, `RSIJEV_SORT_ROWS`, `RSIJEV_TRIM_OPTIONS` |
| **CUDA graphs** | slower on the GB10. On an H100, 12.1 ms for one question on an unreleased branch | — | not on this machine; H100 path not released |
| **batch size** | ±15% at most | — | unchanged (16): the GPU is compute-bound |
| **FP8** weights and activations | slower eager: 45 ms against 24 ms for one question | agreement 0.9795 / 0.9755 / 0.921, roundtrip 0.985 (0.975 compiled) | **not shipped**: fails |
| **W4A16** (GPTQ int4, through vLLM) | one short question 28 → 19 ms against vLLM bf16, +48% req/s at 1 client, about 4% at 32. No gain elsewhere | agreement 0.971 / 0.9735 / **0.772**, roundtrip 0.975 | **not shipped**: fails |
| **NVFP4A16** | same as W4A16 | agreement 0.9085 / 0.905 / **0.689**, roundtrip 0.920 | **not shipped**: fails |
| **W8A16** (GPTQ int8) | slower than bf16 except one question at one client | agreement 0.996 / 0.997 / **0.977**, roundtrip 0.995 | **not shipped**: fails on MMLU-Pro |
| **FlashQLA** DeltaNet kernel | kernel 2.0–2.7x; end to end 1–5% eager, 5–13% compiled, because that kernel is about 9% of the tower's time | agreement 0.9975 / 0.997 / 0.993; max \|dp\| 1.5x what changing the batch size alone causes | **not shipped**: a kernel swap must agree ≈ 1.000 |
| **vLLM backend** | 84 against 34–37 req/s for concurrent single short questions; 3x faster offline scoring (277 against 877 s for the suite); slower with several questions per request (8 questions on 1,052 tokens: 2.9 against 7.0–8.4 req/s) | agreement 0.997 / 0.999 / 0.992, roundtrip 0.995, ECE +0.0007 | passes; **experimental**, on an unreleased branch |

Agreement triples are typed-decisions canonical / typed-decisions reversed / MMLU-Pro 1k.
MMLU-Pro was the most sensitive set every time: quantization lost 23–31 points of agreement
there at 4 bits.

Notes on the ones that did not ship:

- **Quantization.** It only helps where one pass is bound by reading the weights, which
  means one short question. Multi-question and long-document requests are compute-bound,
  and there the Marlin kernels are no faster than bf16. With 4-bit weights, top-1 on single
  benchmarks dropped by up to 5.9 points.
- **vLLM.** It batches separate requests, which this server does not. But it caches a shared
  document only in whole 544-token blocks on this hybrid model, because an attention page
  has to hold one fp32 DeltaNet state. So questions about a short document share nothing,
  and each question re-reads it. This server reads a document once per request.

### The floor, and what remains

One pass cannot go below about 10 ms on the GB10 while the tower reads 2.75 GB of bf16
weights. These would move the floor itself:

- **Early exit:** fewer layers per pass. It is being tested.
- **Distillation** to a smaller tower.
- **Quantization-aware training,** so that 4-bit weights pass the checks above.
- **A datacenter GPU path:** CUDA graphs on H100-class hardware.

Each needs the checks below before it ships.

## Checks every speed change is held to

1. **Release verify agreement.** The full typed-decisions test set in canonical and
   reversed option order, and MMLU-Pro 1k. At least 0.99 of answers must match the bf16
   serve path. A kernel swap must match about 1.000, inside what changing the batch size
   alone moves.
2. **Roundtrip.** 200 decisions, compared by argmax with the training run's own fp32
   record. Agreement must be at least 0.99; bf16 itself gets 0.995.
3. **Calibration.** Suite ECE after calibration must stay within ±0.003 of the bf16 serve
   path.

The serving path also has to equal the evaluated one. `tests/test_serve_parity.py` checks
that the served forward pass matches `evaluate.predict` bit for bit. `tests/test_prefix_cache.py`
and `tests/test_doc_cache.py` check that the cached reads match a fresh read.
