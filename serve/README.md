# Jev-compatible serving

A trained RSI-Jev decision model behind the same HTTP API as Jev, so anything
written against Jev works against this without changes. Getting a checkpoint and
the rest of the project: the [root README](../README.md).

```bash
python scripts/serve.py --ckpt path/to/rsi-jev-v1.0-qwen3.5-2b --port 8000

curl localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": [{"role": "user", "content": "I was charged twice. Please refund."}],
  "questions": {
    "refund":     {"type": "noul",   "instructions": "Does the user request a refund?"},
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
    "urgency":    {"type": "score",  "instructions": "How urgent is the request?",
                   "criteria": ["Routine", "Urgent", "Emergency"]}
  }
}'
```

```json
{
  "model": "jev-latest",
  "answers": {
    "refund":     {"type": "noul", "noul": 0.983},
    "department": {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.979, "technical": 0.021},
                   "confidence": 0.958},
    "urgency":    {"type": "score", "score": 1.197,
                   "legend": {"0": "Routine", "1": "Urgent", "2": "Emergency"},
                   "probabilities": {"0": 0.068, "1": 0.667, "2": 0.265},
                   "confidence": 0.501}
  },
  "usage": {"input_tokens": 136, "output_tokens": 3}
}
```

That is the answer **RSI-Jev-v1.0-2B** actually returns for that request — bf16 on a
GB10, probabilities rounded to three places. v2.0 and v2.1 return different probabilities
for the same request: they ship a fitted calibration that rescales every answer, so their
numbers here would be sharper and their `confidence` higher. The worked example stays on
v1.0 because a test re-runs it against that checkpoint. Note that `score` is an index into the
rubric, so 1.197 is "Urgent", not a fraction of the scale.

**Ask whether you are getting calibrated probabilities.** `GET /v1/limits` reports a
`calibration` field: `none` for a v1.0 checkpoint, or the method a v2.0 or v2.1 one shipped, e.g.
`oof_head_scorefloor`. The chosen option is identical either way — calibration divides each
question's logits by one positive number — but the probabilities are not, so a threshold
tuned against one is not the same threshold against the other. The `version` field beside it
names the release answering, read off the checkpoint's own name, so a client can key a
threshold to a release rather than to a host.

| route | purpose |
|---|---|
| `POST /v1/systemone` | noul, choice and score evaluation |
| `GET /v1/models` | model catalogue, TypeSafe and OpenAI shaped |
| `GET /v1/limits` | admission limits, and how this deployment differs |
| `GET /health`, `GET /health/live` | readiness and liveness, never authenticated |

`jev-latest` is accepted as an alias for the served checkpoint. Apps built on Jev
often pin a version (`"model": "jev-1.13.0"`); `--accept-model jev-1.13.0` (repeatable)
answers those requests too, so switching such an app over is a configuration change.
The answer's `model` field then carries **this** server's model name, not the borrowed
one, because echoing it would claim the answer came from Jev — so an app that checks the
returned name against the one it sent will need that check relaxed. Set
`--api-key` (or `RSIJEV_API_KEY`) to require `Authorization: Bearer …` on
everything except the health routes.

## What is copied exactly

Taken from `reference/openjev-sglang`, which implements
[the TypeSafe/Jev HTTP API](https://docs.typesafe.ai/api), and pinned by
`tests/test_serve.py` (the same assertions its own suite makes):

- the request `{state, model, questions}`, with unknown fields rejected and no
  type coercion;
- the three question types, and their `criteria` shapes — noul's `"true"` /
  `"false"` keys defaulting to `"Yes"` / `"No"`, choice's key→description map,
  score's ascending list of levels;
- the answer shapes: **noul carries only `noul`** (no probabilities, no
  confidence); choice carries `choice`, `probabilities`, `confidence`; score
  adds `legend` and reports the probability-weighted **zero-based** index;
- `confidence` = (K·p_max − 1)/(K − 1), clamped to [0, 1] — the **peak** statistic, which
  TypeSafe documents for three options as "(3 x largest probability - 1) / 2";
- a limit of 1–64 questions per request, rejected before the model runs;
- error envelopes: `{"error": {"message": …}}` for domain errors, FastAPI's
  `{"detail": […]}` for schema failures, `Retry-After` on 429/503/529;
- `x-typesafe-request-id` on every answered request.

## What differs, and why

The wire contract is the compatibility surface; the prompt is not. The
reference says so itself: *"The providers can use different internal prompts and
inference procedures despite receiving equivalent payloads."*

- **The prompt and the readout.** The reference renders a chat template and
  reads the logprobs of single-token labels `A`/`B`/`C`. This is a **base** model
  with a trained readout head, served with exactly the encoder it was trained
  with (`- key: description` option blocks, `Answer:` cue). Using the
  reference's prompt would take the model off its training distribution.
- **Option keys are visible to the model.** The reference hides them, so
  renaming a key provably cannot change the answer. Here the key is part of the
  rendered block, so renaming one *can* move the answer. `GET /v1/limits`
  reports this as `option_keys_visible_to_model: true`.
- **Structured state is compact JSON, not a chat template.** A base model has no
  chat template, and the training states were themselves JSON documents, so
  serializing keeps the request in distribution. Chat transcripts are still
  *validated* exactly as the reference validates them — same roles, text only —
  so a request the reference rejects is rejected here too.
- **`usage.output_tokens`.** This path generates nothing; it reports one readout
  per question. The reference reports N+1 because it decodes one token per
  question plus a prefix warm-up.
- **Up to 160 options per question** (from v3.0; 64 before), because v3.0 trains and
  evaluates with 160. The reference admits 64.
- **Criteria must be strings (or `null`).** Jev also accepts a structured criterion —
  an object such as `{"what": …, "includes": […]}` — and this server rejects it with 422.
  No release so far was trained on a rendering of structured criteria, and serving one the
  model never saw would be guessing; it arrives together with a model trained on it.
- **A state longer than the model's context is cut from the start.** The encoder keeps the
  question, the options and the answer cue whole and drops the *beginning* of the state
  until the request fits (2,048 tokens for v1.0–v3.0; `rsijev/encode.py`). Nothing reports
  that it happened. So a request that puts its query first — `"Query: …"` followed by long
  candidates — loses the query, and the answer is about text the model never saw the
  question for. Keep states under the limit, or put what matters last.

## Opt-in speed paths

Three switches, all off by default. Measured on one GB10 (sm_121, torch 2.13, CUDA 13,
fla 0.5.2), RSI-Jev-v3.0-2B, bf16 tower, serve path (`score_questions_cached`), p50 of 20.
The bf16 baseline was measured in the same session.

| path | 80-token doc, 1 / 8 / 32 q | 1,052-token doc, 1 / 8 / 32 q | verdict |
|---|---|---|---|
| bf16 (default) | 24 / 67 / 189 ms | 75 / 127 / 286 ms | |
| `RSIJEV_DOC_CACHE=1`, same state again | 23 / 49 / 188 ms | 25 / 63 / 240 ms | exact; kept, off by default |
| `RSIJEV_COMPILE=1` | 21 / 62 / 164 ms | 60 / 107 / 243 ms | passed the gates; kept, off by default |
| `RSIJEV_FP8=1` | 45 / 99 / 265 ms | 106 / 169 / 373 ms | failed: slower, and more flips |
| `RSIJEV_FP8=1` + compile | 24 / 59 / 151 ms | 48 / 91 / 223 ms | failed the roundtrip bar |

**Document cache** (`RSIJEV_DOC_CACHE=1`). Keeps document caches across requests, keyed on
the exact token ids and a fingerprint of the weights. A repeated state skips the document
pass. A state whose token ids extend a cached one runs only the new tail. If the tokenizer
re-merges tokens across the old/new boundary, the state is read in full. The last
`RSIJEV_DOC_CACHE_HOLDBACK` tokens (default 8) are re-read with each question, so that a
transcript whose closing bracket becomes a comma still matches. Bounded by
`RSIJEV_DOC_CACHE_ENTRIES` (32) and `RSIJEV_DOC_CACHE_MB` (2048). A cold single question costs
a second pass: 49 ms instead of 31 ms for a 260-token state. An agent transcript growing
from about 260 to 1,880 tokens gives these per-step p50s:

| growth per step | 1 q | 4 q | 8 q |
|---|---|---|---|
| +50 tokens | 78 → 48 ms | 112 → 64 ms | 134 → 89 ms |
| +100 tokens | 79 → 50 ms | 113 → 67 ms | 135 → 91 ms |
| +200 tokens | 84 → 54 ms | 116 → 70 ms | 139 → 96 ms |

The same 1,815-token state asked again: 134 → 27 ms for 1 question, 197 → 77 ms for 8.
`tests/test_doc_cache.py` checks repeated and extended states against a fresh read in fp32:
every argmax is equal, and the worst probability difference is 1.6e-6 on CPU and 4.6e-4 on
the GPU with fla.

**Gates for FP8 and compile.** Each was checked against the bf16 tower on the same items:
the release verify (typed decisions in both orders, MMLU-Pro 1k), the 200-question roundtrip
against the training record, and ECE after the release calibration. Suite ECE here covers 9
of the 12 suite benchmarks (weight 0.70). nimble_public, jev_style_panel, nimble_holdout and
eval_final_v2 were not available on this machine, and no decontamination report was
applied. The ECE bar was ±0.003 of bf16. The roundtrip bar was ≥ 0.99 argmax agreement;
bf16 itself gets 0.995.

| path | top-1 typed can. / rev. / MMLU-Pro | agreement with bf16 | roundtrip | suite ECE (9/12) |
|---|---|---|---|---|
| bf16 | 0.790 / 0.7945 / 0.366 | — | 0.995 | 0.0790 |
| compile | 0.7895 / 0.7945 / 0.364 | 0.9985 / 0.999 / 0.990 | 0.995 | 0.0791 (+0.0001) |
| FP8 | 0.7875 / 0.793 / 0.367 | 0.9795 / 0.9755 / 0.921 | 0.985 | 0.0798 (+0.0008) |
| FP8 + compile | 0.785 / 0.7965 / 0.367 | 0.9745 / 0.9725 / 0.919 | 0.975 | 0.0796 (+0.0006) |

- **Compile** compiles each decoder layer and the scorer, with
  `dynamic=True`. `RSIJEV_COMPILE_MODE` takes `max-autotune-no-cudagraphs`, but on the GB10
  inductor reports too few SMs for GEMM autotuning. `reduce-overhead` is refused. The first
  requests compile: about 13 s for the first shape and 23 s more for the first batched one,
  and about 5 minutes with FP8. `RSIJEV_COMPILE_SCOPE=blocks` (MLP and full attention only,
  DeltaNet mixers eager) gave 24 / 67 / 186 ms at 80 tokens, which is barely a change.
- **FP8** uses torchao 0.18 dynamic float8 (per-row activation and weight scales) on the
  tower's Linear layers with both dimensions ≥ 128. Embeddings, the DeltaNet gate
  projections, the conv, the fla kernels and the scorer are left as they were. In eager mode
  its quantize kernels cost more than the FP8 GEMMs save, even though a bare FP8 GEMM here is
  about 2x a bf16 one. Padded positions reach the projections as zero rows, and a zero amax
  gave NaN until the amax got a floor of 1e-12. NVFP4, tried for speed only, was 2.5–4x
  slower than bf16 in eager mode (torchao without its MSLK kernels). It was not gated.

## Layout

| file | role |
|---|---|
| `wire.py` | the contract: request → `Question`, distribution → answer. Pure Python, no torch, no HTTP |
| `infer.py` | the forward pass, pinned to `evaluate.predict` by `tests/test_serve_parity.py`; the document cache |
| `accel.py` | the opt-in FP8 and compile switches |
| `app.py` | routes, schemas, auth, error envelopes |
| `../scripts/serve.py` | loads a release checkpoint and runs uvicorn |

`infer.py` exists because `evaluate.predict` takes `Case` objects and a `Case`
requires gold, which a served request does not have. Rather than fabricate a
gold label to satisfy a dataclass, the serving path calls the same primitives
directly — and a test asserts the two paths agree bit for bit on real weights.

Tests: `pytest tests/test_serve.py` for the contract, and `tests/test_serve_parity.py` —
marked `slow` — for the bit-for-bit agreement, on real weights.
