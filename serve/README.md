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

## Speed

**Install `fla`.** Without it, the DeltaNet layers of the Qwen3.5 tower run on a plain
PyTorch fallback. With `flash-linear-attention` and `fla-core` 0.5.2 installed, a GB10 runs
about twice as fast. Its release verify still agrees 1.0000 with the training record on
all three sets.

The document is read once per request and every question continues from that read. Below
about 480 saved tokens, reading it once per question is faster, and the server picks the
faster path. `RSIJEV_MIN_SAVED_TOKENS` overrides the threshold.

Two switches, both off by default. Measured on one GB10 with v3.0-2B, a bf16 tower and fla,
p50 for 1 / 8 / 32 questions:

| path | 80-token doc | 1,052-token doc |
|---|---|---|
| default | 24 / 67 / 189 ms | 75 / 127 / 286 ms |
| `RSIJEV_DOC_CACHE=1`, same state again | 23 / 49 / 188 ms | 25 / 63 / 240 ms |
| `RSIJEV_COMPILE=1` | 21 / 62 / 164 ms | 60 / 107 / 243 ms |

**Document cache** (`RSIJEV_DOC_CACHE=1`), for agent loops that ask again about the same or
a growing state. Reads are kept across requests, keyed on the exact token ids and the
weights. A repeated state skips the read. A state that extends a cached one reads only the
new tail. An agent transcript growing by 50–200 tokens per step drops from 78–84 to
48–54 ms per step for one question. The results match a fresh read (every argmax is equal;
`tests/test_doc_cache.py`). A cold single question costs one extra pass: 49 ms instead of
31 ms. Limits are `RSIJEV_DOC_CACHE_ENTRIES` (32) and `RSIJEV_DOC_CACHE_MB` (2048).

**Compile** (`RSIJEV_COMPILE=1`). torch.compile of each decoder layer and the scorer. The
first requests compile, which takes about 40 s. Against the bf16 tower: 99.9% / 99.9% / 99.0%
of answers agree on the release verify sets, and suite ECE after calibration moves by
+0.0001.

## Layout

| file | role |
|---|---|
| `wire.py` | the contract: request → `Question`, distribution → answer. Pure Python, no torch, no HTTP |
| `infer.py` | the forward pass, pinned to `evaluate.predict` by `tests/test_serve_parity.py`; the document cache |
| `accel.py` | the opt-in compile switch |
| `app.py` | routes, schemas, auth, error envelopes |
| `../scripts/serve.py` | loads a release checkpoint and runs uvicorn |

`infer.py` exists because `evaluate.predict` takes `Case` objects and a `Case`
requires gold, which a served request does not have. Rather than fabricate a
gold label to satisfy a dataclass, the serving path calls the same primitives
directly — and a test asserts the two paths agree bit for bit on real weights.

Tests: `pytest tests/test_serve.py` for the contract, and `tests/test_serve_parity.py` —
marked `slow` — for the bit-for-bit agreement, on real weights.
