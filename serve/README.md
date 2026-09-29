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

## Structured criteria and abstention

**Structured criteria.** A choice option's description can be an object, as Jev accepts it:

```json
"criteria": {
  "spam":  {"what": "Unsolicited bulk email", "includes": ["ads from strangers", "lottery scams"],
            "excludes": ["newsletters the recipient signed up for"]},
  "legitimate": "Email the recipient expects"
}
```

`what` is text; `includes` and `excludes` are text or lists of text; other fields are rejected (422).
Strings, `null` and objects can be mixed. The object is rendered by the same function the training
data used (`rsijev.encode.criterion_text`): `Unsolicited bulk email. Includes: ads from strangers;
lottery scams. Excludes: newsletters the recipient signed up for.` String criteria encode exactly as
before.

**Abstention (choice only).** Set `"allow_abstain": true` on a choice question. The model then also
scores a reserved option, "Not enough information", in the same forward pass. The answer keeps its
usual fields over the real options (probabilities renormalized to sum to 1) and adds two:

```json
{"type": "choice", "choice": "billing", "probabilities": {"billing": 0.8, "technical": 0.2},
 "confidence": 0.6, "unknown_probability": 0.12, "abstained": false}
```

- `unknown_probability`: the reserved option's probability.
- `abstained`: `unknown_probability >= tau`. `tau` comes from the checkpoint's `abstain.json`,
  a split-conformal threshold fitted on a dev set of answerable questions (at most `alpha` of
  them are flagged), or `--abstain-tau`. `GET /v1/limits` reports `tau` and where it came from.
- With `allow_abstain`, a choice question takes at most one option fewer than the usual cap,
  and the key "Not enough information" is reserved. It is an ordinary key when abstention is off.
- noul and score questions return 422 with `allow_abstain: true`.
- Without `allow_abstain`, requests and answers are unchanged.

## Layout

| file | role |
|---|---|
| `wire.py` | the contract: request → `Question`, distribution → answer. Pure Python, no torch, no HTTP |
| `infer.py` | the forward pass, pinned to `evaluate.predict` by `tests/test_serve_parity.py` |
| `app.py` | routes, schemas, auth, error envelopes |
| `../scripts/serve.py` | loads a release checkpoint and runs uvicorn |

`infer.py` exists because `evaluate.predict` takes `Case` objects and a `Case`
requires gold, which a served request does not have. Rather than fabricate a
gold label to satisfy a dataclass, the serving path calls the same primitives
directly — and a test asserts the two paths agree bit for bit on real weights.

Tests: `pytest tests/test_serve.py` for the contract, and `tests/test_serve_parity.py` —
marked `slow` — for the bit-for-bit agreement, on real weights.
