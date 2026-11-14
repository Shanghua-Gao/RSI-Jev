# decideit: a local judgment model for your programs

decideit puts an RSI-Jev decision model on your machine. You give it a **state**: text, plus up to four images if
you want. You give it one or more **questions**. It returns a probability for each answer. It doesn't write text.
It reads the input once and answers, so it fits inside scripts, hooks and agent loops that need a quick yes/no or
a pick from a list.

Some questions it fits: "is this safe to run?", "did the build pass?", "which team should get this ticket?",
"does this screenshot show an error?", "how urgent is this: routine, urgent or emergency?".

This page doesn't depend on a model version. decideit uses the current recommended release. The
[README's release table](../README.md#releases) lists the models and their scores.

## Install

| machine | command |
|---|---|
| Apple Silicon Mac (macOS 14+) or Linux (x86_64, aarch64) | `curl -fsSL https://github.com/Shanghua-Gao/RSI-Jev/releases/latest/download/install.sh \| sh` |
| Mac with Homebrew | `brew install shanghua-gao/tap/decideit` |
| any machine with Python 3.10+ | `pip install "rsi-jev[fast,vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev"` |

The first two need no Python: they bring their own and keep everything in `~/.rsi-jev`. On Linux the installer
picks the GPU build of PyTorch when an NVIDIA driver is present and the CPU build otherwise. You get these commands:

- `decide` (also `decideit`): the judgment command;
- `agentguard`: the Claude Code hook described below;
- `rsi-jev`: the server.

The model downloads the first time you ask a question. After that it starts by itself and quits after 10 idle
minutes (`RSIJEV_IDLE_MINUTES` changes that). With a pip install, `decide` asks a running server instead (see
[the HTTP API](#2-the-http-api)).

```sh
echo 'The build failed on line 3.' | decide 'Did the build pass?' && echo yes || echo no
```

To remove the curl install: `rm -rf ~/.rsi-jev ~/.local/bin/decide ~/.local/bin/decideit ~/.local/bin/agentguard ~/.local/bin/rsi-jev`.

## 1. `decide`, from any language or shell script

```sh
git diff | decide "Does this change touch authentication?"             # yes / no
cat ticket.txt | decide "Who should handle this?" --choice billing,technical,other
decide "Does the screen show an error?" --image shot.png                # up to 4 --image
some_cmd | decide "Is this output a failure?" --min-p 0.9 --json
```

- The state comes from **stdin**; the question is the first argument.
- **Exit code:** `0` when the probability reaches `--min-p` (default 0.5), `1` when it doesn't, `2` on any error.
  An error never exits 0, so a failed judgment can't read as "yes".
- **Output:**
  - a yes/no question prints the probability to stderr;
  - `--choice` prints `winner probability` to stdout;
  - `--json` prints one object, `{"p", "choice", "pass", "min_p", "depth", "model", "ms"}`. `p` is P(yes) for a
    yes/no question, and the winner's probability for a choice.
- **Other flags:**
  - `--effort low|medium|high|auto`: how deep the model reads;
  - `--timeout SECONDS`;
  - `--url URL` to use a running server (or set `DECIDE_URL`).
- **Another model:** set `RSIJEV_DECIDE_MODEL` to an alias, a Hugging Face repo id or a local directory.

## 2. The HTTP API

```sh
rsi-jev serve <model> --port 8000        # model: an alias from the release table, a repo id or a directory
```

```sh
curl localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": "I was charged twice. Please refund.",
  "questions": {
    "refund":     {"type": "noul",   "instructions": "Does the user request a refund?"},
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "Payments and refunds", "technical": "Software bugs"}},
    "urgency":    {"type": "score",  "instructions": "How urgent is the request?",
                   "criteria": ["Routine", "Urgent", "Emergency"]}
  }
}'
```

The answer has this shape:

```json
{"model": "jev-latest",
 "answers": {
   "refund":     {"type": "noul", "noul": 0.99},
   "department": {"type": "choice", "choice": "billing",
                  "probabilities": {"billing": 0.99, "technical": 0.01}, "confidence": 0.98},
   "urgency":    {"type": "score", "score": 1.08,
                  "legend": {"0": "Routine", "1": "Urgent", "2": "Emergency"},
                  "probabilities": {"0": 0.18, "1": 0.56, "2": 0.26}, "confidence": 0.35}},
 "usage": {"input_tokens": 136, "output_tokens": 3}}
```

- `state` is a string or a chat list: `[{"role": "user", "content": "..."}]`.
- **Question types:**
  - `noul` is yes/no and returns `noul` = P(yes). An optional `criteria` of `{"true": "...", "false": "..."}`
    names the two sides.
  - `choice` picks one key of `criteria`, which maps each key to its description.
  - `score` takes an ordered list of levels and returns the expected **zero-based** index, so 1.08 means
    "Urgent".
- **Many questions:** up to 64 per request. Asking them together is cheaper than one request each, because the
  state is read once.
- **Images:** `"images": ["data:image/png;base64,..."]` takes PNG, JPEG or WebP, 1 to 4 per request. Put
  `<image>` in the state where each image belongs. http(s) URLs aren't fetched.
- **Effort:** `"effort": "low" | "medium" | "high" | "auto"` sets how many layers answer. `auto` stops early
  once the model is confident.
- **Other routes:**
  - `GET /v1/limits`: limits, calibration and the release answering;
  - `GET /v1/models`;
  - `GET /health`.
- **Auth:** `--api-key` (or `RSIJEV_API_KEY`) requires `Authorization: Bearer …`.
- **Jev clients:** a client written for Jev's API works once its base URL points here.

Full reference: [`serve/README.md`](../serve/README.md). Running it, speed and memory:
[`inference.md`](inference.md) and, on a Mac, [`mac.md`](mac.md).

## 3. Python, without a server (pip install)

```python
from rsijev import Decider
d = Decider("<model>")                          # alias, Hugging Face repo id or directory
answers = d.decide("I was charged twice. Please refund.",
                   {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}})
answers["refund"]["noul"]                       # P(yes)

d.decide("Customer photo: <image>\nThe customer says it arrived damaged.",
         {"damaged": {"type": "noul", "instructions": "Does the photo show visible damage?"}},
         images=["photo.jpg"])                  # paths, bytes, data URLs or PIL images
```

`decide()` returns the same `answers` object the server would.

## Patterns

- **Gate a script:** `cmd | decide "Did it succeed?" --min-p 0.9 || alert`.
- **Route:** use `--choice` or a `choice` question, then dispatch on the winner.
- **Many questions about one input:** put them in one request.
- **Agent permissions:** [`agentguard`](../apps/agentguard/README.md) is a Claude Code `PreToolUse` hook built
  on decideit. It only removes a permission prompt when the model is very sure a call is safe. It never denies,
  and a blocklist runs first.

## Things to know

- **Choose the threshold on your own examples.** Label a few dozen real cases. Set `--min-p` where wrong "yes"
  answers are rare enough for your use. The probabilities are calibrated on our benchmarks, not on your task.
- **The model sees option keys.** Use meaningful keys (`billing`, not `a`) and a short, concrete description for
  each.
- **Ask one thing per question,** and put the context the question needs in the state.
- **Limits:** a request longer than 32,768 text tokens is refused, never cut. A request takes at most four images.
  Long documents are slower, much more so on a Mac than on a GPU.
- **It judges; it does not generate.** It can't write a reply, summarise or explain itself. Use it to decide, and
  a generative model for the words.
- **Local:** after the one-time model download, nothing leaves the machine.

## For an AI agent reading this

1. `decide --help` should work after either installer. With a pip install, start `rsi-jev serve <model>`, then
   check `curl localhost:8000/health`.
2. In a shell, run `decide "<question>" [--choice a,b,c] [--json] < input.txt` and branch on the exit code:
   - `0` means yes;
   - `1` means no;
   - `2` is an error. Treat it as unknown, never as yes.
3. From code, `POST /v1/systemone` with `{"model": "jev-latest", "state": ..., "questions": {...}}` and read
   `answers[<key>]`.
4. Before relying on it, measure it on labelled examples from the real task and set the threshold from them.
