# A judgment model against a chat model, on the same image questions

30 questions from the [gallery](../gallery/), each with a reference answer. Each is asked
two ways on one GPU, one model process at a time, batch 1:

- **RSI-Jev v4.0-VL** through `rsi-jev serve`, over HTTP. It returns a probability for
  every allowed answer.
- **Qwen3.5-2B as a chat model** (the vision chat checkpoint released beside the base
  v4.0-VL is trained from), non-thinking, greedy, asked to reply with a JSON object.

What this shows is **latency and output validity**. The accuracy comparison is
unresolved; see below.

```bash
pip install -e ".[vision]"            # from a clone of this repo
bash examples/chat_vs_rsi/run.sh            # needs a CUDA GPU; writes rsi.json, chat.json, results.md, the GIF
```

Or step by step:

```bash
rsi-jev serve v4.0-vl-2b --port 8000 &
python examples/chat_vs_rsi/rsi_client.py http://127.0.0.1:8000     # -> rsi.json
kill %1                                                              # one model at a time
python examples/chat_vs_rsi/chat_baseline.py                        # -> chat.json
python examples/chat_vs_rsi/summarize.py                            # -> results.json, results.md
python examples/chat_vs_rsi/make_gif.py chat-vs-rsi.gif             # replay GIF from the measured timings
```

## Published result (NVIDIA GB10, bf16)

| | RSI-Jev v4.0-VL | Qwen3.5-2B chat (non-thinking) |
|---|---|---|
| latency, median of per-question p50 | 69 ms | 540 ms |
| latency range (per-question p50) | 52–257 ms | 301–6,848 ms |
| valid output | 30/30 | 20/30 |
| confidence | a calibrated probability per answer | none: text only |
| accuracy, invalid counted as wrong | 26/30 | 16/30 |
| chat, lenient re-read (post hoc) | | valid 27/30, accuracy 23/30 |

RSI-Jev's latency is the whole HTTP call (JSON, image decode, resize, tokenise, forward),
3 warm calls then the p50 of 9. The chat model's is the processor plus `generate`, 2 warm
runs then the p50 of 5.

**Why the accuracy is unresolved.** The strict rule, fixed before the run, reads the value
under the question's id. All ten of the chat model's invalid replies are to questions whose
id is `q`: it echoed the question text under `"q"` and, in most of them, put its answer
under `"answer"`. That is a parsing confound, not a vision error. The lenient re-read (the
last JSON object, any value that is an allowed answer) was defined after seeing the replies,
so it is an upper bound on what a more forgiving parser would get. Against it the gap is 26
against 23 of 30, which 30 questions cannot separate from noise.

## Files

| file | what it does |
|---|---|
| `questions.json` | the 30 questions; images are the gallery's (`../gallery/images/`) |
| `rsi_client.py` | side (a): HTTP requests to `rsi-jev serve` |
| `chat_baseline.py` | side (b): Qwen/Qwen3.5-2B with `transformers`, the prompt and the strict parser |
| `summarize.py` | the table above, paired counts and an exact McNemar test |
| `make_gif.py` | the side-by-side replay GIF (a replay of measured timings, not a live recording) |
| `run.sh` | all four steps |

The questions were drawn from the gallery pools before any run: per category a fixed
quota (VisA 4, road 7, CLEVR 6, "is it there?" 4, charts 5, screens 4), `random.Random(0)`
inside each, one question per item, distinct photos for the road items. Every image is
republishable; per-file licences are in [../gallery/CREDITS.md](../gallery/CREDITS.md).
