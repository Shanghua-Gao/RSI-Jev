# Which questions need more layers? (v6.0-VL)

v6.0-VL answers from layer 16, 20 or 32 of its tower. This page measures what the deeper layers
buy on [Decision Index 0.2.1](https://github.com/apolinario/decision-index): by effort level, by
benchmark and by question. It is the evidence behind the `effort` setting described in
[inference.md](inference.md#effort-levels-v60-vl) and in the [release record](../versions/v6.0-vl.md).

All numbers come from the released package (bf16, one H200), on the same stratified 16,000-row
sample of Decision Index, scored on a sample-only suite (coverage 1), one option order, with the
57 rows that overlap our training data removed. Fixed exits were forced with variants that share
the released weights.

## Effort levels

Skill × 100 by area (chance-corrected); latency is the median request.

| effort | layers | index | knowledge & reasoning | language | retrieval | tools | arts | median ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `low` | 16 | 43.3 | 25.1 | 43.5 | 53.2 | 63.6 | 32.3 | 23 |
| `medium` | 20 | 45.9 | 28.4 | 46.7 | 53.6 | 66.1 | 36.5 | 27 |
| `high` | 32 | 45.9 | 29.2 | 45.7 | 53.2 | 67.1 | 36.1 | 40 |
| `auto` (thresholds 0.95 at 16, 0.59 at 20) | [PLACEHOLDER: new auto DI gate] | | | | | | | |
| earlier `auto` (one threshold, 0.59) | 18.5 on average | 44.9 | 28.8 | 44.8 | 52.9 | 64.5 | 34.9 | 30 |
| unset (default) | mixed | 45.7 | 29.2 | 45.9 | 53.3 | 65.6 | 36.1 | 40 |
| v5.0-VL | 20 | 37.4 | 25.8 | 42.4 | 41.8 | 51.7 | 19.5 | – |

The earlier `auto`, with one threshold of 0.59 at both exits, answered 78% of Decision Index questions at layer 16, 9% at 20 and 13% at 32; the section below shows what that cost and how the released thresholds fix it. The unset
default runs a single question at all 32 layers and uses the cascade only for multi-question
requests, so on this benchmark (mostly single questions) it scores close to `high`.

## Which benchmarks gain from depth

Skill at layer 16 / 20 / 32. About 315 questions per benchmark in the sample (standard error
about 0.03), so only gaps of 0.03 or more count.

- **Done at layer 16:** BFCL function calling (.93 / .94 / .93), CLINC150 and BANKING77 intent,
  When2Call, FinEntity, PhishNChips, BRIGHT and Amazon ESCI retrieval, New Yorker caption
  matching.
- **Needs layer 20:** HellaSwag, ANLI, NLI4CT, BBH fixed-option (.39 / .48 / .47), HoVer claim
  verification (.21 / .32 / .32), CLadder, VAST, BPoMP (.53 / .75 / .76), ToolRet, the home
  appliance simulator.
- **Keeps gaining to layer 32:** API-Bank tool documentation (.60 / .62 / .66), GPQA Diamond
  (.14 / .18 / .22), MuSR (.34 / .36 / .39), CRUXEval (.07 / .10 / .13), POP909 (.52 / .59 / .62).

Smaller gains from 20 to 32 show on GSM8K, WinoGrande and Habermas. BANKING77 (−.035), New Yorker
(−.060) and ESCI (−.037) get worse with depth. HLE, RAGTruth and iSarcasm score zero at every exit.

## Which questions gain from depth

Per-question correctness at layers 16, 20 and 32, on the 27 benchmarks that Decision Index
scores by per-question accuracy (10,232 questions; our per-question accuracy matches the kit's
own score within 0.02 at every exit). Ranking, F1 and case-level benchmarks are covered only in
the section above, because per-question accuracy does not measure them. *Rescued* is wrong at 16
and right at 32; *harmed* is the reverse.

**By layer 16's own confidence** (calibrated top-option probability):

| layer-16 confidence | questions | acc 16 | acc 20 | acc 32 | rescued | harmed | net |
|---|---:|---:|---:|---:|---:|---:|---:|
| below 0.50 | 2,323 | .331 | .354 | .363 | 11.5% | 8.4% | +3.1 |
| 0.50–0.59 | 1,546 | .537 | .673 | .671 | 22.5% | 9.1% | +13.5 |
| 0.59–0.70 | 1,717 | .643 | .690 | .695 | 12.1% | 6.9% | +5.2 |
| 0.70–0.80 | 1,529 | .767 | .763 | .774 | 2.9% | 2.2% | +0.7 |
| 0.80–0.90 | 1,520 | .872 | .876 | .877 | 1.1% | 0.6% | +0.5 |
| 0.90–0.97 | 1,067 | .906 | .907 | .909 | 0.3% | 0.0% | +0.3 |
| 0.97 and above | 530 | .949 | .949 | .949 | 0.0% | 0.0% | 0.0 |

**By question shape:**

| slice | questions | benchmarks | acc 16 | acc 20 | acc 32 | net 16→32 |
|---|---:|---:|---:|---:|---:|---:|
| 2 options | 3,797 | 11 | .694 | .759 | .762 | +6.8 |
| 3–4 options | 4,083 | 16 | .735 | .752 | .760 | +2.4 |
| 5–10 options | 1,404 | 10 | .462 | .467 | .449 | −1.4 |
| 11–50 options | 314 | 3 | .080 | .080 | .089 | +1.0 |
| more than 50 options | 634 | 3 | .568 | .607 | .645 | +7.7 |
| input under 256 tokens | 7,096 | 23 | .673 | .714 | .718 | +4.5 |
| input 256–1k | 1,980 | 18 | .607 | .620 | .619 | +1.2 |
| input 1k–4k | 839 | 7 | .601 | .628 | .638 | +3.7 |
| input 4k–16k | 317 | 3 | .606 | .621 | .666 | +6.0 |
| all accuracy-scored | 10,232 | 27 | .652 | .686 | .690 | +3.8 |

Most of the gain happens between layers 16 and 20 (.652 → .686). From 20 to 32 the average moves
by .004, which hides the knowledge and multi-step reasoning benchmarks above that keep gaining.

## Why `auto` has a threshold per exit

This section is about the earlier setting, one threshold of 0.59 at both exits.

- That cascade answered at layer 16 when the calibrated top-option probability there was at least
  0.59. On Decision Index that is 78% of questions, and layer 16 alone scores 43.3.
- Its threshold and per-exit temperatures were chosen on our dev sets, which resemble our own
  test suite. There layer 16 is nearly as good as 32: any threshold of 0.59 or more matches full
  depth at about 21 layers on average.
- Decision Index has more two-option judgement questions, more knowledge and reasoning, and
  formats the model has not seen. Layer 16 is confident and wrong there more often: the 0.59–0.70
  band still gains 5.2 points from depth but stops at layer 16. About three quarters of the loss
  comes from two-option judgement tasks, where 0.59 is barely above chance.
- `medium` wins on Decision Index because 20 layers happen to be enough there. A pre-registered
  check on fresh dev sets found it is not a better default in general: on suite-like questions it
  trails the cascade (.711 vs .756).

**The fix.** `auto` now has its own threshold at each exit: 0.95 at layer 16 and 0.59 at layer 20.
Layer 16 answers only when it is nearly sure; the questions it used to answer at 0.59–0.95 go on to
layer 20, which is enough for most of them. The thresholds were chosen on our development sets,
picked on one half and confirmed on the other, with average depth capped at 24 layers, then read
once on test: suite 0.771 (`high` 0.770), held-out 0.696, MMLU-Pro 0.444, final ECE 0.024, with 20% of
questions stopping at layer 16, 46% at 20 and 34% at 32 (23.3 layers on average). Decision Index: [PLACEHOLDER: new auto DI gate].

## Choosing an effort level

- `low` for intent, routing, retrieval and sentiment, where layer 16 is already saturated
  (57% of full latency).
- `medium` for classification and judgement workloads: the same Decision Index score as `high` at
  about 70% of the latency.
- `high` for knowledge, multi-step reasoning, code and tool documentation.
- `auto` for mixed workloads: its default thresholds (0.95 at 16, 0.59 at 20) match `high` on our
  suite at about 23 layers. Every response
  reports each question's exit (`usage.depth`) and calibrated confidence (`usage.confidence`), so
  the routing can be checked.

## Limits

- 16,000-row sample reads, not full runs. This page compares exits with each other on the same
  rows, which a sample-to-full offset does not change.
- The per-question section uses the Decision Index test sample. It explains the released
  behaviour; it was not used to choose any setting.
- One seed, one read. Per-benchmark differences under 0.03 are within noise.
