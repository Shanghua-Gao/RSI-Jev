# Contributing

**A contribution here is usually a measurement, not a patch.** RSI-Jev is a model *line*: a chain
of versions, each one a hypothesis that was registered, run, and then shipped or killed by its own
measurement. The most useful thing you can send is something that moves or blocks the next one.

This says what to send, what happens to it, and the discipline it will be held to.

## What to send, and what happens to it

Two templates, one pipeline, and neither ends with a maintainer nodding at it.

- **[The model got it wrong →](https://github.com/Shanghua-Gao/RSI-Jev/issues/new?template=wrong-answer.yml)**
  the document, the question, the answer you expected.
- **[I tried a variant →](https://github.com/Shanghua-Gao/RSI-Jev/issues/new?template=experiment.yml)**
  what you changed, per-seed numbers, and which kernel stack.

**A wrong answer** becomes a test case first. If it generalises — if a class of documents is being
read wrong rather than one document — it becomes a hypothesis with a registered prediction, and
the next cycle either moves the metric or records that it did not. Cases where the model is
*confidently* wrong are the most valuable, because confidence and correctness are supposed to
travel together and a counterexample says exactly where they came apart.

**A variant you measured** removes a branch. The loop's scarcest resource is GPU time, and the
expensive part of a search is not running the winner, it is running the losers. A well-measured
negative from outside is worth more to us than a small positive, which is why the template asks
for paired per-seed deltas rather than a mean — a delta whose sign flips across seeds is noise,
and only the per-seed values show that — and for the kernel stack, because numbers do not compare
across them.

What we will not do is silently fold a suggestion in. If a proposal becomes a version, the version
card says where it came from; if it was tried and failed, that is recorded too, under the same
rule that applies to our own ideas.

## The discipline it will be held to

The repo publishes the **search**, not just the scores, because in our own history the search is
where nearly all the error lived.

- **No contamination, checked rather than asserted** — no release trains on any benchmark's
  **test** split, and no training document shares a state with one. Whether a release may use a
  benchmark's **train** split changed at v2.0, and it changes what its scores mean:
  [`BENCHMARKS.md`](BENCHMARKS.md#what-a-model-may-train-on). A held-out set of three benchmarks
  is read once per release, so two releases can still be compared on equal terms.
- **Null floors are measured, not assumed** — arms provably identical to the control, verified by
  object identity before any GPU time. Their spread *is* the noise floor, so a difference smaller
  than it is not a result.
- **Predictions are registered before the run.** A version that misses its own bar ships as a
  failure rather than getting quietly re-cut.
- **Artifacts are verified.** A checkpoint is reloaded from disk, re-scored, and published only if
  it reproduces its training run's per-question predictions exactly. Both v1.0 checkpoints agree
  at **1.0000**, and v2.0 at 1.0000 on typed-decisions. Its MMLU-Pro agreement is 0.9980 on one
  machine and 1.0000 on another, because a few of that benchmark's answers turn on a logit
  margin below 0.001 — reported rather than rounded, in
  [`versions/v2.0.md`](versions/v2.0.md#3-checkpoints).
- **Failures ship**, including the ones that killed our own champion.

What that buys: the bug behind v1.0 took **seven registered negatives** to find. Each was an
optimiser-side fix that reduced a training instability without removing it, because the cause was
a precision mistake nowhere near the optimiser — and the fix is one line. Not one of the seven was
worth publishing alone; together they are what made it findable. The trail is in
[`versions/v1.0.md`](versions/v1.0.md#10-how-it-got-here).

## How a release is cut

Two rules, pointing in opposite directions on purpose.

**The research record accumulates.** `versions/` keeps one card per release, all of
them, on `main` forever — what was tried, what shipped, what was killed, and by
which measurement. That chain *is* the project: it is the only way to see whether a
self-improving loop is improving. Each card carries its own release's numbers, so
v2's speed and calibration never overwrite v1's. Dropping an old card would drop
the evidence.

**Everything else is the current release only.** The recipe, the scripts, the
checkpoint ids, the pinned stack, the serving and code guides — `main` carries one
version of each. When the next release ships, a branch is cut at the one being
superseded and `main` moves on, so `git checkout v1.0` is the exact tree that
produced v1.0 and nothing about v1.0 is left on `main` to be mistaken for current.

What lets those coexist is that a published checkpoint carries its own code: every
HF release ships a `code/` directory and `load_release` uses it. So the demo
compares every release ever published from `main` alone — it needs repository ids,
not old branches.

The tag goes on at release time, not before: everything here is squashed to a
single commit when a version ships, so a tag applied earlier would name a commit
that does not survive.

## What is running it

[**AutoScientists**](https://github.com/mims-harvard/AutoScientists)
([paper](https://arxiv.org/abs/2605.28655)), running here as an internal next version: agents form
teams around hypotheses, critique each other's proposals before any GPU time is spent, and share
failures so the system stops re-exploring dead ends.

Its task definitions and per-cycle traces stay private. The **version-level record** is what
ships — the models, the code, the numbers, and every attempt that did not make it. That boundary
is deliberate: the traces are the system's working memory and publishing them would let a reader
re-run our search rather than check our result.

**v1.0 itself was built in a human-directed agent session** under the same pre-registration
discipline, not by the swarm. An earlier phase pointed the same loop at a narrower question — how
far an *inference algorithm* alone could go on a frozen model, with no training — and that is
where the earlier multi-agent results come from.

**v2.0 is the first release the multi-agent run produced**, out of about fifty arms of which four
were kept. It is also the first release to show the limit of its own bar: it beats v1.0 by 0.096
on the suite it was judged against, and does **not** beat it on the three benchmarks held out from
the whole search. That is in its record and in [`EXPLORE.md`](EXPLORE.md) rather than left for a
reader to discover, because a search that only publishes the numbers it was optimising is the
failure mode this repo exists to avoid.

## Related work

- **[jaredpalmer/kev](https://github.com/jaredpalmer/kev)** — trainable Jev-like family
  (0.8B/4B/9B) with frozen eval suites.
- **[denis-pplx/autojev](https://github.com/denis-pplx/autojev)** — also agent-built; full-weight
  SFT of Qwen3.8-27B, 84.60% vs Jev 82.79%.
- **[ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang)** — Jev-compatible
  endpoint on a frozen open model; the wire contract our server reproduces.
- **[aisa-group/InferenceBench](https://github.com/aisa-group/InferenceBench)** — its headline
  result is that plain Random/SMAC3/TPE search beat every agent, so **we owe a non-agent baseline
  at equal budget**. It is on the roadmap for exactly that reason.
- Method priors we are measured against, not credited with:
  [PriDe](https://arxiv.org/abs/2309.03882) for option-order debiasing and
  [G-Eval](https://arxiv.org/pdf/2303.16634) for the `Σ i·pᵢ` readout.
