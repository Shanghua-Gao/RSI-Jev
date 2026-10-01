"""One thread owns the model; requests queue for it, and may share a forward pass.

    worker = GpuWorker(ModelRunner(served), window_ms=None)   # one request at a time
    worker = GpuWorker(ModelRunner(served), window_ms=2)      # micro-batching

A request is planned (tokenized, and its path chosen) in the caller's thread and
then queued. The worker thread runs the model. That keeps tokenizing the next
request off the GPU's critical path, and replaces the lock the server used to hold.

With `window_ms` set, the worker takes every request already queued when it
becomes free, and waits up to `window_ms` after the first one arrived for more.
The questions of the requests that read their state in full ("plain" requests, and
multi-question requests on a short state) then run together, in batches of at most
`max_rows` rows and `max_tokens` padded tokens, and the answers are split back.
Requests that read a long state once and continue each question from it run on
their own, as they do without batching, and so do requests with images. A request that finds nobody to share with
runs exactly as it would without batching.

Batching changes which rows share a bf16 forward, and that moves probabilities
slightly (argmax agreement is gated in speed2_report.md), so it is off unless asked
for: `--batch-window-ms`.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Sequence


class CallRunner:
    """Wraps any scorer `(state, questions[, images]) -> (probs, tokens)`; never
    batches. A text request calls the scorer with two arguments, as before images."""

    pools = False

    def __init__(self, scorer: Callable):
        self.scorer = scorer

    def plan(self, state, questions, images=None):
        return (state, questions, images) if images else (state, questions)

    def run(self, plans: list) -> list:
        return [_capture(self.scorer, *p) for p in plans]


def _capture(fn, *a):
    try:
        return fn(*a)
    except BaseException as e:            # delivered to the waiting request, not the worker
        return _Failed(e)


class _Failed:
    def __init__(self, exc: BaseException):
        self.exc = exc


class ModelRunner:
    """The served model: plan with `serve.infer.plan_request`, run with
    `score_planned`, and (when pooling) run several plans' rows together.

    `prep` and `venc` (serve.server.Served) enable image requests: they are planned
    with `plan_image_request`, run on their own with `score_image_planned`, and
    never pooled. Without them an image request is a 422 (`image_error` says why)."""

    pools = True

    def __init__(self, model, tok, enc, *, spec_max_options: int, device, batch_size: int = 16,
                 max_rows: int | None = None, max_tokens: int | None = None,
                 pool_prefix_tokens: int | None = None, prep=None, venc=None,
                 name: str = "this model", image_error: str | None = None):
        env = os.environ.get
        self.prep, self.venc, self.name, self.image_error = prep, venc, name, image_error
        self.model, self.tok, self.enc, self.device = model, tok, enc, device
        self.spec_max_options = spec_max_options
        self.batch_size = batch_size
        self.max_rows = int(max_rows or env("RSIJEV_BATCH_MAX_ROWS", 32))
        # One forward pass of the pooled rows carries at most this many padded tokens.
        # On the GB10, throughput on ~130-token rows rises from 37 req/s (1 row) to 84
        # (8 rows, ~1,100 tokens) and falls after; 1,096-token rows gain nothing from
        # batching (11.2 req/s alone, 9.5 in batches). So a long row runs on its own.
        self.max_tokens = int(max_tokens or env("RSIJEV_BATCH_MAX_TOKENS", 1280))
        # A multi-question request that reads its state once is pooled (each question
        # then read in full) only when its state is shorter than this. Off by default:
        # measured on the GB10, pooling 8-question requests on an 80-token state cut
        # throughput from 13.8 to 8.9 req/s.
        self.pool_prefix_tokens = int(pool_prefix_tokens if pool_prefix_tokens is not None
                                      else env("RSIJEV_BATCH_POOL_PREFIX", 0))
        self._tok_lock = threading.Lock()   # HF fast tokenizers are not safe across threads

    def max_options(self, questions) -> int:
        return max(self.spec_max_options, max(len(q.options) for q in questions))

    def plan(self, state, questions, images=None) -> dict:
        from serve.infer import plan_request
        if images:
            p = self._plan_images(state, questions, images)
        else:
            with self._tok_lock:
                p = plan_request(self.tok, state, questions, self.enc)
        p["max_options"] = self.max_options(questions)
        return p

    def _plan_images(self, state, questions, images) -> dict:
        from serve.images import state_too_long, text_only_error
        from serve.infer import plan_image_request
        if self.prep is None:
            raise text_only_error(self.name, self.image_error)
        try:
            with self._tok_lock:
                return plan_image_request(self.tok, self.prep, state, images, questions,
                                          self.venc)
        except ValueError as e:
            err = state_too_long(e, self.venc.max_length, self.venc)
            if err is None:
                raise
            raise err from None

    def one(self, plan):
        from serve.infer import score_image_planned, score_planned
        run = score_image_planned if plan["path"] == "image" else score_planned
        with self._maybe_tok_lock():
            preds, tokens = run(self.model, self.tok, plan, max_options=plan["max_options"],
                                device=self.device, batch_size=self.batch_size)
        return [list(p.probs) for p in preds], tokens

    def _maybe_tok_lock(self):
        from serve.infer import _needs_option_tokens
        return self._tok_lock if _needs_option_tokens(self.model) else _Null()

    def poolable(self, plan) -> bool:
        return plan["path"] == "plain" or (
            plan["path"] == "cached" and len(plan["prefix"]) < self.pool_prefix_tokens)

    def full(self, plans: list) -> bool:
        """Whether the poolable rows of `plans` already fill one forward pass."""
        rows = [len(e["input_ids"]) for p in plans if self.poolable(p) for e in p["encoded"]]
        return len(rows) >= self.max_rows or (bool(rows) and len(rows) * max(rows) >= self.max_tokens)

    def run(self, plans: list) -> list:
        pooled = [i for i, p in enumerate(plans) if self.poolable(p)] if len(plans) > 1 else []
        if len(pooled) < 2:
            pooled = []
        out: list[Any] = [None] * len(plans)
        for i, p in enumerate(plans):
            if i not in pooled:
                out[i] = _capture(self.one, p)
        if pooled:
            res = _capture(self._pooled, [plans[i] for i in pooled])
            for j, i in enumerate(pooled):
                out[i] = res if isinstance(res, _Failed) else res[j]
        return out

    def _pooled(self, plans: list) -> list:
        from serve.infer import run_rows, speed_options
        from rsijev.contract import Prediction
        rows, owner = [], []
        for j, p in enumerate(plans):
            rows += p["encoded"]
            owner += [j] * len(p["encoded"])
        opt = speed_options()
        with self._maybe_tok_lock():
            probs = run_rows(self.model, self.tok, rows,
                             max_options=max(p["max_options"] for p in plans),
                             device=self.device, batch_size=self.max_rows,
                             max_tokens=self.max_tokens, sort=opt["sort"],
                             trim_options=opt["trim_options"])
        from serve.infer import fixed_depth
        for p in plans:                      # pooled rows always take the fixed exit
            fixed_depth(self.model, p)
        out = [([], 0) for _ in plans]
        for j, r, pr in zip(owner, rows, probs):
            out[j][0].append(list(Prediction(tuple(pr)).probs))
        return [(ps, sum(len(e["input_ids"]) for e in p["encoded"]))
                for (ps, _), p in zip(out, plans)]


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class GpuWorker:
    """A queue and the one thread that runs the model.

    `window_ms=None` runs one request at a time in arrival order, which is what the
    lock did. A number turns on micro-batching (see the module docstring)."""

    def __init__(self, runner, *, window_ms: float | None = None, max_group: int = 64):
        self.runner = runner
        self.window = None if window_ms is None or window_ms < 0 else window_ms / 1000.0
        self.max_group = max_group
        self._q: queue.Queue = queue.Queue()
        self.stats = {"groups": 0, "requests": 0, "pooled_groups": 0, "max_group": 0}
        self._thread = threading.Thread(target=self._loop, name="rsijev-gpu", daemon=True)
        self._thread.start()

    # -- caller side
    def plan(self, state, questions, images=None):
        """Tokenize and choose a path, in the caller's thread. `images` (validated
        PIL images) is passed on only when there are some, so a runner written for
        text alone keeps working."""
        if images:
            return self.runner.plan(state, questions, images)
        return self.runner.plan(state, questions)

    def enqueue(self, plan) -> Future:
        f: Future = Future()
        self._q.put((time.perf_counter(), plan, f))
        return f

    def __call__(self, state, questions, images=None):
        """Synchronous use, as a scorer: plan here, run on the worker, wait."""
        return self.enqueue(self.plan(state, questions, images)).result()

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=10)

    # -- worker side
    def _gather(self, first) -> list:
        jobs = [first]
        if self.window is None or not self.runner.pools:
            return jobs
        deadline = first[0] + self.window
        full = getattr(self.runner, "full", lambda plans: False)
        # Stop at one forward pass's worth of rows: a request that would only fit in a
        # second pass waits for the next group instead of holding this one back.
        while len(jobs) < self.max_group and not full([p for _, p, _ in jobs]):
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                left = deadline - time.perf_counter()
                if left <= 0:
                    break
                try:
                    item = self._q.get(timeout=left)
                except queue.Empty:
                    break
            if item is None:
                self._q.put(None)
                break
            jobs.append(item)
        return jobs

    def _loop(self) -> None:
        while True:
            first = self._q.get()
            if first is None:
                return
            jobs = self._gather(first)
            live = [j for j in jobs if j[2].set_running_or_notify_cancel()]
            if not live:
                continue
            results = self.runner.run([p for _, p, _ in live])
            self.stats["groups"] += 1
            self.stats["requests"] += len(live)
            self.stats["max_group"] = max(self.stats["max_group"], len(live))
            if len(live) > 1:
                self.stats["pooled_groups"] += 1
            for (_, _, f), r in zip(live, results):
                if isinstance(r, _Failed):
                    f.set_exception(r.exc)
                else:
                    f.set_result(r)


def model_worker(served, *, batch_size: int = 16, window_ms: float | None = None,
                 **runner_kw) -> GpuWorker:
    """The server's worker for a loaded release (serve.server.Served)."""
    runner = ModelRunner(served.model, served.tok, served.enc,
                         spec_max_options=served.meta["spec"]["max_options"],
                         device=served.device, batch_size=batch_size,
                         prep=getattr(served, "prep", None), venc=getattr(served, "venc", None),
                         name=served.name, image_error=getattr(served, "vision_error", None),
                         **runner_kw)
    return GpuWorker(runner, window_ms=window_ms)

