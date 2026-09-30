"""The serving speed paths: exact where they claim to be, order-preserving always.

    python -m pytest tests/test_serve_speed.py -q

The tokenizer checks need the Qwen3.5 tokenizer in the Hugging Face cache (no
weights) and skip without it. Everything else runs on stand-ins.
"""
from __future__ import annotations

import random
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.encode import EncodeConfig, encode_question            # noqa: E402
from serve import infer                                             # noqa: E402
from serve.wire import parse_questions                              # noqa: E402


@pytest.fixture(scope="module")
def tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
    except Exception as e:                       # offline without the tokenizer cached
        pytest.skip(f"Qwen3.5 tokenizer not available: {e}")


# Text that stresses the pre-tokenizer at and around the state/question boundary:
# trailing spaces, tabs and newlines, punctuation runs, contractions, digits,
# combining marks, CJK, emoji, CRLF, special-token text, and leading whitespace in
# the instructions (which must fall back).
PIECES = ["Hello", " world", ".", "...", "!!", "'s", "'ll", " 42", "3", "\t", "  ", "\n",
          "\r\n", "é", "é", "́", "日本語", "🙂", "<|endoftext|>", "#", "--",
          "don't", " ", "x", " ", "}", "{\"a\":1}", " ", "\x1c", "Ω", "١٢"]


def _text(rng, n):
    return "".join(rng.choice(PIECES) for _ in range(n))


def _question(rng, i):
    ins = _text(rng, rng.randint(0, 8))
    kind = rng.choice(["noul", "choice", "score"])
    if kind == "noul":
        return {"type": "noul", "instructions": ins or "?"}
    k = rng.randint(2, 6)
    if kind == "score":
        return {"type": "score", "instructions": ins or "?",
                "criteria": [_text(rng, rng.randint(0, 3)) or "lvl" for _ in range(k)]}
    return {"type": "choice", "instructions": ins or "?",
            "criteria": {f"{_text(rng, 1).strip() or 'o'}{j}": (_text(rng, rng.randint(0, 3)) or None)
                         for j in range(k)}}


@pytest.mark.parametrize("order", ["canonical", "reversed"])
def test_encode_questions_matches_encode_question(tok, order):
    """Row for row, the ids and every index equal encode_question's."""
    assert infer.splits_after_blank_line(tok), "the Qwen3.5 tokenizer is the one this is for"
    rng = random.Random(0)
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order=order)
    fast = fallback = 0
    for trial in range(300):
        state = _text(rng, rng.randint(1, 40))
        if not state.strip():
            state += "x"
        qs = parse_questions({f"q{i}": _question(rng, i) for i in range(rng.randint(1, 6))})
        got, lead = infer.encode_questions(tok, state, qs, enc)
        want = [encode_question(tok, state, q, enc) for q in qs]
        assert got == want, f"trial {trial}: {state!r}"
        assert lead == tok(f"{state}\n\n", add_special_tokens=False)["input_ids"]
        for q in qs:
            if q.instructions[:1].isprintable() and q.instructions[:1].strip():
                fast += 1
            else:
                fallback += 1
    assert fast > 500 and fallback > 50                       # both branches exercised


def test_encode_questions_real_requests(tok):
    """The documented and benchmarked requests, including a long state."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "docs" / "assets"))
    from bench_gb10 import DOCS, QUESTIONS
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical")
    qs = parse_questions(QUESTIONS)
    for state in DOCS.values():
        got, _ = infer.encode_questions(tok, state, qs, enc)
        assert got == [encode_question(tok, state, q, enc) for q in qs]


def test_encode_questions_truncation_falls_back(tok):
    enc = EncodeConfig(layout="state_first", option_pool="mean", option_order="canonical",
                       max_length=64)
    qs = parse_questions({"a": {"type": "noul", "instructions": "Is it long?"}})
    state = "A very long document. " * 50
    got, _ = infer.encode_questions(tok, state, qs, enc)
    assert got == [encode_question(tok, state, qs[0], enc)]
    assert len(got[0]["input_ids"]) == 64


def test_other_layouts_and_empty_state_fall_back(tok):
    qs = parse_questions({"a": {"type": "noul", "instructions": "Refund?"}})
    for enc, state in ((EncodeConfig(layout="options_first"), "Charged twice."),
                       (EncodeConfig(option_order="shuffled"), "Charged twice."),
                       (EncodeConfig(), "   ")):
        got, lead = infer.encode_questions(tok, state, qs, enc)
        assert lead is None
        assert got == [encode_question(tok, state, q, enc) for q in qs]


def test_unknown_tokenizer_is_not_trusted():
    class Slow:
        pass
    assert infer.splits_after_blank_line(Slow()) is False


# ---------------------------------------------------------------- row order
def test_row_order_default_is_request_order():
    assert infer.row_order([5, 1, 9, 2, 7], 2) == [[0, 1], [2, 3], [4]]


def test_row_order_sorted_groups_similar_lengths():
    lengths = [5, 100, 6, 99, 7, 98]
    got = infer.row_order(lengths, 3, sort=True)
    assert got == [[1, 3, 5], [4, 2, 0]]
    assert sorted(i for b in got for i in b) == list(range(6))


class RowModel(torch.nn.Module):
    """A batch-invariant stand-in: each option's logit is a function of its own row's
    tokens only, so any batching must give the same numbers."""
    cfg = type("C", (), {"residual": False})()

    def eval(self):
        return self

    def forward(self, *, input_ids, attention_mask, option_index, option_mask, **_):
        x = (input_ids * attention_mask).float()
        s = x.sum(1, keepdim=True)
        lg = torch.log1p(option_index.float() + 1) * torch.sin(s / 7.0)
        return lg.masked_fill(~option_mask, float("-inf"))


class FakeTok:
    pad_token_id = 0
    eos_token_id = 0


def _rows(rng, n):
    rows = []
    for _ in range(n):
        k = rng.randint(2, 7)
        L = rng.randint(k + 2, 40)
        ids = [rng.randint(1, 50) for _ in range(L)]
        oi = sorted(rng.sample(range(1, L - 1), k))
        rows.append({"input_ids": ids, "option_index": oi,
                     "option_span": [(i, i + 1) for i in oi], "decision_index": L - 1,
                     "options": [str(j) for j in range(k)], "option_perm": list(range(k)),
                     "mode": "choice"})
    return rows


@pytest.mark.parametrize("sort,trim", [(True, False), (False, True), (True, True)])
def test_run_rows_restores_order(sort, trim):
    rng = random.Random(1)
    rows = _rows(rng, 23)
    kw = dict(max_options=12, device="cpu", batch_size=5)
    base = infer.run_rows(RowModel(), FakeTok(), rows, **kw)
    got = infer.run_rows(RowModel(), FakeTok(), rows, sort=sort, trim_options=trim, **kw)
    assert [len(p) for p in got] == [len(r["option_index"]) for r in rows]
    for a, b in zip(base, got):
        assert a == pytest.approx(b, abs=1e-6)


def test_speed_options_come_from_the_environment(monkeypatch):
    for k in ("RSIJEV_SORT_ROWS", "RSIJEV_TRIM_OPTIONS", "RSIJEV_FAST_ENCODE"):
        monkeypatch.delenv(k, raising=False)
    assert infer.speed_options() == {"sort": False, "trim_options": False, "fast_encode": True}
    monkeypatch.setenv("RSIJEV_SORT_ROWS", "1")
    monkeypatch.setenv("RSIJEV_TRIM_OPTIONS", "1")
    monkeypatch.setenv("RSIJEV_FAST_ENCODE", "0")
    assert infer.speed_options() == {"sort": True, "trim_options": True, "fast_encode": False}


def test_row_order_token_budget():
    lengths = [10, 10, 10, 50, 10]
    got = infer.row_order(lengths, 8, max_tokens=100)
    assert got == [[0, 1, 2], [3, 4]]
    for b in got:
        assert len(b) * max(lengths[i] for i in b) <= 100 or len(b) == 1
    assert infer.row_order([500], 8, max_tokens=100) == [[0]]     # one long row still runs


# ---------------------------------------------------------------- HTTP front
from fastapi.testclient import TestClient                           # noqa: E402

from serve import app as appmod                                     # noqa: E402
from serve.batcher import CallRunner, GpuWorker                     # noqa: E402

NOUL = {"type": "noul", "instructions": "Refund?"}
CHOICE = {"type": "choice", "instructions": "Team?",
          "criteria": {"billing": "Payments", "technical": "Bugs", "other": None}}


def _scorer(state, questions):
    out = []
    for i, q in enumerate(questions):
        raw = [1.0 / (j + 1 + len(state) % 3 + i) for j in range(len(q.options))]
        out.append([v / sum(raw) for v in raw])
    return out, 7


def test_fast_json_parses_to_the_same_body():
    import json
    body = {"answers": {"a": {"p": [1e-05, 0.1 + 0.2, 1 / 3, 5e-324, 0.0]}}, "k": "é日"}
    fast = json.loads(appmod.FastJSONResponse(body).body)
    slow = json.loads(appmod.JSONResponse(body).body)
    assert fast == slow == json.loads(json.dumps(body))


def test_non_finite_probabilities_are_still_a_server_error():
    c = TestClient(appmod.create_app(lambda s, q: ([[float("nan"), 1.0]], 1),
                                     served_model_name="m"), raise_server_exceptions=False)
    r = c.post("/v1/systemone", json={"model": "m", "state": "s", "questions": {"q": NOUL}})
    assert r.status_code == 500


def test_worker_serves_concurrent_requests_one_at_a_time_in_order():
    active, seen = [0], []

    def slow(state, questions):
        active[0] += 1
        assert active[0] == 1, "two requests on the model at once"
        time.sleep(0.01)
        seen.append(state)
        active[0] -= 1
        return _scorer(state, questions)
    w = GpuWorker(CallRunner(slow))
    futs = [w.enqueue(w.plan(f"s{i}", parse_questions({"q": NOUL}))) for i in range(8)]
    assert [f.result()[1] for f in futs] == [7] * 8
    assert seen == [f"s{i}" for i in range(8)]
    w.close()


class PoolRunner:
    """Pools everything; records the groups it was handed."""
    pools = True

    def __init__(self):
        self.groups = []

    def plan(self, state, questions):
        return (state, questions)

    def run(self, plans):
        self.groups.append(len(plans))
        time.sleep(0.02)
        return [_scorer(*p) for p in plans]


def test_batch_window_groups_requests_and_splits_answers_back():
    runner = PoolRunner()
    w = GpuWorker(runner, window_ms=50)
    qs = [parse_questions({"q": NOUL, "c": CHOICE}) for _ in range(6)]
    futs = [w.enqueue(w.plan(f"state {i}", qs[i])) for i in range(6)]
    got = [f.result() for f in futs]
    assert got == [_scorer(f"state {i}", qs[i]) for i in range(6)]
    assert max(runner.groups) > 1 and sum(runner.groups) == 6
    w.close()


def test_without_a_window_nothing_is_grouped():
    runner = PoolRunner()
    w = GpuWorker(runner)
    futs = [w.enqueue(w.plan(f"s{i}", parse_questions({"q": NOUL}))) for i in range(5)]
    [f.result() for f in futs]
    assert runner.groups == [1] * 5
    w.close()


def test_an_error_reaches_only_its_own_request():
    def picky(state, questions):
        if state == "bad":
            raise ValueError("cannot fit")
        return _scorer(state, questions)
    w = GpuWorker(CallRunner(picky))
    qs = parse_questions({"q": NOUL})
    good, bad = w.enqueue(w.plan("ok", qs)), w.enqueue(w.plan("bad", qs))
    assert good.result()[1] == 7
    with pytest.raises(ValueError):
        bad.result()
    w.close()


def test_http_answers_equal_the_python_api():
    from serve.decider import Decider
    app = appmod.create_app(_scorer, served_model_name="m")
    c = TestClient(app)
    questions = {"q": NOUL, "c": CHOICE}
    r = c.post("/v1/systemone", json={"model": "m", "state": "Charged twice.",
                                      "questions": questions})
    assert r.status_code == 200
    d = Decider.from_scorer(_scorer, name="m")
    assert r.json() == d.request("Charged twice.", questions)


def test_concurrent_http_requests_all_answer():
    import concurrent.futures as cf
    c = TestClient(appmod.create_app(_scorer, served_model_name="m"))

    def one(i):
        r = c.post("/v1/systemone", json={"model": "m", "state": f"s{i}", "questions": {"q": NOUL}})
        return r.status_code, r.json()["answers"]["q"]["noul"]
    with cf.ThreadPoolExecutor(8) as ex:
        got = list(ex.map(one, range(24)))
    assert all(s == 200 for s, _ in got)
    assert [p for _, p in got] == [
        appmod.to_answer(parse_questions({"q": NOUL})[0], _scorer(f"s{i}", parse_questions({"q": NOUL}))[0][0])["noul"]
        for i in range(24)]


def test_a_group_stops_at_one_forward_pass():
    class Capped(PoolRunner):
        def full(self, plans):
            return len(plans) >= 3
    runner = Capped()
    w = GpuWorker(runner, window_ms=100)
    futs = [w.enqueue(w.plan(f"s{i}", parse_questions({"q": NOUL}))) for i in range(7)]
    [f.result() for f in futs]
    assert max(runner.groups) == 3 and sum(runner.groups) == 7
    w.close()


def test_model_runner_pools_only_plain_requests_by_default(monkeypatch):
    from serve.batcher import ModelRunner
    monkeypatch.delenv("RSIJEV_BATCH_POOL_PREFIX", raising=False)
    r = ModelRunner(None, None, None, spec_max_options=160, device="cpu")
    assert r.poolable({"path": "plain", "prefix": None})
    assert not r.poolable({"path": "cached", "prefix": [1] * 81})
    assert not r.poolable({"path": "doc", "prefix": [1] * 81})
    rows = {"path": "plain", "prefix": None, "encoded": [{"input_ids": [0] * 100}] * 4}
    r.max_rows, r.max_tokens = 32, 1000
    assert not r.full([rows, rows]) and r.full([rows, rows, rows])
