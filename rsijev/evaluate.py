"""Evaluation. PROTECTED — arms may not change how they are scored.

Produces one record per (arm, target, split). Primary numbers are selective:
tie-aware coverage at an error budget, and AURC. Accuracy, ECE, proper scores
and the per-mode decision scores are reported beside them, never instead.

A fitted temperature is applied only to a SEPARATE reported row. It moves ECE a
great deal and selective coverage not at all, because it is monotone within a
question -- folding it into a raw number hides which of the two changed.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from . import metrics as M
from .contract import Case, Prediction, Question, gold_label
from .encode import EncodeConfig, collate, encode_question, iter_questions, unpermute_logits


def _float_tensors(module: torch.nn.Module):
    """Every floating-point parameter AND buffer, by name. The cast set."""
    from itertools import chain
    for name, t in chain(module.named_parameters(), module.named_buffers()):
        if t is not None and t.is_floating_point():
            yield name, t


def _bitwise_checksum(module: torch.nn.Module) -> tuple:
    """A bit-exact fingerprint of EVERY floating tensor, not a prefix of them.

    Checking only the first few was worse than checking none: on a real
    DecisionModel the first hundreds of parameters all belong to the tower, so a
    corrupted scorer -- the only part that is trained -- sat entirely outside the
    checked set and the guard reported success. The checked set and the cast set
    are now the same set by construction.
    """
    out = []
    for name, t in _float_tensors(module):
        flat = t.detach().reshape(-1)[:4096]
        bits = flat.view(torch.int16 if flat.element_size() == 2 else torch.int32)
        out.append((name, str(t.dtype), int(bits.to(torch.int64).sum())))
    return tuple(out)


@contextmanager
def eval_precision(module: torch.nn.Module, dtype: torch.dtype | None,
                   verify: bool = True) -> Iterator[None]:
    """Run an evaluation pass at a different precision, then restore exactly.

    Why this exists: the tower's GEMM reductions move the final hidden state by
    one bf16 ulp when the tensor SHAPE changes, so the same question scored in a
    differently-composed batch can flip its argmax. At a logit in [8,16) that ulp
    is 2^-4 = 0.0625, which is enough for a near-tie. Evaluating the tower in
    fp32 removes the effect by construction rather than by remembering to pin the
    batch size.

    **Per-tensor dtypes, not one dtype for the module.** DecisionModel is mixed
    precision on purpose: the tower is bf16 and the trained scorer is fp32.
    Restoring the module with a single `module.to(...)` taken from the first
    parameter sends the scorer back as bf16, which rounds the trained weights and
    is not a restore at all. Every tensor is recorded and restored to its own
    dtype.

    **Up-casts only.** bf16 -> fp32 -> bf16 is exact, fp32 having strictly more
    mantissa bits over the same exponent range. The REVERSE is not: fp32 ->
    bf16 -> fp32 discards 16 mantissa bits and does not come back. Qwen3.5 is
    mixed on load -- 321 bf16 parameters and 2 fp32 buffers, the rotary
    inv_freq -- so a bf16 request would quietly compute RoPE from bf16
    frequencies, a numeric path nothing else has ever used. That is refused at
    entry rather than caught at exit, because the call site is where it can be
    fixed. To evaluate at the model's own precision, pass None.
    """
    if dtype is None:
        yield
        return
    tensors = list(_float_tensors(module))
    if not tensors or all(t.dtype == dtype for _, t in tensors):
        yield
        return
    target = torch.finfo(dtype)
    lossy = [(n, t.dtype) for n, t in tensors
             if t.dtype != dtype and (torch.finfo(t.dtype).eps < target.eps
                                      or torch.finfo(t.dtype).max > target.max)]
    if lossy:
        kinds = sorted({str(d) for _, d in lossy})
        raise ValueError(
            f"eval_precision({dtype}) would DOWN-cast {len(lossy)} tensor(s) of dtype "
            f"{kinds}, e.g. {[n for n, _ in lossy[:3]]}. That is lossy and does not "
            f"restore: fp32 -> bf16 -> fp32 discards 16 mantissa bits. Pass None to "
            f"evaluate at the model's own precision.")
    saved = {name: t.dtype for name, t in tensors}
    before = _bitwise_checksum(module) if verify else None
    for _, t in tensors:
        t.data = t.data.to(dtype)
    try:
        yield
    finally:
        for name, t in tensors:
            t.data = t.data.to(saved[name])
        wrong = [n for n, t in tensors if t.dtype != saved[n]]
        if wrong:
            raise RuntimeError(
                f"{type(module).__name__}: {len(wrong)} tensor(s) not restored to their own "
                f"dtype, e.g. {wrong[:3]}")
        if verify and _bitwise_checksum(module) != before:
            raise RuntimeError(
                f"{type(module).__name__}: the round trip through {dtype} was not bit-exact; "
                "the weights this worker trains next are not the ones it had")


@dataclass
class ModeReport:
    mode: str
    n: int
    accuracy: float
    base_rate: float
    decision_score: float
    coverage_at_5: float
    aurc: float
    ece: float
    mean_log_score: float
    mean_brier: float
    auc: float | None = None
    spearman_expected: float | None = None
    spearman_argmax: float | None = None


@dataclass
class ArmReport:
    target: str
    split: str
    temperature: float | None
    per_mode: dict[str, ModeReport] = field(default_factory=dict)
    pooled_coverage_at_5: float = float("nan")
    pooled_aurc: float = float("nan")

    def min_decision_score(self) -> float:
        return min(r.decision_score for r in self.per_mode.values())


@torch.no_grad()
def predict(model, tokenizer, cases: Sequence[Case], enc: EncodeConfig, *,
            max_options: int, device: str = "cuda", batch_size: int = 16,
            temperature: float = 1.0,
            eval_dtype: torch.dtype | None = None,
            fast: bool | str = False, prefix_cache: bool = False,
            token_budget: int | None = None, max_rows: int | None = None,
            compile: bool = False, stats: dict | None = None
            ) -> list[tuple[Case, Question, Prediction]]:
    """Score every question of every case.

    `eval_dtype` casts the WHOLE model for the pass, tower included. Casting only
    the readout fixes nothing: the ulp originates in the tower's reductions and
    `lm_head` propagates it faithfully. The scorer is already fp32, so this is
    about the tower.

    `fast` selects the accelerated path (`_predict_fast`): length-bucketed
    batches, rows returned in their original order. It is NOT bit-identical to
    the historical batching: the tower's kernels depend on batch shape even in
    fp32. Measured on rc-B over the whole v2 suite (20,919 questions): 2 argmax
    flips, max |dp| 2.3e-3, |d suite_mean| <= 8e-5, 1.7x faster. "auto" takes
    it only when the tower evaluates in fp32; in bf16 a shape change moves the
    hidden state by a full ulp, so a bf16 pass keeps the historical batching
    unless `fast=True` asks otherwise. With `fast=False` (the default) this
    function is exactly what it was.

    `prefix_cache` (default off) additionally encodes each case's shared state
    once. On rc-B its built-in self-check rejected it on every benchmark where
    it applied (|dp| up to 8e-4 against the uncached rows), so it falls back
    and buys nothing; it stays behind the flag.
    """
    model.eval()
    pairs = list(iter_questions(cases))
    out: list[tuple[Case, Question, Prediction]] = []
    with eval_precision(model, eval_dtype):
        use_fast = fast is True or (fast == "auto" and _tower_is_fp32(model))
        if stats is not None:
            stats["eval_path"] = "fast" if use_fast else "legacy"
        if use_fast:
            return _predict_fast(model, tokenizer, pairs, enc, max_options=max_options,
                                 device=device, batch_size=batch_size,
                                 temperature=temperature, prefix_cache=prefix_cache,
                                 token_budget=token_budget, max_rows=max_rows,
                                 compile=compile, stats=stats)
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            batch = collate(tokenizer,
                            [encode_question(tokenizer, c.state, q, enc) for c, q in chunk],
                            max_options=max_options, device=device)
            logits = unpermute_logits(model(**batch), batch["option_perm"],
                                      batch["option_mask"]) / temperature
            probs = F.softmax(logits, dim=-1)
            for r, (c, q) in enumerate(chunk):
                p = probs[r, : len(q.options)].float().tolist()
                out.append((c, q, Prediction(tuple(p))))
    return out


# The gated-delta-net chunk length, in both the fla kernels and transformers'
# torch reference. A cached prefix is cut to a multiple of it, so the suffix's
# chunk grid is the grid the uncached pass would have used: the linear layers
# then add the same chunks in the same order and the recurrent state handed
# across the cut is the one the uncached pass carries internally.
LINEAR_ATTN_CHUNK = 64
# The cached path's first batch in every call is checked against the uncached
# path; beyond this, or on any argmax change, the call runs uncached.
PREFIX_SELFCHECK_TOL = 1e-4


def _tower_is_fp32(model) -> bool:
    tower = getattr(model, "tower", model)
    ts = [t for _, t in _float_tensors(tower)]
    return bool(ts) and all(t.dtype == torch.float32 for t in ts)


def _cacheable_tower(model):
    """The tower, if a cache can be threaded through it; else None.

    The cache is injected at the TOWER, not through DecisionModel's own
    arguments, so it works with any arm's arch.py (most snapshots predate
    `past_key_values` in `_compute`), as long as the arch calls its tower once
    per forward -- which `_TowerContinuation` checks rather than assumes.
    """
    import inspect
    tower = getattr(model, "tower", None)
    if not isinstance(tower, torch.nn.Module):
        return None
    params = inspect.signature(tower.forward).parameters
    if not all(k in params for k in ("past_key_values", "position_ids", "use_cache")):
        return None
    return tower


class _TowerContinuation:
    """Make the tower's next call continue from a prefix cache.

    Inside the block the arch sees an ordinary batch -- the suffix rows, indices
    relative to the suffix -- and calls its tower as it always does. The tower
    receives the cache, positions that continue past the prefix, and a mask
    that also covers the prefix. The hidden states it returns cover only the
    suffix, which is what the shifted indices address. A second tower call
    inside one block would read a cache the first call already advanced, so it
    raises instead.
    """

    def __init__(self, tower, cache, prefix_len: int):
        self.tower, self.cache, self.L, self.calls = tower, cache, prefix_len, 0

    def __enter__(self):
        orig = self.tower.forward

        def forward(input_ids=None, attention_mask=None, **kw):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("the arch called its tower twice in one forward; "
                                   "a prefix cache cannot serve that")
            rows, width = input_ids.shape
            dev = input_ids.device
            if attention_mask is None:
                attention_mask = torch.ones((rows, width), dtype=torch.long, device=dev)
            mask = torch.cat([torch.ones((rows, self.L), dtype=attention_mask.dtype, device=dev),
                              attention_mask], dim=1)
            pos = (torch.arange(width, device=dev) + self.L).unsqueeze(0).expand(rows, width)
            kw.update(past_key_values=self.cache, use_cache=True, position_ids=pos)
            return orig(input_ids=input_ids, attention_mask=mask, **kw)

        self._prev = self.tower.__dict__.get("forward")
        self.tower.forward = forward          # instance attribute shadows the method
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            del self.tower.forward
        else:
            self.tower.forward = self._prev
        return False


def _cached_prefix_len(tokenizer, state: str, enc: EncodeConfig, encoded: list[dict]) -> int:
    """How many leading tokens every question of this case shares, cut to the
    linear-attention chunk grid, or 0 when there is no provably shared prefix.

    The same guard as serve.infer._shared_prefix: the layout must put the state
    first, every question must start with exactly the tokens of `state + "\\n\\n"`
    (so the tokenizer did not merge across the boundary and no question was
    truncated), and nothing the readout reads may sit inside the prefix.
    """
    if enc.layout != "state_first" or not state.strip() or len(encoded) < 2:
        return 0
    ids = tokenizer(f"{state}\n\n", add_special_tokens=False)["input_ids"]
    for e in encoded:
        if e["input_ids"][:len(ids)] != ids:
            return 0
        lo = min([e["decision_index"], *e["option_index"],
                  *(a for a, _ in e.get("option_span") or [])])
        if lo < len(ids):
            return 0
    return (len(ids) // LINEAR_ATTN_CHUNK) * LINEAR_ATTN_CHUNK


def _shift(e: dict, n: int) -> dict:
    """The same example with its first n tokens moved into a cache."""
    return {**e, "input_ids": e["input_ids"][n:],
            "option_index": [i - n for i in e["option_index"]],
            "option_span": [(a - n, b - n) for a, b in e.get("option_span") or []],
            "decision_index": e["decision_index"] - n}


def _pack(units: list[tuple], *, max_rows: int, token_budget: int) -> list[list[tuple]]:
    """Greedy packing of (rows, cost_len, payload) units, already sorted longest
    first, under a row cap and a padded-token budget (rows x longest cost)."""
    batches, cur, rows, width = [], [], 0, 0
    for u in units:
        n, length = u[0], u[1]
        w = max(width, length)
        if cur and (rows + n > max_rows or (rows + n) * w > token_budget):
            batches.append(cur)
            cur, rows, w = [], 0, length
        cur.append(u)
        rows += n
        width = w
    if cur:
        batches.append(cur)
    return batches


def _predict_fast(model, tokenizer, pairs, enc: EncodeConfig, *, max_options: int,
                  device: str, batch_size: int, temperature: float,
                  prefix_cache: bool, token_budget: int | None, max_rows: int | None,
                  compile: bool, stats: dict | None):
    """`predict`, with the same answers computed in less time.

    1. Length buckets. Rows are sorted by length and packed under a padded-token
       budget, so a batch is never padded to one long outlier. The default
       budget, batch_size x max_length, is the largest batch the historical
       path can already build, so peak memory does not grow.
    2. Shared state once. Every question of a case is state + question, so the
       state is a true prefix. Cases whose questions share it are grouped by
       the cached length; their prefixes run once, unpadded (equal length by
       construction), and every question continues on its own row of that
       cache. The cache is hybrid: full-attention layers hold keys and values,
       the gated-delta-net layers hold a causal-conv window and a recurrent
       state. Both are carried: `Cache.reorder_cache` broadcasts every one of
       them from a case to its questions, and the tower's linear layers take the
       conv window and the recurrent state as their initial state when the cache
       has previous state. The cut is on the 64-token chunk grid, see
       LINEAR_ATTN_CHUNK. Padding never enters a prefix: prefixes in one pass
       have equal length, and suffix padding is at the right, after every
       position the readout reads.
    3. A self-check. The first cached batch of every call is also scored the
       historical way; if any argmax differs or a probability moves by more
       than PREFIX_SELFCHECK_TOL, the cache is dropped for the whole call and
       every row is scored uncached. An arch the cache cannot serve therefore
       costs one batch, never a wrong number.
    4. Rows are written back to their original positions.
    """
    import time as _time
    t0 = _time.perf_counter()
    max_rows = max_rows or 4 * batch_size
    token_budget = token_budget or batch_size * enc.max_length
    fwd = torch.compile(model, dynamic=True) if compile else model
    encoded = [encode_question(tokenizer, c.state, q, enc) for c, q in pairs]

    # group the pairs by case, in order; iter_questions yields each case contiguously
    by_case: list[list[int]] = []
    for i, (c, _) in enumerate(pairs):
        if by_case and pairs[by_case[-1][0]][0] is c:
            by_case[-1].append(i)
        else:
            by_case.append([i])

    tower = _cacheable_tower(model) if (prefix_cache and not compile) else None
    plain: list[int] = []
    cached: dict[int, list[tuple]] = {}     # L -> [(rows, L + longest suffix, idxs)]
    for idxs in by_case:
        L = (_cached_prefix_len(tokenizer, pairs[idxs[0]][0].state, enc,
                                [encoded[i] for i in idxs]) if tower is not None else 0)
        if L == 0:
            plain += idxs
            continue
        # a case with more questions than a batch holds is split; each piece
        # runs the prefix once, still one pass per piece rather than per question
        for j in range(0, len(idxs), max_rows):
            part = idxs[j:j + max_rows]
            cached.setdefault(L, []).append(
                (len(part), max(len(encoded[i]["input_ids"]) for i in part), part))

    probs_out: list[list[float] | None] = [None] * len(pairs)
    counts = {"batches": 0, "prefix_tokens": 0, "tokens": 0}

    def plain_probs(idxs):
        batch = collate(tokenizer, [encoded[i] for i in idxs],
                        max_options=max_options, device=device)
        counts["tokens"] += int(batch["attention_mask"].sum())
        counts["batches"] += 1
        logits = unpermute_logits(fwd(**batch), batch["option_perm"], batch["option_mask"])
        return F.softmax(logits / temperature, dim=-1)

    def cached_probs(parts, L):
        pids = torch.tensor([encoded[p[0]]["input_ids"][:L] for p in parts],
                            dtype=torch.long, device=device)
        cache = tower(input_ids=pids, use_cache=True).past_key_values
        counts["prefix_tokens"] += pids.numel()
        row_of = [k for k, p in enumerate(parts) for _ in p]
        cache.reorder_cache(torch.tensor(row_of, dtype=torch.long, device=device))
        idxs = [i for p in parts for i in p]
        batch = collate(tokenizer, [_shift(encoded[i], L) for i in idxs],
                        max_options=max_options, device=device)
        counts["tokens"] += int(batch["attention_mask"].sum())
        counts["batches"] += 1
        with _TowerContinuation(tower, cache, L):
            logits = unpermute_logits(model(**batch), batch["option_perm"],
                                      batch["option_mask"])
        return idxs, F.softmax(logits / temperature, dim=-1)

    def emit(idxs, probs):
        for r, i in enumerate(idxs):
            probs_out[i] = probs[r, :len(pairs[i][1].options)].float().tolist()

    selfcheck = None
    n_cached_rows = 0
    for L in sorted(cached):
        units = sorted(cached[L], key=lambda u: -u[1])
        for b in _pack(units, max_rows=max_rows, token_budget=token_budget):
            parts = [u[2] for u in b]
            if selfcheck is False:
                plain += [i for p in parts for i in p]
                continue
            try:
                idxs, probs = cached_probs(parts, L)
            except RuntimeError as e:
                selfcheck, reason = False, f"cached pass raised: {e}"
                plain += [i for p in parts for i in p]
                continue
            if selfcheck is None:
                ref = plain_probs(idxs)
                n = probs.shape[1]
                finite = torch.isfinite(ref[:, :n]) & (ref[:, :n] > 0)
                dp = float((probs - ref).abs().masked_fill(~finite, 0).max())
                same = bool((probs.argmax(-1) == ref.argmax(-1)).all())
                if same and dp <= PREFIX_SELFCHECK_TOL:
                    selfcheck, reason = True, f"max|dp| {dp:.2e} on {len(idxs)} rows"
                else:
                    selfcheck, reason = False, f"max|dp| {dp:.2e}, argmax same {same}"
                    emit(idxs, ref)
                    continue
            emit(idxs, probs)
            n_cached_rows += len(idxs)

    order = sorted(plain, key=lambda i: -len(encoded[i]["input_ids"]))
    units = [(1, len(encoded[i]["input_ids"]), i) for i in order]
    for b in _pack(units, max_rows=max_rows, token_budget=token_budget):
        idxs = [u[2] for u in b]
        emit(idxs, plain_probs(idxs))

    missing = [i for i, p in enumerate(probs_out) if p is None]
    if missing:
        raise RuntimeError(f"fast eval lost {len(missing)} rows, e.g. {missing[:3]}")
    if stats is not None:
        stats.update(n_questions=len(pairs), n_batches=counts["batches"],
                     n_cached_rows=n_cached_rows, prefix_tokens=counts["prefix_tokens"],
                     tokens=counts["tokens"],
                     legacy_tokens=sum(len(e["input_ids"]) for e in encoded),
                     prefix_selfcheck=None if selfcheck is None else
                     ("pass: " if selfcheck else "FAIL, uncached: ") + reason,
                     seconds=round(_time.perf_counter() - t0, 2))
    return [(c, q, Prediction(tuple(p))) for (c, q), p in zip(pairs, probs_out)]

def _base_rate(items: Sequence[tuple[Question, Sequence[float]]]) -> float:
    """What guessing gets: the majority-class rate, computed WITHIN each label
    space and then weighted by how many questions use it.

    Pooling across label spaces is wrong and not harmlessly so. typed-decisions
    mixes four workflows whose choice options are entirely different sets, and a
    pooled majority over their union reads 0.123 -- far below any real guessing
    rate, which would inflate every choice decision score built on it.
    """
    groups: dict[tuple[str, ...], dict[str, int]] = {}
    for q, g in items:
        counts = groups.setdefault(q.options, {})
        lab = gold_label(q, g)
        counts[lab] = counts.get(lab, 0) + 1
    if not groups:
        return float("nan")
    # A label space with a handful of questions has an upward-biased majority --
    # two questions sharing an answer read as a 100% guessing rate. Spaces below
    # `min_group` are pooled into one remainder instead of each contributing a
    # small-sample maximum.
    min_group = 20
    big = {k: v for k, v in groups.items() if sum(v.values()) >= min_group}
    small: dict[str, int] = {}
    for k, v in groups.items():
        if k not in big:
            for lab, n in v.items():
                small[lab] = small.get(lab, 0) + n
    total = sum(sum(c.values()) for c in groups.values())
    hits = sum(max(c.values()) for c in big.values()) + (max(small.values()) if small else 0)
    return hits / total


def score_predictions(preds: Sequence[tuple[Case, Question, Prediction]], *,
                      target: str, split: str,
                      temperature: float | None = None) -> ArmReport:
    rep = ArmReport(target=target, split=split, temperature=temperature)
    by_mode: dict[str, list[tuple[Case, Question, Prediction]]] = {}
    for c, q, p in preds:
        by_mode.setdefault(q.mode, []).append((c, q, p))

    pooled_conf, pooled_corr = [], []
    for mode, items in by_mode.items():
        conf = [p.confidence() for _, _, p in items]
        corr = [p.choice(q.options) == gold_label(q, c.gold[q.key])
                for c, q, p in items]
        pooled_conf += conf
        pooled_corr += corr

        acc = sum(corr) / len(corr)
        base = _base_rate([(q, c.gold[q.key]) for c, q, _ in items])
        rc = M.risk_coverage(conf, corr)
        logs = [M.log_score(p.probs, c.gold[q.key]) for c, q, p in items]
        briers = [M.brier(p.probs, c.gold[q.key]) for c, q, p in items]

        auc = sp_e = sp_a = None
        if mode == "noul":
            labels = [gold_label(q, c.gold[q.key]) == "true" for c, q, _ in items]
            auc = M.auc([p.noul() for _, _, p in items], labels)
            ds = M.decision_score("noul", auc_value=auc)
        elif mode == "score":
            pred_e = [p.score() for _, _, p in items]
            gold_e = [sum(i * g for i, g in enumerate(c.gold[q.key])) for c, q, _ in items]
            gold_a = [float(max(range(len(c.gold[q.key])),
                                key=c.gold[q.key].__getitem__)) for c, q, _ in items]
            sp_e, sp_a = M.spearman(pred_e, gold_e), M.spearman(pred_e, gold_a)
            ds = M.decision_score("score", rho=sp_e)
        else:
            ds = M.decision_score("choice", accuracy=acc, base_rate=base)

        rep.per_mode[mode] = ModeReport(
            mode=mode, n=len(items), accuracy=acc, base_rate=base, decision_score=ds,
            coverage_at_5=M.coverage_at_error(conf, corr, 0.05), aurc=rc.aurc,
            ece=M.ece(conf, corr), mean_log_score=sum(logs) / len(logs),
            mean_brier=sum(briers) / len(briers), auc=auc,
            spearman_expected=sp_e, spearman_argmax=sp_a)

    rep.pooled_coverage_at_5 = M.coverage_at_error(pooled_conf, pooled_corr, 0.05)
    rep.pooled_aurc = M.risk_coverage(pooled_conf, pooled_corr).aurc
    return rep


def _resolve_code_version() -> str:
    """The commit this PROCESS imported, resolved once at import time.

    Resolving it when the record is written reports whatever is checked out at
    that moment, which is not what produced the record: a worker mid-arm keeps
    running the code it imported while a deploy moves the tree underneath it,
    and every record written across that deploy is stamped with code it never
    ran. Workers restart on a code change, so import time is exactly right.
    """
    import subprocess
    from pathlib import Path
    try:
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        head = out.stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


CODE_VERSION = _resolve_code_version()


def as_record(rep: ArmReport, **extra) -> dict:
    """Flatten into one experiment-log row."""
    rec = {"target": rep.target, "split": rep.split, "temperature": rep.temperature,
           "pooled_coverage_at_5": rep.pooled_coverage_at_5,
           "pooled_aurc": rep.pooled_aurc,
           "min_decision_score": rep.min_decision_score(),
           "code_version": CODE_VERSION}
    for mode, r in rep.per_mode.items():
        for k, v in r.__dict__.items():
            if k != "mode":
                rec[f"{mode}.{k}"] = v
    rec.update(extra)
    return rec
