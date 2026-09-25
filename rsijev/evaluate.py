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
            eval_dtype: torch.dtype | None = None
            ) -> list[tuple[Case, Question, Prediction]]:
    """Score every question of every case.

    `eval_dtype` casts the WHOLE model for the pass, tower included. Casting only
    the readout fixes nothing: the ulp originates in the tower's reductions and
    `lm_head` propagates it faithfully. The scorer is already fp32, so this is
    about the tower.
    """
    model.eval()
    pairs = list(iter_questions(cases))
    out: list[tuple[Case, Question, Prediction]] = []
    with eval_precision(model, eval_dtype):
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
