"""Serving on MLX: the same requests, plans, effort semantics and response fields as the
PyTorch path (serve/infer.py, serve/effort.py, serve/batcher.py), with the model in MLX and
no torch anywhere on the path.

    rsi-jev serve PATH --backend mlx           # PATH: an MLX checkpoint (scripts/convert_mlx.py)
    Decider(PATH, backend="mlx")

What runs where:
  planning      serve/plan.py (the PyTorch path's own code): encode_questions, the shared
                prefix check, adaptive_applies, fixed_depth, record_confidence
  text rows     "plain": each question read in full, in run_rows' batches (row_order, the
                same batch size and token cap); "cached": the state read once
                (MLXDecisionModel.encode_prefix) and every question continued from it, by
                the same rule as PyTorch (shared prefix, >= 2 questions, (Q - 1) x prefix
                >= RSIJEV_MIN_SAVED_TOKENS, default 480)
  adaptive exit the release's policy (meta.json adaptive + calibration.json exits, as
                serve.release.adaptive_policy builds it); unset effort = --adaptive (auto:
                multi-question requests), effort low / medium / high / auto as
                serve/effort.py: low and medium are forced exits, high the main exit, auto
                the cascade with meta.json auto_thresholds
  images        full depth (the aux heads read text only), the vision tower once per
                request, each question read in full with its M-RoPE positions

Not on this path yet (PyTorch only): the document cache (--profile agent), micro-batching
across requests, the image prefix cache.
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np

from rsijev.contract import Prediction, Question
from rsijev.encode import EncodeConfig, encode_question

from .collate import collate, softmax, suffixes, unpermute
from .model import MLXDecisionModel, Policy

DEFAULT_MAX_OPTIONS = 80
MIN_SAVED_TOKENS = 480
FORWARD_MAX_TOKENS = int(os.environ.get("RSIJEV_FORWARD_MAX_TOKENS", "32768"))


def min_saved_tokens() -> int:
    v = os.environ.get("RSIJEV_MIN_SAVED_TOKENS")
    return int(v) if v is not None and v.strip() else MIN_SAVED_TOKENS


def is_mlx_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "mlx.json").exists()


# ----------------------------------------------------------------------------- policy
def attach_policy(model: MLXDecisionModel, meta: dict, ckpt: Path, fixed_exit=None,
                  adaptive=None):
    """serve.release.adaptive_policy for the MLX model: sets adaptive_policy, adaptive_mode,
    effort_base, effort_auto_taus. Returns (policy or None, why it is off or None)."""
    from serve.release import (ADAPTIVE_CAL_FILE, CAL_JSON, adaptive_block, adaptive_mode,
                               auto_thresholds, exit_logts)
    model.adaptive_mode = "off"
    model.effort_base = None
    model.effort_auto_taus = None
    model.adaptive_policy = None
    if not model.cfg.aux_exits:
        return None, None
    exits = model.exit_indices()
    temps = exit_logts(ckpt, exits)                     # validated even when off
    mode, src = adaptive_mode(meta, adaptive, fixed_exit)
    block = adaptive_block(meta)
    if not block or block.get("tau") is None:
        return None, ("the release has aux exits but no tuned tau (meta.json adaptive.tau)"
                      if mode != "off" else f"fixed exit forced ({src})")

    def build():
        if block.get("exits") is not None and [int(x) for x in block["exits"]] != exits:
            raise RuntimeError(f"adaptive.exits {block['exits']} do not match the model's exits {exits}")
        cal_file = block.get("calibration")
        if temps is not None:
            if cal_file not in (None, CAL_JSON):
                raise RuntimeError(f"calibration.json has per-exit temperatures and meta.json "
                                   f"adaptive.calibration names {cal_file}: one source only")
            if (ckpt / ADAPTIVE_CAL_FILE).exists():
                raise RuntimeError(f"both calibration.json \"exits\" and {ADAPTIVE_CAL_FILE}: one source only")
            if model.cal_mode == "none":
                raise RuntimeError("per-exit temperatures need the main head calibrated too "
                                   "(calibration.json cal_mode)")
            cal = {L: {"logT": float(t)} for L, t in temps.items()}
        else:
            p = ckpt / (cal_file or ADAPTIVE_CAL_FILE)
            if not p.exists():
                raise RuntimeError(f"adaptive exit needs {p.name} (one cal-4b per aux exit) or "
                                   f"per-exit temperatures in calibration.json")
            flat = mx.load(str(p))
            cal = {}
            for L in exits[:-1]:
                c = {k.split(".", 1)[1]: v.astype(mx.float32) for k, v in flat.items()
                     if k.split(".", 1)[0] == str(L)}
                if set(c) != {"mean", "W", "mu", "sd", "w", "b"}:
                    raise RuntimeError(f"{p.name}: exit {L} has {sorted(c)}, not a cal-4b")
                cal[L] = c
        return Policy(list(exits), cal, float(block["tau"]))

    if mode == "off":
        try:
            model.effort_base = build()
        except RuntimeError:
            model.effort_base = None
        if model.effort_base is not None:
            model.effort_auto_taus = auto_thresholds(block, model.effort_base.exits)
        return None, f"fixed exit forced ({src})"
    policy = build()
    model.effort_base = policy
    model.effort_auto_taus = auto_thresholds(block, policy.exits)
    model.adaptive_mode = mode
    model.adaptive_policy = policy
    return policy, None


def policy_for(model, effort: str) -> Policy:
    """serve.effort.policy_for with the MLX Policy."""
    from serve.effort import auto_taus
    base = model.effort_base
    cache = model.__dict__.setdefault("_effort_policies", {})
    if effort == "auto":
        if not getattr(model, "effort_auto_taus", None):
            return base
        if "auto" not in cache:
            cache["auto"] = Policy(list(base.exits), dict(base.cal), base.tau, taus=auto_taus(model))
        return cache["auto"]
    if effort not in cache:
        aux = base.exits[:-1]
        if effort == "low":
            cache[effort] = Policy(list(base.exits), dict(base.cal), -1.0)
        elif effort == "medium":
            L = aux[-1]
            cache[effort] = Policy([L, base.exits[-1]], {L: base.cal[L]}, -1.0)
        else:
            raise ValueError(f"no policy for effort {effort!r}")
    return cache[effort]


# ----------------------------------------------------------------------------- planning
def plan_request(tokenizer, state: str, questions: Sequence[Question], enc: EncodeConfig) -> dict:
    """serve.infer.plan_request without the document cache: path "cached" or "plain"."""
    from serve.plan import _shared_prefix, encode_questions, speed_options, truncation_report
    if speed_options()["fast_encode"]:
        encoded, lead = encode_questions(tokenizer, state, questions, enc)
    else:
        encoded, lead = [encode_question(tokenizer, state, q, enc) for q in questions], None
    prefix = _shared_prefix(tokenizer, state, enc, encoded, ids=lead)
    if prefix is None or len(questions) < 2 or (len(questions) - 1) * len(prefix) < min_saved_tokens():
        path = "plain"
    else:
        path = "cached"
    return {"encoded": encoded, "prefix": prefix, "path": path, "doc_cache": None,
            "options": [len(q.options) for q in questions],
            "truncated": truncation_report(encoded, enc)}


def plan_image_request(tokenizer, prep, state: str, images: Sequence,
                       questions: Sequence[Question], enc: EncodeConfig) -> dict:
    """serve.infer.plan_image_request(read_once=False) without torch: the image processor
    once, every question encoded on the expanded state, no image cut."""
    from rsijev.image_text import IMAGE_PAD, expand_state
    from serve.plan import truncation_report
    pv, grid, ntok = prep(images)
    pad = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    encoded = []
    for q in questions:
        e = encode_question(tokenizer, expand_state(state, ntok), q, enc)
        got = sum(1 for t in e["input_ids"] if t == pad)
        if got != sum(ntok):
            raise ValueError(f"{q.key}: {got} image tokens survive of {sum(ntok)} "
                             f"(max_length {enc.max_length} cut into an image; lower the budget)")
        encoded.append(e)
    return {"path": "image", "image_path": "plain", "encoded": encoded, "pixel_values": pv,
            "grid": grid, "options": [len(q.options) for q in questions], "ntok": list(ntok),
            "truncated": truncation_report(encoded, enc)}


# ----------------------------------------------------------------------------- running
def _batches(rows, batch_size, npfx=0):
    from serve.plan import row_order, speed_options
    opt = speed_options()
    lengths = [len(e["input_ids"]) + npfx for e in rows]
    return row_order(lengths, batch_size, opt["sort"], FORWARD_MAX_TOKENS), opt["trim_options"]


def _probs(z: mx.array, batch: dict, temperature: float = 1.0) -> np.ndarray:
    logits = unpermute(np.array(z.astype(mx.float32)), batch["option_perm"], batch["option_mask"])
    return softmax(logits / temperature)


def run_rows(model, tokenizer, rows, *, max_options, batch_size=16, temperature=1.0,
             cache=None, npfx=0) -> list[list[float]]:
    """serve.infer.run_rows: probabilities at the main exit, in the order given."""
    ks = [len(e["option_index"]) for e in rows]
    out: list = [None] * len(rows)
    order, trim = _batches(rows, batch_size, npfx)
    for idx in order:
        part = [rows[i] for i in idx]
        width = max(ks[i] for i in idx) if trim else max_options
        batch = collate(tokenizer, part, width)
        p = _probs(model.forward(batch, cache=cache), batch, temperature)
        for r, i in enumerate(idx):
            out[i] = p[r, :ks[i]].tolist()
    return out


def score_adaptive(model, tokenizer, plan: dict, *, max_options, batch_size=16,
                   temperature=1.0, stats=None):
    """serve.infer.score_adaptive: the staged path with the plan's effort / threshold."""
    if plan.get("effort") is not None:
        policy = policy_for(model, plan["effort"])
    else:
        policy = model.adaptive_policy
    t = plan.get("threshold")
    if isinstance(t, dict):
        policy = Policy(list(policy.exits), dict(policy.cal), policy.tau, taus=dict(t))
    elif t is not None:
        policy = Policy(list(policy.exits), dict(policy.cal), float(t))
    encoded, prefix = plan["encoded"], plan["prefix"]
    cache, npfx = None, 0
    if plan["path"] == "cached":
        npfx = len(prefix)
        cache = model.encode_prefix(prefix)
    rows = suffixes(encoded, npfx) if cache is not None else encoded
    ks = [len(e["option_index"]) for e in rows]
    out: list = [None] * len(rows)
    depth: list = [None] * len(rows)
    order, trim = _batches(rows, batch_size, npfx)
    for idx in order:
        part = [rows[i] for i in idx]
        width = max(ks[i] for i in idx) if trim else max_options
        batch = collate(tokenizer, part, width)
        z, d = model.staged(batch, policy, cache=cache, stats=stats)
        p = _probs(z, batch, temperature)
        for r, i in enumerate(idx):
            out[i] = Prediction(tuple(p[r, :ks[i]].tolist()))
            depth[i] = int(d[r])
    plan["depth"] = depth
    return out, npfx + sum(len(e["input_ids"]) for e in rows)


def score_planned(model, tokenizer, plan: dict, *, max_options, batch_size=16, temperature=1.0):
    """serve.infer.score_planned: adaptive where adaptive_applies, else the fixed exit."""
    from serve.plan import adaptive_applies, fixed_depth, record_confidence
    if adaptive_applies(model, plan):
        preds, tokens = score_adaptive(model, tokenizer, plan, max_options=max_options,
                                       batch_size=batch_size, temperature=temperature)
    else:
        fixed_depth(model, plan)
        encoded, prefix = plan["encoded"], plan["prefix"]
        if plan["path"] == "cached":
            npfx = len(prefix)
            cache = model.encode_prefix(prefix)
            rows = suffixes(encoded, npfx)
            probs = run_rows(model, tokenizer, rows, max_options=max_options, batch_size=batch_size,
                             temperature=temperature, cache=cache, npfx=npfx)
            tokens = npfx + sum(len(e["input_ids"]) for e in rows)
        else:
            probs = run_rows(model, tokenizer, encoded, max_options=max_options,
                             batch_size=batch_size, temperature=temperature)
            tokens = sum(len(e["input_ids"]) for e in encoded)
        preds = [Prediction(tuple(p)) for p in probs]
    record_confidence(model, plan, preds)
    return preds, tokens


def score_image_planned(model, tokenizer, plan: dict, *, max_options, batch_size=16,
                        temperature=1.0):
    """serve.infer.score_image_planned ("plain"): the vision tower once, each question read
    in full at the main exit with its M-RoPE positions."""
    from serve.plan import fixed_depth, record_confidence
    from .vision import mrope_positions
    fixed_depth(model, plan)
    encoded, grid = plan["encoded"], plan["grid"]
    feats = model.image_embeds(plan["pixel_values"], grid)
    out: list[Prediction] = []
    prompt_tokens = 0
    for i in range(0, len(encoded), batch_size):
        part, nopt = encoded[i:i + batch_size], plan["options"][i:i + batch_size]
        prompt_tokens += sum(len(e["input_ids"]) for e in part)
        batch = collate(tokenizer, part, max_options)
        rows = len(part)
        batch["position_ids"] = mrope_positions(batch["input_ids"], batch["attention_mask"],
                                                np.tile(grid, (rows, 1)), model.image_token_id)
        emb = mx.concatenate([feats] * rows, axis=0) if rows > 1 else feats
        p = _probs(model.forward(batch, image_embeds=emb), batch, temperature)
        for r, n in enumerate(nopt):
            out.append(Prediction(tuple(p[r, :n].tolist())))
    record_confidence(model, plan, out)
    return out, prompt_tokens


# ----------------------------------------------------------------------------- loading
def load_for_serving(ref, *, revision=None, max_length=None, truncate=None, fixed_exit=None,
                     adaptive=None, vision: bool | None = None, dtype: str | None = None):
    """serve.server.load_for_serving for an MLX checkpoint -> serve.server.Served.

    `dtype`: the compute dtype of the tower, "bf16" (default, what the weights are stored
    in) or "fp32"; the heads are fp32 either way."""
    from transformers import AutoTokenizer

    from serve.release import (checkpoint_name, own_token_pool, release_version, resolve_ckpt,
                               serving_encoder)
    from serve.server import Served
    from serve.wire import MAX_INPUT_TOKENS
    path = resolve_ckpt(ref, revision=revision)
    if not is_mlx_checkpoint(path):
        raise SystemExit(f"{path} is not an MLX checkpoint (no mlx.json): convert the release "
                         f"first with scripts/convert_mlx.py PKG OUT --bits 8 --group 64")
    meta = json.loads((path / "meta.json").read_text())
    spec = meta["spec"]
    vb = (meta.get("release") or {}).get("vision") or (spec.get("fit_extra") or {}).get("vision")
    dt = {None: mx.bfloat16, "bf16": mx.bfloat16, "fp32": mx.float32}.get(dtype)
    if dt is None:
        raise ValueError(f"dtype must be bf16 or fp32 on the MLX backend, got {dtype!r}")
    model = MLXDecisionModel(path, vision=bool(vb) and vision is not False, dtype=dt)
    tok = AutoTokenizer.from_pretrained(str(path))
    cap, policy = serving_encoder(spec, max_length, truncate)
    enc = EncodeConfig(layout=spec["layout"], option_pool=spec["option_pool"],
                       option_order="canonical", max_length=cap, truncate=policy,
                       option_pool_own_tokens=own_token_pool(spec, meta))
    meta["calibration"] = model.cal_mode
    meta["weights_source"] = str(path)
    ada, why = attach_policy(model, meta, path, fixed_exit, adaptive)
    served_ada = None if ada is None else {"exits": ada.exits, "tau": ada.tau, "mode": model.adaptive_mode}
    meta["serving"] = {"max_length": cap, "truncate": policy, "exit_layer": model.cfg.exit_layer,
                       "adaptive": served_ada, "option_pool_own_tokens": enc.option_pool_own_tokens,
                       "backend": "mlx", "mlx": model.mlx_record.get("tower")}
    if why:
        meta["serving"]["adaptive_off"] = why
    explicit = (max_length is not None or truncate is not None
                or os.environ.get("RSIJEV_MAX_LENGTH", "").strip()
                or os.environ.get("RSIJEV_TRUNCATE", "").strip())
    if not explicit:
        enc = dataclasses.replace(enc, max_length=MAX_INPUT_TOKENS, truncate="none")
        meta["serving"].update(max_length=MAX_INPUT_TOKENS, truncate="none")
    bits = (model.mlx_record.get("tower") or {}).get("bits", 16)
    name = checkpoint_name(ref, path)
    device = f"mlx-{'gpu' if mx.default_device() == mx.gpu else 'cpu'}"
    dname = ("bf16" if dt == mx.bfloat16 else "fp32") if bits == 16 else f"{bits}-bit"
    s = Served(model, tok, enc, meta, device, dname, path, name, release_version(name), [])
    if model.visual is not None:
        from .vision import ImagePrep
        budget = int((vb or {}).get("budget", 1024))
        meta["vision"] = {"image_token_budget": budget, "min_tokens_per_image": 64, "revision": None}
        try:
            import PIL  # noqa: F401
        except ImportError as e:
            s.vision_error = f"image requests need Pillow ({e}); pip install pillow"
        else:
            s.prep = ImagePrep(path, budget=budget)
        s.venc = dataclasses.replace(enc, max_length=enc.max_length + budget, option_order="canonical")
    return s


# ----------------------------------------------------------------------------- the worker
class MLXRunner:
    """serve.batcher's runner protocol (plan / run / pools) for the MLX model. One request
    at a time; no pooling across requests."""

    pools = False

    def __init__(self, served, *, batch_size: int = 16, default_effort: str | None = None):
        import threading
        from serve.effort import canonical, check_supported
        self.s = served
        self.batch_size = batch_size
        self.default_effort = canonical(default_effort)
        check_supported(served.model, self.default_effort)
        self._tok_lock = threading.Lock()

    def plan(self, state, questions, images=None, effort=None, threshold=None) -> dict:
        from serve.effort import canonical, resolve
        from serve.effort import threshold as check_threshold
        from serve.images import state_too_long, text_only_error
        from serve.wire import RequestError
        s = self.s
        try:
            effort = canonical(effort) if effort is not None else self.default_effort
            effort, threshold = resolve(s.model, effort, check_threshold(threshold), bool(images))
        except ValueError as e:
            raise RequestError(str(e)) from None
        with self._tok_lock:
            if images:
                if s.prep is None:
                    raise text_only_error(s.name, s.vision_error)
                try:
                    p = plan_image_request(s.tok, s.prep, state, images, questions, s.venc)
                except ValueError as e:
                    err = state_too_long(e, s.venc.max_length, s.venc)
                    if err is None:
                        raise
                    raise err from None
            else:
                p = plan_request(s.tok, state, questions, s.enc)
        if effort is not None:
            p["effort"] = effort
        if threshold is not None:
            p["threshold"] = threshold
        p["max_options"] = max(s.meta["spec"]["max_options"], max(len(q.options) for q in questions))
        return p

    def one(self, plan):
        s = self.s
        run = score_image_planned if plan["path"] == "image" else score_planned
        preds, tokens = run(s.model, s.tok, plan, max_options=plan["max_options"],
                            batch_size=self.batch_size)
        return [list(p.probs) for p in preds], tokens

    def run(self, plans: list) -> list:
        from serve.batcher import _capture
        return [_capture(self.one, p) for p in plans]


def model_worker(served, *, batch_size: int = 16, default_effort: str | None = None):
    from serve.batcher import GpuWorker
    return GpuWorker(MLXRunner(served, batch_size=batch_size, default_effort=default_effort))


def make_scorer(served, batch_size: int = 16):
    """serve.server.make_scorer for the MLX model (the Decider's scorer)."""
    runner = MLXRunner(served, batch_size=batch_size)

    def scorer(state: str, questions, images=None, effort=None, threshold=None):
        plan = runner.plan(state, questions, images, effort, threshold)
        out = runner.one(plan)
        extras = {k: plan[k] for k in ("truncated", "depth", "effort", "confidence")
                  if plan.get(k) is not None}
        return (*out, extras) if extras else out

    return scorer
