"""The training loop. PROTECTED in its bookkeeping, EDITABLE through its config.

Arms in one comparison share seed, initialisation, data order and optimiser step
count, and differ in exactly one thing (common random numbers). The step count is
fixed rather than the epoch count so that a data-axis arm that changes corpus
size does not silently also change the amount of optimisation.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .contract import Case
from .encode import (EncodeConfig, collate, encode_question, gold_tensor,
                     iter_questions, unpermute_logits)
from .train import Objective, RLConfig, objective_loss, prior_kl, seed_everything


@dataclass
class FitConfig:
    objective: Objective = "soft_ce"
    steps: int = 400                 # FIXED across arms, not epochs
    batch_size: int = 16
    lr_head: float = 1e-3
    # The layer-mixture logits sit behind a saturated softmax by design (a one-
    # hot start is what puts the arm at its fixed-tap baseline), so they need a
    # far larger step than the head or they never leave it. There are only
    # len(MODES) x len(candidates) of them.
    lr_mix: float = 1e-1
    lr_base: float = 5e-6
    # Schedule for the TOWER group only; the head and mix groups keep the
    # constant-after-warmup schedule every frozen-tower arm was run under.
    # A constant 2e-5 on the tower diverged mid-run (loss 1 -> 4588 -> nan).
    base_schedule: str = "cosine"    # "cosine" (to 0) | "constant"
    # The head's schedule. "constant" is what every arm before 2026-09-23 used.
    # With a tuned tower at 2B, logit spikes arrived after step 1200, when the
    # tower lr had decayed and only the head's constant lr was still large.
    head_schedule: str = "constant"  # "constant" | "cosine" (to 0)
    # Two remedies for a head whose logit scale grows without bound once the
    # tower trains. Exactly one-hot gold (12% of noul, 8% of choice in synth)
    # puts the soft-CE optimum at an infinite margin.
    #   label_smoothing: train on (1-e)*gold + e*uniform over the question's
    #     options, so the optimum is finite. Training targets only.
    #   head_weight_decay: AdamW decay on the head group alone.
    label_smoothing: float = 0.0
    head_weight_decay: float = 0.0
    warmup: float = 0.05
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    keep_last_k: int = 3             # checkpoints to average; free variance reduction
    prior_kl: float = 0.0            # weight on KL(base || model); 0 disables
    # bf16 autocast for the forward pass while the weights stay fp32. Needed when
    # the tower trains: fp32 activations for 24 layers at batch 16 used 79 GB.
    autocast_bf16: bool = False
    # Post-hoc calibration on DEV data (rsijev/calibrate.py). "none" reproduces
    # the champion exactly (no dev split, nothing withheld from training).
    #   temp | temp_mode | oof_head
    cal_method: str = "none"
    cal_td_frac: float = 0.20        # typed-decisions TRAIN cases withheld as dev
    cal_synth_frac: float = 0.03
    cal_mc_frac: float = 0.05
    cal_joint_lambda: float = 0.0     # oof_head_joint: weight on score-row soft-gold Brier
    rl: RLConfig = field(default_factory=RLConfig)
    log_every: int = 50
    # Length-bucketed batches: bucket width in tokens; 0 = off (earlier releases).
    length_bucket: int = 0
    # ---- mmlu-keep-1: knowledge retention while the tower trains ----------
    # (a) Retention KL(base || student) on the tower's NEXT-TOKEN distribution
    # (tied lm_head = the frozen embedding) over general-text rows drawn from the
    # training split's MC-replay cases. The base's targets are precomputed ONCE
    # from the tower at step 0 -- which is bit-for-bit the frozen base copy --
    # as the top-k log-probs per position plus the tail mass, so no second 2B
    # model is held in memory. KL is exact on the (top-k, rest) partition of the
    # base's support. 0 disables (champion behaviour).
    retention_kl: float = 0.0
    retention_source_prefixes: str = "dc_"   # Case.source prefixes (mc_replay = dc_*)
    retention_rows: int = 4                  # retention rows per step
    retention_pool: int = 2048               # rows whose base targets are cached
    retention_topk: int = 64
    retention_max_tokens: int = 384
    # v5.0-VL (early exit trained from the base, layers above the exit dropped): the
    # retention term runs the EXIT model -- the first exit_layer decoder layers, the
    # final norm and the tied lm_head -- for both the base targets (the base cut at the
    # same layer) and the student. Nothing then runs layers >= exit_layer, so they are
    # frozen and a step costs the truncated tower only. Needs arch exit_layer with
    # exit_norm. Off = full-depth retention (targets and student at full depth).
    retention_at_exit: bool = False
    # (b) Per-source TOWER gradient weight: {source_prefix: scale}. The scale
    # multiplies that row's gradient into the tower only (arch._RowGradScale);
    # the scorer sees the full loss. {} = every row at 1.0 (champion behaviour).
    source_tower_scale: dict = field(default_factory=dict)

    # v2.1: decoder layers 0..lower_layers_n-1 train at lr_base * lower_layers_lr_scale,
    # in their own param group on the same schedule. n=0 / scale 1.0 is v1.0 and v2.0.
    lower_layers_n: int = 0
    lower_layers_lr_scale: float = 1.0
    # ---- continued training from a saved checkpoint -------------------------
    # init_from: a release checkpoint directory written by run_arm_lib.save_release
    # (tower.safetensors, scorer.safetensors, meta.json). Its tower and scorer are
    # loaded into THIS arm's own fp32 tower copy and scorer before the optimiser
    # is built, so the arm continues training from that model. Requires
    # freeze_base=false (a frozen arm's tower is the worker's resident model).
    # The parent's path and content sha go into history[0], i.e. into the arm
    # record's `history`. "" = start from the base model (champion behaviour).
    # init_sha256: optional; if set, the parent's combined sha (ckpt_sha) must
    # match, or the arm refuses to train.
    # steps=0 with init_from trains nothing: the arm scores the parent as saved
    # (the load check: its suite must equal the parent's own record).
    init_from: str = ""
    init_sha256: str = ""
    # Image states (rsijev/vision_fit.py, v4.0-VL on): {"root" | "roots", "budget",
    # "model", ...}. {} = text only, exactly as before.
    vision: dict = field(default_factory=dict)
    # RL stage (rsijev/rl2.py): {} = plain fit (earlier releases'
    # behaviour); otherwise fit_rl2 runs with this config after init_from.
    rl2: dict = field(default_factory=dict)



def _lr_lambdas(groups, cfg: FitConfig):
    """One LR multiplier per parameter group. Every tower group ("base" and v2.1's
    "base_lower") follows `base_schedule`, the head follows `head_schedule`, and the rest
    only warm up. The lower layers share the tower's schedule, as in the code that trained
    v2.1 onwards; an earlier public copy matched only "base" and held them constant."""
    n_warm = max(1, int(cfg.warmup * cfg.steps))

    def warm(s):
        return min(1.0, (s + 1) / n_warm)

    def warm_cosine(s):
        if s < n_warm:
            return warm(s)
        t = (s - n_warm) / max(1, cfg.steps - n_warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    base_fn = warm_cosine if cfg.base_schedule == "cosine" else warm
    head_fn = warm_cosine if cfg.head_schedule == "cosine" else warm
    return [base_fn if gr["name"].startswith("base") else head_fn if gr["name"] == "head" else warm
            for gr in groups]

def _param_groups(model, cfg: FitConfig):
    """One group per role, plus an optional slower group for the lowest layers.

    v2.1's only optimiser change: decoder layers `0 .. lower_layers_n - 1` get their
    own group at `lr_base * lower_layers_lr_scale`, on the same schedule as the rest of
    the tower. It matters because fitting a decision objective through every layer at one
    rate overwrites the representation the model's general knowledge sits in -- and
    freezing those layers instead costs decision accuracy, because they do have to adapt.
    """
    import re
    head, mix, base, lower = [], [], [], []
    pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("mix_logits"):
            mix.append(p)
        elif n.startswith("scorer"):
            head.append(p)
        else:
            m = pat.search(n) if cfg.lower_layers_n else None
            if m and "visual" not in n and int(m.group(1)) < cfg.lower_layers_n:
                lower.append(p)
            else:
                base.append(p)
    groups = [{"params": head, "lr": cfg.lr_head, "name": "head",
               "weight_decay": cfg.head_weight_decay}]
    if mix:
        groups.append({"params": mix, "lr": cfg.lr_mix, "name": "mix"})
    if base:
        groups.append({"params": base, "lr": cfg.lr_base, "name": "base"})
    if lower:
        groups.append({"params": lower, "lr": cfg.lr_base * cfg.lower_layers_lr_scale,
                       "name": "base_lower"})
        print(f"    lower_layers_n={cfg.lower_layers_n}: "
              f"{sum(p.numel() for p in lower):,} params at lr x{cfg.lower_layers_lr_scale}",
              flush=True)
    return groups


def _text_tower_embed(model):
    emb = model.tower.get_input_embeddings()
    if emb is None:
        raise ValueError("retention_kl needs the tower's input embedding (tied lm_head)")
    return emb.weight


def _retention_rows(tokenizer, cases, cfg: FitConfig, seed: int):
    """Token-id lists for the retention pool: state + question + options, in
    canonical order (the same rendering the model trains on), first question
    of each case, truncated to retention_max_tokens."""
    prefixes = tuple(p for p in cfg.retention_source_prefixes.split(",") if p)
    pool = [c for c in cases if c.source.startswith(prefixes)]
    if not pool:
        raise ValueError(f"retention_kl: no training cases with source prefix {prefixes}")
    rng = random.Random(seed + 2718281)
    rng.shuffle(pool)
    pool = pool[: cfg.retention_pool]
    enc = EncodeConfig()
    rows = []
    for c in pool:
        ids = encode_question(tokenizer, c.state, c.questions[0], enc)["input_ids"]
        rows.append(ids[: cfg.retention_max_tokens])
    return rows


def _pad(tokenizer, rows, device):
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    am = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
        am[i, :len(r)] = 1
    return ids.to(device), am.to(device)


def _retention_ctx(model, cfg: FitConfig):
    """The context retention's tower calls run in: the exit model under
    retention_at_exit, else the full tower."""
    import contextlib
    if not cfg.retention_at_exit:
        return contextlib.nullcontext()
    acfg = model.cfg
    if not getattr(acfg, "exit_layer", None) or not getattr(acfg, "exit_norm", True):
        raise ValueError("retention_at_exit needs arch exit_layer with exit_norm=True")
    return model.exit_tower()


def _freeze_above_exit(model) -> int:
    """requires_grad=False for the decoder layers >= exit_layer, which no pass runs
    under retention_at_exit. Returns the number of parameters frozen."""
    import re
    L = int(model.cfg.exit_layer)
    pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    n = 0
    for name, p in model.tower.named_parameters():
        m = pat.search(name)
        if m and "visual" not in name and int(m.group(1)) >= L and p.requires_grad:
            p.requires_grad_(False)
            n += p.numel()
    return n


@torch.no_grad()
def _retention_targets(model, tokenizer, rows, cfg: FitConfig, device, dev_type):
    """Base top-k next-token log-probs per position, from the UNTRAINED tower."""
    W = _text_tower_embed(model)
    out = []
    was = model.training
    model.eval()
    for i in range(0, len(rows), 8):
        chunk = rows[i:i + 8]
        ids, am = _pad(tokenizer, chunk, device)
        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cfg.autocast_bf16), \
                _retention_ctx(model, cfg):
            final = model.tower(input_ids=ids, attention_mask=am).last_hidden_state
        for r, row in enumerate(chunk):
            n = len(row) - 1                      # positions that predict a next token
            with torch.autocast(dev_type, enabled=False):
                lp = F.log_softmax(final[r, :n].float() @ W.float().T, dim=-1)
            v, ix = lp.topk(cfg.retention_topk, dim=-1)
            out.append((v.cpu(), ix.to(torch.int32).cpu()))
    model.train(was)
    return out


def _row_retention_kl(final_row, W, q_lp, q_ix):
    """KL(base || student) on the partition {top-k of base} + {rest}, mean over
    positions. final_row (n, H) fp32; q_lp, q_ix (n, k)."""
    lp = F.log_softmax(final_row @ W.T, dim=-1)                 # (n, V)
    p_lp = lp.gather(1, q_ix)                                   # (n, k)
    q = q_lp.exp()
    q_rest = (1 - q.sum(-1)).clamp_min(1e-8)
    p_rest = (1 - p_lp.exp().sum(-1)).clamp_min(1e-8)
    kl = (q * (q_lp - p_lp)).sum(-1) + q_rest * (q_rest.log() - p_rest.log())
    return kl.mean()


def _approx_len(pair) -> int:
    """Characters the encoder will render: state, instructions, options and
    their descriptions. Tokenising 100k questions to sort them would cost more
    than it saves; the character count orders them almost the same way."""
    c, q = pair
    return (len(c.state) + len(q.instructions)
            + sum(len(o) + len(q.criteria.get(o, "")) for o in q.options))


def _bucketed_stream(pairs, order, n, batch_size, k, rng) -> list[int]:
    """The first n indices of the cyclic data order, regrouped by length within
    windows of k batches. Every window holds exactly the examples the unbucketed
    stream would have visited over the same steps."""
    flat = [order[i % len(order)] for i in range(n)]
    out: list[int] = []
    w = batch_size * k
    for s in range(0, n, w):
        win = sorted(flat[s:s + w], key=lambda i: _approx_len(pairs[i]))
        batches = [win[j:j + batch_size] for j in range(0, len(win), batch_size)]
        rng.shuffle(batches)
        for b in batches:
            out += b
    return out


CKPT_FILES = ("meta.json", "scorer.safetensors", "tower.safetensors")


def ckpt_sha(ckpt) -> dict:
    """sha256 of each release file and a combined digest over the three
    (sha256 of "name sha" lines, fixed order). The tower is ~5.5 GB: ~20 s."""
    import hashlib
    from pathlib import Path
    ck, out = Path(ckpt), {}
    for name in CKPT_FILES:
        h = hashlib.sha256()
        with open(ck / name, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 24), b""):
                h.update(blk)
        out[name] = h.hexdigest()
    out["combined"] = hashlib.sha256(
        "\n".join(f"{n} {out[n]}" for n in CKPT_FILES).encode()).hexdigest()
    return out


def load_init(model, cfg: FitConfig) -> dict:
    """Load a release checkpoint (tower minus the frozen embedding, scorer) into
    the arm's model in place. Returns the lineage fields for the record."""
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    ck = Path(cfg.init_from)
    missing_f = [n for n in CKPT_FILES if not (ck / n).is_file()]
    if missing_f:
        raise FileNotFoundError(f"init_from {ck}: missing {missing_f}")
    if not any(p.requires_grad for p in model.tower.parameters()):
        raise RuntimeError("init_from needs freeze_base=false: with a frozen tower the arm "
                           "reads the worker's resident model, which must never be overwritten")
    if getattr(model, "mix_logits", None) is not None:
        raise RuntimeError("init_from: a release checkpoint does not store mix_logits; "
                           "a layer-mix arm cannot be continued from it")
    sha = ckpt_sha(ck)
    if cfg.init_sha256 and sha["combined"] != cfg.init_sha256:
        raise RuntimeError(f"init_from {ck}: sha {sha['combined']} != expected {cfg.init_sha256}")
    meta = json.loads((ck / "meta.json").read_text())
    tower_sd = load_file(str(ck / "tower.safetensors"))
    missing, unexpected = model.tower.load_state_dict(tower_sd, strict=False)
    del tower_sd
    bad = [k for k in missing if "embed_tokens" not in k]
    if bad or unexpected:
        raise RuntimeError(f"init_from {ck}: tower mismatch, missing {bad[:5]}, "
                           f"unexpected {list(unexpected)[:5]}")
    model.scorer.load_state_dict(load_file(str(ck / "scorer.safetensors")), strict=True)
    pspec = meta.get("spec", {})
    info = {"init_from": str(ck), "init_sha256": sha["combined"],
            "init_parent_code_dir": pspec.get("code_dir"),
            "init_parent_corpus_dir": pspec.get("corpus_dir"),
            "init_parent_steps": pspec.get("steps"),
            "init_parent_fit_extra_init_from": (pspec.get("fit_extra") or {}).get("init_from", "")}
    print(f"    init_from {ck}: loaded tower + scorer, sha {sha['combined'][:16]}", flush=True)
    return info


@torch.no_grad()
def _probe_loss(model, tokenizer, pairs, idx, enc, cfg: FitConfig, max_options, device) -> float:
    """Loss of the current model on one batch, no update (steps=0 arms)."""
    was = model.training
    model.eval()
    chunk = [pairs[i] for i in idx]
    batch = collate(tokenizer, [encode_question(tokenizer, c.state, q, enc) for c, q in chunk],
                    max_options=max_options, device=device)
    gold = gold_tensor([c for c, _ in chunk], [q.key for _, q in chunk], max_options, device=device)
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cfg.autocast_bf16):
        logits = model(**batch)
    logits = unpermute_logits(logits.float(), batch["option_perm"], batch["option_mask"])
    logits = logits.masked_fill(~batch["option_mask"], float("-inf"))
    loss = objective_loss(cfg.objective, logits, gold, progress=0.0, rl=cfg.rl, generator=None)
    model.train(was)
    return float(loss)


def fit(model, tokenizer, cases: Sequence[Case], enc: EncodeConfig, cfg: FitConfig, *,
        max_options: int, seed: int, device: str = "cuda", _helpers: dict | None = None) -> dict:
    """Train in place. Returns the averaged scorer state plus a small history."""
    g = seed_everything(seed)
    dev_type0 = "cuda" if str(device).startswith("cuda") else "cpu"
    if cfg.retention_at_exit:
        _retention_ctx(model, cfg)          # validates exit_layer / exit_norm
        n_frz = _freeze_above_exit(model)
        print(f"    retention_at_exit: layers >= {model.cfg.exit_layer} dropped from every pass; "
              f"{n_frz:,} params frozen", flush=True)
    ret_rows = ret_tgt = None
    if cfg.retention_kl and cfg.init_from:
        # BEFORE init_from: the retention targets are the BASE model's next-token
        # distribution (cut at the exit under retention_at_exit). After the load they
        # would be the parent's, and a decision-tuned parent's LM is destroyed. No RNG
        # is consumed here, so the data order is unchanged.
        ret_rows = _retention_rows(tokenizer, cases, cfg, seed)
        ret_tgt = _retention_targets(model, tokenizer, ret_rows, cfg, device, dev_type0)
        print(f"    retention_kl={cfg.retention_kl}: {len(ret_rows)} rows, base targets cached "
              f"before init_from{' (base cut at exit ' + str(model.cfg.exit_layer) + ')' if cfg.retention_at_exit else ''}",
              flush=True)
    # Continue from a saved release checkpoint. Loaded after seeding and
    # before the optimiser, so the data order and option shuffling are the same
    # as a from-base arm with this seed.
    init = load_init(model, cfg) if cfg.init_from else None
    if cfg.rl2:
        from .rl2 import fit_rl2
        out = fit_rl2(model, tokenizer, cases, enc, cfg, max_options=max_options, seed=seed,
                      device=device, helpers={"param_groups": _param_groups,
                                              "bucketed_stream": _bucketed_stream,
                                              "seed_everything": seed_everything,
                                              **(_helpers or {})})
        if init is not None:
            first = next((h for h in out["history"] if h.get("step", 0) >= 0), out["history"][0])
            first.update(init)
        out["init"] = init
        return out
    # Its own stream, so turning option shuffling on does not change the data
    # ORDER an arm sees and break the pairing with its control.
    order_rng = random.Random(seed + 104729)
    cal_report = None
    dev = []
    if cfg.cal_method != "none":
        from .calibrate import split_dev
        cases, dev = split_dev(cases, td_frac=cfg.cal_td_frac, synth_frac=cfg.cal_synth_frac,
                               mc_frac=cfg.cal_mc_frac)
        print(f"    calibration: {len(dev)} dev cases withheld from training "
              f"({sum(len(c.questions) for c, _ in dev)} questions); {len(cases)} train cases",
              flush=True)
    if any(c.source.startswith("caldev:") for c in cases):
        raise ValueError("caldev:* cases are calibration-only and must never be trained on")
    pairs = list(iter_questions(cases))
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)          # data order is part of the CRN

    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambdas(opt.param_groups, cfg))

    # Scale probes: where does logit growth come from -- the features the head
    # reads, or the head's own weights? A pre-hook records the mean L2 norm of
    # the head's inputs on the step being logged.
    feat = {}
    def _probe(_m, _args, kwargs):
        if feat.get("on"):
            with torch.no_grad():
                dn = kwargs["decision_h"].detach().float().norm(dim=-1)
                feat["dec"], feat["dec_max"] = float(dn.mean()), float(dn.max())
                oh = kwargs.get("option_h")
                if oh is not None:
                    on = oh.detach().float().norm(dim=-1)
                    feat["opt"], feat["opt_max"] = float(on.mean()), float(on.max())
    hook = model.scorer.register_forward_pre_hook(_probe, with_kwargs=True)
    final_norm = next((p for n, p in model.named_parameters()
                       if n.endswith("tower.norm.weight") or n == "tower.norm.weight"), None)

    ret_order_rng = random.Random(seed + 31415)
    if cfg.retention_kl and ret_rows is None:
        ret_rows = _retention_rows(tokenizer, cases, cfg, seed)
        ret_tgt = _retention_targets(model, tokenizer, ret_rows, cfg, device, dev_type0)
        print(f"    retention_kl={cfg.retention_kl}: {len(ret_rows)} rows, "
              f"{sum(len(r) - 1 for r in ret_rows)} positions, top-{cfg.retention_topk} "
              f"base targets cached", flush=True)
    ret_perm, ret_cur = [], 0
    scale_map = dict(cfg.source_tower_scale or {})

    vis = None
    if cfg.vision:
        from .vision_fit import VisionFeed
        vis = VisionFeed(cfg.vision, tokenizer, model, device)
    model.train()
    history, ckpts = [], []
    stream = None
    if cfg.length_bucket:
        stream = _bucketed_stream(pairs, order, cfg.steps * cfg.batch_size, cfg.batch_size,
                                  cfg.length_bucket, random.Random(seed + 7919))
    cursor = 0
    for step in range(cfg.steps):
        if stream is not None:
            idx = stream[cursor:cursor + cfg.batch_size]
        else:
            idx = [order[(cursor + i) % len(order)] for i in range(cfg.batch_size)]
        cursor += cfg.batch_size
        chunk = [pairs[i] for i in idx]
        if vis is not None:
            batch = vis.batch(chunk, enc, order_rng, max_options)
        else:
            batch = collate(tokenizer,
                            [encode_question(tokenizer, c.state, q, enc, rng=order_rng)
                             for c, q in chunk],
                            max_options=max_options, device=device)
        gold = gold_tensor([c for c, _ in chunk], [q.key for _, q in chunk],
                           max_options, device=device)
        if cfg.label_smoothing:
            # Uniform over the question's own options. The option COUNT does not
            # depend on presentation order, and gold fills the first n slots.
            n = batch["option_mask"].sum(1, keepdim=True).float()
            slots = torch.arange(gold.shape[1], device=gold.device).unsqueeze(0)
            uni = (slots < n).float() / n
            gold = (1 - cfg.label_smoothing) * gold + cfg.label_smoothing * uni

        feat["on"] = step % cfg.log_every == 0 or step == cfg.steps - 1
        dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
        extra = {}
        if scale_map:
            sc = [next((v for k, v in scale_map.items() if c.source.startswith(k)), 1.0)
                  for c, _ in chunk]
            extra["row_grad_scale"] = torch.tensor(sc, dtype=torch.float32, device=device)
        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cfg.autocast_bf16):
            if cfg.prior_kl:
                logits, base = model.forward_with_base(**batch, **extra)   # ONE tower pass
            else:
                logits, base = model(**batch, **extra), None
        logits = logits.float()
        # Back to canonical option order BEFORE the loss: gold indexes
        # q.options, so a permuted presentation compared unmapped would train
        # against the wrong option.
        logits = unpermute_logits(logits, batch["option_perm"], batch["option_mask"])
        # masked options must not receive gradient through the loss
        logits = logits.masked_fill(~batch["option_mask"], float("-inf"))
        loss = objective_loss(cfg.objective, logits.float(), gold,
                              progress=step / max(1, cfg.steps - 1), rl=cfg.rl,
                              generator=None)
        if cfg.prior_kl:
            loss = loss + cfg.prior_kl * prior_kl(logits.float(), base)
        ret_kl = None
        if cfg.retention_kl:
            pick = []
            for _ in range(cfg.retention_rows):
                if ret_cur >= len(ret_perm):
                    ret_perm = list(range(len(ret_rows)))
                    ret_order_rng.shuffle(ret_perm)
                    ret_cur = 0
                pick.append(ret_perm[ret_cur]); ret_cur += 1
            r_ids, r_am = _pad(tokenizer, [ret_rows[i] for i in pick], device)
            with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cfg.autocast_bf16), \
                    _retention_ctx(model, cfg):
                r_final = model.tower(input_ids=r_ids, attention_mask=r_am).last_hidden_state
            W = _text_tower_embed(model).detach().float()
            kls = []
            with torch.autocast(dev_type, enabled=False):
                for j, i in enumerate(pick):
                    q_lp, q_ix = ret_tgt[i]
                    n = q_lp.shape[0]
                    # checkpointed: the (n, V) student log-probs are recomputed in
                    # backward instead of being held for every row at once.
                    kls.append(torch.utils.checkpoint.checkpoint(
                        _row_retention_kl, r_final[j, :n].float(), W,
                        q_lp.to(device, non_blocking=True).float(),
                        q_ix.to(device, non_blocking=True).long(),
                        use_reentrant=False))
            ret_kl = torch.stack(kls).mean()
            loss = loss + cfg.retention_kl * ret_kl
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss {loss.item()} at step {step}/{cfg.steps}; "
                f"last logged {history[-3:]}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        logging = step % cfg.log_every == 0 or step == cfg.steps - 1
        if logging:
            # Pre-clip gradient norm per group and the largest finite logit: the
            # two training-time signals that say WHERE an instability starts,
            # without looking at any evaluation target.
            gnorm = {gr["name"]: float(torch.linalg.vector_norm(torch.stack(
                         [p.grad.detach().float().norm() for p in gr["params"]
                          if p.grad is not None] or [torch.zeros((), device=loss.device)])))
                     for gr in opt.param_groups}
            fin = logits.detach()[torch.isfinite(logits.detach())]
            logit_absmax = float(fin.abs().max()) if fin.numel() else float("nan")
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for gr in opt.param_groups for p in gr["params"]], cfg.grad_clip)
        opt.step()
        sched.step()

        if logging:
            history.append({"step": step, "loss": float(loss.detach()),
                            **({"image_rows": vis.n_image_rows} if vis is not None else {}),
                            **({"ret_kl": round(float(ret_kl.detach()), 5)}
                               if ret_kl is not None else {}),
                            "logit_absmax": round(logit_absmax, 2),
                            "feat_dec": round(feat.get("dec", float("nan")), 2),
                            "feat_opt": round(feat.get("opt", float("nan")), 2),
                            "feat_dec_max": round(feat.get("dec_max", float("nan")), 2),
                            "feat_opt_max": round(feat.get("opt_max", float("nan")), 2),
                            "w_head": round(float(torch.sqrt(sum(
                                (p.detach().float() ** 2).sum()
                                for p in model.scorer.parameters()))), 3),
                            **({"w_final_norm": round(float(final_norm.detach().float().abs().mean()), 4)}
                               if final_norm is not None else {}),
                            **{f"gn_{k}": round(v, 4) for k, v in gnorm.items()}})
        if step >= cfg.steps - cfg.keep_last_k:
            ckpts.append({k: v.detach().cpu().clone()
                          for k, v in model.scorer.state_dict().items()})

    hook.remove()
    if cfg.steps == 0:
        # Nothing trained: the scorer stays exactly as loaded (no averaging).
        if not cfg.init_from:
            raise ValueError("steps=0 is only meaningful with init_from (a load check)")
        history.append({"step": 0, "loss": _probe_loss(
            model, tokenizer, pairs, order[:cfg.batch_size], enc, cfg, max_options, device),
            "eval_only": True})
    else:
        averaged = {k: torch.stack([c[k].float() for c in ckpts]).mean(0) for k in ckpts[0]}
        model.scorer.load_state_dict({k: v.to(next(model.scorer.parameters()).dtype)
                                      for k, v in averaged.items()})
    if cfg.cal_method != "none":
        from .calibrate import calibrate
        cal_report = calibrate(model, tokenizer, dev, enc, method=cfg.cal_method,
                               max_options=max_options, device=device,
                               joint_lambda=cfg.cal_joint_lambda)
        print("    CALIBRATION " + " ".join(f"{k}={v}" for k, v in cal_report.items()), flush=True)
        # Into the record through history: step -1 keeps it out of every stability
        # rule (they read step > 150 / >= 750) and history[-1] stays the final loss.
        # "loss" here is the calibrated dev pool's weighted top-label BCE.
        history.insert(0, {"step": -1, "loss": cal_report["cal_dev_wbce_cal"], **cal_report})

    if init is not None and history:
        history[0].update(init)
    return {"history": history, "averaged_over": len(ckpts), "init": init,
            "examples_seen": cfg.steps * cfg.batch_size, "seed": seed,
            "calibration": cal_report}
