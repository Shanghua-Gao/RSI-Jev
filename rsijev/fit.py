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


def _param_groups(model, cfg: FitConfig):
    head, mix, base = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("mix_logits"):
            mix.append(p)
        elif n.startswith("scorer"):
            head.append(p)
        else:
            base.append(p)
    groups = [{"params": head, "lr": cfg.lr_head, "name": "head",
               "weight_decay": cfg.head_weight_decay}]
    if mix:
        groups.append({"params": mix, "lr": cfg.lr_mix, "name": "mix"})
    if base:
        groups.append({"params": base, "lr": cfg.lr_base, "name": "base"})
    return groups


def fit(model, tokenizer, cases: Sequence[Case], enc: EncodeConfig, cfg: FitConfig, *,
        max_options: int, seed: int, device: str = "cuda") -> dict:
    """Train in place. Returns the averaged scorer state plus a small history."""
    g = seed_everything(seed)
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
    lambdas = [base_fn if gr["name"] == "base" else head_fn if gr["name"] == "head" else warm
               for gr in opt.param_groups]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)

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

    model.train()
    history, ckpts = [], []
    cursor = 0
    for step in range(cfg.steps):
        idx = [order[(cursor + i) % len(order)] for i in range(cfg.batch_size)]
        cursor += cfg.batch_size
        chunk = [pairs[i] for i in idx]
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
        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cfg.autocast_bf16):
            if cfg.prior_kl:
                logits, base = model.forward_with_base(**batch)   # ONE tower pass
            else:
                logits, base = model(**batch), None
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

    return {"history": history, "averaged_over": len(ckpts),
            "examples_seen": cfg.steps * cfg.batch_size, "seed": seed,
            "calibration": cal_report}
