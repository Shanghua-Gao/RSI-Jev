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
from .encode import (EncodeConfig, collate, criterion_text, encode_question, gold_tensor,
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

    # ---- v6.0-VL (multi-exit) training. Every key below is off by default, which is
    # the loop above exactly; the v6.0-VL stage specs (data/v6.0-vl_recipe/) set them.
    # Gradient accumulation: each step's batch_size rows (same rows, same order, same
    # option shuffles) run as grad_accum micro-batches; each micro-batch's task loss is
    # weighted by its row share, and the step-level term (retention) is added once, on
    # the last micro-batch. Optimiser step, schedule and clipping unchanged. 1 = one pass.
    grad_accum: int = 1
    # Split a step into grad_accum micro-batches only when its longest encoded row is
    # longer than this many tokens; shorter steps run as one pass. 0 = every step.
    grad_accum_min_tokens: int = 0
    # Deep supervision of the early exits (arch aux_exits): loss = main-exit loss +
    # sum_L aux_exit_weights[L] x (the same objective at exit L). Keys are exit indices
    # as strings. {} with aux heads present = weight 0 (heads not trained).
    aux_exit_weights: dict = field(default_factory=dict)
    # lr of the aux heads (their own param group, on the head schedule). 0 = lr_head.
    lr_aux: float = 0.0
    # After init_from: copy the loaded main scorer into every aux head.
    aux_init_from_main: bool = True
    # Load the aux heads from this checkpoint directory (or aux_scorers.safetensors
    # file) instead; wins over aux_init_from_main. "" = off.
    aux_init_from: str = ""
    # Where fit() writes the averaged aux heads (aux_scorers.safetensors, keys
    # "<exit>.<param>"). run_arm_lib saves the main scorer only, so a multi-exit spec
    # points this at its save_dir. "" = not saved.
    aux_save_dir: str = ""
    # > 0: every aux exit with no entry in aux_exit_weights gets weight
    # aux_weight_total / len(aux_exits). 0 = aux_exit_weights only.
    aux_weight_total: float = 0.0
    # Self-distillation of each aux exit to the main exit (teacher detached):
    # w_L x (CE_L + aux_distill x T^2 x KL(p_main || p_L)) at T = aux_distill_temp.
    aux_distill: float = 0.0
    aux_distill_temp: float = 1.0
    # The aux exits read norm(h_L).detach() (DecisionModel.forward_exits aux_detach):
    # their losses train only the aux heads, and the tower gets the main loss alone.
    # The aux group is clipped separately, so the tower + main-head step equals the step
    # without aux exits.
    aux_detach_tower: bool = False
    # Per aux exit, + aux_zloss x mean(logsumexp(aux logits)^2) (z-loss), which keeps the
    # aux heads' logit scale from drifting. Aux heads only. 0 = off.
    aux_zloss: float = 0.0
    # Confidence-ranking term on the MAIN exit: + conf_rank_coef x RankNet over
    # (correct, wrong) row pairs of s = logit-odds of the top-1 probability /
    # conf_rank_margin_temp, per micro-batch. It is dropped on a step whose previous
    # step's retention KL exceeded conf_rank_kl_gate (needs retention_kl > 0; <= 0 = no
    # gate), and is off for the first conf_rank_start fraction of the steps. 0 = off.
    conf_rank_coef: float = 0.0
    conf_rank_margin_temp: float = 1.0
    conf_rank_kl_gate: float = 0.20
    conf_rank_start: float = 0.0
    # Sources (Case.source prefixes, comma list) presented in canonical option order
    # instead of the encoder's option_order. "" = every row uses the encoder's.
    canonical_order_sources: str = ""
    # One stdout line every print_every steps (loss terms, s/step, peak memory). 0 = off.
    print_every: int = 0
    # Non-finite guard. > 0: a step whose loss, or whose pre-clip gradient norm in any
    # param group, is non-finite is skipped (no update; the schedule and the data cursor
    # still advance). With aux_detach_tower, a non-finite norm in the aux group alone
    # drops only the aux gradients. More than nonfinite_skip_max consecutive skips raise
    # FloatingPointError. 0 = raise on the first non-finite loss (as before).
    nonfinite_skip_max: int = 0
    # Training rows are encoded at this max_length (the encoder's truncation applies);
    # evaluation keeps the encoder's own cap. 0 = the encoder's cap.
    train_max_length: int = 0
    # The fused CUDA AdamW kernel (same update rule, less memory). Ignored on CPU.
    adam_fused: bool = False



def _lr_lambdas(groups, cfg: FitConfig):
    """One LR multiplier per parameter group. Every tower group ("base" and v2.1's
    "base_lower") follows `base_schedule`, the heads ("head" and the early-exit heads'
    "aux") follow `head_schedule`, and the rest only warm up. The lower layers share the
    tower's schedule, as in the code that trained v2.1 onwards; an earlier public copy
    matched only "base" and held them constant."""
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
    return [base_fn if gr["name"].startswith("base") else head_fn if gr["name"] in ("head", "aux")
            else warm for gr in groups]

def _param_groups(model, cfg: FitConfig):
    """One group per role, plus an optional slower group for the lowest layers.

    v2.1's only optimiser change: decoder layers `0 .. lower_layers_n - 1` get their
    own group at `lr_base * lower_layers_lr_scale`, on the same schedule as the rest of
    the tower. It matters because fitting a decision objective through every layer at one
    rate overwrites the representation the model's general knowledge sits in -- and
    freezing those layers instead costs decision accuracy, because they do have to adapt.
    """
    import re
    head, mix, base, lower, aux = [], [], [], [], []
    pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("mix_logits"):
            mix.append(p)
        elif n.startswith("aux_scorers"):
            # the early-exit heads (arch aux_exits): their own group, lr_aux or lr_head
            aux.append(p)
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
    if aux:
        groups.append({"params": aux, "lr": cfg.lr_aux or cfg.lr_head, "name": "aux",
                       "weight_decay": cfg.head_weight_decay})
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


def _approx_len(pair, enc: EncodeConfig | None = None) -> int:
    """Characters the encoder will render: state, instructions, options and
    their descriptions. Tokenising 100k questions to sort them would cost more
    than it saves; the character count orders them almost the same way.
    Under the long-context encoder (enc.truncate "middle") a structured
    description counts as the text it renders to (criterion_text), as in the
    code that trained v6.0-VL; otherwise as before."""
    c, q = pair
    if enc is not None and enc.truncate == "middle":
        return (len(c.state) + len(q.instructions)
                + sum(len(o) + len(criterion_text(q.criteria.get(o, ""))) for o in q.options))
    return (len(c.state) + len(q.instructions)
            + sum(len(o) + len(q.criteria.get(o, "")) for o in q.options))


def _bucketed_stream(pairs, order, n, batch_size, k, rng, enc: EncodeConfig | None = None) -> list[int]:
    """The first n indices of the cyclic data order, regrouped by length within
    windows of k batches. Every window holds exactly the examples the unbucketed
    stream would have visited over the same steps."""
    flat = [order[i % len(order)] for i in range(n)]
    out: list[int] = []
    w = batch_size * k
    for s in range(0, n, w):
        win = sorted(flat[s:s + w], key=lambda i: _approx_len(pairs[i], enc))
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


def _top_conf_correct(logits: torch.Tensor, gold_idx: torch.Tensor):
    """(log m, log(1 - m), correct), m = the top-1 probability (first-max rule)."""
    lp = F.log_softmax(logits.float(), -1)
    z = logits.float().masked_fill(~torch.isfinite(logits), -1e30)
    top = z.argmax(-1)
    logm = lp.gather(1, top[:, None]).squeeze(1)
    log1m = torch.log(-torch.expm1(logm.clamp(max=-1e-7)))
    return logm, log1m, (top == gold_idx).float()


def conf_rank_loss(logits, gold_idx, *, margin_temp: float):
    """Confidence ranking: RankNet over (correct, wrong) row pairs of s = logit-odds of
    the top-1 probability / margin_temp. Returns (loss, soft AUC, n_pairs); with no
    pair, a zero loss whose gradient path never touches the -inf option mask."""
    logm, log1m, correct = _top_conf_correct(logits, gold_idx)
    s = (logm - log1m) / margin_temp
    pos, neg = correct > 0.5, correct <= 0.5
    if not (bool(pos.any()) and bool(neg.any())):
        safe = logits.float().masked_fill(~torch.isfinite(logits), 0.0)
        return safe.sum() * 0.0, float("nan"), 0
    d = s[pos][:, None] - s[neg][None, :]
    return F.softplus(-d).mean(), float((d.detach() > 0).float().mean()), int(d.numel())


def _group_grad_norms(opt, safe: bool = False, scaled: set | None = None) -> dict:
    """Pre-clip L2 gradient norm per param group, in fp32 as clip_grad_norm_ computes it
    (inf once sum(g^2) overflows, although every gradient is finite). safe=True: such a
    group's norm is recomputed overflow-free as amax x ||g / amax|| and its name added
    to `scaled`; it stays inf/nan only when a gradient really is inf/nan."""
    out = {}
    for gr in opt.param_groups:
        gps = [p.grad.detach() for p in gr["params"] if p.grad is not None]
        n = float(torch.linalg.vector_norm(torch.stack([g.float().norm() for g in gps]))) if gps else 0.0
        if safe and not math.isfinite(n) and gps:
            amax = float(torch.stack([g.float().abs().max() for g in gps]).max())
            if math.isfinite(amax) and amax > 0:
                n = amax * float(torch.linalg.vector_norm(torch.stack([(g.float() / amax).norm() for g in gps])))
                if scaled is not None and math.isfinite(n):
                    scaled.add(gr["name"])
        out[gr["name"]] = n
    return out


def _clip_groups(opt, sel, max_norm: float, gnorm: dict, scaled: set) -> None:
    """clip_grad_norm_ over the groups sel() picks. When one of them overflowed the fp32
    norm (finite but huge gradients, see _group_grad_norms) the clip uses the
    overflow-free norm, so the gradients are scaled down instead of multiplied by
    max_norm / inf = 0. Otherwise exactly clip_grad_norm_."""
    grs = [gr for gr in opt.param_groups if sel(gr)]
    ps = [p for gr in grs for p in gr["params"]]
    if not ps:
        return
    if not any(gr["name"] in scaled for gr in grs):
        torch.nn.utils.clip_grad_norm_(ps, max_norm)
        return
    total = math.sqrt(sum(float(gnorm.get(gr["name"], 0.0)) ** 2 for gr in grs))
    coef = min(1.0, max_norm / (total + 1e-6))
    for p in ps:
        if p.grad is not None:
            p.grad.mul_(coef)


def _nonfinite_decision(loss_finite: bool, gnorm: dict, aux_detached: bool) -> str:
    """'ok' | 'skip' (no update at all) | 'skip_aux' (drop the aux gradients, step the rest)."""
    if not loss_finite:
        return "skip"
    bad = sorted(k for k, v in gnorm.items() if not math.isfinite(v))
    if not bad:
        return "ok"
    return "skip_aux" if (aux_detached and bad == ["aux"]) else "skip"


def _aux_weights(cfg: "FitConfig", aux_heads) -> dict:
    """{exit: weight} of the aux-exit loss terms (aux_exit_weights, then aux_weight_total)."""
    aux_w = {int(k): float(v) for k, v in (cfg.aux_exit_weights or {}).items()}
    if cfg.aux_weight_total and aux_heads is not None:
        for k in aux_heads:
            aux_w.setdefault(int(k), float(cfg.aux_weight_total) / len(aux_heads))
    return aux_w


def aux_exit_term(cfg: "FitConfig", main_logits, aux_logits, gold, option_mask, *, progress: float):
    """One aux exit's loss term (before its weight), and its parts for the log.

    main_logits / aux_logits: canonical option order, -inf where masked. CE (the fit's
    objective) to gold, + aux_distill x T^2 x KL(p_main || p_aux) at T = aux_distill_temp
    with the teacher detached, + aux_zloss x mean(logsumexp(aux)^2)."""
    om = option_mask
    ce_ = objective_loss(cfg.objective, aux_logits, gold, progress=progress, rl=cfg.rl, generator=None)
    term, parts = ce_, {"loss": ce_.detach()}
    if cfg.aux_distill:
        T_ = float(cfg.aux_distill_temp)
        t_lp = torch.log_softmax(main_logits.detach().float() / T_, dim=-1)
        s_lp = torch.log_softmax(aux_logits / T_, dim=-1).masked_fill(~om, 0.0)
        tl = t_lp.masked_fill(~om, 0.0)
        kl_ = (tl.exp() * om * (tl - s_lp)).sum(-1).mean()
        term = term + cfg.aux_distill * (T_ ** 2) * kl_
        parts["kl"] = kl_.detach()
    if cfg.aux_zloss:
        lse = torch.logsumexp(aux_logits, dim=-1)                  # masked options are -inf
        zl_ = (lse ** 2).mean()
        term = term + float(cfg.aux_zloss) * zl_
        parts["z"] = zl_.detach()
    return term, parts


def fit(model, tokenizer, cases: Sequence[Case], enc: EncodeConfig, cfg: FitConfig, *,
        max_options: int, seed: int, device: str = "cuda", _helpers: dict | None = None) -> dict:
    """Train in place. Returns the averaged scorer state plus a small history."""
    g = seed_everything(seed)
    dev_type0 = "cuda" if str(device).startswith("cuda") else "cpu"
    if cfg.train_max_length:
        import dataclasses as _dc
        enc = _dc.replace(enc, max_length=int(cfg.train_max_length))
        print(f"    train_max_length={cfg.train_max_length}: training rows encoded at that cap "
              f"(truncate {enc.truncate})", flush=True)
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
    # ---- early-exit heads (arch aux_exits): which are trained, and where they start
    aux_heads = getattr(model, "aux_scorers", None)
    aux_w = _aux_weights(cfg, aux_heads)
    if cfg.aux_distill and aux_heads is None:
        raise ValueError("aux_distill needs arch aux_exits")
    if aux_w and aux_heads is None:
        raise ValueError("aux_exit_weights needs arch aux_exits")
    if cfg.aux_detach_tower and aux_heads is None:
        raise ValueError("aux_detach_tower needs arch aux_exits")
    if aux_heads is not None:
        if set(aux_w) - {int(k) for k in aux_heads}:
            raise ValueError(f"aux_exit_weights {sorted(aux_w)} not in aux_exits {sorted(aux_heads)}")
        if cfg.prior_kl:
            raise ValueError("aux exits: prior_kl is not supported")
        if cfg.aux_init_from:
            from pathlib import Path
            from safetensors.torch import load_file
            f = Path(cfg.aux_init_from)
            f = f / "aux_scorers.safetensors" if f.is_dir() else f
            sd = load_file(str(f))
            got = sorted({int(k.split(".", 1)[0]) for k in sd})
            if got != sorted(int(k) for k in aux_heads):
                raise ValueError(f"aux_init_from {f}: exits {got} != aux_exits {sorted(aux_heads)}")
            aux_heads.load_state_dict(sd, strict=True)
            print(f"    aux heads {got}: loaded from {f}", flush=True)
        elif init is not None and cfg.aux_init_from_main:
            main_sd = model.scorer.state_dict()
            for k, sc in aux_heads.items():
                sc.load_state_dict(main_sd)
            print(f"    aux heads {sorted(int(k) for k in aux_heads)}: initialised from the "
                  f"loaded main scorer", flush=True)
        print(f"    aux exit weights {aux_w} (main exit {model.cfg.exit_layer} weight 1); "
              f"distill {cfg.aux_distill} T={cfg.aux_distill_temp} detach_tower {cfg.aux_detach_tower}",
              flush=True)
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

    fused = bool(cfg.adam_fused) and str(device).startswith("cuda")
    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay,
                            **({"fused": True} if fused else {}))
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

    import time as _time
    from dataclasses import replace as _dc_replace
    canon_pref = tuple(x for x in (cfg.canonical_order_sources or "").split(",") if x)
    enc_canon = _dc_replace(enc, option_order="canonical") if canon_pref else enc
    cr_on = float(cfg.conf_rank_coef) > 0
    cr_gate = float(cfg.conf_rank_kl_gate or 0)
    if cr_on and cr_gate > 0 and not cfg.retention_kl:
        raise ValueError("conf_rank_kl_gate needs retention_kl > 0 (the fit's only KL to an anchor)")
    cr_start = int(float(cfg.conf_rank_start) * cfg.steps)
    cr_n_on = cr_n_gated = 0
    last_ret_kl = None
    if cr_on:
        print(f"    conf_rank: coef {cfg.conf_rank_coef} margin_temp {cfg.conf_rank_margin_temp} "
              f"kl_gate {cr_gate} (on ret_kl, lagged 1 step) from step {cr_start}; main exit only", flush=True)
    if canon_pref:
        print(f"    canonical_order_sources: {canon_pref}", flush=True)
    n_acc = max(1, int(cfg.grad_accum))
    if cfg.batch_size % n_acc:
        raise ValueError(f"grad_accum={n_acc} must divide batch_size={cfg.batch_size}")
    n_cut = 0                      # training rows whose state was cut to max_length
    n_split = 0                    # steps that ran as grad_accum micro-batches
    nf_max = int(cfg.nonfinite_skip_max)
    nf = {"run": 0, "total": 0, "aux_run": 0, "aux_total": 0}     # consecutive / total skips
    if nf_max:
        print(f"    nonfinite guard: skip non-finite steps, fail after {nf_max} consecutive"
              + (" (aux group alone -> drop aux grads only)" if cfg.aux_detach_tower else "")
              + (f"; aux z-loss {cfg.aux_zloss}" if cfg.aux_zloss else ""), flush=True)
    elif cfg.aux_zloss:
        print(f"    aux z-loss {cfg.aux_zloss}", flush=True)
    # history keys that only these features produce, so a run without them logs as before
    new_log = n_acc > 1 or aux_heads is not None or cr_on or nf_max > 0
    pr = {"t": _time.perf_counter(), "s": 0}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    vis = None
    if cfg.vision:
        from .vision_fit import VisionFeed
        vis = VisionFeed(cfg.vision, tokenizer, model, device)
        if canon_pref:
            raise ValueError("canonical_order_sources is not supported with fit_extra.vision")
    model.train()
    history, ckpts, aux_ckpts = [], [], []
    stream = None
    if cfg.length_bucket:
        stream = _bucketed_stream(pairs, order, cfg.steps * cfg.batch_size, cfg.batch_size,
                                  cfg.length_bucket, random.Random(seed + 7919), enc)
    cursor = 0
    for step in range(cfg.steps):
        if stream is not None:
            idx = stream[cursor:cursor + cfg.batch_size]
        else:
            idx = [order[(cursor + i) % len(order)] for i in range(cfg.batch_size)]
        cursor += cfg.batch_size
        chunk = [pairs[i] for i in idx]
        feat["on"] = step % cfg.log_every == 0 or step == cfg.steps - 1
        dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
        # Encode the whole step first: the same rows, order and option shuffles as one pass.
        if vis is not None:
            enc_all = vis.encode(chunk, enc, order_rng)
        else:
            enc_all = [encode_question(tokenizer, c.state, q,
                                       enc_canon if (canon_pref and c.source.startswith(canon_pref)) else enc,
                                       rng=order_rng) for c, q in chunk]
        cr_step = cr_on and step >= cr_start
        cr_gated = bool(cr_step and cr_gate > 0 and last_ret_kl is not None and last_ret_kl > cr_gate)
        cr_n_on += int(cr_step)
        cr_n_gated += int(cr_gated)
        cr_acc = {"loss": 0.0, "pairs": 0, "auc_num": 0.0}
        ce_main = 0.0
        n_cut += sum(1 for e in enc_all if e.get("state_cut"))
        step_len = max(len(e["input_ids"]) for e in enc_all)
        n_acc_step = n_acc if (not cfg.grad_accum_min_tokens or step_len > cfg.grad_accum_min_tokens) else 1
        n_split += n_acc_step > 1
        mb = cfg.batch_size // n_acc_step
        loss_parts, logit_parts, aux_loss, aux_amax = [], [], {}, {}
        ret_kl = None
        for j in range(0, len(chunk), mb):
            sub = chunk[j:j + mb]
            encoded = enc_all[j:j + mb]
            share = len(sub) / len(chunk)
            batch = (vis.collate(encoded, max_options) if vis is not None else
                     collate(tokenizer, encoded, max_options=max_options, device=device))
            gold = gold_tensor([c for c, _ in sub], [q.key for _, q in sub],
                               max_options, device=device)
            if cfg.label_smoothing:
                # Uniform over the question's own options. The option COUNT does not
                # depend on presentation order, and gold fills the first n slots.
                n = batch["option_mask"].sum(1, keepdim=True).float()
                slots = torch.arange(gold.shape[1], device=gold.device).unsqueeze(0)
                uni = (slots < n).float() / n
                gold = (1 - cfg.label_smoothing) * gold + cfg.label_smoothing * uni

            extra = {}
            if scale_map:
                sc = [next((v for k, v in scale_map.items() if c.source.startswith(k)), 1.0)
                      for c, _ in sub]
                extra["row_grad_scale"] = torch.tensor(sc, dtype=torch.float32, device=device)
            with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cfg.autocast_bf16):
                exits = None
                if cfg.prior_kl:
                    logits, base = model.forward_with_base(**batch, **extra)   # ONE tower pass
                elif aux_heads is not None:
                    exits = model.forward_exits(**batch, **extra,              # ONE tower pass
                                                **({"aux_detach": True} if cfg.aux_detach_tower else {}))
                    logits, base = exits[int(model.cfg.exit_layer)], None
                else:
                    logits, base = model(**batch, **extra), None
            logits = logits.float()
            # Back to canonical option order BEFORE the loss: gold indexes
            # q.options, so a permuted presentation compared unmapped would train
            # against the wrong option.
            logits = unpermute_logits(logits, batch["option_perm"], batch["option_mask"])
            # masked options must not receive gradient through the loss
            logits = logits.masked_fill(~batch["option_mask"], float("-inf"))
            progress = step / max(1, cfg.steps - 1)
            loss = objective_loss(cfg.objective, logits.float(), gold,
                                  progress=progress, rl=cfg.rl,
                                  generator=None)
            ce_main += float(loss.detach()) * share
            if cfg.prior_kl:
                loss = loss + cfg.prior_kl * prior_kl(logits.float(), base)
            if cr_step and not cr_gated:
                rk, auc_, np_ = conf_rank_loss(logits, gold.argmax(-1),
                                               margin_temp=float(cfg.conf_rank_margin_temp))
                loss = loss + float(cfg.conf_rank_coef) * rk
                cr_acc["loss"] += float(rk.detach()) * share
                if np_:
                    cr_acc["pairs"] += np_
                    cr_acc["auc_num"] += auc_ * np_
            if exits is not None:
                om = batch["option_mask"]
                for L_, w_ in aux_w.items():
                    if not w_:
                        continue
                    al = unpermute_logits(exits[L_].float(), batch["option_perm"], om)
                    al = al.masked_fill(~om, float("-inf"))
                    term, parts = aux_exit_term(cfg, logits, al, gold, om, progress=progress)
                    for k_, v_ in parts.items():
                        key = f"aux{L_}_{k_}"
                        aux_loss[key] = aux_loss.get(key, 0.0) + v_ * share
                    if nf_max or feat["on"]:
                        fa = al.detach()[torch.isfinite(al.detach())]
                        aux_amax[L_] = max(aux_amax.get(L_, 0.0),
                                           float(fa.abs().max()) if fa.numel() else float("nan"))
                    loss = loss + w_ * term
            if n_acc_step > 1:
                # every term of the task loss is a mean over rows: weight by row share
                loss = loss * share
            if j + mb >= len(chunk) and cfg.retention_kl:
                # the step-level term, once per step, on the last micro-batch
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
                    for jj, i in enumerate(pick):
                        q_lp, q_ix = ret_tgt[i]
                        n = q_lp.shape[0]
                        # checkpointed: the (n, V) student log-probs are recomputed in
                        # backward instead of being held for every row at once.
                        kls.append(torch.utils.checkpoint.checkpoint(
                            _row_retention_kl, r_final[jj, :n].float(), W,
                            q_lp.to(device, non_blocking=True).float(),
                            q_ix.to(device, non_blocking=True).long(),
                            use_reentrant=False))
                ret_kl = torch.stack(kls).mean()
                loss = loss + cfg.retention_kl * ret_kl
            mb_finite = bool(torch.isfinite(loss))
            if not mb_finite and not nf_max:
                raise FloatingPointError(
                    f"non-finite loss {loss.item()} at step {step}/{cfg.steps}; "
                    f"last logged {history[-3:]}")
            if j == 0:
                opt.zero_grad(set_to_none=True)
            if mb_finite:            # guard on: a non-finite micro-batch is not backpropagated
                loss.backward()
            loss_parts.append(loss.detach())
            logit_parts.append(logits.detach())
        loss = loss_parts[0] if n_acc_step == 1 else torch.stack(loss_parts).sum()
        logits = logit_parts[0] if n_acc_step == 1 else torch.cat([l.reshape(-1) for l in logit_parts])
        logging = step % cfg.log_every == 0 or step == cfg.steps - 1
        nf_scaled: set = set()
        if logging or nf_max:
            # Pre-clip gradient norm per group and the largest finite logit: the
            # two training-time signals that say WHERE an instability starts,
            # without looking at any evaluation target.
            gnorm = _group_grad_norms(opt, safe=bool(nf_max), scaled=nf_scaled)
            fin = logits.detach()[torch.isfinite(logits.detach())]
            logit_absmax = float(fin.abs().max()) if fin.numel() else float("nan")
        nf_dec = "ok"
        if nf_max:
            nf_dec = _nonfinite_decision(bool(torch.isfinite(loss.detach()).all()), gnorm,
                                         bool(cfg.aux_detach_tower))
            if nf_dec == "skip_aux":
                for gr in opt.param_groups:
                    if gr["name"] == "aux":
                        for p in gr["params"]:
                            p.grad = None
                nf["aux_run"] += 1
                nf["aux_total"] += 1
            elif nf_dec == "ok":
                nf["aux_run"] = 0
            if nf_dec == "skip":
                opt.zero_grad(set_to_none=True)
                nf["run"] += 1
                nf["total"] += 1
            else:
                nf["run"] = 0
            if nf_dec != "ok":
                print(f"    nonfinite: step {step + 1}/{cfg.steps} {nf_dec} loss {float(loss.detach()):.4g} "
                      f"gn {{{', '.join(f'{k}: {v:.4g}' for k, v in gnorm.items())}}} "
                      f"aux_absmax {{{', '.join(f'{k}: {v:.4g}' for k, v in sorted(aux_amax.items()))}}} "
                      f"run {nf['run']}/{nf['aux_run']} total {nf['total']}/{nf['aux_total']} (max {nf_max})",
                      flush=True)
            if max(nf["run"], nf["aux_run"]) > nf_max:
                raise FloatingPointError(
                    f"non-finite loss/grad on {max(nf['run'], nf['aux_run'])} consecutive steps (> "
                    f"nonfinite_skip_max {nf_max}) at step {step}/{cfg.steps}: {nf_dec}, loss "
                    f"{float(loss.detach())}, gn {gnorm}; last logged {history[-3:]}")
        if nf_dec == "skip":
            pass                                     # no clip, no update: grads are already None
        elif nf_max and cfg.grad_clip:
            # same clip as below, overflow-safe (_clip_groups == clip_grad_norm_ unless a norm overflowed)
            sels = ((lambda g_: g_["name"] != "aux", lambda g_: g_["name"] == "aux") if cfg.aux_detach_tower
                    else (lambda g_: True,))
            for sel in sels:
                _clip_groups(opt, sel, cfg.grad_clip, gnorm, nf_scaled)
            if nf_scaled:
                print(f"    nonfinite: step {step + 1}/{cfg.steps} fp32 grad norm overflow in "
                      f"{sorted(nf_scaled)} (finite grads) -> clipped with the safe norm", flush=True)
        elif cfg.grad_clip and cfg.aux_detach_tower:
            # detached aux heads are clipped on their own, so the tower + main head step
            # is exactly the step without aux exits
            for sel in (lambda g_: g_["name"] != "aux", lambda g_: g_["name"] == "aux"):
                ps = [p for gr in opt.param_groups if sel(gr) for p in gr["params"]]
                if ps:
                    torch.nn.utils.clip_grad_norm_(ps, cfg.grad_clip)
        elif cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                [p for gr in opt.param_groups for p in gr["params"]], cfg.grad_clip)
        if nf_dec != "skip":
            opt.step()
        sched.step()

        if ret_kl is not None:
            last_ret_kl = float(ret_kl.detach())
        cr_rec = {}
        if cr_on:
            cr_rec = {"cr_loss": round(cr_acc["loss"], 5), "cr_pairs": cr_acc["pairs"],
                      "cr_auc": round(cr_acc["auc_num"] / cr_acc["pairs"], 4) if cr_acc["pairs"] else None,
                      "cr_gated": int(cr_gated), "cr_gated_frac": round(cr_n_gated / max(1, cr_n_on), 4)}
        if cfg.print_every and (step % cfg.print_every == 0 or step == cfg.steps - 1):
            now = _time.perf_counter()
            sps = (now - pr["t"]) / max(1, step + 1 - pr["s"])
            pr["t"], pr["s"] = now, step + 1
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
            terms = " ".join(f"{k} {float(v):.4f}" for k, v in aux_loss.items())
            print(f"    step {step + 1}/{cfg.steps} loss {float(loss.detach()):.4f} ce {ce_main:.4f}"
                  + (f" ret_kl {float(ret_kl.detach()):.5f}" if ret_kl is not None else "")
                  + (f" {terms}" if terms else "")
                  + (f" cr_loss {cr_rec['cr_loss']:.4f} cr_pairs {cr_rec['cr_pairs']} cr_auc {cr_rec['cr_auc']}"
                     f" cr_gated {cr_rec['cr_gated']} gated_frac {cr_rec['cr_gated_frac']}" if cr_on else "")
                  + f" step_len {step_len} accum {n_acc_step} s/step {sps:.3f}"
                  + f" eta_h {sps * (cfg.steps - step - 1) / 3600:.2f} peak_gib {peak:.1f}", flush=True)

        if logging:
            history.append({"step": step, "loss": float(loss.detach()),
                            **({"ce": round(ce_main, 5), **cr_rec,
                                **{k: round(float(v), 5) for k, v in aux_loss.items()},
                                "step_len": step_len, "accum": n_acc_step} if new_log else {}),
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
                            **{f"gn_{k}": round(v, 4) for k, v in gnorm.items()},
                            **({f"aux{k}_absmax": round(v, 2) for k, v in sorted(aux_amax.items())}),
                            **({"w_aux": round(float(torch.sqrt(sum((p.detach().float() ** 2).sum()
                                                                    for p in aux_heads.parameters()))), 3)}
                               if aux_heads is not None else {}),
                            **({"nf_skips": nf["total"], "nf_aux_skips": nf["aux_total"]} if nf_max else {})})
        if step >= cfg.steps - cfg.keep_last_k:
            ckpts.append({k: v.detach().cpu().clone()
                          for k, v in model.scorer.state_dict().items()})
            if aux_heads is not None:
                aux_ckpts.append({k: v.detach().cpu().clone()
                                  for k, v in aux_heads.state_dict().items()})

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
        if aux_ckpts:
            av = {k: torch.stack([c[k].float() for c in aux_ckpts]).mean(0) for k in aux_ckpts[0]}
            aux_heads.load_state_dict({k: v.to(next(aux_heads.parameters()).dtype)
                                       for k, v in av.items()})
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
    if aux_heads is not None and cfg.aux_save_dir:
        from pathlib import Path
        from safetensors.torch import save_file
        d = Path(cfg.aux_save_dir)
        d.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().float().contiguous().cpu() for k, v in aux_heads.state_dict().items()},
                  str(d / "aux_scorers.safetensors"))
        print(f"    saved aux heads -> {d / 'aux_scorers.safetensors'}", flush=True)

    if init is not None and history:
        history[0].update(init)
    out = {"history": history, "averaged_over": len(ckpts), "init": init,
           "examples_seen": cfg.steps * cfg.batch_size, "seed": seed,
           "calibration": cal_report}
    if new_log:
        out.update({"grad_accum": n_acc, "state_cut_rows": n_cut,
                    "grad_accum_min_tokens": cfg.grad_accum_min_tokens, "split_steps": n_split,
                    "conf_rank": ({"steps_on": cr_n_on, "steps_gated": cr_n_gated,
                                   "gated_frac": round(cr_n_gated / max(1, cr_n_on), 5)} if cr_on else None)})
    return out
