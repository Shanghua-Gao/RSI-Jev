"""The training loop. PROTECTED in its bookkeeping, EDITABLE through its config.

Arms in one comparison share seed, initialisation, data order and optimiser step
count, and differ in exactly one thing (common random numbers). The step count is
fixed rather than the epoch count so that a data-axis arm that changes corpus
size does not silently also change the amount of optimisation.
"""
from __future__ import annotations

# longctx: the worker env has no per-arm variable, so the allocator setting lives in the arm's code.
# This module is imported by run_arm_lib before any CUDA allocation, so the caching allocator reads it at init.
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import copy
import dataclasses
import math
import random
from dataclasses import dataclass, field
from typing import Sequence

import torch
try:  # also applies if CUDA was initialised earlier than expected
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
except Exception as _e:  # noqa: BLE001
    print(f"[fit] _set_allocator_settings failed: {_e}", flush=True)
print(f"[fit] PYTORCH_CUDA_ALLOC_CONF={_os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)
import torch.nn.functional as F
import torch.utils.checkpoint

from .contract import Case
from .encode import (EncodeConfig, collate, criterion_text, encode_question, gold_tensor,
                     iter_questions, unpermute_logits)
from .train import Objective, RLConfig, objective_loss, prior_kl, seed_everything


def _top_conf_correct(logits: torch.Tensor, gold_idx: torch.Tensor):
    """(log m, log(1-m), correct), m = top-1 probability (first-max rule). Verbatim rl2.top_conf_correct (c4e5959)."""
    lp = F.log_softmax(logits.float(), -1)
    z = logits.float().masked_fill(~torch.isfinite(logits), -1e30)
    top = z.argmax(-1)
    logm = lp.gather(1, top[:, None]).squeeze(1)
    log1m = torch.log(-torch.expm1(logm.clamp(max=-1e-7)))
    return logm, log1m, (top == gold_idx).float()


def conf_rank_loss(logits, gold_idx, *, margin_temp: float):
    """P4 confidence ranking: RankNet over (correct, wrong) pairs of s = logit-odds of top-1. Verbatim
    rl2.conf_rank_loss (rsi-jev c4e5959). Returns (loss, soft AUC, n_pairs); no pair -> zero loss with a
    gradient path that never touches the -inf option mask."""
    logm, log1m, correct = _top_conf_correct(logits, gold_idx)
    s = (logm - log1m) / margin_temp
    pos, neg = correct > 0.5, correct <= 0.5
    if not (bool(pos.any()) and bool(neg.any())):
        safe = logits.float().masked_fill(~torch.isfinite(logits), 0.0)
        return safe.sum() * 0.0, float("nan"), 0
    d = s[pos][:, None] - s[neg][None, :]
    return F.softplus(-d).mean(), float((d.detach() > 0).float().mean()), int(d.numel())


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
    rl: RLConfig = field(default_factory=RLConfig)
    log_every: int = 50
    # Length-bucketed batches, OFF by default (0): see rsi-jev-internal eval-accel fit.py.
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
    ddp: bool = False                # A94: torchrun data parallel (caller sets device + inits the group); see keyv1 fit
    ref_kl: float = 0.0              # A93: KL(REF || model), main-exit option logits; REF = frozen copy after init
    retention_source_prefixes: str = "dc_"   # Case.source prefixes (mc_replay = dc_*)
    retention_rows: int = 4                  # retention rows per step
    retention_pool: int = 2048               # rows whose base targets are cached
    retention_topk: int = 64
    retention_max_tokens: int = 384
    # (b) Per-source TOWER gradient weight: {source_prefix: scale}. The scale
    # multiplies that row's gradient into the tower only (arch._RowGradScale);
    # the scorer sees the full loss. {} = every row at 1.0 (champion behaviour).
    source_tower_scale: dict = field(default_factory=dict)
    # cand-2b-1 (a): decoder layers 0..lower_layers_n-1 train at lr_base x
    # lower_layers_lr_scale (own param group, same tower schedule). A softer form
    # of mmlu-keep-1c's freeze. n=0 / scale 1.0 = champion.
    lower_layers_n: int = 0
    lower_layers_lr_scale: float = 1.0
    # ---- A16 continued training --------------------------------------------
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
    # Image states (rsijev/vision_fit.py): {"root" | "roots", "budget", "model"}. Empty = text only.
    vision: dict = field(default_factory=dict)
    init_sha256: str = ""
    # Keep a case's questions adjacent in the data order (the case order is still the seeded shuffle), so paired
    # questions on one state (q / not-q) land in the same or the next batch. Off (False) = the historical order.
    group_case_questions: bool = False
    # longctx: gradient accumulation. Each step's batch_size examples (same ones, same order) run as grad_accum
    # micro-batches; each micro task loss is weighted by its row share; the step-level terms (listwise, retention)
    # are added once, on the last micro-batch. Optimiser step, schedule and clipping unchanged. 1 = one pass.
    grad_accum: int = 1
    # ax6: apply grad_accum only where needed. 0 = every step (m4b7 behaviour). >0 = a step is split into
    # grad_accum micro-batches only when its longest encoded row exceeds this many tokens; shorter steps run as
    # one pass (identical to grad_accum 1). The data order and option shuffles do not depend on the split.
    grad_accum_min_tokens: int = 0
    # keyv1 (A94b port): with grad_accum_min_tokens > 0, a step whose longest row is <= that many tokens runs as
    # grad_accum_short micro-batches instead of one pass (0 = one pass, the old behaviour).
    grad_accum_short: int = 0
    # ladder-v1: cap the TRAINING encoder's max_length (middle cut above it; image rows still get
    # their vision source budget on top). 0 = the code default (32k). Eval/scoring encoders are untouched.
    train_max_length: int = 0
    # a120 OOM fix (2026-10-14: drop, never cut): JSON {"cap", "n_drop", "drop": [[case_id, question_key],
    # ...]} of training questions whose encoded length exceeds cap (jobs/a120/cap/lenchk_a120.py). Listed questions are removed
    # before the data order is drawn; every listed pair must exist. "" = off (unchanged behaviour).
    train_drop_file: str = ""
    adam_fused: bool = False
    # S4 (arch_goal_plan): supervised LISTWISE relevance loss. Cases whose source starts with listwise_source are
    # grouped by case_id prefix before '#' (one query's candidates); every listwise_every steps one group of up to
    # listwise_group candidates (all positives up to half, the rest negatives) is scored together, and
    # listwise_weight x CE(softmax over candidates of the noul log-odds, gold = positives normalised) is added to the
    # step's loss. The group stream has its own seeded rng, so the pointwise data order is identical to a
    # listwise_weight=0 arm (the matched pair). 0 = off.
    listwise_weight: float = 0.0
    listwise_source: str = ""
    listwise_group: int = 8
    listwise_every: int = 1
    # arch (early exit, continued training): after init_from loads the
    # parent, put the decoder layers the scorer no longer runs (exit_layer..) and
    # the text model's final norm back to BASE weights. The scorer never reads
    # them; only the LM-retention term does, and a tuned tower's top layers are
    # its most damaged part (layer 23 cos .41 to base, perplexity 25k).
    exit_reset_upper: bool = False
    # big4b (early exit trained from base, layers above the exit DROPPED):
    # the LM-retention term runs the EXIT model, i.e. the first exit_layer layers
    # + final norm + tied lm_head, for both the base targets (the base model cut at
    # the same layer) and the student. Nothing then runs layers >= exit_layer, so
    # they are frozen (no gradient could reach them) and a step costs the
    # truncated tower only. Needs arch exit_layer with exit_norm. Off = the
    # historical full-depth retention (targets and student at full depth).
    retention_at_exit: bool = False
    # exit-reason rt2b (ped-jm-e12rt): full-depth retention with layers >= exit_layer frozen; pool from a text file.
    retention_freeze_upper: bool = False
    retention_text_file: str = ""
    # ---- adaexit: deep supervision of the aux exits (arch aux_exits) ----
    # loss = main-exit loss + sum_L aux_exit_weights[L] x (same objective at aux exit L).
    # Keys are exit indices as strings (JSON). Gradients of the aux terms flow into
    # the tower (layers < L) and into that exit's own head. {} with aux heads present
    # = weight 0 (heads never trained).
    aux_exit_weights: dict = field(default_factory=dict)
    # lr of the aux heads (own param group, the head schedule). 0 = lr_head.
    lr_aux: float = 0.0
    # After init_from: copy the loaded main scorer into every aux head (start each
    # shallow head at the trained deep head, instead of at noise).
    aux_init_from_main: bool = True
    # vis-ax1: load the aux heads from a saved aux_scorers.safetensors (a file, or a dir
    # holding one, e.g. the AX-1 parent's ckpt), strictly. Takes precedence over
    # aux_init_from_main, so a continued multi-exit arm keeps its parent's trained heads.
    aux_init_from: str = ""
    # Start the data stream at step data_offset_steps of this seed's order (the same
    # cyclic order and length buckets as a run of data_offset_steps + steps steps),
    # so a continued arm sees the questions its parent did NOT see. 0 = from the start.
    data_offset_steps: int = 0
    # Where fit() writes the averaged aux heads (aux_scorers.safetensors, keys "<exit>.<param>").
    # run_arm_lib.save_release is protected and saves the main scorer only, so the arm's
    # spec points this at its own save_dir. "" = not saved.
    aux_save_dir: str = ""
    # ax6: the aux exits come from ONE list (arch aux_exits). aux_weight_total > 0 gives every aux exit that has no
    # entry in aux_exit_weights the weight aux_weight_total / len(aux_exits), so changing the exit list keeps the
    # total deep-supervision weight fixed. 0 = only aux_exit_weights (AX-1 behaviour).
    aux_weight_total: float = 0.0
    # ax6: self-distillation of every aux exit to the main exit's distribution (teacher detached: no gradient into
    # the main head or through the teacher). Per aux exit L: w_L x (CE_L + aux_distill x KL(p_main || p_L)) at
    # temperature aux_distill_temp (the KL is scaled by T^2). 0 = off (plain deep supervision).
    aux_distill: float = 0.0
    aux_distill_temp: float = 1.0
    # ---- lad2 port (lad2, 2026-10-04) from the rt9 tree; every key OFF by default (= ladder-v1 behaviour) ----
    # aux_detach_tower (rt9-ladder-detach): aux exits read norm(h_L).detach() (arch._forward_exits aux_detach), so aux CE/KL
    # train only the aux heads and the trunk sees the main loss alone. The aux group is grad-clipped separately.
    aux_detach_tower: bool = False
    # A39: one stdout line every print_every steps (loss terms, s/step, ETA, peak GiB). 0 = off.
    # keyv1 / rt9 (A94b port) P4 confidence-ranking AUX term: loss += conf_rank_coef x RankNet over (correct, wrong)
    # row pairs of the MAIN exit's top-1 logit-odds / conf_rank_margin_temp, per micro-batch. A step whose PREVIOUS
    # step's ret_kl exceeds conf_rank_kl_gate drops the term (needs retention_kl > 0 unless the gate is <= 0).
    # conf_rank_start: fraction of steps before the term turns on. 0 = off.
    conf_rank_coef: float = 0.0
    conf_rank_margin_temp: float = 1.0
    conf_rank_kl_gate: float = 0.20
    conf_rank_start: float = 0.0
    # Case.source prefixes (comma list) presented in CANONICAL option order instead of enc.option_order. "" = off.
    canonical_order_sources: str = ""
    # proxy_v1 P2 arm b (method 2026-10-15, ped-proxyb): permutation-consistency term. Every micro-batch row is re-encoded with its
    # presented option order REVERSED (no rng draw: data order and option shuffles equal the perm_kl=0 control), scored under
    # no_grad, and loss += perm_kl x KL(sg p_rev || p) over the row's own options (row mean). Training only; inference unchanged. 0 = off.
    # canonical_order_sources rows (order = meaning, e.g. pf_rt9__manyopt_seq) get weight 0 in that mean.
    perm_kl: float = 0.0
    # (2026-10-15): listwise CE over a CYCLIC ROTATION of the same item's presented option order, WITH grad. Per micro-batch
    # row a shift k in 1..K-1 (K = the row's option count) is drawn from a keyed rng (seed, step, row): it never touches order_rng, so data
    # order / option shuffles equal the lw_ce=0 control. loss += lw_ce x objective_loss(logits_rot, gold). Gold-supervised (not a KL to a
    # reference): the same question is trained under two orders. canonical_order_sources rows are excluded. Training only. 0 = off.
    lw_ce: float = 0.0
    print_every: int = 0
    # A53/A64 step checkpoints + resume (rsijev/ckpt_resume.py): ckpt_every steps and/or ckpt_minutes minutes; SIGUSR1 ->
    # save + exit 99, SIGTERM -> save + exit 143; 3-part fingerprint (spec/data/code), DONE marker. 0/0 = off.
    ckpt_every: int = 0
    ckpt_dir: str = ""
    ckpt_keep: int = 2
    ckpt_minutes: float = 0.0
    # non-finite guard (rt9d2): skip a step with a non-finite loss / grad norm (aux group alone under aux_detach_tower ->
    # drop only the aux grads); > nonfinite_skip_max consecutive -> FloatingPointError. 0 = off (raise at once).
    nonfinite_skip_max: int = 0
    # aux_zloss > 0: + aux_zloss x mean(logsumexp(aux logits)^2) per aux exit (aux heads only). 0 = off.
    aux_zloss: float = 0.0
    # ISM (vis7): image-swap evidence margin. For an image
    # row (x, I, gold g) with a partner row of the SAME source with the SAME state text, question and option texts whose gold
    # option text differs (so g is not gold on the partner image I' by construction), a second forward on (x, I') with the
    # row's own option presentation adds ism_weight * relu(ism_margin - [log p(g|x,I) - log p(g|x,I')]) (row mean over the
    # micro-batch). Partner pick: keyed RNG (seed, step, row), never order_rng, so the data order / option shuffles equal the
    # ism_weight=0 arm. Training only; inference unchanged. 0 = off.
    ism_weight: float = 0.0
    ism_margin: float = 1.0
    # RTB (vis8): render-twin binding margin. A two-option
    # image row (x, I, gold g) with a twin I_tw on which gold flips to g' (a rendered twin case named by the row's "twin" field,
    # same state/question/options; or, for sources in rtb_reverse_sources, the same images in reversed order) gets a second
    # forward on (x, I_tw) with the row's own option presentation and adds
    #   rtb_weight * relu(rtb_margin - [(s_g(I) - s_g'(I)) + (s_g'(I_tw) - s_g(I_tw))])   (logits, row mean over the micro-batch).
    # The twins' own CE comes from their own rows in the corpus (both arms). The order_rng stream is consumed exactly as in the
    # rtb_weight=0 arm, so data order and option shuffles equal the control. Training only; inference unchanged. 0 = off.
    rtb_weight: float = 0.0
    rtb_margin: float = 2.0
    rtb_reverse_sources: list = field(default_factory=list)
    # TCL (vis10):
    # in-LM token correspondence InfoNCE. For a two-image row whose vision row carries meta.anch ([[r, c, gr, gc, ...]]: I1 token
    # (r, c) maps into I2 token (gr, gc)), the hidden states of hs[tcl_layer] (the SAME forward as the CE; no extra pass) at the
    # 196 + 196 image-pad positions are (optionally per-image mean-centred,) L2-normalised and each anchor gets
    #   CE(cos(x1[a], x2[:]) / tcl_temp, gold token)          (mean over anchors, row mean over the micro-batch, x tcl_weight).
    # No parameters, no projection head; training only, inference unchanged. Rows without anch or without exactly 2 x 196
    # row-major image tokens get no term. order_rng is untouched, so data order / option shuffles equal the tcl_weight=0 arm.
    tcl_weight: float = 0.0
    tcl_layer: int = 12
    tcl_temp: float = 0.07
    tcl_center: bool = True
    # vis13 order lottery (Cell 0): data-order seed only. Feeds the
    # question/case shuffles and order_rng (option presentation); init, dropout and the other streams keep `seed`. None = seed.
    order_seed: int | None = None


def _param_groups(model, cfg: FitConfig):
    import re
    head, mix, base, lower, aux = [], [], [], [], []
    pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("mix_logits"):
            mix.append(p)
        elif n.startswith("aux_scorers"):
            aux.append(p)
        elif n.startswith("scorer"):
            head.append(p)
        else:
            m = pat.search(n) if cfg.lower_layers_n else None
            if m and "visual" not in n and int(m.group(1)) < cfg.lower_layers_n:
                if cfg.lower_layers_lr_scale == 0:
                    # ladder-v1: scale 0 = FROZEN (no grad, no optimiser state): fits a 32-layer
                    # tower + 4 heads on one 96 GB RTX. Any scale > 0 behaves exactly as before.
                    p.requires_grad_(False)
                    continue
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
        print(f"    lower_layers_n={cfg.lower_layers_n}: {sum(p.numel() for p in lower):,} params "
              f"at lr x{cfg.lower_layers_lr_scale}", flush=True)
    return groups


def _text_tower_embed(model):
    emb = model.tower.get_input_embeddings()
    if emb is None:
        raise ValueError("retention_kl needs the tower's input embedding (tied lm_head)")
    return emb.weight


def _retention_text_rows(tokenizer, cases, cfg: FitConfig, seed: int):
    """rt2b: retention pool from cfg.retention_text_file, restricted to this arm's training case_ids."""
    import json as _json
    ids_ok = {c.case_id for c in cases}
    pool = []
    with open(cfg.retention_text_file) as f:
        for ln in f:
            r = _json.loads(ln)
            if r["case_id"] in ids_ok and r.get("text"):
                pool.append(r["text"])
    if len(pool) < cfg.retention_pool:
        raise ValueError(f"retention_text_file: {len(pool)} rows in the training cases < retention_pool {cfg.retention_pool}")
    rng = random.Random(seed + 2718281)
    rng.shuffle(pool)
    pool = pool[: cfg.retention_pool]
    rows = [tokenizer(x, add_special_tokens=False)["input_ids"][-cfg.retention_max_tokens:] for x in pool]
    print(f"    retention_text_file: {len(pool)} rows (tail {cfg.retention_max_tokens} tok) from {cfg.retention_text_file}", flush=True)
    return rows


def _retention_rows(tokenizer, cases, cfg: FitConfig, seed: int):
    """Token-id lists for the retention pool: state + question + options, in
    canonical order (the same rendering the model trains on), first question
    of each case, truncated to retention_max_tokens."""
    if cfg.retention_text_file:
        return _retention_text_rows(tokenizer, cases, cfg, seed)
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
    """The context the LM-retention tower calls run in: the exit model under
    retention_at_exit, else the full tower."""
    import contextlib
    if not cfg.retention_at_exit:
        return contextlib.nullcontext()
    acfg = model.cfg
    if not getattr(acfg, "exit_layer", None) or not getattr(acfg, "exit_norm", True):
        raise ValueError("retention_at_exit needs arch exit_layer with exit_norm=True")
    return model.exit_tower()


def _freeze_above_exit(model) -> int:
    """requires_grad=False for decoder layers >= exit_layer (never run when the
    retention term runs the exit model). Returns the number of frozen params."""
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
            + sum(len(o) + len(criterion_text(q.criteria.get(o, ""))) for o in q.options))


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


def _upper_snapshot(model) -> dict:
    """CPU copies of the tower weights the early-exit scorer does not run: decoder
    layers >= arch.exit_layer and the text model's final norm."""
    import re
    L = getattr(model.cfg, "exit_layer", None)
    if not L:
        raise ValueError("exit_reset_upper needs arch exit_layer")
    pat = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    snap = {}
    for n, p in model.tower.named_parameters():
        if "visual" in n:
            continue
        m = pat.search(n)
        # the final norm is on the scoring path under exit_norm: keep the parent's
        top_norm = not m and n.endswith("norm.weight") and not getattr(model.cfg, "exit_norm", True)
        if (m and int(m.group(1)) >= int(L)) or top_norm:
            snap[n] = p.detach().to("cpu", copy=True)
    return snap


@torch.no_grad()
def _upper_restore(model, snap: dict) -> None:
    params = dict(model.tower.named_parameters())
    for n, t in snap.items():
        params[n].copy_(t.to(params[n].device, params[n].dtype))
    print(f"    exit_reset_upper: {len(snap)} tensors (layers >= {model.cfg.exit_layer}"
          f"{'' if getattr(model.cfg, 'exit_norm', True) else ' + final norm'}) reset to base "
          f"after init_from", flush=True)


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


# ---- lad2 port: verbatim rt9 helpers ----
def _group_grad_norms(opt, device, safe: bool = False, scaled: set | None = None) -> dict:
    """Pre-clip L2 grad norm per param group, fp32 as clip_grad_norm_ computes it (inf once sum(g^2) > 3.4e38, i.e.
    grads >~ 1e19, although every grad is finite). safe=True: such a group's norm is recomputed overflow-free as
    amax x ||g / amax|| and its name is added to `scaled`; it stays inf/nan only when a grad really is inf/nan."""
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
    """clip_grad_norm_ over the groups sel() picks; when one of them overflowed the fp32 norm (finite but huge grads,
    see _group_grad_norms) the clip uses the overflow-free norm, so the grads are scaled down instead of being
    multiplied by max_norm / inf = 0 (which turns every inf element into NaN). Otherwise exactly clip_grad_norm_."""
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
    """'ok' | 'skip' (no update at all) | 'skip_aux' (drop the aux grads, step the rest)."""
    if not loss_finite:
        return "skip"
    bad = sorted(k for k, v in gnorm.items() if not math.isfinite(v))
    if not bad:
        return "ok"
    return "skip_aux" if (aux_detached and bad == ["aux"]) else "skip"


def _step_hook(step: int) -> None:
    """Called after every completed optimiser step (after any step ckpt). Tests patch it to simulate a crash."""


_CKPT_KEYS = ("ckpt_every", "ckpt_dir", "ckpt_keep", "ckpt_minutes", "print_every", "snap_every", "snap_dir")
# train-efficiency keys: left out of the fingerprint at their default, so a ckpt written before they existed
# still resumes; a non-default value is part of it (it changes numerics or the step).
_EFF_DEFAULTS = {"nonfinite_skip_max": 0, "aux_zloss": 0.0, "aux_detach_tower": False,
                 # A94b keyv1 port
                 "conf_rank_coef": 0.0, "conf_rank_margin_temp": 1.0, "conf_rank_kl_gate": 0.20,
                 "conf_rank_start": 0.0, "canonical_order_sources": "", "grad_accum_short": 0}


def _ckpt_fingerprint(model, cfg: "FitConfig", seed: int, n_pairs: int) -> str:
    import dataclasses, hashlib, json
    c = {k: v for k, v in dataclasses.asdict(cfg).items() if k not in _CKPT_KEYS
         and not (k in _EFF_DEFAULTS and v == _EFF_DEFAULTS[k])}
    shapes = [(n, list(p.shape)) for n, p in model.named_parameters() if p.requires_grad]
    blob = json.dumps({"cfg": c, "seed": seed, "n_pairs": n_pairs, "shapes": shapes}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _rng_state() -> dict:
    s = {"py": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as _np
        s["np"] = _np.random.get_state()
    except Exception:  # noqa: BLE001
        pass
    return s


def _rng_restore(s: dict) -> None:
    random.setstate(s["py"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    if "np" in s:
        import numpy as _np
        _np.random.set_state(s["np"])


def _ckpt_root(cfg: "FitConfig"):
    from pathlib import Path
    d = cfg.ckpt_dir or _os.environ.get("RSIJEV_STEP_CKPT_DIR", "")
    if not d:
        raise ValueError("ckpt_every > 0 needs fit_extra.ckpt_dir (or RSIJEV_STEP_CKPT_DIR); exp.py sets <save_dir>/step_ckpt")
    return Path(d)


def _ckpt_parts(model, cfg: "FitConfig", seed: int, pairs) -> dict:
    """A64 three-part fingerprint: spec (A53 hash), data (the question stream), code (this package's sources)."""
    from pathlib import Path
    from . import ckpt_resume as CR
    data = CR.sha_strings(f"{c.case_id}|{len(c.state or '')}|{len(getattr(q, 'options', None) or ())}" for c, q in pairs)
    parts = {"spec": _ckpt_fingerprint(model, cfg, seed, len(pairs)), "data": data,
             "code": CR.code_sha(Path(__file__).resolve().parent)}
    parts["all"] = CR.sha_strings([parts["spec"], parts["data"], parts["code"]])
    return parts


def _ckpt_latest(root):
    from . import ckpt_resume as CR
    return CR.find_latest(root)


def _ckpt_save(root, step_done: int, fp: dict, model, opt, sched, loop: dict, keep: int) -> None:
    from . import ckpt_resume as CR
    state = {"params": {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
             "buffers": {n: b.detach().cpu().clone() for n, b in model.named_buffers()},
             "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": _rng_state(), "loop": loop}
    dst = CR.save_step(root, step_done, state, {"fingerprint": fp["all"], "parts": fp}, keep)   # tmp+fsync+replace, LATEST
    del state
    CR.release_memory()                              # A64b: no CPU copy of the state lingers between saves
    print(f"    step_ckpt: saved {dst} (keep {keep})", flush=True)


def _ckpt_load(d, model, opt, sched) -> dict:
    state = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
    params = dict(model.named_parameters())
    bufs = dict(model.named_buffers())
    with torch.no_grad():
        for n, v in state["params"].items():
            params[n].copy_(v.to(params[n].device, params[n].dtype))
        for n, v in state["buffers"].items():
            if n in bufs:
                bufs[n].copy_(v.to(bufs[n].device, bufs[n].dtype))
    opt.load_state_dict(state["opt"])
    sched.load_state_dict(state["sched"])
    state["loop"]["_rng"] = state["rng"]
    loop = state["loop"]
    del state
    from . import ckpt_resume as CR
    CR.release_memory()                              # A64b: drop the CPU copy of the loaded state
    return loop



def _perm_kl(logits: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
    """KL(sg softmax(ref) || softmax(logits)) over each row's own options (canonical order, mask = real options), row mean.
    rows (B,) 0/1 weights: 0 = a canonical_order_sources row (its order is meaning, not noise), counted as 0 in the mean."""
    lp = torch.log_softmax(logits.float(), -1).masked_fill(~mask, 0.0)
    lq = torch.log_softmax(ref.detach().float(), -1).masked_fill(~mask, 0.0)
    kl = (lq.exp() * (lq - lp)).masked_fill(~mask, 0.0).sum(-1)
    return (kl * rows if rows is not None else kl).mean()


def _lw_ce_logits(model, tokenizer, sub, encoded, enc, max_options, device, extra, dev_type, bf16, seed, step, canon_pref):
    """lw_ce: the same rows with the presented option order cyclically rotated by k in 1..K-1 (keyed rng), scored WITH grad;
    canonical-order logits (B, max_options) and a (B,) bool of the rows that count (K > 1, not a canonical-order source)."""
    perms, keep = [], []
    for i, ((c, q), e) in enumerate(zip(sub, encoded)):
        perm = list(e["option_perm"]); n = len(perm)
        ok = n > 1 and not (canon_pref and c.source.startswith(canon_pref))
        k = 1 + random.Random(seed * 1000003 + step * 1009 + i).randrange(n - 1) if ok else 0
        perms.append(perm[k:] + perm[:k]); keep.append(ok)
    enc2 = [encode_question(tokenizer, c.state, q, enc, order=o) for (c, q), o in zip(sub, perms)]
    b2 = collate(tokenizer, enc2, max_options=max_options, device=device)
    with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=bf16):
        l2 = model(**b2, **extra).float()
    l2 = unpermute_logits(l2, b2["option_perm"], b2["option_mask"])
    return l2.masked_fill(~b2["option_mask"], float("-inf")), torch.tensor(keep, device=l2.device)


def _perm_kl_ref(model, tokenizer, sub, encoded, enc, max_options, device, extra, dev_type, bf16):
    """proxy P2 b: the same rows with each row's presented option order reversed, scored without grad; canonical-order logits."""
    enc2 = [encode_question(tokenizer, c.state, q, enc, order=list(e["option_perm"])[::-1]) for (c, q), e in zip(sub, encoded)]
    b2 = collate(tokenizer, enc2, max_options=max_options, device=device)
    with torch.no_grad(), torch.autocast(dev_type, dtype=torch.bfloat16, enabled=bf16):
        l2 = model(**b2, **extra).float()
    l2 = unpermute_logits(l2, b2["option_perm"], b2["option_mask"])
    return l2.masked_fill(~b2["option_mask"], float("-inf"))


def _ism_gold_text(c, q):
    g = c.gold[q.key]
    gi = max(range(len(g)), key=g.__getitem__)
    crit = q.criteria or {}
    return crit.get(q.options[gi], q.options[gi]), float(g[gi])


def _ism_index(pairs, vis):
    """ISM strict partner index: key (source, state, question, sorted option texts) -> {gold text: [case_id, ...]}.
    Image questions only; soft-gold rows (max gold < .5) are neither swapped nor used as partners."""
    groups, keys = {}, {}
    for i, (c, q) in enumerate(pairs):
        if not (vis.index.get(c.case_id, (None, []))[1]):
            continue
        gt, gp = _ism_gold_text(c, q)
        if gp < 0.5:
            continue
        crit = q.criteria or {}
        k = (c.source, c.state, q.instructions, tuple(sorted(crit.get(o, o) for o in q.options)))
        keys[i] = k
        groups.setdefault(k, {}).setdefault(gt, []).append(c.case_id)
    return groups, keys


def _ism_candidates(groups, k, pair):
    gt, _ = _ism_gold_text(*pair)
    return [cid for t, ids in sorted(groups.get(k, {}).items()) if t != gt for cid in ids]


def _rtb_index(pairs, cases, vis, cfg):
    """RTB twin index: pair index -> (twin case_id, g, g'). Two-option image questions with a hard gold only.
    Rendered twins must exist as cases with the same question text and options and the other option as gold."""
    by_id = {c.case_id: c for c in cases}
    rev = tuple(cfg.rtb_reverse_sources or ())
    out, why = {}, {"no_twin": 0, "twin_missing": 0, "twin_mismatch": 0, "rendered": 0, "reversed": 0}
    for i, (c, q) in enumerate(pairs):
        ims = vis.index.get(c.case_id, (None, []))[1]
        if not ims or len(q.options) != 2:
            continue
        g = c.gold[q.key]
        gi = max(range(2), key=g.__getitem__)
        if float(g[gi]) < 0.5:
            continue
        tid = vis.twin.get(c.case_id)
        if tid is not None:
            t = by_id.get(tid)
            if t is None or tid not in vis.index:
                why["twin_missing"] += 1
                continue
            tq = next((x for x in t.questions if x.key == q.key), None)
            tg = t.gold.get(q.key) if tq is not None else None
            if (tq is None or tq.instructions != q.instructions or list(tq.options) != list(q.options) or t.state != c.state
                    or max(range(2), key=tg.__getitem__) != 1 - gi):
                why["twin_mismatch"] += 1
                continue
            out[i] = (tid, gi, 1 - gi); why["rendered"] += 1
        elif rev and c.source.startswith(rev) and len(ims) == 2:
            out[i] = (vis.add_reversed(c.case_id), gi, 1 - gi); why["reversed"] += 1
        else:
            why["no_twin"] += 1
    return out, why


def _ddp_info(cfg: "FitConfig"):
    if not cfg.ddp:
        return 1, 0
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()):
        raise ValueError("fit ddp=True needs torch.distributed initialised (torchrun + init_process_group)")
    return dist.get_world_size(), dist.get_rank()


def _ddp_allreduce_grads(params: list, world: int) -> None:
    """Mean of every trainable grad over ranks. A param with no grad on any rank stays None (AdamW skips it, as on
    one GPU); one that has a grad on some rank gets zeros elsewhere first, so every rank reduces the same list."""
    import torch.distributed as dist
    if world == 1:
        return
    dev = next((p.device for p in params), torch.device("cuda"))
    has = torch.tensor([p.grad is not None for p in params], dtype=torch.int32, device=dev)
    dist.all_reduce(has)
    live = []
    for p, h in zip(params, has.tolist()):
        if h:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            live.append(p.grad)
    cap = 256 * 2 ** 20                                  # bytes per bucket
    bucket, size = [], 0
    def flush(b):
        if not b:
            return
        flat = torch._utils._flatten_dense_tensors(b)
        dist.all_reduce(flat)
        flat.div_(world)
        for g, s in zip(b, torch._utils._unflatten_dense_tensors(flat, b)):
            g.copy_(s)
    for g in live:
        if bucket and (g.dtype != bucket[0].dtype or size + g.numel() * g.element_size() > cap):
            flush(bucket); bucket, size = [], 0
        bucket.append(g); size += g.numel() * g.element_size()
    flush(bucket)


def _ddp_gather_rows(x: torch.Tensor, fill: float) -> torch.Tensor:
    """All-gather a (rows, k) tensor over ranks WITH autograd (pads k to the max with `fill`). The backward sums
    every rank's gradient into the local slice; since every rank computes the same loss on the gathered tensor,
    the later grad mean over ranks makes the update equal to the one-GPU gradient."""
    import torch.distributed as dist
    import torch.distributed.nn.functional as dnf
    k = torch.tensor([x.shape[-1]], device=x.device)
    dist.all_reduce(k, op=dist.ReduceOp.MAX)
    k = int(k)
    if x.shape[-1] < k:
        x = F.pad(x, (0, k - x.shape[-1]), value=fill)
    return torch.cat(dnf.all_gather(x.contiguous()), dim=0)


def fit(model, tokenizer, cases: Sequence[Case], enc: EncodeConfig, cfg: FitConfig, *,
        max_options: int, seed: int, device: str = "cuda") -> dict:
    """Train in place. Returns the averaged scorer state plus a small history."""
    g = seed_everything(seed)
    W_dp, R_dp = _ddp_info(cfg)
    if W_dp > 1:
        import torch.distributed as _dist
        if cfg.batch_size % W_dp:
            raise ValueError(f"ddp: batch_size {cfg.batch_size} not divisible by world {W_dp}")
        if cfg.retention_kl and cfg.retention_rows % W_dp:
            raise ValueError(f"ddp: retention_rows {cfg.retention_rows} not divisible by world {W_dp}")
        if cfg.listwise_weight or cfg.ism_weight or cfg.rtb_weight or cfg.tcl_weight:
            raise ValueError("ddp: listwise / ism / rtb / tcl are not supported")
        with torch.no_grad():                        # identical start on every rank
            for t_ in list(model.parameters()) + list(model.buffers()):
                if t_.is_floating_point():
                    _dist.broadcast(t_.data, 0)
        print(f"    ddp: rank {R_dp}/{W_dp}, {cfg.batch_size // W_dp} rows per rank per step", flush=True)
    dev_type0 = "cuda" if str(device).startswith("cuda") else "cpu"
    if cfg.train_max_length:
        import dataclasses as _dc
        enc = _dc.replace(enc, max_length=int(cfg.train_max_length))
        print(f"    train_max_length={cfg.train_max_length}: training rows middle-cut above it", flush=True)
    if cfg.retention_at_exit:
        _retention_ctx(model, cfg)          # validates exit_layer / exit_norm
        n_frz = _freeze_above_exit(model)
        print(f"    retention_at_exit: layers >= {model.cfg.exit_layer} dropped from every pass; "
              f"{n_frz:,} params frozen", flush=True)
    elif cfg.retention_freeze_upper:
        if not getattr(model.cfg, "exit_layer", None):
            raise ValueError("retention_freeze_upper needs arch exit_layer")
        n_frz = _freeze_above_exit(model)
        print(f"    retention_freeze_upper: full-depth retention KL, layers >= {model.cfg.exit_layer} frozen; "
              f"{n_frz:,} params frozen", flush=True)
    ret_rows = ret_tgt = None
    if cfg.retention_kl:
        # BEFORE init_from: the targets are the BASE model's next-token
        # distribution. Computed after the load they would be the parent's, and
        # a decision-tuned parent's LM is destroyed (perplexity 25k at tap 23).
        # No RNG is consumed here, so the data order is unchanged.
        ret_rows = _retention_rows(tokenizer, cases, cfg, seed)
        ret_tgt = _retention_targets(model, tokenizer, ret_rows, cfg, device, dev_type0)
        print(f"    retention_kl={cfg.retention_kl}: {len(ret_rows)} rows, "
              f"{sum(len(r) - 1 for r in ret_rows)} positions, top-{cfg.retention_topk} "
              f"base targets cached (from the base tower{' cut at exit ' + str(model.cfg.exit_layer) if cfg.retention_at_exit else ''}, before init_from)", flush=True)
    upper = _upper_snapshot(model) if (cfg.init_from and cfg.exit_reset_upper) else None
    # A16: continue from a saved release checkpoint. Loaded after seeding and
    # before the optimiser, so the data order and option shuffling are the same
    # as a from-base arm with this seed.
    init = load_init(model, cfg) if cfg.init_from else None
    aux_heads = getattr(model, "aux_scorers", None)
    aux_w = {int(k): float(v) for k, v in (cfg.aux_exit_weights or {}).items()}
    if cfg.aux_weight_total and aux_heads is not None:
        for k in aux_heads:
            aux_w.setdefault(int(k), float(cfg.aux_weight_total) / len(aux_heads))
    if cfg.aux_distill and aux_heads is None:
        raise ValueError("aux_distill needs arch aux_exits")
    if aux_w and aux_heads is None:
        raise ValueError("aux_exit_weights needs arch aux_exits")
    if aux_heads is not None:
        if set(aux_w) - {int(k) for k in aux_heads}:
            raise ValueError(f"aux_exit_weights {sorted(aux_w)} not in aux_exits {sorted(aux_heads)}")
        if cfg.prior_kl or cfg.listwise_weight:
            raise ValueError("aux exits: prior_kl / listwise are not supported")
    if cfg.aux_detach_tower and aux_heads is None:
        raise ValueError("aux_detach_tower needs arch aux_exits")
    if aux_heads is not None:
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
              f"distill {cfg.aux_distill} T={cfg.aux_distill_temp} detach_tower {cfg.aux_detach_tower}", flush=True)
    if upper is not None:
        _upper_restore(model, upper)
        import re
        init["exit_reset_upper"] = sorted({m.group(1) for k in upper
                                           for m in [re.search(r"(?:^|\.)layers\.(\d+)\.", k)] if m},
                                          key=int)
    ref_model = None
    if cfg.ref_kl:
        import copy
        ref_model = copy.deepcopy(model).eval()
        for p_ in ref_model.parameters():
            p_.requires_grad_(False)
        print(f"    ref_kl={cfg.ref_kl}: frozen copy of the loaded init scores every batch (KL(REF || model), main exit)", flush=True)
    # Its own stream, so turning option shuffling on does not change the data
    # ORDER an arm sees and break the pairing with its control.
    osd = seed if cfg.order_seed is None else int(cfg.order_seed)
    order_rng = random.Random(osd + 104729)
    pairs = list(iter_questions(cases))
    if cfg.train_drop_file:
        import json as _json
        _dj = _json.load(open(cfg.train_drop_file))
        _dk = {(cid, k) for cid, k in _dj["drop"]}
        if len(_dk) != int(_dj["n_drop"]) or cfg.group_case_questions:
            raise ValueError(f"train_drop_file: {len(_dk)} unique pairs vs n_drop {_dj['n_drop']} (group_case_questions unsupported)")
        _n0 = len(pairs)
        pairs = [p for p in pairs if (p[0].case_id, p[1].key) not in _dk]
        if _n0 - len(pairs) != len(_dk):
            raise ValueError(f"train_drop_file: removed {_n0 - len(pairs)} of {len(_dk)} listed pairs (file does not match the corpus)")
        print(f"    train_drop_file cap {_dj['cap']}: dropped {len(_dk):,} of {_n0:,} training questions -> {len(pairs):,}", flush=True)
    order = list(range(len(pairs)))
    random.Random(osd).shuffle(order)          # data order is part of the CRN
    if cfg.group_case_questions:
        spans, k = [], 0
        for c in cases:
            spans.append(list(range(k, k + len(c.questions))))
            k += len(c.questions)
        corder = list(range(len(spans)))
        random.Random(osd).shuffle(corder)
        order = [i for ci in corder for i in spans[ci]]

    # ladder-v1: adam_fused=True uses the fused CUDA AdamW kernel (no foreach temporaries: the
    # multi-tensor path allocates a sqrt(v) copy the size of every param group, ~10 GB on a 32-layer 4B tower).
    # Same update rule; off (default) = unchanged behaviour.
    _fused = bool(cfg.adam_fused) and str(device).startswith("cuda")
    opt = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay, **({"fused": True} if _fused else {}))
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
    lambdas = [base_fn if gr["name"].startswith("base") else head_fn if gr["name"] in ("head", "aux")
               else warm for gr in opt.param_groups]
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

    ret_order_rng = random.Random(seed + 31415)
    ret_perm, ret_cur = [], 0
    scale_map = dict(cfg.source_tower_scale or {})

    lw_groups, lw_cur = [], 0
    if cfg.listwise_weight:
        by = {}
        for c in cases:
            if c.source.startswith(cfg.listwise_source) and len(c.questions) == 1 and c.questions[0].mode == "noul":
                by.setdefault(c.case_id.split("#")[0], []).append(c)
        lw_rng = random.Random(seed + 271828)
        for k in sorted(by):
            cs = by[k]
            pos = [c for c in cs if c.gold[c.questions[0].key][1] > 0.5]
            neg = [c for c in cs if c.gold[c.questions[0].key][1] <= 0.5]
            if not pos or not neg:
                continue
            lw_rng.shuffle(neg)
            npos = min(len(pos), max(1, cfg.listwise_group // 2))
            lw_groups.append(pos[:npos] + neg[:cfg.listwise_group - npos])
        lw_rng.shuffle(lw_groups)
        print(f"    listwise: {len(lw_groups)} groups from {cfg.listwise_source!r}, weight {cfg.listwise_weight}", flush=True)
        if not lw_groups:
            raise ValueError("listwise_weight > 0 but no candidate groups found")

    n_acc = max(1, int(cfg.grad_accum))
    if cfg.batch_size % n_acc:
        raise ValueError(f"grad_accum={n_acc} must divide batch_size={cfg.batch_size}")
    n_short = max(1, int(cfg.grad_accum_short or 1))
    if cfg.batch_size % n_short:
        raise ValueError(f"grad_accum_short={n_short} must divide batch_size={cfg.batch_size}")
    if W_dp > 1 and n_short > 1 and cfg.grad_accum_min_tokens and n_short % W_dp:
        raise ValueError(f"ddp: grad_accum_short {n_short} needs n_short % world == 0 (world {W_dp})")
    n_cut = 0                      # training rows whose state was cut to max_length
    n_split = 0                    # steps that ran as grad_accum micro-batches

    vis = None
    if cfg.vision:
        from .vision_fit import VisionFeed
        vis = VisionFeed(cfg.vision, tokenizer, model, device)
    ism_groups, ism_key = {}, {}
    ism_tot = {"n": 0, "act": 0, "hinge": 0.0}           # cumulative since (re)start; logged in history
    if cfg.ism_weight:
        if vis is None:
            raise ValueError("ism_weight needs fit_extra.vision")
        ism_groups, ism_key = _ism_index(pairs, vis)
        n_ok = sum(1 for i, k in ism_key.items()
                   if sum(map(len, ism_groups[k].values())) > len(ism_groups[k].get(_ism_gold_text(*pairs[i])[0], ())))
        print(f"    ism: weight {cfg.ism_weight} margin {cfg.ism_margin}; {len(ism_key):,} image questions, "
              f"{n_ok:,} with a strict swap partner (same source/state/question/options, other gold)", flush=True)
        if not n_ok:
            raise ValueError("ism_weight > 0 but no image question has a strict swap partner")
    rtb_map, rtb_tot = {}, {"n": 0, "act": 0, "hinge": 0.0}
    if cfg.rtb_weight:
        if vis is None:
            raise ValueError("rtb_weight needs fit_extra.vision")
        rtb_map, why = _rtb_index(pairs, cases, vis, cfg)
        print(f"    rtb: weight {cfg.rtb_weight} margin {cfg.rtb_margin} reverse {list(cfg.rtb_reverse_sources)}; "
              f"{len(rtb_map):,} questions with a twin {why}", flush=True)
        if not rtb_map or why["twin_mismatch"] or why["twin_missing"]:
            raise ValueError(f"rtb: twin index invalid {why}")
    tcl_cap, tcl_tot = {"on": False, "hs": []}, {"rows": 0, "anch": 0, "loss": 0.0, "hit": 0.0, "skip": 0}
    if cfg.tcl_weight:
        if vis is None:
            raise ValueError("tcl_weight needs fit_extra.vision")
        n_tcl = sum(1 for c, _ in pairs if vis.anch.get(c.case_id))
        print(f"    tcl: weight {cfg.tcl_weight} layer {cfg.tcl_layer} temp {cfg.tcl_temp} center {cfg.tcl_center}; "
              f"{n_tcl:,} questions with anchors (of {len(pairs):,})", flush=True)
        if not n_tcl:
            raise ValueError("tcl_weight > 0 but no training question has meta.anch")
        if aux_heads is None:
            raise ValueError("tcl_weight needs arch aux_exits (forward_exits is the pass that returns all states)")
        _orig_run_tower = model._run_tower

        def _tcl_run_tower(*a_, **k_):
            r_ = _orig_run_tower(*a_, **k_)
            if tcl_cap["on"] and k_.get("all_states"):
                tcl_cap["hs"].append(r_[0][int(cfg.tcl_layer)])     # conv_safe_call may split the batch: cat later
            return r_
        model._run_tower = _tcl_run_tower
    model.train()
    history, ckpts, aux_ckpts = [], [], []
    stream = None
    off = int(cfg.data_offset_steps) * cfg.batch_size
    if cfg.length_bucket:
        stream = _bucketed_stream(pairs, order, off + cfg.steps * cfg.batch_size, cfg.batch_size,
                                  cfg.length_bucket, random.Random(seed + 7919))
    cursor = off
    if off:
        print(f"    data_offset_steps={cfg.data_offset_steps}: stream starts at question {off:,} "
              f"of {len(pairs):,}", flush=True)
    import time as _time
    from dataclasses import replace as _dc_replace
    canon_pref = tuple(x for x in (cfg.canonical_order_sources or "").split(",") if x)
    enc_canon = _dc_replace(enc, option_order="canonical") if canon_pref else enc
    # ped-jm-e12mxp: perm_kl + aux heads allowed (perm_kl reads the main exit only: logits = exits[exit_layer], ref = model() main exit)
    # ped-jm-e12mxrt-pv: perm_kl allowed with vision: applied per row to text rows only (image rows weight 0, see the loop)
    if cfg.perm_kl and (cfg.prior_kl or cfg.ism_weight
                        or cfg.rtb_weight or cfg.ref_kl or enc.option_order == "canonical"):
        raise ValueError("perm_kl: main exit (aux heads ok), no prior/ref KL, ism/rtb, shuffled/reversed order")
    if cfg.lw_ce and (vis is not None or cfg.prior_kl or aux_heads is not None or cfg.ism_weight
                      or cfg.rtb_weight or cfg.perm_kl or enc.option_order != "shuffled"):
        raise ValueError("lw_ce: text rows, main exit, no prior KL, ism/rtb, no perm_kl, shuffled option order")
    if cfg.lw_ce:
        print(f"    lw_ce={cfg.lw_ce}: + CE on a cyclic rotation of the same item's option order, with grad, main exit", flush=True)
    if cfg.perm_kl:
        print(f"    perm_kl={cfg.perm_kl}: KL(sg p(reversed presentation) || p) per row, main exit, no_grad reference pass", flush=True)
        if vis is not None:
            print("    perm_kl (pv): vision run -> text rows only (image rows skipped), scaled n_text/len(micro-batch)", flush=True)
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
    pr = {"t": _time.perf_counter(), "s": 0}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    nf_max = int(cfg.nonfinite_skip_max)
    nf = {"run": 0, "total": 0, "aux_run": 0, "aux_total": 0}
    if nf_max:
        print(f"    nonfinite guard: skip non-finite steps, fail after {nf_max} consecutive"
              + (" (aux group alone -> drop aux grads only)" if cfg.aux_detach_tower else "")
              + (f"; aux z-loss {cfg.aux_zloss}" if cfg.aux_zloss else ""), flush=True)
    elif cfg.aux_zloss:
        print(f"    aux z-loss {cfg.aux_zloss}", flush=True)
    lw_loss = None
    start_step, ck_root, ck_fp, RS = 0, None, None, None
    if (int(cfg.ckpt_every) > 0 or float(cfg.ckpt_minutes) > 0) and cfg.steps > 0:
        from . import ckpt_resume as CR
        ck_root = _ckpt_root(cfg)
        ck_fp = _ckpt_parts(model, cfg, seed, pairs)
        RS = CR.Resilience(ck_root, ck_fp, every=int(cfg.ckpt_every), minutes=float(cfg.ckpt_minutes),
                           keep=int(cfg.ckpt_keep), rank=R_dp)
        if RS.done() is not None:                                    # raises on a fingerprint mismatch
            print(f"    step_ckpt: {ck_root}/DONE matches: arm already trained, restoring the final state (0 steps)",
                  flush=True)
        d, meta = _ckpt_latest(ck_root)
        if d is not None:
            try:
                CR._compare(meta.get("parts") or {"spec": meta.get("fingerprint")}, ck_fp, f"step ckpt {d}")
            except CR.FingerprintMismatch as e:
                raise ValueError(str(e)) from None
            lp = _ckpt_load(d, model, opt, sched)
            start_step = int(meta["step_done"])
            cursor, ret_perm, ret_cur, lw_cur = lp["cursor"], lp["ret_perm"], lp["ret_cur"], lp["lw_cur"]
            last_ret_kl = lp.get("last_ret_kl")
            cr_n_on, cr_n_gated = int(lp.get("cr_n_on", 0)), int(lp.get("cr_n_gated", 0))
            n_cut, n_split, history, ckpts, aux_ckpts = lp["n_cut"], lp["n_split"], lp["history"], lp["ckpts"], lp["aux_ckpts"]
            order_rng.setstate(lp["order_rng"])
            nf.update(lp.get("nf") or {})
            ret_order_rng.setstate(lp["ret_order_rng"])
            if vis is not None:
                vis.n_image_rows = int(lp.get("vis_image_rows", 0))
            _rng_restore(lp["_rng"])
            pr["s"] = start_step
            print(f"    step_ckpt: RESUMED from {d} at step {start_step}/{cfg.steps} (fingerprint {ck_fp['all']})", flush=True)
        else:
            print(f"    step_ckpt: every {cfg.ckpt_every} steps / {cfg.ckpt_minutes} min -> {ck_root} "
                  f"(fingerprint {ck_fp}); no ckpt to resume", flush=True)
        RS.install_signals()
    dp_params = [p for gr in opt.param_groups for p in gr["params"]] if W_dp > 1 else []
    for step in range(start_step, cfg.steps):
        if stream is not None:
            idx = stream[cursor:cursor + cfg.batch_size]
        else:
            idx = [order[(cursor + i) % len(order)] for i in range(cfg.batch_size)]
        cursor += cfg.batch_size
        chunk = [pairs[i] for i in idx]
        feat["on"] = step % cfg.log_every == 0 or step == cfg.steps - 1
        dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
        # encode the whole step first (same rows, same order, same option shuffles as per-micro encoding)
        # ladder-v1: image rows are encoded by VisionFeed (same rng stream); text rows exactly as before
        if cfg.ism_weight or cfg.rtb_weight:
            # same calls in the same order as vis.encode; the per-row rng state lets the swap forward reuse the row's
            # option presentation without touching order_rng
            enc_all, ism_st = [], []
            for c, q in chunk:
                ism_st.append(order_rng.getstate())
                enc_all.append(vis._encode(c, q, enc_canon if (canon_pref and c.source.startswith(canon_pref)) else enc,
                                           order_rng))
        else:
            if canon_pref:      # same calls in the same order as vis.encode / the list below, per-row encoder
                enc_all = [(vis._encode if vis is not None else
                            (lambda c_, q_, e_, r_: encode_question(tokenizer, c_.state, q_, e_, rng=r_)))(
                               c, q, enc_canon if c.source.startswith(canon_pref) else enc, order_rng)
                           for c, q in chunk]
            else:
                enc_all = (vis.encode(chunk, enc, order_rng) if vis is not None else
                           [encode_question(tokenizer, c.state, q, enc, rng=order_rng) for c, q in chunk])
        cr_step = cr_on and step >= cr_start
        cr_gated = bool(cr_step and cr_gate > 0 and last_ret_kl is not None and last_ret_kl > cr_gate)
        cr_n_on += int(cr_step)
        cr_n_gated += int(cr_gated)
        cr_acc = {"loss": 0.0, "pairs": 0, "auc_num": 0.0}
        n_cut += sum(1 for e in enc_all if e.get("state_cut"))
        step_len = max(len(e["input_ids"]) for e in enc_all)
        n_acc_step = n_acc if (not cfg.grad_accum_min_tokens or step_len > cfg.grad_accum_min_tokens) else n_short
        n_split += n_acc_step > 1
        mb = cfg.batch_size // n_acc_step
        dp_gather = False
        if W_dp > 1:
            per = cfg.batch_size // W_dp
            if n_acc_step == 1:      # one-pass step spread over ranks: the rank term gathers rows (keyv1)
                mb, dp_gather = per, True
            elif n_acc_step % W_dp:
                raise ValueError(f"ddp: a {n_acc_step}-way split step needs n_acc % world == 0 (world {W_dp})")
            chunk = chunk[R_dp * per:(R_dp + 1) * per]
            enc_all = enc_all[R_dp * per:(R_dp + 1) * per]
        loss_parts, logit_parts = [], []
        aux_loss = {}
        aux_amax = {}
        for j in range(0, len(chunk), mb):
            sub = chunk[j:j + mb]
            encoded = enc_all[j:j + mb]
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
            tcl_cap["on"], tcl_cap["hs"] = bool(cfg.tcl_weight), []
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
            tcl_cap["on"] = False
            logits = logits.float()
            # Back to canonical option order BEFORE the loss: gold indexes
            # q.options, so a permuted presentation compared unmapped would train
            # against the wrong option.
            logits = unpermute_logits(logits, batch["option_perm"], batch["option_mask"])
            # masked options must not receive gradient through the loss
            logits = logits.masked_fill(~batch["option_mask"], float("-inf"))
            ref_l = None
            if ref_model is not None:
                with torch.no_grad(), torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cfg.autocast_bf16):
                    ref_l = (ref_model.forward_exits(**batch, **extra)[int(model.cfg.exit_layer)]
                             if aux_heads is not None else ref_model(**batch, **extra))
                ref_l = unpermute_logits(ref_l.float(), batch["option_perm"], batch["option_mask"])
                ref_l = ref_l.masked_fill(~batch["option_mask"], float("-inf"))
            loss = objective_loss(cfg.objective, logits.float(), gold,
                                  progress=step / max(1, cfg.steps - 1), rl=cfg.rl,
                                  generator=None)
            if cfg.prior_kl:
                loss = loss + cfg.prior_kl * prior_kl(logits.float(), base)
            if cfg.perm_kl and vis is not None:
                # pv: text rows only (vis._encode == encode_question for them, so the reversed reference pass matches);
                # image rows are skipped; subset mean x n_text/len(sub), like the aux terms. The key is always logged
                # (0 on an image-only micro-batch) so every DDP rank all-reduces the same aux keys.
                ti_ = [i for i, (c, _) in enumerate(sub) if not vis.index.get(c.case_id, (None, []))[1]]
                if ti_:
                    it_ = torch.tensor(ti_, device=logits.device)
                    ex_t = {k: (v[it_.to(v.device)] if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == len(sub) else v)
                            for k, v in extra.items()}
                    sub_t = [sub[i] for i in ti_]
                    pk_ = _perm_kl(logits[it_], _perm_kl_ref(model, tokenizer, sub_t, [encoded[i] for i in ti_], enc, max_options,
                                                             device, ex_t, dev_type, cfg.autocast_bf16),
                                   batch["option_mask"][it_],
                                   torch.tensor([0.0 if c.source.startswith(canon_pref) else 1.0 for c, _ in sub_t],
                                                device=logits.device) if canon_pref else None) * (len(ti_) / len(sub))
                    loss = loss + float(cfg.perm_kl) * pk_
                    aux_loss["perm_kl"] = aux_loss.get("perm_kl", 0.0) + pk_.detach() * (len(sub) / len(chunk))
                else:
                    aux_loss["perm_kl"] = aux_loss.get("perm_kl", 0.0) + 0.0
            elif cfg.perm_kl:
                pk_ = _perm_kl(logits, _perm_kl_ref(model, tokenizer, sub, encoded, enc, max_options, device, extra, dev_type,
                                                    cfg.autocast_bf16), batch["option_mask"],
                              torch.tensor([0.0 if c.source.startswith(canon_pref) else 1.0 for c, _ in sub], device=logits.device)
                              if canon_pref else None)
                loss = loss + float(cfg.perm_kl) * pk_
                aux_loss["perm_kl"] = aux_loss.get("perm_kl", 0.0) + pk_.detach() * (len(sub) / len(chunk))
            if cfg.lw_ce:
                l2_, k2_ = _lw_ce_logits(model, tokenizer, sub, encoded, enc, max_options, device, extra, dev_type, cfg.autocast_bf16,
                                         seed, step, canon_pref)
                if bool(k2_.any()):
                    lw_ = objective_loss(cfg.objective, l2_[k2_].float(), gold[k2_], progress=step / max(1, cfg.steps - 1), rl=cfg.rl,
                                         generator=None) * (float(k2_.sum()) / len(sub))
                    loss = loss + float(cfg.lw_ce) * lw_
                    aux_loss["lw_ce"] = aux_loss.get("lw_ce", 0.0) + lw_.detach() * (len(sub) / len(chunk))
            if ref_l is not None:
                rk_ = prior_kl(logits.float(), ref_l)
                loss = loss + cfg.ref_kl * rk_
                aux_loss["ref_kl"] = aux_loss.get("ref_kl", 0.0) + rk_.detach() * (len(sub) / len(chunk))
            if cr_step and not cr_gated:
                if dp_gather:   # same pairs as the one-GPU one-pass micro-batch
                    rk, auc_, np_ = conf_rank_loss(_ddp_gather_rows(logits, float("-inf")),
                                                   _ddp_gather_rows(gold.detach(), 0.0).argmax(-1),
                                                   margin_temp=float(cfg.conf_rank_margin_temp))
                else:
                    rk, auc_, np_ = conf_rank_loss(logits, gold.argmax(-1), margin_temp=float(cfg.conf_rank_margin_temp))
                loss = loss + float(cfg.conf_rank_coef) * rk
                cr_acc["loss"] += float(rk.detach()) * (len(sub) / len(chunk))
                if np_:
                    cr_acc["pairs"] += np_
                    cr_acc["auc_num"] += auc_ * np_
            if exits is not None:
                om = batch["option_mask"]
                T_ = float(cfg.aux_distill_temp)
                if cfg.aux_distill:
                    t_lp = torch.log_softmax(logits.detach().float() / T_, dim=-1)
                for L_, w_ in aux_w.items():
                    if not w_:
                        continue
                    al = unpermute_logits(exits[L_].float(), batch["option_perm"], om)
                    al = al.masked_fill(~om, float("-inf"))
                    ce_ = objective_loss(cfg.objective, al, gold,
                                         progress=step / max(1, cfg.steps - 1), rl=cfg.rl,
                                         generator=None)
                    term = ce_
                    if cfg.aux_distill:
                        s_lp = torch.log_softmax(al / T_, dim=-1).masked_fill(~om, 0.0)
                        tl = t_lp.masked_fill(~om, 0.0)
                        kl_ = (tl.exp() * om * (tl - s_lp)).sum(-1).mean()
                        term = term + cfg.aux_distill * (T_ ** 2) * kl_
                        aux_loss[f"aux{L_}_kl"] = aux_loss.get(f"aux{L_}_kl", 0.0) + kl_.detach() * (len(sub) / len(chunk))
                    if cfg.aux_zloss:
                        lse = torch.logsumexp(al, dim=-1)                       # masked options are -inf
                        zl_ = (lse ** 2).mean()
                        term = term + float(cfg.aux_zloss) * zl_
                        aux_loss[f"aux{L_}_z"] = aux_loss.get(f"aux{L_}_z", 0.0) + zl_.detach() * (len(sub) / len(chunk))
                    if nf_max or feat.get("on"):
                        fa = al.detach()[torch.isfinite(al.detach())]
                        aux_amax[L_] = max(aux_amax.get(L_, 0.0), float(fa.abs().max()) if fa.numel() else float("nan"))
                    aux_loss[f"aux{L_}_loss"] = aux_loss.get(f"aux{L_}_loss", 0.0) + ce_.detach() * (len(sub) / len(chunk))
                    loss = loss + w_ * term
            if cfg.tcl_weight:
                hl = torch.cat(tcl_cap["hs"], 0) if len(tcl_cap["hs"]) > 1 else tcl_cap["hs"][0]
                tcl_cap["hs"] = []
                if hl.shape[0] != len(sub):
                    raise RuntimeError(f"tcl: captured {hl.shape[0]} rows for a micro-batch of {len(sub)}")
                G_, terms = 14, []
                for r_, (c0, _) in enumerate(sub):
                    an = vis.anch.get(c0.case_id)
                    if not an:
                        continue
                    pos = (batch["input_ids"][r_] == vis.pad_id).nonzero(as_tuple=True)[0]
                    ok = len(pos) == 2 * G_ * G_
                    if ok:
                        p1, p2 = pos[:G_ * G_], pos[G_ * G_:]
                        for pp in (p1, p2):     # row-major 14 x 14 from the LM's own M-RoPE (h, w) ids (as tcl_probe.py)
                            hh, ww = batch["position_ids"][1, r_, pp], batch["position_ids"][2, r_, pp]
                            ok = ok and torch.equal((hh - hh.min()) * G_ + (ww - ww.min()),
                                                    torch.arange(G_ * G_, device=hh.device))
                    if not ok:
                        tcl_tot["skip"] += 1
                        continue
                    x1, x2 = hl[r_, p1].float(), hl[r_, p2].float()
                    if cfg.tcl_center:
                        x1, x2 = x1 - x1.mean(0, keepdim=True), x2 - x2.mean(0, keepdim=True)
                    x1, x2 = torch.nn.functional.normalize(x1, dim=-1), torch.nn.functional.normalize(x2, dim=-1)
                    at = torch.tensor(an, device=hl.device)[:, :4].long()
                    ai, gt_ = at[:, 0] * G_ + at[:, 1], at[:, 2] * G_ + at[:, 3]
                    sim = x1[ai] @ x2.T / float(cfg.tcl_temp)
                    terms.append(torch.nn.functional.cross_entropy(sim, gt_))
                    tcl_tot["rows"] += 1; tcl_tot["anch"] += len(ai)
                    tcl_tot["loss"] += float(terms[-1].detach()); tcl_tot["hit"] += float((sim.detach().argmax(-1) == gt_).float().sum())
                if terms:
                    tl_ = torch.stack(terms)
                    loss = loss + float(cfg.tcl_weight) * tl_.sum() / len(sub)
                    for k__, v__ in (("tcl_n", float(len(terms))), ("tcl_loss_sum", tl_.detach().sum())):
                        aux_loss[k__] = aux_loss.get(k__, 0.0) + v__
            if cfg.ism_weight:
                sw_rows, sw_ex = [], []
                for r_ in range(len(sub)):
                    gi_ = idx[j + r_]
                    k_ = ism_key.get(gi_)
                    cand = _ism_candidates(ism_groups, k_, pairs[gi_]) if k_ is not None else []
                    if not cand:
                        continue
                    pc = random.Random(f"ism:{seed}:{step}:{j + r_}").choice(cand)
                    c0, q0 = sub[r_]
                    rr = random.Random(); rr.setstate(ism_st[j + r_])
                    try:
                        sw_ex.append(vis._encode(dataclasses.replace(c0, case_id=pc), q0, enc, rr))
                    except ValueError:              # partner image does not fit max_length at any budget: no swap
                        continue
                    sw_rows.append(r_)
                if sw_rows:
                    n_img0 = vis.n_image_rows
                    sb = vis.collate(sw_ex, max_options)
                    vis.n_image_rows = n_img0       # image_rows counts the main stream only (same as the ctl)
                    with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cfg.autocast_bf16):
                        if aux_heads is not None:
                            sl = model.forward_exits(**sb, **({"aux_detach": True} if cfg.aux_detach_tower else {}))
                            sl = sl[int(model.cfg.exit_layer)]
                        else:
                            sl = model(**sb)
                    sl = unpermute_logits(sl.float(), sb["option_perm"], sb["option_mask"])
                    sl = sl.masked_fill(~sb["option_mask"], float("-inf"))
                    ri = torch.tensor(sw_rows, device=logits.device)
                    gsel = gold[ri].argmax(-1)
                    lp_i = torch.log_softmax(logits[ri], -1).gather(1, gsel[:, None]).squeeze(1)
                    lp_s = torch.log_softmax(sl, -1).gather(1, gsel[:, None]).squeeze(1)
                    marg = lp_i - lp_s
                    hinge = torch.relu(float(cfg.ism_margin) - marg)
                    loss = loss + float(cfg.ism_weight) * hinge.sum() / len(sub)
                    # per-step SUMS over swapped rows (divide by ism_n): rows swapped, hinge-active rows, hinge, margin
                    for k__, v__ in (("ism_n", float(len(sw_rows))), ("ism_act_n", (hinge.detach() > 0).float().sum()),
                                     ("ism_hinge_sum", hinge.detach().sum()), ("ism_margin_sum", marg.detach().sum())):
                        aux_loss[k__] = aux_loss.get(k__, 0.0) + v__
                    ism_tot["n"] += len(sw_rows); ism_tot["act"] += int((hinge.detach() > 0).sum())
                    ism_tot["hinge"] += float(hinge.detach().sum())
            if cfg.rtb_weight:
                tw_rows, tw_ex, tw_g = [], [], []
                for r_ in range(len(sub)):
                    t_ = rtb_map.get(idx[j + r_])
                    if t_ is None:
                        continue
                    c0, q0 = sub[r_]
                    rr = random.Random(); rr.setstate(ism_st[j + r_])
                    try:
                        tw_ex.append(vis._encode(dataclasses.replace(c0, case_id=t_[0]), q0, enc, rr))
                    except ValueError:              # twin image does not fit max_length at any budget: no term
                        continue
                    tw_rows.append(r_); tw_g.append(t_[1:])
                if tw_rows:
                    n_img0 = vis.n_image_rows
                    tb = vis.collate(tw_ex, max_options)
                    vis.n_image_rows = n_img0       # image_rows counts the main stream only (same as the ctl)
                    with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cfg.autocast_bf16):
                        if aux_heads is not None:
                            tl = model.forward_exits(**tb, **({"aux_detach": True} if cfg.aux_detach_tower else {}))
                            tl = tl[int(model.cfg.exit_layer)]
                        else:
                            tl = model(**tb)
                    tl = unpermute_logits(tl.float(), tb["option_perm"], tb["option_mask"])
                    ri = torch.tensor(tw_rows, device=logits.device)
                    gg = torch.tensor([a for a, _ in tw_g], device=logits.device)
                    gp = torch.tensor([b for _, b in tw_g], device=logits.device)
                    li = logits[ri]
                    s_i = li.gather(1, gg[:, None]).squeeze(1) - li.gather(1, gp[:, None]).squeeze(1)
                    s_t = tl.gather(1, gp[:, None]).squeeze(1) - tl.gather(1, gg[:, None]).squeeze(1)
                    marg = s_i + s_t
                    hinge = torch.relu(float(cfg.rtb_margin) - marg)
                    loss = loss + float(cfg.rtb_weight) * hinge.sum() / len(sub)
                    for k__, v__ in (("rtb_n", float(len(tw_rows))), ("rtb_act_n", (hinge.detach() > 0).float().sum()),
                                     ("rtb_hinge_sum", hinge.detach().sum()), ("rtb_margin_sum", marg.detach().sum())):
                        aux_loss[k__] = aux_loss.get(k__, 0.0) + v__
                    rtb_tot["n"] += len(tw_rows); rtb_tot["act"] += int((hinge.detach() > 0).sum())
                    rtb_tot["hinge"] += float(hinge.detach().sum())
            if n_acc_step > 1:
                # every term of the task loss is a mean over rows: weight by row share
                loss = loss * (len(sub) / len(chunk))
            if j + mb >= len(chunk):
                # step-level terms, once per step, on the last micro-batch
                lw_loss = None
                if lw_groups and step % max(1, cfg.listwise_every) == 0:
                    grp = lw_groups[lw_cur % len(lw_groups)]
                    lw_cur += 1
                    lb = collate(tokenizer, [encode_question(tokenizer, c.state, c.questions[0], enc, rng=order_rng) for c in grp],
                                 max_options=max_options, device=device)
                    with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=cfg.autocast_bf16):
                        ll = model(**lb)
                    ll = unpermute_logits(ll.float(), lb["option_perm"], lb["option_mask"])
                    ti = [c.questions[0].options.index("true") for c in grp]
                    fi = [c.questions[0].options.index("false") for c in grp]
                    r = torch.arange(len(grp), device=ll.device)
                    score = ll[r, torch.tensor(ti, device=ll.device)] - ll[r, torch.tensor(fi, device=ll.device)]
                    tgt = torch.tensor([c.gold[c.questions[0].key][1] for c in grp], device=ll.device, dtype=torch.float32)
                    tgt = tgt / tgt.sum()
                    lw_loss = -(tgt * torch.log_softmax(score, dim=0)).sum()
                    loss = loss + cfg.listwise_weight * lw_loss
                ret_kl = None
                if cfg.retention_kl:
                    pick = []
                    for _ in range(cfg.retention_rows):
                        if ret_cur >= len(ret_perm):
                            ret_perm = list(range(len(ret_rows)))
                            ret_order_rng.shuffle(ret_perm)
                            ret_cur = 0
                        pick.append(ret_perm[ret_cur]); ret_cur += 1
                    if W_dp > 1:        # every rank drew the same rows; each scores its share (mean over ranks)
                        k_ = cfg.retention_rows // W_dp
                        pick = pick[R_dp * k_:(R_dp + 1) * k_]
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
            if dp_gather and cr_step and not cr_gated:
                import torch.distributed as _dist
                fin_ = torch.tensor([int(mb_finite)], device=loss.device)
                _dist.all_reduce(fin_, op=_dist.ReduceOp.MIN)
                mb_finite = bool(int(fin_))
            if not mb_finite and not nf_max:
                raise FloatingPointError(
                    f"non-finite loss {loss.item()} at step {step}/{cfg.steps}; "
                    f"last logged {history[-3:]}")
            if j == 0:
                # same place as the single-pass loop: after the first forward
                opt.zero_grad(set_to_none=True)
            if mb_finite:            # guard on: a non-finite micro-batch is not backpropagated; the step is skipped
                loss.backward()
            loss_parts.append(loss.detach())
            logit_parts.append(logits.detach())
        loss = loss_parts[0] if n_acc_step == 1 else torch.stack(loss_parts).sum()
        logits = logit_parts[0] if n_acc_step == 1 else torch.cat([l.reshape(-1) for l in logit_parts])
        if W_dp > 1:
            import torch.distributed as _dist
            _ddp_allreduce_grads(dp_params, W_dp)
            sc = torch.tensor([float(loss), float(ret_kl.detach()) if ret_kl is not None else 0.0],
                              device=loss.device, dtype=torch.float64)
            _dist.all_reduce(sc)
            sc /= W_dp
            loss = sc[0].to(loss.dtype)
            if ret_kl is not None:
                ret_kl = sc[1].float()
            if aux_loss:                                 # logged aux / ref_kl terms: global mean (same keys on every rank)
                ak_ = sorted(aux_loss)
                av_ = torch.tensor([float(aux_loss[k]) for k in ak_], device=loss.device, dtype=torch.float64)
                _dist.all_reduce(av_)
                aux_loss = {k: v / W_dp for k, v in zip(ak_, av_.tolist())}
            if cr_on and not dp_gather:
                cv_ = torch.tensor([cr_acc["loss"], cr_acc["pairs"], cr_acc["auc_num"]],
                                   device=loss.device, dtype=torch.float64)
                _dist.all_reduce(cv_)
                cr_acc = {"loss": float(cv_[0]) / W_dp, "pairs": int(round(float(cv_[1]))), "auc_num": float(cv_[2])}
        logging = step % cfg.log_every == 0 or step == cfg.steps - 1
        if logging or nf_max:
            # Pre-clip gradient norm per group and the largest finite logit: the
            # two training-time signals that say WHERE an instability starts,
            # without looking at any evaluation target.
            nf_scaled = set()
            gnorm = _group_grad_norms(opt, loss.device, safe=bool(nf_max), scaled=nf_scaled)
            fin = logits.detach()[torch.isfinite(logits.detach())]
            logit_absmax = float(fin.abs().max()) if fin.numel() else float("nan")
        nf_dec = "ok"
        if nf_max:
            nf_dec = _nonfinite_decision(bool(torch.isfinite(loss.detach()).all()), gnorm, bool(cfg.aux_detach_tower))
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
                      f"run {nf['run']}/{nf['aux_run']} total {nf['total']}/{nf['aux_total']} (max {nf_max})", flush=True)
            if max(nf["run"], nf["aux_run"]) > nf_max:
                raise FloatingPointError(
                    f"non-finite loss/grad on {max(nf['run'], nf['aux_run'])} consecutive steps (> nonfinite_skip_max "
                    f"{nf_max}) at step {step}/{cfg.steps}: {nf_dec}, loss {float(loss.detach())}, gn {gnorm}; "
                    f"last logged {history[-3:]}")
        if nf_dec == "skip":
            pass                                     # no clip, no update: grads are already None
        elif nf_max and cfg.grad_clip:
            sels = ((lambda g_: g_["name"] != "aux", lambda g_: g_["name"] == "aux") if cfg.aux_detach_tower
                    else (lambda g_: True,))
            for sel in sels:
                _clip_groups(opt, sel, cfg.grad_clip, gnorm, nf_scaled)
            if nf_scaled:
                print(f"    nonfinite: step {step + 1}/{cfg.steps} fp32 grad norm overflow in {sorted(nf_scaled)} "
                      f"-> clipped with the safe norm", flush=True)
        elif cfg.grad_clip and cfg.aux_detach_tower:
            # detached aux heads are clipped on their own, so the trunk + main head step is exactly the no-aux step
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
        # a120 OOM fix (method 2026-10-14): peak_gib is the MAX over DDP ranks (all ranks reduce, rank 0 prints). The
        # length-sorted chunks give the last rank the longest rows, so rank 0's own peak understated it (81.2 vs 89.4 GiB).
        peak_all = None
        if (cfg.print_every and W_dp > 1 and torch.cuda.is_available()
                and (step % cfg.print_every == 0 or step == cfg.steps - 1)):
            _pk = torch.tensor([torch.cuda.max_memory_allocated() / 2 ** 30], device=device)
            torch.distributed.all_reduce(_pk, op=torch.distributed.ReduceOp.MAX)
            peak_all = float(_pk)
        if cfg.print_every and R_dp == 0 and (step % cfg.print_every == 0 or step == cfg.steps - 1):
            now = _time.perf_counter()
            sps = (now - pr["t"]) / max(1, step + 1 - pr["s"])
            pr["t"], pr["s"] = now, step + 1
            peak = (peak_all if peak_all is not None else
                    torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0)
            terms = " ".join(f"{k} {float(v):.4f}" for k, v in aux_loss.items())
            if cfg.tcl_weight:
                print(f"    tcl_tot: rows {tcl_tot['rows']} anchors {tcl_tot['anch']} skipped {tcl_tot['skip']} "
                      f"mean_loss {tcl_tot['loss'] / max(1, tcl_tot['rows']):.4f} "
                      f"train_nn_exact {tcl_tot['hit'] / max(1, tcl_tot['anch']):.4f}", flush=True)
            if cfg.rtb_weight:
                print(f"    rtb_tot: twins {rtb_tot['n']} active {rtb_tot['act']} "
                      f"({rtb_tot['act'] / max(1, rtb_tot['n']):.3f}) mean_hinge {rtb_tot['hinge'] / max(1, rtb_tot['n']):.4f}", flush=True)
            if cfg.ism_weight:
                print(f"    ism_tot: swapped {ism_tot['n']} active {ism_tot['act']} "
                      f"({ism_tot['act'] / max(1, ism_tot['n']):.3f}) mean_hinge {ism_tot['hinge'] / max(1, ism_tot['n']):.4f}", flush=True)
            print(f"    step {step + 1}/{cfg.steps} loss {float(loss.detach()):.4f}"
                  + (f" {terms}" if terms else "") + f" step_len {step_len} accum {n_acc_step} s/step {sps:.3f}"
                  + f" eta_h {sps * (cfg.steps - step - 1) / 3600:.2f} peak_gib {peak:.1f}"
                  + (f" nf {nf['total']}/{nf['aux_total']}" if nf_max else "")
                  + (f" cr {cr_rec['cr_loss']:.4f} pairs {cr_rec['cr_pairs']} gated {cr_rec['cr_gated']}" if cr_rec else ""),
                  flush=True)

        if logging:
            history.append({"step": step, "loss": float(loss.detach()), **cr_rec,
                            **({"image_rows": vis.n_image_rows} if vis is not None else {}),
                            **({"ism_tot_n": ism_tot["n"], "ism_tot_act": ism_tot["act"],
                                "ism_tot_hinge": round(ism_tot["hinge"], 3)} if cfg.ism_weight else {}),
                            **({"tcl_tot_rows": tcl_tot["rows"], "tcl_tot_anch": tcl_tot["anch"], "tcl_tot_skip": tcl_tot["skip"],
                                "tcl_tot_loss": round(tcl_tot["loss"], 3), "tcl_tot_hit": round(tcl_tot["hit"], 1)}
                               if cfg.tcl_weight else {}),
                            **({"rtb_tot_n": rtb_tot["n"], "rtb_tot_act": rtb_tot["act"],
                                "rtb_tot_hinge": round(rtb_tot["hinge"], 3)} if cfg.rtb_weight else {}),
                            **({"lw_loss": round(float(lw_loss.detach()), 5)} if lw_loss is not None else {}),
                            **{k: round(float(v), 5) for k, v in aux_loss.items()},
                            "step_len": step_len, "accum": n_acc_step,
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
        ck_now = RS is not None and RS.should_save(step + 1, cfg.steps)    # every rank calls (DDP-synced)
        if ck_now and R_dp == 0:
            _ckpt_save(ck_root, step + 1, ck_fp, model, opt, sched,
                       {"cursor": cursor, "ret_perm": list(ret_perm), "ret_cur": ret_cur, "lw_cur": lw_cur,
                        "last_ret_kl": last_ret_kl, "cr_n_on": cr_n_on, "cr_n_gated": cr_n_gated,
                        "n_cut": n_cut, "n_split": n_split, "history": history, "ckpts": ckpts,
                        "aux_ckpts": aux_ckpts, "order_rng": order_rng.getstate(),
                        "ret_order_rng": ret_order_rng.getstate(), "nf": dict(nf),
                        "vis_image_rows": int(vis.n_image_rows) if vis is not None else 0}, cfg.ckpt_keep)
        if W_dp > 1 and ck_now:
            import torch.distributed as _dist
            _dist.barrier()
        if ck_now:
            RS.exit_if_signalled()                   # a signal save: exit 99 / 143 (A64)
        _step_hook(step)

    if RS is not None:
        RS.mark_done(cfg.steps)                      # A64: a re-launch of this arm trains nothing
        RS.close()
    hook.remove()
    if W_dp > 1:
        import torch.distributed as _dist
        _dist.barrier()
        if R_dp != 0:                                # the caller's eval / save runs once, on rank 0
            import sys as _sys
            _sys.stdout.flush()
            _dist.destroy_process_group()
            _os._exit(0)
    # m4b7: free grads + optimizer state before the in-run eval (4B at 32k: the fp32 SDPA path materialises
    # 16 x 32k x 32k scores = 64 GiB per row, which OOMed with the optimizer still resident)
    opt.zero_grad(set_to_none=True)
    del sched, opt
    import gc as _gc; _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"    fit: freed optimizer; allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB before eval", flush=True)
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
    return {"history": history, "averaged_over": len(ckpts), "init": init,
            "examples_seen": cfg.steps * cfg.batch_size, "seed": seed,
            "grad_accum": n_acc, "state_cut_rows": n_cut,
            "grad_accum_min_tokens": cfg.grad_accum_min_tokens, "split_steps": n_split,
            "conf_rank": ({"steps_on": cr_n_on, "steps_gated": cr_n_gated,
                           "gated_frac": round(cr_n_gated / max(1, cr_n_on), 5)} if cr_on else None)}
