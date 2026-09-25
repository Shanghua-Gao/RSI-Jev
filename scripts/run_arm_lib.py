"""One experiment end to end, against state that is already loaded.

This is the shared body: `release_train.py` calls it to cut a release, and
the internal search loop calls the same function for every candidate, which
is what makes a released checkpoint the same code path as an experiment.

Everything expensive -- weights, tokenizer, corpus, both evaluation targets --
is passed in already loaded, so an arm costs only its own compute. The control
is measured in the SAME call as the candidate, on the same weights, and both
rows are returned together.
"""
from __future__ import annotations

import math
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.arch import ArchConfig, DecisionModel, LogprobReadout, hidden_state_layer  # noqa: E402
from rsijev.contract import Prediction                                 # noqa: E402
from rsijev.encode import EncodeConfig, collate, encode_question, iter_questions  # noqa: E402
from rsijev.evaluate import as_record, eval_precision, predict, score_predictions  # noqa: E402
from rsijev.fit import FitConfig, fit                                  # noqa: E402
from rsijev.train import RLConfig, assert_r0_gate, seed_everything     # noqa: E402

def text_hidden_size(lm) -> int:
    """`AutoConfig` returns the multimodal wrapper (which has `.text_config`);
    `AutoModelForCausalLM` returns the text model, whose config IS the text
    config and has no such attribute. Handle both rather than guessing."""
    cfg = lm.config
    inner = getattr(cfg, "text_config", None)
    if inner is None and hasattr(cfg, "get_text_config"):
        inner = cfg.get_text_config()
    return (inner or cfg).hidden_size


DEFAULTS = dict(readout="option_xattn", objective="soft_ce", steps=1500,
                batch_size=16, eval_batch_size=32, max_options=80,
                lr_head=1e-3, lr_base=5e-6, diag_only=False, logit_cap=None, head_input_norm=False, label_smoothing=0.0, head_weight_decay=0.0, base_schedule="cosine", head_schedule="constant", seed=17, readout_layer=-1,
                option_pool="mean", layout="state_first", freeze_base=True,
                sources="", keep_last_k=3, residual=False, prior_kl=0.0,
                freeze_embeddings=True, grad_checkpointing=True,
                eval_dtype="native", repeat_eval=None,
                option_order="canonical", eval_option_orders=None,
                # Passthrough for fields an arm adds to ArchConfig / FitConfig in
                # its own arch.py / fit.py. This file is protected (it holds the
                # evaluation path), so a new knob reaches the model through these
                # dicts instead of through an edit here.
                arch_extra={}, fit_extra={},
                # RLConfig fields (group, sigma_start, sigma_end, reward). The SAME
                # config drives both the R0 gate and training, so a gate pass is a
                # statement about the RL setup that actually trains.
                rl_extra={},
                # Floor on trainable tower parameters when freeze_base=False. It
                # catches a copy that silently kept requires_grad=False. An arm
                # that deliberately trains less (adapters, frozen lower layers)
                # lowers it explicitly; the tower-moved check still applies.
                min_trainable_tower=100_000_000,
                # Write a release checkpoint here after training, before evaluation,
                # so the checkpoint is exactly the model the records score.
                save_dir=None)


# "bfloat16" means the model's NATIVE precision, not a cast: Qwen3.5 loads its
# parameters in bf16 but keeps rotary inv_freq buffers in fp32, and casting the
# whole module to bf16 would compute positions from bf16 frequencies -- a numeric
# path no training step and no earlier result ever used. None = no cast at all.
DTYPES = {"native": None, "float32": torch.float32}
# "bfloat16" is accepted as an alias for "native" and recorded as "native": the
# model is never uniformly bf16 -- its rotary buffers stay fp32 in both legs --
# so a record must not claim that it was.
ALIASES = {"bfloat16": "native"}


def zero_shot(lm, tok, cases, name, enc, *, device, batch_size, max_options, eval_dtype=None):
    with eval_precision(lm, eval_dtype):
        return _zero_shot(lm, tok, cases, name, enc, device=device,
                          batch_size=batch_size, max_options=max_options)


@torch.no_grad()
def _zero_shot(lm, tok, cases, name, enc, *, device, batch_size, max_options):
    ro = LogprobReadout(lm)
    pairs = list(iter_questions(cases))
    preds = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        batch = collate(tok, [encode_question(tok, c.state, q, enc) for c, q in chunk],
                        max_options=max_options, device=device)
        ids = torch.zeros((len(chunk), max_options), dtype=torch.long, device=device)
        for r, (_, q) in enumerate(chunk):
            t = LogprobReadout.option_token_ids(tok, q.options)
            ids[r, :len(t)] = torch.tensor(t, device=device)
        probs = torch.softmax(ro(input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"],
                                 decision_index=batch["decision_index"],
                                 option_token_ids=ids,
                                 option_mask=batch["option_mask"]).float(), dim=-1)
        for r, (c, q) in enumerate(chunk):
            preds.append((c, q, Prediction(tuple(probs[r, :len(q.options)].tolist()))))
    return score_predictions(preds, target=name, split="test"), preds


def item_rows(preds, *, target, role, eval_dtype, option_order="canonical"):
    """Per-question record: enough to find near-ties and to re-score offline.
    `top2_logit_gap` is log(p1/p2) = z1 - z2, recoverable exactly from the
    probabilities, and is the quantity a one-ulp shift competes with."""
    import math
    from rsijev.contract import gold_label
    out = []
    for c, q, p in preds:
        pr = list(p.probs)
        top = sorted(range(len(pr)), key=pr.__getitem__, reverse=True)
        gap = (math.log(max(pr[top[0]], 1e-30)) - math.log(max(pr[top[1]], 1e-30))
               if len(top) > 1 else float("inf"))
        out.append({"target": target, "role": role, "eval_dtype": eval_dtype,
                    "option_order": option_order,
                    "case_id": c.case_id, "key": q.key, "mode": q.mode,
                    "probs": [round(x, 7) for x in pr],
                    "pred": q.options[top[0]], "gold": gold_label(q, c.gold[q.key]),
                    "top2_logit_gap": gap})
    return out


# Probability units. Calibrated on a stub with a real final norm: a correct base
# term reads exactly 0; a wrong lm_head reads 0.97; a correction head that was NOT
# zero-initialised reads 9.7e-3. An earlier 1e-2 let that last case through. The
# two paths run the same computation, so only bf16 kernel-path noise separates
# them on the real weights, which is far below 1e-3.
RESIDUAL_TOL = 1e-3


@torch.no_grad()
def residual_init_gap(model, lm, tok, cases, enc, *, device, max_options, n=128, bs=16) -> float:
    """Max |p_residual_at_init - p_zero_shot| over n real questions."""
    ro = LogprobReadout(lm)
    pairs = list(iter_questions(cases))[:n]
    worst = 0.0
    model.eval()
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        batch = collate(tok, [encode_question(tok, c.state, q, enc) for c, q in chunk],
                        max_options=max_options, device=device)
        p_res = torch.softmax(model(**batch).float(), dim=-1)
        p_ctl = torch.softmax(ro(input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"],
                                 decision_index=batch["decision_index"],
                                 option_token_ids=batch["option_token_ids"],
                                 option_mask=batch["option_mask"]).float(), dim=-1)
        mask = batch["option_mask"]
        worst = max(worst, float((p_res - p_ctl).abs().masked_fill(~mask, 0).max()))
    model.train()
    return worst


# Which gated-delta-net implementation the tower runs: the fla Triton kernels
# or transformers' reference PyTorch fallback. They differ numerically, so
# records from the two must never be mixed within one comparison.
try:
    import fla  # noqa: F401
    LINEAR_ATTN_KERNEL = "fla-" + getattr(fla, "__version__", "?")
except Exception:   # ImportError, or Triton finding no GPU driver
    LINEAR_ATTN_KERNEL = "torch-reference"
import torch as _torch
LINEAR_ATTN_KERNEL += f"/torch-{_torch.__version__}"


def save_release(model, lm, cfg: dict, out: Path, *, tapped, train_seconds, history,
                 n_train_cases) -> None:
    """A loadable checkpoint: tuned tower weights (fp32, the precision they were
    evaluated in) minus the embedding, which is frozen and identical to the base
    model's; the trained scorer; and meta.json with everything needed to rebuild
    the model on top of the public base weights (scripts/load_release.py)."""
    import json as _json
    from safetensors.torch import save_file
    from rsijev.evaluate import CODE_VERSION
    out.mkdir(parents=True, exist_ok=True)
    tower = {k: v.detach().float().contiguous().cpu()
             for k, v in model.tower.state_dict().items() if "embed_tokens" not in k}
    save_file(tower, str(out / "tower.safetensors"))
    save_file({k: v.detach().float().contiguous().cpu()
               for k, v in model.scorer.state_dict().items()}, str(out / "scorer.safetensors"))
    meta = {
        "base_model": getattr(lm.config, "_name_or_path", None),
        "spec": {k: v for k, v in cfg.items() if k != "save_dir"},
        "tapped_layer": tapped[0], "tapped_layer_type": tapped[1],
        "linear_attn_kernel": LINEAR_ATTN_KERNEL, "code_version": CODE_VERSION,
        "train_seconds": train_seconds, "n_train_cases": n_train_cases,
        "final_loss": history[-1]["loss"] if history else None,
        "tower_keys": len(tower), "excluded": ["embed_tokens (frozen; taken from base_model)"],
    }
    (out / "meta.json").write_text(_json.dumps(meta, indent=2) + "\n")
    print(f"    saved release checkpoint -> {out} ({len(tower)} tower tensors)", flush=True)


def run_arm(*, lm, tok, targets, corpus, device, spec, name, items_path=None):
    cfg = {**DEFAULTS, **spec}
    # Training presents options in `option_order`; evaluation may score the SAME
    # trained head under several presentations. The diagnostic that matters: a
    # head trained canonical and scored reversed tells position from content,
    # read in canonical space after unpermute_logits.
    enc = EncodeConfig(layout=cfg["layout"], option_pool=cfg["option_pool"],
                       option_order=cfg["option_order"])
    eval_orders = list(cfg["eval_option_orders"] or ["canonical"])
    encs = {o: EncodeConfig(layout=cfg["layout"], option_pool=cfg["option_pool"],
                            option_order=o) for o in eval_orders}

    # An RL objective may not be read before it reproduces a known truth.
    rl_cfg = RLConfig(**dict(cfg["rl_extra"] or {}))
    if str(cfg["objective"]).startswith("rlcd"):
        assert_r0_gate(cfg["objective"], rl=rl_cfg)

    wanted = set(cfg["sources"].split(",")) if cfg["sources"] else set(corpus)
    pool = [c for k, v in corpus.items() if k in wanted for c in v]
    # In-distribution holdout: 1 case in 10, chosen by a hash of its id so every
    # arm and every seed holds out the SAME cases. Without it an arm that loses
    # to the zero-shot control is ambiguous -- a broken pipeline and a head that
    # fits its training sources but does not transfer look identical on the two
    # out-of-distribution targets. This separates them.
    import hashlib
    def held(c):
        return int(hashlib.sha256(c.case_id.encode()).hexdigest(), 16) % 10 == 0
    train_cases = [c for c in pool if not held(c)]
    holdout = [c for c in pool if held(c)]
    # diag_only: a training-stability diagnostic scores the in-distribution
    # holdout alone, so no test target can steer a fix chosen from it.
    targets = {"in_distribution": holdout, **({} if cfg["diag_only"] else targets)}

    eval_dtypes = [ALIASES.get(d, d) for d in (cfg["repeat_eval"] or [cfg["eval_dtype"]])]
    for d in eval_dtypes:
        if d not in DTYPES:
            raise ValueError(f"eval dtype {d!r} not in {sorted(DTYPES)}")
    items: list[dict] = []
    records = []
    for d, o in [(d, o) for d in eval_dtypes for o in eval_orders]:
        for tname, cases in targets.items():
            rep, preds = zero_shot(lm, tok, cases, tname, encs[o], device=device,
                                   batch_size=cfg["eval_batch_size"],
                                   max_options=cfg["max_options"], eval_dtype=DTYPES[d])
            items += item_rows(preds, target=tname, role="control", eval_dtype=d, option_order=o)
            # the FULL resolved config on control rows too: a control scored at a
            # non-default batch size or precision must be visible in the log.
            records.append(as_record(rep, arm=name, role="control",
                                     control_kind="zero_shot_logprob", trainable=0,
                                     seed=cfg["seed"], eval_dtype=d, option_order=o,
                                     **{"spec." + k: v for k, v in cfg.items()}))


    arch = ArchConfig(readout=cfg["readout"], readout_layer=cfg["readout_layer"],
                      max_options=cfg["max_options"], freeze_base=cfg["freeze_base"],
                      option_pool=cfg["option_pool"], residual=cfg["residual"],
                      logit_cap=cfg["logit_cap"], head_input_norm=cfg["head_input_norm"],
                      **dict(cfg["arch_extra"] or {}))
    hidden = text_hidden_size(lm)
    # Record what the readout index actually taps. Index k is the output of
    # layer k-1, and on this 3:1 hybrid an off-by-one lands on the other
    # attention type: the first sweep's "layer 15" was layer 14, linear.
    _tc = getattr(lm.config, "text_config", lm.config)
    _lt = list(getattr(_tc, "layer_types", []))
    if isinstance(cfg["readout_layer"], dict):
        # per-mode taps: record each mode's real layer, never a single number
        _t = {m: hidden_state_layer(int(i), _lt) for m, i in cfg["readout_layer"].items()}
        tapped = ({m: v[0] for m, v in _t.items()}, {m: v[1] for m, v in _t.items()})
    else:
        tapped = hidden_state_layer(cfg["readout_layer"], _lt)
    # Seed BEFORE the head is built. fit() seeds only after construction, so the
    # head's initial weights used to come from whatever RNG state the worker
    # process was in -- in practice the PREVIOUS arm's seed, or a fresh process's
    # default. Two "seed 17" arms matched only when both were preceded by a seed-17
    # arm; the same spec as a process's first arm did not. The head init now
    # depends on this arm's seed and nothing else.
    seed_everything(cfg["seed"])
    base_tower = getattr(lm, "model", lm)
    if cfg["freeze_base"]:
        tower = base_tower
    else:
        # A tower-training arm must never touch the worker's resident model: every
        # later arm on this worker, and every zero-shot control, reads it. So it
        # trains its own copy -- in fp32, because Adam steps of ~lr=2e-5 on bf16
        # weights of magnitude ~0.02 are below bf16's resolution and round to zero.
        tower = copy.deepcopy(base_tower).to(torch.float32)
        # The copy inherits requires_grad=False: the zero-shot control's
        # LogprobReadout froze every parameter of the resident model before this
        # arm was built. Without this line a "tuned" arm trains only its head.
        for p_ in tower.parameters():
            p_.requires_grad_(True)
        if cfg["grad_checkpointing"]:
            # use_reentrant=False: with the embedding frozen, the reentrant variant
            # sees inputs that do not require grad and returns NO gradients to the
            # layers -- the arm would train its head alone and report otherwise.
            tower.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    model = DecisionModel(tower, hidden, arch,
                          # the base distribution is needed by residual arms AND by
                          # prior-anchored arms; a from-scratch arm with prior_kl
                          # would otherwise raise inside the first training step.
                          lm_head=lm.lm_head if (cfg["residual"] or cfg["prior_kl"]) else None
                          ).to(device)
    model.scorer.to(torch.float32)
    if not cfg["freeze_base"] and cfg["freeze_embeddings"]:
        # The embedding is tied to lm_head (254 M of 752 M). A readout head never
        # generates, so it is kept frozen by default; set freeze_embeddings=false
        # to tune it too.
        for p_ in tower.get_input_embeddings().parameters():
            p_.requires_grad_(False)
    if not cfg["freeze_base"]:
        n_tower = sum(p_.numel() for p_ in tower.parameters() if p_.requires_grad)
        if n_tower < cfg["min_trainable_tower"]:
            raise RuntimeError(f"freeze_base=False but only {n_tower:,} tower parameters are "
                               f"trainable (floor {cfg['min_trainable_tower']:,}); the arm would "
                               "silently train its head alone. An arm that trains fewer on "
                               "purpose sets min_trainable_tower explicitly.")
    from rsijev.evaluate import _bitwise_checksum
    resident_before = _bitwise_checksum(base_tower) if not cfg["freeze_base"] else None

    # A residual arm claims that at step 0 it IS the zero-shot control. Check that
    # on the real weights before training, not on a stub: a stub has no final norm
    # and cannot distinguish a base term taken after the norm from one taken before
    # it, or from the wrong layer.
    init_diff = None
    if cfg["residual"]:
        # canonical presentation: the check compares two readouts of the same
        # rows, and a shuffled training order would only add a permutation to both
        init_diff = residual_init_gap(model, lm, tok, holdout,
                                      EncodeConfig(layout=cfg["layout"], option_pool=cfg["option_pool"]),
                                      device=device, max_options=cfg["max_options"])
        print(f"    residual init vs zero-shot control: max |dp| = {init_diff:.2e} "
              f"({'OK' if init_diff <= RESIDUAL_TOL else 'INVALID'})", flush=True)

    probe = None
    if not cfg["freeze_base"]:
        probe = {n: p_.detach().clone() for n, p_ in list(tower.named_parameters())
                 if p_.requires_grad}
        # The FIRST trainable tensors: every readout tap depends on them. The last
        # ones do not -- a tap at readout 15 reads layer 14, so layers 15-23 and
        # the final norm get zero gradient by construction, and probing them
        # raised a false alarm on a tower that had in fact trained.
        probe = dict(list(probe.items())[:4])
    t0 = time.perf_counter()
    hist = fit(model, tok, train_cases, enc,
               FitConfig(objective=cfg["objective"], steps=cfg["steps"],
                         batch_size=cfg["batch_size"], lr_head=cfg["lr_head"],
                         lr_base=cfg["lr_base"], base_schedule=cfg["base_schedule"],
                         head_schedule=cfg["head_schedule"],
                         label_smoothing=cfg["label_smoothing"],
                         head_weight_decay=cfg["head_weight_decay"],
                         keep_last_k=cfg["keep_last_k"],
                         prior_kl=cfg["prior_kl"], rl=rl_cfg,
                         autocast_bf16=not cfg["freeze_base"],
                         **dict(cfg["fit_extra"] or {})),
               max_options=cfg["max_options"], seed=cfg["seed"], device=device)
    train_s = time.perf_counter() - t0
    if probe is not None:
        moved = {n: float((dict(tower.named_parameters())[n].detach() - v).abs().max())
                 for n, v in probe.items()}
        if not all(math.isfinite(v) for v in moved.values()):
            raise RuntimeError(f"tower weights went non-finite during training ({moved})")
        if max(moved.values()) == 0.0:
            raise RuntimeError(f"freeze_base=False but the tower did not move during training "
                               f"({moved}); gradients never reached it")
        print(f"    tower moved: max |dw| over probe tensors = {max(moved.values()):.3e}", flush=True)
    if resident_before is not None and _bitwise_checksum(base_tower) != resident_before:
        raise RuntimeError("a tower-training arm modified the worker's resident model; "
                           "every later arm and control on this worker would be contaminated")
    print(f"    trained {len(train_cases)} cases in {train_s:.0f}s  "
          f"loss {[round(h['loss'], 4) for h in hist['history']]}", flush=True)
    for h in hist["history"]:
        print("    " + " ".join(f"{k}={v}" for k, v in h.items()), flush=True)

    if cfg["save_dir"]:
        save_release(model, lm, cfg, Path(cfg["save_dir"]), tapped=tapped,
                     train_seconds=round(train_s), history=hist["history"],
                     n_train_cases=len(train_cases))

    for d, o, (tname, cases) in [(d, o, t) for d in eval_dtypes for o in eval_orders
                                 for t in targets.items()]:
        preds = predict(model, tok, cases, encs[o], max_options=cfg["max_options"],
                        device=device, batch_size=cfg["eval_batch_size"],
                        eval_dtype=DTYPES[d])
        items += item_rows(preds, target=tname, role="candidate", eval_dtype=d, option_order=o)
        rep = score_predictions(preds, target=tname, split="test")
        rec = as_record(rep, arm=name, role="candidate", eval_dtype=d, option_order=o,
                        trainable=model.trainable_parameters(),
                        train_seconds=round(train_s), n_train_cases=len(train_cases),
                        final_loss=hist["history"][-1]["loss"], history=hist["history"],
                        seed=cfg["seed"],
                        residual_init_max_abs_prob_diff=init_diff,
                        tapped_layer=tapped[0], tapped_layer_type=tapped[1],
                        linear_attn_kernel=LINEAR_ATTN_KERNEL,
                        valid=(init_diff is None or init_diff <= RESIDUAL_TOL),
                        **{"spec." + k: v for k, v in cfg.items()})
        records.append(rec)
        ctrl = next(r for r in records if r["role"] == "control" and r["target"] == tname
                    and r["eval_dtype"] == d and r["option_order"] == o)
        print(f"    [{d}/{o}] {tname:16s} minDS {rec['min_decision_score']:+7.2f} "
              f"(ctrl {ctrl['min_decision_score']:+7.2f})  "
              f"AURC {rec['pooled_aurc']:.4f} (ctrl {ctrl['pooled_aurc']:.4f})", flush=True)
        for m in ("choice", "noul", "score"):
            if f"{m}.accuracy" in rec:
                print(f"       {m:7s} acc {rec[f'{m}.accuracy']:.4f} "
                      f"(ctrl {ctrl[f'{m}.accuracy']:.4f})  DS {rec[f'{m}.decision_score']:+7.2f} "
                      f"(ctrl {ctrl[f'{m}.decision_score']:+7.2f})", flush=True)
    if items_path is not None:
        import json as _json
        with open(items_path, "w") as fh:
            for row in items:
                fh.write(_json.dumps(row) + "\n")
    return records
