"""EDITABLE — the model axis: how state and options are encoded and read out.

The base is `Qwen/Qwen3.5-0.8B-Base`, which is NOT a plain decoder:

  * multimodal (`Qwen3_5ForConditionalGeneration`): a text tower plus a vision
    tower. The vision tower is frozen and unused here.
  * the text tower repeats `linear_attention x3 + full_attention`, so **only
    every 4th layer can attend freely**. Where a readout is attached is a real
    choice, not a detail.
  * vocabulary 248,320, tied embedding ~254 M parameters -- about 40% of the
    text tower, against ~264 M of MLP across all 24 layers. A decision model
    never generates text, so that matrix is the largest single block of weight
    in the model that the task does not obviously need.

Every mechanism here is switchable so that arms differ in exactly one thing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import contextlib
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .contract import MODES

def hidden_state_layer(index: int, layer_types: "list[str]") -> tuple[int, str]:
    """What a `readout_layer` index actually taps: (layer number, its type).

    `output_hidden_states` yields (embeddings, out_of_layer0, ...,
    out_of_layer_{L-2}, normed_final), so index k is the output of layer k-1.
    Reading `layer_types[k]` instead of `layer_types[k-1]` is off by one, and on
    a 3:1 hybrid that lands on the wrong ATTENTION TYPE, not merely a
    neighbouring depth.
    """
    n = len(layer_types)
    k = n if index in (-1, n) else index
    if k == 0:
        return -1, "embeddings"
    if k >= n:
        return n - 1, layer_types[n - 1] + " (final, normed)"
    return k - 1, layer_types[k - 1]


def _norm_layer(index: int, n_states: int) -> int:
    """Normalise a possibly-negative hidden-state index."""
    return index if index >= 0 else n_states + index


Readout = Literal["option_marker", "pointer", "nli_template", "option_xattn"]
Embedding = Literal["tied", "frozen", "sliced", "replaced"]


@dataclass
class ArchConfig:
    readout: Readout = "option_marker"
    # Which entry of the hidden-states tuple the readout reads. Negative counts
    # from the end. See `hidden_state_layer` for what an index actually means:
    # the tuple is (embeddings, out_of_layer0, ..., out_of_layer_{L-2}, final),
    # so index k is the output of layer k-1 and is OFF BY ONE from layer_types.
    # For Qwen3.5-0.8B the full-attention layers are layer_types 3/7/11/15/19/23,
    # whose outputs live at hidden-state indices 4/8/12/16/20/-1 -- NOT at 3/7/
    # 11/15/19/23, which are all linear-attention outputs.
    #
    # May be a dict {mode: index}, because the layer profiles differ BY MODE:
    # on typed-decisions choice is a band peaking mid-stack and falling off a
    # cliff deeper, score is a single-layer spike, noul is flat, and MMLU-Pro
    # prefers shallow. One tap cannot be optimal for all three. The tower runs
    # once regardless, so per-mode taps are free -- only the indexing changes.
    readout_layer: int | dict = -1
    # Cross-attention readout only.
    xattn_heads: int = 4
    xattn_dim: int | None = None          # defaults to the tower's hidden size
    # How option_xattn combines each option's value v_k with the attended context
    # ctx (arch-fix-1). "sum" (default, every arm so far): proj(v_k + ctx). proj is
    # linear, so proj(ctx) is the SAME constant for all K options and cancels in
    # the softmax: the cross-attention and the decision query contribute nothing.
    #   "mlp":      MLP([v_k, v_k*ctx, v_k-ctx]) -> 1   (variant a)
    #   "bilinear": proj(v_k) + <W v_k, ctx>, W zero-initialised, so the arm starts
    #               exactly at the "sum" model (variant b)
    # Checkpoint layout: q/k/v/attn/proj keep their names; "mlp" adds comb.*,
    # "bilinear" adds bil.weight.
    #   "bilinear_norm": as "bilinear" on RMS-normalised v_k and ctx, divided by d.
    #               Unscaled, one Adam step moves all d^2 entries of W by ~lr and the
    #               term by ~lr*|v|_1*|ctx|_1 (CPU 0.8B check: loss 1.04 -> 17.4 at
    #               lr_head 1e-3); normalised, a worst-case step is ~lr*d.
    xattn_combine: Literal["sum", "mlp", "bilinear", "bilinear_norm"] = "sum"
    xattn_mlp_hidden: int = 512
    # arch-2 readout designs, layered ON TOP of the base option scorer above (so
    # they compose with any xattn_combine). Each starts at the base scorer: the
    # correction's output layer or gate is zero-initialised, or (per_mode) the
    # three heads are copies of one initialisation. Still one tower pass.
    #   "none":     the base scorer alone (every arm so far)
    #   "joint":    + a small transformer over [decision; options] (no position
    #               encoding, so option-order equivariant); per-option linear out
    #               (zero-init). Options attend to each other before the logit.
    #   "ordinal":  + for SCORE rows only, a cumulative-link term: location
    #               s in [0, K-1] and dispersion tau from the decision state,
    #               P(y<=k) = sigmoid((k + 0.5 - s) / tau); added as gate*log P
    #               (gate init 0). Levels are canonical option indices (score
    #               options are "0".."K-1"), read through option_perm.
    #   "per_mode": three copies of the base scorer, one per mode (choice/noul/
    #               score), selected by mode_id.
    # Checkpoint layout (scorer.safetensors): base keys move under "base." for
    # joint/ordinal and "heads.<m>." for per_mode; joint adds jin/jtype/jenc/jout,
    # ordinal adds oloc/odisp/ogate.
    head_design: Literal["none", "joint", "ordinal", "per_mode"] = "none"
    joint_dim: int = 512
    joint_layers: int = 2
    joint_heads: int = 8
    # Encode the state once and answer K questions from it, questions blind to
    # each other. This is how the reference service is described as behaving.
    pack_questions: bool = False
    embedding: Embedding = "tied"
    option_pool: Literal["mean", "last"] = "mean"
    # Start the trained readout AT the zero-shot control instead of at noise.
    # With `residual`, the logit is the base model's own option logprob plus a
    # learned correction whose output layer is zero-initialised, so at step 0 the
    # trained arm and the control are the same function. Any drop below the
    # control is then overfitting you can see, not a cold start being paid for.
    residual: bool = False
    # Soft-cap the scorer's logits: cap * tanh(logits / cap). Near the identity
    # for |logit| << cap, bounded by construction beyond it. With a tuned tower
    # the head inflated the logit scale without bound (max 113 by step 150, 4.7e6
    # by the end); one seed of the stable recipe still reached 1304. None = off.
    logit_cap: float | None = None
    # Instead of choosing ONE layer, learn a per-mode mixture over several.
    # The cross-target disagreement (MMLU-Pro prefers shallow, typed-decisions
    # prefers mid-stack) cannot be resolved by picking a tap, because at
    # inference nothing says which distribution a request came from. A learned
    # mixture does not have to know: it can weight shallow and deep features
    # from the data. It subsumes the fixed tap, which is a one-hot mixture, and
    # it is initialised AS one so an arm starts at its own baseline rather than
    # at noise -- the same discipline as the residual readout.
    layer_mix: tuple | None = None          # candidate hidden-state indices
    layer_mix_init: int = -1                # the tap it starts one-hot on
    layer_mix_temp: float = 10.0            # logit scale for the one-hot start
    # A one-hot start is exact but SATURATED: at temp 10 over 8 candidates the
    # weight is 0.9997 and d(weight)/d(logit) is about 3e-4, roughly 3000x
    # smaller than at uniform, so the mixture starts at its baseline and then
    # cannot move. That is the residual trap in a new place. The fix is not a
    # softer start, which would give up the exact baseline, but a much larger
    # step size on these few parameters -- see FitConfig.lr_mix.
    freeze_base: bool = True              # train the readout only, by default
    # Parameter-free RMS normalisation of the head's inputs (decision and option
    # features), in the spirit of QK-layernorm. With a tuned 2B tower the max
    # logit grew 2 -> 205 while the MEAN feature norm fell and the head weight
    # norm moved 0.1%: the growth is concentrated in a few rows or directions,
    # which a per-row normalisation removes from the head's view.
    head_input_norm: bool = False
    # banking77 needs 77 slots. The pointer readout allocates hidden x max_options
    # whatever the question actually uses, so raising this grows ONLY that arm --
    # another reason readout arms must report their trainable parameter counts.
    max_options: int = 80
    # mmlu-keep-1 (c): freeze the lower fraction of the tower's decoder layers
    # (embeddings are frozen separately by the runner). 0 = train every layer,
    # which is the champion's behaviour. 1/3 at 2B freezes layers 0-7 of 24.
    freeze_lower_frac: float = 0.0
    # arch2: TREE execution of the tower (rsijev/tree.py; needs layout "tree").
    #   "none"           the conventional causal pass over the flat row
    #   "indep"          D1 POD: each option a branch from the question; options
    #                    never see each other inside the tower
    #   "dec_sees_opts"  D3 marker: as indep, but the decision cue sees every
    #                    sibling option in the full-attention layers
    #   "bidir"          D2 prefix-LM: the question block (question, cue, all
    #                    options) is bidirectional in full attention
    #   "late"           D4 late interaction: options encoded ALONE (roots) and
    #                    read at hidden index late_option_layer; the scorer
    #                    cross-attends them to the state/question memory
    #   "chain"          test topology: identical to the conventional pass
    # Order invariance comes from the topology (positions restart per branch,
    # linear-attention branches start from the parent's state) plus a
    # permutation-equivariant head (option_xattn; head_design "joint" adds a
    # set-transformer comparator with no position ids).
    tree_rule: str = "none"
    late_option_layer: int = 8
    late_dim: int = 256
    late_heads: int = 8
    # arch: EARLY EXIT. The scorer reads hidden-state index exit_layer
    # (the un-normalised output of decoder layer exit_layer-1, what
    # readout_layer=exit_layer would tap) and decoder layers exit_layer.. are NOT
    # RUN in the scoring pass: the text model runs its first exit_layer layers
    # (then its final norm, or the identity without exit_norm). The weights stay in the module,
    # so checkpoints keep the full layout, and a direct `tower(...)` call (fit.py's
    # LM-retention term) still runs full depth. Parity with the full tower read at
    # readout_layer=exit_layer: tests/test_early_exit.py. On the 24-layer 2B,
    # 16 = output of layer 15 and 12 = output of layer 11 (both full attention).
    # Requires readout_layer=-1, residual=False, no layer_mix. None = off.
    exit_layer: int | None = None
    # With exit_norm (default) the exit state goes through the text model's own
    # final RMSNorm, i.e. the model IS a truncated Qwen3.5 (LayerSkip-style shared
    # norm). Mid-stack states are ~13x smaller than the normed final (2B base:
    # token norm 11.7 at index 16 vs 153.5 final), so a head continued from a
    # tap-final parent sees inputs at the scale it was trained on. False = the
    # raw tap, exactly readout_layer=exit_layer on the full tower.
    exit_norm: bool = True
    # adaexit: ADAPTIVE early exit. Extra readout heads at shallower exits
    # (hidden-state indices < exit_layer, same convention as exit_layer), each its
    # own OptionScorer reading the text model's final norm applied to that state
    # (exit_norm semantics). The main scorer still reads exit_layer, and forward()
    # returns ONLY its logits, so with the aux heads ignored the model is exactly the
    # fixed-exit model. forward_exits() returns every exit's logits (deep-supervision
    # training, per-exit eval); rsijev/adaptive.py runs the staged per-row exit.
    # Checkpoint layout: the aux heads go to aux_scorers.safetensors, keyed "<L>.<name>".
    # Needs exit_layer, exit_norm, head_design "none", tree_rule "none". () = off.
    aux_exits: tuple = ()
    # ladder-v1: decoder layers 0..freeze_lower_n-1 get requires_grad=False at model build (no grads,
    # no optimiser state). Lets a 32-layer 4B tower + 4 heads train on one 96 GB RTX. 0 = off (unchanged).
    freeze_lower_n: int = 0


def _layer_period(tm) -> int:
    types = list(getattr(tm.config, "layer_types", None) or [])
    for p in range(1, len(types) + 1):
        if all(types[i] == types[i % p] for i in range(len(types))):
            return p
    return 1


class _RowGradScale(torch.autograd.Function):
    """Identity forward; multiplies the gradient of row b by scale[b] backward.

    mmlu-keep-1 (b): applied to the hidden states the scorer reads, it scales a
    row's gradient into the TOWER only; the scorer still sees the full loss.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        (scale,) = ctx.saved_tensors
        return g * scale.to(g.dtype).view(-1, *([1] * (g.dim() - 1))), None


def _decoder_layers(tower: nn.Module):
    """The text tower's decoder-layer ModuleList (Qwen3.5: `.layers`, or
    `.language_model.layers` on the multimodal wrapper)."""
    for path in ("layers", "language_model.layers", "model.layers"):
        m = tower
        try:
            for part in path.split("."):
                m = getattr(m, part)
        except AttributeError:
            continue
        if isinstance(m, nn.ModuleList):
            return m
    raise ValueError("could not locate the tower's decoder layers")


def _text_model(tower: nn.Module) -> nn.Module:
    """The module that owns the decoder `layers` and the final `norm`
    (Qwen3.5: the tower itself, or `.language_model` on the multimodal wrapper)."""
    for path in ("", "language_model", "model"):
        m = tower
        try:
            for part in [p for p in path.split(".") if p]:
                m = getattr(m, part)
        except AttributeError:
            continue
        if isinstance(getattr(m, "layers", None), nn.ModuleList) and hasattr(m, "norm"):
            return m
    raise ValueError("could not locate the tower's text model (layers + norm)")

class OptionScorer(nn.Module):
    """Turns hidden states into one logit per option.

    All four readouts produce the same object -- a vector of option logits in the
    question's own option order -- so they are interchangeable arms.
    """

    def __init__(self, hidden: int, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.xattn_dim or hidden
        if cfg.readout in ("option_marker", "nli_template"):
            self.proj = nn.Linear(hidden, 1)
        elif cfg.readout == "pointer":
            self.proj = nn.Linear(hidden, cfg.max_options)
        elif cfg.readout == "option_xattn":
            self.q = nn.Linear(hidden, d)
            self.k = nn.Linear(hidden, d)
            self.v = nn.Linear(hidden, d)
            self.attn = nn.MultiheadAttention(d, cfg.xattn_heads, batch_first=True)
            self.proj = nn.Linear(d, 1)
            if cfg.xattn_combine == "mlp":
                self.comb = nn.Sequential(nn.Linear(3 * d, cfg.xattn_mlp_hidden), nn.GELU(),
                                          nn.Linear(cfg.xattn_mlp_hidden, 1))
            elif cfg.xattn_combine in ("bilinear", "bilinear_norm"):
                self.bil = nn.Linear(d, d, bias=False)
                nn.init.zeros_(self.bil.weight)
            elif cfg.xattn_combine != "sum":
                raise ValueError(f"xattn_combine {cfg.xattn_combine!r}")
        else:
            raise ValueError(cfg.readout)

    def zero_init_output(self) -> None:
        """Make the correction exactly 0 at initialisation."""
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, *, decision_h: torch.Tensor,
                option_h: torch.Tensor | None = None,
                option_mask: torch.Tensor | None = None,
                mode_id: torch.Tensor | None = None,
                option_perm: torch.Tensor | None = None) -> torch.Tensor:
        """
        mode_id, option_perm: accepted for the arch-2 designs (DesignedScorer);
        the base scorer ignores them.
        decision_h : (B, H)          hidden state at the decision position
        option_h   : (B, K, H)       one hidden state per option span, or None
        option_mask: (B, K) bool     True where an option exists
        returns    : (B, K)          logits, -inf where masked
        """
        cfg = self.cfg
        if cfg.head_input_norm:
            def _rms(x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            decision_h = _rms(decision_h)
            if option_h is not None:
                option_h = _rms(option_h)
        if cfg.readout == "pointer":
            logits = self.proj(decision_h)                       # (B, max_options)
            if option_mask is not None:
                logits = logits[:, : option_mask.shape[1]]
        elif cfg.readout in ("option_marker", "nli_template"):
            if option_h is None:
                raise ValueError(f"{cfg.readout} needs option hidden states")
            logits = self.proj(option_h).squeeze(-1)             # (B, K)
        else:  # option_xattn
            if option_h is None:
                raise ValueError("option_xattn needs option hidden states")
            q = self.q(decision_h).unsqueeze(1)                  # (B, 1, D)
            k, v = self.k(option_h), self.v(option_h)            # (B, K, D)
            pad = None if option_mask is None else ~option_mask
            ctx, _ = self.attn(q, k, v, key_padding_mask=pad, need_weights=False)
            # score each option against the attended context, so the logit for an
            # option depends on the whole option SET, not on that option alone.
            if cfg.xattn_combine == "sum":
                logits = self.proj(v + ctx).squeeze(-1)          # (B, K); ctx cancels (see ArchConfig)
            elif cfg.xattn_combine == "mlp":
                c = ctx.expand_as(v)
                logits = self.comb(torch.cat([v, v * c, v - c], dim=-1)).squeeze(-1)
            elif cfg.xattn_combine == "bilinear":
                logits = self.proj(v).squeeze(-1) + (self.bil(v) * ctx).sum(-1)
            else:  # bilinear_norm
                def _rms(x):
                    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
                bil = (self.bil(_rms(v)) * _rms(ctx)).sum(-1) / v.shape[-1]
                logits = self.proj(v).squeeze(-1) + bil
        if option_mask is not None:
            logits = logits.masked_fill(~option_mask, float("-inf"))
        return logits


class DesignedScorer(nn.Module):
    """arch-2 readout designs around a base OptionScorer (see ArchConfig.head_design).

    Same interface as OptionScorer: one logit per option in PRESENTED order,
    -inf where masked. Every design equals the base scorer at initialisation.
    """

    def __init__(self, hidden: int, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        design = cfg.head_design
        if design == "per_mode":
            one = OptionScorer(hidden, cfg)
            self.heads = nn.ModuleList([copy.deepcopy(one) for _ in MODES])
            return
        self.base = OptionScorer(hidden, cfg)
        if design == "joint":
            dj = cfg.joint_dim
            self.jin = nn.Linear(hidden, dj)
            self.jtype = nn.Parameter(torch.zeros(2, dj))          # 0 = decision, 1 = option
            layer = nn.TransformerEncoderLayer(dj, cfg.joint_heads, dim_feedforward=2 * dj,
                                               dropout=0.0, batch_first=True, norm_first=True,
                                               activation="gelu")
            self.jenc = nn.TransformerEncoder(layer, cfg.joint_layers, enable_nested_tensor=False)
            self.jout = nn.Linear(dj, 1)
            nn.init.zeros_(self.jout.weight)
            nn.init.zeros_(self.jout.bias)
        elif design == "ordinal":
            self.oloc = nn.Linear(hidden, 1)
            self.odisp = nn.Linear(hidden, 1)
            self.ogate = nn.Parameter(torch.zeros(()))
        else:
            raise ValueError(f"head_design {design!r}")

    def zero_init_output(self) -> None:
        for m in (self.heads if self.cfg.head_design == "per_mode" else [self.base]):
            m.zero_init_output()

    def forward(self, *, decision_h, option_h=None, option_mask=None,
                mode_id=None, option_perm=None):
        design = self.cfg.head_design
        kw = dict(decision_h=decision_h, option_h=option_h, option_mask=option_mask)
        if design == "per_mode":
            if mode_id is None:
                raise ValueError("per_mode heads need mode_id in the batch")
            all_l = torch.stack([h(**kw) for h in self.heads])     # (M, B, K)
            b = torch.arange(decision_h.shape[0], device=decision_h.device)
            return all_l[mode_id.long(), b]
        logits = self.base(**kw)
        if option_mask is None:
            raise ValueError(f"head_design {design!r} needs option_mask")
        if design == "joint":
            x = torch.cat([self.jin(decision_h).unsqueeze(1) + self.jtype[0],
                           self.jin(option_h) + self.jtype[1]], dim=1)   # (B, 1+K, dj)
            pad = torch.cat([torch.zeros_like(option_mask[:, :1]), ~option_mask], dim=1)
            y = self.jenc(x, src_key_padding_mask=pad)
            corr = self.jout(y[:, 1:]).squeeze(-1)                  # (B, K)
            logits = logits + corr.masked_fill(~option_mask, 0.0)
        else:  # ordinal
            if mode_id is None or option_perm is None:
                raise ValueError("the ordinal head needs mode_id and option_perm in the batch")
            k = option_mask.sum(-1, keepdim=True).float()           # levels per question
            s = torch.sigmoid(self.oloc(decision_h)) * (k - 1)      # (B, 1) location
            tau = F.softplus(self.odisp(decision_h)) + 0.05         # (B, 1) dispersion
            kmax = option_mask.shape[1]
            lev = torch.arange(kmax, device=decision_h.device).float().unsqueeze(0)
            # cdf at the upper edge of each level; the last real level closes at 1
            cdf = torch.sigmoid((lev + 0.5 - s) / tau)
            cdf = torch.where(lev >= k - 1, torch.ones_like(cdf), cdf)
            lower = torch.cat([torch.zeros_like(cdf[:, :1]), cdf[:, :-1]], dim=1)
            logp = torch.log((cdf - lower).clamp_min(1e-9))         # canonical level order
            logp_pres = logp.gather(1, option_perm.long().clamp(0, kmax - 1))
            is_score = (mode_id == MODES.index("score")).unsqueeze(1)
            ok = option_mask & is_score
            logits = logits + torch.where(ok, self.ogate * logp_pres, torch.zeros_like(logits))
        return logits.masked_fill(~option_mask, float("-inf"))


class LateScorer(nn.Module):
    """arch2 D4: late interaction. Options are encoded without the state; each
    option's pooled vector cross-attends the state/question/cue memory (tap
    states), and a ColBERT MaxSim term scores its tokens against that memory.
    The per-option vectors then go through the ordinary (equivariant) scorer.
    All head parameters live here, under `scorer`, so fit/save/average see them."""

    def __init__(self, hidden: int, cfg: ArchConfig):
        super().__init__()
        self.cfg = cfg
        self.inner = (OptionScorer(hidden, cfg) if cfg.head_design == "none"
                      else DesignedScorer(hidden, cfg))
        self.ln_o = nn.LayerNorm(hidden)
        self.ln_m = nn.LayerNorm(hidden)
        self.xattn = nn.MultiheadAttention(hidden, cfg.late_heads, batch_first=True)
        self.po = nn.Linear(hidden, cfg.late_dim)
        self.pm = nn.Linear(hidden, cfg.late_dim)
        self.ms_scale = nn.Parameter(torch.ones(()))

    def zero_init_output(self) -> None:
        self.inner.zero_init_output()

    def forward(self, *, decision_h, option_h=None, option_mask=None, mode_id=None,
                option_perm=None, memory=None, memory_mask=None, opt_tokens=None,
                opt_slot=None):
        ctx, _ = self.xattn(self.ln_o(option_h), self.ln_m(memory), self.ln_m(memory),
                            key_padding_mask=~memory_mask, need_weights=False)
        opt = option_h + ctx
        logits = self.inner(decision_h=decision_h, option_h=opt, option_mask=option_mask,
                            mode_id=mode_id, option_perm=option_perm)
        # MaxSim: each option token's best match in the memory, averaged per option
        o = F.normalize(self.po(opt_tokens), dim=-1)                 # (B, T, d)
        m = F.normalize(self.pm(memory), dim=-1)                     # (B, M, d)
        sim = torch.einsum("btd,bmd->btm", o, m).masked_fill(~memory_mask[:, None, :], -2.0)
        best = sim.max(-1).values                                    # (B, T)
        K = option_mask.shape[1]
        slot = opt_slot.clamp(min=0)
        on = (opt_slot >= 0).to(best.dtype)
        tot = best.new_zeros((best.shape[0], K)).scatter_add(1, slot, best * on)
        cnt = best.new_zeros((best.shape[0], K)).scatter_add(1, slot, on).clamp_min(1.0)
        ms = tot / cnt
        logits = logits + (self.ms_scale * ms).masked_fill(~option_mask, 0.0)
        return logits.masked_fill(~option_mask, float("-inf"))


# ---- post-hoc calibration (cal-4b), ported verbatim from code_rcrel_exit (ladder-v1) -----
# Applied to the MAIN exit only (the scorer that model() reads); aux exits are never touched by these buffers.
CAL_PCA_DIM = 16          # hidden-state directions the confidence head reads
CAL_N_SCALAR = 7          # p_top, top-2 logit gap, normalised entropy, log K, mode one-hot (3)
CAL_LOGT_CLAMP = 3.0      # |log tau| <= 3: tau in [0.05, 20]


def cal_features(logits: torch.Tensor, decision_h: torch.Tensor, mode_id: torch.Tensor | None,
                 pca_mean: torch.Tensor, pca_W: torch.Tensor) -> torch.Tensor:
    """Input features of the confidence head, all from the SAME forward pass."""
    z = logits.float()
    finite = torch.isfinite(z)
    k = finite.sum(-1).clamp_min(2).float()
    p = torch.softmax(z, dim=-1)
    top2 = torch.topk(z.masked_fill(~finite, -1e9), 2, dim=-1).values
    gap = (top2[:, 0] - top2[:, 1]).clamp(0, 30)
    ent = -(p * torch.log(p.clamp_min(1e-12))).masked_fill(~finite, 0).sum(-1) / torch.log(k)
    ptop = p.max(-1).values
    if mode_id is None:
        mode_id = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
    onehot = F.one_hot(mode_id.long(), len(MODES)).float()
    proj = (decision_h.float() - pca_mean) @ pca_W
    return torch.cat([proj, ptop[:, None], gap[:, None], ent[:, None], torch.log(k)[:, None],
                      onehot], dim=-1)


class DecisionModel(nn.Module):
    """A frozen (or tuned) text tower plus a trained option scorer."""

    def __init__(self, tower: nn.Module, hidden: int, cfg: ArchConfig,
                 lm_head: nn.Module | None = None):
        super().__init__()
        self.tower = tower
        self.cfg = cfg
        if cfg.tree_rule == "late":
            self.scorer = LateScorer(hidden, cfg)
        else:
            self.scorer = (OptionScorer(hidden, cfg) if cfg.head_design == "none"
                           else DesignedScorer(hidden, cfg))
        if cfg.tree_rule not in ("none", "indep", "dec_sees_opts", "bidir", "late", "chain"):
            raise ValueError(f"tree_rule {cfg.tree_rule!r}")
        self.lm_head = lm_head
        if int(getattr(cfg, "freeze_lower_n", 0) or 0) > 0:
            import re as _re
            _pat = _re.compile(r"(?:^|\.)layers\.(\d+)\.")
            _n = 0
            for _name, _p in tower.named_parameters():
                _m = _pat.search(_name)
                if _m and "visual" not in _name and int(_m.group(1)) < int(cfg.freeze_lower_n):
                    _p.requires_grad_(False); _n += _p.numel()
            print(f"    freeze_lower_n={cfg.freeze_lower_n}: {_n:,} tower params frozen", flush=True)
        # cal-4b buffers (identity while cal_mode == "none"; fitted by calibrate.calibrate on DEV after training)
        self.cal_mode = "none"
        k_ = CAL_PCA_DIM
        self.register_buffer("cal_logT", torch.zeros(()))
        self.register_buffer("cal_logT_mode", torch.zeros(len(MODES)))
        self.register_buffer("cal_pca_mean", torch.zeros(hidden))
        self.register_buffer("cal_pca_W", torch.zeros(hidden, k_))
        self.register_buffer("cal_feat_mu", torch.zeros(k_ + CAL_N_SCALAR))
        self.register_buffer("cal_feat_sd", torch.ones(k_ + CAL_N_SCALAR))
        self.register_buffer("cal_w", torch.zeros(k_ + CAL_N_SCALAR))
        self.register_buffer("cal_b", torch.zeros(()))
        self.mix_logits = None
        if cfg.layer_mix:
            cands = list(cfg.layer_mix)
            self.register_buffer("_mix_candidates", torch.tensor(cands), persistent=False)
            init = torch.zeros(len(MODES), len(cands))
            if cfg.layer_mix_init is not None:
                want = cfg.layer_mix_init
                idx = next((i for i, c in enumerate(cands) if c == want), None)
                if idx is None:
                    raise ValueError(f"layer_mix_init {want} is not in layer_mix {cands}")
                init[:, idx] = cfg.layer_mix_temp
            self.mix_logits = nn.Parameter(init)
        if lm_head is not None:
            # Freeze it whenever it is present, not only under `residual`:
            # untied, an arm that merely passes the head for prior anchoring
            # would silently train 254 M parameters while its record claims a
            # few thousand.
            #
            # But freeze ONLY weights the head does not share with the tower.
            # Qwen3.5 ties lm_head to embed_tokens, so an unconditional freeze
            # would also freeze the EMBEDDING -- invisible today because
            # freeze_base is always on, and a silent behaviour change the first
            # time an arm sets freeze_base=False to tune the tower. A tied head
            # is governed by the tower's freeze status; an untied one is frozen
            # here.
            shared = {id(p) for p in self.tower.parameters()}
            for p in self.lm_head.parameters():
                if id(p) not in shared:
                    p.requires_grad_(False)
        if cfg.residual:
            if lm_head is None:
                raise ValueError("residual readout needs the lm_head")
            self.scorer.zero_init_output()
        if cfg.freeze_base:
            for p in self.tower.parameters():
                p.requires_grad_(False)
        self.frozen_lower_layers = 0
        if cfg.freeze_lower_frac and not cfg.freeze_base:
            layers = _decoder_layers(self.tower)
            n_freeze = int(round(len(layers) * float(cfg.freeze_lower_frac)))
            for layer in list(layers)[:n_freeze]:
                for p in layer.parameters():
                    p.requires_grad_(False)
            self.frozen_lower_layers = n_freeze
            print(f"    freeze_lower_frac={cfg.freeze_lower_frac}: froze decoder layers "
                  f"0-{n_freeze - 1} of {len(layers)}", flush=True)
        if cfg.embedding in ("frozen", "sliced", "replaced"):
            emb = getattr(self.tower, "get_input_embeddings", lambda: None)()
            if emb is not None:
                for p in emb.parameters():
                    p.requires_grad_(False)
        if cfg.exit_layer:
            if cfg.readout_layer != -1 or cfg.residual or cfg.layer_mix:
                raise ValueError("exit_layer needs readout_layer=-1, residual=False and no layer_mix")
            tm = _text_model(self.tower)
            n = len(tm.layers)
            # ax6: exit_layer == n is the full-depth model (every layer, then the final norm), so a
            # full-depth main head can carry aux exits below it. Same forward as exit_layer off.
            if not 1 <= int(cfg.exit_layer) <= n:
                raise ValueError(f"exit_layer {cfg.exit_layer} outside 1..{n}")
            # plain attributes (object.__setattr__): registering them would put a
            # second copy of every layer's keys into state_dict()
            object.__setattr__(self, "_exit_text", tm)
            object.__setattr__(self, "_exit_layers",
                               nn.ModuleList(list(tm.layers)[: int(cfg.exit_layer)]))
        if cfg.aux_exits:
            if not cfg.exit_layer or not cfg.exit_norm:
                raise ValueError("aux_exits needs exit_layer with exit_norm=True")
            if cfg.head_design != "none" or cfg.tree_rule != "none":
                raise ValueError("aux_exits supports head_design 'none' and tree_rule 'none' only")
            ex = sorted({int(x) for x in cfg.aux_exits})
            if any(not 1 <= x < int(cfg.exit_layer) for x in ex):
                raise ValueError(f"aux_exits {ex} must lie in 1..{int(cfg.exit_layer) - 1}")
            # built AFTER the main scorer, so the main head's initialisation (and
            # every RNG draw before it) is the fixed-exit arm's
            per = _layer_period(tm)
            if any(x % per for x in ex):
                # training is fine; the staged adaptive runtime (rsijev/adaptive.py) needs stage
                # starts on the layer-type period, so such an exit is read only by forward_exits
                print(f"    WARNING aux_exits {[x for x in ex if x % per]} not multiples of the "
                      f"layer-type period {per}: adaptive.run_layers cannot stage there", flush=True)
            self.aux_scorers = nn.ModuleDict({str(x): OptionScorer(hidden, cfg) for x in ex})
        else:
            self.aux_scorers = None

    def exit_indices(self) -> list[int]:
        """Every exit this model can read, shallow to deep (the last is exit_layer)."""
        aux = sorted(int(k) for k in self.aux_scorers) if self.aux_scorers is not None else []
        return aux + [int(self.cfg.exit_layer)]

    def exit_scorer(self, L: int) -> nn.Module:
        return self.scorer if int(L) == int(self.cfg.exit_layer) else self.aux_scorers[str(int(L))]

    @contextlib.contextmanager
    def exit_tower(self):
        """Inside this context a direct `self.tower(...)` call IS the truncated
        model: the text model runs its first exit_layer decoder layers, then its
        final norm (or the identity without exit_norm). With exit_norm and the
        tied embedding, last_hidden_state @ W.T is the exit model's own LM
        (fit.py retention_at_exit). A no-op when exit_layer is off."""
        if not self.cfg.exit_layer:
            yield
            return
        tm = self._exit_text
        full_layers, full_norm = tm.layers, tm.norm
        tm.layers = self._exit_layers
        if not self.cfg.exit_norm:
            tm.norm = nn.Identity()
        try:
            yield
        finally:
            tm.layers, tm.norm = full_layers, full_norm

    def _run_tower(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                   all_states: bool = False, position_ids: torch.Tensor | None = None,
                   inputs_embeds: torch.Tensor | None = None):
        """One tower pass, returning (state the correction reads, FINAL state).

        The two are not the same thing and must not be conflated. The correction
        may tap any layer; the residual base term is the model's own output and
        has to come from the final, normalised state whatever the correction
        reads -- otherwise `residual` at layer 19 would apply `lm_head` to a
        mid-stack, un-normalised residual-stream vector, which is not the
        control but arbitrary logits wearing its name.
        """
        # Image states (rsijev/vision.py) arrive as embeddings with the vision
        # features scattered over the image-pad tokens, plus M-RoPE position ids.
        # Text-only batches set neither, so their call is unchanged. Qwen3.5 injects
        # image features at the input embeddings only (deepstack_visual_indexes is
        # empty), so an early exit never cuts the image path.
        io = {} if position_ids is None else {"position_ids": position_ids}
        if inputs_embeds is None:
            io["input_ids"] = input_ids
        else:
            io["inputs_embeds"] = inputs_embeds
        if self.cfg.exit_layer:
            # Early exit: run only the first exit_layer decoder layers. The swap
            # is undone before returning, so state_dict(), direct tower calls and
            # saving all see the full model.
            with self.exit_tower():
                out = self.tower(attention_mask=attention_mask,
                                 output_hidden_states=True, **io)
            final = out.last_hidden_state           # output of layer exit_layer-1 (normed if exit_norm)
            # the tapped state IS the last entry; never trust a recorder's
            # convention for it (it is what readout_layer=-1 reads)
            hs = tuple(out.hidden_states[:-1]) + (final,)
            if all_states:
                return hs, final
            return final, final
        out = self.tower(attention_mask=attention_mask, output_hidden_states=True, **io)
        hs = out.hidden_states                      # tuple, embeddings first
        # `last_hidden_state` is unambiguously post-norm; hs[-1] is post-norm in
        # HF text models but that is a convention, not a guarantee, so prefer the
        # explicit field when the model provides it.
        final = getattr(out, "last_hidden_state", None)
        if final is None:
            final = hs[-1]
        if all_states:
            return hs, final
        if self.mix_logits is not None:
            if mode_id is None:
                raise ValueError("a layer mixture needs mode_id in the batch")
            n = len(hs)
            stack = torch.stack([hs[_norm_layer(int(c), n)] for c in self._mix_candidates])
            w = torch.softmax(self.mix_logits, dim=-1)[mode_id]      # (B, C)
            h = torch.einsum("cbth,bc->bth", stack.to(w.dtype), w)
            return self._readout(h, final, b, decision_index, option_index,
                                 option_span_start, option_span_end,
                                 option_token_ids, option_mask, want_base,
                                 mode_id=mode_id, option_perm=option_perm)
        layer = self.cfg.readout_layer
        if isinstance(layer, dict):
            raise ValueError("hidden_states() is single-layer; a per-mode readout "
                             "must go through forward()")
        return hs[layer], final

    def hidden_states(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        return self._run_tower(input_ids, attention_mask)[0]

    def _compute(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 decision_index: torch.Tensor,
                 option_index: torch.Tensor | None = None,
                 option_span_start: torch.Tensor | None = None,
                 option_span_end: torch.Tensor | None = None,
                 option_token_ids: torch.Tensor | None = None,
                 option_mask: torch.Tensor | None = None,
                 mode_id: torch.Tensor | None = None,
                 # Accepted and ignored: the permutation is undone downstream by
                 # encode.unpermute_logits, so the model never sees canonical
                 # option order and must not try to.
                 option_perm: torch.Tensor | None = None,
                 want_base: bool = False,
                 row_grad_scale: torch.Tensor | None = None,
                 # arch2 tree layout (encode.encode_tree / collate)
                 tree_seg: torch.Tensor | None = None,
                 tree_kind: torch.Tensor | None = None,
                 tree_parent: torch.Tensor | None = None,
                 tree_slot: torch.Tensor | None = None,
                 position_ids: torch.Tensor | None = None,
                 inputs_embeds: torch.Tensor | None = None):
        """One tower pass, returning (logits, base logits or None).

        The base term and the correction both come from this single pass. A
        separate `base_logits` call would run the frozen tower twice, and the
        tower IS the step: 752 M parameters against a 7.3 M head. An anchored arm
        would then take about twice the wall time of its own control, which
        would not change the metric but would corrupt every cost comparison in
        the report.
        """
        if (position_ids is not None or inputs_embeds is not None) and self.cfg.tree_rule != "none":
            raise ValueError("image states need the flat layout (tree_rule none)")
        if self.cfg.tree_rule != "none":
            return self._compute_tree(
                input_ids=input_ids, decision_index=decision_index, option_index=option_index,
                option_span_start=option_span_start, option_span_end=option_span_end,
                option_mask=option_mask, mode_id=mode_id, option_perm=option_perm,
                want_base=want_base, row_grad_scale=row_grad_scale, tree_seg=tree_seg,
                tree_kind=tree_kind, tree_parent=tree_parent, tree_slot=tree_slot)
        with _prefix_bidir_scope(self, attention_mask, option_span_end, option_mask):
            hs, final = self._run_tower(input_ids, attention_mask, all_states=True,
                                        position_ids=position_ids, inputs_embeds=inputs_embeds)
        b = torch.arange(final.shape[0], device=final.device)
        if self.mix_logits is not None:
            if mode_id is None:
                raise ValueError("a layer mixture needs mode_id in the batch")
            n = len(hs)
            stack = torch.stack([hs[_norm_layer(int(c), n)] for c in self._mix_candidates])
            w = torch.softmax(self.mix_logits, dim=-1)[mode_id]      # (B, C)
            h = torch.einsum("cbth,bc->bth", stack.to(w.dtype), w)
            if row_grad_scale is not None:
                h = _RowGradScale.apply(h, row_grad_scale)
            return self._readout(h, final, b, decision_index, option_index,
                                 option_span_start, option_span_end,
                                 option_token_ids, option_mask, want_base,
                                 mode_id=mode_id, option_perm=option_perm)
        layer = self.cfg.readout_layer
        if isinstance(layer, dict):
            if mode_id is None:
                raise ValueError("a per-mode readout_layer needs mode_id in the batch")
            h = torch.empty_like(hs[_norm_layer(list(layer.values())[0], len(hs))])
            for m, name in enumerate(MODES):
                rows = (mode_id == m).nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                if name not in layer:
                    raise ValueError(f"readout_layer has no entry for mode {name!r}")
                h[rows] = hs[_norm_layer(layer[name], len(hs))][rows]
        else:
            h = hs[layer]
        if row_grad_scale is not None:
            h = _RowGradScale.apply(h, row_grad_scale)
        return self._readout(h, final, b, decision_index, option_index,
                             option_span_start, option_span_end,
                             option_token_ids, option_mask, want_base,
                                 mode_id=mode_id, option_perm=option_perm)

    def _compute_tree(self, *, input_ids, decision_index, option_index, option_span_start,
                      option_span_end, option_mask, mode_id, option_perm, want_base,
                      row_grad_scale, tree_seg, tree_kind, tree_parent, tree_slot):
        """arch2: the tower over the segment tree (rsijev/tree.py), same readout."""
        from .tree import build_plan, text_model, tree_hidden
        from .encode import SEG_OPTION
        if tree_seg is None:
            raise ValueError(f"tree_rule {self.cfg.tree_rule!r} needs the tree layout (layout='tree')")
        if self.cfg.residual or want_base or self.mix_logits is not None:
            raise ValueError("tree execution supports neither residual/base logits nor layer_mix")
        layer = self.cfg.readout_layer
        if isinstance(layer, dict):
            raise ValueError("tree execution needs a single readout_layer")
        tm = text_model(self.tower)
        n = len(tm.layers)
        k = n if layer in (-1, n) else int(layer)
        conv_k = next(l.linear_attn.conv_kernel_size for l in tm.layers
                      if l.block_type == "linear_attention")
        plan = build_plan(tree_seg, tree_kind, tree_parent, self.cfg.tree_rule, conv_k=conv_k)
        late = self.cfg.tree_rule == "late"
        cap = (int(self.cfg.late_option_layer),) if late else ()
        h, caps = tree_hidden(self.tower, input_ids, plan, upto=k, final_norm=(k == n), capture=cap)
        if row_grad_scale is not None:
            h = _RowGradScale.apply(h, row_grad_scale)
        b = torch.arange(h.shape[0], device=h.device)
        late_kw = None
        if late:
            ho = caps[int(self.cfg.late_option_layer)]
            if row_grad_scale is not None:
                ho = _RowGradScale.apply(ho, row_grad_scale)
            dtype = next(self.scorer.parameters()).dtype
            slot = torch.where(plan.kind == SEG_OPTION,
                               tree_slot.gather(1, tree_seg.clamp(min=0)), torch.full_like(tree_seg, -1))
            late_kw = dict(memory=h.to(dtype), memory_mask=(plan.kind >= 0) & (plan.kind != SEG_OPTION),
                           opt_tokens=ho.to(dtype), opt_slot=slot)
            return self._readout(h, None, b, decision_index, option_index, option_span_start,
                                 option_span_end, None, option_mask, False, mode_id=mode_id,
                                 option_perm=option_perm, opt_src=ho, late_kw=late_kw)
        return self._readout(h, None, b, decision_index, option_index, option_span_start,
                             option_span_end, None, option_mask, False, mode_id=mode_id,
                             option_perm=option_perm)

    def _readout(self, h, final, b, decision_index, option_index,
                 option_span_start, option_span_end, option_token_ids,
                 option_mask, want_base, mode_id=None, option_perm=None,
                 opt_src=None, late_kw=None, scorer=None):
        scorer = self.scorer if scorer is None else scorer
        decision_h = h[b, decision_index]                        # (B, H)
        option_h = None
        if opt_src is not None:
            h = opt_src                                          # D4: options pooled from their own layer
        if option_span_start is not None and self.cfg.option_pool == "mean":
            # Mean-pool each option's whole block. The last token of an option is
            # usually punctuation, and a representation that does not differ
            # between options makes every readout blind by construction.
            t = torch.arange(h.shape[1], device=h.device).view(1, 1, -1)
            inside = ((t >= option_span_start.unsqueeze(-1)) &
                      (t < option_span_end.unsqueeze(-1))).to(h.dtype)   # (B,K,T)
            denom = inside.sum(-1, keepdim=True).clamp_min(1.0)
            option_h = torch.einsum("bkt,bth->bkh", inside, h) / denom
        elif option_index is not None:
            option_h = h[b.unsqueeze(1), option_index]           # (B, K, H)

        # The tower runs in bf16 and the scorer in fp32 (a few thousand
        # parameters, so the precision is free and the optimiser is better
        # behaved). Cast at the boundary rather than requiring every caller to
        # remember: a mismatch here is a runtime error, not a silent wrong answer,
        # but it costs an allocation every time.
        dtype = next(scorer.parameters()).dtype
        decision_h = decision_h.to(dtype)
        if option_h is not None:
            option_h = option_h.to(dtype)
        # Autocast OFF for the scorer. Under the bf16 autocast that tower-training
        # arms use to fit in memory, nn.Linear and MultiheadAttention run in bf16
        # WHATEVER the weight dtype, so the "fp32 scorer" above was computing its
        # attention softmax, projections and gradients in bf16. Every tower arm
        # showed seed-dependent logit spikes (up to 12096) that no lr, schedule,
        # input norm, smoothing or decay removed; frozen arms, which never use
        # autocast, never spiked. The scorer is 7 M parameters: fp32 is free.
        with torch.autocast(decision_h.device.type, enabled=False):
            logits = scorer(decision_h=decision_h, option_h=option_h,
                                 option_mask=option_mask, mode_id=mode_id,
                                 option_perm=option_perm, **(late_kw or {}))
        if self.cfg.logit_cap:
            c = float(self.cfg.logit_cap)
            logits = c * torch.tanh(logits / c)
            # tanh(-inf) = -1: re-mask, or padded options come back at -cap and
            # take probability mass.
            if option_mask is not None:
                logits = logits.masked_fill(~option_mask, float("-inf"))
        if getattr(self, "cal_mode", "none") != "none" and scorer is self.scorer:
            with torch.autocast(decision_h.device.type, enabled=False):
                log_t = self.cal_log_temperature(logits.float(), decision_h.float(), mode_id)
                logits = logits.float() / torch.exp(log_t).unsqueeze(-1)

        base = None
        if self.cfg.residual or want_base:
            if option_token_ids is None:
                raise ValueError("residual / prior anchoring needs option_token_ids")
            if self.lm_head is None:
                raise ValueError("residual / prior anchoring needs the lm_head")
            with torch.no_grad():
                # FINAL state, never `h`: the base term is the control at every
                # readout_layer, so the layer sweep moves only the correction.
                vocab = self.lm_head(final[b, decision_index])       # (B, V)
                base = vocab.gather(1, option_token_ids).float()
                if option_mask is not None:
                    base = base.masked_fill(~option_mask, float("-inf"))
        if self.cfg.residual:
            logits = base.to(logits.dtype) + torch.nan_to_num(logits, neginf=0.0)
            if option_mask is not None:
                logits = logits.masked_fill(~option_mask, float("-inf"))
        return logits, base

    def forward(self, **kw) -> torch.Tensor:
        return conv_safe_call(self.tower, self._compute, kw)[0]

    def forward_exits(self, **kw) -> dict:
        """_forward_exits, micro-batched when one tower call would overflow conv1d (conv_safe_call)."""
        return conv_safe_call(self.tower, self._forward_exits, kw)

    def _forward_exits(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                      decision_index: torch.Tensor,
                      option_index: torch.Tensor | None = None,
                      option_span_start: torch.Tensor | None = None,
                      option_span_end: torch.Tensor | None = None,
                      option_token_ids: torch.Tensor | None = None,
                      option_mask: torch.Tensor | None = None,
                      mode_id: torch.Tensor | None = None,
                      option_perm: torch.Tensor | None = None,
                      row_grad_scale: torch.Tensor | None = None,
                      position_ids: torch.Tensor | None = None,
                      inputs_embeds: torch.Tensor | None = None,
                      aux_detach: bool = False, **_) -> dict:
        """{exit index: logits} for every exit, from ONE pass of the exit tower.

        The main exit's entry is computed exactly as forward() computes it (same
        state, same readout, same scorer); each aux exit reads norm(hidden_states[L])
        through its own scorer. Presented option order, -inf where masked.
        aux_detach (lad2 port of rt9-ladder-detach, fit_extra.aux_detach_tower): each aux exit reads
        norm(hidden_states[L]).detach(), so aux losses train only their own heads and
        the trunk (incl. the shared final norm) gets the main-path gradient only."""
        if self.aux_scorers is None:
            raise ValueError("forward_exits needs arch aux_exits")
        # image states (vision_fit / VisionDecisionModel) come as inputs_embeds + M-RoPE
        # position ids, exactly as _compute passes them; text batches set neither
        with _prefix_bidir_scope(self, attention_mask, option_span_end, option_mask):
            hs, final = self._run_tower(input_ids, attention_mask, all_states=True,
                                        position_ids=position_ids, inputs_embeds=inputs_embeds)
        b = torch.arange(final.shape[0], device=final.device)
        norm = self._exit_text.norm

        def read(h, sc):
            if row_grad_scale is not None:
                h = _RowGradScale.apply(h, row_grad_scale)
            return self._readout(h, final, b, decision_index, option_index,
                                 option_span_start, option_span_end, option_token_ids,
                                 option_mask, False, mode_id=mode_id, option_perm=option_perm,
                                 scorer=sc)[0]
        out = {int(self.cfg.exit_layer): read(final, None)}
        for k, sc in self.aux_scorers.items():
            h_k = norm(hs[int(k)])
            out[int(k)] = read(h_k.detach() if aux_detach else h_k, sc)
        return dict(sorted(out.items()))

    def cal_log_temperature(self, logits: torch.Tensor, decision_h: torch.Tensor,
                            mode_id: torch.Tensor | None) -> torch.Tensor:
        """log tau per row, (B,). Argmax-preserving (one positive scalar per row)."""
        B = logits.shape[0]
        if self.cal_mode == "temp":
            return self.cal_logT.expand(B)
        if self.cal_mode == "temp_mode":
            if mode_id is None:
                raise ValueError("per-mode temperature needs mode_id in the batch")
            return self.cal_logT_mode[mode_id]
        if self.cal_mode in ("oof_head", "oof_head_scorefloor", "oof_head_joint"):
            f = cal_features(logits, decision_h, mode_id, self.cal_pca_mean, self.cal_pca_W)
            f = (f - self.cal_feat_mu) / self.cal_feat_sd
            lt = (f @ self.cal_w + self.cal_b).clamp(-CAL_LOGT_CLAMP, CAL_LOGT_CLAMP)
            if self.cal_mode == "oof_head_scorefloor":
                if mode_id is None:
                    raise ValueError("oof_head_scorefloor needs mode_id in the batch")
                lt = torch.where(mode_id == MODES.index("score"), lt.clamp_min(0.0), lt)
            return lt
        raise ValueError(f"unknown cal_mode {self.cal_mode!r}")

    def forward_with_base(self, **kw):
        """(logits, base logits) from ONE tower pass. Use this when anchoring."""
        return conv_safe_call(self.tower, self._compute, {**kw, "want_base": True})

    @torch.no_grad()
    def base_logits(self, **kw) -> torch.Tensor:
        """_base_logits, micro-batched when one tower call would overflow conv1d (conv_safe_call)."""
        return conv_safe_call(self.tower, self._base_logits, kw)

    @torch.no_grad()
    def _base_logits(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                    decision_index: torch.Tensor, option_token_ids: torch.Tensor,
                    option_mask: torch.Tensor | None = None, **_) -> torch.Tensor:
        """The frozen model's own option logits -- the zero-shot control's view.

        Available whether or not `residual` is on, because an objective may want
        to ANCHOR to the prior without starting from it. Those are different
        things: residual start did not stop training from damaging the prior out
        of distribution, so the interesting question moved from initialisation to
        what keeps a head from overwriting a prior it cannot see.

        This runs the tower on its own. Inside a training step use
        `forward_with_base`, which returns both from one pass.
        """
        if self.lm_head is None:
            raise ValueError("base_logits needs the lm_head")
        _, final = self._run_tower(input_ids, attention_mask)
        b = torch.arange(final.shape[0], device=final.device)
        vocab = self.lm_head(final[b, decision_index])
        out = vocab.gather(1, option_token_ids).float()
        if option_mask is not None:
            out = out.masked_fill(~option_mask, float("-inf"))
        return out

    @staticmethod
    def probs(logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits, dim=-1)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LogprobReadout(nn.Module):
    """ZERO-SHOT control: score each option by the base model's own logprob.

    **Zero trainable parameters.** This is the arm every trained readout has to
    beat, and until it is measured there is no scale on which a trained number
    means anything. It is the same mechanism the frozen phase-1 baseline uses --
    label logprobs at the answer position, normalised over the option set -- so
    a phase-2 gain decomposes into readout versus training rather than being
    confounded with model size.

    Options are scored by the FIRST token of their key, which is what makes the
    option set a closed vocabulary at the answer position. Where two options
    share a first token the scores collide; `colliding_options` reports that
    rather than letting it pass silently.
    """

    def __init__(self, causal_lm: nn.Module):
        super().__init__()
        self.lm = causal_lm
        for p in self.lm.parameters():
            p.requires_grad_(False)

    @staticmethod
    def option_token_ids(tokenizer, options: "tuple[str, ...]",
                         prefix: str = " ") -> list[int]:
        """First tokens that actually distinguish the options.

        A leading space is right for word options (" stop" is what follows
        "Answer:") and catastrophic for short numeric ones: this tokenizer emits
        the space as its own token, so " 0", " 1", " 2", " 3" all begin with
        token 220 and every score question scores identically. When the prefixed
        form collides, fall back to the bare form before giving up.
        """
        def first(o: str, pre: str) -> int:
            ids = tokenizer(pre + o, add_special_tokens=False)["input_ids"]
            return ids[0]
        ids = [first(o, prefix) for o in options]
        if len(set(ids)) == len(ids) or not prefix:
            return ids
        bare = [first(o, "") for o in options]
        return bare if len(set(bare)) == len(bare) else ids

    @staticmethod
    def colliding_options(tokenizer, options: "tuple[str, ...]",
                          prefix: str = " ") -> bool:
        ids = LogprobReadout.option_token_ids(tokenizer, options, prefix)
        return len(set(ids)) != len(ids)

    @torch.no_grad()
    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                decision_index: torch.Tensor, option_token_ids: torch.Tensor,
                option_mask: torch.Tensor, **_) -> torch.Tensor:
        """Project ONLY the decision rows through the vocabulary.

        Calling the causal LM produces logits at every position: (B, T, V) with
        V = 248,320. At eval batch 32 over 850-token rows that is about 13.5 GB
        in bf16 and 27 GB in fp32, for a tensor of which one row per example is
        ever read. It OOMed three arms. Running the tower and applying lm_head to
        the gathered (B, H) slice is (B, V) instead -- some 850x smaller, and
        faster.

        It also makes this the SAME computation as the residual base term in
        `_compute`, rather than merely the same tokens: control and base can no
        longer drift apart by one of them changing.
        """
        tower = getattr(self.lm, "model", self.lm)
        kw = {"input_ids": input_ids, "attention_mask": attention_mask, "decision_index": decision_index,
              "option_token_ids": option_token_ids, "option_mask": option_mask}
        return conv_safe_call(tower, lambda **k: self._pick(tower, **k), kw)

    def _pick(self, tower, *, input_ids, attention_mask, decision_index, option_token_ids, option_mask):
        out = tower(input_ids=input_ids, attention_mask=attention_mask)
        final = getattr(out, "last_hidden_state", None)
        if final is None:
            final = out.hidden_states[-1]
        b = torch.arange(final.shape[0], device=final.device)
        vocab = self.lm.lm_head(final[b, decision_index])      # (B, V)
        picked = vocab.gather(1, option_token_ids)             # (B, K)
        return picked.masked_fill(~option_mask, float("-inf"))


# ax6 conv1d 32-bit fix. Without the causal_conv1d kernel (the fla env has none) transformers' qwen3_5
# gated-delta-net layers run the reference F.conv1d, which refuses any input/output above 2^31-1 elements:
# B x conv_dim x (T + kernel-1) (conv_dim 8192 on the 4B, kernel 4). The 4B's AX-6 zero-shot control at eval
# batch 32 hit it on its first typed-decisions batch with a long (32k middle-cut) row; the legacy bf16 eval path
# (batch 32, no token budget) would hit it the same way. Any tower call that would overflow is run as micro-batches
# along the batch dim with the SAME padded length, and the per-row outputs are concatenated: every row's
# computation and the loss over the concatenated logits are those of the unsplit call (gradients flow through the
# cat, so no loss rescaling is needed). Calls that fit (every x20-sized batch) run exactly as before.
_I32_MAX = 2 ** 31 - 1
_CONV_GEOM: dict = {}
_CONV_SPLITS = [0]


def conv_geometry(root: nn.Module) -> tuple:
    """(conv_dim, kernel-1) of the first linear-attention layer under root; (0, 0) when there is none."""
    k = id(root)
    if k not in _CONV_GEOM:
        g = (0, 0)
        for m in root.modules():
            cd, conv = getattr(m, "conv_dim", None), getattr(m, "conv1d", None)
            if isinstance(cd, int) and conv is not None and hasattr(conv, "kernel_size"):
                g = (int(cd), int(conv.kernel_size[0]) - 1)
                break
        _CONV_GEOM[k] = g
    return _CONV_GEOM[k]


def conv_safe_rows(root: nn.Module, batch: int, seq: int, limit: int | None = None):
    """Rows per micro-batch so one conv1d call stays within `limit` elements, or None when no split is needed."""
    cd, pad = conv_geometry(root)
    if not cd:
        return None
    limit = _I32_MAX if limit is None else limit
    per_row = cd * (int(seq) + pad)
    if int(batch) * per_row <= limit:
        return None
    return max(1, limit // per_row)


def _cat_rows(outs):
    o = outs[0]
    if o is None:
        return None
    if torch.is_tensor(o):
        return torch.cat(outs, 0)
    if isinstance(o, dict):
        return {k: _cat_rows([x[k] for x in outs]) for k in o}
    if isinstance(o, (tuple, list)):
        return type(o)(_cat_rows(list(x)) for x in zip(*outs))
    raise TypeError(f"conv_safe_call: cannot concatenate {type(o)}")


def conv_safe_call(root: nn.Module, fn, kw: dict, limit: int | None = None):
    """fn(**kw), split along the batch dim when B x conv_dim x (T+k-1) would exceed `limit`."""
    ids = kw["input_ids"]
    B, T = int(ids.shape[0]), int(ids.shape[1])
    rows = conv_safe_rows(root, B, T, limit)
    if rows is None:
        return fn(**kw)
    if _CONV_SPLITS[0] < 5:
        print(f"[arch] conv1d split: batch {B} x seq {T} x conv_dim {conv_geometry(root)[0]} -> micro-batches "
              f"of {rows} rows", flush=True)
    _CONV_SPLITS[0] += 1
    outs = []
    for s in range(0, B, rows):
        # ladder-v1: M-RoPE position_ids are (3, B, T) -> split along dim 1 (vision rows)
        sub = {k: (v[:, s:s + rows] if k == "position_ids" and torch.is_tensor(v) and v.dim() == 3 else
                   v[s:s + rows] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v)
               for k, v in kw.items()}
        outs.append(fn(**sub))
    return _cat_rows(outs)


# m4b7 memory settings, ax6: applied ONLY to long rows (key length > _LONG_KEY tokens; the lc3 / 32k rows).
# m4b7 found: (1) with a padding mask the SDPA math kernel materialises B x 16 x L x L scores (64 GiB at L=32k on
# the 4B) -> no math kernel; (2) the fp32 eval with no mask takes the GQA path, which only flash (bf16) or math
# serve -> repeat KV (GQA off) so the memory-efficient kernel serves fp32 too. m4b7 set both globally; here short
# rows keep the stock kernels and GQA path (x20's numerics), long rows get mem-efficient/flash only + GQA off.
_LONG_KEY = 4096
try:
    import transformers.integrations.sdpa_attention as _sdpa_ax6
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF_ax6
    from torch.nn.attention import SDPBackend as _SDPB, sdpa_kernel as _sdpa_kernel
    _stock_use_gqa = _sdpa_ax6.use_gqa_in_sdpa
    _stock_sdpa_fwd = _sdpa_ax6.sdpa_attention_forward

    def _use_gqa_ax6(attention_mask, key, value):
        if key.shape[-2] > _LONG_KEY:
            return False
        return _stock_use_gqa(attention_mask, key, value)

    def _sdpa_fwd_ax6(module, query, key, value, *a, **k):
        if key.shape[-2] > _LONG_KEY and query.is_cuda:
            with _sdpa_kernel([_SDPB.EFFICIENT_ATTENTION, _SDPB.FLASH_ATTENTION]):
                return _stock_sdpa_fwd(module, query, key, value, *a, **k)
        return _stock_sdpa_fwd(module, query, key, value, *a, **k)
    _sdpa_ax6.use_gqa_in_sdpa = _use_gqa_ax6
    _AAF_ax6["sdpa"] = _sdpa_fwd_ax6
except Exception as _e:  # noqa: BLE001
    print(f"[arch] ax6 long-row SDPA settings NOT applied: {_e}", flush=True)


# E2 prefix-bidirectional full attention (2026-10-16; patch_bidir.py).
# Env RSIJEV_PREFIX_BIDIR=1: on full-attention layers the prefix [0, P) (prompt + options, P = end of the last option span)
# is non-causal; positions >= P stay causal; linear-attention layers unchanged. Env unset: nothing below changes a call.
import os as _os_e2


def _prefix_bidir_on() -> bool:
    return _os_e2.environ.get("RSIJEV_PREFIX_BIDIR", "1") == "1"


try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF_e2
    _prev_sdpa_e2 = _AAF_e2["sdpa"]

    def _sdpa_e2(module, query, key, value, attention_mask, *a, **k):
        if not getattr(attention_mask, "_jm_bidir", False):
            return _prev_sdpa_e2(module, query, key, value, attention_mask, *a, **k)
        k.pop("is_causal", None)
        T, rows = query.shape[2], []
        for b, (P, n) in enumerate(attention_mask._jm_rows):
            q, kk, vv = query[b:b + 1], key[b:b + 1], value[b:b + 1]
            parts = []
            if P > 0:                              # prefix queries see the whole prefix
                parts.append(_prev_sdpa_e2(module, q[:, :, :P], kk[:, :, :P], vv[:, :, :P], None, *a, is_causal=False, **k)[0])
            if n > P:                              # cue/answer queries: causal over [0, n)
                m = torch.arange(P, n, device=q.device)[:, None] >= torch.arange(n, device=q.device)[None]
                parts.append(_prev_sdpa_e2(module, q[:, :, P:n], kk[:, :, :n], vv[:, :, :n], m[None, None], *a, is_causal=False, **k)[0])
            if T > n:                              # right padding: never read by a valid position
                parts.append(query.new_zeros(1, T - n, query.shape[1], value.shape[-1]))
            rows.append(torch.cat(parts, 1))
        return torch.cat(rows, 0), None
    _AAF_e2["sdpa"] = _sdpa_e2
except Exception as _e:  # noqa: BLE001
    _AAF_e2 = None
    print(f"[arch] E2 prefix-bidir sdpa wrapper NOT installed: {_e}", flush=True)


def _bidir_pre_e2(mod, args, kwargs):
    rows = getattr(mod, "_jm_bidir_rows", None)
    am = kwargs.get("attention_mask")
    if rows is None or isinstance(am, dict):
        return None
    if args:
        raise ValueError("E2 prefix-bidir: the text model must be called with keyword arguments")
    x = kwargs.get("input_ids") if kwargs.get("input_ids") is not None else kwargs.get("inputs_embeds")
    B, T = int(x.shape[0]), int(x.shape[1])
    if len(rows) != B:
        raise ValueError(f"E2 prefix-bidir: {len(rows)} prefix rows for a batch of {B}")
    marker = torch.tensor(rows, dtype=torch.long, device=x.device)
    marker._jm_bidir, marker._jm_rows = True, rows
    lin = None if (am is None or T == 1 or bool((am == 1).all())) else am[:, -T:].contiguous()
    kwargs["attention_mask"] = {"full_attention": marker, "linear_attention": lin}
    return args, kwargs


@contextlib.contextmanager
def _prefix_bidir_scope(model, attention_mask, option_span_end, option_mask):
    """Inside: the text model's forward runs full-attention layers prefix-bidirectional. No-op when the env is unset."""
    if not _prefix_bidir_on():
        yield
        return
    if _AAF_e2 is None or option_span_end is None:
        raise ValueError("E2 prefix-bidir needs the sdpa wrapper and option spans in the batch")
    tm = _text_model(model.tower)
    if getattr(tm.config, "_attn_implementation", None) != "sdpa":
        raise ValueError(f"E2 prefix-bidir needs sdpa attention, not {tm.config._attn_implementation!r}")
    if not getattr(tm, "_jm_bidir_hooked", False):
        tm.register_forward_pre_hook(_bidir_pre_e2, with_kwargs=True)
        tm._jm_bidir_hooked = True
    om = option_mask if option_mask is not None else torch.ones_like(option_span_end, dtype=torch.bool)
    P = torch.where(om, option_span_end, torch.zeros_like(option_span_end)).max(dim=1).values
    n = attention_mask.sum(dim=1) if attention_mask is not None else None
    P, n = P.tolist(), (n.tolist() if n is not None else None)
    rows = [(min(int(p), int(m)), int(m)) for p, m in zip(P, n)] if n is not None else None
    if rows is None:
        raise ValueError("E2 prefix-bidir needs the 2D attention_mask (right padding)")
    tm._jm_bidir_rows = rows
    try:
        yield
    finally:
        tm._jm_bidir_rows = None
