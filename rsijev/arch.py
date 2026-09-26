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
        else:
            raise ValueError(cfg.readout)

    def zero_init_output(self) -> None:
        """Make the correction exactly 0 at initialisation."""
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, *, decision_h: torch.Tensor,
                option_h: torch.Tensor | None = None,
                option_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
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
            logits = self.proj(v + ctx).squeeze(-1)              # (B, K)
        if option_mask is not None:
            logits = logits.masked_fill(~option_mask, float("-inf"))
        return logits


CAL_PCA_DIM = 16          # hidden-state directions the confidence head reads
CAL_N_SCALAR = 7          # p_top, top-2 logit gap, normalised entropy, log K, mode one-hot (3)
CAL_LOGT_CLAMP = 3.0      # |log tau| <= 3: tau in [0.05, 20]


def cal_features(logits: torch.Tensor, decision_h: torch.Tensor, mode_id: torch.Tensor | None,
                 pca_mean: torch.Tensor, pca_W: torch.Tensor) -> torch.Tensor:
    """Input features of the confidence head, all from the SAME forward pass:
    the uncalibrated option distribution's shape, the question's mode and option
    count, and a low-rank projection of the decision hidden state (the only
    feature that can tell an in-distribution question from an OOD one)."""
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
        self.scorer = OptionScorer(hidden, cfg)
        self.lm_head = lm_head
        self.mix_logits = None
        # Post-hoc calibration (the cal-1 / cal-4 arms). Buffers only, never
        # parameters: nothing here is trained by the optimiser or counted as
        # trainable. They stay at identity (cal_mode "none") during training and
        # are fitted by fit.calibrate() on DEV data after the SFT steps; the
        # forward pass then rescales the logits it already computed, so a
        # decision is still ONE forward pass.
        self.cal_mode = "none"            # "none" | "temp" | "temp_mode" | "oof_head" | "oof_head_scorefloor"
        k = CAL_PCA_DIM
        self.register_buffer("cal_logT", torch.zeros(()))
        self.register_buffer("cal_logT_mode", torch.zeros(len(MODES)))
        self.register_buffer("cal_pca_mean", torch.zeros(hidden))
        self.register_buffer("cal_pca_W", torch.zeros(hidden, k))
        self.register_buffer("cal_feat_mu", torch.zeros(k + CAL_N_SCALAR))
        self.register_buffer("cal_feat_sd", torch.ones(k + CAL_N_SCALAR))
        self.register_buffer("cal_w", torch.zeros(k + CAL_N_SCALAR))
        self.register_buffer("cal_b", torch.zeros(()))
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
        if cfg.embedding in ("frozen", "sliced", "replaced"):
            emb = getattr(self.tower, "get_input_embeddings", lambda: None)()
            if emb is not None:
                for p in emb.parameters():
                    p.requires_grad_(False)

    def encode_prefix(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """Run a shared prefix once and return its cache.

        Every question of a case is `state + question + options + cue`, so the
        state is a true prefix of all of them. Under a causal mask, running the
        state once and continuing each question on its cache computes the same
        activations as running each full sequence -- so this is an exact
        optimisation, not an approximation, and `tests/test_prefix_cache.py`
        checks that rather than trusting it.
        """
        with torch.no_grad():
            return self.tower(input_ids=input_ids, attention_mask=attention_mask,
                              use_cache=True).past_key_values

    def _run_tower(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                   all_states: bool = False, past_key_values=None,
                   position_ids: torch.Tensor | None = None):
        """One tower pass, returning (state the correction reads, FINAL state).

        The two are not the same thing and must not be conflated. The correction
        may tap any layer; the residual base term is the model's own output and
        has to come from the final, normalised state whatever the correction
        reads -- otherwise `residual` at layer 19 would apply `lm_head` to a
        mid-stack, un-normalised residual-stream vector, which is not the
        control but arbitrary logits wearing its name.
        """
        extra = {}
        if past_key_values is not None:
            # The mask spans prefix + this chunk; positions continue past the
            # prefix. hidden_states then cover only the chunk, which is why the
            # readout indices are given relative to it.
            extra = {"past_key_values": past_key_values, "use_cache": True}
        if position_ids is not None:
            extra["position_ids"] = position_ids
        out = self.tower(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, **extra)
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
                                 option_token_ids, option_mask, want_base)
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
                 past_key_values=None,
                 position_ids: torch.Tensor | None = None,
                 want_base: bool = False):
        """One tower pass, returning (logits, base logits or None).

        The base term and the correction both come from this single pass. A
        separate `base_logits` call would run the frozen tower twice, and the
        tower IS the step: 752 M parameters against a 7.3 M head. An anchored arm
        would then take about twice the wall time of its own control, which
        would not change the metric but would corrupt every cost comparison in
        the report.
        """
        hs, final = self._run_tower(input_ids, attention_mask, all_states=True,
                                    past_key_values=past_key_values,
                                    position_ids=position_ids)
        b = torch.arange(final.shape[0], device=final.device)
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
                                 mode_id=mode_id)
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
        return self._readout(h, final, b, decision_index, option_index,
                             option_span_start, option_span_end,
                             option_token_ids, option_mask, want_base,
                             mode_id=mode_id)

    def _readout(self, h, final, b, decision_index, option_index,
                 option_span_start, option_span_end, option_token_ids,
                 option_mask, want_base, mode_id=None):
        decision_h = h[b, decision_index]                        # (B, H)
        option_h = None
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
        dtype = next(self.scorer.parameters()).dtype
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
            logits = self.scorer(decision_h=decision_h, option_h=option_h,
                                 option_mask=option_mask)
        if self.cfg.logit_cap:
            c = float(self.cfg.logit_cap)
            logits = c * torch.tanh(logits / c)
            # tanh(-inf) = -1: re-mask, or padded options come back at -cap and
            # take probability mass.
            if option_mask is not None:
                logits = logits.masked_fill(~option_mask, float("-inf"))

        if self.cal_mode != "none":
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
        return self._compute(**kw)[0]

    def forward_with_base(self, **kw):
        """(logits, base logits) from ONE tower pass. Use this when anchoring."""
        return self._compute(want_base=True, **kw)

    @torch.no_grad()
    def base_logits(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
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

    def cal_log_temperature(self, logits: torch.Tensor, decision_h: torch.Tensor,
                            mode_id: torch.Tensor | None) -> torch.Tensor:
        """log tau per row, (B,). Argmax-preserving: every row is divided by one
        positive scalar, so top-1 is exactly the uncalibrated model's."""
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
                # score rows soften but never sharpen (cal-4b)
                lt = torch.where(mode_id == MODES.index("score"), lt.clamp_min(0.0), lt)
            return lt
        raise ValueError(f"unknown cal_mode {self.cal_mode!r}")

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
        out = tower(input_ids=input_ids, attention_mask=attention_mask)
        final = getattr(out, "last_hidden_state", None)
        if final is None:
            final = out.hidden_states[-1]
        b = torch.arange(final.shape[0], device=final.device)
        vocab = self.lm.lm_head(final[b, decision_index])      # (B, V)
        picked = vocab.gather(1, option_token_ids)             # (B, K)
        return picked.masked_fill(~option_mask, float("-inf"))
