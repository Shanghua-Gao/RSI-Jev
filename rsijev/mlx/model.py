"""The MLX decision model: text tower + exit heads + calibration (+ vision tower), loaded
from an MLX checkpoint directory (scripts/convert_mlx.py).

`MLXDecisionModel` mirrors what serving reads off the PyTorch DecisionModel:

  forward(batch)          the main exit's calibrated logits (fixed exit), presented order
  forward_exits(batch)    every exit's logits from one pass, as DecisionModel.forward_exits:
                          aux exits read norm(h_L) with the shared final norm; the main
                          head's calibration (cal_mode) applies to every entry
  staged(batch, policy)   adaptive exit: the tower runs exit to exit (16 -> 20 -> 32) on the
                          rows still undecided; a row stops at the first aux exit whose
                          calibrated top-1 probability reaches that exit's tau and is answered
                          with that exit's calibration (rsijev.adaptive.staged_scores_fast +
                          serve/infer._adaptive_finalizer)
  encode_prefix(ids)      a prefix cache for the "cached" path
  image_embeds(pv, grid)  the vision tower's features

and the attributes serve/effort.py and serve/infer.py read: cfg (exit_layer, aux_exits,
option_pool, logit_cap, residual), effort_base, effort_auto_taus, adaptive_policy,
adaptive_mode, cal_mode, image_token_id.

Batches are numpy dicts from rsijev.mlx.collate.collate; logits come back as mx arrays.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .heads import Calibration, OptionScorer, cap_logits, calibrated_conf, exit_logt, pool
from .text import PrefixCache, TextConfig, TextTower, rotary_cos_sin, text_positions

F32 = mx.float32


@dataclass
class Policy:
    """rsijev.adaptive.Policy without torch: exits shallow to deep (the last always answers),
    a calibrator per aux exit ({"logT": x} or a cal-4b dict), the threshold, and optional
    per-exit thresholds. tau > 1 never exits early; tau < 0 always stops at the first exit."""
    exits: list
    cal: dict = field(default_factory=dict)
    tau: float = 2.0
    taus: dict = field(default_factory=dict)

    def tau_at(self, L) -> float:
        return self.taus.get(int(L), self.tau)


@dataclass
class ServeCfg:
    """The ArchConfig fields serving reads."""
    exit_layer: int
    aux_exits: tuple = ()
    option_pool: str = "mean"
    logit_cap: float | None = None
    readout: str = "option_xattn"
    residual: bool = False
    head_input_norm: bool = False
    xattn_heads: int = 4
    xattn_combine: str = "sum"
    max_options: int = 80


def _spec_cfg(spec: dict, n_layers: int) -> ServeCfg:
    extra = dict(spec.get("arch_extra") or {})
    if spec.get("readout") != "option_xattn":
        raise RuntimeError(f"the MLX backend serves readout option_xattn; this release uses "
                           f"{spec.get('readout')!r}")
    if spec.get("residual"):
        raise RuntimeError("the MLX backend does not serve the residual readout")
    if extra.get("head_design", "none") != "none" or extra.get("layer_mix"):
        raise RuntimeError("the MLX backend serves head_design 'none' without a layer mixture")
    if spec.get("readout_layer", -1) != -1:
        raise RuntimeError("the MLX backend serves readout_layer -1 (the exit's normed state)")
    if extra.get("exit_layer") and not extra.get("exit_norm", True):
        raise RuntimeError("the MLX backend serves exit_norm=True")
    return ServeCfg(exit_layer=int(extra.get("exit_layer") or n_layers),
                    aux_exits=tuple(sorted(int(x) for x in extra.get("aux_exits") or ())),
                    option_pool=spec.get("option_pool", "mean"), logit_cap=spec.get("logit_cap"),
                    head_input_norm=bool(spec.get("head_input_norm", False)),
                    xattn_heads=int(extra.get("xattn_heads", 4)),
                    xattn_combine=extra.get("xattn_combine", "sum"),
                    max_options=int(spec.get("max_options", 80)))


class MLXDecisionModel:
    def __init__(self, path: str | Path, *, dtype=mx.bfloat16, vision: bool = True):
        path = Path(path)
        self.path = path
        self.meta = json.loads((path / "meta.json").read_text())
        self.config = json.loads((path / "config.json").read_text())
        mj = path / "mlx.json"
        self.mlx_record = json.loads(mj.read_text()) if mj.exists() else {}
        spec = self.meta["spec"]
        self.tcfg = TextConfig.from_config(self.config)
        self.cfg = _spec_cfg(spec, self.tcfg.num_hidden_layers)
        self.dtype = dtype
        group = (self.mlx_record.get("tower") or {}).get("group_size") or \
            (self.mlx_record.get("embed_tokens") or {}).get("group_size")
        params = mx.load(str(path / "tower.safetensors"))
        self.tower = TextTower(self.tcfg, params, group=group, dtype=dtype,
                               n_layers=self.cfg.exit_layer)
        del params
        kw = dict(heads=self.cfg.xattn_heads, combine=self.cfg.xattn_combine,
                  head_input_norm=self.cfg.head_input_norm)
        self.scorer = OptionScorer(mx.load(str(path / "scorer.safetensors")), **kw)
        self.aux_scorers = {}
        if self.cfg.aux_exits:
            flat = mx.load(str(path / "aux_scorers.safetensors"))
            for L in self.cfg.aux_exits:
                sd = {k.split(".", 1)[1]: v for k, v in flat.items() if k.split(".", 1)[0] == str(L)}
                self.aux_scorers[L] = OptionScorer(sd, **kw)
        self.calibration = Calibration()
        if (path / "calibration.safetensors").exists():
            mode = json.loads((path / "calibration.json").read_text())["cal_mode"]
            self.calibration = Calibration(mode, mx.load(str(path / "calibration.safetensors")))
        self.cal_mode = self.calibration.mode
        self.image_token_id = self.config.get("image_token_id")
        self.visual = None
        if vision and (path / "visual.safetensors").exists():
            from .vision import VisionTower
            self.visual = VisionTower(self.config, mx.load(str(path / "visual.safetensors")))
        # set by rsijev.mlx.serve.attach_policy (as serve.release.adaptive_policy does)
        self.adaptive_policy = None
        self.adaptive_mode = "off"
        self.effort_base = None
        self.effort_auto_taus = None

    # ------------------------------------------------------------------ structure
    def exit_indices(self) -> list[int]:
        return sorted(self.aux_scorers) + [int(self.cfg.exit_layer)]

    def exit_scorer(self, L: int) -> OptionScorer:
        return self.scorer if int(L) == int(self.cfg.exit_layer) else self.aux_scorers[int(L)]

    def eval(self):
        return self

    # ------------------------------------------------------------------ pieces
    def embed(self, ids: np.ndarray, image_embeds: mx.array | None = None) -> mx.array:
        """Token embeddings, image features written over the image-pad rows in order."""
        h = self.tower.embed(mx.array(ids))
        if image_embeds is None:
            return h
        rows, cols = np.nonzero(np.asarray(ids) == self.image_token_id)
        if len(rows) != image_embeds.shape[0]:
            raise ValueError(f"{len(rows)} image tokens for {image_embeds.shape[0]} image features")
        h[mx.array(rows), mx.array(cols)] = image_embeds.astype(h.dtype)
        return h

    def image_embeds(self, pixel_values, grid) -> mx.array:
        if self.visual is None:
            raise RuntimeError("this checkpoint was loaded without its vision tower")
        return self.visual(pixel_values, grid)

    def _rope(self, batch: dict, B: int, T: int, start: int = 0):
        pos = batch.get("position_ids")
        if pos is None:
            pos = text_positions(B, T, start)
        else:
            pos = mx.array(np.asarray(pos))
        return rotary_cos_sin(pos, self.tcfg, self.dtype)

    def _readout(self, scorer, h, batch, rows=None):
        """(raw logits (n, K) fp32, decision state fp32) for normed states h of `rows`."""
        g = (lambda k: mx.array(batch[k])) if rows is None else (lambda k: mx.array(batch[k][rows]))
        mask = g("option_mask")
        dh, oh = pool(h, g("decision_index"), g("option_span_start"), g("option_span_end"),
                      g("option_index"), self.cfg.option_pool)
        z = scorer(dh, oh, mask)
        return cap_logits(z, self.cfg.logit_cap, mask), dh

    def _start(self, batch, cache: PrefixCache | None, image_embeds=None):
        ids = np.asarray(batch["input_ids"])
        B, T = ids.shape
        h = self.embed(ids, image_embeds)
        rope = self._rope(batch, B, T, cache.length if cache is not None else 0)
        return h, rope

    # ------------------------------------------------------------------ passes
    def forward_exits(self, batch: dict, *, cache: PrefixCache | None = None,
                      image_embeds=None, calibrate: bool = True) -> dict:
        """{exit: logits (B, K)} from one pass (presented order, -inf where masked)."""
        h, rope = self._start(batch, cache, image_embeds)
        mode = mx.array(batch["mode_id"])
        out, lo = {}, 0
        for L in self.exit_indices():
            h = self.tower.run(h, lo, L, rope, cache=cache)
            z, dh = self._readout(self.exit_scorer(L), self.tower.norm(h), batch)
            out[L] = self.calibration.apply(z, dh, mode) if calibrate else z
            lo = L
        return out

    def forward(self, batch: dict, *, cache: PrefixCache | None = None, image_embeds=None):
        """The main exit's calibrated logits (B, K): the fixed-exit model."""
        h, rope = self._start(batch, cache, image_embeds)
        h = self.tower.run(h, 0, self.cfg.exit_layer, rope, cache=cache)
        z, dh = self._readout(self.scorer, self.tower.norm(h), batch)
        return self.calibration.apply(z, dh, mx.array(batch["mode_id"]))

    def finalize(self, policy: Policy, L: int, z, dh, mode):
        """The calibration a row answered at exit L gets (serve/infer._adaptive_finalizer)."""
        if self.cal_mode == "none":
            return z
        if int(L) == int(self.cfg.exit_layer):
            return self.calibration.apply(z, dh, mode)
        return z.astype(F32) / mx.exp(exit_logt(policy.cal[int(L)], z, dh, mode))[:, None]

    def staged(self, batch: dict, policy: Policy, *, cache: PrefixCache | None = None,
               stats: dict | None = None):
        """Adaptive exit for one collated batch -> (logits (B, K) mx, exit per row list)."""
        ids = np.asarray(batch["input_ids"])
        B, T = ids.shape
        exits = list(policy.exits)
        if not any(policy.tau_at(L) <= 1.0 for L in exits[:-1]):
            exits = exits[-1:]                      # no aux exit can ever be taken
        per = self.tcfg.layer_period()
        if any(L % per for L in exits[:-1]):
            raise ValueError(f"exits {exits} are not multiples of the layer-type period {per}")
        h, rope = self._start(batch, cache)
        mode_all = np.asarray(batch["mode_id"])
        rows = np.arange(B)
        out: list = [None] * B
        depth = [int(exits[-1])] * B
        lo = 0
        for i, L in enumerate(exits):
            last = i == len(exits) - 1
            h = self.tower.run(h, lo, L, rope, cache=cache)
            z, dh = self._readout(self.exit_scorer(L), self.tower.norm(h), batch, rows)
            mode = mx.array(mode_all[rows])
            if stats is not None:
                stats.setdefault("rows_at", {})[L] = stats.get("rows_at", {}).get(L, 0) + len(rows)
            if last:
                zf = self.finalize(policy, L, z, dh, mode)
                for j, r in enumerate(rows.tolist()):
                    out[r] = zf[j]
                break
            conf = np.array(calibrated_conf(z, dh, mode, policy.cal[int(L)]))
            stop = conf >= policy.tau_at(L)
            if stop.any():
                zf = self.finalize(policy, L, z, dh, mode)
                for j in np.nonzero(stop)[0].tolist():
                    out[int(rows[j])] = zf[j]
                    depth[int(rows[j])] = int(L)
                keep = np.nonzero(~stop)[0]
                if len(keep) == 0:
                    break
                rows = rows[keep]
                kk = mx.array(keep)
                h = h[kk]
                cos, sin = rope
                rope = (cos[kk], sin[kk])
            lo = L
        return mx.stack(out), depth

    def encode_prefix(self, ids: list[int] | np.ndarray, *, positions=None, image_embeds=None,
                      cache: PrefixCache | None = None) -> PrefixCache:
        """Run a shared prefix (1, P) through every layer and keep its cache. With `cache`,
        continue that cache over `ids` (positions continue from cache.length unless given)."""
        ids = np.asarray(ids, dtype=np.int32).reshape(1, -1)
        start = cache.length if cache is not None else 0
        h = self.embed(ids, image_embeds)
        pos = (text_positions(1, ids.shape[1], start) if positions is None
               else mx.array(np.asarray(positions)))
        rope = rotary_cos_sin(pos, self.tcfg, self.dtype)
        rec = PrefixCache()
        self.tower.run(h, 0, self.cfg.exit_layer, rope, cache=cache, record=rec)
        if cache is not None:
            rec = _join(cache, rec)
        mx.eval([a for c in rec.layers.values() for a in (c.keys, c.values, c.conv, c.state)
                 if a is not None])
        return rec


def _join(old: PrefixCache, new: PrefixCache) -> PrefixCache:
    """A cache continued over more tokens: keys/values appended, DeltaNet state replaced."""
    from .text import LayerCache
    out = PrefixCache(length=new.length)
    for i, c in new.layers.items():
        o = old.layers.get(i)
        if c.keys is not None and o is not None and o.keys is not None:
            out.layers[i] = LayerCache(keys=mx.concatenate([o.keys, c.keys], axis=2),
                                       values=mx.concatenate([o.values, c.values], axis=2))
        else:
            out.layers[i] = c
    return out


def load(path: str | Path, *, dtype=mx.bfloat16, vision: bool = True) -> MLXDecisionModel:
    return MLXDecisionModel(path, dtype=dtype, vision=vision)
