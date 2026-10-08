"""Right-padded, CUDA-graphed text passes (v6.1-VL 27B and later multi-exit releases).

Numerics. With a pad multiple m > 0 (meta.json `serving.pad_multiple`; RSIJEV_PAD_MULTIPLE
overrides it, and RSIJEV_PAD_MULTIPLE=0 is the unpadded eager path, byte for byte the code
that ran before), every text pass of serve.infer.run_rows and serve.infer.score_adaptive
right-pads its rows to a multiple of m. Fixed exit (effort high, or no effort), medium,
low and auto all run one function, `_Runner._stage`: decoder layers lo..hi-1, the exit's
norm and head, the stop statistic and the exit's calibration. A CUDA graph is a capture of
that same function on the same static tensors, so graphs on and graphs off
(RSIJEV_CUDA_GRAPHS=0) return the same bits.

Which batches pad. Only short ones: a "plain" text batch pads iff rows x padded width <=
RSIJEV_PAD_MAX_TOKENS (default 768), a pure function of the batch's shape. Longer batches
and every prefix-cache suffix run the unpadded eager code unchanged (the RSIJEV_PAD_MULTIPLE=0
path, same batches), so they are bit-identical to it. Why (27B, padded against unpadded, per
request, one H200): graphs pay only while a pass is launch-bound;
1-question requests gained below ~600 tokens (x.45 at <128, x.54 at 128-256, x.88 at
384-512), were flat at 512-1536 and lost above, where padding waste and one-off captures
(about 4 passes each; up to 256 widths per row count under the old 16384-token graph limit)
cost more than replay saved; multi-question suffix passes were padded but never graphed
(+7-18% at 1-8k tokens). Every padded pass is graphed (RSIJEV_GRAPH_MAX_TOKENS defaults to
the pad limit), up to RSIJEV_GRAPH_MAX shapes (default 1024; the padded shapes number a few
hundred: widths <= 768/rows x stages x option widths); a shape past the cap runs the same
function eager (same bits).

Masks. Each row's length is known on the host, so the masks are built on the device from
the lengths, without the host synchronisation (`padding_mask.all()`) that keeps
transformers' own mask builder out of a graph. They are the masks transformers 5.17 builds
for a right-padded batch (create_causal_mask with sdpa, create_recurrent_attention_mask):
  full attention    bool (n, 1, T, T): key j is visible to query i iff j <= i and j < len[row]
  linear attention  the 2D padding mask (n, T), long, which zeroes the pad inputs
and both None when no row has a pad, as transformers then passes is_causal / no mask.
A single padded row takes no full-attention mask (is_causal; see own_masks): identical on
every position the readout reads, and the same SDPA kernel as the unpadded eager row.
tests/test_padded.py checks them against transformers' masks (hf_masks).

Rows of one batch are never padded with extra rows; a graph is keyed by (stage, rows,
width, ...). Rows that stop at an exit leave the batch; the rest continue at the same
width, so a row's numbers do not depend on how many rows stopped beside it beyond the
batch composition that eager staged serving already has.

Prefix-cache plans are never padded (prefix nor suffix). Pads after a document would enter the DeltaNet state:
a zeroed input still applies the layer's decay gate (g = -exp(A_log) * softplus(dt_bias))
to the recurrent state, and shifts zeros into the conv state. So the planner's path is kept:
a plan that reads the state once ("cached") or from the document cache ("doc") runs its
prefix and its suffixes unpadded and eager, as before; only short "plain" batches run on the
padded graphed function. The planner picks "cached" only when it saves >= 2048 tokens on an H200
(compute-bound passes, where graphs buy little); forcing those plans to "plain" to graph
them measured 0.69x (4 requests of 39-64 questions) and moved answers
through the path change, so it is not done. Image requests keep the unpadded eager path.
"""
from __future__ import annotations

import os
import threading
import warnings

import torch

ENV_PAD = "RSIJEV_PAD_MULTIPLE"
ENV_GRAPHS = "RSIJEV_CUDA_GRAPHS"
PAD_MAX_TOKENS = int(os.environ.get("RSIJEV_PAD_MAX_TOKENS", "768"))
GRAPH_MAX_TOKENS = int(os.environ.get("RSIJEV_GRAPH_MAX_TOKENS", str(PAD_MAX_TOKENS)))
GRAPH_MAX = int(os.environ.get("RSIJEV_GRAPH_MAX", "1024"))
# tests only: fill pads with this id instead of the tokenizer's pad id
PAD_ID_OVERRIDE: int | None = None


def pad_multiple(model) -> int:
    """The pad multiple this model serves with: RSIJEV_PAD_MULTIPLE if set (0 = unpadded
    eager), else meta.json serving.pad_multiple (load_release sets model.serving_pad_multiple)."""
    raw = os.environ.get(ENV_PAD, "").strip()
    if raw:
        return max(0, int(raw))
    return int(getattr(model, "serving_pad_multiple", 0) or 0)


def graphs_on() -> bool:
    return os.environ.get(ENV_GRAPHS, "1").strip().lower() not in ("0", "false", "no", "off")


def ceil_to(n: int, m: int) -> int:
    return -(-int(n) // m) * m


def supported(model) -> bool:
    """A DecisionModel whose exit readout this module reproduces: early-exit text model,
    normed exit, single readout layer at the exit, no residual base term, no layer mix."""
    cfg = getattr(model, "cfg", None)
    if cfg is None or getattr(model, "_exit_text", None) is None or not getattr(cfg, "exit_layer", 0):
        return False
    rl = getattr(cfg, "readout_layer", -1)
    return (not getattr(cfg, "residual", True) and getattr(model, "mix_logits", None) is None
            and bool(getattr(cfg, "exit_norm", False)) and rl in (-1, int(cfg.exit_layer)))


def active(model, device) -> int:
    """The pad multiple when the padded path applies to this model on this device, else 0."""
    m = pad_multiple(model)
    if m <= 0 or torch.device(device).type != "cuda" or not supported(model):
        return 0
    return m


def pads(n_rows: int, width: int, m: int) -> bool:
    """Whether a plain text batch of n_rows rows, longest `width` tokens, runs padded (and
    graphed): rows x padded width <= PAD_MAX_TOKENS. A function of the batch shape only, so
    a request's numbers never depend on what ran before it."""
    return m > 0 and int(n_rows) * ceil_to(width, m) <= PAD_MAX_TOKENS


def plan_path(model, plan: dict, batch_size: int, device) -> str:
    """The path a text plan runs on: the planner's own (prefix paths are kept; see the
    module docstring). One place to change the rule."""
    return plan["path"]


def hf_masks(tm, x, lengths):
    """transformers' own masks for this padded batch (tests only): what tm.forward builds."""
    import sys
    from transformers.cache_utils import DynamicCache
    mod = sys.modules[type(tm).__module__]
    n, T = x.shape[0], x.shape[1]
    am = (torch.arange(T, device=x.device).view(1, -1) < lengths.view(-1, 1)).long()
    cache = DynamicCache(config=tm.config) if getattr(tm.config, "use_cache", None) else None
    pos = torch.arange(T, device=x.device).view(1, -1).expand(n, -1)
    mk = {"config": tm.config, "inputs_embeds": x, "attention_mask": am, "past_key_values": cache,
          "position_ids": pos}
    rec = getattr(mod, "create_recurrent_attention_mask", None)
    return {"full_attention": mod.create_causal_mask(**mk),
            "linear_attention": rec(**mk) if rec is not None else tm._update_linear_attn_mask(am, cache)}


def own_masks(T: int, lengths: torch.Tensor, has_pad: bool, device):
    """Masks of a right-padded batch. One row: full attention takes no mask (SDPA
    is_causal, the flash kernel the unpadded one-row eager pass uses). For every real query
    position i < len that is the same mask as causal & key-padding (all keys <= i are real);
    the two differ only on pad queries, which nothing reads. The explicit mask there moved
    1-question answers 3-4/1000 (near-ties, max|dp| .082). Several rows: the
    explicit causal & key-padding mask, as transformers builds for the eager padded batch."""
    if not has_pad:
        return {"full_attention": None, "linear_attention": None}
    t = torch.arange(T, device=device)
    valid = t.view(1, -1) < lengths.view(-1, 1)                                  # (n, T)
    if lengths.shape[0] == 1:
        return {"full_attention": None, "linear_attention": valid.long()}
    full = (t.view(1, 1, -1, 1) >= t.view(1, 1, 1, -1)) & valid[:, None, None, :]
    return {"full_attention": full, "linear_attention": valid.long()}


class _Runner:
    """Padded stage passes for one model, with their CUDA graphs (one shared memory pool)."""

    def __init__(self, model):
        tm = model._exit_text
        self.model, self.tm, self.cfg = model, tm, tm.config
        self.layers = list(tm.layers)
        self.types = list(self.cfg.layer_types)
        self.use_cache = getattr(self.cfg, "use_cache", None)
        self.kw = {}
        if getattr(self.cfg, "is_causal", None) is not None:
            self.kw["is_causal"] = self.cfg.is_causal
        self.main = int(model.cfg.exit_layer)
        self.pool = None
        self.graphs: dict = {}
        self.failed: set = set()
        self.arena: dict = {}
        self.lock = threading.RLock()
        self.stats = {"replay": 0, "capture": 0, "eager": 0, "capture_failed": 0}

    # ------------------------------------------------------------------ the one function
    def _stage(self, st: dict, lo: int, hi: int, has_pad: bool, cal, want_conf: bool,
               write_x: bool) -> dict:
        from rsijev import adaptive as A
        tm, model = self.tm, self.model
        x = tm.embed_tokens(st["input_ids"]) if lo == 0 else st["x_in"]
        n, T = x.shape[0], x.shape[1]
        masks = own_masks(T, st["lengths"], has_pad, x.device)
        pos = torch.arange(T, device=x.device).view(1, 1, -1).expand(4, n, -1)
        tpos, rpos = pos[0], pos[1:]
        pe = tm.rotary_emb(x, rpos)
        cache = None
        if self.use_cache:
            from transformers.cache_utils import DynamicCache
            cache = DynamicCache(config=self.cfg)
        for j in range(lo, hi):
            x = self.layers[j](x, position_embeddings=pe, attention_mask=masks[self.types[j]],
                               position_ids=tpos, past_key_values=cache, use_cache=self.use_cache,
                               **self.kw)
        if write_x:
            st["x_out"].copy_(x)
        h = tm.norm(x)
        pool = A._pool_cache(st, T, x.dtype, x.device)
        z, dh = A._readout_pooled(model.exit_scorer(hi), h, st, None, pool,
                                  option_pool=model.cfg.option_pool, logit_cap=model.cfg.logit_cap)
        mode = st["mode_id"]
        out = {"z": z}
        if want_conf:
            out["conf"] = A.calibrated_conf(z, dh, mode, cal)
        # the calibration the answered rows get (serve.infer._adaptive_finalizer)
        if model.cal_mode == "none":
            out["zf"] = z
        else:
            with torch.autocast(z.device.type, enabled=False):
                if hi == self.main:
                    lt = model.cal_log_temperature(z.float(), dh.float(), mode)
                    out["zf"] = (z.float() / torch.exp(lt).unsqueeze(-1)).to(z.dtype)
                else:
                    lt = A.calibrate_logt(z.float(), dh.float(), mode, cal)
                    out["zf"] = (z.float() / torch.exp(lt)[:, None]).to(z.dtype)
        return out

    # ------------------------------------------------------------------ graphs
    def _flat(self, name: str, numel: int, dtype, device) -> torch.Tensor:
        buf = self.arena.get(name)
        if buf is None:
            buf = self.arena[name] = torch.empty(numel, dtype=dtype, device=device)
        return buf

    def _capture(self, key, fn):
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):          # Triton autotuning (fla) must not run under capture
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = fn()
            self.stats["capture"] += 1
            return g, out
        except Exception as e:              # noqa: BLE001 -- any failure: that shape runs eager
            self.failed.add(key)
            self.stats["capture_failed"] += 1
            warnings.warn(f"CUDA graph capture failed for {key[:6]}, running it eager: {e!r}"[:500])
            torch.cuda.synchronize()
            return None

    def run_stage(self, inputs: dict, lo: int, hi: int, has_pad: bool, cal, want_conf: bool,
                  write_x: bool):
        """(outputs, x_out or None) of one padded stage; inputs hold input_ids (lo == 0) or
        x_in, lengths, and the readout fields."""
        x0 = inputs["input_ids"] if lo == 0 else inputs["x_in"]
        n, T = int(x0.shape[0]), int(x0.shape[1])
        H, dt = int(self.cfg.hidden_size), self.tm.embed_tokens.weight.dtype
        dev = x0.device
        with self.lock:
            use = graphs_on() and n * T <= GRAPH_MAX_TOKENS
            if use:
                shapes = tuple((k, tuple(v.shape), v.dtype) for k, v in sorted(inputs.items()))
                key = (lo, hi, n, T, has_pad, want_conf, write_x, id(cal), shapes)
                entry = self.graphs.get(key)
                if entry is None and key not in self.failed and len(self.graphs) < GRAPH_MAX:
                    st = {}
                    for k, v in inputs.items():
                        if k == "x_in":
                            st[k] = self._flat("x_in", GRAPH_MAX_TOKENS * H, dt, dev)[: n * T * H].view(n, T, H)
                        else:
                            st[k] = torch.empty_like(v)
                    if write_x:
                        st["x_out"] = self._flat("x_out", GRAPH_MAX_TOKENS * H, dt, dev)[: n * T * H].view(n, T, H)
                    for k, v in inputs.items():
                        st[k].copy_(v)
                    got = self._capture(key, lambda: self._stage(st, lo, hi, has_pad, cal, want_conf, write_x))
                    if got is not None:
                        entry = self.graphs[key] = (got[0], got[1], st)
                if entry is not None:
                    g, out, st = entry
                    for k, v in inputs.items():
                        st[k].copy_(v)
                    g.replay()
                    self.stats["replay"] += 1
                    res = {k: v.clone() for k, v in out.items()}
                    return res, (st["x_out"].clone() if write_x else None)
            st = dict(inputs)
            if write_x:
                st["x_out"] = torch.empty((n, T, H), dtype=dt, device=dev)
            self.stats["eager"] += 1
            res = self._stage(st, lo, hi, has_pad, cal, want_conf, write_x)
            return res, (st["x_out"] if write_x else None)

    # ------------------------------------------------------------------ entry points
    def inputs_for(self, batch: dict, lengths: list[int], m: int, pad_id: int):
        """Padded stage-0 inputs from a collated (unpadded) batch, and has_pad."""
        ids = batch["input_ids"]
        n, w = ids.shape
        T = ceil_to(w, m)
        pid = pad_id if PAD_ID_OVERRIDE is None else PAD_ID_OVERRIDE
        pids = ids.new_full((n, T), pid)
        valid = torch.arange(w, device=ids.device).view(1, -1) < torch.tensor(lengths, device=ids.device).view(-1, 1)
        pids[:, :w] = torch.where(valid, ids, torch.full_like(ids, pid))
        inp = {"input_ids": pids, "lengths": torch.tensor(lengths, dtype=torch.long, device=ids.device)}
        for k in ("decision_index", "option_index", "option_span_start", "option_span_end",
                  "option_mask", "mode_id", "option_perm"):
            if k in batch:
                inp[k] = batch[k]
        return inp, any(int(L) < T for L in lengths)

    def fixed(self, batch: dict, lengths: list[int], m: int, pad_id: int) -> torch.Tensor:
        """DecisionModel.forward logits (presented order) of the padded batch: the main exit."""
        inp, has_pad = self.inputs_for(batch, lengths, m, pad_id)
        res, _ = self.run_stage(inp, 0, self.main, has_pad, None, False, False)
        return res["zf"]

    def staged(self, policy, batch: dict, lengths: list[int], m: int, pad_id: int):
        """rsijev.adaptive.staged_scores_fast on the padded batch, with the served
        finalizer: (logits presented order, depth (B,) long)."""
        inp, has_pad = self.inputs_for(batch, lengths, m, pad_id)
        B = int(inp["input_ids"].shape[0])
        dev = inp["input_ids"].device
        # exits a row can stop at (tau <= 1); an exit with tau <= 0 answers every row
        eff = []
        for L in policy.exits[:-1]:
            t = policy.tau_at(L)
            if t <= 1.0:
                eff.append(int(L))
                if t <= 0.0:
                    break
        if not eff or policy.tau_at(eff[-1]) > 0.0:
            eff.append(int(policy.exits[-1]))
        rows = list(range(B))
        out = None
        depth = torch.full((B,), int(policy.exits[-1]), dtype=torch.long, device=dev)
        lo = 0
        for L in eff:
            last = L == int(policy.exits[-1])
            tau = 2.0 if last else policy.tau_at(L)
            all_stop = last or tau <= 0.0
            cal = None if L == self.main else policy.cal[L]
            res, xo = self.run_stage(inp, lo, L, has_pad, cal, not all_stop, not all_stop)
            if out is None:
                out = torch.full((B, res["z"].shape[1]), float("-inf"), dtype=res["z"].dtype, device=dev)
            if all_stop:
                ridx = torch.tensor(rows, device=dev)
                out[ridx] = res["zf"]
                depth[ridx] = L
                break
            stop = (res["conf"] >= tau).tolist()                 # the one host read at this exit
            si = [k for k, s in enumerate(stop) if s]
            ki = [k for k, s in enumerate(stop) if not s]
            if si:
                sidx = torch.tensor(si, device=dev)
                ridx = torch.tensor([rows[k] for k in si], device=dev)
                out[ridx] = res["zf"][sidx]
                depth[ridx] = L
            if not ki:
                break
            kidx = torch.tensor(ki, device=dev)
            nxt = {k: v[kidx] for k, v in inp.items() if k not in ("input_ids", "x_in")}
            nxt["x_in"] = xo[kidx] if len(ki) < len(stop) else xo
            inp, rows, lo = nxt, [rows[k] for k in ki], L
        return out, depth


def runner(model) -> _Runner:
    r = getattr(model, "_rsijev_padded", None)
    if r is None:
        r = _Runner(model)
        object.__setattr__(model, "_rsijev_padded", r)
    return r


def stats(model) -> dict:
    r = getattr(model, "_rsijev_padded", None)
    return {} if r is None else {**r.stats, "graphs": len(r.graphs)}
