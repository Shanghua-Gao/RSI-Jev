"""The Qwen3.5 text tower in MLX: gated DeltaNet x3 + gated full attention, repeated.

A port of transformers' `Qwen3_5TextModel` (5.17) as the released PyTorch path runs it in
bf16, op for op where precision is decided:

  * RMSNorm (zero-centred weight): normalise in fp32, multiply by (1 + w) in fp32, cast back
  * gated RMSNorm (DeltaNet output): normalise in fp32, cast, * w in the tower dtype, * silu(z)
    in fp32, cast back
  * the DeltaNet core (`chunk_gated_delta`) in fp32 from tower-dtype q/k/v: l2-normalised
    q/k, the chunked delta rule (chunk 64) of transformers' torch_chunk_gated_delta_rule; the
    causal depthwise conv1d accumulates in fp32 and rounds once, as a bf16 conv does
  * rotary (M-RoPE, interleaved sections, partial factor 0.25): cos/sin in fp32, cast to the
    tower dtype, applied in that dtype. Positions are (3, B, T): equal rows for text, the
    image grid for image tokens (`rsijev.mlx.vision.mrope_positions`)
  * SiLU in fp32, rounded once (what torch's bf16 silu does)
  * attention: mx.fast.scaled_dot_product_attention (causal, grouped KV heads)

The layers run in stages (`run(h, lo, hi, ...)`), so exits 16 / 20 / 32 read the residual
stream between stages; the shared final RMSNorm is `norm`. Rows are right-padded, so under
the causal mask no pad token reaches a real one and no padding mask is needed.

`run(..., cache=PrefixCache)` continues every row from a shared prefix: per layer the
attention keys/values and the DeltaNet conv window + recurrent state of the prefix, broadcast
over the rows (serve/infer.py's cached path, in PyTorch). `run(..., record=PrefixCache())`
fills one.

The DeltaNet recurrence runs in chunked form (`chunk_gated_delta`) everywhere; on an Apple
GPU with mlx-lm installed, RSIJEV_MLX_DELTA=kernel uses mlx-lm's Metal kernel instead (to be
measured on a Mac: speed and agreement).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import mlx.core as mx

F32 = mx.float32


@dataclass
class TextConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    layer_types: list = field(default_factory=list)
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    mrope_section: tuple = (11, 11, 10)
    mrope_interleaved: bool = True

    @classmethod
    def from_config(cls, cfg: dict) -> "TextConfig":
        tc = cfg.get("text_config", cfg)
        rp = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
        n = int(tc["num_hidden_layers"])
        every = int(tc.get("full_attention_interval", 4))
        types = list(tc.get("layer_types") or [
            "full_attention" if (i + 1) % every == 0 else "linear_attention" for i in range(n)])
        return cls(hidden_size=tc["hidden_size"], intermediate_size=tc["intermediate_size"],
                   num_hidden_layers=n, num_attention_heads=tc["num_attention_heads"],
                   num_key_value_heads=tc["num_key_value_heads"],
                   head_dim=tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"],
                   linear_num_key_heads=tc["linear_num_key_heads"],
                   linear_num_value_heads=tc["linear_num_value_heads"],
                   linear_key_head_dim=tc["linear_key_head_dim"],
                   linear_value_head_dim=tc["linear_value_head_dim"],
                   linear_conv_kernel_dim=tc["linear_conv_kernel_dim"],
                   vocab_size=tc["vocab_size"], rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
                   layer_types=types,
                   rope_theta=float(rp.get("rope_theta", tc.get("rope_theta", 10_000_000.0))),
                   partial_rotary_factor=float(rp.get("partial_rotary_factor", 0.25)),
                   mrope_section=tuple(rp.get("mrope_section", (11, 11, 10))),
                   mrope_interleaved=bool(rp.get("mrope_interleaved", True)))

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def layer_period(self) -> int:
        t = self.layer_types
        for p in range(1, len(t) + 1):
            if all(t[i] == t[i % p] for i in range(len(t))):
                return p
        return len(t)


# ------------------------------------------------------------------------------- numerics
def rms_norm(x: mx.array, w1: mx.array, eps: float) -> mx.array:
    """Qwen3.5 RMSNorm; `w1` is (1 + w) in fp32 (made at load)."""
    return mx.fast.rms_norm(x.astype(F32), w1, eps).astype(x.dtype)


def silu32(x: mx.array) -> mx.array:
    """silu in fp32, rounded once to x's dtype."""
    y = x.astype(F32)
    return (y * mx.sigmoid(y)).astype(x.dtype)


def softplus(x: mx.array) -> mx.array:
    """torch.nn.functional.softplus (beta 1, threshold 20)."""
    return mx.where(x > 20, x, mx.log1p(mx.exp(mx.minimum(x, 20.0))))


def text_positions(B: int, T: int, start: int = 0) -> mx.array:
    p = mx.arange(start, start + T, dtype=mx.int32)
    return mx.broadcast_to(p[None, None, :], (3, B, T))


def rotary_cos_sin(positions: mx.array, cfg: TextConfig, dtype) -> tuple[mx.array, mx.array]:
    """positions (3, B, T) -> cos, sin (B, T, rotary_dim) in `dtype`.

    transformers' Qwen3_5TextRotaryEmbedding: frequencies in fp32 per axis, frequency i taken
    from axis h for i = 1, 4, ... < 3 * section[1], from axis w for i = 2, 5, ... <
    3 * section[2], from axis t otherwise; then [f, f], cos/sin, cast."""
    if not cfg.mrope_interleaved:
        raise NotImplementedError("only interleaved M-RoPE (Qwen3.5) is ported")
    dim = cfg.rotary_dim
    half = dim // 2
    inv = 1.0 / (cfg.rope_theta ** (mx.arange(0, dim, 2, dtype=F32) / dim))
    freqs = positions.astype(F32)[..., None] * inv                    # (3, B, T, half)
    axis = [0] * half
    for a, offset in ((1, 1), (2, 2)):
        for i in range(offset, min(cfg.mrope_section[a] * 3, half), 3):
            axis[i] = a
    onehot = mx.array([[1.0 if axis[i] == a else 0.0 for i in range(half)] for a in range(3)],
                      dtype=F32)                                       # (3, half)
    f = (freqs * onehot[:, None, None, :]).sum(0)                      # exact: one non-zero term
    emb = mx.concatenate([f, f], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x (B, H, T, D); cos/sin (B, T, r): rotate the first r dims, in x's dtype."""
    r = cos.shape[-1]
    c, s = cos[:, None], sin[:, None]
    xr = x[..., :r]
    out = xr * c + _rotate_half(xr) * s
    return out if r == x.shape[-1] else mx.concatenate([out, x[..., r:]], axis=-1)


def _unit_lower_inverse(S: mx.array) -> mx.array:
    """(I + S)^-1 for strictly lower-triangular S (..., C, C), C a power of two, by recursive
    2x2 block inversion: inv([[A, 0], [B, D]]) = [[A^-1, 0], [-D^-1 B A^-1, D^-1]], both
    diagonal blocks inverted together at each level. As stable as forward substitution (what
    torch.linalg.solve_triangular computes); a Neumann product (I - S)(I + S^2)... is not:
    its powers of S overflow on real DeltaNet inputs."""
    C = S.shape[-1]
    if C & (C - 1):
        raise ValueError(f"chunk size {C} is not a power of two")
    L = S + mx.eye(C, dtype=S.dtype)
    lead = L.shape[:-2]
    # blocks: (..., nb, b, b) diagonal blocks of the current size, starting at 1x1 (= 1)
    inv = mx.ones((*lead, C, 1, 1), dtype=S.dtype)
    b = 1
    while b < C:
        # the (2b x 2b) diagonal blocks of L, split into [[A, 0], [Bm, D]]
        blocks = mx.stack([L[..., i:i + 2 * b, i:i + 2 * b] for i in range(0, C, 2 * b)], axis=-3)
        Bm = blocks[..., b:, :b]
        Ai = inv[..., 0::2, :, :]
        Di = inv[..., 1::2, :, :]
        X = -(Di @ Bm @ Ai)
        z = mx.zeros_like(Ai)
        top = mx.concatenate([Ai, z], axis=-1)
        bot = mx.concatenate([X, Di], axis=-1)
        inv = mx.concatenate([top, bot], axis=-2)
        b *= 2
    return inv[..., 0, :, :]


def chunk_gated_delta(q, k, v, g, beta, state=None, chunk: int = 64):
    """transformers' torch_chunk_gated_delta_rule (use_qk_l2norm_in_kernel=True) in MLX.

    q, k (B, T, H, Dk), v (B, T, H, Dv) in the tower dtype, heads already repeated to H;
    g (B, T, H) fp32 log-decay; beta (B, T, H). state (B, H, Dk, Dv) fp32 or None.
    Returns (out (B, T, H, Dv) in q's dtype, final state (B, H, Dk, Dv) fp32)."""
    in_dtype = q.dtype
    B, T, H, Dk = k.shape
    Dv = v.shape[-1]
    q, k, v, beta, g = (x.astype(F32).swapaxes(1, 2) for x in (q, k, v, beta, g))
    q = q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6)
    k = k * mx.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    q = q * (Dk ** -0.5)
    pad = (chunk - T % chunk) % chunk
    if pad:
        q, k, v = (mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)]) for x in (q, k, v))
        beta, g = (mx.pad(x, [(0, 0), (0, 0), (0, pad)]) for x in (beta, g))
    N = (T + pad) // chunk
    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, k_beta, v_beta = (x.reshape(B, H, N, chunk, x.shape[-1]) for x in (q, k, k_beta, v_beta))
    g = g.reshape(B, H, N, chunk)
    cum = mx.cumsum(g, axis=-1)
    lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    decay = mx.exp(mx.where(lower, cum[..., :, None] - cum[..., None, :], -mx.inf))
    kt = k.swapaxes(-1, -2)
    ut = (k_beta @ kt) * decay
    attn = (q @ kt) * decay
    inv = _unit_lower_inverse(mx.tril(ut, -1))
    ecum = mx.exp(cum)
    new_values = inv @ v_beta
    k_cumdecay = inv @ (k_beta * ecum[..., None])
    qd = q * ecum[..., None]
    kd = k * mx.exp(cum[..., -1:] - cum)[..., None]
    chunk_decay = mx.exp(cum[..., -1])[..., None, None]
    st = mx.zeros((B, H, Dk, Dv), dtype=F32) if state is None else state.astype(F32)
    outs = []
    for i in range(N):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ st
        outs.append(qd[:, :, i] @ st + attn[:, :, i] @ v_new)
        st = st * chunk_decay[:, :, i] + kd[:, :, i].swapaxes(-1, -2) @ v_new
    out = outs[0] if N == 1 else mx.concatenate(outs, axis=2)
    return out[:, :, :T].swapaxes(1, 2).astype(in_dtype), st


def recurrent_gated_delta(q, k, v, g, beta, state=None):
    """The same rule one token at a time (fp32): the reference the chunked form is tested
    against, and what mlx-lm's Metal kernel computes."""
    in_dtype = q.dtype
    B, T, H, Dk = k.shape
    q, k, v, beta, g = (x.astype(F32) for x in (q, k, v, beta, g))
    q = q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6) * (Dk ** -0.5)
    k = k * mx.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    st = mx.zeros((B, H, Dk, v.shape[-1]), dtype=F32) if state is None else state.astype(F32)
    ys = []
    for t in range(T):
        st = st * mx.exp(g[:, t])[..., None, None]
        kv = (st * k[:, t, :, :, None]).sum(-2)                         # (B, H, Dv)
        delta = (v[:, t] - kv) * beta[:, t, :, None]
        st = st + k[:, t, :, :, None] * delta[:, :, None, :]
        ys.append((st * q[:, t, :, :, None]).sum(-2))
    return mx.stack(ys, axis=1).astype(in_dtype), st


def kernel_gated_delta(q, k, v, g, beta, state=None):
    """mlx-lm's gated-delta recurrence (its Metal kernel on an Apple GPU), same contract as
    chunk_gated_delta. mlx-lm keeps the state as (B, H, Dv, Dk)."""
    from mlx_lm.models.gated_delta import gated_delta_kernel
    in_dtype = q.dtype
    Dk = k.shape[-1]
    q32, k32 = q.astype(F32), k.astype(F32)
    q32 = q32 * mx.rsqrt((q32 * q32).sum(-1, keepdims=True) + 1e-6) * (Dk ** -0.5)
    k32 = k32 * mx.rsqrt((k32 * k32).sum(-1, keepdims=True) + 1e-6)
    B, T, H, _ = k.shape
    st = (mx.zeros((B, H, v.shape[-1], Dk), dtype=F32) if state is None
          else state.astype(F32).swapaxes(-1, -2))
    y, st = gated_delta_kernel(q32.astype(in_dtype), k32.astype(in_dtype), v, mx.exp(g),
                               beta.astype(in_dtype), st)
    return y.astype(in_dtype), st.swapaxes(-1, -2)


def delta_backend() -> str:
    """chunk (default) or kernel (RSIJEV_MLX_DELTA=kernel: Apple GPU + mlx-lm only)."""
    want = os.environ.get("RSIJEV_MLX_DELTA", "chunk").strip().lower() or "chunk"
    if want not in ("chunk", "kernel"):
        raise ValueError(f"RSIJEV_MLX_DELTA must be chunk or kernel, got {want!r}")
    if want == "kernel" and not (mx.metal.is_available() and mx.default_device() == mx.gpu):
        raise RuntimeError("RSIJEV_MLX_DELTA=kernel needs an Apple GPU (Metal)")
    return want


# ------------------------------------------------------------------------------- weights
class Linear:
    """y = x W^T (+ b); W either a plain array or MLX affine-quantized (codes/scales/biases)."""
    __slots__ = ("w", "scales", "biases", "b", "group", "bits")

    def __init__(self, w, b=None, scales=None, biases=None, group=None, bits=None):
        self.w, self.b, self.scales, self.biases = w, b, scales, biases
        self.group, self.bits = group, bits

    @classmethod
    def from_params(cls, p: dict, name: str, group: int | None = None) -> "Linear":
        w = p[f"{name}.weight"]
        if f"{name}.scales" not in p:
            return cls(w, p.get(f"{name}.bias"))
        s = p[f"{name}.scales"]
        in_features = s.shape[-1] * group
        bits = w.shape[-1] * 32 // in_features
        return cls(w, p.get(f"{name}.bias"), s, p[f"{name}.biases"], group, bits)

    @property
    def quantized(self) -> bool:
        return self.scales is not None

    def __call__(self, x):
        if self.scales is not None:
            y = mx.quantized_matmul(x, self.w, self.scales, self.biases, transpose=True,
                                    group_size=self.group, bits=self.bits)
        else:
            y = x @ self.w.T
        return y if self.b is None else y + self.b


class Embedding:
    __slots__ = ("w", "scales", "biases", "group", "bits", "dtype")

    def __init__(self, p: dict, group: int | None, dtype):
        self.w = p["embed_tokens.weight"]
        self.scales = p.get("embed_tokens.scales")
        self.biases = p.get("embed_tokens.biases")
        self.group, self.dtype = group, dtype
        self.bits = None
        if self.scales is not None:
            self.bits = self.w.shape[-1] * 32 // (self.scales.shape[-1] * group)

    def __call__(self, ids: mx.array) -> mx.array:
        if self.scales is None:
            return self.w[ids].astype(self.dtype)
        return mx.dequantize(self.w[ids], self.scales[ids], self.biases[ids],
                             group_size=self.group, bits=self.bits).astype(self.dtype)


# ------------------------------------------------------------------------------- layers
@dataclass
class LayerCache:
    """One layer's state over a prefix: keys/values (attention, (1, Hkv, P, D)) or the
    DeltaNet conv window ((1, K-1, C), pre-activation inputs) and recurrent state
    ((1, H, Dk, Dv) fp32)."""
    keys: mx.array | None = None
    values: mx.array | None = None
    conv: mx.array | None = None
    state: mx.array | None = None


@dataclass
class PrefixCache:
    layers: dict = field(default_factory=dict)       # layer index -> LayerCache
    length: int = 0

    def nbytes(self) -> int:
        n = 0
        for c in self.layers.values():
            for a in (c.keys, c.values, c.conv, c.state):
                if a is not None:
                    n += a.nbytes
        return n


class DeltaNet:
    def __init__(self, cfg: TextConfig, p: dict, pre: str, group, dtype):
        self.cfg, self.dtype = cfg, dtype
        self.nk, self.nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.key_dim, self.value_dim = self.nk * self.dk, self.nv * self.dv
        self.K = cfg.linear_conv_kernel_dim
        L = lambda n: Linear.from_params(p, f"{pre}.{n}", group)      # noqa: E731
        self.in_proj_qkv, self.in_proj_z = L("in_proj_qkv"), L("in_proj_z")
        self.in_proj_b, self.in_proj_a, self.out_proj = L("in_proj_b"), L("in_proj_a"), L("out_proj")
        cw = p[f"{pre}.conv1d.weight"]                                  # (C, 1, K) torch layout
        self.conv_w = cw.reshape(cw.shape[0], self.K).astype(F32).T     # (K, C) fp32 of the bf16 weight
        self.A = -mx.exp(p[f"{pre}.A_log"].astype(F32))
        self.dt_bias = p[f"{pre}.dt_bias"].astype(F32)
        self.norm_w = p[f"{pre}.norm.weight"].astype(dtype)
        self.eps = cfg.rms_norm_eps

    def __call__(self, x, cache: LayerCache | None = None, record: LayerCache | None = None):
        B, T, _ = x.shape
        qkv = self.in_proj_qkv(x)                                        # (B, T, C)
        z = self.in_proj_z(x).reshape(B, T, self.nv, self.dv)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)
        K = self.K
        if cache is not None and cache.conv is not None:
            win = mx.broadcast_to(cache.conv, (B, K - 1, qkv.shape[-1]))
        else:
            win = mx.zeros((B, K - 1, qkv.shape[-1]), dtype=qkv.dtype)
        xs = mx.concatenate([win, qkv], axis=1)
        if record is not None:
            record.conv = xs[:, -(K - 1):]
        xf = xs.astype(F32)
        conv = xf[:, 0:T] * self.conv_w[0]
        for j in range(1, K):
            conv = conv + xf[:, j:j + T] * self.conv_w[j]
        conv = conv.astype(qkv.dtype)
        conv = silu32(conv)
        q = conv[..., :self.key_dim].reshape(B, T, self.nk, self.dk)
        k = conv[..., self.key_dim:2 * self.key_dim].reshape(B, T, self.nk, self.dk)
        v = conv[..., 2 * self.key_dim:].reshape(B, T, self.nv, self.dv)
        beta = mx.sigmoid(b.astype(F32)).astype(b.dtype)
        g = self.A * softplus(a.astype(F32) + self.dt_bias)
        rep = self.nv // self.nk
        if rep > 1:
            q = mx.repeat(q, rep, axis=2)
            k = mx.repeat(k, rep, axis=2)
        state = None
        if cache is not None and cache.state is not None:
            state = mx.broadcast_to(cache.state, (B, *cache.state.shape[1:]))
        fn = kernel_gated_delta if delta_backend() == "kernel" else chunk_gated_delta
        out, st = fn(q, k, v, g, beta, state)
        if record is not None:
            record.state = st
        # gated RMSNorm: normalise in fp32, cast, * w (tower dtype), * silu(z) in fp32, cast
        o = out.astype(F32)
        o = (o * mx.rsqrt((o * o).mean(-1, keepdims=True) + self.eps)).astype(self.dtype)
        o = (self.norm_w * o).astype(F32)
        zf = z.astype(F32)
        o = (o * (zf * mx.sigmoid(zf))).astype(self.dtype)
        return self.out_proj(o.reshape(B, T, self.value_dim))


class Attention:
    def __init__(self, cfg: TextConfig, p: dict, pre: str, group, dtype):
        self.cfg, self.dtype = cfg, dtype
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        L = lambda n: Linear.from_params(p, f"{pre}.{n}", group)      # noqa: E731
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = L("q_proj"), L("k_proj"), L("v_proj"), L("o_proj")
        self.q_norm = 1.0 + p[f"{pre}.q_norm.weight"].astype(F32)
        self.k_norm = 1.0 + p[f"{pre}.k_norm.weight"].astype(F32)
        self.eps = cfg.rms_norm_eps
        self.scale = self.hd ** -0.5

    def __call__(self, x, cos, sin, cache: LayerCache | None = None, record: LayerCache | None = None):
        B, T, _ = x.shape
        qg = self.q_proj(x).reshape(B, T, self.nh, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:]
        gate = gate.reshape(B, T, self.nh * self.hd)
        q = rms_norm(q, self.q_norm, self.eps).transpose(0, 2, 1, 3)
        k = rms_norm(self.k_proj(x).reshape(B, T, self.nkv, self.hd), self.k_norm, self.eps).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.nkv, self.hd).transpose(0, 2, 1, 3)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if record is not None:
            record.keys, record.values = k, v
        if cache is not None and cache.keys is not None:
            P = cache.keys.shape[2]
            k = mx.concatenate([mx.broadcast_to(cache.keys, (B, self.nkv, P, self.hd)), k], axis=2)
            v = mx.concatenate([mx.broadcast_to(cache.values, (B, self.nkv, P, self.hd)), v], axis=2)
            # the T new queries sit at the end: query i sees keys 0..P+i
            mask = mx.arange(P + T)[None, :] <= (mx.arange(T)[:, None] + P)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        else:
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        o = o.transpose(0, 2, 1, 3).reshape(B, T, self.nh * self.hd)
        return self.o_proj(o * mx.sigmoid(gate))


class MLP:
    def __init__(self, p: dict, pre: str, group):
        L = lambda n: Linear.from_params(p, f"{pre}.{n}", group)      # noqa: E731
        self.gate_proj, self.up_proj, self.down_proj = L("gate_proj"), L("up_proj"), L("down_proj")

    def __call__(self, x):
        return self.down_proj(silu32(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer:
    def __init__(self, cfg: TextConfig, p: dict, i: int, group, dtype):
        pre = f"layers.{i}"
        self.linear = cfg.layer_types[i] == "linear_attention"
        self.mixer = (DeltaNet(cfg, p, f"{pre}.linear_attn", group, dtype) if self.linear
                      else Attention(cfg, p, f"{pre}.self_attn", group, dtype))
        self.mlp = MLP(p, f"{pre}.mlp", group)
        self.ln1 = 1.0 + p[f"{pre}.input_layernorm.weight"].astype(F32)
        self.ln2 = 1.0 + p[f"{pre}.post_attention_layernorm.weight"].astype(F32)
        self.eps = cfg.rms_norm_eps

    def __call__(self, x, cos, sin, cache=None, record=None):
        h = rms_norm(x, self.ln1, self.eps)
        if self.linear:
            h = self.mixer(h, cache, record)
        else:
            h = self.mixer(h, cos, sin, cache, record)
        x = x + h
        return x + self.mlp(rms_norm(x, self.ln2, self.eps))


class TextTower:
    """Embeddings, decoder layers [0, n_layers) and the shared final norm."""

    def __init__(self, cfg: TextConfig, params: dict, *, group: int | None = None,
                 dtype=mx.bfloat16, n_layers: int | None = None, embed_group: int | None = None):
        self.cfg, self.dtype = cfg, dtype
        n = n_layers or cfg.num_hidden_layers
        self.embed_tokens = Embedding(params, embed_group or group, dtype)
        self.layers = [DecoderLayer(cfg, params, i, group, dtype) for i in range(n)]
        self.norm_w = 1.0 + params["norm.weight"].astype(F32)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def embed(self, ids: mx.array) -> mx.array:
        return self.embed_tokens(ids)

    def norm(self, h: mx.array) -> mx.array:
        return rms_norm(h, self.norm_w, self.cfg.rms_norm_eps)

    def rotary(self, positions: mx.array):
        return rotary_cos_sin(positions, self.cfg, self.dtype)

    def run(self, h: mx.array, lo: int, hi: int, rope, *, cache: PrefixCache | None = None,
            record: PrefixCache | None = None) -> mx.array:
        """Decoder layers lo..hi-1 on h (B, T, D); un-normed output. `rope` = rotary(positions)."""
        cos, sin = rope
        for i in range(lo, hi):
            c = cache.layers.get(i) if cache is not None else None
            r = None
            if record is not None:
                r = record.layers.setdefault(i, LayerCache())
            h = self.layers[i](h, cos, sin, c, r)
        if record is not None:
            record.length = h.shape[1] + (cache.length if cache is not None else 0)
        return h
