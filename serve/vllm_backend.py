"""Serve the tower through vLLM; keep the scorer and calibration in PyTorch.

    python scripts/serve.py --ckpt path/to/ckpt --backend vllm

vLLM runs only the Qwen3.5 tower, as a pooling model that returns per-token
hidden states (`PoolingParams(task="token_embed", use_activation=False)`, the
approach of mode-io/vllm-jev). Those rows are the final, normed hidden state --
exactly what `readout_layer: -1` reads -- and they go through the checkpoint's
own `DecisionModel._readout`: option-span mean pooling, the option_xattn scorer
and the fitted calibration, all unchanged and in fp32. The encoding
(`rsijev.encode`) is ours too, so vLLM only ever sees token ids.

What vLLM buys is concurrency: requests from many clients are batched into the
same engine steps instead of queueing behind one lock.

vLLM loads an HF-format directory, which `build_model_dir` writes once: the
checkpoint's fine-tuned tower (stored fp32 without the embedding) cast to bf16,
the base model's embedding, the base text config with architecture
`Qwen3_5ForCausalLM` (vLLM's text-only Qwen3.5, hybrid DeltaNet + attention),
and the base tokenizer files. bf16 is what the PyTorch serve path runs, and
the cast rounds each weight the same way `load_release(..., bf16)` does.

Prefix caching. vLLM 0.29 disables prefix caching by default for pooling
models on a hybrid architecture, and for `token_embed` requests it also sets
`skip_reading_prefix_cache=True`, because a cache hit returns rows only for the
tokens it computed. Here that is what we want: every row the readout needs
(option spans, decision token) lies after the shared state. So the engine is
started with `enable_prefix_caching=True, mamba_cache_mode="align"` and requests
set `skip_reading_prefix_cache=False`; a request whose returned rows do not
cover what the readout needs is re-run with cache reading off (this happens
when the same question about the same state is asked again).

What that buys is limited by the block size. On this hybrid model vLLM sizes an
attention block so its page holds one DeltaNet state (fp32), which is 544
tokens here, and a DeltaNet state is only cached at block boundaries. So the
questions of one request share floor(state / 544) * 544 tokens of their state,
not all of it: nothing for states under 544 tokens. vLLM finds that shared
prefix among requests submitted together by itself; running one question ahead
of the rest (`hidden_rows(..., lead=True)`) measured no faster.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from rsijev.arch import ArchConfig, DecisionModel

FORMAT = "rsijev-vllm-tower-v1"
ARCHITECTURE = "Qwen3_5ForCausalLM"
MANIFEST = "rsijev_vllm.json"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "chat_template.jinja", "special_tokens_map.json")
# Default engine limits. This box shares its memory between CPU and GPU, and
# vLLM's own default (90% of it) would take the machine; keep the pool small.
DEFAULT_GPU_MEMORY_UTILIZATION = 0.2
# A fixed KV + DeltaNet-state pool instead of one sized by profiling: vLLM's
# profiling asserts when another process on the machine frees memory meanwhile,
# and 8 GiB already holds ~300k tokens (64 sequences of 2,560 need 164k).
DEFAULT_KV_CACHE_GB = 8.0
DEFAULT_MAX_MODEL_LEN = 2560           # EncodeConfig.max_length is 2048
DEFAULT_MAX_NUM_SEQS = 64


def _quick_fingerprint(path: Path) -> str:
    """Size plus hashes of the first and last MiB: cheap, and enough to notice a
    model dir built from a different checkpoint."""
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, size - (1 << 20)))
        h.update(f.read(1 << 20))
    return h.hexdigest()


def _base_snapshot(base_model: str) -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(base_model, allow_patterns=[
        "config.json", "model.safetensors*", "*.json", "merges.txt", "*.jinja"]))


def build_model_dir(ckpt: str | Path, out: str | Path, *, dtype: str = "bf16",
                    base_dir: str | Path | None = None) -> Path:
    """Write the HF-format tower vLLM loads. Idempotent: an existing dir built
    from the same checkpoint and dtype is returned as is."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    ckpt, out = Path(ckpt), Path(out)
    meta = json.loads((ckpt / "meta.json").read_text())
    tower_path = ckpt / "tower.safetensors"
    fp = _quick_fingerprint(tower_path)
    if (out / MANIFEST).exists():
        man = json.loads((out / MANIFEST).read_text())
        if man.get("tower_fingerprint") == fp and man.get("dtype") == dtype:
            return out
        raise FileExistsError(f"{out} holds a different build; remove it first")
    torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    base = Path(base_dir) if base_dir else _base_snapshot(meta["base_model"])
    tmp = out.with_name(out.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        cfg = json.loads((base / "config.json").read_text())
        text = dict(cfg.get("text_config") or cfg)
        text["architectures"] = [ARCHITECTURE]
        text["dtype"] = {"bf16": "bfloat16", "fp32": "float32"}[dtype]
        text.setdefault("tie_word_embeddings", cfg.get("tie_word_embeddings", True))
        (tmp / "config.json").write_text(json.dumps(text, indent=2) + "\n")

        tensors = {}
        with safe_open(str(tower_path), framework="pt") as f:
            for k in f.keys():
                # Same rounding as load_state_dict into a bf16 module: one
                # round-to-nearest-even of the fp32 value.
                tensors["model." + k] = f.get_tensor(k).to(torch_dtype).contiguous()
        index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
        emb = "model.language_model.embed_tokens.weight"
        with safe_open(str(base / index[emb]), framework="pt") as f:
            tensors["model.embed_tokens.weight"] = f.get_tensor(emb).to(torch_dtype).contiguous()
        n_layers = text["num_hidden_layers"]
        have = {k.split(".")[2] for k in tensors if k.startswith("model.layers.")}
        if len(have) != n_layers or "model.norm.weight" not in tensors:
            raise ValueError(f"tower has layers {sorted(have)} and "
                             f"{'a' if 'model.norm.weight' in tensors else 'no'} final norm; "
                             f"the base config wants {n_layers}")
        save_file(tensors, str(tmp / "model.safetensors"), metadata={"format": "pt"})
        del tensors
        for name in TOKENIZER_FILES:
            if (base / name).is_file():
                shutil.copy2(base / name, tmp / name)
        (tmp / MANIFEST).write_text(json.dumps({
            "format": FORMAT, "architecture": ARCHITECTURE, "dtype": dtype,
            "source": ckpt.resolve().name, "base_model": meta["base_model"],
            "tower_fingerprint": fp, "readout_layer": meta["spec"]["readout_layer"],
        }, indent=2) + "\n")
        if out.exists():
            shutil.rmtree(out)
        tmp.replace(out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return out


def default_model_dir(ckpt: str | Path, dtype: str = "bf16") -> Path:
    root = Path(os.environ.get("RSIJEV_VLLM_CACHE",
                               Path.home() / ".cache" / "rsijev" / "vllm"))
    return root / f"{Path(ckpt).resolve().name}-{dtype}"


class VllmTower:
    """A vLLM AsyncLLM pooling engine on its own event-loop thread, callable from
    ordinary (threaded) code. Returns per-token final hidden states."""

    def __init__(self, model_dir: str | Path, *,
                 gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
                 kv_cache_gb: float | None = DEFAULT_KV_CACHE_GB,
                 max_model_len: int = DEFAULT_MAX_MODEL_LEN,
                 max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
                 max_num_batched_tokens: int | None = None,
                 prefix_caching: bool = True,
                 chunked_prefill: bool | None = None,
                 enforce_eager: bool = False,
                 dtype: str = "bfloat16",
                 engine_kwargs: dict | None = None):
        if gpu_memory_utilization > 0.25:
            raise ValueError("gpu_memory_utilization above 0.25 is refused: on a "
                             "unified-memory machine vLLM would take the host's RAM")
        from vllm import AsyncEngineArgs
        self.model_dir = Path(model_dir)
        self.prefix_caching = prefix_caching
        kw = dict(model=str(self.model_dir), runner="pooling", convert="embed",
                  pooler_config={"task": "token_embed"},
                  # Return the hidden states in the model dtype (bf16), not the
                  # fp32 pooling-head default: the values are the same, the
                  # transfer from the engine process is half the size.
                  hf_overrides={"head_dtype": "model"},
                  dtype=dtype, skip_tokenizer_init=True,
                  gpu_memory_utilization=gpu_memory_utilization,
                  max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                  enable_prefix_caching=prefix_caching, enforce_eager=enforce_eager)
        if kv_cache_gb is not None:
            kw["kv_cache_memory_bytes"] = int(kv_cache_gb * 2**30)
        if prefix_caching:
            kw["mamba_cache_mode"] = "align"
        if chunked_prefill is not None:
            kw["enable_chunked_prefill"] = chunked_prefill
        if max_num_batched_tokens is not None:
            kw["max_num_batched_tokens"] = max_num_batched_tokens
        kw.update(engine_kwargs or {})
        self.engine_kwargs = kw
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True,
                                        name="vllm-tower-loop")
        self._thread.start()
        self.engine = self._run(self._make(AsyncEngineArgs(**kw)))
        self.stats = {"requests": 0, "rerun_uncached": 0, "rows": 0, "prompt_tokens": 0}

    async def _make(self, args):
        from vllm.v1.engine.async_llm import AsyncLLM
        return AsyncLLM.from_engine_args(args)

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    async def _one(self, ids: list[int], read_cache: bool) -> torch.Tensor:
        from vllm import PoolingParams
        params = PoolingParams(task="token_embed", use_activation=False,
                               skip_reading_prefix_cache=not read_cache)
        rid = f"rsijev-{uuid.uuid4().hex}"
        final = None
        try:
            async for out in self.engine.encode({"prompt_token_ids": ids}, params, rid):
                final = out
        except BaseException:
            try:
                await self.engine.abort(rid)
            except Exception:
                pass
            raise
        if final is None or not final.finished:
            raise RuntimeError("vLLM returned no final pooling output")
        data = final.outputs.data
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        if data.ndim != 2 or not 1 <= data.shape[0] <= len(ids):
            raise RuntimeError(f"vLLM returned hidden states of shape {tuple(data.shape)} "
                               f"for {len(ids)} tokens")
        return data

    async def _rows(self, ids: list[int], need_from: int) -> tuple[int, torch.Tensor]:
        """(offset, rows): rows[i] is the hidden state at position offset + i,
        and offset <= need_from."""
        read = self.prefix_caching
        data = await self._one(ids, read)
        offset = len(ids) - data.shape[0]
        if offset > need_from:                 # the cache hit reached into the readout
            self.stats["rerun_uncached"] += 1
            data = await self._one(ids, False)
            offset = len(ids) - data.shape[0]
            if offset:
                raise RuntimeError("vLLM returned a partial sequence with cache reading off")
        self.stats["requests"] += 1
        self.stats["rows"] += data.shape[0]
        self.stats["prompt_tokens"] += len(ids)
        return offset, data

    async def _many(self, seqs, need_from, lead):
        start = 0
        if lead and len(seqs) > 1:
            first = await self._rows(seqs[0], need_from[0])
            start = 1
        rest = await asyncio.gather(*(self._rows(s, n)
                                      for s, n in zip(seqs[start:], need_from[start:])))
        return ([first] if start else []) + list(rest)

    def hidden_rows(self, seqs: Sequence[Sequence[int]], need_from: Sequence[int],
                    lead: bool = False) -> list[tuple[int, torch.Tensor]]:
        """[(offset, rows)] per sequence: rows[i] is the final hidden state at
        position offset + i, and offset <= need_from.

        All sequences go to the engine at once; `lead` runs the first one alone
        before the rest."""
        seqs = [list(map(int, s)) for s in seqs]
        return self._run(self._many(seqs, list(need_from), lead and len(seqs) > 1))

    def shutdown(self) -> None:
        engine, self.engine = self.engine, None
        if engine is None:
            return

        async def stop():                # on the engine's own loop, or its output
            engine.shutdown()            # handler reports the engine as dead
        try:
            self._run(stop())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)


class VllmDecisionModel(DecisionModel):
    """DecisionModel whose tower is a vLLM engine.

    `forward(**collate(...))` returns the same logits DecisionModel.forward
    would, computed from vLLM's hidden states by the same `_readout`, so every
    caller of the model -- `rsijev.evaluate.predict`, `serve.infer.score_questions`
    -- works on it unchanged. Only a final-layer readout is supported: that is
    the tensor a pooling model returns.
    """

    def __init__(self, tower: VllmTower, hidden: int, cfg: ArchConfig,
                 hidden_dtype: torch.dtype = torch.bfloat16):
        if cfg.readout_layer not in (-1,) or cfg.layer_mix or cfg.residual:
            raise ValueError("the vLLM backend reads the final normed hidden state only "
                             f"(readout_layer -1, no layer mix, no residual); got "
                             f"readout_layer={cfg.readout_layer!r}")
        super().__init__(nn.Module(), hidden, cfg)
        self.engine = tower
        self.hidden_dtype = hidden_dtype
        self._rsijev_numerics = "vllm"

    def encode_prefix(self, *a, **k):           # no HF cache to hand out
        raise NotImplementedError("the vLLM backend caches prefixes inside vLLM")

    extend_prefix = encode_prefix

    @torch.no_grad()
    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                decision_index: torch.Tensor, option_index=None, option_span_start=None,
                option_span_end=None, option_token_ids=None, option_mask=None,
                mode_id=None, option_perm=None, past_key_values=None,
                position_ids=None, **_):
        if past_key_values is not None:
            raise NotImplementedError("the vLLM backend takes whole sequences")
        dev = decision_index.device
        B, W = input_ids.shape
        lens = attention_mask.sum(1).tolist()
        ids_cpu = input_ids.cpu()
        seqs = [ids_cpu[r, :n].tolist() for r, n in enumerate(lens)]
        need = []
        for r in range(B):
            first = int(decision_index[r])
            if option_mask is not None:
                m = option_mask[r]
                src = option_span_start if (option_span_start is not None
                                            and self.cfg.option_pool == "mean") else option_index
                if src is not None and bool(m.any()):
                    first = min(first, int(src[r][m].min()))
            need.append(first)
        rows = self.engine.hidden_rows(seqs, need)
        h = torch.zeros((B, W, self.scorer_hidden()), dtype=self.hidden_dtype, device=dev)
        for r, (off, data) in enumerate(rows):
            h[r, off:lens[r]] = data.to(device=dev, dtype=self.hidden_dtype)
        b = torch.arange(B, device=dev)
        logits, _ = self._readout(h, h, b, decision_index, option_index,
                                  option_span_start, option_span_end, option_token_ids,
                                  option_mask, False, mode_id=mode_id, option_perm=option_perm)
        return logits

    def scorer_hidden(self) -> int:
        return self.cal_pca_mean.shape[0]


def load_vllm_release(ckpt: str | Path, device: str = "cuda", *,
                      model_dir: str | Path | None = None, dtype: str = "bf16",
                      **engine_kwargs):
    """`load_release` for the vLLM backend: (model, tokenizer, enc, meta)."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from rsijev.encode import EncodeConfig
    ckpt = Path(ckpt)
    meta = json.loads((ckpt / "meta.json").read_text())
    spec = meta["spec"]
    model_dir = Path(model_dir) if model_dir else default_model_dir(ckpt, dtype)
    build_model_dir(ckpt, model_dir, dtype=dtype)
    cfg = json.loads((model_dir / "config.json").read_text())
    arch = ArchConfig(readout=spec["readout"], readout_layer=spec["readout_layer"],
                      max_options=spec["max_options"], freeze_base=True,
                      option_pool=spec["option_pool"], residual=spec["residual"],
                      logit_cap=spec.get("logit_cap"),
                      head_input_norm=spec.get("head_input_norm", False),
                      **dict(spec.get("arch_extra") or {}))
    tok = AutoTokenizer.from_pretrained(meta["base_model"])
    tower = VllmTower(model_dir, dtype={"bf16": "bfloat16", "fp32": "float32"}[dtype],
                      **engine_kwargs)
    model = VllmDecisionModel(tower, cfg["hidden_size"], arch,
                              hidden_dtype={"bf16": torch.bfloat16,
                                            "fp32": torch.float32}[dtype]).to(device)
    model.scorer.load_state_dict(load_file(str(ckpt / "scorer.safetensors")))
    if (ckpt / "calibration.safetensors").exists():
        from rsijev.calibrate import load_calibration
        meta["calibration"] = load_calibration(model, ckpt)
    else:
        meta["calibration"] = "none"
    model.scorer.to(torch.float32)
    model.eval()
    meta["backend"] = {"name": "vllm", "model_dir": str(model_dir),
                       "engine": {k: v for k, v in tower.engine_kwargs.items()
                                  if k != "model"}}
    enc = EncodeConfig(layout=spec["layout"], option_pool=spec["option_pool"],
                       option_order="canonical")
    return model, tok, enc, meta


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="build the HF-format tower dir vLLM loads")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    a = ap.parse_args()
    out = build_model_dir(a.ckpt, a.out or default_model_dir(a.ckpt, a.dtype), dtype=a.dtype)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
