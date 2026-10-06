"""EDITABLE (model axis). Image states: Qwen3.5's own vision tower feeding the
text tower, in the same single decision pass.

Qwen3.5-*-Base is `Qwen3_5ForConditionalGeneration`: the checkpoint carries a
24-block ViT (patch 16, 2x2 spatial merge, output = the text hidden size) at
`model.visual.*`. `AutoModelForCausalLM` drops it on load, which is why the
text-only releases never had it. This module loads it back beside the text
tower and routes image features into the SAME decision pass:

  state "<image>\\nWhat is ..."  ->  "<|vision_start|><|image_pad|>*N<|vision_end|>\\n..."
  ids -> embed_tokens -> image_pad rows replaced by visual(pixels) -> text tower
  -> the unchanged option scorer.

Nothing is generated; the vision tower adds one encoder pass per image.

Position ids are Qwen's multimodal RoPE (3 rows: t, h, w). A batch without
images takes the text path exactly (no position_ids, no inputs_embeds).

A state holds one literal "<image>" per image, where that image goes, or none
(the images are then put before the state, in order). encode.py and contract.py
are untouched: the expanded state is an ordinary string whose image run is
tokenised as N special tokens.

Ported from the release code the vision releases were trained and gated with.
What is new here only serves requests faster
and does not change a number: `prepare` runs the image processor once per
request instead of once per question, `VisionDecisionModel` takes
precomputed `image_embeds` so the ViT runs once per request as well, and
`encode_image_prefix` runs the shared state (image tokens included) once so
every question continues from its cache (serve/infer.py).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
import torch.nn as nn

from .arch import DecisionModel
from .contract import Question
from .encode import EncodeConfig, collate, encode_question

IMAGE_MARKER = "<image>"
VISION_START, IMAGE_PAD, VISION_END = "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"
# Qwen's multimodal special tokens. A state may not spell them itself: they
# would tokenise as the real tokens and be counted as image positions.
VISION_SPECIAL_TOKENS = (VISION_START, IMAGE_PAD, VISION_END, "<|video_pad|>")


# The base-model snapshot each vision release was verified against. The vision tower's
# weights and the image processor come from the base repo, not from the release, so
# without a pinned revision an upstream change to that repo would change answers
# silently. A release's meta.json may name its own (`vision.revision`).
# 4B: the snapshot the 4B exit models (b4-exit20, rt5-4b-x20, vis-v4k) were trained
# and gated on.
PINNED_REVISIONS = {"Qwen/Qwen3.5-2B-Base": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
                    "Qwen/Qwen3.5-4B-Base": "1001bb4d826a52d1f399e183466143f4da7b741b"}


def vision_revision(block: dict | None, model_id: str) -> str | None:
    """The base-model revision for the vision tower and image processor: the release's
    own `revision` if its vision block has one, else the pinned one for `model_id`."""
    return (block or {}).get("revision") or PINNED_REVISIONS.get(model_id)


@dataclass
class VisionConfig:
    # One LLM token covers a 32x32 pixel block (patch 16, 2x2 merge). The budget is
    # per DECISION and is split evenly over its images, so 4 images at 1024 tokens
    # total cost what 1 image at 1024 does in the text tower.
    image_token_budget: int = 1024
    min_tokens_per_image: int = 64
    freeze_visual: bool = True


def vision_block(meta: dict) -> dict | None:
    """The vision settings a checkpoint was built with, or None for a text model."""
    rel = (meta.get("release") or {}).get("vision")
    if rel:
        return rel
    return ((meta.get("spec") or {}).get("fit_extra") or {}).get("vision")


class ImagePrep:
    """PIL images -> (pixel_values, grid_thw, llm tokens per image), budgeted."""

    def __init__(self, model_id: str, cfg: VisionConfig, revision: str | None = None):
        from transformers import AutoImageProcessor
        self.model_id, self.cfg = model_id, cfg
        self.revision = revision if revision is not None else PINNED_REVISIONS.get(model_id)
        self._procs: dict[int, object] = {}
        self._base = AutoImageProcessor.from_pretrained(model_id, revision=self.revision)
        self.unit = self._base.patch_size * self._base.merge_size      # 32 px

    def tokens_per_image(self, n_images: int) -> int:
        """The most LLM tokens one of `n_images` images may take."""
        return max(self.cfg.min_tokens_per_image, self.cfg.image_token_budget // max(1, n_images))

    def _proc(self, max_tokens: int):
        if max_tokens not in self._procs:
            from transformers import AutoImageProcessor
            u2 = self.unit * self.unit
            mn = min(self.cfg.min_tokens_per_image, max_tokens) * u2
            self._procs[max_tokens] = AutoImageProcessor.from_pretrained(
                self.model_id, revision=self.revision, size={"shortest_edge": mn, "longest_edge": max_tokens * u2})
        return self._procs[max_tokens]

    def __call__(self, images: Sequence) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        per = self.tokens_per_image(len(images))
        out = self._proc(per)([im.convert("RGB") for im in images], return_tensors="pt")
        grid = out["image_grid_thw"]
        m = self._base.merge_size
        return out["pixel_values"], grid, [int(g.prod()) // (m * m) for g in grid]


def expand_state(state: str, n_tokens: Sequence[int]) -> str:
    runs = [VISION_START + IMAGE_PAD * n + VISION_END for n in n_tokens]
    k = state.count(IMAGE_MARKER)
    if k == 0:
        return "\n".join(runs) + ("\n" + state if state.strip() else "")
    if k != len(runs):
        raise ValueError(f"state has {k} {IMAGE_MARKER} markers for {len(runs)} images")
    parts = state.split(IMAGE_MARKER)
    return "".join(p + (runs[i] if i < len(runs) else "") for i, p in enumerate(parts))


def encode_vision_question(tokenizer, prep: ImagePrep, state: str, images: Sequence,
                           q: Question, enc: EncodeConfig,
                           rng: random.Random | None = None, prepared=None) -> dict:
    """encode_question on the expanded state, plus the pixels. No image may be cut.

    `prepared` is `prep(images)` computed once for every question of a request."""
    if not images:
        return encode_question(tokenizer, state, q, enc, rng=rng)
    pv, grid, ntok = prepared if prepared is not None else prep(images)
    e = encode_question(tokenizer, expand_state(state, ntok), q, enc, rng=rng)
    pad = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    got = sum(1 for t in e["input_ids"] if t == pad)
    if got != sum(ntok):
        raise ValueError(f"{q.key}: {got} image tokens survive of {sum(ntok)} "
                         f"(max_length {enc.max_length} cut into an image; lower the budget)")
    e["pixel_values"], e["image_grid_thw"] = pv, grid
    return e


def mrope_position_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       grid_thw: torch.Tensor, image_token_id: int,
                       spatial_merge_size: int = 2) -> torch.Tensor:
    """(3, B, T) M-RoPE positions, computed by HF's own Qwen3_5Model.get_rope_index."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    shim = SimpleNamespace(config=SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size)))
    shim.get_vision_position_ids = lambda *a, **k: Qwen3_5Model.get_vision_position_ids(shim, *a, **k)
    pos, _ = Qwen3_5Model.get_rope_index(
        shim, input_ids, mm_token_type_ids=(input_ids == image_token_id).int(),
        image_grid_thw=grid_thw, attention_mask=attention_mask)
    return pos


def vision_collate(tokenizer, examples: Sequence[dict], max_options: int,
                   device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    batch = collate(tokenizer, examples, max_options, device="cpu")
    imgs = [e for e in examples if "pixel_values" in e]
    if imgs:
        grid = torch.cat([e["image_grid_thw"] for e in imgs])
        batch["pixel_values"] = torch.cat([e["pixel_values"] for e in imgs])
        batch["image_grid_thw"] = grid
        batch["position_ids"] = mrope_position_ids(
            batch["input_ids"], batch["attention_mask"], grid,
            tokenizer.convert_tokens_to_ids(IMAGE_PAD))
    return {k: v.to(device) for k, v in batch.items()}


def load_visual(model_id: str, dtype=torch.bfloat16, revision: str | None = None) -> nn.Module:
    """The checkpoint's vision tower (ViT + merger), weights from model.visual.*, at
    `revision` (default: the pinned one for `model_id`, see PINNED_REVISIONS)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    local = Path(model_id).is_dir() and (Path(model_id) / "visual.safetensors").exists()
    if revision is None and not local:
        revision = PINNED_REVISIONS.get(model_id)
    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    vc = cfg.vision_config
    vc._attn_implementation = "sdpa"
    visual = Qwen3_5VisionModel(vc)
    sd = {}
    if local:      # a self-contained release: visual.safetensors, keys without "model.visual."
        with safe_open(str(Path(model_id) / "visual.safetensors"), framework="pt") as fh:
            sd = {k: fh.get_tensor(k) for k in fh.keys()}
    else:
        idx = json.loads(Path(hf_hub_download(model_id, "model.safetensors.index.json",
                                                        revision=revision)).read_text())
        files = sorted({f for k, f in idx["weight_map"].items() if k.startswith("model.visual.")})
        for f in files:
            with safe_open(hf_hub_download(model_id, f, revision=revision), framework="pt") as fh:
                for k in fh.keys():
                    if k.startswith("model.visual."):
                        sd[k[len("model.visual."):]] = fh.get_tensor(k)
    missing, unexpected = visual.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary" not in m]      # buffers, rebuilt at init
    if missing or unexpected:
        raise RuntimeError(f"visual load: missing {missing[:5]} unexpected {unexpected[:5]}")
    # Rotary frequency buffers stay fp32, as HF's from_pretrained keeps them: a blanket
    # .to(bf16) rounds inv_freq, and the ViT's features then differ from HF's by up to
    # 4.0 (the decision state by 1.3); kept in fp32 they were bitwise equal.
    keep = {n: b.detach().clone() for n, b in visual.named_buffers() if "inv_freq" in n}
    visual = visual.to(dtype)
    for n, b in keep.items():
        mod, _, attr = n.rpartition(".")
        visual.get_submodule(mod).register_buffer(attr, b.float(), persistent=False)
    return visual.eval()


class VisionDecisionModel(DecisionModel):
    """DecisionModel whose state may carry images. Text-only batches take the
    parent's path untouched."""

    def __init__(self, tower, hidden, cfg, lm_head=None, *, visual: nn.Module,
                 image_token_id: int, vcfg: VisionConfig | None = None):
        super().__init__(tower, hidden, cfg, lm_head=lm_head)
        # The vision tower must be the base model's own: its merger emits the text
        # tower's width (2B: 2048, 4B: 2560). A ViT from another size would scatter
        # features of the wrong width over the image tokens.
        vo = getattr(getattr(visual, "config", None), "out_hidden_size", None)
        he = tower.get_input_embeddings().embedding_dim
        if vo is not None and vo != he:
            raise ValueError(f"the vision tower emits {vo}-wide features; the text tower "
                             f"embeds {he}: load the base model's own vision tower")
        self.visual = visual
        self.image_token_id = image_token_id
        self.vcfg = vcfg or VisionConfig()
        if self.vcfg.freeze_visual:
            for p in self.visual.parameters():
                p.requires_grad_(False)

    def image_embeds(self, pixel_values, image_grid_thw) -> torch.Tensor:
        dt = next(self.visual.parameters()).dtype
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.vcfg.freeze_visual):
            out = self.visual(pixel_values.to(dt), grid_thw=image_grid_thw, return_dict=True)
        return out.pooler_output

    def embed(self, input_ids, image_embeds=None) -> torch.Tensor:
        """Token embeddings with the image features written over the image-pad rows,
        in order. Without features, the plain token embeddings."""
        emb = self.tower.get_input_embeddings()(input_ids)
        if image_embeds is None:
            return emb
        mask = (input_ids == self.image_token_id).unsqueeze(-1)
        if int(mask.sum()) != image_embeds.shape[0]:
            raise ValueError(f"{int(mask.sum())} image tokens for {image_embeds.shape[0]} image features")
        return emb.masked_scatter(mask, image_embeds.to(emb.dtype))

    def encode_image_prefix(self, input_ids, position_ids, image_embeds=None, cache=None):
        """Run a prefix that may hold image tokens once and return its cache, or
        continue `cache` over the next `input_ids` (mutating it).

        `position_ids` are this chunk's (3, 1, T) M-RoPE positions, sliced from the
        positions of the WHOLE sequence (`mrope_position_ids`), never recomputed from
        the chunk alone: after an image the text positions jump by the image's grid,
        not by its token count, so a chunk's positions depend on what came before
        it. `image_embeds` are the features of the image-pad tokens in this chunk,
        in order. Under a causal mask this computes what the full pass computes for
        these tokens -- the same argument as `DecisionModel.encode_prefix`.

        An early-exit model (ArchConfig.exit_layer) runs its first exit_layer layers
        here as everywhere else, and the cache keeps only those layers. Qwen3.5 puts
        image features in at the input embeddings only (deepstack_visual_indexes is
        empty for 0.8B/2B/4B), so the exit never cuts the image path."""
        with torch.no_grad(), self.exit_tower():
            cache = self.tower(inputs_embeds=self.embed(input_ids, image_embeds),
                               position_ids=position_ids, past_key_values=cache,
                               use_cache=True).past_key_values
        return self.exit_cache(cache)

    def _compute(self, *, input_ids, pixel_values=None, image_grid_thw=None,
                 image_embeds=None, **kw):
        if pixel_values is None and image_embeds is None:
            return super()._compute(input_ids=input_ids, **kw)
        img = image_embeds if image_embeds is not None else self.image_embeds(pixel_values, image_grid_thw)
        return super()._compute(input_ids=input_ids, inputs_embeds=self.embed(input_ids, img), **kw)

    def forward_exits(self, *, input_ids, pixel_values=None, image_grid_thw=None,
                      image_embeds=None, **kw) -> dict:
        """Every exit's logits (arch aux_exits) for a state that may hold images: the
        image features go in exactly as in _compute. Used by the per-exit dumps of the
        exit-policy selection (scripts/dump_exits.py); serving does not call it."""
        if pixel_values is None and image_embeds is None:
            return super().forward_exits(input_ids=input_ids, **kw)
        img = image_embeds if image_embeds is not None else self.image_embeds(pixel_values, image_grid_thw)
        return super().forward_exits(input_ids=input_ids, inputs_embeds=self.embed(input_ids, img), **kw)
