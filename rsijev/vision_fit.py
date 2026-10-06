"""Training-time image feed for fit.py and rl2.py (`fit_extra.vision`).

Corpora are loaded with contract.load_cases, which keeps no `images` field, so
images are looked up by case_id in the vision corpus roots: their vis*.jsonl
rows carry `images`, paths relative to the root. A case whose id is in no root
is text and is encoded exactly as by the plain encoder, so a text row costs
nothing and changes nothing.

    fit_extra.vision = {"root": <dir> | "roots": [<dir>, ...],
                        "budget": 1024,                  # image tokens per question
                        "model": "Qwen/Qwen3.5-2B-Base", # whose frozen ViT encodes the images
                        "rl_budget": 256,                # optional: RL-side rows (rl2)
                        "source_budget": {"vis_docvqa": 2048}}   # optional, per source prefix

The roots are built by scripts/build_vision_v1.py, build_vision_v2.py and
build_vision_v3.py. v4.0-VL trained with roots vision_v1 + vision_v2 + vision_v3
and budget 1024. The ViT is frozen and runs in bf16; its features are scattered
into the tower's own (frozen) token embeddings at the image-pad positions, and
the text tower then runs on `inputs_embeds` with multimodal RoPE positions.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .encode import collate, encode_question
from .vision import (IMAGE_PAD, ImagePrep, VisionConfig, encode_vision_question,
                     load_visual, vision_collate)


class VisionFeed:
    def __init__(self, spec: dict, tokenizer, model, device):
        roots = [Path(r) for r in (spec.get("roots") or [spec["root"]])]
        self.index: dict[str, tuple[Path, list[str]]] = {}
        for root in roots:
            for f in sorted(root.glob("vis*.jsonl")):          # vis_* (v1), vis2_* (v2)
                for line in f.open():
                    if line.strip():
                        r = json.loads(line)
                        if r["case_id"] in self.index:
                            raise ValueError(f"case_id {r['case_id']} in two vision roots")
                        self.index[r["case_id"]] = (root, r.get("images") or [])
        self.root = roots
        budget = int(spec.get("budget", 1024))
        mid = spec.get("model", "Qwen/Qwen3.5-2B-Base")
        # fall back to a smaller budget when an image would not fit max_length
        self.preps = [ImagePrep(mid, VisionConfig(image_token_budget=b))
                      for b in (budget, budget // 2, budget // 4)]
        # RL-side rows (rl2 listwise image reranking: 16 one-image rows per group) may use
        # a smaller per-row budget; rl_budget unset = the replay budget.
        rlb = int(spec.get("rl_budget", budget))
        self.rl_preps = self.preps if rlb == budget else [
            ImagePrep(mid, VisionConfig(image_token_budget=b, min_tokens_per_image=min(64, b)))
            for b in (rlb, max(16, rlb // 2))]
        self.visual = load_visual(mid).to(device)
        # the vision tower must be the base model's own (4B: out 2560, 2B: out 2048)
        vo = self.visual.config.out_hidden_size
        he = model.tower.get_input_embeddings().embedding_dim
        if vo != he:
            raise ValueError(f"vision model {mid!r} emits {vo}-wide features; the tower "
                             f"embeds {he}: set fit_extra.vision.model to the base model")
        for p in self.visual.parameters():
            p.requires_grad_(False)
        # per-source budgets (e.g. {"vis_docvqa": 2048}): prefix match on case.source; such rows
        # get max_length raised by their budget so the image is never the part that is cut
        self.src_preps = {k: [ImagePrep(mid, VisionConfig(image_token_budget=int(b)))
                              for b in (int(b), int(b) // 2)]
                          for k, b in (spec.get("source_budget") or {}).items()}
        self.src_budget = {k: int(b) for k, b in (spec.get("source_budget") or {}).items()}
        self.tok, self.model, self.device = tokenizer, model, device
        self.pad_id = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
        self.n_image_rows = 0
        print(f"    vision: {len(self.index)} image cases indexed from {self.root}, "
              f"budget {budget}", flush=True)

    def _encode(self, c, q, enc, rng, preps=None):
        root, ims = self.index.get(c.case_id, (None, []))
        if not ims:
            return encode_question(self.tok, c.state, q, enc, rng=rng)
        key = next((k for k in self.src_preps if c.source.startswith(k)), None)
        if key is not None and preps is None:
            import dataclasses
            preps = self.src_preps[key]
            enc = dataclasses.replace(enc, max_length=enc.max_length + self.src_budget[key])
        from PIL import Image
        pil = [Image.open(root / p).convert("RGB") for p in ims]
        err = None
        for prep in (preps or self.preps):  # noqa
            st = rng.getstate() if rng is not None else None
            try:
                return encode_vision_question(self.tok, prep, c.state, pil, q, enc, rng=rng)
            except ValueError as e:                    # image cut by max_length
                err = e
                if rng is not None:
                    rng.setstate(st)
        raise err

    def batch(self, chunk, enc, rng, max_options, rl: bool = False):
        return self.collate(self.encode(chunk, enc, rng, rl), max_options)

    # encode / collate are split so fit() can encode a whole step and run it as
    # grad_accum micro-batches (fit_extra.grad_accum). batch() == collate(encode()).
    def encode(self, chunk, enc, rng, rl: bool = False):
        return [self._encode(c, q, enc, rng, self.rl_preps if rl else None) for c, q in chunk]

    def collate(self, ex, max_options):
        b = vision_collate(self.tok, ex, max_options, device=self.device)
        if "pixel_values" in b:
            pv, grid = b.pop("pixel_values"), b.pop("image_grid_thw")
            ids = b["input_ids"]
            with torch.no_grad():
                img = self.visual(pv.to(torch.bfloat16), grid_thw=grid,
                                  return_dict=True).pooler_output
            emb = self.model.tower.get_input_embeddings()(ids)
            mask = (ids == self.pad_id).unsqueeze(-1)
            if int(mask.sum()) != img.shape[0]:
                raise ValueError(f"{int(mask.sum())} image tokens for {img.shape[0]} features")
            b["inputs_embeds"] = emb.masked_scatter(mask, img.to(emb.dtype))
            self.n_image_rows += sum(1 for e in ex if "pixel_values" in e)
        return b
