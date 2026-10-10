"""EDITABLE (model axis). The text side of image states, without torch.

How a state with images is spelled for the text tower: one literal "<image>" per image
where it goes (or none: the images then lead the state, in order), each expanded to
"<|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>" with n the image's LLM tokens.
rsijev/vision.py (PyTorch) and rsijev/mlx (MLX) both use this, so the two backends expand a
state identically.
"""
from __future__ import annotations

from typing import Sequence

IMAGE_MARKER = "<image>"
VISION_START, IMAGE_PAD, VISION_END = "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"
# Qwen's multimodal special tokens. A state may not spell them itself: they
# would tokenise as the real tokens and be counted as image positions.
VISION_SPECIAL_TOKENS = (VISION_START, IMAGE_PAD, VISION_END, "<|video_pad|>")


def expand_state(state: str, n_tokens: Sequence[int]) -> str:
    runs = [VISION_START + IMAGE_PAD * n + VISION_END for n in n_tokens]
    k = state.count(IMAGE_MARKER)
    if k == 0:
        return "\n".join(runs) + ("\n" + state if state.strip() else "")
    if k != len(runs):
        raise ValueError(f"state has {k} {IMAGE_MARKER} markers for {len(runs)} images")
    parts = state.split(IMAGE_MARKER)
    return "".join(p + (runs[i] if i < len(runs) else "") for i, p in enumerate(parts))
