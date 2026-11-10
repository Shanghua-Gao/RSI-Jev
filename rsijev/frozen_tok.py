"""A release's tokenizer without transformers, for the MLX path.

`tokenizer.json` is not the tokenizer a release was trained with: transformers 5.17
rebuilds the Qwen3.5 pre-tokenizer when it loads one, and registers more added tokens
from tokenizer_config.json. `tokenizer.frozen.json` is what transformers actually ended up
with, written once at conversion time, so this reader gives the same ids with no transformers
installed and no dependence on its version.
"""
from __future__ import annotations

import json
from pathlib import Path

FROZEN = "tokenizer.frozen.json"


def has_frozen(path: str | Path) -> bool:
    return (Path(path) / FROZEN).is_file()


def freeze(src: str | Path, out: str | Path | None = None) -> Path:
    """Write `FROZEN` into `out` (default `src`) from what transformers loads from `src`."""
    from transformers import AutoTokenizer
    target = Path(out if out is not None else src) / FROZEN
    target.write_text(AutoTokenizer.from_pretrained(str(src)).backend_tokenizer.to_str(), encoding="utf-8")
    return target


def _content(token) -> str | None:
    return token.get("content") if isinstance(token, dict) else token


class FrozenTokenizer:
    """The part of a transformers tokenizer the MLX path uses."""

    split_special_tokens = False

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer
        path = Path(path)
        self.backend_tokenizer = Tokenizer.from_file(str(path / FROZEN))
        config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.pad_token, self.eos_token = _content(config.get("pad_token")), _content(config.get("eos_token"))
        self.pad_token_id = self.convert_tokens_to_ids(self.pad_token) if self.pad_token else None
        self.eos_token_id = self.convert_tokens_to_ids(self.eos_token) if self.eos_token else None

    def __call__(self, text, add_special_tokens: bool = False) -> dict:
        t = self.backend_tokenizer
        if isinstance(text, str):
            return {"input_ids": t.encode(text, add_special_tokens=add_special_tokens).ids}
        return {"input_ids": [e.ids for e in t.encode_batch(list(text), add_special_tokens=add_special_tokens)]}

    def decode(self, ids) -> str:
        return self.backend_tokenizer.decode(list(ids), skip_special_tokens=False)

    def convert_tokens_to_ids(self, token: str):
        return self.backend_tokenizer.token_to_id(token)


def load_tokenizer(path: str | Path):
    """The frozen tokenizer when the package has one; older packages load through transformers."""
    if has_frozen(path):
        return FrozenTokenizer(path)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path))
