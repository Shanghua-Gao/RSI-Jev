"""Find and load a released checkpoint.

A checkpoint (written by scripts/run_arm_lib.save_release) holds the fine-tuned
tower in fp32 WITHOUT the embedding, which was frozen during training and is taken
from the public base model; the trained option scorer; and meta.json with the full
recipe. `load_release` rebuilds the exact model that was scored.

`resolve_ckpt` turns what a user typed into a local directory: a directory is used
as it is, a Hugging Face repo id (`shgao/rsi-jev-v3.0-qwen3.5-2b`) or a short alias
(`v3.0-2b`) is fetched with `huggingface_hub.snapshot_download` into the standard
Hugging Face cache, so a second run downloads nothing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from rsijev.arch import ArchConfig, DecisionModel
from rsijev.encode import EncodeConfig
from rsijev.vision import vision_block

# Keyed by release, because a key that means "the 2B one" stops being useful the
# moment there are two of them.
REPOS = {"v4.0-vl-2b": "shgao/rsi-jev-v4.0-vl-qwen3.5-2b",
         "v3.0-2b": "shgao/rsi-jev-v3.0-qwen3.5-2b",
         "v2.1-2b": "shgao/rsi-jev-v2.1-qwen3.5-2b",
         "v2.0-2b": "shgao/rsi-jev-v2.0-qwen3.5-2b",
         "v1.0-2b": "shgao/rsi-jev-v1.0-qwen3.5-2b",
         "v1.0-0.8b": "shgao/rsi-jev-v1.0-qwen3.5-0.8b"}
LATEST = "v3.0-2b"

_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def repo_id_for(ref: str | Path) -> str | None:
    """The Hugging Face repo `ref` names, or None when it names a local path.

    An existing directory always wins, so a local folder that happens to be called
    `shgao/...` is never swapped for a download. Anything written as a path
    (`./x`, `/x`, `~/x`, `a/b/c`) stays a path, so a typo in one fails as a missing
    directory instead of turning into a Hub lookup."""
    ref = str(ref)
    if Path(ref).expanduser().is_dir():
        return None
    if ref in REPOS:
        return REPOS[ref]
    if _REPO_ID.match(ref) and not ref.startswith((".", "~", "/")):
        return ref
    return None


def resolve_ckpt(ref: str | Path, *, revision: str | None = None) -> Path:
    """A local checkpoint directory for `ref`: a directory, an HF repo id or an alias."""
    path = Path(str(ref)).expanduser()
    if path.is_dir():
        return path
    repo = repo_id_for(ref)
    if repo is None:
        raise FileNotFoundError(
            f"{ref!r} is not a checkpoint directory, a Hugging Face repo id such as "
            f"{REPOS[LATEST]!r}, or one of the aliases {sorted(REPOS)}")
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo, revision=revision))
    except Exception as e:
        if Path(str(ref)).parent.is_dir() and ref not in REPOS:
            raise FileNotFoundError(f"{ref!r} is neither a local checkpoint directory nor "
                                    f"a Hugging Face repo that could be fetched: {e}") from e
        raise


def checkpoint_name(ref: str | Path, path: str | Path | None = None) -> str:
    """The name a checkpoint is served under: the repo's own name for an HF id or
    alias (`rsi-jev-v3.0-qwen3.5-2b`, never a snapshot hash), the directory's name
    otherwise."""
    repo = repo_id_for(ref)
    if repo is not None:
        return repo.rsplit("/", 1)[-1]
    return Path(path if path is not None else ref).expanduser().resolve().name


def release_version(name: str) -> str | None:
    """`v3.0` out of `rsi-jev-v3.0-qwen3.5-2b`, or None."""
    found = re.search(r"v\d+\.\d+", name)
    return found.group(0) if found else None


def load_release(ckpt: str | Path, device: str = "cuda", infer_dtype=None,
                 vision: bool | None = None):
    """`infer_dtype` casts the TOWER for inference only: bf16 is 3-6x faster and
    moved pooled top-1 by at most 0.003 over the full test set. The scorer always
    stays fp32 -- running it in reduced precision is the bug that cost this
    project a whole version. Evaluation leaves this None and gets fp32.

    A checkpoint trained with images (v4.0 on: a `vision` block in meta.json) is
    loaded with the base model's own vision tower beside the text tower, as
    `rsijev.vision.VisionDecisionModel`; a text request runs exactly the text
    path. `vision=False` loads it text-only. `meta["vision"]` then says what an
    image request may carry; a text-only model's meta has no such key.
    """
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt = Path(ckpt)
    meta = json.loads((ckpt / "meta.json").read_text())
    spec = meta["spec"]
    tok = AutoTokenizer.from_pretrained(meta["base_model"])
    # Load straight into the precision the tower will run in, rather than fp32
    # then a cast: the fp32 checkpoint rounds to the target once either way, and
    # this never materialises a second copy. That matters on a laptop, and on a
    # unified-memory box holding two checkpoints at once. The base weights are
    # bf16 on the Hub, so fp32 is an exact upcast and bf16 is a no-op.
    lm = AutoModelForCausalLM.from_pretrained(meta["base_model"],
                                              dtype=infer_dtype or torch.float32)
    tower = getattr(lm, "model", lm)
    missing, unexpected = tower.load_state_dict(load_file(str(ckpt / "tower.safetensors")),
                                                strict=False)
    bad = [k for k in missing if "embed_tokens" not in k]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint does not match {meta['base_model']}: "
                           f"missing {bad[:5]}, unexpected {list(unexpected)[:5]}")
    cfg = getattr(lm.config, "text_config", None) or lm.config
    arch = ArchConfig(readout=spec["readout"], readout_layer=spec["readout_layer"],
                      max_options=spec["max_options"], freeze_base=True,
                      option_pool=spec["option_pool"], residual=spec["residual"],
                      logit_cap=spec.get("logit_cap"),
                      head_input_norm=spec.get("head_input_norm", False),
                      **dict(spec.get("arch_extra") or {}))
    vb = vision_block(meta) if vision is not False else None
    if vb:
        from rsijev.vision import IMAGE_PAD, VisionConfig, VisionDecisionModel, load_visual
        vcfg = VisionConfig(image_token_budget=int(vb.get("budget", 1024)))
        # The ViT runs in bf16 whatever the tower does (fp32 rotary buffers), which
        # is how the vision releases were trained and gated.
        model = VisionDecisionModel(tower, cfg.hidden_size, arch,
                                    visual=load_visual(meta["base_model"]),
                                    image_token_id=tok.convert_tokens_to_ids(IMAGE_PAD),
                                    vcfg=vcfg).to(device)
        meta["vision"] = {"image_token_budget": vcfg.image_token_budget,
                          "min_tokens_per_image": vcfg.min_tokens_per_image}
    else:
        model = DecisionModel(tower, cfg.hidden_size, arch).to(device)
    model.scorer.load_state_dict(load_file(str(ckpt / "scorer.safetensors")))
    # v2.0 onward a checkpoint may ship a fitted calibration (calibration.safetensors
    # + calibration.json, rsijev/calibrate.py). It is part of the released model, not
    # an extra: the forward pass divides each question's logits by one positive
    # temperature, so the answer is unchanged and it is still one forward pass. A
    # checkpoint without those files -- v1.0 -- loads exactly as it always did.
    if (ckpt / "calibration.safetensors").exists():
        from rsijev.calibrate import load_calibration
        meta["calibration"] = load_calibration(model, ckpt)
    else:
        meta["calibration"] = "none"
    model.scorer.to(torch.float32)          # never follows the tower down
    model.eval()
    enc = EncodeConfig(layout=spec["layout"], option_pool=spec["option_pool"],
                       option_order="canonical")
    return model, tok, enc, meta


def artifact_name(path) -> str:
    """The checkpoint's own name, never where it lived.

    A record written here ships inside the published checkpoint, so an absolute
    path would publish the machine it was trained on. v1.0's verify.json went out
    carrying a lab filesystem path before this existed.
    """
    return Path(path).name
