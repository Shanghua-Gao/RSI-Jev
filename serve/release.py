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
import os
import re
from pathlib import Path

import torch

from rsijev.arch import ArchConfig, DecisionModel
from rsijev.encode import EncodeConfig
from rsijev.vision import vision_block

# Keyed by release, because a key that means "the 2B one" stops being useful the
# moment there are two of them.
REPOS = {"v6.0-vl-4b": "shgao/rsi-jev-v6.0-vl-4b",
         "v5.0-vl-3b": "shgao/rsi-jev-v5.0-vl-3b",
         "v4.0-vl-2b": "shgao/rsi-jev-v4.0-vl-qwen3.5-2b",
         "v3.0-2b": "shgao/rsi-jev-v3.0-qwen3.5-2b",
         "v2.1-2b": "shgao/rsi-jev-v2.1-qwen3.5-2b",
         "v2.0-2b": "shgao/rsi-jev-v2.0-qwen3.5-2b",
         "v1.0-2b": "shgao/rsi-jev-v1.0-qwen3.5-2b",
         "v1.0-0.8b": "shgao/rsi-jev-v1.0-qwen3.5-0.8b"}
LATEST = "v6.0-vl-4b"

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
    """`v3.0` out of `rsi-jev-v3.0-qwen3.5-2b`, `v4.0-VL` out of
    `rsi-jev-v4.0-vl-qwen3.5-2b`, or None. The only suffix read is `-vl`: any other
    word after the number is the base model's name, not the release's."""
    found = re.search(r"v\d+\.\d+(-vl)?(?![a-z])", name, re.IGNORECASE)
    if not found:
        return None
    return found.group(0)[:len(found.group(0)) - len(found.group(1) or "")] + \
        (found.group(1) or "").upper()


# The input cap of every release that does not record its own (v1.0 - v4.0-VL).
DEFAULT_MAX_LENGTH = 2048


def serving_encoder(spec: dict, max_length: int | None = None,
                    truncate: str | None = None) -> tuple[int, str]:
    """(input cap, over-cap policy) a release is served with.

    The cap is the one the checkpoint was trained at, `spec.max_length` in its
    meta.json, or 2048 when it records none (every release up to v4.0-VL). The
    policy is `spec.truncate`: "left" (the default, every release so far) or
    "middle" (the long-context encoder, rsijev/encode.py). An explicit argument, or
    RSIJEV_MAX_LENGTH / RSIJEV_TRUNCATE, overrides either; that is a choice made at
    serving time, not what the model was gated with.

    TODO (not in this release): order averaging (score each question in 2 or 5
    option orders and average; 2x/5x cost) is a kept result that is not served yet.
    Adaptive exit is served for multi-question requests (see `adaptive_mode`)."""
    if max_length is None:
        env = os.environ.get("RSIJEV_MAX_LENGTH", "").strip()
        max_length = int(env) if env else None
    if truncate is None:
        truncate = os.environ.get("RSIJEV_TRUNCATE", "").strip() or None
    cap = int(max_length if max_length is not None
              else (spec.get("max_length") or DEFAULT_MAX_LENGTH))
    policy = truncate or spec.get("truncate") or "left"
    if policy not in ("left", "middle"):
        raise ValueError(f"truncate must be 'left' or 'middle', got {policy!r}")
    if cap < 64:
        raise ValueError(f"max_length {cap} is too small to hold a question")
    return cap, policy


ADAPTIVE_CAL_FILE = "adaptive_calibration.safetensors"


def adaptive_block(meta: dict) -> dict | None:
    """The adaptive-exit policy block of a release's meta.json, or None."""
    return meta.get("adaptive") or (meta.get("release") or {}).get("adaptive")


ADAPTIVE_MODES = ("auto", "on", "off")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")


def adaptive_mode(meta: dict, adaptive: str | None = None,
                  fixed_exit: bool | None = None) -> tuple[str, str]:
    """(mode, where it came from) for a release with aux exits.

      auto  adaptive exit for multi-question requests, the fixed exit for one question
            (the default; GB10 five-repeat check: adaptive 6-9% faster on 8- and
            32-question batches, 1-7% slower on single questions)
      on    adaptive exit for every text request the model runs on its own
      off   the fixed exit for every request

    `adaptive` (--adaptive, else RSIJEV_ADAPTIVE) wins, then `fixed_exit` (--fixed-exit,
    else RSIJEV_FIXED_EXIT=1, an alias for off), then meta.json `adaptive.serving`, then
    auto. Asking for on or auto together with the fixed exit is an error."""
    if adaptive is None:
        adaptive = os.environ.get("RSIJEV_ADAPTIVE", "").strip().lower() or None
    if fixed_exit is None:
        fixed_exit = _env_flag("RSIJEV_FIXED_EXIT")
    if adaptive is not None:
        adaptive = str(adaptive).strip().lower()
        if adaptive not in ADAPTIVE_MODES:
            raise ValueError(f"adaptive must be one of {ADAPTIVE_MODES}, got {adaptive!r}")
        if fixed_exit and adaptive != "off":
            raise ValueError(f"--adaptive {adaptive} contradicts --fixed-exit / RSIJEV_FIXED_EXIT")
        return adaptive, "--adaptive / RSIJEV_ADAPTIVE"
    if fixed_exit:
        return "off", "--fixed-exit / RSIJEV_FIXED_EXIT"
    block = adaptive_block(meta) or {}
    if block.get("serving") is not None:
        mode = str(block["serving"]).strip().lower()
        if mode not in ADAPTIVE_MODES:
            raise ValueError(f"meta.json adaptive.serving must be one of {ADAPTIVE_MODES}, got {mode!r}")
        return mode, "meta.json adaptive.serving"
    return "auto", "default"


def adaptive_policy(meta: dict, ckpt: Path, model, device, fixed_exit: bool | None = None,
                    adaptive: str | None = None):
    """(rsijev.adaptive.Policy or None, why it is off or None). Sets model.adaptive_mode.

    A release serves adaptive exit when it has aux exits (spec.arch_extra.aux_exits,
    heads in aux_scorers.safetensors) AND a policy: meta.json `adaptive` with the tau
    tuned on DEV, and one calibrator per aux exit, either
      - a scalar temperature per exit in calibration.json, next to the main head's own
        calibration: "exits": {"12": {"cal_mode": "temp", "logT": x}, ...}
        (`exit_temperatures`), or
      - one cal-4b per aux exit in adaptive_calibration.safetensors (keys
        "<exit>.<mean|W|mu|sd|w|b>").
    scripts/pack_adaptive_policy.py writes either from the tuning run's policy. Nothing
    about the policy is a code constant. Which requests use it is `adaptive_mode` (auto:
    multi-question requests only); mode off serves the fixed exit and loads no policy."""
    model.adaptive_mode = "off"
    model.effort_base = None
    model.effort_auto_taus = None
    if not getattr(model.cfg, "aux_exits", ()):
        return None, None
    temps = exit_temperatures(ckpt, model.exit_indices())      # validated even when off
    mode, src = adaptive_mode(meta, adaptive, fixed_exit)
    block = adaptive_block(meta)
    if not block or block.get("tau") is None:
        return None, ("the release has aux exits but no tuned tau (meta.json adaptive.tau)"
                      if mode != "off" else f"fixed exit forced ({src})")
    if mode == "off":
        # Not served by default, but per-request `effort` (serve/effort.py) can still ask
        # for it; a policy that does not build leaves effort at full.
        try:
            model.effort_base = _build_policy(meta, ckpt, model, device, block, temps)
        except RuntimeError:
            model.effort_base = None
        if model.effort_base is not None:
            model.effort_auto_taus = auto_thresholds(block, model.effort_base.exits)
        return None, f"fixed exit forced ({src})"
    policy = _build_policy(meta, ckpt, model, device, block, temps)
    model.effort_base = policy
    model.effort_auto_taus = auto_thresholds(block, policy.exits)
    model.adaptive_mode = mode
    return policy, None


def auto_thresholds(block: dict, exits: list) -> dict | None:
    """meta.json adaptive.auto_thresholds {"<aux exit>": tau in (0, 1]}: per-exit thresholds
    for effort auto only (serve/effort.py); the default (unset effort) keeps the single tau.
    A partial block is filled from that tau. Absent: None."""
    raw = block.get("auto_thresholds")
    if raw is None:
        return None
    aux = [str(L) for L in exits[:-1]]
    if not isinstance(raw, dict) or not set(map(str, raw)) <= set(aux):
        raise RuntimeError(f"meta.json adaptive.auto_thresholds must map aux exits {aux} to "
                           f"thresholds in (0, 1]; got {raw!r}")
    out = {}
    for k, v in raw.items():
        t = float(v)
        if not (0.0 < t <= 1.0):
            raise RuntimeError(f"meta.json adaptive.auto_thresholds[{k!r}] = {v!r} is not in (0, 1]")
        out[int(k)] = 2.0 if t >= 1.0 else t
    return out


def _build_policy(meta: dict, ckpt: Path, model, device, block: dict, temps):
    """The release's adaptive Policy (exits, per-exit calibration, tau)."""
    from safetensors.torch import load_file
    from rsijev import adaptive as A
    exits = model.exit_indices()
    if block.get("exits") is not None and [int(x) for x in block["exits"]] != exits:
        raise RuntimeError(f"adaptive.exits {block['exits']} do not match the model's exits {exits}")
    cal_file = block.get("calibration")
    if temps is not None:
        if cal_file not in (None, CAL_JSON):
            raise RuntimeError(f"calibration.json has per-exit temperatures and meta.json "
                               f"adaptive.calibration names {cal_file}: one source only")
        if (ckpt / ADAPTIVE_CAL_FILE).exists():
            raise RuntimeError(f"both calibration.json \"exits\" and {ADAPTIVE_CAL_FILE}: one source only")
        if getattr(model, "cal_mode", "none") == "none":
            raise RuntimeError("per-exit temperatures need the main head calibrated too "
                               "(calibration.json cal_mode)")
        cal = {L: A.cal_to(t, device) for L, t in temps.items()}
    else:
        path = ckpt / (cal_file or ADAPTIVE_CAL_FILE)
        if not path.exists():
            raise RuntimeError(f"adaptive exit needs {path.name} (one cal-4b per aux exit) or "
                               f"per-exit temperatures in calibration.json")
        flat = load_file(str(path))
        cal = {}
        for L in exits[:-1]:
            c = {k.split(".", 1)[1]: v for k, v in flat.items() if k.split(".", 1)[0] == str(L)}
            if set(c) != {"mean", "W", "mu", "sd", "w", "b"}:
                raise RuntimeError(f"{path.name}: exit {L} has {sorted(c)}, not a cal-4b")
            cal[L] = A.cal_to(c, device)
    return A.Policy(exits, cal, float(block["tau"]))


CAL_JSON = "calibration.json"


def exit_temperatures(ckpt: Path, exits: list[int]) -> dict | None:
    """The per-exit scalar temperatures in calibration.json, or None when it has none.

    Format: "exits": {"<aux exit>": {"cal_mode": "temp", "logT": <finite number>}, ...},
    one entry for every aux exit (the main exit's calibration is the file's own cal_mode
    and calibration.safetensors). Returns {exit: {"logT": float32 scalar}} (the
    rsijev.adaptive temp_cal form); any other shape is refused."""
    import math
    from rsijev import adaptive as A
    p = Path(ckpt) / CAL_JSON
    if not p.exists():
        return None
    block = json.loads(p.read_text()).get("exits")
    if block is None:
        return None
    want = [str(L) for L in exits[:-1]]
    if not isinstance(block, dict) or sorted(block, key=int) != want:
        got = sorted(block) if isinstance(block, dict) else type(block).__name__
        raise RuntimeError(f"calibration.json exits: need one entry per aux exit {want}, got {got}")
    out = {}
    for k in want:
        e = block[k]
        if not isinstance(e, dict) or e.get("cal_mode") != "temp" or set(e) != {"cal_mode", "logT"}:
            raise RuntimeError(f"calibration.json exits[{k}]: need {{\"cal_mode\": \"temp\", \"logT\": x}}, got {e!r}")
        v = e["logT"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise RuntimeError(f"calibration.json exits[{k}].logT must be a finite number, got {v!r}")
        out[int(k)] = A.temp_cal(v)
    return out


def _layer_at_or_above(key: str, n: int) -> bool:
    """True for a decoder-layer weight with index >= n ("layers.<i>." in the key)."""
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
    return bool(m) and int(m.group(1)) >= n


def strip_exit_tower(src: str | Path, dst: str | Path, exit_layer: int) -> dict:
    """Write a tower.safetensors without the decoder layers an early-exit release never
    runs (index >= exit_layer). Every kept tensor is copied unchanged."""
    from safetensors.torch import load_file, save_file
    sd = load_file(str(src))
    kept = {k: v.contiguous() for k, v in sd.items() if not _layer_at_or_above(k, exit_layer)}
    save_file(kept, str(dst), metadata={"format": "pt"})
    return {"kept": len(kept), "dropped": len(sd) - len(kept),
            "params_kept": sum(v.numel() for v in kept.values()),
            "params_dropped": sum(v.numel() for k, v in sd.items() if k not in kept)}


def own_token_pool(spec: dict, meta: dict) -> bool:
    """Own-token option pooling: meta's flag, or RSIJEV_OPTION_POOL_OWN_TOKENS=1|0 to override."""
    env = os.environ.get("RSIJEV_OPTION_POOL_OWN_TOKENS")
    if env is not None:
        return env == "1"
    return bool(spec.get("option_pool_own_tokens") or meta.get("option_pool_own_tokens"))


# spec.arch_extra keys that only shape training (which layers train, at what rate); serving ignores them.
TRAINING_ONLY_ARCH_KEYS = ("freeze_lower_n", "lower_lr_mult", "retention_at_exit")


def serving_arch_extra(spec: dict) -> dict:
    """spec.arch_extra minus the training-only keys. Any other key ArchConfig doesn't know is an
    error: it may change what the model computes."""
    import dataclasses
    extra = {k: v for k, v in dict(spec.get("arch_extra") or {}).items() if k not in TRAINING_ONLY_ARCH_KEYS}
    known = {f.name for f in dataclasses.fields(ArchConfig)}
    unknown = sorted(set(extra) - known)
    if unknown:
        raise RuntimeError(f"spec.arch_extra has keys this server does not know: {unknown}")
    return extra


def load_release(ckpt: str | Path, device: str = "cuda", infer_dtype=None,
                 vision: bool | None = None, max_length: int | None = None,
                 truncate: str | None = None, fixed_exit: bool | None = None,
                 adaptive: str | None = None):
    """`infer_dtype` casts the TOWER for inference only: bf16 is 3-6x faster and
    moved pooled top-1 by at most 0.003 over the full test set. The scorer always
    stays fp32 -- running it in reduced precision is the bug that cost this
    project a whole version. Evaluation leaves this None and gets fp32.

    A checkpoint trained with images (v4.0-VL on: a `vision` block in meta.json) is
    loaded with the base model's own vision tower beside the text tower, as
    `rsijev.vision.VisionDecisionModel`; a text request runs exactly the text
    path. `vision=False` loads it text-only. `meta["vision"]` then says what an
    image request may carry; a text-only model's meta has no such key.

    An early-exit checkpoint (the 4B releases: `spec.arch_extra.exit_layer`, e.g. 20
    of 32) is read the same way: ArchConfig takes exit_layer / exit_norm from
    arch_extra, and every tower call -- the scoring pass, the prefix and document
    caches, image states -- runs only the first exit_layer layers and the head reads
    that layer. A checkpoint without exit_layer loads exactly as before.

    `max_length` / `truncate`: see `serving_encoder`. `meta["serving"]` records what
    was used.

    Adaptive exit (aux exits plus a tuned policy, see `adaptive_policy`): the aux heads
    load from aux_scorers.safetensors, `model.adaptive_policy` holds the policy and
    `model.adaptive_mode` which requests use it (`adaptive_mode`; auto: multi-question
    requests, each question stopping at the first exit confident enough,
    serve/infer.score_adaptive). Every other model gets `adaptive_policy = None`.
    """
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ckpt = Path(ckpt)
    meta = json.loads((ckpt / "meta.json").read_text())
    spec = meta["spec"]
    # A self-contained release (config.json next to meta.json) carries everything it
    # runs: the tokenizer, the image processor, the text tower with its embeddings and
    # the vision tower. It never touches the base model on the Hub. Older releases
    # take those from `base_model`.
    local = (ckpt / "config.json").exists()
    src = str(ckpt) if local else meta["base_model"]
    meta["weights_source"] = src
    tok = AutoTokenizer.from_pretrained(src)
    # Load straight into the precision the tower will run in, rather than fp32
    # then a cast: the fp32 checkpoint rounds to the target once either way, and
    # this never materialises a second copy. That matters on a laptop, and on a
    # unified-memory box holding two checkpoints at once. The base weights are
    # bf16 on the Hub, so fp32 is an exact upcast and bf16 is a no-op.
    # An early-exit release runs only its first exit_layer decoder layers, so build the
    # tower with just those: the deeper layers are never loaded, and a release may ship
    # a tower.safetensors without them (strip_exit_tower). A full-depth file loads too;
    # its deeper layers are dropped here.
    exit_layer = (spec.get("arch_extra") or {}).get("exit_layer")
    kw = {}
    if exit_layer:
        from transformers import AutoConfig
        bc = AutoConfig.from_pretrained(src)
        tc = getattr(bc, "text_config", None) or bc
        tc.num_hidden_layers = int(exit_layer)
        if getattr(tc, "layer_types", None):
            tc.layer_types = list(tc.layer_types)[: int(exit_layer)]
        kw["config"] = bc
    if local:
        from transformers import AutoConfig
        lm = AutoModelForCausalLM.from_config(kw.get("config") or AutoConfig.from_pretrained(src),
                                              dtype=infer_dtype or torch.float32)
    else:
        lm = AutoModelForCausalLM.from_pretrained(meta["base_model"],
                                                  dtype=infer_dtype or torch.float32, **kw)
    tower = getattr(lm, "model", lm)
    sd = load_file(str(ckpt / "tower.safetensors"))
    if exit_layer:
        sd = {k: v for k, v in sd.items() if not _layer_at_or_above(k, int(exit_layer))}
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    # a base-model release takes its embeddings from the Hub; a self-contained one ships them
    bad = list(missing) if local else [k for k in missing if "embed_tokens" not in k]
    if exit_layer:                     # layers at or past the exit never run
        bad = [k for k in bad if not _layer_at_or_above(k, int(exit_layer))]
    if bad or unexpected:
        raise RuntimeError(f"checkpoint does not match {meta['base_model']}: "
                           f"missing {bad[:5]}, unexpected {list(unexpected)[:5]}")
    cfg = getattr(lm.config, "text_config", None) or lm.config
    arch = ArchConfig(readout=spec["readout"], readout_layer=spec["readout_layer"],
                      max_options=spec["max_options"], freeze_base=True,
                      option_pool=spec["option_pool"], residual=spec["residual"],
                      logit_cap=spec.get("logit_cap"),
                      head_input_norm=spec.get("head_input_norm", False),
                      **serving_arch_extra(spec))
    vb = vision_block(meta) if vision is not False else None
    if vb:
        from rsijev.vision import (IMAGE_PAD, VisionConfig, VisionDecisionModel, load_visual,
                                   vision_revision)
        # Training loads the vision tower named in the vision block; serving loads the
        # base model's. They must be the same model (a 2B ViT emits 2048-wide features,
        # the 4B tower embeds 2560).
        if vb.get("model") and vb["model"] != meta["base_model"]:
            raise RuntimeError(f"the checkpoint was trained with the vision tower of "
                               f"{vb['model']}, but its base model is {meta['base_model']}")
        vcfg = VisionConfig(image_token_budget=int(vb.get("budget", 1024)))
        revision = None if local else vision_revision(vb, meta["base_model"])
        # The ViT runs in bf16 whatever the tower does (fp32 rotary buffers), which
        # is how the vision releases were trained and gated.
        model = VisionDecisionModel(tower, cfg.hidden_size, arch,
                                    visual=load_visual(src, revision=revision),
                                    image_token_id=tok.convert_tokens_to_ids(IMAGE_PAD),
                                    vcfg=vcfg).to(device)
        meta["vision"] = {"image_token_budget": vcfg.image_token_budget,
                          "min_tokens_per_image": vcfg.min_tokens_per_image,
                          "revision": revision}
    else:
        model = DecisionModel(tower, cfg.hidden_size, arch).to(device)
    model.scorer.load_state_dict(load_file(str(ckpt / "scorer.safetensors")))
    if arch.aux_exits:
        aux = ckpt / "aux_scorers.safetensors"
        if not aux.exists():
            raise RuntimeError(f"spec.arch_extra.aux_exits {list(arch.aux_exits)} needs {aux.name}")
        model.aux_scorers.load_state_dict(load_file(str(aux)))
        model.aux_scorers.to(device=device, dtype=torch.float32)
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
    cap, policy = serving_encoder(spec, max_length, truncate)
    enc = EncodeConfig(layout=spec["layout"], option_pool=spec["option_pool"],
                       option_order="canonical", max_length=cap, truncate=policy,
                       option_pool_own_tokens=own_token_pool(spec, meta))
    ada, why = adaptive_policy(meta, ckpt, model, device, fixed_exit, adaptive)
    model.adaptive_policy = ada
    served_ada = None if ada is None else {"exits": ada.exits, "tau": ada.tau, "mode": model.adaptive_mode}
    meta["serving"] = {"max_length": cap, "truncate": policy,
                       "exit_layer": arch.exit_layer, "adaptive": served_ada,
                       "option_pool_own_tokens": enc.option_pool_own_tokens}
    if why:
        meta["serving"]["adaptive_off"] = why
    return model, tok, enc, meta


def artifact_name(path) -> str:
    """The checkpoint's own name, never where it lived.

    A record written here ships inside the published checkpoint, so an absolute
    path would publish the machine it was trained on. v1.0's verify.json went out
    carrying a lab filesystem path before this existed.
    """
    return Path(path).name
