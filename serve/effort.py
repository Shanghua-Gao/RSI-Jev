"""`effort`: how many layers a multi-exit release runs, per request or as the server default.

A release with aux exits (spec.arch_extra.aux_exits) and a tuned policy (meta.json
`adaptive`) can answer at more than one depth. `effort` picks one, using the heads the
release already has; no other weights are involved:

  light     the shallowest aux exit for every question (v6.0-VL: layer 16)
  balanced  the deepest aux exit for every question (v6.0-VL: layer 20)
  full      the main exit for every question (v6.0-VL: layer 32), i.e. --adaptive off
  auto      the release's confidence cascade (exit to exit, stop at the first whose
            calibrated top-1 probability reaches tau) for every text request, a single
            question included, i.e. --adaptive on

Aliases: low = light, medium = balanced, high = max = full.

Left unset (no `effort` in the request, no --effort / RSIJEV_EFFORT), serving is exactly
what it was before `effort` existed: the release's adaptive mode (--adaptive, meta.json
`adaptive.serving`, else auto: the cascade for multi-question requests, the main exit for
one question). Requests with images run at full depth whatever the effort, because the aux
heads read text only; their response reports effort "full". A release without aux exits
answers effort full (or unset) as before and refuses light, balanced and auto with a 422.
"""
from __future__ import annotations

import os

EFFORTS = ("light", "balanced", "full", "auto")
ALIASES = {"low": "light", "medium": "balanced", "high": "full", "max": "full"}
NEEDS_EXITS = ("light", "balanced", "auto")


def canonical(value) -> str | None:
    """'Medium' -> 'balanced'; None or '' -> None. Raises ValueError for anything else."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if not v:
        return None
    v = ALIASES.get(v, v)
    if v not in EFFORTS:
        names = ", ".join(list(EFFORTS) + [f"{a} ({b})" for a, b in ALIASES.items()])
        raise ValueError(f"effort must be one of {names}; got {value!r}")
    return v


def threshold(value) -> float | None:
    """A request's `confidence_threshold`: None, or a number in (0, 1]. Raises ValueError."""
    if value is None:
        return None
    try:
        t = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"confidence_threshold must be a number in (0, 1]; got {value!r}") from None
    if not (0.0 < t <= 1.0):
        raise ValueError(f"confidence_threshold must be in (0, 1]; got {value!r}")
    return t


def resolve(model, effort, conf_threshold, images: bool) -> tuple[str | None, float | None]:
    """(effort, threshold) a plan runs with, after validation; raises ValueError.

    `conf_threshold` replaces the release's tau for the cascade, so it applies to effort
    auto and to the default adaptive path; with light, balanced or full it is refused. 1.0
    means never exit early: the plan runs at full depth (exactly the main exit), which also
    covers a single-option question whose confidence is exactly 1. Images run at full depth
    whatever is asked."""
    if conf_threshold is not None:
        if effort in ("light", "balanced", "full"):
            raise ValueError(f"confidence_threshold applies to effort 'auto' or the default "
                             f"adaptive path, not to effort {effort!r}")
        if getattr(model, "effort_base", None) is None:
            raise ValueError("confidence_threshold needs a multi-exit release with an adaptive policy")
    if images:
        return ("full" if effort is not None or conf_threshold is not None else None), None
    check_supported(model, effort)
    if conf_threshold == 1.0:
        return "full", None
    return effort, conf_threshold


def default_effort(cli_value=None) -> str | None:
    """The server default: --effort, else RSIJEV_EFFORT, else None (unset)."""
    if cli_value is not None:
        return canonical(cli_value)
    return canonical(os.environ.get("RSIJEV_EFFORT"))


def check_supported(model, effort: str | None) -> None:
    """Raise ValueError when this release cannot serve `effort`."""
    if effort in NEEDS_EXITS and getattr(model, "effort_base", None) is None:
        if not getattr(getattr(model, "cfg", None), "aux_exits", ()):
            raise ValueError(f"effort {effort!r} needs a release with aux exits; this one has a "
                             f"single exit, so only effort 'full' (or no effort) is served")
        raise ValueError(f"effort {effort!r} needs the release's adaptive policy (meta.json "
                         f"adaptive.tau and per-exit calibration), which this release lacks")


def policy_for(model, effort: str):
    """The rsijev.adaptive.Policy a text plan runs with at `effort` (light/balanced/auto)."""
    from rsijev.adaptive import Policy
    base = model.effort_base
    if effort == "auto":
        return base
    cache = model.__dict__.setdefault("_effort_policies", {})
    if effort not in cache:
        aux = base.exits[:-1]
        if effort == "light":
            # tau -1: every row stops at the first exit; the policy's other stages never run
            cache[effort] = Policy(list(base.exits), dict(base.cal), -1.0)
        elif effort == "balanced":
            L = aux[-1]
            cache[effort] = Policy([L, base.exits[-1]], {L: base.cal[L]}, -1.0)
        else:
            raise ValueError(f"no policy for effort {effort!r}")
    return cache[effort]
