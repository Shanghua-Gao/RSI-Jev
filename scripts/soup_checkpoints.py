"""Average release checkpoints of one architecture into one (v6.1-VL stage 3). Nothing is trained.

Every tensor of tower.safetensors, scorer.safetensors (the main head) and, when present,
aux_scorers.safetensors (the early-exit heads) is the weighted mean of the members' (uniform by
default). The mean is taken in fp32; the tower is stored in the first member's dtype, the heads
in fp32. The members must agree on everything that shapes the forward pass (readout, layout,
option pooling, option cap, arch_extra, residual, logit cap, head input norm, readout layer) and
hold the same tensor names. meta.json is the first member's plus a "soup" block naming the
members and weights; serving reads only the architecture and readout fields, which are equal.

  python scripts/soup_checkpoints.py OUT CKPT_A CKPT_B [--weights 0.5 0.5]

v6.1-VL is the uniform average of v6.0-VL's head-stage checkpoint and a second fine-tune of the
same base (data/v6.1-vl_recipe/3-soup.json).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

FILES = ("tower.safetensors", "scorer.safetensors", "aux_scorers.safetensors")
SAME = ("readout", "layout", "option_pool", "max_options", "arch_extra", "residual", "logit_cap",
        "head_input_norm", "readout_layer")


def check_specs(metas: list) -> None:
    """Refuse members whose forward pass differs."""
    for k in SAME:
        vals = [m["spec"].get(k) for m in metas]
        if any(v != vals[0] for v in vals):
            raise ValueError(f"members differ in spec.{k}: {vals}")


def average(sds: list, weights: list, dtype=None) -> dict:
    """The weighted mean of state dicts with the same keys, computed in fp32; stored in `dtype`
    (or each tensor's own dtype in the first member when None)."""
    if any(set(sd) != set(sds[0]) for sd in sds):
        raise ValueError("members hold different tensor names")
    if abs(sum(weights) - 1.0) > 1e-6 or len(weights) != len(sds):
        raise ValueError(f"weights {weights} must be one per member and sum to 1")
    out = {}
    for k in sds[0]:
        acc = sum(float(w) * sd[k].float() for w, sd in zip(weights, sds))
        out[k] = acc.to(dtype or sds[0][k].dtype).contiguous()
    return out


def soup(out: Path, members: list, weights: list | None = None) -> dict:
    from safetensors.torch import load_file, save_file
    out, members = Path(out), [Path(m) for m in members]
    weights = weights or [1.0 / len(members)] * len(members)
    metas = [json.loads((m / "meta.json").read_text()) for m in members]
    check_specs(metas)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for fn in FILES:
        have = [(m / fn).exists() for m in members]
        if not any(have):
            continue
        if not all(have):
            raise ValueError(f"{fn} is missing in some members")
        sds = [load_file(str(m / fn)) for m in members]
        avg = average(sds, weights, None if fn == "tower.safetensors" else torch.float32)
        save_file(avg, str(out / (fn + ".tmp")))
        os.replace(out / (fn + ".tmp"), out / fn)
        written.append(fn)
        del sds, avg
    meta = dict(metas[0])
    meta["soup"] = {"members": [m.name for m in members], "weights": [float(w) for w in weights],
                    "kind": "uniform" if len(set(weights)) == 1 else "weighted"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return {"out": str(out), "files": written, "soup": meta["soup"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("members", nargs="+")
    ap.add_argument("--weights", type=float, nargs="+")
    a = ap.parse_args()
    if len(a.members) < 2:
        raise SystemExit("need at least two members")
    print(json.dumps(soup(Path(a.out), a.members, a.weights)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
