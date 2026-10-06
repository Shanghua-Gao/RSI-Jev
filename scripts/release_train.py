"""Train one release checkpoint: the given recipe, one model, one seed.

Same code path as a worker arm (run_arm_lib.run_arm), plus save_dir, so the saved
checkpoint is exactly the model its records score. Run from a dedicated release
tree, so a release never depends on any other working copy.

    python scripts/release_train.py --model Qwen/Qwen3.5-2B-Base --seed 17 \
        --spec spec.json --save-dir CKPT --out OUTDIR --name NAME --corpus DIR [--root DIR]

A published meta.json names its parent checkpoint (fit_extra.init_from), its image
corpora (fit_extra.vision.roots), where an RL stage saves its calibration
(fit_extra.rl2.cal_save_dir) and, for a multi-exit model, where the early-exit heads
are loaded from and saved to (fit_extra.aux_init_from, aux_save_dir) by relative names
such as "calA-ce/s17" or "vision_v1". --root DIR resolves those against DIR; without it
they are used as given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["TRITON_CACHE_DIR"] = os.path.join(
    os.environ.get("TMPDIR", "/tmp"),
    f"triton-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}")

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_arm_lib as lib                                        # noqa: E402
from rsijev.contract import load_cases                          # noqa: E402
from rsijev.targets import load_mmlu_pro_1k, load_typed_decisions  # noqa: E402


def resolve_paths(spec: dict, root: Path) -> dict:
    """The spec with its relative checkpoint / corpus paths placed under `root`."""
    import copy
    spec = copy.deepcopy(spec)
    fe = spec.get("fit_extra") or {}

    def at(p):
        return str(p) if not p or Path(p).is_absolute() else str(root / p)
    for k in ("init_from", "aux_init_from", "aux_save_dir"):
        if fe.get(k):
            fe[k] = at(fe[k])
    vis = fe.get("vision") or {}
    if vis.get("root"):
        vis["root"] = at(vis["root"])
    if vis.get("roots"):
        vis["roots"] = [at(r) for r in vis["roots"]]
    rl2 = fe.get("rl2") or {}
    if rl2.get("cal_save_dir"):
        rl2["cal_save_dir"] = at(rl2["cal_save_dir"])
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--root", default=None,
                    help="directory that relative init_from / vision roots / cal_save_dir are under")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    corpus = {f.stem: load_cases(str(f)) for f in sorted(Path(a.corpus).glob("*.jsonl"))}
    targets = {"mmlu_pro_1k": load_mmlu_pro_1k(), "typed_decisions": load_typed_decisions("test")}
    spec = {**json.loads(Path(a.spec).read_text()), "seed": a.seed, "save_dir": a.save_dir}
    if a.root:
        spec = resolve_paths(spec, Path(a.root))
    for k in lib.unknown_spec_keys(spec):
        print(f"WARNING: spec key {k!r} is not used by this code", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = lib.run_arm(lm=lm, tok=tok, targets=targets, corpus=corpus, device=dev,
                       spec=spec, name=a.name, items_path=out / f"{a.name}.items.jsonl")
    with open(out / f"{a.name}.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
