"""Train one release checkpoint: the given recipe, one model, one seed.

Same code path as a worker arm (run_arm_lib.run_arm), plus save_dir, so the saved
checkpoint is exactly the model its records score. Run from a dedicated release
tree, so a release never depends on any other working copy.

    python scripts/release_train.py --model Qwen/Qwen3.5-2B-Base --seed 17 \
        --spec spec.json --save-dir CKPT --out OUTDIR --name NAME --corpus DIR
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--corpus", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    lm = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    corpus = {f.stem: load_cases(str(f)) for f in sorted(Path(a.corpus).glob("*.jsonl"))}
    targets = {"mmlu_pro_1k": load_mmlu_pro_1k(), "typed_decisions": load_typed_decisions("test")}
    spec = {**json.loads(Path(a.spec).read_text()), "seed": a.seed, "save_dir": a.save_dir}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = lib.run_arm(lm=lm, tok=tok, targets=targets, corpus=corpus, device=dev,
                       spec=spec, name=a.name, items_path=out / f"{a.name}.items.jsonl")
    with open(out / f"{a.name}.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
