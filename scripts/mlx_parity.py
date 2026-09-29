"""MLX vs PyTorch parity for a release checkpoint.

    # PyTorch fp32 reference -> fixture (needs torch + the typed-decisions test set)
    python scripts/mlx_parity.py reference --ckpt DIR --out tests/fixtures/mlx_parity_v3.0.json

    # MLX against the fixture (needs mlx + mlx-lm; no torch, no dataset)
    python scripts/mlx_parity.py check --ckpt DIR_OR_REPO [--dtype float32]

The fixture holds the inputs (state and typed questions, from every 16th case of
the typed-decisions test split) and the reference's calibrated probabilities,
so the check runs anywhere MLX runs, including a Mac with no PyTorch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# MLX's CUDA backend runs fp32 matmuls as TF32 unless told not to; parity means fp32.
os.environ.setdefault("MLX_ENABLE_TF32", "0")
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "mlx_parity_v3.0.json"


def _questions(case) -> list:
    from rsijev.contract import Question
    return [Question(key=q["key"], mode=q["mode"], instructions=q["instructions"],
                     options=tuple(q["options"]), criteria=q["criteria"])
            for q in case["questions"]]


def reference(a) -> int:
    import torch
    sys.path.append(str(ROOT / "scripts"))      # after ROOT: scripts/serve.py shadows serve/
    from load_release import load_release
    from rsijev.targets import load_typed_decisions
    from serve.infer import score_questions
    torch.set_num_threads(a.threads)
    model, tok, enc, meta = load_release(a.ckpt, "cpu", infer_dtype=torch.float32)
    cases = load_typed_decisions("test")[::a.stride]
    out = []
    t0 = time.perf_counter()
    for c in cases:
        preds, _ = score_questions(model, tok, c.state, list(c.questions), enc, device="cpu",
                                   batch_size=8, max_options=meta["spec"]["max_options"])
        out.append({"case_id": c.case_id, "state": c.state,
                    "questions": [{"key": q.key, "mode": q.mode, "instructions": q.instructions,
                                   "options": list(q.options), "criteria": q.criteria}
                                  for q in c.questions],
                    "probs": {q.key: [round(p, 7) for p in pr.probs]
                              for q, pr in zip(c.questions, preds)}})
    n = sum(len(c["questions"]) for c in out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "release": meta["release"]["name"], "calibration": meta["calibration"],
        "reference": f"PyTorch {torch.__version__} fp32 CPU, scripts/load_release.py + "
                     "serve.infer.score_questions",
        "source": f"LocalLLaMA/typed-decisions test, every {a.stride}th case",
        "n_questions": n, "cases": out}, indent=1) + "\n")
    print(f"wrote {a.out}: {len(out)} cases, {n} questions, {time.perf_counter() - t0:.0f}s")
    return 0


def compare(fixture: dict, model, tok, meta, verbose: bool = False) -> dict:
    """Run MLX on the fixture's inputs; argmax agreement and max |dp| per option."""
    from rsijev_mlx.infer import encode_config, score_questions
    enc = encode_config(meta)
    n = agree = 0
    max_dp, flips, dps, worst = 0.0, [], [], None
    for i, c in enumerate(fixture["cases"]):
        if verbose and i % 5 == 0:
            print(f"  case {i}/{len(fixture['cases'])}", flush=True)
        qs = _questions(c)
        probs, _ = score_questions(model, tok, c["state"], qs, enc)
        for q, p in zip(qs, probs):
            ref = c["probs"][q.key]
            n += 1
            am, rm = max(range(len(p)), key=p.__getitem__), max(range(len(ref)), key=ref.__getitem__)
            agree += int(am == rm)
            if am != rm:
                top = sorted(ref, reverse=True)
                flips.append({"case": c["case_id"], "key": q.key,
                              "ref_top2_gap": top[0] - top[1]})
            dp = max(abs(x - y) for x, y in zip(p, ref))
            dps.append(dp)
            if dp > max_dp:
                max_dp, worst = dp, {"case": c["case_id"], "key": q.key, "mode": q.mode,
                                     "ref_max_p": max(ref)}
    dps.sort()
    return {"n": n, "argmax_agreement": agree / n, "max_abs_dp": max_dp,
            "median_abs_dp": dps[len(dps) // 2], "worst": worst, "flips": flips}


def check(a) -> int:
    from rsijev_mlx import load
    fixture = json.loads(Path(a.fixture).read_text())
    t0 = time.perf_counter()
    model, tok, meta = load(a.ckpt, dtype=a.dtype)
    import mlx.core as mx
    print(f"loaded in {time.perf_counter() - t0:.0f}s ({a.dtype}, calibration "
          f"{meta['calibration']}, device {mx.default_device()})", flush=True)
    t0 = time.perf_counter()
    r = compare(fixture, model, tok, meta, verbose=True)
    r["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(r, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--ckpt", required=True)
    r.add_argument("--out", default=str(FIXTURE))
    r.add_argument("--stride", type=int, default=16)
    r.add_argument("--threads", type=int, default=16)
    c = sub.add_parser("check")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--fixture", default=str(FIXTURE))
    c.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    a = ap.parse_args()
    return reference(a) if a.cmd == "reference" else check(a)


if __name__ == "__main__":
    raise SystemExit(main())
