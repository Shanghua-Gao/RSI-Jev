"""Score a checkpoint on the twelve-benchmark suite: the headline in every release record.

    python scripts/suite.py --ckpt DIR                    # all twelve, weighted mean
    python scripts/suite.py --ckpt DIR --only typed_decisions_test,kev_hard_v1
    python scripts/suite.py --ckpt DIR --json out.json

Each benchmark's TEST split is fetched from a pinned upstream repository, so this needs
the roots those live under. `--list` prints every benchmark with its origin, its licence
and the variable that points at it, and exits; nothing is redistributed by this repo.

Two benchmarks are read in canonical and reversed option order and the pair is reported;
`nimble_public` and `jev_style_panel` are averaged over their subsets rather than pooled,
which is what their authors do. Both conventions live in `rsijev/targets_suite.py`, so a
number here and a number in a release record are the same measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch                                                       # noqa: E402

from rsijev.contract import gold_label                             # noqa: E402
from rsijev.encode import EncodeConfig                             # noqa: E402
from rsijev.evaluate import predict                                # noqa: E402
from rsijev.metrics import ece                                     # noqa: E402
from rsijev import targets_suite as ts                             # noqa: E402

BOTH_ORDERS = {"typed_decisions_test", "mmlu_pro_1k"}


def score(model, tok, cases, enc, *, max_options, device, batch_size):
    """pooled top-1, per-source top-1, mean confidence and ECE for one benchmark."""
    preds = predict(model, tok, cases, enc, max_options=max_options, device=device,
                    batch_size=batch_size)
    hits, conf, correct, by_src = 0, [], [], {}
    for c, q, p in preds:
        i = max(range(len(p.probs)), key=p.probs.__getitem__)
        ok = q.options[i] == gold_label(q, c.gold[q.key])
        hits += int(ok)
        conf.append(p.probs[i])
        correct.append(ok)
        s = by_src.setdefault(getattr(c, "source", "") or "", [0, 0])
        s[0] += int(ok); s[1] += 1
    n = len(correct)
    return {"n": n, "top1": round(hits / n, 4),
            "mean_conf": round(sum(conf) / n, 4),
            "ece": round(ece(conf, correct, bins=15), 4),
            "by_source": {k: round(v[0] / v[1], 4) for k, v in sorted(by_src.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--only", default=None, help="comma-separated benchmark names")
    ap.add_argument("--json", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp32"],
                    help="tower precision; published suite numbers are fp32")
    ap.add_argument("--no-decontam", action="store_true",
                    help="keep eval cases that the decontamination report drops")
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    a = ap.parse_args()

    if a.list:
        print(f"{'benchmark':22s} {'weight':>7s}  licence / origin")
        for name, w in ts.RECOMMENDED.items():
            b = ts.SUITE[name]
            print(f"{name:22s} {w:7.2f}  {b.licence}\n{'':32s}{b.origin}")
        return 0
    if not a.ckpt:
        ap.error("--ckpt is required unless --list")

    # APPEND, never insert: scripts/serve.py shadows the serve/ package, so putting this
    # directory at the front of sys.path breaks every import of serve.infer.
    sys.path.append(str(ROOT / "scripts"))
    from load_release import load_release
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}.get(a.dtype)
    model, tok, _, meta = load_release(a.ckpt, dev, infer_dtype=dt)
    spec = meta["spec"]
    print(f"loaded: calibration {meta['calibration']}, tower {a.dtype or 'fp32'}, "
          f"kernel {meta['linear_attn_kernel']}")

    names = a.only.split(",") if a.only else list(ts.RECOMMENDED)
    suite = ts.load_suite(names, decontam=not a.no_decontam)
    out, weighted, wsum = {}, 0.0, 0.0
    for name in names:
        cases = suite[name]
        rows = {}
        for order in (["canonical", "reversed"] if name in BOTH_ORDERS else ["canonical"]):
            enc = EncodeConfig(layout=spec["layout"], option_pool=spec["option_pool"],
                              option_order=order)
            rows[order] = score(model, tok, cases, enc, max_options=spec["max_options"],
                                device=dev, batch_size=a.batch_size)
        r = rows["canonical"]
        # The suite metric: macro over subsets where the authors report a macro.
        top1 = (sum(r["by_source"].values()) / len(r["by_source"])
                if name in ts.MACRO_OVER_SOURCE and r["by_source"] else r["top1"])
        w = ts.RECOMMENDED.get(name, 0.0)
        weighted += w * top1
        wsum += w
        out[name] = {**rows, "metric_top1": round(top1, 4), "weight": w,
                     "macro_over_source": name in ts.MACRO_OVER_SOURCE}
        extra = f"  reversed {rows['reversed']['top1']}" if "reversed" in rows else ""
        print(f"  {name:22s} top-1 {top1:.4f}  ECE {r['ece']:.4f}  n={r['n']}{extra}")

    summary = {"suite_mean": round(weighted / wsum, 4) if wsum else None,
               "weight_covered": round(wsum, 4),
               "suite_ece": round(sum(ts.RECOMMENDED.get(k, 0) * v["canonical"]["ece"]
                                      for k, v in out.items()) / wsum, 4) if wsum else None}
    print(f"\n  suite mean {summary['suite_mean']}  suite ECE {summary['suite_ece']}"
          f"  (weight covered {summary['weight_covered']})")
    if wsum < 0.999:
        print("  NOTE: a partial suite. The release records' suite mean is over all twelve.")
    if a.json:
        Path(a.json).write_text(json.dumps({"benchmarks": out, **summary,
                                            "calibration": meta["calibration"]}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
