"""Check an MLX build on this Mac: answers vs the shipped references, latency per effort, peak memory.

    python scripts/mac_check.py shgao/rsi-jev-v6.1-vl-4b-mlx-8bit          # or a local directory
    python scripts/mac_check.py PKG --ref scripts/mac_check_ref/rsi-jev-v6.1-vl-4b-mlx-8bit.json

Runs the reference rows (200 questions: 180 text, 20 with images; public benchmarks) through
the served MLX path, one question per request, at effort low / medium / high / auto, and
prints per effort:

  * median and p90 latency per request (ms), after two warm-up requests
  * argmax agreement and mean |dp| (largest absolute probability difference over the options)
    with the bf16 PyTorch release, served the same way on a GPU, and with this same build run
    by MLX on Linux (CUDA backend), which is what the build was validated with
  * the mean exit depth (auto)

then the peak memory MLX allocated over those rows (weights + activations), the load time,
and peak memory and latency for one question on a long document (1k to 32k tokens, stopping
before a length that would not fit). Pass --json to
save everything (send that file back). Building references (maintainers): --backend torch
or --backend mlx with --rows ROWS.json --write-ref OUT.json, then --merge-ref; --make-rows
DATA OUT.json picks the rows.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EFFORTS = ("low", "medium", "high", "auto")
REF_DIR = ROOT / "scripts" / "mac_check_ref"


# ----------------------------------------------------------------------------- rows
def make_rows(data: Path, out: Path, per_final=30, n_clinc=30, n_vision=20, seed=0):
    """Public-benchmark rows: eval_final_v2 (5 sets x 30), CLINC150+OOS (30, 151 options),
    eval_vision_v1 (4 per set, the image files' original bytes embedded)."""
    rng = random.Random(seed)
    rows = []
    man = json.loads((data / "eval_final_v2" / "manifest.json").read_text())["benchmarks"]
    for name, b in sorted(man.items()):
        recs = [json.loads(ln) for ln in open(data / "eval_final_v2" / b["file"]) if ln.strip()]
        rng.shuffle(recs)
        n = 0
        for r in recs:
            for q in r["questions"]:
                if n < per_final and len(q["options"]) <= 160:
                    rows.append({"id": f"{_public(name)}:{r['case_id']}:{q['key']}", "set": _public(name),
                                 "state": r["state"], "question": q,
                                 "gold": int(np.argmax(r["gold"][q["key"]]))})
                    n += 1
    clinc = [json.loads(ln) for ln in open(data / "clinc_items300.jsonl")]
    rng.shuffle(clinc)
    for it in clinc[:n_clinc]:
        qd = it["payload"]["questions"]["q1"]
        keys = list(qd["criteria"])
        rows.append({"id": f"clinc:{it['id']}", "set": "clinc150", "state": "",
                     "question": {"key": "q1", "mode": "choice", "instructions": qd["instructions"],
                                  "options": keys, "criteria": dict(qd["criteria"])},
                     "gold": keys.index(it["gold"])})
    vroot = data / "eval_vision_v1"
    vman = json.loads((vroot / "manifest.json").read_text())["benchmarks"]
    per = max(1, n_vision // len(vman))
    for name, b in sorted(vman.items()):
        recs = [json.loads(ln) for ln in open(vroot / b["file"]) if ln.strip()]
        rng.shuffle(recs)
        n = 0
        for r in recs:
            for q in r["questions"]:
                if n < per and len(q["options"]) <= 160:
                    rows.append({"id": f"{_public(name)}:{r['case_id']}:{q['key']}", "set": _public(name),
                                 "state": r["state"], "question": q,
                                 "gold": int(np.argmax(r["gold"][q["key"]])),
                                 "images": [base64.b64encode((vroot / p).read_bytes()).decode()
                                            for p in r["images"]]})
                    n += 1
    out.write_text(json.dumps({"rows": rows}))
    print(f"wrote {out}: {len(rows)} rows ({sum('images' in r for r in rows)} with images), "
          f"{out.stat().st_size / 1e6:.1f} MB")


def _public(name: str) -> str:
    """The benchmark's own name (efv2_bbh -> bbh, efvis_pope -> pope)."""
    return name.split("_", 1)[1] if name.startswith(("efv2_", "efvis_")) else name


def _question(q):
    from rsijev.contract import Question
    return Question(q["key"], q["mode"], q["instructions"], tuple(q["options"]), dict(q["criteria"]))


def _images(r):
    if not r.get("images"):
        return None
    from PIL import Image
    return [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in r["images"]]


# ----------------------------------------------------------------------------- memory
def _mx_mem(name):
    import mlx.core as mx
    f = getattr(mx, name, None) or getattr(getattr(mx, "metal", None), name, None)
    return f() if f else None


def _reset_peak():
    import mlx.core as mx
    f = getattr(mx, "reset_peak_memory", None) or getattr(getattr(mx, "metal", None), "reset_peak_memory", None)
    if f:
        f()


def machine() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    if sys.platform == "darwin":
        for k in ("machdep.cpu.brand_string", "hw.memsize"):
            try:
                info[k] = subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
            except OSError:
                pass
    return info


# ----------------------------------------------------------------------------- run
def run(model, rows, backend, efforts, cpu=False, batch_size=16):
    t0 = time.time()
    if backend == "mlx":
        import mlx.core as mx
        if cpu:
            mx.set_default_device(mx.cpu)
        from rsijev.mlx.serving import load_for_serving, make_scorer
        served = load_for_serving(model)
        info = {"mlx": mx.__version__, "device": str(mx.default_device())}
    else:
        from serve.server import load_for_serving, make_scorer
        served = load_for_serving(model)
        import torch
        info = {"torch": torch.__version__, "device": served.device, "dtype": served.dtype_name}
    scorer = make_scorer(served, batch_size)
    load_s = time.time() - t0
    info.update(name=served.name, load_s=round(load_s, 1), tower=served.dtype_name)
    if backend == "mlx":
        info["weights_gb_active_after_load"] = round((_mx_mem("get_active_memory") or 0) / 1e9, 2)
        _reset_peak()
    print(f"loaded {served.name} ({backend}, {info['device']}, {served.dtype_name}) in {load_s:.0f} s", flush=True)
    qs = [_question(r["question"]) for r in rows]
    ims = [_images(r) for r in rows]
    for i in (0, len(rows) - 1):                                   # warm-up: one text, one image
        scorer(rows[i]["state"], [qs[i]], ims[i], "high")
    out = {}
    for effort in efforts:
        res = []
        for r, q, im in zip(rows, qs, ims):
            t = time.perf_counter()
            got = scorer(r["state"], [q], im, effort)
            ms = (time.perf_counter() - t) * 1e3
            probs, extras = got[0][0], (got[2] if len(got) > 2 else {})
            depth = extras.get("depth")
            res.append({"ms": round(ms, 2), "p": [round(float(x), 6) for x in probs],
                        "depth": depth[0] if isinstance(depth, list) else depth})
        out[effort] = res
        ms = [x["ms"] for x in res]
        print(f"  {effort:6s} median {statistics.median(ms):8.1f} ms", flush=True)
    if backend == "mlx":
        info["peak_memory_gb"] = round((_mx_mem("get_peak_memory") or 0) / 1e9, 2)
        info["memory_by_length"] = memory_curve(scorer)
    return info, out


def memory_curve(scorer, lengths=(1000, 4000, 8000, 16000, 32000)):
    """Peak MLX memory and latency for one question on a long synthetic document, by input
    length; stops before a length that would need more than ~60% of this machine's memory."""
    from rsijev.contract import Question
    total = None
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    except (OSError, ValueError):
        pass
    q = [Question("a", "choice", "Which department should handle this?", ("billing", "technical", "other"),
                  {"billing": "Payments and refunds", "technical": "Software bugs", "other": "Anything else"})]
    out, last = [], None
    for n in lengths:
        if total and last and last["peak_gb"] * 1e9 * (n / last["tokens"]) > 0.6 * total:
            break
        state = "The quarterly report lists revenue, costs and open support tickets by region. " * (n // 14)
        _reset_peak()
        t = time.perf_counter()
        got = scorer(state, q, None, "high")
        last = {"tokens": int(got[1]), "peak_gb": round((_mx_mem("get_peak_memory") or 0) / 1e9, 2),
                "ms": round((time.perf_counter() - t) * 1e3, 1)}
        out.append(last)
        print(f"  {last['tokens']:6d} tokens: peak {last['peak_gb']:6.2f} GB, {last['ms']:8.1f} ms", flush=True)
    return out


def compare(rows, got, ref, efforts):
    """Agreement / mean |dp| of `got` vs `ref` outputs, per effort and text / image."""
    res = {}
    for e in efforts:
        if e not in ref or e not in got:
            continue
        for part, sel in (("all", lambda r: True), ("text", lambda r: not r.get("images")),
                          ("images", lambda r: bool(r.get("images")))):
            idx = [i for i, r in enumerate(rows) if sel(r)]
            if not idx:
                continue
            pa = [np.asarray(got[e][i]["p"]) for i in idx]
            pb = [np.asarray(ref[e][i]["p"]) for i in idx]
            agree = float(np.mean([a.argmax() == b.argmax() for a, b in zip(pa, pb)]))
            dp = [float(np.abs(a - b).max()) for a, b in zip(pa, pb)]
            res[f"{e}.{part}"] = {"n": len(idx), "agree": round(agree, 4), "dp_mean": round(float(np.mean(dp)), 5),
                                  "dp_max": round(float(np.max(dp)), 4)}
    return res


def accuracy(rows, got, efforts):
    return {e: round(float(np.mean([np.argmax(got[e][i]["p"]) == r["gold"] for i, r in enumerate(rows)])), 4)
            for e in efforts if e in got}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("model", nargs="?", help="MLX build: local directory or Hugging Face repo id")
    ap.add_argument("--ref", default=None, help="reference JSON (default: scripts/mac_check_ref/<name>.json)")
    ap.add_argument("--json", default=None, help="save the results here")
    ap.add_argument("--efforts", default=",".join(EFFORTS))
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (quick look)")
    ap.add_argument("--backend", choices=("mlx", "torch"), default="mlx")
    ap.add_argument("--cpu", action="store_true", help="MLX on the CPU instead of the GPU")
    ap.add_argument("--rows", default=None, help="rows JSON (to build a reference)")
    ap.add_argument("--write-ref", default=None, help="write this run's outputs as a reference")
    ap.add_argument("--make-rows", nargs=2, metavar=("DATA", "OUT"), default=None)
    ap.add_argument("--merge-ref", nargs=3, metavar=("OUT", "TORCH_REF", "MLX_REF"), default=None,
                    help="bundle a PyTorch reference and an MLX reference of the same rows")
    a = ap.parse_args(argv)
    if a.make_rows:
        make_rows(Path(a.make_rows[0]), Path(a.make_rows[1]))
        return 0
    if a.merge_ref:
        out, tp, mp = a.merge_ref
        t, m = json.loads(Path(tp).read_text()), json.loads(Path(mp).read_text())
        if [r["id"] for r in t["rows"]] != [r["id"] for r in m["rows"]]:
            raise SystemExit("the two references hold different rows")
        slim = lambda o: {e: [{"p": x["p"], "depth": x["depth"]} for x in v] for e, v in o.items()}  # noqa: E731
        rows_file = Path(out).parent / "rows.json"         # the rows (with images) once, beside the refs
        if not rows_file.exists() or json.loads(rows_file.read_text())["rows"] != t["rows"]:
            rows_file.write_text(json.dumps({"rows": t["rows"]}))
        Path(out).write_text(json.dumps({"rows": rows_file.name, "torch_bf16": slim(t["outputs"]),
                                         "mlx_linux": slim(m["outputs"]),
                                         "info": {"torch_bf16": t["info"], "mlx_linux": m["info"]}}))
        print(f"wrote {out}: {Path(out).stat().st_size / 1e6:.1f} MB")
        return 0
    if not a.model:
        ap.error("model is required")
    efforts = [e for e in a.efforts.split(",") if e]
    name = Path(a.model.rstrip("/")).name
    ref_path = Path(a.ref) if a.ref else REF_DIR / f"{name}.json"
    ref = json.loads(ref_path.read_text()) if (not a.rows and ref_path.exists()) else None
    if a.rows:
        rows = json.loads(Path(a.rows).read_text())["rows"]
    elif ref is not None:
        rows = ref["rows"]
        if isinstance(rows, str):                          # rows.json beside the reference
            rows = json.loads((ref_path.parent / rows).read_text())["rows"]
    else:
        raise SystemExit(f"no reference at {ref_path}; pass --ref")
    if a.limit:
        rows = rows[: a.limit]
    info, got = run(a.model, rows, a.backend, efforts, cpu=a.cpu)
    if a.write_ref:
        Path(a.write_ref).write_text(json.dumps({"rows": rows, "info": info, "outputs": got}))
        print("wrote", a.write_ref)
        return 0
    result = {"machine": machine(), "info": info, "accuracy": accuracy(rows, got, efforts),
              "latency_ms": {}, "vs": {}}
    for e in efforts:
        ms = [x["ms"] for x in got[e]]
        tm = [x["ms"] for x, r in zip(got[e], rows) if not r.get("images")]
        im = [x["ms"] for x, r in zip(got[e], rows) if r.get("images")]
        result["latency_ms"][e] = {"median": round(statistics.median(ms), 1),
                                   "p90": round(float(np.quantile(ms, .9)), 1),
                                   "median_text": round(statistics.median(tm), 1) if tm else None,
                                   "median_images": round(statistics.median(im), 1) if im else None,
                                   "mean_depth": round(float(np.mean([x["depth"] for x in got[e]
                                                                      if x["depth"] is not None] or [0])), 2)}
    for which, title in (("torch_bf16", "bf16 PyTorch release (GPU)"), ("mlx_linux", "this build, MLX on Linux")):
        if ref and which in ref:
            out = {e: v[: len(rows)] for e, v in ref[which].items()}
            result["vs"][which] = compare(rows, got, out, efforts)
    print()
    print(f"{info['name']}: mlx {info.get('mlx')} on {info['device']}; load {info['load_s']} s; "
          f"peak memory {info.get('peak_memory_gb')} GB (after load {info.get('weights_gb_active_after_load')} GB)")
    print(f"{'effort':7s} {'median ms':>9s} {'p90 ms':>8s} {'text ms':>8s} {'image ms':>9s} {'depth':>6s} {'acc':>6s}"
          + "".join(f" {'agree ' + w:>17s} {'|dp| ' + w:>16s}" for w in result["vs"]))
    for e in efforts:
        lat = result["latency_ms"][e]
        line = (f"{e:7s} {lat['median']:9.1f} {lat['p90']:8.1f} {lat['median_text'] or 0:8.1f} "
                f"{lat['median_images'] or 0:9.1f} {lat['mean_depth']:6.1f} {100 * result['accuracy'][e]:5.1f}%")
        for w, c in result["vs"].items():
            x = c.get(f"{e}.all", {})
            line += f" {100 * x.get('agree', float('nan')):16.1f}% {x.get('dp_mean', float('nan')):16.4f}"
        print(line)
    for x in info.get("memory_by_length") or []:
        print(f"one question on a {x['tokens']}-token document: peak memory {x['peak_gb']} GB, {x['ms']} ms")
    for w, c in result["vs"].items():
        im = [c.get(f"{e}.images") for e in efforts if c.get(f"{e}.images")]
        if im:
            print(f"images vs {w}: agree " + " / ".join(f"{100 * x['agree']:.0f}%" for x in im)
                  + f" ({im[0]['n']} rows, efforts {'/'.join(efforts)})")
    if a.json:
        Path(a.json).write_text(json.dumps({**result, "outputs": got}, indent=1))
        print("saved", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
