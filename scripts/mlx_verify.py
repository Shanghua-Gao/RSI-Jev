"""Check an MLX checkpoint against what was measured, bit for bit, with MLX itself (CPU build is enough).

    python scripts/mlx_verify.py MLXDIR --pkg PKG [--codes CODES]

For every quantized tensor of MLXDIR/tower.safetensors (MLX's own `mx.load`):
  * from --codes (a precomputed quantization, e.g. GPTQ): the codes MLX's words hold equal the
    file's codes, the stored scales / biases equal the file's in bf16, and `mx.dequantize`
    equals the effective bf16 weight the codes define (rsijev.mlx.quant.dequantize, the weight
    the quality numbers were measured with);
  * otherwise (round-to-nearest): the words, scales and biases equal `mx.quantize` of the
    package's bf16 weight.
Every other tower tensor equals the package's. Heads: equal to the package's (fp32), or to
their bf16 rounding when mlx.json says bf16. Then SHA256SUMS. Exit status 0 only if all pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    import mlx.core as mx
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    from rsijev.mlx.convert import HEAD_FILES
    from rsijev.mlx.quant import dequantize

    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("mlx_dir")
    ap.add_argument("--pkg", required=True, help="the source release package")
    ap.add_argument("--codes", default=None, help="the precomputed codes the tower was packed from")
    ap.add_argument("--json", default=None, help="write the per-tensor result here")
    a = ap.parse_args(argv)
    mx.set_default_device(mx.cpu)
    out, pkg = Path(a.mlx_dir), Path(a.pkg)
    rec = json.loads((out / "mlx.json").read_text())
    T = mx.load(str(out / "tower.safetensors"))
    codes = load_file(a.codes) if a.codes else {}
    tb = rec["tower"]
    eb = rec["embed_tokens"]
    res = {"checked": 0, "failed": [], "kinds": {}}

    def f32(x):
        return np.array(x.astype(mx.float32))

    def fail(k, why):
        res["failed"].append(f"{k}: {why}")
        print("FAIL", k, why, flush=True)

    with safe_open(str(pkg / "tower.safetensors"), "pt") as fh:
        src_keys = set(fh.keys())
        for key in sorted(src_keys):
            name = key[: -len(".weight")] if key.endswith(".weight") else None
            if name and name + ".scales" in T:
                w = T[key]
                s, b = T[name + ".scales"], T[name + ".biases"]
                emb = key == "embed_tokens.weight"
                bits, g = (eb["bits"], eb["group_size"]) if emb else (tb["bits"], tb["group_size"])
                deq = f32(mx.dequantize(w, s, b, group_size=g, bits=bits))
                if not emb and codes:
                    q = codes[name + ".q"]
                    cs, cb = codes[name + ".s"].bfloat16(), codes[name + ".b"].bfloat16()
                    per = 32 // bits
                    words = np.array(w).view(np.uint32)
                    un = np.stack([(words >> (bits * i)) & ((1 << bits) - 1) for i in range(per)], -1)
                    un = un.reshape(q.shape)
                    ok_codes = np.array_equal(un, q.numpy())
                    ok_s = np.array_equal(f32(s), cs.float().numpy()) and np.array_equal(f32(b), cb.float().numpy())
                    ref = dequantize(q, cs, cb, g).float().numpy()
                    ok_deq = np.array_equal(deq, ref)
                    kind = "codes"
                    if not (ok_codes and ok_s and ok_deq):
                        fail(key, f"codes {ok_codes} scales/biases {ok_s} dequantize {ok_deq}")
                else:
                    wt = fh.get_tensor(key).to(torch.bfloat16)
                    wm = mx.array(wt.float().numpy()).astype(mx.bfloat16)
                    qm, sm, bm = mx.quantize(wm, group_size=g, bits=bits)
                    ok = (np.array_equal(np.array(qm), np.array(w)) and np.array_equal(f32(sm), f32(s))
                          and np.array_equal(f32(bm), f32(b)))
                    kind = "rtn"
                    if not ok:
                        fail(key, "differs from mx.quantize of the package weight")
                res["kinds"][kind] = res["kinds"].get(kind, 0) + 1
            elif key in T:
                if not np.array_equal(f32(T[key]), fh.get_tensor(key).float().numpy()):
                    fail(key, "copied tensor differs")
                res["kinds"]["copied"] = res["kinds"].get("copied", 0) + 1
            else:
                res["kinds"]["dropped"] = res["kinds"].get("dropped", 0) + 1
                continue
            res["checked"] += 1
    extra = {k for k in T if not k.endswith((".scales", ".biases"))} - src_keys
    if extra:
        fail("tower", f"keys not in the package: {sorted(extra)[:5]}")
    if codes:
        used = {k.rsplit(".", 1)[0] for k in codes}
        missing = [n for n in used if n + ".scales" not in T]
        if missing:
            fail("codes", f"{len(missing)} matrices of the codes file not quantized in the tower")
    bf16_heads = "bf16" in rec.get("heads", "")
    for f in HEAD_FILES + ("calibration.safetensors",):
        if not (pkg / f).exists():
            continue
        A, B = load_file(str(pkg / f)), load_file(str(out / f))
        want_bf16 = bf16_heads and f in HEAD_FILES
        for k, v in A.items():
            exp = v.bfloat16() if (want_bf16 and v.is_floating_point()) else v
            if B[k].dtype != exp.dtype or not torch.equal(B[k], exp):
                fail(f"{f}:{k}", f"{B[k].dtype} vs expected {exp.dtype}")
        res["kinds"][f] = len(A)
    for line in (out / "SHA256SUMS").read_text().splitlines():
        h, name = line.split("  ", 1)
        d = hashlib.sha256()
        with open(out / name, "rb") as fh2:
            for chunk in iter(lambda: fh2.read(1 << 24), b""):
                d.update(chunk)
        if d.hexdigest() != h:
            fail(name, "SHA256SUMS mismatch")
    res["ok"] = not res["failed"]
    print(json.dumps({k: v for k, v in res.items() if k != "failed"}), "failed", len(res["failed"]))
    print("ALL OK" if res["ok"] else "FAILED", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
