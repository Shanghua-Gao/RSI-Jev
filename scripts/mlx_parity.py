"""MLX vs PyTorch parity on the quantization study's 2,200 rows: python scripts/mlx_parity.py {mlx,torch,compare,effort} ...

The rows are tools/quant/study.py's (quant-mac branch), rebuilt the same way: eval_suite_v2
(13 sets x 100 questions) + eval_final_v2 (5 x 60) + CLINC150+OOS (151 options, 300) as text,
and eval_vision_v1 (5 x 60 image questions), seed 0, canonical option order, the release's
own encoder (max_length 32,768, own-token pooling). Every exit (16 / 20 / 32) is read from
one pass, raw (no calibration), as the study's reference was.

    # logits from an MLX checkpoint (bf16 or 8-bit), any MLX device
    python scripts/mlx_parity.py mlx --ckpt MLXDIR --data DATA --out mlx_8bit.npz
    # logits from the PyTorch path (bf16; --qdq 8:64 = the MLX 8-bit weights, simulated)
    python scripts/mlx_parity.py torch --pkg PKG --data DATA --out torch_q8.npz --qdq 8:64
    # agreement / |dp| per exit and subset, probabilities after the shipped per-exit T
    python scripts/mlx_parity.py compare A.npz B.npz --pkg PKG [--json out.json]
    # effort paths on MLX: low / medium / high equal forced exits, auto = the cascade
    python scripts/mlx_parity.py effort --ckpt MLXDIR --data DATA --exits mlx_8bit.npz --out effort.json

DATA holds eval_suite_v2/, eval_final_v2/, clinc_items300.jsonl and eval_vision_v1/ (with the
300 images used). A reference .npz may also be the study's ref_logits.pt (H200, bf16).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ------------------------------------------------------------------------------- rows
def _load_set(path: Path) -> dict:
    from rsijev.contract import load_cases
    if path.is_dir():
        man = json.loads((path / "manifest.json").read_text())
        out = {}
        for name, b in man["benchmarks"].items():
            p = path / b["file"]
            if b.get("sha256") and hashlib.sha256(p.read_bytes()).hexdigest() != b["sha256"]:
                raise SystemExit(f"{p.name}: sha256 mismatch")
            out[name] = load_cases(str(p))
        return out
    return {path.stem: load_cases(str(path))}


def text_rows(data: Path, per_suite=100, per_final=60):
    from rsijev.contract import Case, Question
    rows = []
    for tag, sub, per in (("suite", "eval_suite_v2", per_suite), ("final", "eval_final_v2", per_final)):
        for name, cases in _load_set(data / sub).items():
            cs = list(cases)
            random.Random(0).shuffle(cs)
            n = 0
            for c in cs:
                for q in c.questions:
                    if n >= per or len(q.options) > 160:
                        continue
                    rows.append((c, q, f"{tag}.{name}"))
                    n += 1
    for line in open(data / "clinc_items300.jsonl"):
        it = json.loads(line)
        qd = it["payload"]["questions"]["q1"]
        keys = list(qd["criteria"])
        Q = Question(key="q1", mode="choice", instructions=qd["instructions"], options=tuple(keys),
                     criteria=dict(qd["criteria"]))
        c = Case(case_id=it["id"], source="clinc_eval", state="", questions=(Q,),
                 gold={"q1": tuple(1.0 if o == it["gold"] else 0.0 for o in keys)})
        rows.append((c, Q, "clinc151"))
    return rows


def vision_rows(data: Path, per=60):
    from rsijev.contract import Case, Question
    root = data / "eval_vision_v1"
    man = json.loads((root / "manifest.json").read_text())["benchmarks"]
    out = []
    for b, m in man.items():
        cs = []
        for line in (root / m["file"]).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            qs = tuple(Question(q["key"], q["mode"], q["instructions"], tuple(q["options"]), q["criteria"])
                       for q in r["questions"])
            cs.append((Case(r["case_id"], r["source"], r["state"], qs,
                            {k: tuple(v) for k, v in r["gold"].items()}), r.get("images", [])))
        random.Random(0).shuffle(cs)
        n = 0
        for c, ims in cs:
            for q in c.questions:
                if n >= per or len(q.options) > 160:
                    continue
                out.append((c, q, ims, b))
                n += 1
    return root, out


def gold_index(c, q) -> int:
    g = c.gold[q.key]
    return max(range(len(g)), key=g.__getitem__)


def token_budget_batches(lengths, tok_budget, batch_size):
    """scripts/dump_exits.token_budget_batches: sorted by length, rows x longest <= budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[i] > tok_budget or len(cur) >= batch_size:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def encoder(pkg_or_ckpt: Path):
    from rsijev.encode import EncodeConfig
    from serve.release import own_token_pool, serving_encoder
    meta = json.loads((pkg_or_ckpt / "meta.json").read_text())
    spec = meta["spec"]
    cap, policy = serving_encoder(spec)
    return EncodeConfig(layout=spec["layout"], option_pool=spec["option_pool"],
                        option_order="canonical", max_length=cap, truncate=policy,
                        option_pool_own_tokens=own_token_pool(spec, meta))


def _save(out, Z, y, tags, extra=None):
    np.savez_compressed(out, **{f"z{L}": z for L, z in Z.items()}, y=np.asarray(y),
                        tags=np.asarray(tags), **(extra or {}))
    log("wrote", out)


# ------------------------------------------------------------------------------- MLX
def run_mlx(a):
    import mlx.core as mx
    from PIL import Image
    from transformers import AutoTokenizer

    from rsijev.encode import encode_question
    from rsijev.image_text import IMAGE_PAD, expand_state
    from rsijev.mlx.collate import collate, unpermute
    from rsijev.mlx.model import MLXDecisionModel
    from rsijev.mlx.vision import ImagePrep, mrope_positions
    if a.cpu:
        mx.set_default_device(mx.cpu)
    ck, data = Path(a.ckpt), Path(a.data)
    model = MLXDecisionModel(ck, vision=not a.text_only)
    tok = AutoTokenizer.from_pretrained(str(ck))
    enc = encoder(ck)
    log("loaded", ck.name, "device", mx.default_device(), "exits", model.exit_indices(),
        "tower", model.mlx_record.get("tower"))
    text = text_rows(data)
    vroot, vis = vision_rows(data)
    if a.limit:
        text = [r for t in sorted({r[2] for r in text}) for r in [x for x in text if x[2] == t][: a.limit]]
        vis = vis[: a.limit]
    if a.text_only:
        vis = []
    tags = [t for *_, t in text] + [f"vis.{b}" for *_, b in vis]
    y = [gold_index(c, q) for c, q, _ in text] + [gold_index(c, q) for c, q, _, _ in vis]
    n = len(tags)
    exits = model.exit_indices()
    Z = {L: np.full((n, 160), -np.inf, dtype=np.float32) for L in exits}
    encd = [encode_question(tok, c.state, q, enc) for c, q, _ in text]
    lengths = [len(e["input_ids"]) for e in encd]
    log(f"text rows {len(text)}: {sum(lengths)} tokens, longest {max(lengths)}")
    batches = token_budget_batches(lengths, a.tok_budget, a.batch_size)
    for bi, ix in enumerate(batches):
        b = collate(tok, [encd[i] for i in ix], 160)
        out = model.forward_exits(b, calibrate=False)
        mx.eval(list(out.values()))
        for L, z in out.items():
            Z[L][ix] = unpermute(np.array(z.astype(mx.float32)), b["option_perm"], b["option_mask"])
        if bi % 20 == 0:
            log(f"text batch {bi + 1}/{len(batches)}")
    if vis:
        prep = ImagePrep(ck, budget=1024)
        log("image processor:", prep.backend)
        import dataclasses
        venc = dataclasses.replace(enc, max_length=max(enc.max_length, 1024 + 2048))
        pad = tok.convert_tokens_to_ids(IMAGE_PAD)
        off = len(text)
        for i in range(0, len(vis), 8):
            part = vis[i:i + 8]
            rows, feats, grids = [], [], []
            for c, q, ims, _ in part:
                pil = [Image.open(vroot / p).convert("RGB") for p in ims]
                pv, grid, ntok = prep(pil)
                e = encode_question(tok, expand_state(c.state, ntok), q, venc)
                assert sum(1 for t in e["input_ids"] if t == pad) == sum(ntok)
                rows.append(e)
                feats.append(model.image_embeds(pv, grid))
                grids.append(grid)
            b = collate(tok, rows, 160)
            b["position_ids"] = mrope_positions(b["input_ids"], b["attention_mask"],
                                                np.concatenate(grids), model.image_token_id)
            out = model.forward_exits(b, image_embeds=mx.concatenate(feats, axis=0), calibrate=False)
            mx.eval(list(out.values()))
            for L, z in out.items():
                Z[L][off + i: off + i + len(part)] = unpermute(np.array(z.astype(mx.float32)),
                                                               b["option_perm"], b["option_mask"])
        log(f"vision rows {len(vis)} done")
    _save(a.out, Z, y, tags, {"elapsed_s": np.asarray(time.time() - T0)})


# ------------------------------------------------------------------------------- torch
def run_torch(a):
    import torch
    from PIL import Image

    from rsijev.encode import collate, encode_question, unpermute_logits
    from rsijev.vision import ImagePrep, VisionConfig, encode_vision_question, vision_collate
    from serve.release import load_release
    from serve.runtime import keep_fused_kernels_off
    dev = a.device
    keep_fused_kernels_off(dev)
    pkg, data = Path(a.pkg), Path(a.data)
    model, tok, enc, meta = load_release(pkg, dev, infer_dtype=torch.bfloat16)
    model.cal_mode = "none"
    enc.option_order = "canonical"
    if a.qdq:
        from rsijev.mlx.convert import is_quantized_linear
        from rsijev.mlx.quant import qdq
        bits, group = (int(x) for x in a.qdq.split(":"))
        n = 0
        with torch.no_grad():
            for name, p in model.tower.named_parameters():
                key = name.split("model.", 1)[-1] if name.startswith("model.") else name
                if is_quantized_linear(key):
                    p.copy_(qdq(p.data, bits, group))
                    n += 1
        log(f"tower linears replaced by their {bits}-bit g{group} MLX weights: {n}")
    text = text_rows(data)
    vroot, vis = vision_rows(data)
    if a.limit:
        text = [r for t in sorted({r[2] for r in text}) for r in [x for x in text if x[2] == t][: a.limit]]
        vis = vis[: a.limit]
    tags = [t for *_, t in text] + [f"vis.{b}" for *_, b in vis]
    y = [gold_index(c, q) for c, q, _ in text] + [gold_index(c, q) for c, q, _, _ in vis]
    exits = model.exit_indices()
    Z = {L: np.full((len(tags), 160), -np.inf, dtype=np.float32) for L in exits}
    encd = [encode_question(tok, c.state, q, enc) for c, q, _ in text]
    batches = token_budget_batches([len(e["input_ids"]) for e in encd], a.tok_budget, a.batch_size)
    with torch.no_grad():
        for ix in batches:
            b = collate(tok, [encd[i] for i in ix], max_options=160, device=dev)
            for L, z in model.forward_exits(**b).items():
                Z[L][ix] = unpermute_logits(z.float(), b["option_perm"], b["option_mask"]).cpu().numpy()
        log("text done")
        prep = ImagePrep(meta["weights_source"], VisionConfig(image_token_budget=1024),
                         revision=(meta.get("vision") or {}).get("revision"))
        import dataclasses
        e2 = dataclasses.replace(enc, max_length=max(enc.max_length, 1024 + 2048))
        ex = [encode_vision_question(tok, prep, c.state, [Image.open(vroot / p).convert("RGB") for p in ims], q, e2)
              for c, q, ims, _ in vis]
        off = len(text)
        for i in range(0, len(ex), 8):
            bt = vision_collate(tok, ex[i:i + 8], 160, device=dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=a.autocast):
                out = model.forward_exits(**bt)
            for L, z in out.items():
                Z[L][off + i: off + i + z.shape[0]] = unpermute_logits(
                    z.float(), bt["option_perm"], bt["option_mask"]).cpu().numpy()
        log("vision done")
    _save(a.out, Z, y, tags)


# ------------------------------------------------------------------------------- compare
def load_logits(path):
    path = str(path)
    if path.endswith(".pt"):
        import torch
        r = torch.load(path)
        Z = {int(L): z.numpy() for L, z in r["z"].items()}
        return Z, r["y"].numpy(), None
    d = np.load(path)
    Z = {int(k[1:]): d[k] for k in d.files if k.startswith("z") and k[1:].isdigit()}
    return Z, d["y"], d["tags"] if "tags" in d.files else None


def temps_of(pkg: Path) -> dict:
    return {int(k): float(v) for k, v in json.loads((pkg / "calibration.json").read_text())["T"].items()}


def softmax(z):
    z = z.astype(np.float64)
    m = np.max(z, -1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(-1, keepdims=True)


def compare(A, B, y, tags, temps) -> dict:
    groups = {"all": np.ones(len(tags), bool),
              "text": np.array([not t.startswith("vis.") and t != "clinc151" for t in tags]),
              "clinc151": np.array([t == "clinc151" for t in tags]),
              "vision": np.array([t.startswith("vis.") for t in tags])}
    out = {}
    for L in sorted(A):
        pa, pb = softmax(A[L] / temps[L]), softmax(B[L] / temps[L])
        aa, ab = pa.argmax(-1), pb.argmax(-1)
        dp = np.abs(pa - pb).max(-1)
        for g, m in groups.items():
            if not m.any():
                continue
            out[f"{g}@{L}"] = {"n": int(m.sum()), "agree": round(float((aa[m] == ab[m]).mean()), 5),
                               "flips": int((aa[m] != ab[m]).sum()),
                               "dp_mean": round(float(dp[m].mean()), 5),
                               "dp_p99": round(float(np.quantile(dp[m], .99)), 4),
                               "dp_max": round(float(dp[m].max()), 4),
                               "acc_a": round(float((aa[m] == y[m]).mean()), 4),
                               "acc_b": round(float((ab[m] == y[m]).mean()), 4)}
    return out


def run_compare(a):
    A, y, tags = load_logits(a.a)
    B, yb, tb = load_logits(a.b)
    tags = tags if tags is not None else tb
    assert tags is not None, "one side must carry tags (an .npz from this script)"
    if not np.array_equal(y, yb):
        raise SystemExit("the two files hold different rows (gold labels differ)")
    res = compare(A, B, y, list(tags), temps_of(Path(a.pkg)))
    print(f"{'subset@exit':16s} {'n':>5s} {'agree':>7s} {'flips':>5s} {'dp_mean':>8s} {'dp_p99':>7s} {'dp_max':>7s} {'acc_a':>6s} {'acc_b':>6s}")
    for k, v in res.items():
        print(f"{k:16s} {v['n']:5d} {100 * v['agree']:6.2f}% {v['flips']:5d} {v['dp_mean']:8.5f} "
              f"{v['dp_p99']:7.4f} {v['dp_max']:7.4f} {100 * v['acc_a']:5.1f} {100 * v['acc_b']:5.1f}")
    if a.json:
        Path(a.json).write_text(json.dumps({"a": a.a, "b": a.b, "results": res}, indent=1))


# ------------------------------------------------------------------------------- effort
def cascade(Z: dict, temps: dict, taus: dict, exits: list, modes=None) -> np.ndarray:
    """The exit each row stops at under scalar-temperature calibration: the first aux exit
    whose top-1 softmax(z / T) reaches its tau, else the main exit."""
    n = next(iter(Z.values())).shape[0]
    route = np.full(n, exits[-1])
    open_ = np.ones(n, bool)
    for L in exits[:-1]:
        conf = softmax(Z[L] / temps[L]).max(-1)
        stop = open_ & (conf >= taus[L])
        route[stop] = L
        open_ &= ~stop
    return route


def run_effort(a):
    """Every text row through the served staged path at each effort (one question per
    request: the staged path runs per batch exactly as score_adaptive does), against the
    forced exits read from one pass (--exits, this checkpoint's own `mlx` dump) and the
    cascade simulated on those logits and on a PyTorch reference (--ref)."""
    import mlx.core as mx
    from transformers import AutoTokenizer

    from rsijev.encode import encode_question
    from rsijev.mlx.collate import collate, unpermute
    from rsijev.mlx.serving import load_for_serving, policy_for
    if a.cpu:
        mx.set_default_device(mx.cpu)
    ck, data = Path(a.ckpt), Path(a.data)
    s = load_for_serving(ck, vision=False)
    model = s.model
    tok = AutoTokenizer.from_pretrained(str(ck))
    enc = encoder(ck)
    text = text_rows(data)
    if a.limit:
        text = text[: a.limit]
    encd = [encode_question(tok, c.state, q, enc) for c, q, _ in text]
    temps = {L: float(np.exp(model.calibration.b["cal_logT"])) if L == model.cfg.exit_layer
             else float(np.exp(model.effort_base.cal[L]["logT"])) for L in model.exit_indices()}
    T_ship = temps_of(ck)
    Zx, _, _ = load_logits(a.exits)
    Zx = {L: z[: len(text)] for L, z in Zx.items()}
    batches = token_budget_batches([len(e["input_ids"]) for e in encd], a.tok_budget, a.batch_size)
    res = {"temps_served": temps, "temps_shipped": T_ship}
    exits = model.exit_indices()
    for effort in ("low", "medium", "high", "auto"):
        P = np.zeros((len(text), 160), dtype=np.float64)
        depth = np.zeros(len(text), dtype=int)
        pol = policy_for(model, effort) if effort != "high" else None
        for ix in batches:
            b = collate(tok, [encd[i] for i in ix], 160)
            if pol is None:
                z, d = model.forward(b), [exits[-1]] * len(ix)
            else:
                z, d = model.staged(b, pol)
            P[ix] = softmax(unpermute(np.array(z.astype(mx.float32)), b["option_perm"], b["option_mask"]))
            depth[ix] = d
        if effort in ("low", "medium", "high"):
            L = {"low": exits[0], "medium": exits[-2], "high": exits[-1]}[effort]
            ref = softmax(Zx[L] / temps[L])
            r = {"exit": L, "all_at_exit": bool((depth == L).all()),
                 "argmax_eq_forced": float((P.argmax(-1) == ref.argmax(-1)).mean()),
                 "max_abs_dp_vs_forced": float(np.abs(P - ref).max())}
        else:
            taus = {int(k): float(v) for k, v in (model.effort_auto_taus or {}).items()}
            sim = cascade(Zx, temps, taus, exits)
            r = {"taus": taus, "route_eq_cascade_same_logits": float((sim == depth).mean()),
                 "shares": {int(L): float((depth == L).mean()) for L in exits},
                 "mean_depth": float(depth.mean())}
            if a.ref:
                Zr, _, _ = load_logits(a.ref)
                Zr = {L: z[: len(text)] for L, z in Zr.items()}
                simr = cascade(Zr, temps, taus, exits)
                r.update({"route_eq_torch_cascade": float((simr == depth).mean()),
                          "torch_shares": {int(L): float((simr == L).mean()) for L in exits},
                          "torch_mean_depth": float(simr.mean()),
                          "answer_eq_torch_cascade": float((P.argmax(-1) == np.array(
                              [softmax(Zr[int(L)][i:i + 1] / temps[int(L)])[0].argmax()
                               for i, L in enumerate(simr)])).mean())})
        res[effort] = r
        log(effort, json.dumps(r))
    Path(a.out).write_text(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("mlx")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--tok-budget", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(func=run_mlx)
    p = sub.add_parser("torch")
    p.add_argument("--pkg", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--qdq", default=None, help="bits:group, e.g. 8:64")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--autocast", action=argparse.BooleanOptionalAction, default=True,
                   help="image rows under bf16 autocast, as the study ran them")
    p.add_argument("--tok-budget", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(func=run_torch)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--pkg", required=True, help="for the shipped per-exit temperatures")
    p.add_argument("--json", default=None)
    p.set_defaults(func=run_compare)
    p = sub.add_parser("effort")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--exits", required=True, help="this checkpoint's `mlx` dump (.npz)")
    p.add_argument("--ref", default=None, help="a PyTorch dump (.npz or ref_logits.pt) for the cascade")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--tok-budget", type=int, default=65536)
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(func=run_effort)
    a = ap.parse_args(argv)
    return a.func(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
