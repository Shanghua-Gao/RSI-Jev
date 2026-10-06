"""Refit the early-exit heads of a multi-exit checkpoint on its frozen tower (v6.0-VL stage 2).

The tower and the main head are not trained (their files' sha256 are checked before and
after). The exit heads start from the heads stage 1 trained, which learned against a moving
tower, and are refit on the converged one:

  * forward = model.forward_exits(aux_detach=True), the stage-1 training readout;
  * loss = sum over the early exits L of soft-CE(gold) + distill x KL(p_main || p_L) at
    T = 1, the main exit's distribution detached (fit.py's aux term); --distill -1 takes
    the stage-1 spec's fit_extra.aux_distill (0.5 for v6.0-VL);
  * AdamW (weight decay = the spec's head_weight_decay), lr --lr with a linear warmup of
    --warmup steps then cosine to 0, gradient clip 1.0;
  * training rows: the checkpoint's own corpus and sources (meta.json spec), minus its
    in-distribution holdout (sha256(case_id) % 10 == 0, run_arm_lib's rule, so the
    calibration DEV pool stays clean), minus every case whose state also appears in an
    evaluation set (--exclude, and with --exclude-public the typed-decisions test split
    and MMLU-Pro 1k), compared by whitespace- and case-normalised sha256. The rows are
    water-filled to --target-q questions over the sources, options shuffled except the
    spec's canonical_order_sources, length-sorted into token-budget batches in a seeded
    order.

Writes OUT/aux_scorers.safetensors (keys "<exit>.<param>", the layout fit.py and
serve/release.py use) and OUT/train_log.json. With --link-into DIR it also makes DIR a
checkpoint: every file of --ckpt linked, except aux_scorers.safetensors, which links to
the refit heads (the main path stays bit-identical).

  python scripts/refit_exit_heads.py --ckpt D/trunk/s17 --corpus D/text-corpus --out D/refit \\
      --steps 3000 --exclude-public --exclude D/eval_suite_v2 --exclude D/eval_final_v2 \\
      --link-into D/trunk-refit/s17
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def nhash(s: str) -> str:
    """sha256 of a state with whitespace collapsed and lower-cased."""
    return hashlib.sha256(re.sub(r"\s+", " ", str(s)).strip().lower().encode()).hexdigest()


def held(cid: str) -> bool:
    """run_arm_lib's in-distribution holdout (one case in ten, by a hash of its id)."""
    return int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 10 == 0


def fsha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()[:16]


def waterfill(avail: dict, total: int) -> dict:
    """Per-source quotas summing to at most `total`: every source gets an equal share,
    sources with fewer questions than their share give the rest to the others."""
    q, left, rest = {}, total, dict(avail)
    while rest and left > 0:
        share = left // len(rest) or 1
        small = {k: v for k, v in rest.items() if v <= share}
        if not small:
            for k in rest:
                q[k] = share
            break
        for k, v in small.items():
            q[k] = v
            left -= v
            del rest[k]
    return q


def token_batches(lengths: list, tok_budget: int, max_bs: int, rng: random.Random) -> list:
    """Row indices sorted by length and cut into batches of at most max_bs rows whose
    rows x longest row stays within tok_budget, in a seeded random order."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur = [], []
    for i in order:
        if cur and ((len(cur) + 1) * lengths[i] > tok_budget or len(cur) >= max_bs):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def warmup_cosine(warmup: int, steps: int):
    """The LR multiplier: linear warmup over `warmup` steps, times a cosine to 0 over `steps`."""
    return lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))


def refit_loss(exits: dict, main: int, gold: torch.Tensor, option_perm, option_mask, distill: float):
    """(loss, per-exit record) of one batch: per early exit, soft-CE to gold plus
    distill x KL(p_main || p_L), the main exit detached. exits: {exit: presented-order
    logits}; gold: canonical order, rows normalised."""
    from rsijev.encode import unpermute_logits
    om = option_mask
    zm = unpermute_logits(exits[main].float(), option_perm, om).masked_fill(~om, float("-inf"))
    t_lp = torch.log_softmax(zm.detach(), -1).masked_fill(~om, 0.0)
    loss, rec = 0.0, {}
    for L, z in exits.items():
        if L == main:
            continue
        al = unpermute_logits(z.float(), option_perm, om).masked_fill(~om, float("-inf"))
        s_lp = torch.log_softmax(al, -1).masked_fill(~om, 0.0)
        ce = -(gold * s_lp).sum(-1).mean()
        kl = (t_lp.exp() * om * (t_lp - s_lp)).sum(-1).mean()
        loss = loss + ce + distill * kl
        rec[f"ce{L}"], rec[f"kl{L}"] = round(float(ce.detach()), 4), round(float(kl.detach()), 4)
    return loss, rec


def link_checkpoint(src: Path, out: Path, aux: Path) -> None:
    """out = every file of src linked, except aux_scorers.safetensors -> aux."""
    src, out = Path(src).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name == "aux_scorers.safetensors" or not f.is_file():
            continue
        d = out / f.name
        if not (d.exists() or d.is_symlink()):
            os.symlink(f, d)
    d = out / "aux_scorers.safetensors"
    if d.exists() or d.is_symlink():
        d.unlink()
    os.symlink(Path(aux).resolve(), d)
    print(f"[refit] checkpoint {out.name}: main path from {src.name}, exit heads refit", flush=True)


def _excluded_states(paths: list, public: bool) -> set:
    from rsijev.contract import load_cases
    cases = []
    if public:
        from rsijev.targets import load_mmlu_pro_1k, load_typed_decisions
        cases += load_typed_decisions("test") + load_mmlu_pro_1k()
    for p in paths:
        p = Path(p)
        if p.is_dir():
            man = p / "manifest.json"
            files = ([p / b["file"] for b in json.loads(man.read_text())["benchmarks"].values()]
                     if man.exists() else sorted(p.glob("*.jsonl")))
        else:
            files = [p]
        for f in files:
            cases += load_cases(str(f))
    return {nhash(c.state) for c in cases}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="the stage-1 checkpoint (tower, scorer, aux_scorers, meta.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", required=True, help="the stage-1 corpus directory (<source>.jsonl)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--distill", type=float, default=-1.0, help="-1: the spec's fit_extra.aux_distill")
    ap.add_argument("--target-q", type=int, default=96000)
    ap.add_argument("--tok-budget", type=int, default=32768)
    ap.add_argument("--max-bs", type=int, default=32)
    ap.add_argument("--exclude", action="append", default=[],
                    help="a case file, or a directory of them (manifest.json or *.jsonl), whose states "
                         "must not be trained on; repeatable")
    ap.add_argument("--exclude-public", action="store_true",
                    help="also exclude the typed-decisions test split and MMLU-Pro 1k")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: questions per source")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--link-into", default="", help="make this directory the refit checkpoint")
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(a.ckpt)
    sha0 = {f: fsha(ck / f) for f in ("tower.safetensors", "scorer.safetensors")}
    print(f"frozen sha before {sha0}", flush=True)

    from rsijev.contract import load_cases
    from rsijev.encode import collate, encode_question
    from serve.release import load_release
    model, tok, enc, meta = load_release(str(ck), dev, infer_dtype=torch.bfloat16 if dev == "cuda" else None,
                                         vision=False)
    spec = meta["spec"]
    fe = spec.get("fit_extra") or {}
    distill = float(fe.get("aux_distill", 0.5)) if a.distill < 0 else a.distill
    if getattr(model, "aux_scorers", None) is None:
        raise SystemExit("checkpoint has no early exits (spec.arch_extra.aux_exits)")
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.aux_scorers.parameters():
        p.requires_grad_(True)
    heads = list(model.aux_scorers.parameters())
    a0 = {k: v.detach().clone() for k, v in model.aux_scorers.state_dict().items()}
    opt = torch.optim.AdamW([{"params": heads, "lr": a.lr}], weight_decay=float(spec.get("head_weight_decay") or 0.0))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(a.warmup, a.steps))
    MAIN = int(model.cfg.exit_layer)
    print(f"loaded {ck.name} exits {model.exit_indices()} distill {distill} lr {a.lr} steps {a.steps} "
          f"enc max_length {enc.max_length} own_tokens {enc.option_pool_own_tokens} ({time.time() - t0:.0f}s)",
          flush=True)

    # ---- training rows
    corpus = Path(a.corpus)
    srcs = [s for s in spec["sources"].split(",") if s]
    bench = _excluded_states(a.exclude, a.exclude_public)
    cand, avail, nheld, nbench = {}, {}, 0, 0
    for s in srcs:
        f = corpus / f"{s}.jsonl"
        if not f.exists():
            print(f"  WARNING source {s}: no {f.name}", flush=True)
            continue
        keep = []
        for c in load_cases(str(f)):
            if held(c.case_id):
                nheld += 1
                continue
            if nhash(c.state) in bench:
                nbench += 1
                continue
            keep.append(c)
        keep.sort(key=lambda c: hashlib.sha256(("refit:" + c.case_id).encode()).hexdigest())
        cand[s], avail[s] = keep, sum(len(c.questions) for c in keep)
    quota = {s: a.limit for s in cand} if a.limit else waterfill(avail, a.target_q)
    canon = tuple(x for x in (fe.get("canonical_order_sources") or "").split(",") if x)
    enc_sh = dataclasses.replace(enc, option_order=spec.get("option_order", "shuffled"))
    enc_cn = dataclasses.replace(enc, option_order="canonical")
    rows, encd, skipped = [], [], 0
    for s, cs in cand.items():
        n = 0
        for c in cs:
            if n >= quota.get(s, 0):
                break
            for q in c.questions:
                if len(q.options) > int(spec["max_options"]):
                    skipped += 1
                    continue
                try:
                    encd.append(encode_question(tok, c.state, q,
                                                enc_cn if c.source.startswith(canon or ("\0",)) else enc_sh, rng))
                    rows.append((c, q))
                    n += 1
                except ValueError:
                    skipped += 1
    print(f"TRAIN {len(rows)} q from {len(cand)} sources (holdout cases dropped {nheld}, evaluation-state "
          f"cases dropped {nbench}, skipped {skipped}) ({time.time() - t0:.0f}s)", flush=True)
    batches = token_batches([len(e["input_ids"]) for e in encd], a.tok_budget, a.max_bs, rng)
    print(f"{len(batches)} batches; steps {a.steps} (epochs {a.steps / max(1, len(batches)):.2f})", flush=True)

    model.eval()
    model.aux_scorers.train(True)
    mo = int(spec["max_options"])
    log, ts = [], time.time()
    dt = torch.bfloat16 if dev == "cuda" else torch.float32
    for step in range(a.steps):
        ix = batches[step % len(batches)]
        b = collate(tok, [encd[i] for i in ix], max_options=mo, device=dev)
        with torch.autocast(dev, dtype=dt, enabled=dev == "cuda"):
            ex = model.forward_exits(**b, aux_detach=True)
        om = b["option_mask"]
        K = om.shape[1]
        gold = torch.zeros(len(ix), K, device=dev)
        for r, i in enumerate(ix):
            c, q = rows[i]
            g = torch.tensor(c.gold[q.key], dtype=torch.float)[:K]
            gold[r, :len(g)] = g / g.sum().clamp_min(1e-9)
        loss, rec = refit_loss(ex, MAIN, gold, b["option_perm"], om, distill)
        rec = {"step": step, **rec}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(heads, float(spec.get("grad_clip", 1.0) or 1.0))
        opt.step()
        sched.step()
        log.append(rec)
        if step % 50 == 0 or step == a.steps - 1:
            el = time.time() - ts
            print(f"step {step + 1}/{a.steps} {rec} s/step {el / (step + 1):.2f} eta_h "
                  f"{el / (step + 1) * (a.steps - step - 1) / 3600:.2f}", flush=True)

    from safetensors.torch import save_file
    sd = model.aux_scorers.state_dict()
    moved = sum(int(not torch.equal(sd[k], a0[k])) for k in sd)
    save_file({k: v.detach().float().contiguous().cpu() for k, v in sd.items()}, str(out / "aux_scorers.safetensors"))
    sha1 = {f: fsha(ck / f) for f in ("tower.safetensors", "scorer.safetensors")}
    if sha1 != sha0:
        raise SystemExit(f"FROZEN FILES CHANGED {sha0} -> {sha1}")
    args = {k: v for k, v in vars(a).items() if k not in ("ckpt", "out", "corpus", "exclude", "link_into")}
    json.dump({"args": args, "ckpt": ck.name, "frozen_sha": sha0, "n_train_q": len(rows),
               "n_batches": len(batches), "aux_tensors_moved": moved,
               "train_seconds": round(time.time() - t0), "log": log}, open(out / "train_log.json", "w"))
    print(f"frozen sha after {sha1} (unchanged); aux tensors moved {moved}/{len(sd)}; saved {out.name} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if a.link_into:
        link_checkpoint(ck, Path(a.link_into), out / "aux_scorers.safetensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
