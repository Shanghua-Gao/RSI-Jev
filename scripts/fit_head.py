"""Fine-tune only the decision head (option scorer) of a release checkpoint, tower frozen.

v5.0-VL's third stage: the head of the image stage, retrained for the own-token readout
(`RSIJEV_OPTION_POOL_OWN_TOKENS=1`, or `spec.option_pool_own_tokens` in the parent's meta).
Rows come from the parent's own training corpus, minus every case the calibration DEV split
holds out (sha256(case_id) % 10 == 0) and every state of those cases in the stage-1 corpus, so
the calibrator is never fitted on rows the head trained on. Writes a loadable checkpoint
directory: the new scorer.safetensors, the parent's tower linked, and meta.json with a
`head_stage` record.

  RSIJEV_OPTION_POOL_OWN_TOKENS=1 python scripts/fit_head.py --ckpt D/vis-v4k/s17 \\
      --corpus D/vis-v4k --dev-corpus D/rt4-tap16-ret-b --out D/headft-B/s17 --seed 1

A multi-exit checkpoint (spec.arch_extra.aux_exits, v6.0-VL's stage 4) can retrain every
decision head at once with --heads all (or a list such as 32,20): one tower pass per step
(forward_exits), loss = the sum over the tuned heads of each head's soft CE to gold. The
heads are independent parameters and the tower is frozen, so this equals separate runs on
the same data order. --sdpa-math runs every head's attention on PyTorch's MATH SDPA
backend: in trained option_xattn heads the attention saturates and the efficient backend's
backward returned wrong, huge gradients. --max-length sets the training encoder's cap
(4096 for v6.0-VL; the default keeps the checkpoint's). --skip-prefix names the image
sources to leave out ("vis_" by default, as for v5.0-VL; v6.0-VL used "vis").
"""
import argparse
import dataclasses
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from rsijev.contract import load_cases  # noqa: E402
from rsijev.encode import collate, encode_question, unpermute_logits  # noqa: E402
from serve.release import load_release  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="parent checkpoint (the image stage)")
ap.add_argument("--corpus", required=True, help="the parent stage's corpus directory")
ap.add_argument("--dev-corpus", required=True, help="the corpus the calibration DEV pool is drawn from")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--heads", default="main",
                help='"main" (the decision head only), "all" (every exit\'s head) or exits, e.g. 32,20')
ap.add_argument("--max-length", type=int, default=0, help="training encoder cap; 0 = the checkpoint's")
ap.add_argument("--skip-prefix", default="vis_", help="sources with this prefix (image rows) are left out")
ap.add_argument("--sdpa-math", action="store_true", help="run every head's attention on the MATH SDPA backend")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def held(case_id: str) -> bool:
    return int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 10 == 0


meta = json.loads((Path(a.ckpt) / "meta.json").read_text())
spec = meta["spec"]
sources = [s for s in spec["sources"].split(",") if s and not s.startswith(a.skip_prefix)]
dev_states = set()
for f in glob.glob(str(Path(a.dev_corpus) / "*.jsonl")):
    for line in open(f):
        if line.strip():
            j = json.loads(line)
            if held(j["case_id"]):
                dev_states.add(j["state"])
max_options = spec["max_options"]
cases = []
for s in sources:
    f = Path(a.corpus) / f"{s}.jsonl"
    if f.exists():
        cases += [c for c in load_cases(str(f)) if not held(c.case_id) and c.state not in dev_states
                  and all(len(q.options) <= max_options for q in c.questions)]
random.Random(a.seed).shuffle(cases)
rows = [(c, q) for c in cases for q in c.questions]
need = a.steps * a.bs
if len(rows) < 2 * need:
    rows = rows * (2 * need // max(1, len(rows)) + 1)

torch.manual_seed(a.seed)
model, tok, enc, _ = load_release(a.ckpt, dev, infer_dtype=torch.float32)
print(f"readout {'own tokens' if enc.option_pool_own_tokens else 'whole block'}; {len(cases)} cases; "
      f"{len(dev_states)} DEV states excluded; {len(rows)} rows", flush=True)
exits = model.exit_indices() if getattr(model, "aux_scorers", None) is not None else [None]
main_L = exits[-1]
if a.heads == "main":
    heads = [main_L]
elif a.heads == "all":
    heads = list(exits)
else:
    heads = [int(x) for x in a.heads.split(",")]
    if not set(heads) <= set(exits):
        raise SystemExit(f"--heads {heads}: the checkpoint's exits are {exits}")
multi = len(exits) > 1
for p in model.parameters():
    p.requires_grad_(False)
params = []
for L in heads:
    for p in (model.exit_scorer(L) if L is not None else model.scorer).parameters():
        p.requires_grad_(True)
        params.append(p)
if a.sdpa_math:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _math(mod):
        f = mod.forward

        def w(*x, **k):
            with sdpa_kernel([SDPBackend.MATH]):
                return f(*x, **k)
        mod.forward = w
    for L in exits:
        _math(model.exit_scorer(L) if L is not None else model.scorer)
print(f"exits {exits}; tuning heads {heads}: {sum(p.numel() for p in params):,} params; lr {a.lr}; "
      f"sdpa_math {a.sdpa_math}", flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
enc_c = dataclasses.replace(enc, option_order="canonical",
                            **({"max_length": a.max_length} if a.max_length else {}))
model.eval()
model.cal_mode = "none"
t0, run, ptr, skipped = time.time(), {L: 0.0 for L in heads}, 0, 0
for i in range(a.steps):
    chunk, exs = [], []
    while len(chunk) < a.bs:
        c, q = rows[ptr]
        ptr += 1
        try:
            exs.append(encode_question(tok, c.state, q, enc_c))
            chunk.append((c, q))
        except ValueError:
            skipped += 1                       # options alone exceed the cap: skipped, as when scoring
    batch = collate(tok, exs, max_options=max_options, device=dev)
    out = model.forward_exits(**batch) if multi else {main_L: model(**batch)}
    g, loss = None, 0.0
    for L in heads:
        z = unpermute_logits(out[L].float(), batch["option_perm"], batch["option_mask"])
        fin = torch.isfinite(z)
        logp = torch.log_softmax(z.masked_fill(~fin, -1e9), -1)
        if g is None:
            g = torch.zeros_like(z)
            for r, (c, q) in enumerate(chunk):
                gg = torch.tensor(c.gold[q.key], dtype=torch.float32)
                g[r, :len(gg)] = gg / gg.sum()
        lL = -(g * logp.masked_fill(~fin, 0)).sum(-1).mean()
        loss = loss + lL
        run[L] = lL.item() if i == 0 else 0.98 * run[L] + 0.02 * lL.item()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if i % 50 == 0 or i == a.steps - 1:
        print(f"step {i + 1}/{a.steps} " + " ".join(f"loss{'' if L is None else L} {run[L]:.4f}" for L in heads)
              + f" skipped {skipped} {time.time() - t0:.0f}s", flush=True)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
save_file({k: v.detach().float().contiguous().cpu() for k, v in model.scorer.state_dict().items()},
          str(out / "scorer.safetensors"))
if multi:
    save_file({k: v.detach().float().contiguous().cpu() for k, v in model.aux_scorers.state_dict().items()},
              str(out / "aux_scorers.safetensors"))
tower = out / "tower.safetensors"
if not tower.exists():
    os.symlink(str((Path(a.ckpt) / "tower.safetensors").resolve()), str(tower))
m2 = json.loads(json.dumps(meta))
m2["spec"]["option_pool_own_tokens"] = bool(enc.option_pool_own_tokens)
m2["head_stage"] = {"parent": Path(a.ckpt).name, "readout": "own tokens" if enc.option_pool_own_tokens else "whole block",
                    "steps": a.steps, "batch": a.bs, "lr": a.lr, "seed": a.seed, "tower_frozen": True,
                    "rows_used": ptr - skipped, "rows_skipped": skipped, "train_seconds": round(time.time() - t0),
                    **({"heads": heads, "max_length": enc_c.max_length, "sdpa_math": a.sdpa_math} if multi else {})}
(out / "meta.json").write_text(json.dumps(m2, indent=2) + "\n")
print("saved", out, flush=True)
