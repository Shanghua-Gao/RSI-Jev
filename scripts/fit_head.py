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
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"


def held(case_id: str) -> bool:
    return int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 10 == 0


meta = json.loads((Path(a.ckpt) / "meta.json").read_text())
spec = meta["spec"]
sources = [s for s in spec["sources"].split(",") if s and not s.startswith("vis_")]
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
for p in model.tower.parameters():
    p.requires_grad_(False)
params = list(model.scorer.parameters())
for p in params:
    p.requires_grad_(True)
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
enc_c = dataclasses.replace(enc, option_order="canonical")
model.eval()
model.cal_mode = "none"
t0, run, ptr, skipped = time.time(), 0.0, 0, 0
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
    z = unpermute_logits(model(**batch).float(), batch["option_perm"], batch["option_mask"])
    fin = torch.isfinite(z)
    logp = torch.log_softmax(z.masked_fill(~fin, -1e9), -1)
    g = torch.zeros_like(z)
    for r, (c, q) in enumerate(chunk):
        gg = torch.tensor(c.gold[q.key], dtype=torch.float32)
        g[r, :len(gg)] = gg / gg.sum()
    loss = -(g * logp.masked_fill(~fin, 0)).sum(-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    run = loss.item() if i == 0 else 0.98 * run + 0.02 * loss.item()
    if i % 50 == 0 or i == a.steps - 1:
        print(f"step {i + 1}/{a.steps} loss {run:.4f} skipped {skipped} {time.time() - t0:.0f}s", flush=True)

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
save_file({k: v.detach().float().contiguous().cpu() for k, v in model.scorer.state_dict().items()},
          str(out / "scorer.safetensors"))
tower = out / "tower.safetensors"
if not tower.exists():
    os.symlink(str((Path(a.ckpt) / "tower.safetensors").resolve()), str(tower))
m2 = json.loads(json.dumps(meta))
m2["spec"]["option_pool_own_tokens"] = bool(enc.option_pool_own_tokens)
m2["head_stage"] = {"parent": Path(a.ckpt).name, "readout": "own tokens" if enc.option_pool_own_tokens else "whole block",
                    "steps": a.steps, "batch": a.bs, "lr": a.lr, "seed": a.seed, "tower_frozen": True,
                    "rows_used": ptr - skipped, "rows_skipped": skipped, "train_seconds": round(time.time() - t0)}
(out / "meta.json").write_text(json.dumps(m2, indent=2) + "\n")
print("saved", out, flush=True)
