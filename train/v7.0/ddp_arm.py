#!/usr/bin/env python3
"""A94 (2026-10-11): run ONE queue arm spec with DDP on N GPUs, through the unchanged evaluator path.

  torchrun --standalone --nproc_per_node 4 ddp_arm.py --code-dir C --spec S --name N --out O --model M --corpus D [--runner R]

Same six arguments as worker.py passes to scripts/run_arm_isolated.py (R = <deployed tree>/scripts/run_arm_isolated.py, as the
worker uses). This file does only what run_arm_isolated cannot do itself under torchrun: pin each rank to its GPU, start the NCCL
process group, check the spec against fit.py's DDP rules, and hand run_arm_isolated a spec copy with fit_extra.ddp = true.
Everything else (integrity check, overlap checks, control eval, fit, eval, records) is run_arm_isolated / run_arm_lib, unchanged.
fit.py (ddp=True): each rank trains B/W rows of every step, global batch unchanged; rank 0 alone saves step ckpts, and ranks > 0
exit after training, so the post-train eval and the <name>.jsonl records are written once, by rank 0.
Pre-fit work (model load, control eval / control cache) runs on every rank: correct (atomic cache writes) but duplicated.
Exit 2 = refused (DDP rule broken or the code snapshot's fit.py has no DDP), like run_arm_isolated.
"""
import argparse, datetime, json, os, runpy, sys
from pathlib import Path

DEPLOYED = str(Path(__file__).resolve().parent / "scripts" / "run_arm_isolated.py")


def ddp_problems(spec, world, fit_src):
    """-> list of reasons this spec cannot run DDP on `world` ranks (empty = ok). Mirrors fit.py's own checks."""
    p = []; fe = spec.get("fit_extra") or {}
    if "ddp: bool" not in fit_src: p.append("the code snapshot's rsijev/fit.py has no DDP support (FitConfig.ddp); rebuild code_dir from a tree that has it")
    b = int(spec.get("batch_size", 16)); acc = max(1, int(fe.get("grad_accum", 1) or 1))
    if b % world: p.append(f"batch_size {b} % world {world} != 0")
    if acc > 1 and acc % world: p.append(f"grad_accum {acc} % world {world} != 0 (fit: a split step needs n_acc % world == 0)")
    if fe.get("retention_kl") and int(fe.get("retention_rows", 0) or 0) % world:
        p.append(f"retention_rows {fe.get('retention_rows')} % world {world} != 0")
    if "listwise" in str(spec.get("objective", "")) or fe.get("listwise"): p.append("listwise objective is not supported under DDP")
    return p


def main():
    ap = argparse.ArgumentParser()
    for a in ("--code-dir", "--spec", "--name", "--out", "--model", "--corpus"): ap.add_argument(a, required=True)
    ap.add_argument("--runner", default=os.environ.get("DDP_RUNNER", DEPLOYED))
    ap.add_argument("--check", action="store_true", help="only check the spec (no torch); world from --world")
    ap.add_argument("--world", type=int, default=0)
    a = ap.parse_args()
    world = a.world or int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    spec = json.loads(Path(a.spec).read_text())
    fit = Path(a.code_dir) / "rsijev/fit.py"
    bad = ddp_problems(spec, world, fit.read_text(errors="replace") if fit.exists() else "")
    if bad:
        if rank == 0: print("DDP_REFUSED: " + "; ".join(bad), file=sys.stderr, flush=True)
        return 2
    if a.check: print(f"ddp_arm: spec ok for world {world}"); return 0
    import torch, torch.distributed as dist
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)                    # run_arm_isolated loads the model .to("cuda") = this rank's GPU
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))   # pre-fit control eval can be long; ranks do equal work
    fe = dict(spec.get("fit_extra") or {}); fe["ddp"] = True; spec = {**spec, "fit_extra": fe}
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"ddp_arm-{os.environ.get('SLURM_JOB_ID', 'local')}-r{rank}"
    tmp.mkdir(parents=True, exist_ok=True); sp = tmp / Path(a.spec).name; sp.write_text(json.dumps(spec))
    if rank == 0: print(f"ddp_arm: world {world}, spec {a.spec} (fit_extra.ddp=true), runner {a.runner}", flush=True)
    sys.argv = [a.runner, "--code-dir", a.code_dir, "--spec", str(sp), "--name", a.name, "--out", a.out, "--model", a.model, "--corpus", a.corpus]
    try:
        runpy.run_path(a.runner, run_name="__main__")    # ends in sys.exit(main()); ranks > 0 os._exit(0) inside fit
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if dist.is_initialized(): dist.destroy_process_group()
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
