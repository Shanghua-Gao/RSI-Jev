"""A persistent worker: hold one GPU, load everything once, drain an arm queue.

Submitting one job per arm pays the Slurm queue twice over -- the wait, and then
a fresh model load and a fresh corpus parse for every arm. This holds the
allocation for its full two days instead and keeps the model, the tokenizer, the
corpus and both evaluation targets resident, so an arm costs only its own
compute.

Queue protocol, on shared storage so several workers cooperate without a server:

    $QUEUE/pending/<name>.json     an arm spec, written by whoever enqueues
    $QUEUE/running/<name>.json     claimed; the claim is an atomic rename
    $QUEUE/done/<name>.json        finished, with the result path
    $QUEUE/failed/<name>.json      raised; the traceback is in the record
    $QUEUE/release.request         any worker that sees this exits cleanly

Two workers never run the same arm: `os.rename` is atomic on POSIX, so exactly
one claim can succeed.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

# Triton's compile cache defaults to ~/.triton on the shared NFS home. Workers
# on different nodes writing it at once fail with "Stale file handle" (three fla
# diag arms died this way), and home is nearly full. Keep it node-local and
# per-process. Set before torch/fla import so every compile sees it.
# Unconditional: the user's shell profile exports TRITON_CACHE_DIR on shared
# netscratch, and a setdefault here silently deferred to it (a fourth arm died
# the same way after the first fix). This affects only worker processes.
os.environ["TRITON_CACHE_DIR"] = os.path.join(
    os.environ.get("TMPDIR", "/tmp"),
    f"triton-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}")

import torch


def _write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


SIDECARS = (".error.json", ".result.json", ".tmp")


def is_spec(path: Path) -> bool:
    """An arm spec is `<name>.json` and nothing else. Error and result records
    also end in .json, and a requeue glob that swept them into pending/ made
    workers run an error record as if it were an arm."""
    return path.suffix == ".json" and not any(path.name.endswith(x) for x in SIDECARS)


RESTART_EXIT = 3


def run_isolated(spec_path: Path, spec: dict, name: str, out: Path, args, root: Path) -> list:
    """Run an arm whose spec names a code snapshot, in a fresh process.

    stdout streams into this worker's log; stderr goes to a file so a refusal or a
    crash comes back as this arm's error record rather than a bare exit code.
    """
    import subprocess
    err = out / f"{name}.stderr.txt"
    with open(err, "w") as fh:
        rc = subprocess.run(
            [sys.executable, str(root / "scripts" / "run_arm_isolated.py"),
             "--code-dir", spec["code_dir"], "--spec", str(spec_path), "--name", name,
             "--out", str(out), "--model", args.model, "--corpus", args.corpus],
            stderr=fh, env=os.environ.copy()).returncode
    tail = err.read_text()[-3000:]
    if rc != 0:
        kind = "REFUSED" if rc == 2 else f"exit {rc}"
        raise RuntimeError(f"isolated arm {kind}:\n{tail}")
    return [json.loads(line) for line in open(out / f"{name}.jsonl") if line.strip()]


def code_fingerprint(root: Path) -> str:
    """Hash of every source file a worker executes.

    A running worker never re-imports anything: the corpus and targets it holds
    were built by the classes it imported at startup, and reloading modules under
    them would make class identity silently wrong. So when this fingerprint moves,
    the worker exits with RESTART_EXIT between arms and worker_loop.sh starts a
    fresh process in the same allocation. The GPU is never given back.
    """
    import hashlib
    h = hashlib.sha256()
    for f in sorted([*root.glob("rsijev/*.py"), *root.glob("scripts/*.py")]):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def claim(pending: Path, running: Path, job: str) -> Path | None:
    for spec in sorted(p for p in pending.glob("*.json") if is_spec(p)):
        target = running / spec.name
        try:
            os.rename(spec, target)      # atomic: exactly one worker wins
        except OSError:
            continue                     # another worker took it
        _write(running / f"{spec.stem}.owner", {"job_id": job, "claimed_at": time.time()})
        return target
    return None


def reclaim_stale(queue: Path, stale_after: float = 900.0) -> list[str]:
    """Return abandoned arms to pending.

    A worker killed mid-arm -- preempted on a requeue partition, or cut off at
    its two-day limit -- leaves its spec in running/ forever. An arm is stale
    when its owner's heartbeat is older than `stale_after`; it goes back to
    pending and the next free worker reruns it from scratch.
    """
    moved = []
    running, pending = queue / "running", queue / "pending"
    for owner in running.glob("*.owner"):
        spec = running / f"{owner.stem}.json"
        try:
            job = json.loads(owner.read_text())["job_id"]
            hb = json.loads((queue / f"worker_{job}.json").read_text()).get("heartbeat_at", 0)
        except Exception:
            hb = 0
        if time.time() - hb > stale_after and spec.exists():
            try:
                os.rename(spec, pending / spec.name)
                owner.unlink(missing_ok=True)
                moved.append(spec.stem)
            except OSError:
                pass
    return moved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--idle-exit", type=float, default=0.0,
                    help="seconds of empty queue before exiting; 0 means hold")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rsijev.contract import load_cases
    from rsijev.targets import load_kg_v1, load_typed_decisions
    import run_arm_lib as lib                                      # noqa: F401

    queue = Path(args.queue)
    dirs = {k: queue / k for k in ("pending", "running", "done", "failed")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    job = os.environ.get("SLURM_JOB_ID", "local")
    status = queue / f"worker_{job}.json"
    info = {"job_id": job, "node": socket.gethostname(), "started_at": time.time()}

    # The heartbeat must keep beating WHILE an arm trains: the fit loop blocks
    # for minutes, and a silent worker is indistinguishable from a dead one, so
    # without this thread another worker would reclaim a live arm and run it twice.
    live: dict = {"state": "loading"}

    def beat() -> None:
        while True:
            try:
                _write(status, {**info, **live, "heartbeat_at": time.time()})
            except Exception:
                pass
            time.sleep(60)

    threading.Thread(target=beat, daemon=True).start()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev).eval()
    targets = {"kg_v1": load_kg_v1(),
               "typed_decisions": load_typed_decisions("test")}
    corpus = {f.stem: load_cases(str(f)) for f in sorted(Path(args.corpus).glob("*.jsonl"))}
    load_s = time.perf_counter() - t0
    print(f"[worker {job}] loaded in {load_s:.0f}s on {info['node']} ({dev}); "
          f"{sum(len(v) for v in corpus.values())} training cases, "
          f"{sum(len(v) for v in targets.values())} eval cases", flush=True)

    done_count, idle_since, last_reclaim = 0, time.time(), 0.0
    root = Path(__file__).resolve().parent.parent
    booted_on = code_fingerprint(root)
    live["code_fingerprint"] = booted_on
    exit_code = 0

    def stop_requested() -> bool:
        nonlocal exit_code
        if (queue / "release.request").exists():
            return True
        if code_fingerprint(root) != booted_on or (queue / f"restart.{job}").exists():
            (queue / f"restart.{job}").unlink(missing_ok=True)
            exit_code = RESTART_EXIT
            return True
        return False

    while not stop_requested():
        if time.time() - last_reclaim > 300:
            back = reclaim_stale(queue)
            if back:
                print(f"[worker {job}] reclaimed abandoned arms: {back}", flush=True)
            last_reclaim = time.time()
        spec_path = claim(dirs["pending"], dirs["running"], job)
        if spec_path is None:
            live.update(state="idle", arm=None, completed=done_count)
            if args.idle_exit and time.time() - idle_since > args.idle_exit:
                print(f"[worker {job}] idle for {args.idle_exit}s, exiting", flush=True)
                break
            time.sleep(args.poll)
            continue

        idle_since = time.time()
        spec = json.loads(spec_path.read_text())
        name = spec_path.stem
        live.update(state="running", arm=name, completed=done_count)
        print(f"\n[worker {job}] === arm {name}: {json.dumps(spec)} ===", flush=True)
        t1 = time.perf_counter()
        try:
            path = out / f"{name}.jsonl"
            if spec.get("code_dir"):
                # An agent's own code snapshot: fresh process, integrity- and
                # overlap-checked by run_arm_isolated.py before anything loads.
                records = run_isolated(spec_path, spec, name, out, args, root)
            else:
                records = lib.run_arm(lm=lm, tok=tok, targets=targets, corpus=corpus,
                                      device=dev, spec=spec, name=name,
                                      items_path=out / f"{name}.items.jsonl")
                with open(path, "w") as fh:
                    for r in records:
                        fh.write(json.dumps(r) + "\n")
            spec_path.replace(dirs["done"] / spec_path.name)
            (dirs["running"] / f"{name}.owner").unlink(missing_ok=True)
            _write(dirs["done"] / f"{name}.result.json",
                   {"arm": name, "spec": spec, "result": str(path),
                    "seconds": round(time.perf_counter() - t1), "worker": job})
            done_count += 1
            cv = records[-1].get("code_version", "?") if records else "?"
            print(f"[worker {job}] arm {name} done in "
                  f"{time.perf_counter() - t1:.0f}s code={cv} -> {path}", flush=True)
        except Exception:
            tb = traceback.format_exc()
            try:
                spec_path.replace(dirs["failed"] / spec_path.name)
            except OSError:
                pass       # another worker already moved it; the record below still lands
            (dirs["running"] / f"{name}.owner").unlink(missing_ok=True)
            _write(dirs["failed"] / f"{name}.error.json",
                   {"arm": name, "spec": spec, "traceback": tb, "worker": job})
            print(f"[worker {job}] arm {name} FAILED\n{tb}", flush=True)
            # one bad arm must not take the allocation down with it

    live.update(state="restarting" if exit_code == RESTART_EXIT else "released",
                completed=done_count)
    _write(status, {**info, **live, "heartbeat_at": time.time()})
    print(f"[worker {job}] {'restarting on new code' if exit_code == RESTART_EXIT else 'released'} "
          f"after {done_count} arms", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
