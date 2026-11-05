"""Training resilience: atomic step checkpoints, auto-resume, exact data position, signal save, DONE marker (A64).

Canonical copy: tools/resilience/ckpt_resume.py. Repos vendor it byte-identical as rsijev/ckpt_resume.py
(tests/test_resilience.py checks the sha). Dependencies: python stdlib + torch (numpy optional). Patterns taken from
torchtitan (step-N dirs, commit marker, keep_latest_k), Megatron (tracker file, signal -> save at step end),
NeMo (finished run refuses to retrain), Lightning/Slurm (SIGUSR1 -> save -> requeue).

Layout of a checkpoint root:
    root/step_0000400/state.pt      full state (torch.save)
    root/step_0000400/meta.json     written LAST inside the tmp dir: {"step", "fingerprint", "parts", "time"}
    root/LATEST                      name of the newest committed step dir (pointer; a scan is the fallback)
    root/DONE                        {"step", "fingerprint", ...} written after the FINAL save; a re-launch is a no-op

Use from a training loop:
    R = Resilience(root, fingerprint=make_fingerprint(spec, data_sha, code_sha), every=200, keep=2)
    if R.done(): exit(0)                                   # finished arm: nothing to do (fingerprint-checked)
    state = R.load()                                       # None, or the dict passed to save(); refuses a mismatch
    if state: model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); ...; restore_rng(state["rng"])
    R.install_signals()                                    # SIGUSR1 / SIGTERM -> flag; saved at the next step end
    for step in range(start, steps):
        ...train one step...
        if R.should_save(step + 1, steps):  R.save(step + 1, {..., "rng": rng_state(), "sampler": sampler.state_dict()})
        R.exit_if_signalled()                              # after a signal save: exit 99 (USR1, requeue) / 143 (TERM)
    R.mark_done(steps)
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch

REQUEUE_EXIT = 99          # "saved on SIGUSR1, requeue me" (tools/resilience/run_resilient.sh requeues on it)
TERM_EXIT = 143            # 128 + SIGTERM


# ---------------------------------------------------------------- fingerprints
def sha_strings(items: Iterable[str]) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(str(s).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def files_sha(paths: Iterable[str | Path], content: bool = True) -> str:
    """sha of (name, size[, content]) of files, sorted by path. content=False = name + size + mtime (fast, large data)."""
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        st = p.stat()
        h.update(f"{p.name}|{st.st_size}".encode())
        if content:
            with open(p, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
        else:
            h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:16]


def code_sha(root: str | Path, patterns: Iterable[str] = ("*.py",)) -> str:
    """sha of the source files a trainer runs (sorted; names + bytes). Pass the package dir(s)."""
    root = Path(root)
    files = sorted({f for pat in patterns for f in root.glob(pat) if f.is_file()})
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def make_fingerprint(spec: Any, data: str = "", code: str = "") -> dict:
    """Parts are kept separately so a refusal can say WHICH part moved (spec, data or code)."""
    s = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]
    parts = {"spec": s, "data": data, "code": code}
    parts["all"] = sha_strings([parts["spec"], parts["data"], parts["code"]])
    return parts


class FingerprintMismatch(ValueError):
    pass


def _compare(saved: dict, now: dict, where: str) -> None:
    keys = ["spec", "data", "code"]
    if os.environ.get("RSIJEV_RESUME_IGNORE_CODE") == "1":     # a bugfix-only code change: owner-approved override
        keys.remove("code")
    bad = [k for k in keys if saved.get(k) != now.get(k)]
    if bad:
        raise FingerprintMismatch(
            f"{where}: fingerprint differs in {bad} (saved {[saved.get(k) for k in bad]} vs now {[now.get(k) for k in bad]}); "
            "refusing to resume. Move the ckpt dir away or point the run elsewhere"
            + (" (code-only change after review: RSIJEV_RESUME_IGNORE_CODE=1)" if bad == ["code"] else ""))


# ---------------------------------------------------------------- RNG
def rng_state() -> dict:
    s = {"py": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        s["np"] = np.random.get_state()
    except Exception:  # noqa: BLE001
        pass
    return s


def restore_rng(s: dict) -> None:
    random.setstate(s["py"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    if "np" in s:
        import numpy as np
        np.random.set_state(s["np"])


# ---------------------------------------------------------------- exact data position
class EpochPermutationSampler:
    """Index stream: per-epoch permutation from (seed, epoch) + an offset within the epoch. state_dict() is two ints,
    so a resumed run yields exactly the remaining indices of the uninterrupted run, with no batch replay."""

    def __init__(self, n: int, seed: int, shuffle: bool = True):
        self.n, self.seed, self.shuffle = int(n), int(seed), shuffle
        self.epoch, self.offset = 0, 0
        self._perm, self._perm_epoch = None, -1

    def _order(self) -> list[int]:
        if self._perm_epoch != self.epoch:
            if self.shuffle:
                g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
                self._perm = torch.randperm(self.n, generator=g).tolist()
            else:
                self._perm = list(range(self.n))
            self._perm_epoch = self.epoch
        return self._perm

    def take(self, k: int) -> list[int]:
        out = []
        while len(out) < k:
            order = self._order()
            m = min(k - len(out), self.n - self.offset)
            out += order[self.offset:self.offset + m]
            self.offset += m
            if self.offset >= self.n:
                self.epoch, self.offset = self.epoch + 1, 0
        return out

    def state_dict(self) -> dict:
        return {"n": self.n, "seed": self.seed, "epoch": self.epoch, "offset": self.offset}

    def load_state_dict(self, s: dict) -> None:
        if (s["n"], s["seed"]) != (self.n, self.seed):
            raise FingerprintMismatch(f"sampler state is for n={s['n']} seed={s['seed']}, this run n={self.n} seed={self.seed}")
        self.epoch, self.offset = int(s["epoch"]), int(s["offset"])


def stateful_dataloader_cls():
    """torchdata.stateful_dataloader.StatefulDataLoader when the env has it (cluster envs/rsijev: torchdata 0.11), else
    None: use EpochPermutationSampler. Never pip-install into a shared env for this."""
    try:
        from torchdata.stateful_dataloader import StatefulDataLoader
        return StatefulDataLoader
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- atomic files
def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(str(d), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def commit_dir(tmp: Path, dst: Path) -> None:
    """Move a fully written tmp dir to dst. An existing dst is first renamed aside (os.replace cannot overwrite a
    non-empty dir), so at every instant either the old or the new complete dir is visible under some committed name."""
    _fsync_dir(tmp)
    old = None
    if dst.exists():
        old = dst.with_name(f".old_{dst.name}_{os.getpid()}")
        shutil.rmtree(old, ignore_errors=True)
        os.replace(dst, old)
    os.replace(tmp, dst)
    _fsync_dir(dst.parent)
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def release_memory() -> None:
    """After a save/load: collect cycles and hand freed heap back to the OS (glibc malloc_trim). The CPU copies of a
    multi-GB state otherwise stay as fragmented arena pages and RSS climbs save after save (A64 fix, 2026-10-04)."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def _drop_page_cache(fh) -> None:
    """The written file is fsynced (clean pages): drop it from the page cache, which counts against the job's cgroup."""
    try:
        os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


# ---------------------------------------------------------------- checkpoint root
def _is_committed(d: Path) -> bool:
    return d.is_dir() and (d / "meta.json").exists() and (d / "state.pt").exists()


def find_latest(root: Path) -> tuple[Path | None, dict | None]:
    """LATEST pointer first; if it is missing or names an uncommitted dir, the newest committed step_* dir."""
    root = Path(root)
    p = root / "LATEST"
    if p.exists():
        d = root / p.read_text().strip()
        if _is_committed(d):
            return d, json.loads((d / "meta.json").read_text())
    for d in sorted(root.glob("step_*"), reverse=True):
        if _is_committed(d):
            return d, json.loads((d / "meta.json").read_text())
    return None, None


def save_step(root: Path, step: int, state: dict, meta: dict, keep: int, write_state=None) -> Path:
    """state.pt + meta.json into root/.tmp_step_N, fsync, commit to root/step_N, LATEST -> step_N, keep the newest
    `keep` dirs. write_state(tmp_dir) replaces torch.save (e.g. DCP) if given."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"step_{int(step):07d}"
    tmp = root / f".tmp_{name}_{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    if write_state is not None:
        write_state(tmp)
    else:
        with open(tmp / "state.pt", "wb") as fh:
            torch.save(state, fh)
            fh.flush()
            os.fsync(fh.fileno())
            _drop_page_cache(fh)
    m = {"step_done": int(step), **meta, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(tmp / "meta.json", "w") as fh:          # meta LAST: a dir without it is never resumed
        fh.write(json.dumps(m, default=str))
        fh.flush()
        os.fsync(fh.fileno())
    dst = root / name
    commit_dir(tmp, dst)
    atomic_write_text(root / "LATEST", name + "\n")
    olds = sorted(d for d in root.glob("step_*") if d.is_dir())
    for d in olds[:-max(1, int(keep))]:
        shutil.rmtree(d, ignore_errors=True)
    for d in root.glob(".tmp_step_*"):                # debris of a run killed mid-save (not ours: other pid)
        if not d.name.endswith(f"_{os.getpid()}"):
            shutil.rmtree(d, ignore_errors=True)
    return dst


# ---------------------------------------------------------------- signals
class SignalFlag:
    """SIGUSR1 / SIGTERM set a flag (never save inside the handler); the loop saves at the next step end."""

    def __init__(self):
        self.sig: int | None = None
        self._old: dict = {}

    def install(self, sigs=(signal.SIGUSR1, signal.SIGTERM)) -> bool:
        import threading
        if threading.current_thread() is not threading.main_thread():
            return False
        for s in sigs:
            try:
                self._old[s] = signal.signal(s, self._handler)
            except (ValueError, OSError):
                pass
        return bool(self._old)

    def _handler(self, signum, frame):  # noqa: ARG002
        if self.sig is None or signum == signal.SIGTERM:
            self.sig = signum
        print(f"[resilience] got signal {signal.Signals(signum).name}: saving at the end of this step", flush=True)

    def restore(self) -> None:
        for s, h in self._old.items():
            try:
                signal.signal(s, h)
            except (ValueError, OSError, TypeError):
                pass
        self._old = {}



def _sync_max(vals: list[int]) -> list[int]:
    """Element-wise MAX over torch.distributed ranks (identity when not distributed), so every rank takes the same
    save / exit decision (a time- or signal-triggered save on one rank only would deadlock the barrier)."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            t = torch.tensor(vals, dtype=torch.int64, device="cuda" if dist.get_backend() == "nccl" else "cpu")
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return [int(x) for x in t.tolist()]
    except Exception:  # noqa: BLE001
        pass
    return vals


def exit_code_for(sig: int | None) -> int:
    return TERM_EXIT if sig == signal.SIGTERM else REQUEUE_EXIT


# ---------------------------------------------------------------- the manager
class Resilience:
    def __init__(self, root: str | Path, fingerprint: dict, every: int = 0, minutes: float = 0.0, keep: int = 2,
                 rank: int = 0, signal_continue: bool | None = None, clock=time.monotonic):
        self.root = Path(root)
        self.fp = fingerprint
        self.every, self.minutes, self.keep, self.rank = int(every), float(minutes), int(keep), int(rank)
        self.flag = SignalFlag()
        self._clock = clock
        self._last_save = clock()
        self._signalled: int | None = None
        self.signal_continue = (os.environ.get("RSIJEV_SIGNAL_CONTINUE") == "1") if signal_continue is None \
            else signal_continue

    # finished arm
    def done(self) -> dict | None:
        p = self.root / "DONE"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        _compare(d.get("parts", {}), self.fp, f"DONE marker {p}")
        return d

    def mark_done(self, step: int, **info) -> None:
        if self.rank == 0:
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_write_text(self.root / "DONE", json.dumps(
                {"step": int(step), "fingerprint": self.fp["all"], "parts": self.fp,
                 "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **info}, default=str) + "\n")

    # resume
    def load(self, map_location="cpu") -> dict | None:
        d, meta = find_latest(self.root)
        if d is None:
            return None
        _compare(meta.get("parts", {}), self.fp, f"step ckpt {d}")
        st = torch.load(d / "state.pt", map_location=map_location, weights_only=False)
        self.mark_saved()                                   # the timer runs from the resume, not from __init__
        st["_step"] = int(meta["step_done"])
        st["_dir"] = str(d)
        return st

    # save
    def install_signals(self) -> bool:
        return self.flag.install()

    def should_save(self, step_done: int, total: int) -> bool:
        """Call on EVERY rank after every step (the decision is synced across ranks). A True answer restarts the
        ckpt_minutes timer, so callers that write the state with their own function (rsijev fit.py _ckpt_save) still
        save once per `minutes`, not on every step after the first timed save (the A64 bug: one run saved
        steps 588..599 back to back and was OOM-killed on host RAM)."""
        want = (step_done >= total or (self.every > 0 and step_done % self.every == 0)
                or (self.minutes > 0 and (self._clock() - self._last_save) >= 60 * self.minutes))
        sig, want = _sync_max([int(self.flag.sig or 0), int(want)])
        self._signalled = sig or None
        if want or sig:
            self.mark_saved()
        return bool(want or sig)

    def mark_saved(self) -> None:
        """Restart the ckpt_minutes timer (should_save and save call it; call it after any save of your own)."""
        self._last_save = self._clock()

    def save(self, step_done: int, state: dict, write_state=None) -> Path | None:
        dst = None
        if self.rank == 0:
            dst = save_step(self.root, step_done, state, {"fingerprint": self.fp["all"], "parts": self.fp},
                            self.keep, write_state=write_state)
            print(f"[resilience] saved {dst} (keep {self.keep})", flush=True)
        self.mark_saved()
        release_memory()
        return dst

    def exit_if_signalled(self) -> None:
        """Call after the step's save. USR1: exit 99 (wrapper requeues); TERM: exit 143. RSIJEV_SIGNAL_CONTINUE=1:
        keep training after the save (the worker path, where an exit would mark the arm failed)."""
        if not self._signalled:
            return
        sig, self._signalled, self.flag.sig = self._signalled, None, None
        if self.signal_continue and sig != signal.SIGTERM:
            print("[resilience] signal save done; continuing (RSIJEV_SIGNAL_CONTINUE=1)", flush=True)
            return
        code = exit_code_for(sig)
        print(f"[resilience] signal save done; exiting {code} for requeue", flush=True)
        sys.stdout.flush()
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)

    def close(self) -> None:
        self.flag.restore()
