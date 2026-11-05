"""One numeric path per process for the fla kernels (SERVE-AUTO-NONDET).

fla picks its Triton configs by a timed race once per process. Block sizes (BK, BV) change how sums round, so two servers
on one GPU could answer differently depending on who won. install() drops the block-size choice from the kernels where it
reaches the answer. num_warps still races there and everywhere else; the pin gate forces those races to test that they move nothing.
The values are the winners of the most common class on RTX PRO 6000, and each is in fla 0.5.2's list on every GPU.

    RSIJEV_TUNE_PIN=0         stock autotune
    RSIJEV_TUNE_TABLE=<path>  optional: also load fixed winners (same file as fla_tune_table.py merge, recorded with the pin on)

Never fatal: a kernel that cannot be pinned keeps stock autotune, and the startup line says so.
"""
import importlib, json, os

PINS = (("fla.ops.common.chunk_o", "chunk_fwd_kernel_o", {"BK": 64, "BV": 64}),
        ("fla.ops.gated_delta_rule.chunk_fwd", "chunk_gated_delta_rule_fwd_kkt_solve_kernel", {"BK": 64}),
        ("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", {"BV": 32}))
COMPILE_ERRORS = {"OutOfResources", "CompileTimeAssertionFailure", "PTXASError"}  # what Autotuner._bench treats as "config does not fit"
STATE = {"pinned": [], "unpinned": [], "released": [], "loaded": 0, "bad": 0}


def _autotuner(kernel):
    """The Autotuner under any triton.heuristics wrappers."""
    from triton.runtime.autotuner import Autotuner
    while kernel is not None and not isinstance(kernel, Autotuner): kernel = getattr(kernel, "fn", None)
    return kernel


def _pin(at, want, name):
    keep = [c for c in at.configs if all(c.kwargs.get(k) == v for k, v in want.items())]
    if not keep: return False
    full, at.configs = at.configs, keep

    def run(*a, **k):
        try:
            return type(at).run(at, *a, **k)
        except Exception as ex:
            if type(ex).__name__ not in COMPILE_ERRORS or at.configs is full: raise
            print(f"tune pin: {name} {want} does not fit this GPU ({type(ex).__name__}); stock autotune for it, answers can differ between processes", flush=True)
            STATE["released"].append(name); at.configs = full
            return type(at).run(at, *a, **k)

    at.run = run
    return True


def _table(path):
    from triton.runtime.autotuner import Autotuner
    table, seen, run0 = json.load(open(path)), set(), Autotuner.run

    def name(at):
        fn = getattr(at, "base_fn", None) or at.fn
        return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}#{len(at.configs)}"

    def run(self, *a, **k):
        if id(self) not in seen:
            seen.add(id(self))
            for key, idx, s in table.get(name(self), []):
                if 0 <= idx < len(self.configs) and str(self.configs[idx]) == s:
                    self.cache[tuple(key)] = self.configs[idx]; STATE["loaded"] += 1
                else:
                    STATE["bad"] += 1
        return run0(self, *a, **k)

    Autotuner.run = run
    return sum(len(v) for v in table.values())


def install():
    """Call once, on a GPU device, before the first fla kernel runs. Returns a one-line status for the startup print."""
    if STATE.get("on"): return "tune pin: already on"
    try:
        out = []
        if os.environ.get("RSIJEV_TUNE_PIN", "1").strip() == "0":
            out.append("off, stock autotune, answers can differ between processes")
        else:
            for mod, fn, want in PINS:
                try:
                    at = _autotuner(getattr(importlib.import_module(mod), fn))
                    ok = at is not None and _pin(at, want, fn)
                except (ImportError, AttributeError):
                    ok = False
                STATE["pinned" if ok else "unpinned"].append(fn)
            out.append(f"{len(STATE['pinned'])} of {len(PINS)} kernels")
            if STATE["unpinned"]: out.append(f"NOT pinned {STATE['unpinned']}, answers can differ between processes")
        path = os.environ.get("RSIJEV_TUNE_TABLE", "").strip()
        if path: out.append(f"table {path} ({_table(path)} winners)")
        STATE["on"] = 1
        return "tune pin: " + "; ".join(out)
    except Exception as ex:
        return f"tune pin: failed {type(ex).__name__}: {ex}; stock autotune, answers can differ between processes"
