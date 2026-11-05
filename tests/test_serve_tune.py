"""serve_tune on the real triton Autotuner with a fake kernel (no GPU: the Triton driver is stubbed, kernels never compile)."""
import json, sys, types
import pytest

triton = pytest.importorskip("triton")
from triton.runtime import autotuner as AT
from triton.runtime.errors import OutOfResources
from serve import serve_tune as T


def k(): pass
k.__module__ = "fla.x"


class Fake:
    """Stands in for a triton.jit kernel. fails = kwargs of configs that do not compile."""
    fn = staticmethod(k)
    def __init__(self, fails=None): self.calls, self.fails = [], fails or {}
    def run(self, *a, **k):
        for f, v in self.fails.items():
            if k.get(f) == v: raise OutOfResources(2, 1, "registers")
        self.calls.append(k)


CONFIGS = lambda: [triton.Config({"BK": b, "BV": b}, num_warps=w) for b, w in ((128, 8), (64, 4), (32, 2))]


def kernel(fake, bench, configs=None):
    return AT.Autotuner(fake, ["x"], configs or CONFIGS(), ["x"], None, None, do_bench=bench)


def configs_for(must, prefer):
    """The pinned config plus one that differs in warps and one in block size (or stages)."""
    want = {**must, **prefer}
    kw = {k: v for k, v in want.items() if k not in ("num_warps", "num_stages")}
    w, st = want.get("num_warps", 4), want.get("num_stages", 3)
    return [triton.Config(dict(kw), num_warps=w, num_stages=st), triton.Config(dict(kw), num_warps=1 if w > 1 else 2, num_stages=st),
            triton.Config({k: v // 2 for k, v in kw.items()}, num_warps=w if kw else 8, num_stages=st)]


O = "chunk_fwd_kernel_o"
N = len(T.PINS)


@pytest.fixture
def fla(monkeypatch):
    """A kernel inside a heuristics wrapper, as fla has it, at the module paths in PINS."""
    for k in ("pinned", "unpinned", "released", "racing"): monkeypatch.setitem(T.STATE, k, [])
    monkeypatch.delitem(T.STATE, "on", raising=False)
    monkeypatch.delenv("RSIJEV_TUNE_PIN", raising=False); monkeypatch.delenv("RSIJEV_TUNE_TABLE", raising=False)
    raced = []
    def bench(call, quantiles=None): raced.append(1); call(); return [float(len(raced))] * 3
    made = {}
    for mod, fn, must, prefer in T.PINS:
        f = Fake(); m = types.ModuleType(mod); inner = kernel(f, bench, None if fn == O else configs_for(must, prefer))
        setattr(m, fn, types.SimpleNamespace(fn=inner)); monkeypatch.setitem(sys.modules, mod, m); made[fn] = (inner, f)
    return types.SimpleNamespace(made=made, raced=raced)


def test_o_has_one_config_and_no_race(fla):
    assert f"{N} of {N}" in T.install()
    at, f = fla.made[O]
    at.run(7)
    assert not fla.raced and f.calls[-1]["BK"] == 64 and f.calls[-1]["BV"] == 64 and f.calls[-1]["num_warps"] == 4


def test_off_leaves_stock(fla, monkeypatch):
    monkeypatch.setenv("RSIJEV_TUNE_PIN", "0")
    assert "stock autotune" in T.install()
    at, _ = fla.made[O]
    at.run(7)
    assert len(fla.raced) == 3 and "run" not in vars(at)


def test_warps_still_race_on_a_block_size_pin(fla):
    at, _ = fla.made[O]
    at.configs = [triton.Config({"BK": 64, "BV": 64}, num_warps=w) for w in (1, 2)] + [triton.Config({"BK": 32, "BV": 32}, num_warps=1)]
    T.install()
    at.run(7)
    assert len(fla.raced) == 2 and all(c.kwargs["BK"] == 64 for c in at.configs)


def test_kernel_gone_or_value_gone_is_not_fatal(fla, monkeypatch):
    monkeypatch.setitem(sys.modules, T.PINS[1][0], types.ModuleType(T.PINS[1][0]))
    at, _ = fla.made[O]; at.configs = [c for c in at.configs if c.kwargs["BK"] != 64]
    s = T.install()
    assert f"{N - 2} of {N}" in s and "NOT pinned" in s and O in s and T.PINS[1][1] in s and len(at.configs) == 2


def test_import_failure_is_not_fatal(fla, monkeypatch):
    """No GPU driver: importing fla raises RuntimeError at Autotuner.__init__, not ImportError."""
    class Boom(types.ModuleType):
        def __getattr__(self, n): raise RuntimeError("0 active drivers")
    monkeypatch.setitem(sys.modules, T.PINS[0][0], Boom(T.PINS[0][0]))
    s = T.install()
    assert "failed RuntimeError" in s and "stock autotune" in s


def test_does_not_fit_releases_to_stock_once(fla):
    at, f = fla.made[O]
    f.fails = {"BK": 64}
    T.install()
    at.run(7)
    assert f.calls[-1]["BK"] != 64 and T.STATE["released"] == [O] and len(at.configs) == 3


def test_other_errors_propagate(fla):
    at, f = fla.made[O]
    T.install()
    f.run = lambda *a, **k: (_ for _ in ()).throw(ValueError("real bug"))
    with pytest.raises(ValueError): at.run(7)


def test_table_seeds_and_refuses(fla, tmp_path, monkeypatch):
    """Winner table on top of the pin, on a kernel the pin does not touch. Never written back."""
    def bench(call, quantiles=None): fla.raced.append(1); call(); return [1.0] * 3
    other = kernel(Fake(), bench)
    cfg = other.configs[2]
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"fla.x.k#3": [[[7], 2, str(cfg)], [[8], 2, "WRONG"]]}))
    before = p.read_text()
    monkeypatch.setenv("RSIJEV_TUNE_TABLE", str(p))
    monkeypatch.setitem(T.STATE, "loaded", 0); monkeypatch.setitem(T.STATE, "bad", 0)
    monkeypatch.setattr(AT.Autotuner, "run", AT.Autotuner.run)  # undoes the class patch after the test
    assert "2 winners" in T.install()
    other.run(7)
    assert not fla.raced and other.best_config is cfg and T.STATE["loaded"] == 1 and T.STATE["bad"] == 1
    other.run(8)
    assert len(fla.raced) == 3, "a refused entry runs stock"
    assert p.read_text() == before


def test_every_kernel_left_one_config_nothing_races(fla):
    s = T.install()
    assert f"{N} of {N}" in s and "race" not in s
    for mod, fn, must, prefer in T.PINS:
        at, f = fla.made[fn]
        at.run(7)
        assert len(at.configs) == 1 and T._has(at.configs[0], {**must, **prefer}), fn
    assert not fla.raced


def test_l2norm_pins_bt_and_warps_together(fla):
    """Pin gate 11-05: l2norm BT16 w16 (RTX) and BT8 w8 (H200) moved bits; the warps are part of MUST."""
    must = next(m for _, fn, m, _ in T.PINS if fn == "l2norm_fwd_kernel")
    assert set(must) == {"BT", "num_warps"} and 2 * must["num_warps"] <= must["BT"]
    at, _ = fla.made["l2norm_fwd_kernel"]
    at.configs = [c for c in at.configs if c.num_warps != must["num_warps"]]
    s = T.install()
    assert "NOT pinned" in s and "l2norm_fwd_kernel" in s


def test_prefer_missing_keeps_must_and_says_warps_race(fla):
    kkt = "chunk_gated_delta_rule_fwd_kkt_solve_kernel"
    at, _ = fla.made[kkt]
    at.configs = [triton.Config({"BK": 64}, num_warps=w) for w in (1, 4)] + [triton.Config({"BK": 32}, num_warps=2)]
    s = T.install()
    at.run(7)
    assert f"{N} of {N}" in s and kkt in s and "warps still race" in s
    assert len(fla.raced) == 2 and all(c.kwargs["BK"] == 64 for c in at.configs)
