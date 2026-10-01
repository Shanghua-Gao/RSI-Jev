"""The ways in: HF ids, the Python API, profiles, and an honest startup log.

Everything except the last test runs with no weights, no GPU and no network.

    python -m pytest tests/test_easy_infer.py -q
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient                   # noqa: E402

from serve import release, runtime                          # noqa: E402
from serve.app import create_app                            # noqa: E402
from serve.decider import Decider                           # noqa: E402

REQUEST = {
    "model": "rsi-jev-test",
    "state": [{"role": "user", "content": "I was charged twice. Please refund."}],
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the user request a refund?"},
        "department": {"type": "choice", "instructions": "Which department should handle this?",
                       "criteria": {"billing": "Payments and refunds",
                                    "technical": "Software bugs"}},
        "urgency": {"type": "score", "instructions": "How urgent is the request?",
                    "criteria": ["Routine", "Urgent", "Emergency"]},
    },
}


# ---------------------------------------------------------------------------
# A checkpoint by directory, HF repo id or alias
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    import huggingface_hub
    calls = []

    def snapshot_download(repo_id, revision=None, **kw):
        calls.append((repo_id, revision))
        d = tmp_path / "hub" / repo_id.replace("/", "--") / "snapshots" / "0123abcd"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return calls


def test_a_repo_id_is_downloaded_to_the_hf_cache(fake_hub):
    path = release.resolve_ckpt("shgao/rsi-jev-v3.0-qwen3.5-2b")
    assert fake_hub == [("shgao/rsi-jev-v3.0-qwen3.5-2b", None)]
    assert path.name == "0123abcd"


def test_an_alias_is_the_repo_bench_uses(fake_hub):
    release.resolve_ckpt("v3.0-2b", revision="main")
    assert fake_hub == [("shgao/rsi-jev-v3.0-qwen3.5-2b", "main")]


def test_a_directory_is_used_as_it_is(fake_hub, tmp_path):
    d = tmp_path / "shgao" / "rsi-jev-v3.0-qwen3.5-2b"
    d.mkdir(parents=True)
    assert release.resolve_ckpt(d) == d
    assert release.resolve_ckpt(str(d)) == d
    assert fake_hub == [], "an existing directory must never trigger a download"


def test_a_repo_id_wins_over_a_folder_named_like_the_org(fake_hub, tmp_path, monkeypatch):
    (tmp_path / "shgao").mkdir()
    monkeypatch.chdir(tmp_path)
    release.resolve_ckpt("shgao/rsi-jev-v3.0-qwen3.5-2b")
    assert fake_hub == [("shgao/rsi-jev-v3.0-qwen3.5-2b", None)]


def test_a_missing_local_path_is_not_sent_to_the_hub(fake_hub, tmp_path):
    with pytest.raises(FileNotFoundError):
        release.resolve_ckpt(str(tmp_path / "no-such-ckpt"))
    with pytest.raises(FileNotFoundError):
        release.resolve_ckpt("not a repo")
    assert fake_hub == []


def test_the_served_name_is_the_repo_name_not_a_snapshot_hash(tmp_path):
    assert release.checkpoint_name("shgao/rsi-jev-v3.0-qwen3.5-2b",
                                   tmp_path / "snapshots" / "0123abcd") == "rsi-jev-v3.0-qwen3.5-2b"
    assert release.checkpoint_name("v1.0-0.8b", tmp_path) == "rsi-jev-v1.0-qwen3.5-0.8b"
    d = tmp_path / "rsi-jev-v2.1-qwen3.5-2b"
    d.mkdir()
    assert release.checkpoint_name(str(d)) == "rsi-jev-v2.1-qwen3.5-2b"     # as before
    assert release.release_version("rsi-jev-v3.0-qwen3.5-2b") == "v3.0"
    assert release.release_version("rsi-jev-v4.0-vl-qwen3.5-2b") == "v4.0-VL"


def test_the_script_loader_is_the_package_loader():
    sys.path.insert(0, str(ROOT / "scripts"))
    import load_release
    assert load_release.load_release is release.load_release
    assert load_release.artifact_name is release.artifact_name


# ---------------------------------------------------------------------------
# Decider is the HTTP API in-process
# ---------------------------------------------------------------------------

def _stub_scorer(state, questions):
    """Deterministic, state-dependent, non-uniform: a wrong wiring shows up."""
    out = []
    for i, q in enumerate(questions):
        k = len(q.options)
        w = [(len(state) + 7 * i + 3 * j) % 11 + 1 for j in range(k)]
        out.append([x / sum(w) for x in w])
    return out, 42


def test_decide_equals_the_http_answers():
    d = Decider.from_scorer(_stub_scorer, name="rsi-jev-test")
    client = TestClient(create_app(_stub_scorer, served_model_name="rsi-jev-test"))
    http = client.post("/v1/systemone", json=REQUEST)
    assert http.status_code == 200
    assert d.decide(REQUEST["state"], REQUEST["questions"]) == http.json()["answers"]
    assert d.request(REQUEST["state"], REQUEST["questions"]) == http.json()


def test_decide_rejects_what_the_server_rejects():
    from pydantic import ValidationError
    d = Decider.from_scorer(_stub_scorer)
    bad = {"q": {"type": "choice", "instructions": "?", "criteria": {"only": "one"}}}
    with pytest.raises(ValidationError):
        d.decide("state", bad)


def test_decider_is_importable_from_rsijev():
    import rsijev
    assert rsijev.Decider is Decider


# ---------------------------------------------------------------------------
# Profiles only set defaults
# ---------------------------------------------------------------------------

def test_profiles_map_to_the_env_flags():
    assert runtime.PROFILES == {"agent": {"RSIJEV_DOC_CACHE": "1"},
                                "server": {"RSIJEV_COMPILE": "1"}}
    env = {}
    assert runtime.apply_profile("agent", env) == {"RSIJEV_DOC_CACHE": "1"} and env == {
        "RSIJEV_DOC_CACHE": "1"}
    env = {}
    runtime.apply_profile("server", env)
    assert env == {"RSIJEV_COMPILE": "1"}


def test_no_profile_changes_nothing():
    env = {}
    assert runtime.apply_profile(None, env) == {} and env == {}


def test_an_env_var_already_set_wins_over_the_profile():
    env = {"RSIJEV_DOC_CACHE": "0"}
    assert runtime.apply_profile("agent", env) == {}
    assert env == {"RSIJEV_DOC_CACHE": "0"}


def test_an_unknown_profile_is_refused():
    with pytest.raises(ValueError):
        runtime.apply_profile("turbo", {})


def test_the_profiles_reach_the_switches(monkeypatch):
    """agent turns on the document cache serve.infer reads; server turns on the
    compile serve.accel applies."""
    from serve import accel, infer
    for k in ("RSIJEV_DOC_CACHE", "RSIJEV_COMPILE"):
        monkeypatch.setenv(k, "0")          # so teardown removes what the profile sets
        monkeypatch.delenv(k)
    monkeypatch.setattr(infer, "_DOC_CACHE", None)
    assert infer.default_doc_cache() is None
    runtime.apply_profile("agent")
    assert infer.default_doc_cache() is not None
    calls = []
    monkeypatch.setattr(accel, "enable_compile", lambda m, **k: calls.append(k) or 1)
    assert accel.apply_env(object()) == []
    runtime.apply_profile("server")
    assert len(accel.apply_env(object())) == 1 and calls


def test_the_cli_accepts_a_profile_and_a_positional_model():
    from serve.cli import build_parser
    a = build_parser().parse_args(["serve", "shgao/rsi-jev-v3.0-qwen3.5-2b",
                                   "--profile", "agent", "--port", "8123"])
    assert (a.model, a.profile, a.port, a.ckpt) == ("shgao/rsi-jev-v3.0-qwen3.5-2b",
                                                    "agent", 8123, None)
    a = build_parser().parse_args(["bench"])
    assert a.model is None and a.repeat == 10


# ---------------------------------------------------------------------------
# The startup log reports the running kernels, not the training ones
# ---------------------------------------------------------------------------

def _fake_install(monkeypatch, *, fla: str | None, conv: str | None):
    versions = {"flash-linear-attention": fla, "causal-conv1d": conv}
    monkeypatch.setattr(runtime, "_dist_version", lambda *names: versions.get(names[0]))
    monkeypatch.setattr(runtime, "_importable",
                        lambda m: (fla is not None) if m.startswith("fla") else conv is not None)


def test_fla_is_reported_active_only_when_it_is_used_on_cuda(monkeypatch):
    _fake_install(monkeypatch, fla="0.5.2", conv=None)
    r = runtime.kernel_report("cuda")
    assert r["fla"] == {"installed": True, "version": "0.5.2", "importable": True, "used": True}
    assert r["causal_conv1d"]["used"] is False and r["causal_conv1d"]["installed"] is False
    lines = runtime.describe_kernels(r, "cuda")
    assert lines == ["kernels: fla 0.5.2 active; causal_conv1d not installed (optional)"]


def test_missing_fla_on_cuda_prints_the_install_command(monkeypatch):
    _fake_install(monkeypatch, fla=None, conv=None)
    lines = runtime.describe_kernels(runtime.kernel_report("cuda"), "cuda")
    assert "fla not installed" in lines[0]
    assert len(lines) == 2 and 'pip install "rsi-jev[fast] @ git+' in lines[1]


def test_fla_off_cuda_is_installed_but_not_used(monkeypatch):
    _fake_install(monkeypatch, fla="0.5.2", conv=None)
    r = runtime.kernel_report("cpu")
    assert r["fla"]["installed"] and not r["fla"]["used"]
    lines = runtime.describe_kernels(r, "cpu")
    assert lines == ["kernels: fla 0.5.2 installed, not used on cpu; "
                     "causal_conv1d not installed (optional)"], "no install hint off CUDA"


def test_the_loaded_model_overrides_importability(monkeypatch):
    """If the layer says it calls the torch reference, fla is not active, whatever
    is installed (transformers <= 5.16 keeps the chosen function on the layer)."""
    import torch.nn as nn
    _fake_install(monkeypatch, fla="0.5.2", conv=None)

    def torch_chunk_gated_delta_rule(*a, **k):
        pass

    class GatedDeltaNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkv = nn.Linear(2, 2)
            self.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            self.causal_conv1d_fn = None

    model = nn.Sequential(GatedDeltaNet())
    r = runtime.kernel_report("cuda", model)
    assert r["fla"]["used"] is False and r["causal_conv1d"]["used"] is False

    def fla_fn(*a, **k):
        pass
    fla_fn.__module__ = "fla.ops.gated_delta_rule.chunk"
    model[0].chunk_gated_delta_rule = fla_fn
    assert runtime.kernel_report("cuda", model)["fla"]["used"] is True


def test_the_wrapped_hook_of_newer_transformers_is_followed():
    """transformers 5.17 wraps the torch reference and keeps the chosen kernel in
    the wrapper's closure as `implementation`."""
    def chosen(*a, **k):
        pass
    chosen.__module__ = "fla.ops.gated_delta_rule"

    def make(implementation, torch_function):
        def wrapped(*a, **k):
            return implementation(*a, **k) if implementation else torch_function(*a, **k)
        return wrapped

    assert runtime._impl_module(make(chosen, print)) == "fla.ops.gated_delta_rule"


def test_the_startup_log_does_not_quote_the_training_stack(monkeypatch):
    import torch
    _fake_install(monkeypatch, fla=None, conv=None)
    monkeypatch.delenv("RSIJEV_MIN_SAVED_TOKENS", raising=False)
    lines = runtime.startup_lines(torch=torch, device="cpu", dtype_name="fp32")
    text = "\n".join(lines)
    assert f"torch {torch.__version__}" in text and "tower fp32" in text
    assert "torch-2.7.1" not in text
    assert ">= 480" in text, "the threshold in effect is printed"


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

def test_the_package_ships_everything_the_cli_imports():
    """`pip install rsi-jev` must carry every module `rsi-jev` reaches."""
    import ast
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packages = set(cfg["tool"]["setuptools"]["packages"])
    assert cfg["project"]["scripts"]["rsi-jev"] == "serve.cli:main"
    for py in (ROOT / "serve").glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module
                     and node.level == 0 else [])
            for n in names:
                top = n.split(".")[0]
                assert top not in {"load_release", "scripts", "bench"}, \
                    f"serve/{py.name} imports {n}, which lives in scripts/ and is not installed"
                if top in {"rsijev", "serve"}:
                    assert top in packages


def test_requirements_and_package_agree_on_the_runtime():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    reqs = {line.split(";")[0].strip() for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")}
    assert set(cfg["project"]["dependencies"]) <= reqs
    fast = cfg["project"]["optional-dependencies"]["fast"]
    assert any(d.startswith("flash-linear-attention") for d in fast)
    assert any(d.startswith("fla-core") for d in fast)
    assert not any("causal" in d for d in fast), "causal-conv1d needs nvcc; it stays optional"


# ---------------------------------------------------------------------------
# On real weights: the Python API and the HTTP API return the same answers
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_decider_equals_http_on_real_weights():
    from huggingface_hub import snapshot_download
    from serve.server import make_scorer
    path = snapshot_download("shgao/rsi-jev-v1.0-qwen3.5-2b")
    d = Decider(path)
    app = create_app(make_scorer(d.served), served_model_name=d.name)
    http = TestClient(app).post("/v1/systemone", json={**REQUEST, "model": d.name})
    assert http.status_code == 200
    assert d.decide(REQUEST["state"], REQUEST["questions"]) == http.json()["answers"]


def test_a_cpu_run_hides_the_cuda_only_kernels(monkeypatch):
    """transformers 5.17 binds fla at import whatever the device; on CPU that
    crashes in Triton. load_for_serving hides fla before the model is built."""
    import importlib.metadata
    import importlib.util
    import types
    monkeypatch.delitem(sys.modules, "transformers.models.qwen3_5.modeling_qwen3_5",
                        raising=False)
    for n in ("fla", "causal_conv1d"):
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))   # restored afterwards
    monkeypatch.setattr(importlib.util, "find_spec", lambda n, *a: object())
    monkeypatch.setattr(importlib.metadata, "version", lambda n: "5.17.0")
    assert runtime.keep_fused_kernels_off("cuda") == []
    assert runtime.keep_fused_kernels_off("cpu") == ["fla", "causal_conv1d"]
    assert sys.modules["fla"] is None and sys.modules["causal_conv1d"] is None


def test_older_transformers_are_left_alone(monkeypatch):
    import importlib.metadata
    monkeypatch.delitem(sys.modules, "transformers.models.qwen3_5.modeling_qwen3_5",
                        raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda n: "5.16.2")
    assert runtime.keep_fused_kernels_off("cpu") == []
