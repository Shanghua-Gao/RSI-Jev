"""encode_question over-cap policy: middle-cut with a visible marker (the long-context
encoder, ported from the private branch longctx e1069bf with its tests).

The public default stays truncate="left", so `MID` below asks for "middle" where the
private tests relied on it being the default.

1. A row that fits max_length encodes exactly as the v1.0-v4.0-VL encoder did
   (the legacy module: `git show 0af7fe0:rsijev/encode.py`, or LEGACY_ENCODE=<path>).
2. An over-cap row keeps the state's leading query line and its last line,
   the question, every option and the cue; it carries the marker and
   state_cut > 0, fits max_length, and its indices point where they should.
3. truncate="left" is the legacy behaviour, byte for byte, plus state_cut.
4. Tail-weighted split: the tail keeps ~3x the head's tokens, and the first line survives.
5. Option-block policy (Janus WoS, 145 options, ~4k-token option block): never refuse; descriptions trimmed
   evenly, labels intact, options_cut > 0; question text cut in the middle only when the labels alone fill the cap.

Needs the Qwen3.5 tokenizer (HF cache); CPU only, seconds.
    python -m pytest tests/test_encode_truncation.py -q
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsijev.contract import Question                                   # noqa: E402
from rsijev.encode import EncodeConfig as _EncodeConfig, encode_question  # noqa: E402

LEGACY_REV = "0af7fe0"          # v4-vision-serve, the encoder of every release so far
FLAGS = ("state_cut", "options_cut", "question_cut")


def EncodeConfig(**kw):                                                 # noqa: N802
    """The long-context encoder unless a test asks for another policy."""
    kw.setdefault("truncate", "middle")
    return _EncodeConfig(**kw)


def _legacy():
    path = os.environ.get("LEGACY_ENCODE")
    if not path:
        try:
            src = subprocess.run(["git", "-C", str(ROOT), "show", f"{LEGACY_REV}:rsijev/encode.py"],
                                 check=True, capture_output=True, text=True).stdout
        except Exception:                                                # noqa: BLE001
            pytest.skip("legacy encoder unavailable (no git, no LEGACY_ENCODE)")
        path = str(Path(tempfile.mkdtemp()) / "legacy_encode.py")
        Path(path).write_text(src.replace("from .contract import", "from rsijev.contract import"))
    else:
        src = Path(path).read_text()
        path = str(Path(tempfile.mkdtemp()) / "legacy_encode.py")
        Path(path).write_text(src.replace("from .contract import", "from rsijev.contract import"))
    spec = importlib.util.spec_from_file_location("legacy_encode", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["legacy_encode"] = mod              # dataclasses look the module up
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER", "Qwen/Qwen3.5-2B-Base"))


def _q(mode="choice"):
    if mode == "noul":
        return Question(key="k", mode="noul", instructions="Does the candidate answer the query?",
                        options=("false", "true"), criteria={"false": "No", "true": "Yes"})
    return Question(key="k", mode="choice", instructions="Which category is page 3?",
                    options=("report", "article", "other"),
                    criteria={"report": "A report.", "article": "A research article.", "other": "None."})


FILLER = " ".join(f"Sentence {i} of the middle of a long document." for i in range(3000))
LONG = "Query: where did I park the blue car?\n\n" + FILLER + "\n\nLAST LINE: the end of the state."


@pytest.mark.parametrize("mode", ["choice", "noul"])
@pytest.mark.parametrize("layout", ["state_first", "options_first"])
@pytest.mark.parametrize("state", ["", "A short state.", "Query: q\n\n" + FILLER[:4000]],
                         ids=["empty", "short", "mid"])
def test_under_cap_identical(tok, mode, layout, state):
    L = _legacy()
    cfg = EncodeConfig(layout=layout, max_length=4096)
    old = L.encode_question(tok, state, _q(mode), L.EncodeConfig(layout=layout, max_length=4096))
    new = encode_question(tok, state, _q(mode), cfg)
    assert len(old["input_ids"]) <= 4096
    assert new["state_cut"] == 0 and new["options_cut"] == 0
    assert {k: v for k, v in new.items() if k not in FLAGS} == old


@pytest.mark.parametrize("mode", ["choice", "noul"])
@pytest.mark.parametrize("cap", [2048, 8192])
def test_over_cap_keeps_query_and_tail(tok, mode, cap):
    q = _q(mode)
    e = encode_question(tok, LONG, q, EncodeConfig(max_length=cap))
    ids = e["input_ids"]
    text = tok.decode(ids)
    assert len(ids) <= cap
    assert e["state_cut"] > 0
    assert text.startswith("Query: where did I park the blue car?")
    assert "LAST LINE: the end of the state." in text
    assert f"[... {e['state_cut']} tokens of the state omitted ...]" in text
    assert text.endswith("\n\nAnswer:")
    assert q.instructions + "\nOptions:\n" in text
    assert e["decision_index"] == len(ids) - 1
    for (a, b), o, oi in zip(e["option_span"], e["options"], e["option_index"]):
        assert tok.decode(ids[a:b]).startswith(f"\n- {o}")
        assert oi == b - 1
    # the full state minus the cut is what remains
    n_state = len(tok(LONG, add_special_tokens=False)["input_ids"])
    assert e["state_cut"] < n_state


def test_over_cap_options_first(tok):
    q = _q()
    e = encode_question(tok, LONG, q, EncodeConfig(layout="options_first", max_length=2048))
    text = tok.decode(e["input_ids"])
    assert len(e["input_ids"]) <= 2048 and e["state_cut"] > 0
    assert text.startswith(q.instructions + "\nOptions:\n")
    assert "Query: where did I park the blue car?" in text and "LAST LINE" in text
    assert text.endswith("\n\nAnswer:") and e["decision_index"] == len(e["input_ids"]) - 1


def test_left_is_legacy(tok):
    L = _legacy()
    old = L.encode_question(tok, LONG, _q(), L.EncodeConfig(max_length=2048))
    new = encode_question(tok, LONG, _q(), EncodeConfig(max_length=2048, truncate="left"))
    assert new["state_cut"] > 0
    assert {k: v for k, v in new.items() if k not in FLAGS} == old
    assert not tok.decode(old["input_ids"]).startswith("Query:")        # the bug being fixed


def test_tail_weighted(tok):
    e = encode_question(tok, LONG, _q(), EncodeConfig(max_length=2048))
    text = tok.decode(e["input_ids"])
    marker = f"[... {e['state_cut']} tokens of the state omitted ...]"
    head, tail = text.split(marker)
    n_head = len(tok(head, add_special_tokens=False)["input_ids"])
    n_tail = len(tok(tail, add_special_tokens=False)["input_ids"])
    assert head.startswith("Query: where did I park the blue car?")
    assert 2.0 < n_tail / n_head < 4.5, (n_head, n_tail)


def test_query_line_kept_when_long(tok):
    """a first line longer than head_frac of the room is still kept whole (up to half the room)"""
    st = "Query: " + "a very long question " * 60 + "\n\n" + FILLER
    e = encode_question(tok, st, _q(), EncodeConfig(max_length=2048))
    assert tok.decode(e["input_ids"]).startswith(st.split("\n")[0])


def _janus(n=145):
    labels = [f"wos_{i:03d}" for i in range(n)]
    crit = {o: f"Research area {i}: papers on topic {i} including methods, datasets, theory and applications "
               f"in subfield {i} and its neighbours." for i, o in enumerate(labels)}
    return Question(key="field", mode="choice", instructions="Which Web of Science research area fits the abstract?",
                    options=tuple(labels), criteria=crit)


@pytest.mark.parametrize("state", ["", "Abstract: " + FILLER[:12000]], ids=["no_state", "long_state"])
def test_option_block_never_refuses(tok, state):
    q = _janus()
    L = _legacy()
    full = L.encode_question(tok, state, q, L.EncodeConfig(max_length=10 ** 9))
    n_opt = full["option_span"][-1][1] - full["option_span"][0][0]
    assert 3500 < n_opt < 6000, n_opt                                   # a ~4k-token option block
    with pytest.raises(ValueError):                                      # the legacy encoder refused
        L.encode_question(tok, state, q, L.EncodeConfig(max_length=2048))
    e = encode_question(tok, state, q, EncodeConfig(max_length=2048))
    ids = e["input_ids"]
    assert len(ids) <= 2048 and e["options_cut"] > 0
    assert e["decision_index"] == len(ids) - 1 and tok.decode(ids).endswith("\n\nAnswer:")
    assert q.instructions in tok.decode(ids)
    assert len(e["option_span"]) == 145
    for (a, b), o, oi in zip(e["option_span"], e["options"], e["option_index"]):
        assert tok.decode(ids[a:b]).startswith(f"\n- {o}")                 # every label intact
        assert oi == b - 1
    if state:
        assert tok.decode(ids).startswith("Abstract:") and e["state_cut"] > 0
    # at 8k the same row fits whole and encodes exactly as before
    e8 = encode_question(tok, state, q, EncodeConfig(max_length=8192))
    old8 = L.encode_question(tok, state, q, L.EncodeConfig(max_length=8192))
    if len(old8["input_ids"]) <= 8192 and not e8["state_cut"]:
        assert {k: v for k, v in e8.items() if k not in FLAGS} == old8


def test_question_cut_last_resort(tok):
    q = Question(key="k", mode="choice", instructions="x " * 3000, options=("a", "b"),
                 criteria={"a": "a", "b": "b"})
    e = encode_question(tok, LONG, q, EncodeConfig(max_length=512))
    assert len(e["input_ids"]) <= 512 and e["question_cut"] > 0
    assert tok.decode(e["input_ids"]).endswith("\n\nAnswer:")


def test_labels_alone_too_long_raises(tok):
    with pytest.raises(ValueError):
        encode_question(tok, "", _janus(), EncodeConfig(max_length=256))


def test_structured_criteria_identical_to_rt4(tok):
    """Structured option descriptions (criterion_text, retrain_v4 corpora) encode exactly as the rt4 encoder under
    the cap. LEGACY_CRIT=<rt4 encode.py>; skipped without it."""
    path = os.environ.get("LEGACY_CRIT")
    if not path:
        pytest.skip("LEGACY_CRIT not set")
    src = Path(path).read_text().replace("from .contract import", "from rsijev.contract import")
    p2 = Path(tempfile.mkdtemp()) / "rt4_encode.py"; p2.write_text(src)
    spec = importlib.util.spec_from_file_location("rt4_encode", p2)
    mod = importlib.util.module_from_spec(spec); sys.modules["rt4_encode"] = mod; spec.loader.exec_module(mod)
    q = Question(key="k", mode="choice", instructions="Which tier?", options=("low", "high"),
                 criteria={"low": {"what": "Low effort", "includes": ["typos", "renames"], "excludes": ["refactors"]},
                           "high": {"what": "High effort", "examples": 3}})
    for st in ("", "A short state.", "Query: q\n\n" + FILLER[:4000]):
        old = mod.encode_question(tok, st, q, mod.EncodeConfig(max_length=4096))
        new = encode_question(tok, st, q, EncodeConfig(max_length=4096))
        assert {k: v for k, v in new.items() if k not in FLAGS} == old
    e = encode_question(tok, LONG, q, EncodeConfig(max_length=2048))
    assert "Low effort. Includes: typos; renames. Excludes: refactors." in tok.decode(e["input_ids"])


def test_public_default_is_the_legacy_encoder(tok):
    """Every release so far was trained with left truncation: the default config is it,
    byte for byte, structured descriptions included."""
    L = _legacy()
    assert _EncodeConfig().truncate == "left"
    q = Question(key="k", mode="choice", instructions="Which tier?", options=("low", "high"),
                 criteria={"low": {"what": "Low effort"}, "high": "High effort."})
    for st in ("", "A short state.", LONG):
        old = L.encode_question(tok, st, q, L.EncodeConfig(max_length=2048))
        new = encode_question(tok, st, q, _EncodeConfig(max_length=2048))
        assert {k: v for k, v in new.items() if k not in FLAGS} == old
