"""RSI-Jev v3.0 interactive demo (a Gradio app for a Hugging Face Space).

    python space/app.py                         # from a checkout of the repo
    RSIJEV_CKPT=path/to/release python space/app.py
    PORT=7861 python space/app.py

Loads v3.0 with the repo's own `scripts/load_release.py` and answers through the
same wire mapping (`serve.wire`) and forward pass (`serve.infer`) as the API
server, so what the page shows is what `POST /v1/systemone` returns.

Runs on CPU (fp32 tower) or GPU (bf16 tower); the scorer is fp32 either way.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# In the repo, the code lives one level up; in an assembled Space, next to app.py.
ROOT = HERE if (HERE / "rsijev").is_dir() else HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "scripts"))        # after ROOT: scripts/serve.py shadows serve/

import gradio as gr                            # noqa: E402
import torch                                   # noqa: E402

from serve.wire import RequestError, parse_questions, state_to_text, to_answer  # noqa: E402

CKPT = os.environ.get("RSIJEV_CKPT", "shgao/rsi-jev-v3.0-qwen3.5-2b")
THREADS = int(os.environ.get("RSIJEV_THREADS", "0")) or None
# Tower precision, bf16 by default on CPU and GPU; "auto" is serve.py's rule (bf16 on
# GPU, fp32 on CPU). The scorer and calibration stay fp32 either way.
# bf16 halves memory: the fp32 load peaks at 16.9 GB, over a free CPU Space's 16 GB.
DTYPE = os.environ.get("RSIJEV_DTYPE", "bf16")

try:                                           # ZeroGPU Spaces provide `spaces`
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn


def presets() -> dict[str, tuple[str, str]]:
    """Worked examples the repo already documents: serve/README.md's request and
    serve/examples.json. Name -> (state text, questions JSON)."""
    out = {}
    readme = ROOT / "serve" / "README.md"
    if readme.exists():
        m = re.search(r"-d '(\{.*?\})'\n```", readme.read_text(), re.S)
        if m:
            req = json.loads(m.group(1))
            out["Refund request (serve/README.md)"] = (state_to_text(req["state"]),
                                                       json.dumps(req["questions"], indent=2))
    ex = ROOT / "serve" / "examples.json"
    if ex.exists():
        for e in json.loads(ex.read_text()):
            out[e["title"]] = (state_to_text(e["state"]), json.dumps(e["questions"], indent=2))
    return out


PRESETS = presets()
_model = None


def model():
    global _model
    if _model is None:
        from load_release import load_release
        path = CKPT
        if not Path(path).exists():
            from huggingface_hub import snapshot_download
            path = snapshot_download(CKPT)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and THREADS:
            torch.set_num_threads(THREADS)
        name = DTYPE if DTYPE != "auto" else ("bf16" if device == "cuda" else "fp32")
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[name]
        m, tok, enc, meta = load_release(path, device, infer_dtype=dtype)
        _model = (m, tok, enc, meta, device, name)
    return _model


@gpu
def score(state: str, questions):
    from serve.infer import score_questions_cached
    m, tok, enc, meta, device, _ = model()
    preds, tokens = score_questions_cached(
        m, tok, state, questions, enc, device=device, batch_size=16,
        max_options=max(meta["spec"]["max_options"], max(len(q.options) for q in questions)))
    return [list(p.probs) for p in preds], tokens


def bars(questions, answers) -> str:
    rows = []
    for q in questions:
        a = answers[q.key]
        if q.mode == "noul":
            dist = {"false": 1 - a["noul"], "true": a["noul"]}
            head = f"p(true) = {a['noul']:.3f}"
        elif q.mode == "score":
            dist = a["probabilities"]
            head = f"score = {a['score']:.2f} of {len(dist) - 1}"
        else:
            dist = a["probabilities"]
            head = f"choice = {html.escape(a['choice'])}"
        best = max(dist, key=dist.get)
        lines = []
        for k, p in dist.items():
            label = k if q.mode != "score" else f"{k}: {q.criteria[k]}"
            color = "#2f6fdf" if k == best else "#9db5e0"
            lines.append(
                f"<div class='row'><span class='lab' title='{html.escape(label)}'>"
                f"{html.escape(label)}</span><span class='bar'><span style='width:{100 * p:.1f}%;"
                f"background:{color}'></span></span><span class='num'>{p:.3f}</span></div>")
        rows.append(f"<div class='q'><div class='qh'><b>{html.escape(q.key)}</b> "
                    f"<span class='mode'>{q.mode}</span> &nbsp;{head}</div>{''.join(lines)}</div>")
    return "<div class='bars'>" + "".join(rows) + "</div>"


def answer(state_text: str, questions_json: str):
    try:
        questions = parse_questions(json.loads(questions_json))
    except json.JSONDecodeError as e:
        raise gr.Error(f"Questions are not valid JSON: {e}")
    except RequestError as e:
        raise gr.Error(str(e))
    state = state_to_text(state_text)
    t0 = time.perf_counter()
    probs, tokens = score(state, questions)
    ms = (time.perf_counter() - t0) * 1000
    answers = {q.key: to_answer(q, p) for q, p in zip(questions, probs)}
    *_, device, dtype = model()
    info = (f"{len(questions)} question(s), {tokens} prompt tokens, "
            f"**{ms:.0f} ms** on {device} ({dtype} tower)")
    return bars(questions, answers), info, {"answers": answers,
                                            "usage": {"input_tokens": tokens,
                                                      "output_tokens": len(questions)}}


def load_preset(name: str):
    return PRESETS[name]


CSS = """
.bars .q {margin: 0 0 14px 0}
.bars .qh {margin-bottom: 4px}
.bars .mode {font-size: 12px; padding: 1px 6px; border-radius: 8px; background: #e6ecf7; color: #333}
.bars .row {display: flex; align-items: center; gap: 8px; font-size: 13px; margin: 2px 0}
.bars .lab {width: 38%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap}
.bars .bar {flex: 1; height: 12px; background: #eef1f6; border-radius: 3px; overflow: hidden}
.bars .bar span {display: block; height: 100%}
.bars .num {width: 48px; text-align: right; font-variant-numeric: tabular-nums}
"""

INTRO = """# RSI-Jev v3.0 — typed decisions
Give it a **state** (a document, a ticket, a chat transcript as JSON) and **typed questions**:
`choice` (pick one key of `criteria`), `noul` (yes/no; `p(true)`), `score` (a rubric of levels;
the expected level). Nothing is generated: one forward pass returns a probability for every option.
Option keys are visible to the model, so renaming a key can move the answer.
"""


def build() -> gr.Blocks:
    first = next(iter(PRESETS)) if PRESETS else None
    state0, q0 = PRESETS[first] if first else ("", "{}")
    with gr.Blocks(title="RSI-Jev v3.0") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                preset = gr.Dropdown(list(PRESETS), value=first, label="Example")
                state = gr.Textbox(value=state0, label="State", lines=12, max_lines=30)
                questions = gr.Code(value=q0, language="json", label="Questions (Jev wire format)",
                                    lines=14)
                go = gr.Button("Answer", variant="primary")
            with gr.Column(scale=1):
                info = gr.Markdown()
                out = gr.HTML()
                raw = gr.JSON(label="Response body (answers, usage)")
        preset.change(load_preset, preset, [state, questions])
        go.click(answer, [state, questions], [out, info, raw], api_name="answer")
    return demo


if __name__ == "__main__":
    if os.environ.get("RSIJEV_EAGER", "1") == "1":
        model()                                    # load before the first request
    build().launch(server_name=os.environ.get("HOST", "0.0.0.0"),
                   server_port=int(os.environ.get("PORT", "7860")), css=CSS)
