"""RSI-Jev v5.0-VL 3B on a Hugging Face Space: ask a typed question about text and, optionally, an image.

The answer comes from the same `Decider` the package ships (`from rsijev import Decider`),
which runs the server's own request path. RSIJEV_MODEL picks the checkpoint: a Hugging Face
repo id, an alias or a local directory. On ZeroGPU hardware (the `spaces` package is present)
each answer runs inside a GPU slot; elsewhere the decorator is a no-op.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr

try:                                               # ZeroGPU: a GPU slot per call
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(fn):
        return fn

MODEL = os.environ.get("RSIJEV_MODEL", "shgao/rsi-jev-v5.0-vl-3b")
HERE = Path(__file__).resolve().parent
EX = HERE / "examples"

EXAMPLES = [
    [None, "Ticket #4471: The invoice PDF downloads but every page is blank. Tried Chrome and "
     "Safari. This is blocking our month-end close.",
     "Is the customer blocked from finishing their work?", "yes/no", ""],
    [str(EX / "stop.jpg"), "A frame from the car's front camera.",
     "What should the car do?", "choice",
     "stop_then_go: Stop, wait two seconds, then drive on\n"
     "wait_for_green: Stop and wait for a green light\n"
     "slow_down: Drive on at half speed\n"
     "continue: Drive on at the current speed"],
    [str(EX / "dog.jpg"), "", "Is there a cat in the image?", "yes/no", ""],
    [str(EX / "capsules.jpg"), "Production-line inspection photo of green gel capsules.",
     "Is everything in the photo good, or is at least one part defective?", "choice",
     "good: All parts look fine\ndefective: At least one part is damaged"],
    [str(EX / "checkout.png"), "Goal: finish paying for the order.",
     "Did the payment go through?", "choice",
     "paid: The payment succeeded\ndeclined: The payment was refused\n"
     "pending: The payment is still processing"],
    [str(EX / "chart.png"), "Weekly product dashboard.",
     "Is this chart's trend up, down or flat?", "choice", "up\ndown\nflat"],
]


def parse_options(text: str) -> dict[str, str | None]:
    """One option per line, `key` or `key: description`."""
    out: dict[str, str | None] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        key, _, desc = line.partition(":")
        key = key.strip()
        if key in out:
            raise gr.Error(f"option {key!r} is listed twice")
        out[key] = desc.strip() or None
    return out


def build_request(image, context: str, question: str, kind: str, options: str):
    """(state, questions) for Decider.decide, from the form fields."""
    if not (question or "").strip():
        raise gr.Error("Type a question.")
    context = (context or "").strip()
    if image is None:
        if not context:
            raise gr.Error("Give the model something to read: text, an image, or both.")
        state = context.replace("<image>", "").strip()
    else:
        state = context if "<image>" in context else ("<image>\n" + context).strip()
    if kind == "yes/no":
        q = {"type": "noul", "instructions": question.strip()}
    else:
        crit = parse_options(options)
        if len(crit) < 2:
            raise gr.Error("A choice question needs at least two options, one per line.")
        q = {"type": "choice", "instructions": question.strip(), "criteria": crit}
    return state, {"answer": q}


def make_answer(decider):
    @gpu
    def answer(image, context, question, kind, options):
        state, questions = build_request(image, context, question, kind, options)
        t = time.perf_counter()
        try:
            out = decider.request(state, questions, images=[image] if image is not None else None)
        except Exception as e:                     # a 422 from the request checks, shown as is
            raise gr.Error(str(e)) from e
        ms = (time.perf_counter() - t) * 1000
        a = out["answers"]["answer"]
        if a["type"] == "noul":
            probs = {"yes": a["noul"], "no": 1 - a["noul"]}
            line = f"**{'yes' if a['noul'] >= 0.5 else 'no'}** · p(yes) = {a['noul']:.2f}"
        else:
            probs = a["probabilities"]
            line = (f"**{a['choice']}** · p = {probs[a['choice']]:.2f} · "
                    f"confidence {a['confidence']:.2f}")
        line += f" · {ms:.0f} ms, {out['usage']['input_tokens']} input tokens"
        return probs, line, out
    return answer


def build_demo(decider) -> gr.Blocks:
    with gr.Blocks(title="RSI-Jev v5.0-VL 3B") as demo:
        gr.Markdown(
            "# RSI-Jev v5.0-VL 3B\n"
            "Ask a yes/no or multiple-choice question about a text, an image or both. The model "
            "runs the first 20 of Qwen3.5-4B-Base's 32 layers and scores every "
            "allowed answer in one forward pass and returns calibrated probabilities; it does not "
            "write text. [Code](https://github.com/Shanghua-Gao/RSI-Jev) · "
            f"model `{decider.name}` · calibration `{decider.calibration}`")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image (optional)")
                context = gr.Textbox(label="Text", lines=3,
                                     placeholder="The document, message or context. With an image, "
                                                 "put <image> where it belongs.")
                question = gr.Textbox(label="Question", placeholder="Is the part defective?")
                kind = gr.Radio(["yes/no", "choice"], value="yes/no", label="Answer type")
                options = gr.Textbox(label="Options (choice only), one per line: key or key: description",
                                     lines=4)
                go = gr.Button("Ask", variant="primary")
            with gr.Column():
                probs = gr.Label(label="Probabilities", num_top_classes=10)
                summary = gr.Markdown()
                raw = gr.JSON(label="Response, as POST /v1/systemone returns it")
        gr.Examples(EXAMPLES, [image, context, question, kind, options])
        go.click(make_answer(decider), [image, context, question, kind, options],
                 [probs, summary, raw], api_name="answer")
        gr.Markdown(
            "Example images: stop sign by Dori (public domain) and boxer dog by Joselodos (CC0), "
            "both via Wikimedia Commons; capsules from the VisA dataset, Zou et al. 2022, "
            "CC BY 4.0, resized; the checkout screen and the chart are ours (CC0).")
    return demo


if __name__ == "__main__":
    from rsijev import Decider
    build_demo(Decider(MODEL)).queue(max_size=16).launch()
