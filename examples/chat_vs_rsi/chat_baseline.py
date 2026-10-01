"""Side (b): the same base, Qwen3.5-2B, as a chat model (non-thinking), writing a JSON answer.
Writes chat.json. Needs a CUDA GPU.

    pip install "transformers>=5.17" pillow torchvision
    python examples/chat_vs_rsi/chat_baseline.py [--out chat.json]

Qwen/Qwen3.5-2B is the vision-capable chat checkpoint released beside Qwen3.5-2B-Base, the
base RSI-Jev v4.0-VL is trained from. bf16, greedy, at most 256 new tokens, one process.
The prompt: the same state and question, the allowed answers listed with each option's
description, "reply with only a JSON object". Images go where the state's <image>
markers are. 2 warm runs, 5 timed, p50; timed = processor (image preprocessing,
tokenising) + generate. One more streamed run per question records when each token
arrived, for the side-by-side GIF.

A reply is valid when it parses as JSON and the question's id holds one of the allowed
answers. That strict rule was fixed before the run; summarize.py also reports a
lenient re-read, defined after seeing the replies.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.generation.streamers import BaseStreamer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rsi_client import images_of, load_questions          # noqa: E402

M = "Qwen/Qwen3.5-2B"
proc = None
model = None


def load(im) -> Image.Image:
    if isinstance(im, Image.Image):
        return im.convert("RGB")
    return ImageOps.exif_transpose(Image.open(im)).convert("RGB")


def allowed(spec: dict) -> dict:
    if spec["type"] == "noul":
        return {"yes": "Yes", "no": "No"}
    if spec["type"] == "choice":
        return {k: (v or "") for k, v in spec["criteria"].items()}
    return {str(i): t for i, t in enumerate(spec["criteria"])}


def prompt_content(q: dict, n_images: int) -> tuple[list, dict]:
    """The user turn: text parts and image slots in state order, then the instruction."""
    opts = allowed(q["spec"])
    lines = "\n".join(f'  "{k}"' + (f": {d}" if d and d.lower() != k.lower() else "") for k, d in opts.items())
    ask = (f"Answer this question. Reply with only a JSON object whose key is the question id and "
           f'whose value is your answer, nothing else.\n- "{q["key"]}": {q["spec"]["instructions"]} '
           f"Answer with one of:\n{lines}")
    state = q["state"] if "<image>" in q["state"] else "<image>" * n_images + q["state"]
    parts = state.split("<image>")
    content = []
    for i, part in enumerate(parts):
        if part.strip():
            content.append({"type": "text", "text": part.strip() + "\n"})
        if i < len(parts) - 1:
            content.append({"type": "image"})
    content.append({"type": "text", "text": "\n" + ask})
    return content, opts


def messages(q: dict):
    imgs = [load(im) for im in images_of(q)]
    content, opts = prompt_content(q, len(imgs))
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    return text, imgs, opts


class Clock(BaseStreamer):
    def __init__(self, t0):
        self.t0, self.first, self.events = t0, True, []

    def put(self, value):
        if self.first:           # the prompt
            self.first = False
            return
        torch.cuda.synchronize()
        self.events.append(((time.perf_counter() - self.t0) * 1000, value.reshape(-1).tolist()))

    def end(self):
        pass


def run(text, imgs, streamer=None):
    torch.cuda.synchronize()
    t = time.perf_counter()
    if streamer is not None:
        streamer.t0 = t
    x = proc(text=[text], images=imgs, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**x, max_new_tokens=256, do_sample=False, streamer=streamer)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) * 1000
    new = out[0, x["input_ids"].shape[1]:]
    return proc.decode(new, skip_special_tokens=True), ms, int(new.shape[0]), int(x["input_ids"].shape[1])


def parse(text: str, key: str, opts: dict):
    """(answer, valid_json, valid): the strict rule fixed before the run."""
    try:
        s = text.strip().strip("`")
        s = s[s.index("{"): s.rindex("}") + 1]
        obj = json.loads(s)
    except Exception:
        return None, False, False
    v = obj.get(key) if isinstance(obj, dict) else None
    if isinstance(v, bool):
        v = "yes" if v else "no"
    v = str(v).strip().strip('"').lower() if v is not None else None
    lower = {k.lower(): k for k in opts}
    return (lower.get(v) if v is not None else None), True, v in lower


def main(argv=None) -> int:
    global proc, model
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "chat.json"))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    proc = AutoProcessor.from_pretrained(M)
    model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16).to("cuda").eval()
    qs = load_questions()[: a.limit or None]
    res = []
    for q in qs:
        text, imgs, opts = messages(q)
        run(text, imgs); run(text, imgs)
        runs = [run(text, imgs) for _ in range(5)]
        out_text, _, ntok, nin = runs[0]
        clock = Clock(0.0)
        s_text, s_ms, _, _ = run(text, imgs, clock)
        stream, ids = [], []
        for ms, toks in clock.events:
            ids += toks
            stream.append([round(ms, 1), proc.decode(ids, skip_special_tokens=True)])
        ans, valid_json, valid = parse(out_text, q["key"], opts)
        pick = ("true" if ans == "yes" else "false") if q["spec"]["type"] == "noul" and ans is not None else ans
        res.append(dict(id=q["id"], text=out_text, valid_json=valid_json, valid=valid, pick=pick,
                        correct=bool(valid and pick in q["ref"]), confidence_available=False,
                        ms_p50=round(statistics.median(r[1] for r in runs), 1),
                        ms_all=[round(r[1], 1) for r in runs], output_tokens=ntok, input_tokens=nin,
                        streamed=dict(ms_total=round(s_ms, 1), same_text=s_text == out_text, tokens=stream)))
        print(q["id"], res[-1]["ms_p50"], "ms", ntok, "tok", valid, pick, q["ref"],
              out_text[:120].replace("\n", " "), flush=True)
    Path(a.out).write_text(json.dumps(dict(model=M, prompt_example=messages(qs[0])[0], results=res), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
