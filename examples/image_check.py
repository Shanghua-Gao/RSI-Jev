"""Ask a vision release about images, in-process.

    pip install "rsi-jev[vision]"
    python examples/image_check.py                 # v4.0-vl-2b from the Hugging Face Hub
    python examples/image_check.py path/to/release

Two checks an agent might run before acting: is a dashboard chart trending up, down
or flat, and does a screenshot show an error dialog. The pictures are drawn by
serve/demo_images.py, so the right answers are known: down, and yes (a disk-full
dialog). Both questions about one image read the image once.

Output, measured with v4.0-VL on an HP ZGX Nano (bf16):

    Decider('rsi-jev-v4.0-vl-qwen3.5-2b', device='cuda', dtype='bf16', calibration='oof_head_scorefloor')
    chart       trend=down   p(up/down/flat) = 0.00/1.00/0.00
                below_target: p(yes) = 0.95
    screenshot  error dialog: p(yes) = 0.99
                cause=storage p = 0.99
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # from a clone, uninstalled

from rsijev import Decider                                          # noqa: E402
from serve.demo_images import chart, screenshot                    # noqa: E402

d = Decider(sys.argv[1] if len(sys.argv) > 1 else "v4.0-vl-2b")
print(d)

# A PIL image, a file path, raw bytes or a data URL all work as an image.
chart_png, _ = chart()
a = d.decide("Weekly product dashboard: <image>", {
    "trend": {"type": "choice", "instructions": "Is this chart's trend up, down or flat?",
              "criteria": {"up": "It rises over the period.", "down": "It falls over the period.",
                           "flat": "It stays roughly level."}},
    "below_target": {"type": "noul",
                     "instructions": "Is the most recent value below the dashed target line?"},
}, images=[chart_png])
p = a["trend"]["probabilities"]
print(f"chart       trend={a['trend']['choice']:<6} "
      f"p(up/down/flat) = {p['up']:.2f}/{p['down']:.2f}/{p['flat']:.2f}")
print(f"            below_target: p(yes) = {a['below_target']['noul']:.2f}")

shot, _ = screenshot()
a = d.decide("A user attached this screenshot to a ticket: <image>", {
    "error_dialog": {"type": "noul", "instructions": "Does this screenshot show an error dialog?"},
    "cause": {"type": "choice", "instructions": "What is the error about?",
              "criteria": {"storage": "Disk space or storage.", "network": "Connectivity.",
                           "permission": "Access rights or sign-in.", "none": "There is no error."}},
}, images=[shot])
print(f"screenshot  error dialog: p(yes) = {a['error_dialog']['noul']:.2f}")
c = a["cause"]
print(f"            cause={c['choice']:<6} p = {c['probabilities'][c['choice']]:.2f}")
