"""The demo's image examples, drawn here so they carry no licence.

    python -m serve.demo_images            # rewrite the image examples in serve/examples.json

Three pictures, each with questions whose answers are true by construction:

  * a line chart of weekly active users, falling below a dashed target line;
  * a screenshot of a notes app with a "Couldn't save: disk full" dialog;
  * a grocery receipt, paid by card, total under 25.

Everything is drawn with Pillow (no matplotlib, no fonts downloaded: DejaVu when
the system has it, Pillow's own font otherwise), so the files are small and the
script reruns anywhere. Each image is embedded in examples.json as a data URL
and stays under 150 KB.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 150 * 1024


def _font(size: int, bold: bool = False, mono: bool = False):
    from PIL import ImageFont
    name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu",
              "/Library/Fonts", "C:/Windows/Fonts"):
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def chart():
    """Weekly active users over 12 weeks, falling from ~9.8k to ~6.1k, below a target."""
    from PIL import Image, ImageDraw
    W, H = 640, 400
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((24, 16), "Weekly active users", fill="#111", font=_font(20, bold=True))
    d.text((24, 44), "Last 12 weeks", fill="#666", font=_font(13))
    x0, x1, y0, y1 = 80, W - 30, 80, H - 60           # plot box
    lo, hi = 5000, 10500
    ys = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)
    for v in range(5000, 10501, 1000):
        d.line([(x0, ys(v)), (x1, ys(v))], fill="#e6e6e6")
        d.text((x0 - 10, ys(v)), f"{v // 1000}k", fill="#555", font=_font(12), anchor="rm")
    d.line([(x0, y0), (x0, y1), (x1, y1)], fill="#333", width=1)
    vals = [9800, 9650, 9700, 9300, 8950, 8800, 8400, 7900, 7600, 7050, 6600, 6100]
    xs = [x0 + 20 + i * (x1 - x0 - 40) / (len(vals) - 1) for i in range(len(vals))]
    for i, x in enumerate(xs):
        d.text((x, y1 + 8), f"W{i + 1}", fill="#555", font=_font(11), anchor="mt")
    t = ys(8000)                                       # dashed target line
    for x in range(x0, x1, 12):
        d.line([(x, t), (min(x + 6, x1), t)], fill="#c0392b", width=2)
    d.text((x1 - 4, t - 6), "target 8k", fill="#c0392b", font=_font(12), anchor="rb")
    pts = [(x, ys(v)) for x, v in zip(xs, vals)]
    d.line(pts, fill="#1f6fb2", width=3)
    for x, y in pts:
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill="#1f6fb2")
    d.text((W / 2, H - 18), "Week", fill="#333", font=_font(12), anchor="mm")
    return im, "PNG"


def screenshot():
    """A notes app with a modal "Couldn't save" dialog: the disk is full."""
    from PIL import Image, ImageDraw
    W, H = 720, 450
    im = Image.new("RGB", (W, H), "#f3f3f3")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 34], fill="#dcdcdc")
    for i, c in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        d.ellipse([12 + i * 20, 11, 24 + i * 20, 23], fill=c)
    d.text((W / 2, 17), "Notes — Quarterly plan", fill="#222", font=_font(14), anchor="mm")
    d.rectangle([0, 34, W, 70], fill="#fafafa")
    for i, label in enumerate(("File", "Edit", "Format", "View", "Help")):
        d.text((16 + i * 64, 52), label, fill="#333", font=_font(13), anchor="lm")
    d.rectangle([40, 90, W - 40, H - 30], fill="white", outline="#ddd")
    lines = ["Quarterly plan", "", "1. Hire two support engineers by March.",
             "2. Move billing to the new provider.", "3. Cut first-response time to 4 hours.",
             "4. Review the churn dashboard every Monday."]
    for i, line in enumerate(lines):
        d.text((64, 110 + i * 26), line, fill="#222" if i else "#000",
               font=_font(18 if i == 0 else 14, bold=(i == 0)))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 90))      # dim the window behind the dialog
    im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    d = ImageDraw.Draw(im)
    bx0, by0, bx1, by1 = 170, 130, W - 170, 330
    d.rounded_rectangle([bx0, by0, bx1, by1], radius=10, fill="white", outline="#bbb")
    d.ellipse([bx0 + 22, by0 + 24, bx0 + 62, by0 + 64], fill="#d93025")
    d.text((bx0 + 42, by0 + 44), "!", fill="white", font=_font(26, bold=True), anchor="mm")
    d.text((bx0 + 80, by0 + 30), "Couldn't save", fill="#111", font=_font(18, bold=True))
    body = ["There is not enough disk space to save", "\"Quarterly plan\". Free up space and",
            "try again. Unsaved changes will be lost", "if you close the app."]
    for i, line in enumerate(body):
        d.text((bx0 + 80, by0 + 62 + i * 20), line, fill="#333", font=_font(13))
    d.rounded_rectangle([bx1 - 220, by1 - 50, bx1 - 124, by1 - 18], radius=6,
                        fill="#eee", outline="#bbb")
    d.text((bx1 - 172, by1 - 34), "Cancel", fill="#222", font=_font(13), anchor="mm")
    d.rounded_rectangle([bx1 - 112, by1 - 50, bx1 - 16, by1 - 18], radius=6, fill="#1a73e8")
    d.text((bx1 - 64, by1 - 34), "Try again", fill="white", font=_font(13), anchor="mm")
    return im, "PNG"


def receipt():
    """A grocery receipt, card payment approved, total 23.95."""
    from PIL import Image, ImageDraw
    W, H = 380, 600
    im = Image.new("RGB", (W, H), "#e9e4da")
    d = ImageDraw.Draw(im)
    d.rectangle([30, 20, W - 30, H - 20], fill="#fdfcf8")
    mono, big = _font(14, mono=True), _font(18, bold=True, mono=True)
    y = 40
    for text, f in (("CORNER MARKET", big), ("12 Elm Street", mono), ("2026-09-14  17:42", mono)):
        d.text((W / 2, y), text, fill="#222", font=f, anchor="mt")
        y += 26
    y += 10
    items = [("Bananas 1.2 lb", 1.29), ("Whole milk 1 gal", 3.49), ("Sourdough bread", 2.99),
             ("Eggs, dozen", 4.19), ("Coffee beans 12oz", 11.99)]
    for name, price in items:
        d.text((50, y), name, fill="#222", font=mono)
        d.text((W - 50, y), f"{price:.2f}", fill="#222", font=mono, anchor="ra")
        y += 24
    total = sum(p for _, p in items)
    d.line([(50, y + 4), (W - 50, y + 4)], fill="#999")
    y += 14
    for name, v, f in (("SUBTOTAL", total, mono), ("TAX", 0.0, mono), ("TOTAL", total, big)):
        d.text((50, y), name, fill="#111", font=f)
        d.text((W - 50, y), f"{v:.2f}", fill="#111", font=f, anchor="ra")
        y += 28
    y += 8
    for line in ("VISA **** 4821", "AUTH 083311  APPROVED", "", "THANK YOU FOR SHOPPING"):
        d.text((W / 2, y), line, fill="#333", font=mono, anchor="mt")
        y += 22
    d.rounded_rectangle([W / 2 - 70, y + 16, W / 2 + 70, y + 66], radius=6, outline="#b03a2e",
                        width=4)
    d.text((W / 2, y + 41), "PAID", fill="#b03a2e", font=_font(28, bold=True), anchor="mm")
    return im, "JPEG"


def data_url(im, fmt: str) -> str:
    buf = io.BytesIO()
    im.save(buf, format=fmt, **({"quality": 85} if fmt == "JPEG" else {"optimize": True}))
    b = buf.getvalue()
    assert len(b) < MAX_BYTES, (fmt, len(b))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(b).decode("ascii")


def examples() -> list[dict]:
    yes_no = lambda t, f: {"true": t, "false": f}
    return [
        {"id": "image_chart_000000", "title": "Image: dashboard chart",
         "state": json.dumps({"source": "Weekly product dashboard, exported as an image",
                              "chart": "<image>",
                              "note": "Posted in #growth with no comment."}, indent=2),
         "images": [data_url(*chart())],
         "questions": {
             "trend": {"type": "choice", "instructions": "What is the overall trend of the plotted metric?",
                       "criteria": {"up": "It rises over the period.",
                                    "down": "It falls over the period.",
                                    "flat": "It stays roughly level."}},
             "below_target": {"type": "noul",
                              "instructions": "Is the most recent value below the target line?",
                              "criteria": yes_no("The last point is under the target.",
                                                 "The last point is at or above the target.")},
             "chart_kind": {"type": "choice", "instructions": "What kind of chart is this?",
                            "criteria": {"line": "A line chart.", "bar": "A bar chart.",
                                         "pie": "A pie chart.", "table": "A table of numbers."}}},
         "gold": {"trend": "down", "below_target": "true", "chart_kind": "line"}},
        {"id": "image_screenshot_000000", "title": "Image: app screenshot",
         "state": json.dumps({"ticket": "App keeps showing something when I try to save, help",
                              "plan": "free", "screenshot": "<image>"}, indent=2),
         "images": [data_url(*screenshot())],
         "questions": {
             "error_dialog": {"type": "noul", "instructions": "Does the screenshot show an error dialog?",
                              "criteria": yes_no("An error or warning dialog is on screen.",
                                                 "No error is shown.")},
             "cause": {"type": "choice", "instructions": "What is the error about?",
                       "criteria": {"storage": "Disk space or storage.",
                                    "network": "Connectivity or a server.",
                                    "permission": "Access rights or sign-in.",
                                    "crash": "The app crashed.",
                                    "none": "There is no error."}},
             "impact": {"type": "score", "instructions": "How badly is the user's work affected?",
                        "criteria": ["Not at all.", "A cosmetic annoyance.",
                                     "Work is blocked until the problem is fixed.",
                                     "Work has already been lost."]}},
         "gold": {"error_dialog": "true", "cause": "storage", "impact": "2"}},
        {"id": "image_receipt_000000", "title": "Image: expense receipt",
         "state": json.dumps({"expense_claim": {"employee": "J. Ortiz", "claimed_usd": 23.95,
                                                "policy": "groceries are not reimbursable"},
                              "receipt": "<image>"}, indent=2),
         "images": [data_url(*receipt())],
         "questions": {
             "paid": {"type": "noul", "instructions": "Does the receipt show that it was paid?",
                      "criteria": yes_no("Payment is recorded as made.",
                                         "It is unpaid or the payment is unclear.")},
             "category": {"type": "choice", "instructions": "What kind of purchase is this?",
                          "criteria": {"groceries": "Food and household goods from a store.",
                                       "restaurant": "A meal at a restaurant or cafe.",
                                       "fuel": "Petrol or charging.",
                                       "electronics": "Devices or accessories."}},
             "matches_claim": {"type": "noul",
                               "instructions": "Does the receipt total match the amount claimed?",
                               "criteria": yes_no("The totals are the same.",
                                                  "They differ.")}},
         "gold": {"paid": "true", "category": "groceries", "matches_claim": "true"}},
    ]


def main() -> int:
    path = ROOT / "examples.json"
    data = [e for e in json.loads(path.read_text()) if not e.get("images")]
    data += examples()
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    for e in data:
        if e.get("images"):
            kb = [len(u) * 3 // 4 // 1024 for u in e["images"]]
            print(f"{e['id']}: {kb} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
