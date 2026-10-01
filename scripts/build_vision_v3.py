"""vision_v3: synthetic image decisions with CODE-COMPUTED labels (arm vis-ct-3 of v4.0-VL).

    python scripts/build_vision_v3.py --root OUT/vision_v3 --font-dir /usr/share/fonts [--n 1500] [--probe 300] [--only a,b]
    python scripts/decontam_vision.py v3 --root OUT/vision_v3 --eval-build OUT/eval_vision_v1_build [--demo DIR]

All data here is SYNTHETIC: every image is drawn by this script with PIL and every label comes from the
generator's own state (a solver / arithmetic / rule check), never from a model or a dataset. Nothing is
downloaded. Six generators: ui_state, change_detect, line_rule, grid_games, severity, receipt_math.
v4.0-VL: --n 1500 --probe 300 --seed 3 (10,347 train questions after balancing; vis-ct-3 used them all).

Template families: TRAIN draws layouts A and B; the PROBE uses layout family C only (different
layout, fonts and palette), so a probe measures transfer to an unseen template, not memorised pixels.
The release demo's scenes and strings are not reused.

Writes <root>/vis3_<gen>.jsonl + images/<gen>/..., <root>/probes_raw/probe_<gen>.jsonl + probes_raw/images/...,
<root>/gen_hashes.jsonl, <root>/probes_raw/gen_hashes.jsonl. `decontam_vision.py v3` then filters, writes
<root>/manifest.json and freezes <root>/probes/ (after which probes_raw/ can be deleted). gen_hashes.jsonl
files are appended to, so start from an empty root.

Fonts (not bundled; pass --font-dir). Searched (the build used a RHEL host's system fonts):
  A (train sans): DejaVu Sans / Sans Bold / Serif / Sans Condensed (Bitstream Vera + public-domain changes
      licence, free incl. commercial), Droid Sans / Bold (Apache-2.0)
  MONO (train):   DejaVu Sans Mono / Bold, Liberation Mono Regular / Bold (SIL OFL 1.1)
  C (probe only): Cantarell (SIL OFL 1.1), URW base35 OpenType (AGPL-3.0 with font exception)
The rng draws two fonts per image from a pool, so the pool SIZE feeds the random stream: the labels (every
vis3_*.jsonl byte) reproduce the v4.0-VL files when the sans pool (A) and the mono pool each hold 1, 2 or 4
fonts (random.choice consumes the same random words for those sizes); 3, 5 or 6 give different data. The
build host most likely had the four DejaVu faces of A (no Droid). Checked: with 4 sans + 4 mono fonts, all six
vis3_*.jsonl files are byte-identical to v4.0-VL's; the PNGs have the same sizes and differ in 0-3% of pixels
(glyph rendering, font and zlib versions). Missing fonts fall back to PIL's default font.
"""
from __future__ import annotations

import argparse, glob, json, os, random, sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vision_common as C  # noqa: E402

# ------------------------------------------------------------------ fonts / drawing
# Font files, as paths relative to a font directory (the Fedora/RHEL layout of /usr/share/fonts, where the
# v4.0-VL build ran). Each name is looked up under every --font-dir, first at that relative path, then
# anywhere below the directory by the same last-two-components or file name (Debian/Ubuntu layouts put them
# under truetype/ or opentype/). Order is kept: the pool size and order feed the rng.
A_NAMES = ("dejavu/DejaVuSans.ttf", "dejavu/DejaVuSans-Bold.ttf", "google-droid/DroidSans.ttf",
           "google-droid/DroidSans-Bold.ttf", "dejavu/DejaVuSerif.ttf", "dejavu/DejaVuSansCondensed.ttf")
MONO_NAMES = ("dejavu/DejaVuSansMono.ttf", "dejavu/DejaVuSansMono-Bold.ttf",
              "liberation-mono/LiberationMono-Regular.ttf", "liberation-mono/LiberationMono-Bold.ttf")
# probe family C: fonts never used in training (Cantarell, URW base35 OpenType)
C_DIRS = (("abattis-cantarell", "cantarell"), ("urw-base35",))
FONTS_A: list[str] = []
FONTS_MONO: list[str] = []
FONTS_C: list[str] = []


def _find(dirs, rel):
    for d in dirs:
        hit = glob.glob(os.path.join(d, rel))
        if hit:
            return hit
    base = os.path.basename(rel)
    for d in dirs:
        hit = sorted(glob.glob(os.path.join(d, "**", base), recursive=True))
        if hit:
            same = [h for h in hit if os.path.basename(os.path.dirname(h)) == os.path.dirname(rel)]
            return (same or hit)[:1]
    return []


def set_fonts(dirs):
    """Resolve the three font pools under `dirs` (a list of font directories)."""
    dirs = [str(d) for d in dirs]
    FONTS_A[:] = [f for p in A_NAMES for f in _find(dirs, p)]
    FONTS_MONO[:] = [f for p in MONO_NAMES for f in _find(dirs, p)]
    out = []
    for group in C_DIRS:
        found = []
        for name in group:
            for d in dirs:
                cands = glob.glob(os.path.join(d, name)) + glob.glob(os.path.join(d, "**", name), recursive=True)
                for cd in cands:
                    exts = ("*.otf", "*.ttf") if group[0] == "abattis-cantarell" else ("*.otf",)
                    found += [f for e in exts for f in glob.glob(os.path.join(cd, e))]
                if found:
                    break
            if found:
                break
        out += sorted(dict.fromkeys(found), key=os.path.basename)
    FONTS_C[:] = out
    return {"A": FONTS_A, "MONO": FONTS_MONO, "C": FONTS_C}


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default(size=size)


class Pen:
    def __init__(self, rng, fam, mono=False):
        pool = FONTS_C if fam == "C" and FONTS_C else (FONTS_MONO if mono else FONTS_A)
        pool = pool or FONTS_A or [None]
        self.face = rng.choice(pool)
        self.bold = rng.choice(pool)

    def f(self, size, bold=False):
        return _font(self.bold if bold else self.face, size)


def tw(d, text, font):
    b = d.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


def wrap(d, text, font, width):
    out, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if tw(d, t, font)[0] > width and cur:
            out.append(cur); cur = w
        else:
            cur = t
    if cur:
        out.append(cur)
    return out


def jitter(rng, c, k=18):
    return tuple(max(0, min(255, v + rng.randint(-k, k))) for v in c)


def money(x, cur="$"):
    return f"{cur}{x:,.2f}"


LETTERS = C.LETTERS


def choice(key, instr, texts, gi, rng):
    if len(set(texts)) != len(texts) or len(texts) < 2:
        return None                      # ambiguous option set: skip the question
    return C.choice_q(key, instr, texts, gi, rng)


def noul(key, instr, yes):
    return C.noul_q(key, instr), C.one_hot(2, 1 if yes else 0)


# =================================================================== 1. ui_state
FIELDS = [("Full name", "Maria Okafor"), ("Email", "t.nguyen@mailbox.org"), ("Phone", "+44 20 7946 0321"),
          ("Street address", "18 Harbour Road"), ("City", "Lisbon"), ("Postal code", "1100-148"),
          ("Card number", "5555 4444 3333 1111"), ("Expiry date", "11/29"), ("Security code", "731"),
          ("Company", "Northwind Ltd"), ("Username", "kestrel_88"), ("Password", "••••••••••"),
          ("Date of birth", "04/02/1991"), ("Promo code", "SPRING25"), ("Delivery notes", "Leave at reception")]
ACTIONS = [("Place order", "place the order"), ("Create account", "create the account"), ("Submit", "submit the form"),
           ("Book now", "complete the booking"), ("Save changes", "save the profile"), ("Continue to payment", "continue to payment"),
           ("Send request", "send the request"), ("Register", "register")]
BAD = {"Email": "t.nguyen@", "Phone": "12", "Postal code": "ABC", "Card number": "5555 4444", "Expiry date": "13/19",
       "Security code": "7", "Date of birth": "31/31/1991", "Username": "a", "Full name": "", "City": "",
       "Street address": "", "Company": "", "Password": "••", "Promo code": "XX", "Delivery notes": ""}


def gen_ui_state(rng, fam, ident):
    pen = Pen(rng, fam)
    n = rng.randint(3, 5)
    fields = rng.sample([f for f in FIELDS if BAD.get(f[0])], n) if rng.random() < .3 else rng.sample(FIELDS, n)
    action, goal = rng.choice(ACTIONS)
    # per-field state
    st = []
    n_block = rng.choices([0, 1, 2], [0.45, 0.45, 0.10])[0]
    blockers = set(rng.sample(range(n), min(n_block, n)))
    for i, (lab, val) in enumerate(fields):
        req = True if i in blockers else rng.random() < .7
        if i in blockers:
            s = "invalid" if (BAD.get(lab) and rng.random() < .4) else "empty"
        else:
            s = rng.choices(["filled", "empty", "disabled"], [0.7, 0.15, 0.15])[0]
            if s == "empty":
                req = False   # an empty optional field does not block
            if s == "disabled":
                req = False
        st.append(dict(label=lab, val=val, req=req, s=s))
    blocking = [x for x in st if x["req"] and x["s"] in ("empty", "invalid")]
    W, H = (900, 170 + 95 * n) if fam != "B" else (1000, 200 + 60 * n)
    if fam == "C":
        W, H = 560, 230 + 105 * n
    dark = fam == "C" and rng.random() < .6
    bg = jitter(rng, (24, 26, 32) if dark else (245, 246, 248))
    fg = (235, 235, 235) if dark else (25, 25, 30)
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    title = rng.choice(["Checkout", "Your details", "Sign up", "Booking", "Account", "Shipping", "Profile"])
    if fam == "B":
        d.rectangle((0, 0, W, 70), fill=jitter(rng, (30, 41, 59)))
        d.text((30, 18), title, font=pen.f(30, True), fill="white")
    else:
        d.text((32, 22), title, font=pen.f(32, True), fill=fg)
    y = 90 if fam != "B" else 100
    acc = jitter(rng, rng.choice([(37, 99, 235), (22, 163, 74), (124, 58, 237), (234, 88, 12), (13, 148, 136)]))
    for x in st:
        lab = x["label"] + (" *" if x["req"] and fam != "C" else "") + (" (required)" if x["req"] and fam == "C" else "")
        if fam == "B":
            d.text((30, y + 12), lab, font=pen.f(20), fill=fg)
            box = (300, y, 940, y + 46)
        else:
            d.text((32, y), lab, font=pen.f(20), fill=fg)
            box = (32, y + 28, W - 40, y + 72) if fam == "A" else (32, y + 30, W - 32, y + 80)
        if x["s"] == "disabled":
            d.rectangle(box, fill=(70, 72, 80) if dark else (225, 226, 230), outline=(160, 160, 165), width=2)
            d.text((box[0] + 12, box[1] + 10), x["val"] if rng.random() < .5 else "—", font=pen.f(19), fill=(140, 140, 145))
        else:
            fillc = (40, 42, 50) if dark else "white"
            oc = (220, 38, 38) if x["s"] == "invalid" else (150, 150, 155)
            d.rectangle(box, fill=fillc, outline=oc, width=3 if x["s"] == "invalid" else 2)
            if x["s"] == "filled":
                d.text((box[0] + 12, box[1] + 10), x["val"], font=pen.f(20), fill=fg)
            elif x["s"] == "invalid":
                d.text((box[0] + 12, box[1] + 10), BAD[x["label"]] or "?", font=pen.f(20), fill=fg)
                if fam != "B":
                    d.text((box[0], box[3] + 2), rng.choice(["Invalid value", "Please check this field", "Format not recognised"]),
                           font=pen.f(15), fill=(220, 38, 38))
            elif fam == "C" and x["s"] == "empty":
                d.text((box[0] + 12, box[1] + 12), "Enter " + x["label"].lower(), font=pen.f(18), fill=(120, 120, 128))
        y += 95 if fam != "B" else 60
    # buttons: the primary button ALWAYS looks enabled (the model must check the fields)
    bw = tw(d, action, pen.f(22, True))[0] + 60
    bx = 32 if fam != "B" else 300
    by = y + 15
    d.rounded_rectangle((bx, by, bx + bw, by + 54), radius=8 if fam != "C" else 26, fill=acc)
    d.text((bx + 30, by + 13), action, font=pen.f(22, True), fill="white")
    sec = rng.choice(["Cancel", "Back", "Save draft"])
    d.rounded_rectangle((bx + bw + 24, by, bx + bw + 24 + tw(d, sec, pen.f(22))[0] + 50, by + 54), radius=8,
                        outline=(140, 140, 145), width=2)
    d.text((bx + bw + 49, by + 13), sec, font=pen.f(22), fill=fg)
    im = im.crop((0, 0, W, min(H, by + 90)))
    qs = []
    # Q1 next action
    if blocking:
        b0 = blocking[0]
        gold = f"Fill in the {b0['label']} field" if b0["s"] == "empty" else f"Correct the {b0['label']} field"
        others = [x for x in st if x is not b0]
        ds = [f"Click \"{action}\""]
        o = rng.choice(others) if others else None
        if o:
            ds.append(f"Correct the {o['label']} field" if o["s"] == "filled" else f"Fill in the {o['label']} field")
        ds.append(f"Click \"{sec}\"")
        opts = [gold] + ds[:3]
    else:
        f1 = rng.choice(st)
        opts = [f"Click \"{action}\"", f"Fill in the {f1['label']} field" if f1["s"] != "empty" else f"Correct the {f1['label']} field",
                f"Click \"{sec}\"", f"Enable the {rng.choice(st)['label']} field"]
        gold = opts[0]
    if len(set(opts)) == len(opts):
        qs.append(choice("next", f"The user wants to {goal}. Required fields are marked. What is the correct next step?",
                         opts, 0, rng))
    # Q2 can submit
    qs.append(noul("ready", f"Can the user {goal} right now (every required field filled in and valid)?", not blocking))
    # Q3 field filled / disabled
    x = rng.choice(st)
    if rng.random() < .5 and x["s"] != "disabled":
        qs.append(noul("filled", f"Does the {x['label']} field contain a value?", x["s"] in ("filled", "invalid")))
    else:
        qs.append(noul("disabled", f"Is the {x['label']} field disabled (greyed out)?", x["s"] == "disabled"))
    # Q4 which field blocks
    if len(blocking) <= 1 and len(st) >= 3:
        if blocking:
            g = blocking[0]["label"]
            pool = [y_["label"] for y_ in st if y_["label"] != g]
            opts = [g] + rng.sample(pool, min(2, len(pool))) + ["Nothing, the form is complete"]
        else:
            opts = ["Nothing, the form is complete"] + rng.sample([y_["label"] for y_ in st], 3)
        qs.append(choice("blocker", f"Which field stops the user from being able to {goal}?", opts, 0, rng))
    return [im], "<image>", qs


# =================================================================== 2. change_detect
ELEMS_A = ["the header bar", "the navigation menu", "the main heading", "the hero image", "the primary button", "the footer"]
ELEMS_B = ["the top bar", "the profile card", "the list of items", "the search field", "the tab bar", "the floating button"]
ELEMS_C = ["the title strip", "the revenue card", "the orders card", "the chart panel", "the filter chips", "the export button"]
CHANGES = ["none", "moved", "missing", "overlap", "cutoff", "color", "text"]
CH_TEXT = {"none": "Nothing changed", "moved": "An element moved to a different position", "missing": "An element disappeared",
           "overlap": "An element now overlaps another element", "cutoff": "An element is cut off at the edge of the screen",
           "color": "An element changed color", "text": "The text of an element changed"}


def _layout(fam, rng):
    """-> (W, H, list of element dicts {name, box, kind, color, text})"""
    if fam == "A":
        W, H = 720, 540
        els = [dict(name=ELEMS_A[0], box=[0, 0, W, 60], kind="bar", text=rng.choice(["Orbit Labs", "Pinecrest", "Northdesk"])),
               dict(name=ELEMS_A[1], box=[20, 80, 180, 440], kind="menu", text=None),
               dict(name=ELEMS_A[2], box=[210, 85, 600, 125], kind="text", text=rng.choice(["Release notes", "Welcome back", "Pricing plans", "Team settings"])),
               dict(name=ELEMS_A[3], box=[210, 145, 500, 320], kind="img", text=None),
               dict(name=ELEMS_A[4], box=[210, 350, 380, 400], kind="btn", text=rng.choice(["Get started", "Upgrade", "Download"])),
               dict(name=ELEMS_A[5], box=[0, 480, W, 540], kind="bar2", text="© 2026")]
    elif fam == "B":
        W, H = 400, 720
        els = [dict(name=ELEMS_B[0], box=[0, 0, W, 64], kind="bar", text=rng.choice(["Inbox", "Wallet", "Trips", "Tasks"])),
               dict(name=ELEMS_B[3], box=[16, 80, 384, 124], kind="input", text="Search"),
               dict(name=ELEMS_B[1], box=[16, 140, 384, 260], kind="card", text=rng.choice(["Ana Ruiz", "Tomas Berg", "Lina Haddad"])),
               dict(name=ELEMS_B[2], box=[16, 280, 384, 560], kind="list", text=None),
               dict(name=ELEMS_B[5], box=[300, 580, 370, 640], kind="fab", text="+"),
               dict(name=ELEMS_B[4], box=[0, 656, W, 720], kind="bar2", text=None)]
    else:  # C: dark dashboard grid (probe)
        W, H = 760, 520
        els = [dict(name=ELEMS_C[0], box=[0, 0, W, 56], kind="bar", text=rng.choice(["Ops overview", "Store metrics", "Q3 dashboard"])),
               dict(name=ELEMS_C[4], box=[20, 70, 460, 104], kind="chips", text=None),
               dict(name=ELEMS_C[1], box=[20, 120, 360, 250], kind="card", text="Revenue"),
               dict(name=ELEMS_C[2], box=[390, 120, 740, 250], kind="card", text="Orders"),
               dict(name=ELEMS_C[3], box=[20, 270, 740, 470], kind="chart", text=None),
               dict(name=ELEMS_C[5], box=[600, 70, 740, 104], kind="btn", text="Export")]
    pal = [(37, 99, 235), (22, 163, 74), (124, 58, 237), (234, 88, 12), (219, 39, 119), (13, 148, 136), (202, 138, 4)]
    for e in els:
        e["color"] = jitter(rng, rng.choice(pal), 10)
    return W, H, els


def _draw_layout(W, H, els, fam, pen, seed):
    r = random.Random(seed)
    dark = fam == "C"
    im = Image.new("RGB", (W, H), (22, 24, 30) if dark else (255, 255, 255))
    d = ImageDraw.Draw(im)
    for e in els:
        x0, y0, x1, y1 = e["box"]; k = e["kind"]; c = e["color"]
        fg = (235, 235, 235) if dark else (30, 30, 30)
        if k in ("bar", "bar2"):
            d.rectangle(e["box"], fill=c)
            if e["text"]:
                d.text((x0 + 18, y0 + 16), e["text"], font=pen.f(24, True), fill="white")
        elif k == "menu":
            d.rectangle(e["box"], fill=(243, 244, 246))
            for i in range(5):
                d.rectangle((x0 + 14, y0 + 20 + 50 * i, x0 + 14 + r.randint(70, 130), y0 + 34 + 50 * i), fill=c)
        elif k == "text":
            d.text((x0, y0), e["text"], font=pen.f(30, True), fill=c)
        elif k == "img":
            d.rectangle(e["box"], fill=c)
            d.line((x0, y1, (x0 + x1) // 2, y0 + 40, x1, y1), fill="white", width=4)
            d.ellipse((x1 - 70, y0 + 20, x1 - 30, y0 + 60), fill="white")
        elif k == "btn":
            d.rounded_rectangle(e["box"], radius=10, fill=c)
            d.text((x0 + 16, y0 + 12), e["text"], font=pen.f(20, True), fill="white")
        elif k == "input":
            d.rounded_rectangle(e["box"], radius=20, outline=c, width=3, fill=(248, 248, 250))
            d.text((x0 + 18, y0 + 10), e["text"], font=pen.f(18), fill=(120, 120, 120))
        elif k == "card":
            d.rounded_rectangle(e["box"], radius=12, fill=(40, 44, 54) if dark else (246, 246, 250), outline=c, width=3)
            d.text((x0 + 16, y0 + 14), e["text"], font=pen.f(22, True), fill=fg)
            d.text((x0 + 16, y0 + 54), f"{r.randint(10, 99)},{r.randint(100, 999)}", font=pen.f(28, True), fill=c)
        elif k == "list":
            for i in range(5):
                yy = y0 + 8 + 54 * i
                d.ellipse((x0 + 4, yy, x0 + 40, yy + 36), fill=c)
                d.rectangle((x0 + 54, yy + 6, x0 + 54 + r.randint(120, 260), yy + 18), fill=(200, 200, 205))
                d.rectangle((x0 + 54, yy + 24, x0 + 54 + r.randint(60, 180), yy + 32), fill=(225, 225, 230))
        elif k == "fab":
            d.ellipse(e["box"], fill=c)
            d.text((x0 + 22, y0 + 10), "+", font=pen.f(34, True), fill="white")
        elif k == "chips":
            xx = x0
            for t in ("Today", "7 days", "30 days", "Custom"):
                wdt = tw(d, t, pen.f(16))[0] + 24
                d.rounded_rectangle((xx, y0, xx + wdt, y1), radius=14, outline=c, width=2)
                d.text((xx + 12, y0 + 7), t, font=pen.f(16), fill=fg)
                xx += wdt + 10
        elif k == "chart":
            d.rectangle(e["box"], outline=(80, 84, 96), width=2)
            pts = [(x0 + 20 + i * (x1 - x0 - 40) / 11, y1 - 20 - r.random() * (y1 - y0 - 60)) for i in range(12)]
            d.line(pts, fill=c, width=4)
    return im


def gen_change_detect(rng, fam, ident):
    pen = Pen(rng, fam)
    W, H, els = _layout(fam, rng)
    ch = rng.choices(CHANGES, [0.30, 0.10, 0.15, 0.15, 0.15, 0.075, 0.075])[0]
    seed = rng.randrange(1 << 30)
    before = _draw_layout(W, H, els, fam, pen, seed)
    after_els = [dict(e, box=list(e["box"])) for e in els]
    tgt = None
    if ch != "none":
        cand = [i for i, e in enumerate(after_els) if not (ch == "text" and not e["text"])]
        ti = rng.choice(cand)
        tgt = after_els[ti]
        x0, y0, x1, y1 = tgt["box"]
        if ch == "moved":
            dx = rng.choice([-1, 1]) * rng.randint(60, 140); dy = rng.choice([-1, 1]) * rng.randint(40, 100)
            nb = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
            if nb[0] < 0 or nb[2] > W or nb[1] < 0 or nb[3] > H:
                nb = [x0 - dx, y0 - dy, x1 - dx, y1 - dy]
            nb = [max(0, min(W - (x1 - x0), nb[0])), max(0, min(H - (y1 - y0), nb[1]))]
            nb = [nb[0], nb[1], nb[0] + x1 - x0, nb[1] + y1 - y0]
            if nb == tgt["box"]:
                nb = [x0, max(0, y0 - 50), x1, max(0, y0 - 50) + y1 - y0]
            tgt["box"] = nb
        elif ch == "missing":
            after_els.pop(ti)
        elif ch == "overlap":
            other = rng.choice([e for e in after_els if e is not tgt])
            ox0, oy0, ox1, oy1 = other["box"]
            w, h = x1 - x0, y1 - y0
            nx = (ox0 + ox1) // 2 - w // 3; ny = (oy0 + oy1) // 2 - h // 3
            tgt["box"] = [nx, ny, nx + w, ny + h]
            after_els.remove(tgt); after_els.append(tgt)   # drawn on top
        elif ch == "cutoff":
            w = x1 - x0
            shift = W - x0 - w // 2
            tgt["box"] = [x0 + shift, y0, x1 + shift, y1]
        elif ch == "color":
            tgt["color"] = tuple(255 - v for v in tgt["color"])
        elif ch == "text":
            tgt["text"] = rng.choice(["Coming soon", "Error 404", "Lorem ipsum", "TODO", "Untitled"])
    after = _draw_layout(W, H, after_els, fam, pen, seed)
    names = [e["name"] for e in els]
    qs = []
    broke = ch in ("overlap", "cutoff", "missing")
    qs.append(noul("broken", "Compared with the Before screenshot, is the After layout broken "
                             "(an element missing, overlapping another element, or cut off at the edge)?", broke))
    qs.append(noul("same", "Do the two screenshots look the same?", ch == "none"))
    opts = [CH_TEXT[ch]] + rng.sample([CH_TEXT[c] for c in CHANGES if c != ch], 3)
    qs.append(choice("what", "What changed from the Before screenshot to the After screenshot?", opts, 0, rng))
    if tgt is not None:
        g = tgt["name"]
        opts = [g] + rng.sample([n for n in names if n != g], 3)
        qs.append(choice("which", "Which element changed between the two screenshots?", opts, 0, rng))
    return [before, after], "Before: <image>\nAfter: <image>", qs


# =================================================================== 3. line_rule
CATALOG = [  # name, category, price range
    ("Chianti, bottle", "alcohol", (18, 42)), ("Pale ale pint", "alcohol", (5, 9)), ("Gin and tonic", "alcohol", (8, 14)),
    ("Prosecco glass", "alcohol", (7, 12)), ("Whisky, 50 ml", "alcohol", (9, 22)), ("Lager 6-pack", "alcohol", (8, 15)),
    ("Negroni", "alcohol", (10, 15)), ("Cider, half pint", "alcohol", (3, 6)),
    ("Club sandwich", "food", (7, 14)), ("Tomato soup", "food", (4, 8)), ("Caesar salad", "food", (8, 13)),
    ("Espresso", "food", (2, 4)), ("Orange juice", "food", (3, 5)), ("Pasta al pesto", "food", (11, 18)),
    ("Still water 1L", "food", (2, 4)), ("Fruit bowl", "food", (4, 7)), ("Chicken wrap", "food", (6, 10)),
    ("Taxi to airport", "travel", (25, 70)), ("Train ticket", "travel", (12, 95)), ("Parking, 4h", "travel", (8, 22)),
    ("Hotel night", "travel", (80, 240)), ("USB-C cable", "electronics", (9, 25)), ("Wireless mouse", "electronics", (18, 45)),
    ("Headphones", "electronics", (40, 180)), ("Printer paper", "office", (5, 12)), ("Notebooks x3", "office", (6, 15)),
    ("Whiteboard pens", "office", (4, 10)), ("Cinema tickets", "entertainment", (18, 35)), ("Concert ticket", "entertainment", (45, 120)),
    ("Gift card", "gift", (25, 100)), ("Flowers", "gift", (20, 60)), ("Cigarettes", "tobacco", (9, 14)), ("Cigar", "tobacco", (8, 25)),
]


def gen_line_rule(rng, fam, ident):
    pen = Pen(rng, fam, mono=(fam == "A"))
    kind = rng.choice(["alcohol", "maxprice", "category", "maxqty"])
    n = rng.randint(4, 7)
    violate = rng.random() < .5
    clean_pool = [c for c in CATALOG]
    if kind == "alcohol":
        rule = rng.choice(["Alcoholic drinks cannot be reimbursed.", "Company policy: no alcohol on expense claims.",
                           "Alcohol is not an allowed expense."])
        bad = lambda it: it["cat"] == "alcohol"
    elif kind == "category":
        cat = rng.choice(["tobacco", "entertainment", "gift", "electronics"])
        rule = {"tobacco": "Tobacco products may not be claimed.", "entertainment": "Entertainment (cinema, concerts) is not reimbursable.",
                "gift": "Gifts and gift cards are not allowed on this card.", "electronics": "Electronics must be ordered through IT, not expensed."}[cat]
        bad = lambda it, cat=cat: it["cat"] == cat
    elif kind == "maxprice":
        lim = rng.choice([25, 30, 40, 50, 75, 100])
        rule = f"No single line may exceed {money(lim)}."
        bad = lambda it, lim=lim: it["amt"] > lim
    else:
        lim = rng.choice([2, 3, 4])
        rule = f"At most {lim} units of any one item may be ordered."
        bad = lambda it, lim=lim: it["qty"] > lim
    items = []
    tries = 0
    while len(items) < n and tries < 500:
        tries += 1
        name, cat, (lo, hi) = rng.choice(clean_pool)
        qty = rng.choices([1, 2, 3, 4, 5, 6], [5, 3, 2, 1, 1, 1])[0] if kind == "maxqty" else rng.choice([1, 1, 1, 2])
        unit = round(rng.uniform(lo, hi), 2)
        it = dict(name=name, cat=cat, qty=qty, unit=unit, amt=round(qty * unit, 2))
        if any(i_["name"] == name for i_ in items) or bad(it):
            continue
        items.append(it)
    if violate:
        for _ in range(500):
            name, cat, (lo, hi) = rng.choice(CATALOG)
            qty = rng.randint(3, 8) if kind == "maxqty" else rng.choice([1, 1, 2])
            unit = round(rng.uniform(lo, hi), 2)
            if kind == "maxprice":
                unit = round(rng.uniform(lim + 1, lim * 2.5) / qty, 2)
            it = dict(name=name, cat=cat, qty=qty, unit=unit, amt=round(qty * unit, 2))
            if bad(it) and not any(i_["name"] == name for i_ in items):
                items.insert(rng.randrange(len(items) + 1), it)
                break
        else:
            violate = False
    viol = [it for it in items if bad(it)]
    assert len(viol) == (1 if violate else 0), (kind, viol)
    # render
    if fam == "A":       # thermal receipt
        W = 500; im = Image.new("RGB", (W, 200 + 44 * len(items)), jitter(rng, (250, 250, 244), 5)); d = ImageDraw.Draw(im)
        shop = rng.choice(["CORNER BISTRO", "MARKET HALL 7", "RIVERSIDE CAFE", "STATION STORES", "HILLTOP DELI"])
        d.text((W // 2 - tw(d, shop, pen.f(24, True))[0] // 2, 20), shop, font=pen.f(24, True), fill="black")
        y = 80
        for it in items:
            q = f"{it['qty']}x " if it["qty"] > 1 or kind == "maxqty" else ""
            d.text((24, y), q + it["name"], font=pen.f(20), fill="black")
            s = money(it["amt"]); d.text((W - 24 - tw(d, s, pen.f(20))[0], y), s, font=pen.f(20), fill="black"); y += 44
        d.line((24, y + 6, W - 24, y + 6), fill="black", width=2)
        tot = money(sum(i_["amt"] for i_ in items)); d.text((24, y + 20), "TOTAL", font=pen.f(22, True), fill="black")
        d.text((W - 24 - tw(d, tot, pen.f(22, True))[0], y + 20), tot, font=pen.f(22, True), fill="black")
    elif fam == "B":     # expense report table
        W = 900; im = Image.new("RGB", (W, 170 + 46 * len(items)), "white"); d = ImageDraw.Draw(im)
        d.text((30, 20), "Expense report", font=pen.f(28, True), fill=(20, 20, 20))
        cols = [30, 120, 480, 640, 720]; hd = ["Line", "Item", "Category", "Qty", "Amount"]
        d.rectangle((20, 70, W - 20, 110), fill=jitter(rng, (226, 232, 240)))
        for x, h in zip(cols, hd):
            d.text((x, 80), h, font=pen.f(18, True), fill=(30, 30, 30))
        y = 118
        for k, it in enumerate(items):
            if k % 2:
                d.rectangle((20, y - 4, W - 20, y + 38), fill=(248, 250, 252))
            for x, v in zip(cols, [str(k + 1), it["name"], it["cat"], str(it["qty"]), money(it["amt"])]):
                d.text((x, y + 6), v, font=pen.f(18), fill=(30, 30, 30))
            y += 46
    else:                # C: order confirmation cards (probe)
        W = 620; im = Image.new("RGB", (W, 150 + 70 * len(items)), jitter(rng, (239, 246, 255), 6)); d = ImageDraw.Draw(im)
        d.text((28, 24), f"Order #{rng.randint(10000, 99999)} confirmed", font=pen.f(26, True), fill=(15, 23, 42))
        y = 80
        for it in items:
            d.rounded_rectangle((20, y, W - 20, y + 60), radius=14, fill="white", outline=(203, 213, 225), width=2)
            d.text((40, y + 8), it["name"], font=pen.f(20, True), fill=(15, 23, 42))
            d.text((40, y + 34), f"qty {it['qty']} · {money(it['unit'])} each", font=pen.f(15), fill=(100, 116, 139))
            s = money(it["amt"]); d.text((W - 40 - tw(d, s, pen.f(20, True))[0], y + 18), s, font=pen.f(20, True), fill=(15, 23, 42))
            y += 70
    state = f"Policy: {rule}\n<image>"
    qs = [noul("violates", "Does anything in this list break the policy?", violate)]
    names = [it["name"] for it in items]
    if violate:
        g = viol[0]["name"]
        opts = [g] + rng.sample([x for x in names if x != g], min(2, len(names) - 1)) + ["No line breaks the policy"]
    else:
        opts = ["No line breaks the policy"] + rng.sample(names, 3)
    qs.append(choice("line", "Which line breaks the policy?", opts, 0, rng))
    return [im], state, qs


# =================================================================== 4. grid_games (tic-tac-toe, solver labels)
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
CELL_AB = ["top-left", "top-middle", "top-right", "middle-left", "center", "middle-right", "bottom-left", "bottom-middle", "bottom-right"]
CELL_C = [f"row {r}, column {c}" for r in (1, 2, 3) for c in (1, 2, 3)]


def winner(b):
    for a, b_, c in LINES:
        if b[a] != "." and b[a] == b[b_] == b[c]:
            return b[a]
    return None


def win_moves(b, p):
    out = []
    for i in range(9):
        if b[i] == ".":
            t = b[:i] + p + b[i + 1:]
            if winner(t) == p:
                out.append(i)
    return out


def rand_board(rng, n_moves):
    b = "........."
    for k in range(n_moves):
        p = "X" if k % 2 == 0 else "O"
        empt = [i for i in range(9) if b[i] == "."]
        i = rng.choice(empt)
        b = b[:i] + p + b[i + 1:]
        if winner(b):
            return b, True
    return b, False


def draw_board(b, fam, rng, pen):
    if fam == "C":
        S = 540; im = Image.new("RGB", (S, S), (17, 24, 39)); d = ImageDraw.Draw(im)
        cs = 160
        for i, c in enumerate(b):
            r, col = divmod(i, 3); x, y = 30 + col * (cs + 10), 30 + r * (cs + 10)
            d.rounded_rectangle((x, y, x + cs, y + cs), radius=18, fill=(31, 41, 55))
            if c != ".":
                f = pen.f(110, True); w, h = tw(d, c, f)
                d.text((x + (cs - w) / 2, y + (cs - h) / 2 - 18), c, font=f, fill=(250, 204, 21) if c == "X" else (56, 189, 248))
        return im
    S = 600 if fam == "A" else 640
    im = Image.new("RGB", (S, S), jitter(rng, (255, 255, 255), 6)); d = ImageDraw.Draw(im)
    off = 20 if fam == "A" else 60; cs = (S - off - 20) // 3
    lw = rng.randint(4, 8); lc = jitter(rng, (30, 30, 30), 20)
    for k in (1, 2):
        d.line((off + k * cs, off, off + k * cs, off + 3 * cs), fill=lc, width=lw)
        d.line((off, off + k * cs, off + 3 * cs, off + k * cs), fill=lc, width=lw)
    if fam == "B":
        for k in range(3):
            d.text((off + k * cs + cs // 2 - 8, 18), "ABC"[k], font=pen.f(26, True), fill=(90, 90, 90))
            d.text((20, off + k * cs + cs // 2 - 14), "123"[k], font=pen.f(26, True), fill=(90, 90, 90))
    xc = jitter(rng, rng.choice([(220, 38, 38), (30, 30, 30), (234, 88, 12)]), 10)
    oc = jitter(rng, rng.choice([(37, 99, 235), (22, 163, 74), (30, 30, 30)]), 10)
    m = cs // 4
    for i, c in enumerate(b):
        r, col = divmod(i, 3); cx, cy = off + col * cs + cs // 2, off + r * cs + cs // 2
        if c == "X":
            d.line((cx - m, cy - m, cx + m, cy + m), fill=xc, width=lw + 4); d.line((cx - m, cy + m, cx + m, cy - m), fill=xc, width=lw + 4)
        elif c == "O":
            d.ellipse((cx - m, cy - m, cx + m, cy + m), outline=oc, width=lw + 4)
    return im


def gen_grid_games(rng, fam, ident):
    pen = Pen(rng, fam)
    names = CELL_C if fam == "C" else CELL_AB
    if fam == "B" and rng.random() < .5:
        names = [f"{'ABC'[c]}{r + 1}" for r in range(3) for c in range(3)]
    qs = []
    want_over = rng.random() < .5         # half the boards already have three in a row
    for _ in range(2000):
        b, over = rand_board(rng, rng.randint(5, 9) if want_over else rng.randint(2, 7))
        if over == want_over:
            break
    w = winner(b)
    nx, no = b.count("X"), b.count("O")
    turn = "X" if nx == no else "O"
    other = "O" if turn == "X" else "X"
    state = "Tic-tac-toe. X moves first; players alternate.\n<image>"
    qs.append(noul("won", "Has either player already completed three in a row?", w is not None))
    if w is not None:
        qs.append(choice("who", "Who has won the game?", [w, "O" if w == "X" else "X", "Nobody yet"], 0, rng))
    else:
        if "." in b:
            qs.append(choice("turn", "Whose turn is it?", [turn, other], 0, rng))
        wm, bm = win_moves(b, turn), win_moves(b, other)
        empt = [i for i in range(9) if b[i] == "."]
        if len(wm) == 1 and len(empt) >= 4:
            g = wm[0]
            opts = [names[g]] + [names[i] for i in rng.sample([i for i in empt if i != g], 3)]
            qs.append(choice("win", f"It is {turn}'s turn. Which cell wins the game immediately?", opts, 0, rng))
        elif not wm and len(bm) == 1 and len(empt) >= 4:
            g = bm[0]
            opts = [names[g]] + [names[i] for i in rng.sample([i for i in empt if i != g], 3)]
            qs.append(choice("block", f"It is {turn}'s turn and {turn} cannot win this move. "
                                      f"Which cell must {turn} take to stop {other} from winning next move?", opts, 0, rng))
        if empt:
            qs.append(noul("canwin", f"It is {turn}'s turn. Can {turn} win with a single move?", bool(wm)))
    return [draw_board(b, fam, rng, pen)], state, qs


# =================================================================== 5. severity (rule-computed levels)
SEV_SCALE = ("Severity scale: 0 info (no action needed), 1 low (act when convenient), 2 medium (degraded or at risk; "
             "handle today), 3 high (users affected now; act within the hour), 4 critical (outage, security breach or "
             "data loss; act immediately).")
SEV_LEVELS = ["info", "low", "medium", "high", "critical"]
MSGS = {
    0: [("Backup completed", "Nightly backup of 214 GB finished in 38 minutes."), ("Report ready", "Your monthly usage report is ready to download."),
        ("Settings saved", "Notification preferences were updated."), ("Welcome", "Three new team members joined this week.")],
    1: [("New version", "Version 5.2 is available. Install at your next restart."), ("Tip", "You can archive old projects to save space."),
        ("Password age", "Your password is 80 days old. Consider changing it."), ("Minor warning", "2 images in the gallery have no alt text.")],
    2: [("Certificate expiring", "The TLS certificate for api.example.net expires in 9 days."), ("Storage at 86%", "Volume data-02 is 86% full at the current growth rate."),
        ("Sync delayed", "Replication is 25 minutes behind. The system is retrying."), ("Elevated latency", "p95 latency is 1.8 s, above the 1.2 s target, for 40 minutes."),
        ("Job retries", "The invoice export job failed twice and will retry tonight."), ("Quota warning", "API usage reached 90% of the monthly quota.")],
    3: [("Payments failing", "Card payments are failing for about 12% of customers."), ("Login errors", "Users in the EU region cannot sign in to the mobile app."),
        ("Disk almost full", "Volume logs-01 is 98% full; writes will fail soon."), ("Checkout degraded", "Checkout takes over 20 s for most shoppers.")],
    4: [("Database down", "The primary database is unreachable. All requests are failing."), ("Data loss", "A migration deleted rows in the orders table. Restore needed."),
        ("Security breach", "Unauthorized admin logins detected from an unknown IP range."), ("Full outage", "The website returns 503 for every visitor.")],
}
SEV_COL = [(59, 130, 246), (20, 184, 166), (245, 158, 11), (234, 88, 12), (220, 38, 38)]
ICON = ["i", "-", "!", "!!", "X"]


def gen_severity(rng, fam, ident):
    pen = Pen(rng, fam, mono=(fam == "C"))
    lvl = rng.choices(range(5), [0.18, 0.18, 0.28, 0.18, 0.18])[0]
    title, body = rng.choice(MSGS[lvl])
    styled = rng.random() < .65          # else neutral grey styling: the wording alone carries the level
    col = SEV_COL[lvl] if styled else (107, 114, 128)
    if fam == "A":       # modal dialog
        W, H = 760, 380
        im = Image.new("RGB", (W, H), jitter(rng, (226, 228, 233), 8)); d = ImageDraw.Draw(im)
        d.rounded_rectangle((60, 40, W - 60, H - 40), radius=14, fill="white", outline=(170, 170, 175), width=2)
        d.ellipse((90, 70, 150, 130), fill=col); d.text((110, 78), ICON[lvl] if styled else "•", font=pen.f(34, True), fill="white")
        d.text((170, 82), title, font=pen.f(26, True), fill=(20, 20, 20))
        for i, l in enumerate(wrap(d, body, pen.f(21), W - 240)):
            d.text((170, 140 + 32 * i), l, font=pen.f(21), fill=(50, 50, 55))
        d.rounded_rectangle((W - 220, H - 110, W - 100, H - 66), radius=8, fill=(55, 65, 81)); d.text((W - 180, H - 102), "OK", font=pen.f(20), fill="white")
        extra = []
    elif fam == "B":     # notification stack: most severe of 1-3 toasts
        W = 700
        k = rng.randint(1, 3)
        lv = [lvl] + [rng.randint(0, lvl) for _ in range(k - 1)]
        rng.shuffle(lv)
        im = Image.new("RGB", (W, 60 + 120 * k), jitter(rng, (243, 244, 246), 6)); d = ImageDraw.Draw(im)
        d.text((24, 16), "Notifications", font=pen.f(22, True), fill=(30, 30, 30))
        y = 56
        for l_ in lv:
            t_, b_ = (title, body) if l_ == lvl else rng.choice(MSGS[l_])
            c_ = SEV_COL[l_] if styled else (107, 114, 128)
            d.rounded_rectangle((20, y, W - 20, y + 108), radius=10, fill="white")
            d.rectangle((20, y, 30, y + 108), fill=c_)
            d.text((46, y + 10), t_, font=pen.f(21, True), fill=(20, 20, 20))
            for i, l in enumerate(wrap(d, b_, pen.f(18), W - 90)[:2]):
                d.text((46, y + 44 + 26 * i), l, font=pen.f(18), fill=(60, 60, 65))
            y += 120
        extra = lv
    else:                # C: log panel (probe)
        W = 900
        lines = [("INFO", rng.choice(MSGS[0])[1]) for _ in range(rng.randint(2, 4))]
        tag = ["INFO", "NOTICE", "WARN", "ERROR", "FATAL"][lvl]
        lines.insert(rng.randrange(len(lines) + 1), (tag, f"{title}: {body}"))
        im = Image.new("RGB", (W, 40 + 60 * len(lines)), (12, 12, 14)); d = ImageDraw.Draw(im)
        y = 20
        for t_, m_ in lines:
            ts = f"12:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"
            c_ = {"INFO": (160, 160, 165), "NOTICE": (94, 234, 212), "WARN": (250, 204, 21), "ERROR": (251, 146, 60), "FATAL": (248, 113, 113)}[t_] if styled else (200, 200, 200)
            wl = wrap(d, f"{ts} [{t_}] {m_}", pen.f(16), W - 40)
            for i, l in enumerate(wl[:2]):
                d.text((20, y + 22 * i), l, font=pen.f(16), fill=c_)
            y += 60
        extra = []
    what = "this message" if fam == "A" else "the most severe message shown"
    crit = {str(i): f"{i}: {SEV_LEVELS[i]}" for i in range(5)}
    qs = [(C.Question("level", "score", f"Rate the severity of {what} on the scale above.", tuple(str(i) for i in range(5)), crit),
           C.one_hot(5, lvl))]
    resp = ["No action needed", "Handle when convenient", "Handle today", "Act within the hour", "Act immediately (page on-call)"]
    opts = [resp[lvl]] + rng.sample([r for i, r in enumerate(resp) if i != lvl], 3)
    qs.append(choice("resp", f"Which response fits {what}?", opts, 0, rng))
    qs.append(noul("today", f"Does {what} need someone to act today or sooner?", lvl >= 2))
    return [im], SEV_SCALE + "\n<image>", qs


# =================================================================== 6. receipt_math
GOODS = ["Oat milk 1L", "Sourdough loaf", "Olive oil 500ml", "Basmati rice 2kg", "Coffee beans", "Greek yogurt", "Dish soap",
         "AA batteries x4", "Notebook A5", "Tea bags x80", "Cherry tomatoes", "Cheddar 400g", "Paper towels", "Hand cream",
         "Desk lamp", "HDMI cable", "Printer ink", "Stapler", "Consulting hour", "Design review", "Hosting (monthly)", "Support plan"]


def fmt(x, cur):
    return f"{cur}{x:,.2f}"


def gen_receipt_math(rng, fam, ident):
    pen = Pen(rng, fam, mono=(fam == "A"))
    cur = "€" if fam == "C" else "$"
    n = rng.randint(3, 6)
    names = rng.sample(GOODS, n)
    lines = []
    for nm in names:
        q = rng.choice([1, 1, 1, 2, 3, 4])
        u = round(rng.uniform(1.2, 60 if fam != "B" else 180), 2)
        lines.append(dict(name=nm, qty=q, unit=u, amt=round(q * u, 2)))
    tax_rate = rng.choice([0.0, 0.05, 0.07, 0.08, 0.10, 0.2])
    err = rng.choices(["none", "total", "line"], [0.5, 0.3, 0.2])[0]
    bad_line = None
    printed = [dict(l_) for l_ in lines]
    if err == "line":
        bad_line = rng.randrange(n)
        delta = rng.choice([1.0, 2.0, 10.0, -1.0, 0.9, 5.0])
        printed[bad_line]["amt"] = round(printed[bad_line]["amt"] + delta, 2)
    sub_p = round(sum(l_["amt"] for l_ in printed), 2)
    tax_p = round(sub_p * tax_rate, 2)
    total_p = round(sub_p + tax_p, 2)
    if err == "total":
        total_p = round(total_p + rng.choice([1.0, -1.0, 0.1, 10.0, -0.9, 2.5, 20.0]), 2)
    sub_true = round(sum(l_["amt"] for l_ in lines), 2)
    total_true = round(sub_true + round(sub_true * tax_rate, 2), 2)
    # render
    if fam == "A":
        W = 520; im = Image.new("RGB", (W, 290 + 62 * n), jitter(rng, (252, 252, 247), 4)); d = ImageDraw.Draw(im)
        shop = rng.choice(["FRESHWAY GROCER", "BYTE & BOLT", "PAPERLEAF", "GREEN BASKET"])
        d.text((W // 2 - tw(d, shop, pen.f(24, True))[0] // 2, 18), shop, font=pen.f(24, True), fill="black")
        y = 70
        for l_ in printed:
            d.text((20, y), l_["name"], font=pen.f(19), fill="black")
            d.text((40, y + 26), f"{l_['qty']} @ {fmt(l_['unit'], cur)}", font=pen.f(17), fill=(70, 70, 70))
            s = fmt(l_["amt"], cur); d.text((W - 20 - tw(d, s, pen.f(19))[0], y + 12), s, font=pen.f(19), fill="black"); y += 62
    elif fam == "B":
        W = 900; im = Image.new("RGB", (W, 330 + 44 * n), "white"); d = ImageDraw.Draw(im)
        d.text((30, 24), "INVOICE", font=pen.f(34, True), fill=jitter(rng, (30, 64, 175)))
        d.text((W - 300, 30), f"No. INV-{rng.randint(1000, 9999)}", font=pen.f(18), fill=(60, 60, 60))
        cols = [30, 430, 560, 730]; hd = ["Description", "Qty", "Unit price", "Amount"]
        d.line((30, 100, W - 30, 100), fill=(30, 30, 30), width=2)
        for x, h in zip(cols, hd):
            d.text((x, 108), h, font=pen.f(18, True), fill=(30, 30, 30))
        y = 146
        for l_ in printed:
            for x, v in zip(cols, [l_["name"], str(l_["qty"]), fmt(l_["unit"], cur), fmt(l_["amt"], cur)]):
                d.text((x, y), v, font=pen.f(18), fill=(30, 30, 30))
            y += 44
    else:
        W = 640; im = Image.new("RGB", (W, 300 + 66 * n), jitter(rng, (255, 247, 237), 5)); d = ImageDraw.Draw(im)
        d.text((28, 22), rng.choice(["Trattoria Lume", "Café Nord", "Bistro Vela"]) + " · Rechnung", font=pen.f(26, True), fill=(67, 20, 7))
        y = 76
        for l_ in printed:
            d.rounded_rectangle((20, y, W - 20, y + 56), radius=10, fill="white")
            d.text((36, y + 6), f"{l_['qty']} × {l_['name']}", font=pen.f(19, True), fill=(40, 40, 40))
            d.text((36, y + 32), f"je {fmt(l_['unit'], cur)}", font=pen.f(15), fill=(120, 113, 108))
            s = fmt(l_["amt"], cur); d.text((W - 36 - tw(d, s, pen.f(19, True))[0], y + 16), s, font=pen.f(19, True), fill=(40, 40, 40)); y += 66
    y += 12
    d.line((20, y, W - 20, y), fill=(60, 60, 60), width=2); y += 14
    rows = [("Subtotal", sub_p), (f"Tax {int(round(tax_rate * 100))}%", tax_p), ("TOTAL", total_p)]
    for lab, v in rows:
        f = pen.f(22 if lab == "TOTAL" else 19, lab == "TOTAL")
        d.text((W // 2 - 40, y), lab, font=f, fill="black"); s = fmt(v, cur); d.text((W - 30 - tw(d, s, f)[0], y), s, font=f, fill="black"); y += 40
    qs = [noul("total_ok", "Is the printed total equal to the printed subtotal plus tax?", err != "total"),
          noul("all_ok", "Is every figure on this receipt arithmetically correct (each line = quantity × unit price, "
                         "subtotal = sum of lines, total = subtotal + tax)?", err == "none")]
    g = total_true
    ds, seen = [], {fmt(g, cur)}
    for x in [total_p, g + 1, g - 1, g + 0.1, g - 0.1, g + 10, round(g * 1.1, 2), g + 0.9]:
        s = fmt(round(x, 2), cur)
        if s not in seen and x > 0:
            seen.add(s); ds.append(s)
    opts = [fmt(g, cur)] + rng.sample(ds[:5], 3)
    qs.append(choice("true_total", "What should the total be if every line is priced correctly (quantity × unit price, "
                                   "plus the stated tax)?", opts, 0, rng))
    if err == "line" or rng.random() < .35:
        nm = [l_["name"] for l_ in printed]
        if err == "line":
            gname = nm[bad_line]
            opts = [gname] + rng.sample([x for x in nm if x != gname], min(2, n - 1)) + ["Every line is correct"]
        else:
            opts = ["Every line is correct"] + rng.sample(nm, 3)
        qs.append(choice("bad_line", "Which line's amount does not equal quantity × unit price?", opts, 0, rng))
    return [im], "<image>", qs


GENS = {"ui_state": gen_ui_state, "change_detect": gen_change_detect, "line_rule": gen_line_rule,
        "grid_games": gen_grid_games, "severity": gen_severity, "receipt_math": gen_receipt_math}


# =================================================================== driver
def balance_noul(cases):
    """Trim majority-label noul questions (from the end) until yes == no exactly. Returns (cases, [no, yes])."""
    cnt = [0, 0]
    for c in cases:
        for q in c["questions"]:
            if q["mode"] == "noul":
                cnt[int(c["gold"][q["key"]][1] == 1.0)] += 1
    maj = 0 if cnt[0] > cnt[1] else 1
    excess = abs(cnt[0] - cnt[1])
    for c in reversed(cases):
        if excess <= 0:
            break
        keep = []
        for q in reversed(c["questions"]):
            if excess > 0 and q["mode"] == "noul" and int(c["gold"][q["key"]][1] == 1.0) == maj:
                excess -= 1
                del c["gold"][q["key"]]
                continue
            keep.append(q)
        c["questions"] = list(reversed(keep))
    cases = [c for c in cases if c["questions"]]
    cnt = [0, 0]
    for c in cases:
        for q in c["questions"]:
            if q["mode"] == "noul":
                cnt[int(c["gold"][q["key"]][1] == 1.0)] += 1
    return cases, cnt


def build(gname, fam_set, n_q, root, rng, tag):
    fn = GENS[gname]
    cases, hashes, nq, k = [], [], 0, 0
    while nq < n_q * 1.35:
        fam = rng.choice(fam_set)
        ims, state, qs = fn(rng, fam, k)
        qs = [q for q in qs if q][:4]
        if not qs:
            continue
        rels = [C.save_image(root, gname, f"{tag}{k}_{j}", im, True, hashes) for j, im in enumerate(ims)]
        c = C.make_case(f"vis3_{gname}:{tag}{k}", f"vis3_{gname}", state, rels, qs)
        c["family"] = fam
        cases.append(c)
        nq += len(qs); k += 1
    # trim to ~n_q, then balance yes/no exactly
    out, t = [], 0
    for c in cases:
        if t >= n_q * 1.25:
            break
        out.append(c); t += len(c["questions"])
    out, cnt = balance_noul(out)
    for c in out:   # re-validate after trimming
        C.make_case(c["case_id"], c["source"], c["state"], c["images"],
                    [(C.Question(q["key"], q["mode"], q["instructions"], tuple(q["options"]), q["criteria"]),
                      tuple(c["gold"][q["key"]])) for q in c["questions"]])
    keep = {p for c in out for p in c["images"]}
    return out, [h for h in hashes if h["path"] in keep], cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--probe", type=int, default=300)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--font-dir", action="append", default=[],
                    help="font directory to search (repeatable), e.g. the system font dir; without one, "
                         "PIL's built-in default font is used for every text (images then differ from v4.0-VL)")
    a = ap.parse_args()
    pools = set_fonts(a.font_dir)
    print("fonts:", json.dumps({k: [os.path.basename(f) for f in v] for k, v in pools.items()}), flush=True)
    if len(FONTS_A) not in (1, 2, 4) or len(FONTS_MONO) not in (1, 2, 4):
        print(f"WARNING: {len(FONTS_A)} sans / {len(FONTS_MONO)} mono fonts found; with a pool size other than 1, 2 or 4 "
              "the rng stream differs from the v4.0-VL build, so the labels as well as the pixels will differ. "
              "Point --font-dir at a directory with 4 (or 2) of the listed fonts.", file=sys.stderr, flush=True)
    root = Path(a.root); proot = root / "probes_raw"
    only = set(filter(None, a.only.split(",")))
    for gname in GENS:
        if only and gname not in only:
            continue
        rng = random.Random(f"{a.seed}:{gname}:train")
        tr, th, tc = build(gname, ["A", "B"], a.n, root, rng, "t")
        C.write_jsonl(root / f"vis3_{gname}.jsonl", tr)
        prng = random.Random(f"{a.seed}:{gname}:probe")
        pr, ph, pc = build(gname, ["C"], a.probe, proot, prng, "p")
        C.write_jsonl(proot / f"probe_{gname}.jsonl", pr)
        with open(root / "gen_hashes.jsonl", "a") as fh:
            for h in th:
                fh.write(json.dumps(dict(h, gen=gname)) + "\n")
        with open(proot / "gen_hashes.jsonl", "a") as fh:
            for h in ph:
                fh.write(json.dumps(dict(h, gen=gname)) + "\n")
        print(gname, "train q", sum(len(c["questions"]) for c in tr), "noul no/yes", tc,
              "| probe q", sum(len(c["questions"]) for c in pr), "noul", pc, flush=True)


if __name__ == "__main__":
    main()
