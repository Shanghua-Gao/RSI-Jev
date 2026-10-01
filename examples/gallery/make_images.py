"""Redraw the gallery's generated pictures (charts, screens, receipts, shapes).

    python examples/gallery/make_images.py [OUT_DIR]      # default: a fresh dir, never images/generated

The PNGs in images/generated/ are the exact pictures the published numbers came from;
this script is how they were drawn. It needs DejaVu Sans (the `fonts-dejavu-core`
package on Debian/Ubuntu; set GALLERY_FONT to another path). With the same font file
and Pillow it reproduces the shipped PNGs pixel for pixel; `--check` compares.
A different font or Pillow version moves text by a pixel or two, which can move an
answer that was a close call, so score the shipped PNGs when you compare numbers.

All of these pictures are ours and are released under CC0.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

FONT = os.environ.get("GALLERY_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def font(n):
    try:
        return ImageFont.truetype(FONT, n)
    except OSError:
        return ImageFont.load_default()


# ------------------------------------------------------------------ screens
def ui_form(done_fields, path):
    """Checkout form; done_fields decides which step is next."""
    im = Image.new('RGB', (900, 640), (246, 247, 249)); d = ImageDraw.Draw(im)
    d.text((40, 24), 'Checkout', font=font(34), fill=(20, 20, 20))
    fields = [('Email', 'ana@example.com'), ('Card number', '4242 4242 4242 4242'), ('Expiry', '09/28')]
    y = 100
    for i, (lab, val) in enumerate(fields):
        d.text((40, y), lab, font=font(20), fill=(60, 60, 60))
        d.rectangle((40, y + 28, 600, y + 72), outline=(150, 150, 150), fill='white', width=2)
        if i < done_fields:
            d.text((52, y + 38), val, font=font(20), fill=(20, 20, 20))
        y += 100
    d.rectangle((40, y + 10, 300, y + 64), fill=(37, 99, 235)); d.text((80, y + 24), 'Pay $42.00', font=font(24), fill='white')
    d.rectangle((330, y + 10, 520, y + 64), outline=(120, 120, 120), width=2); d.text((372, y + 24), 'Cancel', font=font(24), fill=(60, 60, 60))
    im.save(path)


def layout(broken, path):
    im = Image.new('RGB', (900, 560), 'white'); d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 900, 70), fill=(24, 24, 27)); d.text((30, 20), 'Acme Docs', font=font(28), fill='white')
    d.rectangle((30, 100, 250, 520), fill=(244, 244, 245))
    for i, t in enumerate(['Overview', 'Install', 'API', 'FAQ']):
        d.text((50, 120 + 50 * i), t, font=font(22), fill=(40, 40, 40))
    x = 480 if broken else 290
    d.text((x, 110), 'Getting started', font=font(32), fill='black')
    body = ['Install the package with pip.', 'Create a client with your key.', 'Call decide() with a state.']
    for i, t in enumerate(body):
        d.text((x, 170 + 40 * i), t, font=font(22), fill=(50, 50, 50))
    if broken:
        d.rectangle((600, 60, 880, 300), fill=(37, 99, 235)); d.text((630, 160), 'Sign up!', font=font(30), fill='white')
    im.save(path)


def dialog(severity, path):
    im = Image.new('RGB', (800, 420), (230, 232, 236)); d = ImageDraw.Draw(im)
    d.rectangle((100, 60, 700, 360), fill='white', outline=(120, 120, 120), width=2)
    title, body, col = {
        'low': ('Update available', 'Version 3.4.1 is ready to install. Restart when convenient.', (37, 99, 235)),
        'medium': ('Sync paused', 'We could not reach the server. Retrying in 5 minutes.', (234, 179, 8)),
        'high': ('Disk failure detected', 'Drive /dev/sdb reported unrecoverable read errors. Data loss is likely.', (220, 38, 38)),
    }[severity]
    d.rectangle((100, 60, 700, 110), fill=col); d.text((120, 72), title, font=font(24), fill='white')
    words, lines, cur = body.split(), [], ''
    for w in words:
        if len(cur) + len(w) > 42:
            lines.append(cur); cur = ''
        cur += w + ' '
    lines.append(cur)
    for i, ln in enumerate(lines):
        d.text((120, 140 + 36 * i), ln, font=font(22), fill='black')
    d.rectangle((560, 300, 680, 344), fill=(37, 99, 235)); d.text((595, 310), 'OK', font=font(22), fill='white')
    im.save(path)


def confirm_delete(path):
    im = Image.new('RGB', (800, 420), (230, 232, 236)); d = ImageDraw.Draw(im)
    d.rectangle((100, 60, 700, 360), fill='white', outline=(120, 120, 120), width=2)
    d.text((130, 90), 'Delete "Q3-report.xlsx"?', font=font(28), fill='black')
    d.text((130, 150), 'This file will be removed for everyone.', font=font(22), fill=(60, 60, 60))
    d.text((130, 186), 'This cannot be undone.', font=font(22), fill=(60, 60, 60))
    d.rectangle((370, 290, 500, 336), outline=(120, 120, 120), width=2); d.text((395, 300), 'Cancel', font=font(22), fill=(40, 40, 40))
    d.rectangle((530, 290, 670, 336), fill=(220, 38, 38)); d.text((560, 300), 'Delete', font=font(22), fill='white')
    im.save(path)


def payment_failed(path):
    im = Image.new('RGB', (900, 520), (246, 247, 249)); d = ImageDraw.Draw(im)
    d.text((40, 24), 'Checkout', font=font(34), fill=(20, 20, 20))
    d.rectangle((40, 100, 860, 190), fill=(254, 226, 226), outline=(220, 38, 38), width=2)
    d.text((64, 116), 'Your card was declined.', font=font(24), fill=(153, 27, 27))
    d.text((64, 152), 'Try a different card or contact your bank.', font=font(20), fill=(153, 27, 27))
    d.rectangle((40, 230, 330, 284), fill=(37, 99, 235)); d.text((66, 244), 'Use another card', font=font(22), fill='white')
    d.rectangle((360, 230, 560, 284), outline=(120, 120, 120), width=2); d.text((400, 244), 'Try again', font=font(22), fill=(60, 60, 60))
    d.text((40, 330), 'Order total: $42.00', font=font(22), fill=(60, 60, 60))
    im.save(path)


# ------------------------------------------------------------------ charts and diagrams
def bars(values, labels, path, axis_zero=True):
    im = Image.new('RGB', (900, 560), 'white'); d = ImageDraw.Draw(im)
    lo = 0 if axis_zero else min(values) * 0.9; hi = max(values) * 1.1
    d.line((80, 480, 860, 480), fill='black', width=2); d.line((80, 40, 80, 480), fill='black', width=2)
    d.text((20, 470), f'{lo:.0f}', font=font(16), fill='black'); d.text((20, 40), f'{hi:.0f}', font=font(16), fill='black')
    w = 700 // len(values)
    for i, (v, lab) in enumerate(zip(values, labels)):
        h = (v - lo) / (hi - lo) * 440; x = 110 + i * w
        d.rectangle((x, 480 - h, x + w - 40, 480), fill=(59, 130, 246)); d.text((x, 490), lab, font=font(18), fill='black')
    im.save(path)


def line(series, path):
    im = Image.new('RGB', (900, 500), 'white'); d = ImageDraw.Draw(im)
    d.line((60, 440, 860, 440), fill='black', width=2); d.line((60, 30, 60, 440), fill='black', width=2)
    lo, hi = min(series), max(series)
    pts = [(60 + i * 800 / (len(series) - 1), 440 - (v - lo) / (hi - lo + 1e-9) * 380) for i, v in enumerate(series)]
    d.line(pts, fill=(220, 38, 38), width=4); d.text((70, 10), 'Weekly active users', font=font(20), fill='black')
    im.save(path)


def ttt(board, path):
    im = Image.new('RGB', (600, 600), 'white'); d = ImageDraw.Draw(im)
    for k in (1, 2):
        d.line((k * 200, 20, k * 200, 580), fill='black', width=6); d.line((20, k * 200, 580, k * 200), fill='black', width=6)
    for i, c in enumerate(board):
        r, col = divmod(i, 3); cx, cy = col * 200 + 100, r * 200 + 100
        if c == 'X':
            d.line((cx - 60, cy - 60, cx + 60, cy + 60), fill=(220, 38, 38), width=12)
            d.line((cx - 60, cy + 60, cx + 60, cy - 60), fill=(220, 38, 38), width=12)
        if c == 'O':
            d.ellipse((cx - 62, cy - 62, cx + 62, cy + 62), outline=(37, 99, 235), width=12)
        if c == '.':
            d.text((cx - 10, cy - 14), str(i + 1), font=font(28), fill=(170, 170, 170))
    im.save(path)


def receipt(item, amount, alcohol, path):
    im = Image.new('RGB', (520, 760), (252, 252, 248)); d = ImageDraw.Draw(im)
    d.text((140, 30), 'THE ANCHOR TAVERN', font=font(26), fill='black'); d.text((180, 70), '14 Quay St', font=font(18), fill='black')
    y = 140
    lines = [(item, amount * 0.7), ('Sparkling water', amount * 0.1)] + \
        ([('Bottle of Rioja', amount * 0.2)] if alcohol else [('Side salad', amount * 0.2)])
    for n, a in lines:
        d.text((40, y), n, font=font(22), fill='black'); d.text((380, y), f'${a:.2f}', font=font(22), fill='black'); y += 44
    d.line((40, y + 10, 480, y + 10), fill='black', width=2)
    d.text((40, y + 30), 'TOTAL', font=font(26), fill='black'); d.text((360, y + 30), f'${amount:.2f}', font=font(26), fill='black')
    im.save(path)


def pie(values, labels, path):
    im = Image.new('RGB', (900, 560), 'white'); d = ImageDraw.Draw(im)
    cols = [(59, 130, 246), (234, 88, 12), (22, 163, 74), (168, 85, 247), (234, 179, 8)]
    tot, a0 = sum(values), -90
    for i, v in enumerate(values):
        a1 = a0 + 360 * v / tot
        d.pieslice((80, 60, 520, 500), a0, a1, fill=cols[i], outline='white', width=3)
        a0 = a1
    for i, lab in enumerate(labels):
        d.rectangle((600, 150 + i * 60, 630, 180 + i * 60), fill=cols[i]); d.text((645, 150 + i * 60), lab, font=font(24), fill='black')
    d.text((80, 14), 'Traffic by channel', font=font(24), fill='black')
    im.save(path)


def two_lines(a, b, path):
    im = Image.new('RGB', (900, 500), 'white'); d = ImageDraw.Draw(im)
    d.line((60, 440, 860, 440), fill='black', width=2); d.line((60, 30, 60, 440), fill='black', width=2)
    lo, hi = min(a + b), max(a + b)
    for s, col, lab in ((a, (37, 99, 235), 'Plan A'), (b, (234, 88, 12), 'Plan B')):
        pts = [(60 + i * 800 / (len(s) - 1), 440 - (v - lo) / (hi - lo) * 380) for i, v in enumerate(s)]
        d.line(pts, fill=col, width=5); y0 = pts[0][1]
        d.text((pts[0][0] + 14, y0 + 12 if y0 < 120 else y0 - 36), lab, font=font(22), fill=col)
    for i in range(len(a)):
        d.text((60 + i * 800 / (len(a) - 1) - 10, 450), f'M{i + 1}', font=font(16), fill='black')
    d.text((70, 6), 'Monthly cost', font=font(20), fill='black')
    im.save(path)


def scatter(r, seed, path):
    rng = random.Random(seed)
    im = Image.new('RGB', (700, 560), 'white'); d = ImageDraw.Draw(im)
    d.line((60, 500, 680, 500), fill='black', width=2); d.line((60, 20, 60, 500), fill='black', width=2)
    for _ in range(60):
        x = rng.random(); y = r * x + (1 - abs(r)) * rng.random() + (1 - r) / 2 * (r < 0) + 0.15 * rng.gauss(0, 1)
        y = min(max(y, 0), 1.15)
        px, py = 70 + x * 590, 490 - y * 400
        d.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(37, 99, 235))
    d.text((300, 515), 'Hours studied', font=font(18), fill='black'); d.text((70, 2), 'Test score', font=font(18), fill='black')
    im.save(path)


def food_chain(path):
    im = Image.new('RGB', (900, 380), 'white'); d = ImageDraw.Draw(im)
    names = ['Grass', 'Rabbit', 'Fox', 'Eagle']
    cols = [(22, 163, 74), (161, 98, 7), (234, 88, 12), (71, 85, 105)]
    for i, (n, c) in enumerate(zip(names, cols)):
        x = 40 + i * 225
        d.rounded_rectangle((x, 150, x + 150, 230), 16, fill=c); d.text((x + 22, 174), n, font=font(28), fill='white')
        if i < 3:
            d.line((x + 158, 190, x + 215, 190), fill='black', width=5)
            d.polygon([(x + 222, 190), (x + 206, 180), (x + 206, 200)], fill='black')
    d.text((40, 40), 'Food chain (arrows point from food to eater)', font=font(26), fill='black')
    im.save(path)


# ------------------------------------------------------------------ counting scenes
COLORS = {'red': (220, 38, 38), 'blue': (37, 99, 235), 'green': (22, 163, 74), 'orange': (234, 88, 12)}


def shapes(seed, counts, path):
    rng = random.Random(seed)
    im = Image.new('RGB', (800, 560), (250, 250, 247)); d = ImageDraw.Draw(im)
    placed = []
    for (col, shape), k in counts.items():
        for _ in range(k):
            for _t in range(500):
                x, y = rng.randint(70, 730), rng.randint(70, 490)
                if all((x - a) ** 2 + (y - b) ** 2 > 110 ** 2 for a, b in placed):
                    break
            placed.append((x, y)); r = 40
            if shape == 'circle':
                d.ellipse((x - r, y - r, x + r, y + r), fill=COLORS[col])
            elif shape == 'square':
                d.rectangle((x - r, y - r, x + r, y + r), fill=COLORS[col])
            else:
                d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=COLORS[col])
    im.save(path)


def draw_all(out: Path) -> list[str]:
    """Every generated picture the gallery uses, under the file names questions.json uses."""
    out.mkdir(parents=True, exist_ok=True)
    o = lambda name: str(out / name)                           # noqa: E731
    shapes(1, {('red', 'circle'): 4, ('blue', 'square'): 3, ('green', 'triangle'): 2}, o('count_1.png'))
    shapes(2, {('blue', 'circle'): 2, ('blue', 'square'): 5, ('orange', 'triangle'): 3}, o('count_2.png'))
    shapes(3, {('green', 'triangle'): 3, ('red', 'square'): 1, ('orange', 'circle'): 6}, o('count_3.png'))
    shapes(4, {('orange', 'square'): 7, ('red', 'circle'): 2}, o('count_4.png'))
    shapes(5, {('red', 'circle'): 2, ('blue', 'square'): 2}, o('count_5.png'))
    bars([42, 55, 71, 48], ['Q1', 'Q2', 'Q3', 'Q4'], o('bars.png'))
    bars([63, 58, 61, 49], ['North', 'South', 'East', 'West'], o('bars_close.png'))
    line([90, 86, 88, 80, 74, 70, 63, 60], o('line_down.png'))
    line([40, 44, 43, 50, 57, 55, 66, 72], o('line_up.png'))
    bars([96, 98, 101, 99], ['A', 'B', 'C', 'D'], o('bars_trunc.png'), axis_zero=False)
    pie([22, 41, 18, 12, 7], ['Search', 'Direct', 'Social', 'Email', 'Ads'], o('pie.png'))
    two_lines([50, 52, 55, 58, 62, 66], [70, 66, 61, 57, 52, 48], o('two_lines.png'))
    scatter(0.8, 1, o('scatter_pos.png'))
    food_chain(o('food_chain.png'))
    ttt('XX.OO....', o('ttt1.png'))
    ttt('X..OO.X..', o('ttt2.png'))
    for item, amt, alc in [('Grilled fish', 38.0, False), ('Grilled fish', 38.0, True), ('Tasting menu', 180.0, False)]:
        receipt(item, amt, alc, o(f'rcpt_{amt:.0f}_{alc}.png'))
    for done in (1, 2, 3):
        ui_form(done, o(f'ui_{done}.png'))
    for sev in ('low', 'medium', 'high'):
        dialog(sev, o(f'dlg_{sev}.png'))
    layout(False, o('layout_before.png'))
    for brk in (False, True):
        layout(brk, o(f'layout_after_{brk}.png'))
    confirm_delete(o('confirm_delete.png'))
    payment_failed(o('payment_failed.png'))
    return sorted(p.name for p in out.glob('*.png'))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out', nargs='?', default='generated_redrawn')
    ap.add_argument('--check', action='store_true', help='compare with the shipped images/generated/')
    a = ap.parse_args()
    out = Path(a.out)
    names = draw_all(out)
    print(f'{len(names)} pictures -> {out}')
    if a.check:
        ship = Path(__file__).resolve().parent / 'images' / 'generated'
        diff = [n for n in names
                if ImageChops.difference(Image.open(out / n).convert('RGB'),
                                         Image.open(ship / n).convert('RGB')).getbbox() is not None]
        print('identical to the shipped PNGs' if not diff else f'differ from the shipped PNGs: {diff}')
        return 1 if diff else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
