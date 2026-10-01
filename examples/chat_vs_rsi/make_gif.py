"""Side-by-side GIF from the measured runs (rsi.json, chat.json): a replay, not a live recording.

    python examples/chat_vs_rsi/make_gif.py [OUT.gif] [--rsi rsi.json] [--chat chat.json]

Left: RSI-Jev v4.0-VL; its probabilities appear at the question's measured p50 latency.
Right: Qwen3.5-2B chat; its text appears token by token at the times recorded in its
streamed run. Clock: 100 ms per frame. The four questions were fixed before any run
(GIF_IDS): one screen, one chart, one photo, one factory part. Needs DejaVu fonts
(set GIF_FONT_DIR if they are not in /usr/share/fonts/truetype/dejavu/).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rsi_client import images_of, load_questions          # noqa: E402
from summarize import lenient                              # noqa: E402

GIF_IDS = ['ui-declined/status', 'chart-down/q', 'pope-dog/has_dog',
           'visa-capsules_Data_Images_Anomaly_045.JPG/inspection']
W, H, TOP, IMG = 960, 470, 150, 132
STEP = 100            # ms per frame
HOLD = 1800           # ms to hold the finished frame
F = os.environ.get('GIF_FONT_DIR', '/usr/share/fonts/truetype/dejavu/').rstrip('/') + '/'
INK, MUTED, LINE, BG, BAR, OK, BAD = '#1d1d1f', '#6e6e73', '#d2d2d7', '#ffffff', '#3a6fd8', '#1a7f37', '#c62828'


def fonts():
    f = lambda n, s: ImageFont.truetype(F + n, s)            # noqa: E731
    return f('DejaVuSans.ttf', 15), f('DejaVuSans-Bold.ttf', 16), f('DejaVuSansMono.ttf', 14), f('DejaVuSans.ttf', 12)


def label(q, key):
    if q['spec']['type'] == 'noul':
        return {'true': 'yes', 'false': 'no'}[key]
    return (q.get('labels') or {}).get(key, key)


def options(q):
    if q['spec']['type'] == 'noul':
        return ['true', 'false']
    return list(q['spec']['criteria'])


def rsi_probs(q, a):
    if a['type'] == 'noul':
        return {'true': a['noul'], 'false': 1 - a['noul']}
    return a['probabilities']


def frame(q, r, c, t, thumbs, fnt):
    SANS, BOLD, MONO, SMALL = fnt
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    x = 16
    for th in thumbs:
        im.paste(th, (x, 10))
        x += th.width + 8
    qx = x + 8
    d.text((qx, 12), 'Question', font=SMALL, fill=MUTED)
    lines = textwrap.wrap(q['spec']['instructions'], 58)[:3]
    for i, ln in enumerate(lines):
        d.text((qx, 30 + i * 20), ln, font=BOLD, fill=INK)
    y = 34 + len(lines) * 20
    ctx = ' '.join(q['state'].replace('<image>', ' ').split())
    if ctx:
        d.text((qx, y), textwrap.shorten('context: ' + ctx, 70), font=SANS, fill=MUTED)
        y += 20
    ref = ', '.join(label(q, k) for k in q['ref'])
    d.text((qx, y), f'reference answer: {ref}', font=SANS, fill=MUTED)
    d.text((W - 16, 12), f't = {t / 1000:.1f} s', font=BOLD, fill=INK, anchor='ra')
    d.text((W - 16, 34), 'REPLAY of measured timings', font=SMALL, fill=MUTED, anchor='ra')
    d.line([(0, TOP - 4), (W, TOP - 4)], fill=LINE)
    d.line([(W // 2, TOP), (W // 2, H - 26)], fill=LINE)
    L = 16
    d.text((L, TOP + 6), 'RSI-Jev v4.0-VL', font=BOLD, fill=INK)
    d.text((L, TOP + 28), 'one forward pass, a probability per answer', font=SMALL, fill=MUTED)
    if t >= r['ms_p50']:
        pr = rsi_probs(q, r['answer'])
        keys = sorted(options(q), key=lambda k: -pr[k])[:5]
        for i, k in enumerate(keys):
            y = TOP + 56 + i * 30
            d.text((L, y), label(q, k)[:16], font=SANS, fill=INK)
            bw = int(250 * pr[k])
            d.rectangle([L + 140, y + 2, L + 140 + max(bw, 1), y + 18], fill=BAR if i == 0 else '#9db7ec')
            d.text((L + 146 + bw, y + 1), f'{pr[k]:.2f}', font=SANS, fill=INK)
        ok = r['correct']
        d.text((L, H - 60), f'answered in {r["ms_p50"]:.0f} ms  ' + ('✓ matches reference' if ok else '✗ differs from reference'),
               font=SANS, fill=OK if ok else BAD)
    R = W // 2 + 16
    d.text((R, TOP + 6), 'Qwen3.5-2B chat (non-thinking)', font=BOLD, fill=INK)
    d.text((R, TOP + 28), 'same base model, writes a JSON answer; no probability', font=SMALL, fill=MUTED)
    shown = ''
    for ms, txt in c['streamed']['tokens']:
        if ms <= t:
            shown = txt
    body = []
    for para in shown.split('\n'):
        body += textwrap.wrap(para, 52) or ['']
    for i, ln in enumerate(body[:9]):
        d.text((R, TOP + 56 + i * 20), ln, font=MONO, fill=INK)
    done = t >= c['streamed']['ms_total']
    if not done and (t // 300) % 2 == 0:
        d.text((R, TOP + 56 + min(len(body), 9) * 20), '▌', font=MONO, fill=MUTED)
    if done:
        if not c['valid']:
            got = lenient(c['text'], q)
            msg, col = ((f'answer "{label(q, got)}" under the wrong key: strict parse fails', BAD) if got
                        else ('no valid answer in the reply', BAD))
        else:
            ok = c['correct']
            msg, col = ('✓ matches reference' if ok else '✗ differs from reference'), (OK if ok else BAD)
        d.text((R, H - 60), f'finished in {c["streamed"]["ms_total"]:.0f} ms  {msg}', font=SANS, fill=col)
    d.text((16, H - 22), 'Replay at measured speed, batch 1, bf16. Left: p50 of 9 calls to rsi-jev serve. '
           'Right: the streamed run of each question.', font=SMALL, fill=MUTED)
    return im


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('out', nargs='?', default='chat-vs-rsi.gif')
    ap.add_argument('--rsi', default=str(HERE / 'rsi.json'))
    ap.add_argument('--chat', default=str(HERE / 'chat.json'))
    a = ap.parse_args(argv)
    fnt = fonts()
    qs = {q['id']: q for q in load_questions()}
    rsi = {r['id']: r for r in json.load(open(a.rsi))['results']}
    chat = {r['id']: r for r in json.load(open(a.chat))['results']}
    frames, durs = [], []
    for qid in GIF_IDS:
        q, r, c = qs[qid], rsi[qid], chat[qid]
        thumbs = []
        for im in images_of(q):
            im = im if isinstance(im, Image.Image) else Image.open(im)
            thumbs.append(ImageOps.contain(im.convert('RGB'), (IMG * 3 // 2, IMG)))
        total = max(r['ms_p50'], c['streamed']['ms_total'])
        t = 0
        while t < total + STEP:
            frames.append(frame(q, r, c, t, thumbs, fnt))
            durs.append(STEP)
            t += STEP
        durs[-1] = HOLD
    pal = [f.convert('P', palette=Image.ADAPTIVE, colors=64) for f in frames]
    out = Path(a.out)
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True, disposal=1)
    print(out, f'{out.stat().st_size / 2**20:.1f} MB', len(frames), 'frames')
    return 0


if __name__ == '__main__':
    sys.exit(main())
