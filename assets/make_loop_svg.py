"""The RSI-Jev loop figure, in the original v1.0 style, extended to every release.

Top: the champion line (15-benchmark suite, one evaluator for every release) climbs release by
release; grey dots are the experiments tried in each stretch that did not clear it (count only:
their height does not encode a score). Bottom: the loop itself (propose, experiment, learn), the
gate, the release, and the published negatives.

    python assets/make_loop_svg.py assets/loop-social.svg --v20 0.7089
    node assets/render_loop.mjs assets/loop-social.svg FRAMES_DIR 10 16   # frames for the GIF (needs playwright)

v2.0 = 0.7089 is its score on the same evaluator as the v3.0 card (scored separately; the card has v1.0, v2.1, v3.0).
"""
import argparse
import re
from pathlib import Path

EXPLORE = Path(__file__).resolve().parents[1] / "EXPLORE.md"
THEMES = {
    "light": dict(bg="#fbfcfd", ink="#212529", muted="#868e96", line="#e9ecef", teal="#0b7285",
                  grey="#adb5bd", box="#fff", soft="#f1f3f5", onteal="#fff"),
    "dark": dict(bg="#0c0e13", ink="#eceff4", muted="#8e98a8", line="#262c38", teal="#7c95ff",
                 grey="#4a5361", box="#131820", soft="#1b212c", onteal="#0c0e13"),
}


def stretch_counts():
    secs = [s for s in re.split(r"\n## ", EXPLORE.read_text())[1:] if any(l.startswith("| `") for l in s.splitlines())]
    return [sum(1 for l in s.splitlines() if l.startswith("| `")) for s in secs]


def build(theme, v20):
    t = THEMES[theme]
    DUR = 16.0
    rel = [("v1.0", 0.622), ("v2.0", v20), ("v2.1", 0.736), ("v3.0", 0.756)]
    lo, hi = 0.55, 0.80
    top, bot = 196, 392
    y = lambda v: top + (bot - top) * (hi - v) / (hi - lo)
    counts = stretch_counts()                       # experiments tried in v1.0→v2.0, v2.0→v2.1, v2.1→v3.0
    x_rel = [300, 610, 860, 1320]                   # release x positions
    o = []
    add = o.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 792" width="1440" height="792" '
        f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">')
    add(f'<rect width="1440" height="792" fill="{t["bg"]}"/>')
    add('<defs>'
        f'<marker id="t1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{t["teal"]}"/></marker>'
        f'<marker id="g1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="{t["grey"]}"/></marker>'
        '<path id="thru" d="M 62 556 L 960 556" fill="none"/>'
        '<path id="down" d="M 62 556 L 884 556 C 948 556, 948 678, 1020 678" fill="none"/></defs>')
    add(f'<text x="64" y="80" font-size="53" font-weight="700" fill="{t["ink"]}">RSI-Jev</text>')
    add(f'<text x="1376" y="80" text-anchor="end" font-size="27" fill="{t["teal"]}">github.com/Shanghua-Gao/RSI-Jev</text>')
    add(f'<text x="64" y="120" font-size="26" fill="{t["muted"]}">recursively self-improving AutoScientists, exploring how to train Jev-like System One models</text>')
    add(f'<text x="64" y="152" font-size="22" fill="{t["ink"]}">the models, and every experiment behind them &#8212; what worked and what did not</text>')
    add(f'<line x1="64" y1="178" x2="1376" y2="178" stroke="{t["line"]}" stroke-width="1.5"/>')
    for v in (0.60, 0.65, 0.70, 0.75, 0.80):
        add(f'<line x1="92" y1="{y(v):.1f}" x2="1376" y2="{y(v):.1f}" stroke="{t["line"]}" stroke-width="1.5"/>'
            f'<text x="82" y="{y(v) + 7:.1f}" text-anchor="end" font-size="19" fill="{t["muted"]}">{v:.2f}</text>')
    add(f'<text x="1376" y="{bot + 26}" text-anchor="end" font-size="17" fill="{t["muted"]}">15-benchmark suite, one evaluator for every release</text>')

    # timeline: before v1.0 (six turns), then each release
    k = lambda s: f"{s / DUR:.4f}"
    t_rel = [2.0, 5.2, 7.8, 11.4]                   # when each release lands (s)
    # champion line: steps at each release
    ys = [y(0.55)] + [y(v) for _, v in rel]
    times = [0.0] + t_rel
    vals = ";".join(f"{v:.1f}" for v in ys + [ys[-1]])
    kts = ";".join(k(s) for s in times) + ";1"
    add(f'<line x1="92" x2="1376" stroke="{t["teal"]}" stroke-width="3.5" stroke-dasharray="10 6">'
        f'<animate attributeName="y1" values="{vals}" keyTimes="{kts}" calcMode="discrete" dur="{DUR}s" repeatCount="indefinite"/>'
        f'<animate attributeName="y2" values="{vals}" keyTimes="{kts}" calcMode="discrete" dur="{DUR}s" repeatCount="indefinite"/></line>')
    # six turns before v1.0, as a bracket
    add(f'<g opacity="0"><path d="M 110 {bot - 8} L 110 {bot} L 250 {bot} L 250 {bot - 8}" fill="none" stroke="{t["muted"]}" stroke-width="2"/>'
        f'<text x="180" y="{bot - 18}" text-anchor="middle" font-size="19" fill="{t["muted"]}">six turns</text>'
        f'<animate attributeName="opacity" values="0;0;1;1" keyTimes="0;{k(0.3)};{k(0.5)};1" dur="{DUR}s" repeatCount="indefinite"/></g>')
    # releases
    for i, ((name, v), x) in enumerate(zip(rel, x_rel)):
        big = i == len(rel) - 1
        add(f'<g opacity="0"><circle cx="{x}" cy="{y(v):.1f}" r="{11 if big else 9}" fill="{t["teal"]}"/>'
            f'<text x="{x}" y="{y(v) - 30:.1f}" text-anchor="middle" font-size="{21 if big else 19}" font-weight="{700 if big else 400}" fill="{t["ink"]}" stroke="{t["bg"]}" stroke-width="7" paint-order="stroke" stroke-linejoin="round">{name} · {v:.3f}</text>'
            f'<animate attributeName="opacity" values="0;0;1;1" keyTimes="0;{k(t_rel[i])};{k(t_rel[i] + 0.2)};1" dur="{DUR}s" repeatCount="indefinite"/></g>')
    # experiments tried in each stretch: grey dots in rows just under the champion line (count only)
    for s, n in enumerate(counts):
        x0, x1 = x_rel[s] + 26, x_rel[s + 1] - 26
        start, end = t_rel[s] + 0.3, t_rel[s + 1] - 0.3
        line_y = y(rel[s][1])
        pitch = 12
        per_row = max(1, int((x1 - x0) // pitch))
        for j in range(n):
            cx = x0 + (j % per_row) * pitch + pitch / 2
            cy = line_y + 16 + (j // per_row) * pitch
            ts = start + (end - start) * j / max(1, n - 1)
            add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{t["grey"]}" opacity="0">'
                f'<animate attributeName="opacity" values="0;0;1;1" keyTimes="0;{k(ts)};{k(ts + 0.1)};1" dur="{DUR}s" repeatCount="indefinite"/></circle>')
        add(f'<text x="{(x0 + x1) / 2:.1f}" y="{line_y + 16 + ((n - 1) // per_row) * pitch + 34:.1f}" text-anchor="middle" font-size="18" fill="{t["muted"]}" opacity="0">'
            f'{n} tried<animate attributeName="opacity" values="0;0;1;1" keyTimes="0;{k(end)};{k(end + 0.2)};1" dur="{DUR}s" repeatCount="indefinite"/></text>')
    total = sum(counts)
    add(f'<text x="136" y="452" font-size="22" fill="{t["ink"]}">four releases in four days</text>')
    add(f'<text x="1376" y="452" text-anchor="end" font-size="22" fill="{t["muted"]}" opacity="0">{total} experiments since v1.0, every one published'
        f'<animate attributeName="opacity" values="0;0;1;1" keyTimes="0;{k(t_rel[-1])};{k(t_rel[-1] + 0.2)};1" dur="{DUR}s" repeatCount="indefinite"/></text>')

    # the loop (unchanged from the original, with the current release)
    for i, beg in enumerate([0.0, 0.81, 1.62, 2.44, 3.25, 4.06, 4.88, 5.69]):
        win = i == 5
        add(f'<g opacity="0"><rect x="-17" y="-12" width="34" height="24" rx="5" fill="{t["teal"] if win else t["grey"]}"/>'
            f'<animateMotion dur="6.5s" begin="{beg:.2f}s" repeatCount="indefinite"><mpath href="#{"thru" if win else "down"}"/></animateMotion>'
            f'<animate attributeName="opacity" values="0;1;1;0" keyTimes="0;.05;.9;1" dur="6.5s" begin="{beg:.2f}s" repeatCount="indefinite"/></g>')
    for x, label in ((114, "propose"), (340, "experiment"), (566, "learn")):
        add(f'<rect x="{x}" y="532" width="184" height="56" rx="28" fill="{t["box"]}" stroke="{t["teal"]}" stroke-width="3"/>'
            f'<text x="{x + 92}" y="569" text-anchor="middle" font-size="25" fill="{t["teal"]}">{label}</text>')
    add(f'<path d="M 658 528 C 658 498, 206 498, 206 522" fill="none" stroke="{t["teal"]}" stroke-width="2.5" stroke-dasharray="7 5" marker-end="url(#t1)"/>')
    add(f'<text x="432" y="486" text-anchor="middle" font-size="22" fill="{t["ink"]}">what it learns changes what it proposes</text>')
    add(f'<line x1="856" y1="508" x2="856" y2="606" stroke="{t["teal"]}" stroke-width="3" stroke-dasharray="6 5"/>'
        f'<text x="856" y="497" text-anchor="middle" font-size="22" fill="{t["ink"]}">beats the champion?</text>')
    add(f'<rect x="920" y="648" width="376" height="62" rx="12" fill="{t["soft"]}"/><text x="1108" y="687" text-anchor="middle" font-size="23" fill="{t["ink"]}">published, every one</text>')
    add(f'<rect x="976" y="530" width="132" height="62" rx="12" fill="{t["teal"]}"/><text x="1042" y="574" text-anchor="middle" font-size="29" fill="{t["onteal"]}">v3.0</text>')
    add(f'<path d="M 1110 556 L 1134 556" stroke="{t["grey"]}" stroke-width="2.5" stroke-dasharray="5 4"/>')
    add(f'<rect x="1138" y="530" width="132" height="62" rx="12" fill="none" stroke="{t["grey"]}" stroke-width="3" stroke-dasharray="7 5"/>'
        f'<text x="1204" y="574" text-anchor="middle" font-size="29" fill="{t["muted"]}">next<animate attributeName="opacity" values="0.3;1;0.3" dur="2.4s" repeatCount="indefinite"/></text>')
    add(f'<path d="M 1204 592 L 1204 612 Q 1204 626 1218 626 L 1344 626 Q 1358 626 1358 640 L 1358 744 Q 1358 758 1344 758 L 220 758 Q 206 758 206 744 L 206 594" fill="none" stroke="{t["grey"]}" stroke-width="2.5" stroke-dasharray="6 5" marker-end="url(#g1)"/>')
    add(f'<text x="566" y="750" text-anchor="middle" font-size="22" fill="{t["muted"]}">and becomes the bar to clear</text></svg>')
    return "\n".join(o)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--theme", default="light", choices=THEMES)
    ap.add_argument("--v20", type=float, required=True)
    a = ap.parse_args()
    Path(a.out).write_text(build(a.theme, a.v20))
    print(a.out, "counts", stretch_counts())
