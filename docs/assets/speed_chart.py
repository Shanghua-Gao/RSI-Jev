"""Draw the as-first-released -> now chart for docs/speed.md from bench_gb10.json.

A dumbbell per workload: v3.0 as first released (grey), today's default (light blue), and today with the
profile that fits the workload (blue), plus the overall speed-up. Writes a light and a dark SVG.

    python docs/assets/speed_chart.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
RUNS = json.loads((HERE / "bench_gb10.json").read_text())["runs"]


def p(cfg, section, key):
    return RUNS[cfg][section][key]["p50_ms"]


# (label, sub-label, release day, default, best, which profile gives best)
ROWS = [
    ("1 question", "80-token document", p("A", "grid", "80x1"), p("B", "grid", "80x1"), p("C", "grid", "80x1"), "server"),
    ("1 question", "1,052-token document", p("A", "grid", "1052x1"), p("B", "grid", "1052x1"), p("C", "grid", "1052x1"), "server"),
    ("32 questions", "80-token document", p("A", "grid", "80x32"), p("B", "grid", "80x32"), p("C", "grid", "80x32"), "server"),
    ("32 questions", "1,052-token document", p("A", "grid", "1052x32"), p("B", "grid", "1052x32"), p("C", "grid", "1052x32"), "server"),
    ("Agent asks again", "same 1,052-token state", p("A", "repeat", "1052x1"), p("B", "repeat", "1052x1"), p("D", "repeat", "1052x1"), "agent"),
    ("Agent state grows", "~100 tokens a step", p("A", "growth", "x1"), p("B", "growth", "x1"), p("D", "growth", "x1"), "agent"),
]

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", sub="#52514e", grid="#e6e5e0", old="#a9a8a1", mid="#9cc1ee", best="#2a78d6"),
    "dark": dict(bg="#1a1a19", ink="#ffffff", sub="#c3c2b7", grid="#34332f", old="#6f6e68", mid="#2f5f9e", best="#3987e5"),
}


def svg(theme):
    t = THEMES[theme]
    W, top, rowh = 1000, 118, 62
    L, R = 300, 800                               # plot x range
    xmax = 450.0
    x = lambda ms: L + (R - L) * ms / xmax
    H = top + rowh * len(ROWS) + 56
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>',
         f'<text x="32" y="44" font-size="22" font-weight="700" fill="{t["ink"]}">RSI-Jev v3.0 on an HP ZGX Nano (NVIDIA GB10): as first released → now</text>',
         f'<text x="32" y="72" font-size="15" fill="{t["sub"]}">Milliseconds per request, median of 20 runs. Shorter is faster. Same model, same answers.</text>']
    lx = 32
    for label, col, r in (("v3.0 as first released", t["old"], 6), ("now, default", t["mid"], 6), ("now, with the profile that fits", t["best"], 7)):
        o.append(f'<circle cx="{lx + 6}" cy="96" r="{r}" fill="{col}"/><text x="{lx + 18}" y="101" font-size="14" fill="{t["sub"]}">{label}</text>')
        lx += 18 + len(label) * 7.4 + 26
    y0 = top
    for ms in (0, 100, 200, 300, 400):
        o.append(f'<line x1="{x(ms):.1f}" y1="{y0 - 8}" x2="{x(ms):.1f}" y2="{y0 + rowh * len(ROWS)}" stroke="{t["grid"]}" stroke-width="1"/>'
                 f'<text x="{x(ms):.1f}" y="{y0 + rowh * len(ROWS) + 22}" text-anchor="middle" font-size="13" fill="{t["sub"]}">{ms}</text>')
    o.append(f'<text x="{R:.1f}" y="{y0 + rowh * len(ROWS) + 44}" text-anchor="end" font-size="13" fill="{t["sub"]}">ms</text>')
    for i, (lab, sub, a, b, best, prof) in enumerate(ROWS):
        cy = y0 + rowh * i + rowh / 2
        o.append(f'<text x="32" y="{cy - 3:.1f}" font-size="16" font-weight="600" fill="{t["ink"]}">{lab}</text>'
                 f'<text x="32" y="{cy + 17:.1f}" font-size="13.5" fill="{t["sub"]}">{sub}</text>')
        o.append(f'<line x1="{x(best):.1f}" y1="{cy:.1f}" x2="{x(a):.1f}" y2="{cy:.1f}" stroke="{t["grid"]}" stroke-width="4" stroke-linecap="round"/>')
        o.append(f'<circle cx="{x(a):.1f}" cy="{cy:.1f}" r="6" fill="{t["old"]}"><title>v3.0 as first released: {a:.1f} ms</title></circle>')
        o.append(f'<circle cx="{x(b):.1f}" cy="{cy:.1f}" r="6" fill="{t["mid"]}"><title>now, default: {b:.1f} ms</title></circle>')
        o.append(f'<circle cx="{x(best):.1f}" cy="{cy:.1f}" r="7" fill="{t["best"]}"><title>now, --profile {prof}: {best:.1f} ms</title></circle>')
        o.append(f'<text x="{x(a) + 12:.1f}" y="{cy + 5:.1f}" font-size="13" fill="{t["sub"]}">{a:.0f}</text>')
        o.append(f'<text x="{x(best) - 12:.1f}" y="{cy - 12:.1f}" text-anchor="middle" font-size="13" fill="{t["ink"]}">{best:.0f}</text>')
        o.append(f'<text x="{W - 32}" y="{cy - 2:.1f}" text-anchor="end" font-size="19" font-weight="700" fill="{t["ink"]}">{a / best:.1f}×</text>'
                 f'<text x="{W - 32}" y="{cy + 16:.1f}" text-anchor="end" font-size="12.5" fill="{t["sub"]}">--profile {prof}</text>')
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    for th in THEMES:
        (HERE / f"speed_gb10_{th}.svg").write_text(svg(th))
    for lab, sub, a, b, best, prof in ROWS:
        print(f"{lab:18s} {sub:24s} {a:7.1f} → {b:7.1f} → {best:6.1f} ({prof})  {a / b:.2f}x default, {a / best:.2f}x best")
