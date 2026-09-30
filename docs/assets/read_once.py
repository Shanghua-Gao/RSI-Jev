"""Draw the 'read the document once' diagram for docs/speed.md (light and dark SVG).

    python docs/assets/read_once.py
"""
from pathlib import Path

HERE = Path(__file__).parent
THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", sub="#52514e", doc="#9cc1ee", q="#2a78d6", line="#e6e5e0"),
    "dark": dict(bg="#1a1a19", ink="#ffffff", sub="#c3c2b7", doc="#2f5f9e", q="#3987e5", line="#34332f"),
}


def svg(t):
    W, H = 1000, 380
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{t["bg"]}"/>']
    doc_w, q_w, h, gap = 250, 34, 16, 6

    def row(x, y, with_doc=True):
        s = ""
        if with_doc:
            s += f'<rect x="{x}" y="{y}" width="{doc_w}" height="{h}" rx="3" fill="{t["doc"]}"/>'
            x += doc_w + 3
        s += f'<rect x="{x}" y="{y}" width="{q_w}" height="{h}" rx="3" fill="{t["q"]}"/>'
        return s

    # left: read 8 times
    o.append(f'<text x="40" y="44" font-size="20" font-weight="700" fill="{t["ink"]}">Before: the ticket is read 8 times</text>')
    o.append(f'<text x="40" y="68" font-size="14" fill="{t["sub"]}">each question is its own sequence</text>')
    for i in range(8):
        o.append(row(40, 90 + i * (h + gap)))
    # right: read once
    X = 540
    o.append(f'<text x="{X}" y="44" font-size="20" font-weight="700" fill="{t["ink"]}">After: read once, every question continues</text>')
    o.append(f'<text x="{X}" y="68" font-size="14" fill="{t["sub"]}">the questions branch off one reading</text>')
    y0 = 90 + 3.5 * (h + gap)
    o.append(f'<rect x="{X}" y="{y0}" width="{doc_w}" height="{h}" rx="3" fill="{t["doc"]}"/>')
    bx = X + doc_w + 30
    for i in range(8):
        y = 90 + i * (h + gap)
        o.append(f'<path d="M {X + doc_w} {y0 + h / 2} C {X + doc_w + 16} {y0 + h / 2}, {bx - 16} {y + h / 2}, {bx} {y + h / 2}" '
                 f'fill="none" stroke="{t["sub"]}" stroke-opacity="0.45" stroke-width="2"/>')
        o.append(f'<rect x="{bx}" y="{y}" width="{q_w}" height="{h}" rx="3" fill="{t["q"]}"/>')
    # legend + totals
    ly = 300
    o.append(f'<rect x="40" y="{ly}" width="16" height="16" rx="3" fill="{t["doc"]}"/>'
             f'<text x="64" y="{ly + 13}" font-size="15" fill="{t["sub"]}">ticket, ~1,000 tokens</text>'
             f'<rect x="250" y="{ly}" width="16" height="16" rx="3" fill="{t["q"]}"/>'
             f'<text x="274" y="{ly + 13}" font-size="15" fill="{t["sub"]}">one question and its options, ~20–60 tokens</text>')
    o.append(f'<text x="40" y="{ly + 52}" font-size="16" fill="{t["ink"]}">8 × (1,000 + 50) ≈ 8,400 tokens read</text>'
             f'<text x="{X}" y="{ly + 52}" font-size="16" fill="{t["ink"]}">1,000 + 8 × 50 ≈ 1,400 tokens read</text>')
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    for name, t in THEMES.items():
        (HERE / f"read_once_{name}.svg").write_text(svg(t))
