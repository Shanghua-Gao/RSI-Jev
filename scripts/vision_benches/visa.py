"""VisA anomaly detection, zero-shot: image AUROC over 12 categories, paired against Jev-Omni and Gemma.

    python scripts/vision_benches/visa.py --visa-root DATA/visa --download      # 1.9 GB, once
    python scripts/vision_benches/visa.py --visa-root DATA/visa [--model v4.0-vl-2b | --server URL]

The data is the official archive VisA_20220922.tar (Amazon Science, CC BY 4.0), checked
against its sha256; the test split is the official 1cls split (2,162 images: 962 good,
1,200 defective). The prompt is jev-omni-inspection's frozen one (`jev_inspection.visa.prompt`,
config "single"), imported from a checkout at the pinned commit: one photo, a state that
names the part, "Is everything in the photo good, or is at least one part defective?",
options A/B. Each image is asked twice, once per option order, and the score is the mean
of the two log-odds of "defective". The photo is scaled to fit 1,536 px first, as that
harness does. The model has seen no example of a good part.

Statistics are jev-omni-inspection's (`jev_inspection.score`): macro image AUROC over the
12 categories, a category-stratified bootstrap (2,000 draws, seed 0) with the same draws
for every system, so the differences are paired. A0 (Jev-Omni) and B0 (Gemma 4 12B) are
recomputed from the per-image predictions that repository publishes
(`results/jev_visa.jsonl`, `results/base_visa.jsonl`); the PatchCore rows are its
published summary.

Published: v4.0-VL 86.6 (85.1-88.1); vs A0 +5.5 [+3.5, +7.5], vs B0 +3.7 [+1.8, +5.6].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                              # noqa: E402

MAX_SIDE = 1536          # the harness's pre-downscale (>= what the 1,024-token image budget uses)
B, SEED = 2000, 0


def inspection(ins: Path):
    """Their prompt and split code (the statistics module, which needs scipy, is imported by analyze)."""
    if str(ins / "src") not in sys.path:
        sys.path.insert(0, str(ins / "src"))
    from jev_inspection import visa                             # noqa: E402  (their code)
    return visa


def download(root: Path, visa) -> None:
    """The official archive, verified, unpacked into root."""
    if (root / "split_csv" / "1cls.csv").exists():
        print(root, "already has VisA")
        return
    root.mkdir(parents=True, exist_ok=True)
    tar = root / "VisA_20220922.tar"
    if not tar.exists():
        subprocess.run(["curl", "-fL", "-o", str(tar), visa.ARCHIVE_URL], check=True)
    h = hashlib.sha256()
    with tar.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != visa.ARCHIVE_SHA256:
        raise SystemExit(f"{tar}: sha256 {h.hexdigest()} != {visa.ARCHIVE_SHA256}")
    subprocess.run(["tar", "xf", str(tar), "-C", str(root)], check=True)
    tar.unlink()


def load_image(path: Path):
    from PIL import Image
    im = Image.open(path)
    im.draft("RGB", (MAX_SIDE, MAX_SIDE))
    im = im.convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return im


def item_request(prompt) -> tuple[str, dict]:
    """(state, questions) for one frozen prompt (one option order)."""
    return prompt.state, {"inspection": C.choice_question(prompt.question, list(prompt.options))}


def score_image(client, visa, item, root: Path, orders=(0, 1)) -> list[dict]:
    img = load_image(root / item.image)
    recs = []
    for order in orders:
        p = visa.prompt(item.obj, "single", order)
        state, questions = item_request(p)
        answers, ms = client.ask(state, questions, [img])
        pr = C.answer_probs(answers["inspection"], list(questions["inspection"]["criteria"]))
        d = p.defective_index
        recs.append({"key": f"{item.image}|single|{order}", "image": item.image, "obj": item.obj, "label": item.label,
                     "types": list(item.types), "config": "single", "order": order, "probs": pr,
                     "logodds_defective": C.log_odds(pr[d], pr[1 - d]), "ms": round(ms, 2)})
    return recs


def run(client, visa, root: Path, out: Path, limit: int | None = None) -> None:
    os.environ.setdefault("VISA_ROOT", str(root))
    tests = [i for i in visa.load(root) if i.split == "test"]
    done = C.done_keys(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for n, it in enumerate(tests[: limit or None]):
            if f"{it.image}|single|0" in done and f"{it.image}|single|1" in done:
                continue
            for rec in score_image(client, visa, it, root):
                if rec["key"] not in done:
                    fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 100 == 0:
                print(n, "of", len(tests), it.image, flush=True)


def analyze(results: Path, root: Path, ins: Path) -> tuple[list[str], dict]:
    import numpy as np
    import pandas as pd
    inspection(ins)
    from jev_inspection import score as S                       # noqa: E402  (their statistics)
    tab = S.Table(root)
    vlm, bias = S.load_vlm([ins / "results/jev_visa.jsonl", ins / "results/base_visa.jsonl"])
    systems = {n: vlm[n].reindex(tab.images).to_numpy() for n in ("A0", "B0")}
    ours = pd.DataFrame(C.read_jsonl(results))
    ours = ours[ours.config == "single"]
    wide = ours.pivot_table(index="image", columns="order", values="logodds_defective")
    s = wide.mean(axis=1).reindex(tab.images).to_numpy()
    missing = int(np.isnan(s).sum())
    if missing:
        return [f"VisA: {missing} of {len(s)} test images have no score yet (partial run); no AUROC."], {"missing": missing}
    systems["ours"] = s
    bias["ours"] = float((wide[0] - wide[1]).mean())
    draws = tab.resample(B, np.random.default_rng(SEED))
    boot = {n: S.macro_boot(tab, draws, v) for n, v in systems.items()}
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]      # noqa: E731
    pc = {n: S.per_category(tab, v) for n, v in systems.items()}
    point = {n: float(np.mean(list(v.values()))) for n, v in pc.items()}
    pub = json.loads((ins / "results/summary.json").read_text())
    res = {"point": point, "ci": {n: ci(b) for n, b in boot.items()}, "per_category": pc, "bias_logodds": bias}
    L = [f"## VisA: {len(tab.images)} test images, 12 categories, frozen prompt, one image, both option orders averaged", "",
         "| system | good parts seen | macro image AUROC (95% CI) | v4.0-VL minus system, paired bootstrap (95% CI) |",
         "|---|---|---|---|"]
    names = {"ours": "v4.0-VL (as served)", "A0": "A0 Jev-Omni (recomputed)", "B0": "B0 Gemma 4 12B (recomputed)"}
    for n in ("ours", "A0", "B0"):
        cell = ""
        if n == "ours":
            parts = []
            for o in ("A0", "B0"):
                dd = boot["ours"] - boot[o]
                p2 = max(1 / B, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))
                parts.append(f"vs {o}: {100 * (point['ours'] - point[o]):+.1f} [{100 * ci(dd)[0]:+.1f}, "
                             f"{100 * ci(dd)[1]:+.1f}], p~{p2:.3g}")
                res[f"ours_minus_{o}"] = {"diff": point["ours"] - point[o], "ci": ci(dd), "p": p2}
            cell = "; ".join(parts)
        L.append(f"| {names[n]} | 0 | {100 * point[n]:.1f} ({100 * ci(boot[n])[0]:.1f}-{100 * ci(boot[n])[1]:.1f}) | {cell} |")
    for k, lab, gp in (("C16", "PatchCore 256 px, k=16 (published)", "16"),
                       ("Call", "PatchCore 256 px, all (published)", "449-904")):
        m = pub["macro_auroc"].get(k)
        if m:
            L.append(f"| {lab} | {gp} | {100 * m['point']:.1f} ({100 * m['ci95'][0]:.1f}-{100 * m['ci95'][1]:.1f}) "
                     f"| not paired (seeded k-samples) |")
    L += ["", f"Check: recomputed A0 {100 * point['A0']:.1f} vs published {100 * pub['macro_auroc']['A0']['point']:.1f}.",
          f"Positional bias (log-odds, order 0 minus order 1): v4.0-VL {bias['ours']:.2f}, A0 {bias['A0']:.2f}, "
          f"B0 {bias['B0']:.2f}.", "",
          "| category | v4.0-VL | A0 Jev-Omni | B0 Gemma | PatchCore k=16 (256 px) |", "|---|---|---|---|---|"]
    for c in tab.categories:
        L.append(f"| {c} | {100 * pc['ours'][c]:.1f} | {100 * pc['A0'][c]:.1f} | {100 * pc['B0'][c]:.1f} | "
                 f"{100 * pub['per_category_auroc']['C16'][c]:.1f} |")
    return L, res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VisA: zero-shot image AUROC through the public API")
    C.add_client_args(ap)
    ap.add_argument("--visa-root", required=True, help="where VisA is (or goes, with --download)")
    ap.add_argument("--download", action="store_true", help="fetch and verify the official archive first")
    ap.add_argument("--out", default="bench-results/visa.jsonl")
    ap.add_argument("--limit", type=int, help="first N test images only (a smoke run; no AUROC)")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args(argv)
    ins = C.upstream("jev-omni-inspection")
    visa = inspection(ins)
    root, out = Path(a.visa_root), Path(a.out)
    if a.download:
        download(root, visa)
    if not a.analyze_only:
        client = C.Client(a.model, a.server)
        print(client.name, flush=True)
        run(client, visa, root, out, a.limit)
    lines, res = analyze(out, root, ins)
    print("\n".join(lines))
    out.with_suffix(".summary.json").write_text(json.dumps(res, indent=1, default=float))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
