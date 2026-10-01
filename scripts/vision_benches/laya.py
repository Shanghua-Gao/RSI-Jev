"""Laya Vision's validation sets: v4.0-VL against Laya Vision's own predictions, paired.

    B=scripts/vision_benches/laya.py
    python $B prep      --data DATA/laya                       # CPU: rebuild the 34 sets, 200-item samples
    python $B run       --data DATA/laya --out preds_v4.jsonl  [--model v4.0-vl-2b | --server URL]
    python $B reference --data DATA/laya --out preds_laya.jsonl   # Laya Vision 201M through its own code
    python $B compare   --ours preds_v4.jsonl --laya preds_laya.jsonl [--md laya.md]

**prep** replays laya-vision's own data preparation (laya_prep.py) from a checkout of
r33drichards/laya-vision at d4075b0: the same sources, seeds and record functions, so the
validation records are the ones Laya scored; check.json per set confirms the count. Then
200 items per set are drawn with a fixed seed: 6,357 questions over 34 sets.

**run** asks v4.0-VL each record as one request: the record's image(s), its state text,
and its question. choice -> a choice over the option texts (no descriptions); noul -> noul
with empty descriptions; score -> a choice over "0".."n-1" whose criteria are the level
texts. This is the encoding the published comparison used.

**reference** runs Laya Vision itself (thaitea/laya-vision-201m @ 0b6228f7, CC BY-NC-SA 4.0)
with its own loader and scorer on the same records, and keeps its calibrated probabilities
(softmax(logits / T[question type]), the ones its README reports). Laya does not publish
per-question predictions for these samples, so they are recomputed from its public weights
and code; nothing of Laya's is shipped here. Install its code first:
`pip install -e ~/.cache/rsi-jev-benches/laya-vision` (prep clones it there).

**compare**: accuracy = argmax == label, Wilson 95% CI; paired difference = mean per-item
difference +- 1.96 SE, with an exact two-sided McNemar p; ECE = 15 equal-width bins on the
top probability, each model on the probabilities it serves. "Clean sets" drops the 14
sets whose source training split v4.0-VL's image data drew from (CLEAN_EXCLUDED).

Published: choice and yes/no, clean sets (n 2,516): 0.828 against 0.713, +0.115
(+0.094, +0.136); all types, all sets (6,357): 0.763 against 0.717, +0.046 (+0.032, +0.060);
all types, clean sets (3,716): 0.683 against 0.664, +0.019 (-0.001, +0.038).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                              # noqa: E402

LAYA_MODEL, LAYA_REVISION = "thaitea/laya-vision-201m", "0b6228f7a0762566de1c4539e9aa4eb1c1aef5f4"
LAYA_WEIGHTS_SHA256 = "31220803f38b44b8b1649b207c632612c9a35dee0da2ef1ea35d21e4b424be68"   # model.safetensors

# Sets whose source TRAIN split v4.0-VL's image corpora drew from. The cauldron_* validation
# rows are Laya's 5% re-split of the_cauldron TRAIN rows (the revision our data also read);
# aokvqa, scienceqa and vqav2_yesno are official validation splits of sources we trained on.
CLEAN_EXCLUDED = {
    "aokvqa": "A-OKVQA train", "scienceqa": "ScienceQA train", "vqav2_yesno": "VQAv2 train",
    "cauldron_ai2d": "the_cauldron ai2d", "cauldron_aokvqa": "A-OKVQA train",
    "cauldron_scienceqa": "the_cauldron scienceqa", "cauldron_tqa": "the_cauldron tqa",
    "cauldron_hateful_memes": "the_cauldron hateful_memes", "cauldron_nlvr2": "the_cauldron nlvr2",
    "cauldron_vqav2": "the_cauldron vqav2", "cauldron_chartqa": "the_cauldron chartqa",
    "cauldron_iconqa": "IconQA", "cauldron_visual7w": "Visual7W", "cauldron_ocrvqa": "OCR-VQA",
}


def sets(data: Path) -> list[str]:
    return sorted(n for n in os.listdir(data) if (data / n / "check.json").exists())


def load(data: Path, name: str) -> list[dict]:
    return C.read_jsonl(data / name / "sample.jsonl")


# ---------------------------------------------------------------------------- v4.0-VL
def record_request(rec: dict, base: Path) -> tuple[str, dict, list]:
    """(state, questions, images) for one Laya record."""
    q = rec["question"]
    t = q["type"]
    if t == "choice":
        spec = {"type": "choice", "instructions": q["instructions"], "criteria": {o: None for o in q["criteria"]}}
    elif t == "noul":
        spec = {"type": "noul", "instructions": q["instructions"], "criteria": {"true": "", "false": ""}}
    else:
        spec = {"type": "choice", "instructions": q["instructions"],
                "criteria": {str(i): lv for i, lv in enumerate(q["criteria"])}}
    paths = [rec["image"]] if rec.get("image") else list(rec.get("images") or [])
    return rec.get("state_text") or "", {"q": spec}, [str(base / p) for p in paths][: C.MAX_IMAGES]


def run(client, data: Path, out: Path, only: list[str] | None, limit: int | None) -> None:
    done = set()
    if out.exists():
        done = {(r["dataset"], r["id"]) for r in C.read_jsonl(out)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for name in only or sets(data):
            t0, k = time.time(), 0
            for rec in load(data, name)[: limit or None]:
                if (name, rec["id"]) in done:
                    continue
                state, questions, images = record_request(rec, data / name)
                row = {"dataset": name, "id": rec["id"], "index": rec["index"], "qtype": rec["question"]["type"],
                       "label": rec["label"]}
                try:
                    answers, ms = client.ask(state, questions, images)
                    opts = list(questions["q"]["criteria"]) if questions["q"]["type"] == "choice" else None
                    row.update(probs=C.answer_probs(answers["q"], opts), ms=round(ms, 1))
                except Exception as e:                  # recorded, never silently dropped
                    row.update(probs=None, error=str(e)[:300])
                fh.write(json.dumps(row) + "\n")
                k += 1
            fh.flush()
            print(name, k, f"{time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------- Laya Vision
def reference(data: Path, out: Path, only: list[str] | None, device: str, batch: int) -> None:
    """Laya's own loader and scorer (laya.vlm_train.load_jsonl_examples + collect_logits) on the
    same records, the checkpoint's temperatures for the calibrated probabilities."""
    lv = C.upstream("laya-vision")
    sys.path.insert(0, str(lv))
    import torch
    from huggingface_hub import hf_hub_download
    from laya.vlm import VLMAgent
    from laya.vlm_train import collect_logits, load_jsonl_examples
    import hashlib
    w = hf_hub_download(LAYA_MODEL, "model.safetensors", revision=LAYA_REVISION)
    h = hashlib.sha256(Path(w).read_bytes()).hexdigest()
    if h != LAYA_WEIGHTS_SHA256:
        print(f"!! {LAYA_MODEL}@{LAYA_REVISION}: weights sha256 {h} is not the one compared", flush=True)
    agent = VLMAgent(LAYA_MODEL, revision=LAYA_REVISION, device=device)
    T = list(agent.temperature)
    done = {r["dataset"] for r in C.read_jsonl(out)} if out.exists() else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for name in only or sets(data):
            if name in done:
                continue
            exs = load_jsonl_examples(str(data), name, "sample")
            recs = load(data, name)
            assert len(exs) == len(recs), (name, len(exs), len(recs))
            for rec, r in zip(recs, collect_logits(agent.model, agent.processor, exs, batch_size=batch, num_workers=4)):
                lg = r["logits"].float()
                fh.write(json.dumps({"dataset": name, "id": rec["id"], "index": rec["index"],
                                     "qtype": rec["question"]["type"], "label": rec["label"],
                                     "logits": [round(float(x), 6) for x in lg],
                                     "probs": torch.softmax(lg, -1).tolist(),
                                     "probs_calibrated": torch.softmax(lg / T[r["qtype"]], -1).tolist()}) + "\n")
            fh.flush()
            print(name, len(recs), flush=True)


# ---------------------------------------------------------------------------- comparison
def ece(conf, correct, bins: int = 15) -> float:
    """Laya's ECE: 15 equal-width bins on the top probability, (lo, hi] intervals."""
    import numpy as np
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def mcnemar(a, b) -> tuple[int, int, float]:
    """a, b: 0/1 over the same items -> (a only, b only, exact two-sided p)."""
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    return n10, n01, min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    return m, m - 1.96 * se, m + 1.96 * se, *mcnemar(a, b)


def correct(r, key):
    p = r[key]
    return int(max(range(len(p)), key=p.__getitem__) == r["label"])


def block(G, L, keys):
    g = [correct(G[k], "probs") for k in keys]
    la = [correct(L[k], "probs_calibrated") for k in keys]
    return {"n": len(keys), "ours": sum(g) / len(g), "laya": sum(la) / len(la),
            "ours_ci": wilson(sum(g), len(g)), "laya_ci": wilson(sum(la), len(la)),
            "ours_ece": ece([max(G[k]["probs"]) for k in keys], g),
            "laya_ece": ece([max(L[k]["probs_calibrated"]) for k in keys], la),
            "diff": paired(g, la)}


def compare(ours: Path, laya: Path) -> tuple[list[str], dict]:
    G = {(r["dataset"], r["id"]): r for r in C.read_jsonl(ours) if r.get("probs") is not None}
    L = {(r["dataset"], r["id"]): r for r in C.read_jsonl(laya)}
    keys = [k for k in L if k in G]
    f3 = lambda x: f"{x:.3f}"                                                  # noqa: E731
    fd = lambda x: f"{x:+.3f}"                                                 # noqa: E731
    fp = lambda p: "<.001" if p < 0.001 else f"{p:.3f}"                        # noqa: E731
    out = [f"v4.0-VL items scored: {len(keys)} of {len(L)} Laya items", "",
           "| questions | sets | n | v4.0-VL [95% CI] | Laya Vision [95% CI] | difference [95% CI], McNemar p | ECE v4.0-VL / Laya |",
           "|---|---|---:|---:|---:|---:|---:|"]
    res = {"pooled": {}, "sets": {}}
    for tn, tf in (("all types", lambda k: True), ("choice and yes/no", lambda k: L[k]["qtype"] != "score")):
        for cn, cf in (("all sets", lambda k: True), ("clean sets", lambda k: k[0] not in CLEAN_EXCLUDED)):
            ks = [k for k in keys if tf(k) and cf(k)]
            if not ks:
                continue
            b = block(G, L, ks)
            d = b["diff"]
            out.append(f"| {tn} | {cn} ({len({k[0] for k in ks})}) | {b['n']} | {f3(b['ours'])} [{f3(b['ours_ci'][0])}, "
                       f"{f3(b['ours_ci'][1])}] | {f3(b['laya'])} [{f3(b['laya_ci'][0])}, {f3(b['laya_ci'][1])}] | "
                       f"{fd(d[0])} [{fd(d[1])}, {fd(d[2])}], p {fp(d[5])} | {f3(b['ours_ece'])} / {f3(b['laya_ece'])} |")
            res["pooled"][f"{tn}: {cn}"] = b
    out += ["", "| set | type | trained on its source | n | v4.0-VL | Laya Vision | difference, McNemar p |",
            "|---|---|---|---:|---:|---:|---:|"]
    for name in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == name]
        b = block(G, L, ks)
        d = b["diff"]
        out.append(f"| {name} | {L[ks[0]]['qtype']} | {'yes' if name in CLEAN_EXCLUDED else 'no'} | {b['n']} | "
                   f"{f3(b['ours'])} | {f3(b['laya'])} | {fd(d[0])}, p {fp(d[5])} |")
        res["sets"][name] = b
    return out, res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Laya Vision validation sets: v4.0-VL against Laya Vision")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep", help="rebuild the 34 validation sets and their samples (CPU, downloads)")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--only", help="comma-separated set names")
    p = sub.add_parser("run", help="score v4.0-VL")
    C.add_client_args(p)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("bench-results/laya_v4.jsonl"))
    p.add_argument("--only")
    p.add_argument("--limit", type=int, help="first N records per set (a smoke run)")
    p = sub.add_parser("reference", help="score Laya Vision 201M with its own code")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("bench-results/laya_reference.jsonl"))
    p.add_argument("--only")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=8)
    p = sub.add_parser("compare", help="the tables (CPU)")
    p.add_argument("--ours", type=Path, default=Path("bench-results/laya_v4.jsonl"))
    p.add_argument("--laya", type=Path, default=Path("bench-results/laya_reference.jsonl"))
    p.add_argument("--md", type=Path)
    a = ap.parse_args(argv)
    only = a.only.split(",") if getattr(a, "only", None) else None
    if a.cmd == "prep":
        import laya_prep
        laya_prep.setup(a.data)
        laya_prep.build(only or laya_prep.ALL)
    elif a.cmd == "run":
        client = C.Client(a.model, a.server)
        print(client.name, flush=True)
        run(client, a.data, a.out, only, a.limit)
    elif a.cmd == "reference":
        reference(a.data, a.out, only, a.device, a.batch)
    else:
        lines, res = compare(a.ours, a.laya)
        print("\n".join(lines))
        if a.md:
            a.md.write_text("\n".join(lines) + "\n")
            a.md.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
