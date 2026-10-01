"""The five held-out image benchmarks of the v4.0-VL card (section 4.3), through the public API.

Build the set first (CPU; it downloads the public splits at pinned revisions):

    python scripts/build_eval_vision.py --build D/eval_build
    python scripts/build_vision_v1.py --root D/vision_v1                     # only for the decontamination
    python scripts/decontam_vision.py eval --build D/eval_build --train D/vision_v1 --out D/eval_vision_v1

then score it:

    python scripts/vision_benches/heldout.py --eval D/eval_vision_v1 [--model v4.0-vl-2b | --server URL]

`--eval` is the frozen set the card reports (4,173 questions: candidates whose image or
text overlaps the training corpus vision_v1 are dropped). Without vision_v1, score the
candidates directly with `--build D/eval_build`. That set is slightly larger (for example
MMBench 1,000 and POPE 900 candidates, against 992 and 853 kept), so its numbers differ a
little from the card's.

Per benchmark it reports top-1 accuracy, top-label ECE over 15 equal-width bins (the
release's evaluator), and "blank": the same questions with every image replaced by a
grey (128, 128, 128) image of the same size, which shows how much the answers depend on
the picture. The mean is over the five benchmarks, equal weights.

Published (bf16 tower, calibration on): MMBench 0.845, RealWorldQA 0.707, POPE 0.912,
HallusionBench 0.695, InfographicVQA 0.858; mean 0.803, mean ECE 0.069, mean blank 0.458.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                              # noqa: E402

GREY = (128, 128, 128)
ECE_BINS = 15


def bench_files(eval_dir: Path | None, build_dir: Path | None) -> tuple[Path, dict[str, Path]]:
    """(the directory image paths are relative to, {benchmark: case file})."""
    if eval_dir is not None:
        man = json.loads((eval_dir / "manifest.json").read_text())
        out = {}
        for b, m in man["benchmarks"].items():
            f = eval_dir / m["file"]
            if hashlib.sha256(f.read_bytes()).hexdigest() != m["sha256"]:
                raise SystemExit(f"{f}: sha256 differs from manifest.json; the frozen set was modified")
            out[b] = f
        return eval_dir, out
    files = sorted((build_dir / "cand").glob("efvis_*.jsonl"))
    if not files:
        raise SystemExit(f"no cand/efvis_*.jsonl under {build_dir}; run scripts/build_eval_vision.py first")
    return build_dir, {f.stem: f for f in files}


def wire_question(q: dict) -> dict:
    """A contract question (key, mode, instructions, options, criteria) -> its wire spec."""
    crit = q["criteria"]
    if q["mode"] == "noul":
        return {"type": "noul", "instructions": q["instructions"],
                "criteria": {"true": crit["true"], "false": crit["false"]}}
    if q["mode"] == "score":
        return {"type": "score", "instructions": q["instructions"], "criteria": [crit[o] for o in q["options"]]}
    return {"type": "choice", "instructions": q["instructions"], "criteria": {o: crit[o] for o in q["options"]}}


def case_request(case: dict, root: Path, blank: bool = False) -> tuple[str, dict, list]:
    """(state, questions, images) for one case; blank swaps each image for a grey one of its size."""
    from PIL import Image
    images = [str(root / p) for p in case.get("images", [])]
    if blank:
        sizes = []
        for p in images:
            with Image.open(p) as im:
                sizes.append(im.size)
        images = [Image.new("RGB", s, GREY) for s in sizes]
    return case["state"], {q["key"]: wire_question(q) for q in case["questions"]}, images


def score_case(client, case: dict, root: Path, blank: bool) -> list[dict]:
    state, questions, images = case_request(case, root, blank)
    answers, ms = client.ask(state, questions, images)
    rows = []
    for q in case["questions"]:
        p = C.answer_probs(answers[q["key"]], q["options"])
        g = case["gold"][q["key"]]
        gold = max(range(len(g)), key=g.__getitem__)
        pred = max(range(len(p)), key=p.__getitem__)
        rows.append({"key": f"{case['case_id']}|{q['key']}|{'blank' if blank else 'image'}", "case_id": case["case_id"],
                     "qkey": q["key"], "mode": q["mode"], "blank": blank, "probs": p, "pred": pred, "gold": gold,
                     "correct": pred == gold, "ms": round(ms, 2)})
    return rows


def run(client, root: Path, files: dict[str, Path], out: Path, limit: int | None, blank: bool) -> None:
    done = C.done_keys(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for b, f in files.items():
            cases = C.read_jsonl(f)[: limit or None]
            for cond in ((False, True) if blank else (False,)):
                for n, case in enumerate(cases):
                    tag = "blank" if cond else "image"
                    if all(f"{case['case_id']}|{q['key']}|{tag}" in done for q in case["questions"]):
                        continue
                    for r in score_case(client, case, root, cond):
                        fh.write(json.dumps({"bench": b, **r}) + "\n")
                    fh.flush()
                print(b, tag, len(cases), "cases", flush=True)


def analyze(out: Path) -> tuple[list[str], dict]:
    rows = C.read_jsonl(out)
    benches = sorted({r["bench"] for r in rows})
    res, L = {}, ["| benchmark | n | top-1 | ECE (15 bins) | blank top-1 |", "|---|---|---|---|---|"]
    for b in benches:
        img = [r for r in rows if r["bench"] == b and not r["blank"]]
        blk = [r for r in rows if r["bench"] == b and r["blank"]]
        top1 = sum(r["correct"] for r in img) / len(img)
        e = C.ece([max(r["probs"]) for r in img], [float(r["correct"]) for r in img], ECE_BINS)
        bl = sum(r["correct"] for r in blk) / len(blk) if blk else None
        res[b] = {"n": len(img), "top1": top1, "ece": e, "blank_top1": bl}
        L.append(f"| {b} | {len(img)} | {top1:.3f} | {e:.3f} | {'' if bl is None else f'{bl:.3f}'} |")
    if benches:
        mean = lambda k: sum(res[b][k] for b in benches) / len(benches)          # noqa: E731
        bl = [res[b]["blank_top1"] for b in benches]
        res["mean"] = {"top1": mean("top1"), "ece": mean("ece"),
                       "blank_top1": sum(bl) / len(bl) if None not in bl else None}
        L.append(f"| **mean** | {sum(res[b]['n'] for b in benches)} | **{res['mean']['top1']:.3f}** | "
                 f"**{res['mean']['ece']:.3f}** | " +
                 ("" if res["mean"]["blank_top1"] is None else f"{res['mean']['blank_top1']:.3f}") + " |")
    return L, res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score the held-out image set through the public API")
    C.add_client_args(ap)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--eval", type=Path, help="the frozen eval_vision_v1 dir (decontam_vision.py eval)")
    src.add_argument("--build", type=Path, help="the candidates (build_eval_vision.py --build), not decontaminated")
    ap.add_argument("--out", default="bench-results/heldout.jsonl")
    ap.add_argument("--no-blank", action="store_true", help="skip the blank-image control")
    ap.add_argument("--limit", type=int, help="first N cases per benchmark (a smoke run)")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if not a.analyze_only:
        root, files = bench_files(a.eval, a.build)
        client = C.Client(a.model, a.server)
        print(client.name, {b: f.name for b, f in files.items()}, flush=True)
        run(client, root, files, out, a.limit, blank=not a.no_blank)
    lines, res = analyze(out)
    print("\n".join(lines))
    out.with_suffix(".summary.json").write_text(json.dumps(res, indent=1))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
