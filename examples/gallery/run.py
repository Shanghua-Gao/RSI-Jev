"""Score the demo gallery: every picture and question in questions.json, through the public API.

    pip install "rsi-jev[vision]"
    python examples/gallery/run.py                          # v4.0-vl-2b from the Hub, in-process (Decider)
    python examples/gallery/run.py --model path/to/checkpoint
    python examples/gallery/run.py --server http://127.0.0.1:8000    # against `rsi-jev serve`
    python examples/gallery/run.py --only charts,ui --compare

One request per question, batch 1, as the published run was made, so the latency is per
call. Prints every answer (OK or XX against the reference), then accuracy per category
and overall. --compare also lists the questions whose top answer differs from the
release run in expected.json.

Published (card section 4.3, NVIDIA GB10, bf16 tower, calibration on): 187 of 227
questions right (82%), median 62 ms per call. Per category: road 99/124,
hallucination 26/26, clevr 37/44, charts 13/15, ui 8/12, visa 4/6.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))        # from a clone, uninstalled


def load_items(path: Path = HERE / "questions.json") -> list[dict]:
    return json.loads(path.read_text())["items"]


def item_images(item: dict, root: Path = HERE) -> list:
    """The images as the published run sent them: the file's bytes, or for VisA the
    photo scaled to fit max_side (the VisA harness does this before the model sees it)."""
    if not item.get("max_side"):
        return [str(root / p) for p in item["images"]]
    from PIL import Image
    out = []
    for p in item["images"]:
        s = item["max_side"]
        im = Image.open(root / p)
        im.draft("RGB", (s, s))
        im = im.convert("RGB")
        im.thumbnail((s, s), Image.LANCZOS)
        out.append(im)
    return out


def pick(answer: dict) -> tuple[str, float]:
    """The top answer and its probability, in the questions' own option keys."""
    if answer["type"] == "noul":
        p = answer["noul"]
        return ("true", p) if p > 0.5 else ("false", 1 - p)
    probs = answer["probabilities"]
    k = max(probs, key=probs.get)
    return k, probs[k]


class Client:
    """`Decider` in-process, or the same request over HTTP to `rsi-jev serve`."""

    def __init__(self, model: str | None = None, server: str | None = None):
        self.server = server.rstrip("/") if server else None
        if self.server is None:
            from rsijev import Decider
            self.decider = Decider(model or "v4.0-vl-2b")
            self.name = repr(self.decider)
        else:
            self.name = self.server

    def body(self, state, questions: dict, images: list) -> dict:
        from serve.images import to_data_url
        return {"model": "jev-latest", "state": state, "questions": questions,
                "images": [to_data_url(im) for im in images]}

    def ask(self, state, questions: dict, images: list) -> tuple[dict, float]:
        t = time.perf_counter()
        if self.server is None:
            answers = self.decider.decide(state, questions, images=images)
        else:
            req = urllib.request.Request(self.server + "/v1/systemone",
                                         json.dumps(self.body(state, questions, images)).encode(),
                                         {"Content-Type": "application/json"})
            answers = json.loads(urllib.request.urlopen(req).read())["answers"]
        return answers, (time.perf_counter() - t) * 1000


def run(items: list[dict], client, log=print) -> list[dict]:
    rows = []
    for it in items:
        images = item_images(it)
        for q in it["questions"]:
            answers, ms = client.ask(it["state"], {q["key"]: q["spec"]}, images)
            k, p = pick(answers[q["key"]])
            ok = None if not q.get("ref") else k in q["ref"]
            rows.append({"id": f"{it['id']}/{q['key']}", "category": it["category"], "pick": k, "p": p,
                         "ref": q.get("ref"), "correct": ok, "ms": ms, "answer": answers[q["key"]]})
            mark = {True: "OK", False: "XX", None: "--"}[ok]
            log(f"{mark} {rows[-1]['id']:<52} {k:>15} p={p:.2f}  ref={','.join(q.get('ref') or [])}  {ms:.0f} ms")
    return rows


def summary(rows: list[dict]) -> list[str]:
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["correct"] is None:
            continue
        by[r["category"]][0] += r["correct"]
        by[r["category"]][1] += 1
    out = [f"{c:<14} {k:>3}/{n:<3} {k / n:.0%}" for c, (k, n) in by.items()]
    k, n = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
    out.append(f"{'all':<14} {k:>3}/{n:<3} {k / n:.0%}" if n else "no scored questions")
    if rows:
        out.append(f"median {statistics.median(r['ms'] for r in rows):.0f} ms per call ({len(rows)} calls)")
    return out


def compare(rows: list[dict], path: Path = HERE / "expected.json") -> list[str]:
    exp = json.loads(path.read_text())["answers"]
    diff = [f"  {r['id']}: now {r['pick']} ({r['p']:.2f}), release run {exp[r['id']]['pick']} ({exp[r['id']]['p']:.2f})"
            for r in rows if r["id"] in exp and exp[r["id"]]["pick"] != r["pick"]]
    return [f"{len(diff)} of {len(rows)} top answers differ from the release run"] + diff


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score the RSI-Jev v4.0-VL demo gallery.")
    ap.add_argument("--model", default="v4.0-vl-2b", help="alias, Hub repo id or checkpoint dir (in-process)")
    ap.add_argument("--server", help="URL of a running `rsi-jev serve` (instead of --model)")
    ap.add_argument("--only", help="comma-separated categories: road,hallucination,clevr,charts,ui,visa")
    ap.add_argument("--limit", type=int, help="first N items only")
    ap.add_argument("--compare", action="store_true", help="list answers that differ from expected.json")
    ap.add_argument("--out", help="write every answer to this JSON file")
    a = ap.parse_args(argv)
    items = load_items()
    if a.only:
        items = [i for i in items if i["category"] in a.only.split(",")]
    items = items[: a.limit or None]
    client = Client(a.model, a.server)
    print(client.name, f"- {len(items)} items, {sum(len(i['questions']) for i in items)} questions", flush=True)
    client.ask("<image>", {"warm": {"type": "noul", "instructions": "Is the image blank?"}},
               [str(HERE / "images/generated/count_5.png")])               # one warm-up call, not scored
    rows = run(items, client, log=lambda s: print(s, flush=True))
    print("\n".join(["", *summary(rows)]))
    if a.compare:
        print("\n".join(compare(rows)))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
