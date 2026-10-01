"""MMStar and BLINK (val): run and analyse. The entry points are mmstar.py and blink.py.

Items, option texts, question stems and state texts come from jev-omni-eval's own
loaders (`src/eval_img.py`: `mmstar_questions`, `blink_questions`), imported from a
checkout at the pinned commit, which also pins the dataset revisions (MMStar
`bc98d668`, BLINK `a3666eb2`). Each item becomes one request: the item's images, its
state text, and one letter-keyed choice question (A, B, ... with the option texts as
criteria). Items with more than four images would be excluded (none are).

The statistics are jev-omni-eval's (`src/analyze.py`): 10-bin top-label ECE, paired
McNemar and bootstrap intervals with that benchmark's resampling unit and seed 0, Holm
within the benchmark. The other systems' per-question predictions are the ones that
repository publishes (`output/results/<bench>__<arm>.json`): A Jev-Omni, B Gemma 4 12B,
C JevDigits. They are read from the checkout, not copied here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                              # noqa: E402

ARMS = {"jev": "A Jev-Omni", "base": "B Gemma 4 12B", "jevdigits": "C JevDigits"}


def eval_img(ev: Path):
    sys.path.insert(0, str(ev / "src"))
    import eval_img as E                                        # noqa: E402  (their loader)
    return E


def items(bench: str, ev: Path):
    E = eval_img(ev)
    return E.blink_questions(E.TASKS) if bench == "blink" else E.mmstar_questions()


def item_request(q: dict) -> tuple[str, dict, list]:
    """(state, questions, images) for one upstream item."""
    return q["state"], {"answer": C.choice_question(q["question"], q["options"])}, q["images"]


def score_item(client, q: dict) -> dict:
    rec = {"key": q["id"], "id": q["id"], "task": q["task"], "l2": q.get("l2"), "n_images": len(q["images"]),
           "n_options": len(q["options"]), "gold": q["gold"]}
    if len(q["images"]) > C.MAX_IMAGES:
        rec["excluded"] = f"{len(q['images'])} images > {C.MAX_IMAGES}"
        return rec
    state, questions, images = item_request(q)
    answers, ms = client.ask(state, questions, images)
    p = C.answer_probs(answers["answer"], list(questions["answer"]["criteria"]))
    pred = max(range(len(p)), key=p.__getitem__)
    rec.update(probs=p, pred=pred, correct=pred == q["gold"], confidence=p[pred], ms=round(ms, 2))
    return rec


def run(bench: str, client, out: Path, ev: Path, limit: int | None = None) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = C.done_keys(out)
    with out.open("a") as fh:
        for n, q in enumerate(items(bench, ev)):
            if limit and n >= limit:
                break
            if q["id"] in done:
                continue
            rec = score_item(client, q)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if n % 50 == 0:
                print(n, rec["id"], "correct" if rec.get("correct") else "wrong", f"{rec.get('ms', 0):.0f} ms", flush=True)
    return out


# ---------------------------------------------------------------------------- analysis
def fmt_p(p):
    return f"{p:.2g}" if p >= 1e-4 else "<1e-4"


def analyze(bench: str, results: Path, ev: Path) -> tuple[list[str], dict]:
    UP = C.load_module("jev_omni_eval_analyze", ev / "src" / "analyze.py")
    theirs, T = {}, {}
    for arm in ARMS:
        f = ev / "output" / "results" / f"{bench}__{arm}.json"
        if f.exists():
            theirs[arm] = {UP.key(r): r for r in json.loads(f.read_text())}
        m = ev / "output" / "results" / f"medium__{arm}.json"
        if m.exists():                                   # their temperature, fit as they fit it
            T[arm] = UP.fit_T(json.loads(m.read_text()))
    ours, excl = {}, []
    for r in C.read_jsonl(results):
        if "excluded" in r:
            excl.append(r["key"])
            continue
        ours[UP.key(r)] = r
    common = set(ours)
    for v in theirs.values():
        common &= set(v)
    keys = sorted(common)
    O = {k: ours[k] for k in keys}
    S = {a: {k: theirs[a][k] for k in keys} for a in theirs}
    res = {"n": len(keys), "n_ours": len(ours), "excluded": excl}
    st = UP.stats(list(ours.values()))
    acc_all = 100 * sum(r["correct"] for r in ours.values()) / max(1, len(ours))
    L = [f"## {bench.upper()}: {len(ours)} items scored, {len(excl)} excluded",
         "",
         f"v4.0-VL as served: accuracy {acc_all:.1f}%, ECE {st['ece']:.3f} (10 bins), mean confidence "
         f"{st['mean_conf']:.3f}, wrong with p >= 0.99: {st['wrong_ge_099']}",
         ""]
    res["ours"] = {"acc": acc_all, **st}
    if not theirs:
        return L + ["(no published predictions found for the other systems)"], res
    blink = bench == "blink"
    L += [f"Paired on the {len(keys)} items every system answered:", "",
          "| system | acc % [95% CI] | " + ("task-macro % [95% CI] | " if blink else "") +
          "ECE | ECE after T (T) | v4.0-VL-only / system-only, McNemar p, difference pp [95% CI] |",
          "|---|---|" + ("---|" if blink else "") + "---|---|---|"]

    def acc_ci(rows):
        dummy = {k: {**r, "correct": False} for k, r in rows.items()}
        p = UP.paired(bench, rows, dummy)
        return p["micro_diff_pp"], p["micro_diff_ci95_pp"], p["native_diff_pp"], p["native_diff_ci95_pp"]

    for name, rows, arm in [("v4.0-VL", O, None)] + [(ARMS[a], S[a], a) for a in ARMS if a in S]:
        mi, mci, na, nci = acc_ci(rows)
        s = UP.stats(list(rows.values()))
        t = T.get(arm) if arm else None
        sc = f"{UP.stats(list(rows.values()), t)['ece']:.3f} ({t})" if t else ""
        cell = ""
        if arm:
            p = UP.paired(bench, O, rows)
            cell = (f"{p['a_only']} / {p['b_only']}, p={fmt_p(p['mcnemar_p'])}, {p['micro_diff_pp']:+.1f} "
                    f"[{p['micro_diff_ci95_pp'][0]:+.1f}, {p['micro_diff_ci95_pp'][1]:+.1f}]")
            if blink:
                cell += (f"; task-macro {p['native_diff_pp']:+.1f} "
                         f"[{p['native_diff_ci95_pp'][0]:+.1f}, {p['native_diff_ci95_pp'][1]:+.1f}]")
            res[f"paired_vs_{arm}"] = p
        L.append(f"| {name} | {mi:.1f} [{mci[0]:.1f}, {mci[1]:.1f}] | " +
                 (f"{na:.1f} [{nci[0]:.1f}, {nci[1]:.1f}] | " if blink else "") + f"{s['ece']:.3f} | {sc} | {cell} |")
        res[arm or "ours_common"] = {"acc": mi, "acc_ci": mci, "native": na, "native_ci": nci, "ece": s["ece"], "T": t}
    if "jev" in S and "base" in S:
        L += ["", f"Per {'subtask' if blink else 'category'} (accuracy %; McNemar p, Holm-adjusted within the benchmark):", "",
              "| " + ("subtask" if blink else "category") + " | n | v4.0-VL | A | B | C | vs A p (Holm) | vs B p (Holm) |",
              "|---|---|---|---|---|---|---|---|"]
        tasks = sorted({S["jev"][k]["task"] for k in keys})
        rows_t, pa, pb = [], [], []
        for t in tasks:
            ks = [k for k in keys if S["jev"][k]["task"] == t]
            acc = lambda R: 100 * sum(R[k]["correct"] for k in ks) / len(ks)          # noqa: E731
            mc = lambda A, B: UP.mcnemar(sum(A[k]["correct"] and not B[k]["correct"] for k in ks),  # noqa: E731
                                         sum(B[k]["correct"] and not A[k]["correct"] for k in ks))
            rows_t.append((t, len(ks), acc(O), acc(S["jev"]), acc(S["base"]),
                           acc(S["jevdigits"]) if "jevdigits" in S else float("nan")))
            pa.append(mc(O, S["jev"]))
            pb.append(mc(O, S["base"]))
        for (t, n, o, a, b, c), x, y in zip(rows_t, UP.holm(pa), UP.holm(pb)):
            L.append(f"| {t} | {n} | {o:.1f} | {a:.1f} | {b:.1f} | {c:.1f} | {fmt_p(x)} | {fmt_p(y)} |")
    return L, res


def main(bench: str, argv=None) -> int:
    ap = argparse.ArgumentParser(description=f"{bench}: score v4.0-VL and compare with the published predictions")
    C.add_client_args(ap)
    ap.add_argument("--out", default=f"bench-results/{bench}.jsonl")
    ap.add_argument("--limit", type=int, help="first N items only (a smoke run)")
    ap.add_argument("--analyze-only", action="store_true", help="skip the model; analyse an existing --out")
    a = ap.parse_args(argv)
    ev = C.upstream("jev-omni-eval")
    out = Path(a.out)
    if not a.analyze_only:
        client = C.Client(a.model, a.server)
        print(client.name, flush=True)
        run(bench, client, out, ev, a.limit)
    lines, res = analyze(bench, out, ev)
    print("\n".join(lines))
    out.with_suffix(".summary.json").write_text(json.dumps(res, indent=1, default=str))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return 0
