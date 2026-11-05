"""trace-verify step 1: base-model greedy CoT traces on TRAIN rows (and nv7.ho_kcur) for the trace-reading scorer
and for the exit-reason study. Prompt/parse = nv7_cot.py (COT, boxed, cut, letter; imported, not copied).
One trace per (case_id, key) for mode=choice questions with 2..8 options and prompt <= --max-prompt tokens; the rest are
counted as skipped. `holdout` = sha256(case_id)%10==0 (run_arm never trains these rows; usable as DEV).
Writes <out>/<name>.<tag>.traces.jsonl (resumable: done (case_id,key) skipped) and <out>/<name>.<tag>.summary.json."""
import argparse, glob, hashlib, json, os, sys, time
from collections import defaultdict

sys.path.insert(0, os.environ.get("NV7COT_DIR", os.path.dirname(os.path.abspath(__file__))))
from nv7_cot import COT, L, boxed, cut, letter, snap  # noqa: E402


def holdout(cid):
    return int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 10 == 0


def render_q(r, q):
    opts = [q["criteria"][o] for o in q["options"]]
    body = r["state"].strip() + "\n" + q["instructions"].strip() + "\n" + "\n".join(f"{L[i]}. {t}" for i, t in enumerate(opts))
    g = r["gold"][q["key"]]
    return body, len(opts), L[max(range(len(opts)), key=lambda i: g[i])]


def items(paths, skip):
    for p in paths:
        for ln in open(p):
            r = json.loads(ln)
            fam = r.get("source") or r.get("meta", {}).get("task") or os.path.basename(p)[:-6]
            for q in r["questions"]:
                if q.get("mode", "choice") != "choice" or not (2 <= len(q["options"]) <= len(L)):
                    skip["mode_or_nopt"] += 1; continue
                if (r["case_id"], q["key"]) in skip["_done"]:
                    skip["already"] += 1; continue
                b, n, g = render_q(r, q)
                yield {"case_id": r["case_id"], "key": q["key"], "src": fam, "file": os.path.basename(p), "holdout": holdout(r["case_id"]),
                       "nopt": n, "gold": g, "prompt": COT.format(q=b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--base", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--inputs", nargs="+", required=True, help="jsonl files or dirs of *.jsonl")
    ap.add_argument("--out", required=True); ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--max-prompt", type=int, default=2560); ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N items only"); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    paths = sorted(p for x in a.inputs for p in (glob.glob(os.path.join(x, "*.jsonl")) if os.path.isdir(x) else [x]))
    os.makedirs(a.out, exist_ok=True)
    fo_p = os.path.join(a.out, f"{a.name}.{a.tag}.traces.jsonl")
    done = set()
    if os.path.exists(fo_p):
        for ln in open(fo_p):
            try:
                d = json.loads(ln); done.add((d["case_id"], d["key"]))
            except Exception:
                pass  # torn last line of a killed run
    skip = defaultdict(int); skip["_done"] = done
    its = list(items(paths, skip))
    if a.limit:
        its = its[:a.limit]
    print(f"{a.name}.{a.tag}: {len(paths)} files, {len(its)} todo, done {len(done)}, skip {dict((k, v) for k, v in skip.items() if k != '_done')}", flush=True)
    if a.dry:
        print(its[0]["prompt"] if its else "(none)"); return
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    mdir = snap(a.base); tok = AutoTokenizer.from_pretrained(mdir)
    keep = []
    for it in its:
        if len(tok(it["prompt"]).input_ids) > a.max_prompt:
            skip["long_prompt"] += 1
        else:
            keep.append(it)
    llm = LLM(model=mdir, tokenizer=mdir, max_model_len=4096, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0, "video": 0},
              enable_prefix_caching=True, max_num_seqs=256, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop=["\nQuestion:", "\n\nQuestion"])
    t0 = time.time(); agg = defaultdict(lambda: [0, 0, 0, 0])  # ok, n, nobox, trunc
    with open(fo_p, "a") as fo:
        for i in range(0, len(keep), a.chunk):
            ch = keep[i:i + a.chunk]
            outs = llm.generate([it["prompt"] for it in ch], sp)
            for it, o in zip(ch, outs):
                txt = o.outputs[0].text; bx = boxed(cut(txt)); p = letter(bx, it["nopt"]); tr = o.outputs[0].finish_reason == "length"
                rec = {k: v for k, v in it.items() if k != "prompt"}
                rec.update(pred=p, ok=p == it["gold"], nobox=bx is None, trunc=tr, ntok=len(o.outputs[0].token_ids), text=txt)
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                for k in (it["src"], "ALL"):
                    s = agg[k]; s[0] += rec["ok"]; s[1] += 1; s[2] += rec["nobox"]; s[3] += tr
            fo.flush()
            print(f"  chunk {i // a.chunk}: {i + len(ch)}/{len(keep)} {time.time() - t0:.0f}s acc {agg['ALL'][0] / max(agg['ALL'][1], 1):.3f}", flush=True)
    summ = {"name": a.name, "base": a.base, "tag": a.tag, "prompt": COT, "max_tokens": a.max_tokens, "max_prompt": a.max_prompt, "inputs": paths,
            "skip": {k: v for k, v in skip.items() if k != "_done"}, "seconds": round(time.time() - t0, 1),
            "by_src": {k: {"acc": round(v[0] / v[1], 4), "n": v[1], "nobox": round(v[2] / v[1], 4), "trunc": round(v[3] / v[1], 4)} for k, v in sorted(agg.items())}}
    json.dump(summ, open(os.path.join(a.out, f"{a.name}.{a.tag}.summary.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summ["by_src"].get("ALL")), json.dumps(summ["skip"]), flush=True)


if __name__ == "__main__":
    main()
