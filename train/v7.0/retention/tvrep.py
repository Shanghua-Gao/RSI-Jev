#!/usr/bin/env python3
"""Pre-train report on Instruct CoT traces (boxed rate + trace length median
vs base4b ~613 chars; boxed < 50% -> HOLD). Same stats for the base4b reference traces (v2m pool source) side by side.
clean = boxed, not truncated, no <think>/</think> and no chat special token in the text.
usage: tvrep.py <instruct.traces.jsonl> <base4b.traces.jsonl|-> <out.json> [max_ref_rows]
prints the report; exit 3 if the Instruct boxed rate < .50 (HOLD), else 0."""
import json, re, statistics, sys

SPECIAL = re.compile(r"<think>|</think>|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|[a-z_]+\|>")


def stats(path, cap=0):
    n = box = trunc = ok = spec = think = clean = 0
    L_all, L_box, L_clean, ntok = [], [], [], []
    for i, ln in enumerate(open(path)):
        if cap and i >= cap: break
        try: d = json.loads(ln)
        except Exception: continue
        t = d["text"]; n += 1
        b = not d["nobox"]; tr = bool(d["trunc"]); sp = bool(SPECIAL.search(t)); th = "<think>" in t or "</think>" in t
        box += b; trunc += tr; ok += bool(d["ok"]); spec += sp; think += th
        L_all.append(len(t)); ntok.append(d.get("ntok", 0))
        if b: L_box.append(len(t))
        if b and not tr and not sp: clean += 1; L_clean.append(len(t))
    med = lambda x: statistics.median(x) if x else None
    return {"n": n, "boxed_rate": round(box / max(n, 1), 4), "trunc_rate": round(trunc / max(n, 1), 4), "acc": round(ok / max(n, 1), 4),
            "special_tok_rate": round(spec / max(n, 1), 4), "think_tag_rate": round(think / max(n, 1), 4), "clean": clean,
            "clean_rate": round(clean / max(n, 1), 4), "median_chars_all": med(L_all), "median_chars_boxed": med(L_box),
            "median_chars_clean": med(L_clean), "median_ntok": med(ntok),
            "clean_len_q90": (sorted(L_clean)[int(.9 * len(L_clean))] if L_clean else None)}


def main():
    ins, ref, out = sys.argv[1], sys.argv[2], sys.argv[3]
    cap = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    rep = {"instruct": stats(ins), "instruct_file": ins}
    if ref != "-":
        rep["base4b_ref"] = stats(ref, cap); rep["base4b_file"] = ref
    b = rep["instruct"]["boxed_rate"]
    rep["verdict"] = "HOLD (boxed < .50)" if b < .5 else "OK (boxed >= .50)"
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep, indent=1))
    sys.exit(3 if b < .5 else 0)


if __name__ == "__main__":
    main()
