"""exit-reason step 0b: does the base LLM's own generative reasoning solve nv7.ho_kcur items the exit scorer misses?

Stratified subset of nv7.ho_kcur (first --per rows of every (task, k) cell; gold is program-verified, LESSONS nv7-gold).
Options rendered as letters; two modes on the same items and model:
  cot    = "reason step by step, put the letter in \\boxed{}" (greedy, --max-tokens)
  direct = "Answer: (" -> first letter of a 4-token greedy continuation (no reasoning)
Writes <out>/<name>.<mode>.items.jsonl and <out>/<name>.nv7.summary.json (acc per mode, per task, per task x k)."""
import argparse, glob, json, os, re, sys, time
from collections import defaultdict

L = "ABCDEFGH"


def snap(model_id):
    d = glob.glob(os.path.join(os.environ["HF_HOME"], "hub", "models--" + model_id.replace("/", "--"), "snapshots", "*"))
    assert len(d) == 1, d
    return d[0]


def boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    dep = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            dep += 1
        elif s[k] == "}":
            dep -= 1
            if dep == 0:
                return s[j + 1:k]
    return None


def cut(text):
    for stop in ("\nQuestion:", "\n\nQuestion", "<|endoftext|>", "<|im_end|>"):
        i = text.find(stop)
        if i >= 0:
            text = text[:i]
    return text


def letter(s, n):
    if s is None:
        return None
    m = re.search(r"\b([A-H])\b", s.replace("\\text", " ").replace("(", " ").replace(")", " "))
    return m.group(1) if m and L.index(m.group(1)) < n else None


def render(r):
    q = r["questions"][0]; opts = [q["criteria"][o] for o in q["options"]]
    body = r["state"].strip() + "\n" + q["instructions"].strip() + "\n" + "\n".join(f"{L[i]}. {t}" for i, t in enumerate(opts))
    gold = max(range(len(opts)), key=lambda i: r["gold"]["answer"][i])
    return body, len(opts), L[gold]


COT = "Question: {q}\nPlease reason step by step, and put the letter of your final answer within \\boxed{{}}.\nAnswer:"
DIRECT = "Question: {q}\nAnswer with the letter of the correct option.\nAnswer: ("


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tower", default=""); ap.add_argument("--per", type=int, default=40); ap.add_argument("--max-tokens", type=int, default=1536)
    a = ap.parse_args()
    cells = defaultdict(list)
    for ln in open(a.data):
        r = json.loads(ln); m = r["meta"]
        if len(cells[(m["task"], m["k"])]) < a.per:
            cells[(m["task"], m["k"])].append(r)
    items = [r for c in sorted(cells, key=lambda c: (c[0], c[1])) for r in cells[c]]
    rend = [render(r) for r in items]
    print(f"nv7 subset {len(items)} items, {len(cells)} cells", flush=True)

    from vllm import LLM, SamplingParams
    base_dir = snap(a.base); mdir = base_dir; drift = None
    if a.tower:  # trained trunk = base with tower layers+norm merged (reason_eval.merge, same as step 0)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from reason_eval import merge
        mdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "nv7cot_%s_%s" % (a.name, os.environ.get("SLURM_JOB_ID", os.getpid())))
        drift = merge(base_dir, a.tower, mdir)
        print(f"merged {a.tower} -> {mdir}; drift {json.dumps(drift)}", flush=True)
    llm = LLM(model=mdir, tokenizer=base_dir, max_model_len=4096, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0, "video": 0},
              enable_prefix_caching=True, max_num_seqs=256, seed=0)
    os.makedirs(a.out, exist_ok=True)
    summ = {"name": a.name, "base": a.base, "tower": a.tower, "drift": drift, "n": len(items), "per": a.per, "prompts": {"cot": COT, "direct": DIRECT}, "modes": {}}
    for mode, tpl, sp in (("direct", DIRECT, SamplingParams(temperature=0.0, max_tokens=4)),
                          ("cot", COT, SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop=["\nQuestion:", "\n\nQuestion"]))):
        t1 = time.time()
        outs = llm.generate([tpl.format(q=b) for b, _, _ in rend], sp)
        agg = defaultdict(lambda: [0, 0]); s = {"n": 0, "ok": 0, "nobox": 0, "trunc": 0, "tok": 0}
        with open(os.path.join(a.out, f"{a.name}.{mode}.items.jsonl"), "w") as fo:
            for r, (b, n, g), o in zip(items, rend, outs):
                txt = o.outputs[0].text
                if mode == "cot":
                    bx = boxed(cut(txt)); p = letter(bx, n); s["nobox"] += bx is None
                else:
                    p = letter(txt[:3], n)
                ok = p == g; t, k = r["meta"]["task"], r["meta"]["k"]
                s["n"] += 1; s["ok"] += ok; s["tok"] += len(o.outputs[0].token_ids); s["trunc"] += o.outputs[0].finish_reason == "length"
                for key in (t, f"{t}:k{k}"):
                    agg[key][0] += ok; agg[key][1] += 1
                fo.write(json.dumps({"case_id": r["case_id"], "task": t, "k": k, "nopt": n, "gold": g, "pred": p, "ok": ok,
                                     "ntok": len(o.outputs[0].token_ids), "text": txt}) + "\n")
        s["acc"] = round(s["ok"] / s["n"], 4); s["tok_per_item"] = round(s["tok"] / s["n"], 1); s["seconds"] = round(time.time() - t1, 1)
        s["by"] = {kk: round(v[0] / v[1], 4) for kk, v in sorted(agg.items())}
        summ["modes"][mode] = s
        print("MODE", mode, json.dumps({k: v for k, v in s.items() if k != "by"}), flush=True)
    json.dump(summ, open(os.path.join(a.out, a.name + ".nv7.summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
