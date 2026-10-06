"""Per-exit logits of a multi-exit checkpoint, for the exit-policy selection (v6.0-VL stages 5-6).

One GPU, the tower in fp32 (the checkpoint's own evaluation precision), canonical option
order. For every question and every exit L: the logits of exit L's head (canonical order,
-inf where masked), from ONE pass of the tower (rsijev.adaptive.readout on norm(h_L); the
main exit is checked bitwise against model() on the first batch). Rows are length-sorted
into token-budget batches. Three text modes and one image mode:

  dev     the checkpoint's in-distribution holdout (sha256(case_id) % 10 == 0 of its training
          sources, which run_arm_lib never trains on), at most 1,500 questions per source
          group (rsijev.adaptive.group_of); tag "<group>|cal" or "<group>|tau" by a salted
          case hash. One cal-4b per exit is fitted on the "cal" half (A.fit_cal4b).
          -> OUT/dev_dump.pt (z, decision states h, mode, y, tag, case_id, correct, conf =
          the cal-4b confidence, raw_conf, cals)
  eval    evaluation sets, read only (nothing is fitted): --set NAME=PATH (a case file or a
          directory with manifest.json), --public for the typed-decisions test split and
          MMLU-Pro 1k; tag = the set name; held-out sets are named "final.<name>".
          -> OUT/eval_dump.pt, the same fields with the dev dump's calibrators applied
  cases   every case file under PD/text/<slug>.jsonl (a policy development set); tag
          "<slug>|A" or "<slug>|B" by sha256("policy3-half:" + case_id) % 2.
          -> OUT (z, mode, y, tag, case_id)
  vision  every probe_*.jsonl (or the manifest's benchmarks) under --root, images and all,
          through forward_exits of the vision model. -> OUT (z, bench, gold, mode)

  python scripts/dump_exits.py dev    --ckpt D/heads-s1 --corpus D/image-corpus --out D/dumps/s1
  python scripts/dump_exits.py eval   --ckpt D/heads-s1 --out D/dumps/s1 --public \\
      --set D/eval_suite_v2 --set final=D/eval_final_v2
  python scripts/dump_exits.py cases  --ckpt D/heads-s1 --pd D/policy-dev-v1 --out D/dumps/pdtext.pt
  python scripts/dump_exits.py vision --ckpt D/heads-s1 --root D/policy-dev-v1/vision --out D/dumps/pdvis.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

CAP_PER_GROUP = 1500


def token_budget_batches(lengths: list, tok_budget: int, batch_size: int) -> list:
    """Indices sorted by length (the last row of a batch is its longest), cut so a batch
    holds at most batch_size rows and rows x longest row stays within tok_budget."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[i] > tok_budget or len(cur) >= batch_size:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def gold_index(c, q) -> int:
    g = c.gold[q.key]
    return max(range(len(g)), key=g.__getitem__)


@torch.no_grad()
def collect(model, tok, enc, rows, *, max_options, device, tok_budget, batch_size, want_h=False, t0=None):
    """rows: [(case, question, tag)] -> {"z": {exit: (n, W) canonical logits}, "mode", "y",
    "tag", "case_id", "correct"} (+ "h": {exit: decision states} with want_h)."""
    from rsijev import adaptive as A
    from rsijev.contract import MODES
    from rsijev.encode import collate, encode_question, unpermute_logits
    t0 = t0 or time.time()
    exits = model.exit_indices()
    main_L = exits[-1]
    tm = model._exit_text
    encd = [encode_question(tok, c.state, q, enc) for c, q, _ in rows]
    batches = token_budget_batches([len(e["input_ids"]) for e in encd], tok_budget, batch_size)
    Z = {L: [None] * len(rows) for L in exits}
    H = {L: [None] * len(rows) for L in exits} if want_h else None
    checked = False
    print(f"  {len(rows)} q in {len(batches)} batches (tok budget {tok_budget})", flush=True)
    for bi, ix in enumerate(batches):
        if bi % 100 == 0:
            print(f"  batch {bi}/{len(batches)} ({time.time() - t0:.0f}s)", flush=True)
        b = collate(tok, [encd[i] for i in ix], max_options=max_options, device=device)
        with model.exit_tower():
            o = model.tower(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                            output_hidden_states=True)
        for L in exits:
            h = o.last_hidden_state if L == main_L else tm.norm(o.hidden_states[L])
            z, dh = A.readout(model.exit_scorer(L), h, b, option_pool=model.cfg.option_pool,
                              logit_cap=model.cfg.logit_cap)
            if L == main_L and not checked:
                ref = model(**b)
                fin = torch.isfinite(ref)
                print(f"check: main-exit readout vs model(): bitwise {torch.equal(z, ref)}, "
                      f"max |dz| {float((z[fin] - ref[fin]).abs().max()):.2e}", flush=True)
                checked = True
            z = unpermute_logits(z.float(), b["option_perm"], b["option_mask"])
            for j, i in enumerate(ix):
                Z[L][i] = z[j].cpu()
                if want_h:
                    H[L][i] = dh[j].cpu()
    mode = torch.tensor([MODES.index(q.mode) for _, q, _ in rows])
    y = torch.tensor([gold_index(c, q) for c, q, _ in rows])
    W = max(int(v.shape[-1]) for v in Z[main_L])
    Zt = {L: torch.stack([torch.nn.functional.pad(v, (0, W - v.shape[-1]), value=float("-inf")) for v in vs])
          for L, vs in Z.items()}
    res = {"exits": exits, "z": Zt, "mode": mode, "y": y, "tag": [t for *_, t in rows],
           "case_id": [c.case_id for c, _, _ in rows],
           "correct": {L: (v.masked_fill(~torch.isfinite(v), -1e30).argmax(-1) == y).float() for L, v in Zt.items()}}
    if want_h:
        res["h"] = {L: torch.stack(v) for L, v in H.items()}
    return res


def dev_rows(spec: dict, corpus: Path, limit: int = 0) -> list:
    """The in-distribution holdout, grouped and halved as described above."""
    from rsijev import adaptive as A
    from rsijev.contract import load_cases
    per, rows = {}, []
    for s in spec["sources"].split(","):
        f = corpus / f"{s}.jsonl"
        if not s or not f.exists():
            continue
        g = A.group_of(s)
        for c in load_cases(str(f)):
            if A.hash_int("", c.case_id) % 10 == 0 and per.get(g, 0) < (limit or CAP_PER_GROUP):
                half = "cal" if A.hash_int("adaexit-half:", c.case_id) % 2 == 0 else "tau"
                for q in c.questions:
                    rows.append((c, q, f"{g}|{half}"))
                per[g] = per.get(g, 0) + len(c.questions)
    return rows


def policy_dev_rows(pd: Path, max_options: int, limit: int = 0) -> list:
    from rsijev import adaptive as A
    from rsijev.contract import load_cases
    rows = []
    for f in sorted((pd / "text").glob("*.jsonl")):
        n = 0
        for c in load_cases(str(f)):
            half = "A" if A.hash_int("policy3-half:", c.case_id) % 2 == 0 else "B"
            for q in c.questions:
                if len(q.options) > max_options or (limit and n >= limit):
                    continue
                rows.append((c, q, f"{f.stem}|{half}"))
                n += 1
    return rows


def _load_set(path: Path) -> dict:
    """{name: cases} from a case file, or a directory with manifest.json (sha256-checked)."""
    from rsijev.contract import load_cases
    if path.is_dir():
        man = json.loads((path / "manifest.json").read_text())
        out = {}
        for name, b in man["benchmarks"].items():
            p = path / b["file"]
            if b.get("sha256") and hashlib.sha256(p.read_bytes()).hexdigest() != b["sha256"]:
                raise SystemExit(f"{p.name}: sha256 mismatch")
            out[name] = load_cases(str(p))
        return out
    return {path.stem: load_cases(str(path))}


def _calibrated(R: dict, cals: dict) -> dict:
    from rsijev import adaptive as A
    return {L: A.calibrated_conf(R["z"][L], R["h"][L], R["mode"], cals[L]) for L in R["exits"]}


def _pack(R: dict, cals: dict | None) -> dict:
    raw = {L: torch.softmax(v, -1).max(-1).values for L, v in R["z"].items()}
    return {**{k: R[k] for k in ("exits", "z", "mode", "y", "tag", "case_id", "correct")},
            "h": {L: v.half() for L, v in R["h"].items()},
            "conf": _calibrated(R, cals) if cals else raw, "raw_conf": raw}


@torch.no_grad()
def dump_vision(ckpt: str, root: Path, out: str, *, batch: int, budget: int, limit: int, device: str) -> None:
    from PIL import Image
    from rsijev.contract import Case, Question
    from rsijev.encode import unpermute_logits
    from rsijev.vision import ImagePrep, VisionConfig, encode_vision_question, vision_collate
    from serve.release import load_release
    model, tok, enc, meta = load_release(ckpt, device, infer_dtype=torch.bfloat16)
    if not hasattr(model, "image_embeds"):
        raise SystemExit("vision dump: the checkpoint has no vision block")
    enc.option_order = "canonical"
    enc.max_length = max(enc.max_length, budget + 2048)
    prep = ImagePrep(meta["weights_source"], VisionConfig(image_token_budget=budget),
                     revision=(meta.get("vision") or {}).get("revision"))
    if (root / "manifest.json").exists():
        man = json.loads((root / "manifest.json").read_text())["benchmarks"]
    else:
        man = {f.stem: {"file": f.name} for f in sorted(root.glob("probe_*.jsonl"))}
    dump = {"bench": [], "gold": [], "mode": [], "z": {}}
    for b, m in man.items():
        cases = []
        for line in (root / m["file"]).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                qs = tuple(Question(q["key"], q["mode"], q["instructions"], tuple(q["options"]), q["criteria"])
                           for q in r["questions"])
                cases.append((Case(r["case_id"], r["source"], r["state"], qs,
                                   {k: tuple(v) for k, v in r["gold"].items()}), r.get("images", [])))
        cases = cases[: limit or None]
        for i in range(0, len(cases), batch):
            ex, gold = [], []
            for c, ims in cases[i:i + batch]:
                pil = [Image.open(root / p).convert("RGB") for p in ims]
                for q in c.questions:
                    ex.append(encode_vision_question(tok, prep, c.state, pil, q, enc))
                    gold.append(gold_index(c, q))
            bt = vision_collate(tok, ex, model.cfg.max_options, device=device)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                Z = model.forward_exits(**bt)
            for L, z in Z.items():
                z = unpermute_logits(z.float(), bt["option_perm"], bt["option_mask"]).cpu()
                zz = torch.full((z.shape[0], 160), float("-inf"))
                zz[:, :z.shape[1]] = z
                dump["z"].setdefault(L, []).append(zz)
            dump["bench"] += [b] * len(gold)
            dump["gold"] += gold
            dump["mode"] += bt["mode_id"].tolist() if bt.get("mode_id") is not None else [0] * len(gold)
        print(b, len(cases), "cases", flush=True)
    dump["z"] = {L: torch.cat(v) for L, v in dump["z"].items()}
    dump["gold"], dump["mode"] = torch.tensor(dump["gold"]), torch.tensor(dump["mode"])
    torch.save(dump, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["dev", "eval", "cases", "vision"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="a directory (dev, eval) or a .pt file (cases, vision)")
    ap.add_argument("--corpus", help="dev: the checkpoint's training corpus directory")
    ap.add_argument("--pd", help="cases: a policy development set (PD/text/*.jsonl)")
    ap.add_argument("--root", help="vision: a directory of probe_*.jsonl (or with manifest.json) and images")
    ap.add_argument("--set", action="append", default=[], help="eval: [NAME=]PATH, repeatable")
    ap.add_argument("--public", action="store_true", help="eval: add typed-decisions test and MMLU-Pro 1k")
    ap.add_argument("--batch-size", type=int, default=0, help="default: 32 (text), 16 (vision)")
    ap.add_argument("--tok-budget", type=int, default=0, help="default: 65536 (dev, eval), 16384 (cases)")
    ap.add_argument("--budget", type=int, default=1024, help="vision: image tokens per decision")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: questions per set / group")
    a = ap.parse_args()
    dev_ = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if a.mode == "vision":
        dump_vision(a.ckpt, Path(a.root), a.out, batch=a.batch_size or 16,
                    budget=a.budget, limit=a.limit, device=dev_)
        return 0
    from serve.release import load_release
    model, tok, enc, meta = load_release(a.ckpt, dev_, infer_dtype=None, vision=False)
    spec, mo = meta["spec"], meta["spec"]["max_options"]
    kw = dict(max_options=mo, device=dev_, batch_size=a.batch_size or 32, t0=t0,
              tok_budget=a.tok_budget or (16384 if a.mode == "cases" else 65536))
    print(f"loaded {Path(a.ckpt).name} exits {model.exit_indices()} own_tokens {enc.option_pool_own_tokens} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if a.mode == "cases":
        R = collect(model, tok, enc, policy_dev_rows(Path(a.pd), mo, a.limit), **kw)
        R.pop("correct")
        torch.save(R, a.out)
        return 0
    from rsijev import adaptive as A
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "dev":
        D = collect(model, tok, enc, dev_rows(spec, Path(a.corpus), a.limit), want_h=True, **kw)
        grp = [t.split("|")[0] for t in D["tag"]]
        sel = torch.tensor([t.endswith("|cal") for t in D["tag"]]).nonzero().squeeze(1)
        cals, diag = {}, {}
        for L in D["exits"]:
            r = A.fit_cal4b(D["z"][L][sel], D["h"][L][sel], D["mode"][sel], D["y"][sel],
                            [grp[i] for i in sel.tolist()], [D["case_id"][i] for i in sel.tolist()])
            cals[L], diag[L] = r["cal"], r["diag"]
            print(f"cal-4b exit {L}: {r['diag']}", flush=True)
        torch.save({**_pack(D, cals), "cal_diag": diag,
                    "cals": {L: {k: v.cpu() for k, v in cals[L].items()} for L in D["exits"]}}, out / "dev_dump.pt")
        return 0
    sets = {}
    if a.public:
        from rsijev.targets import load_mmlu_pro_1k, load_typed_decisions
        sets.update({"typed_decisions": load_typed_decisions("test"), "mmlu_pro_1k": load_mmlu_pro_1k()})
    for s in a.set:
        name, _, path = s.rpartition("=")
        loaded = _load_set(Path(path))
        sets.update({(f"{name}." if name else "") + k: v for k, v in loaded.items()})
    rows = []
    for t, cs in sets.items():
        n = 0
        for c in cs:
            for q in c.questions:
                if a.limit and n >= a.limit:
                    break
                rows.append((c, q, t))
                n += 1
    E = collect(model, tok, enc, rows, want_h=True, **kw)
    cals = None
    if (out / "dev_dump.pt").exists():
        cals = torch.load(out / "dev_dump.pt", map_location="cpu", weights_only=False)["cals"]
    torch.save(_pack(E, cals), out / "eval_dump.pt")
    print(f"saved eval_dump.pt ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
