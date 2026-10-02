"""Fit a release's calibration (cal-4b) on the in-distribution holdout of its corpus.

    python scripts/fit_release_calibration.py --ckpt CKPT --dev-corpus CORPUS --out RELEASE_DIR \\
        [--dev-sources a,b,...] [--trained-corpus DIR --trained-sources a,b,...] [--limit]

cal-4b is rsijev.calibrate.calibrate(method="oof_head_scorefloor"), unchanged. What
this script decides is the DEV pool it is fitted on:

- Out-of-fold rows only: a case of --dev-corpus is eligible when it falls in the
  1-in-10 in-distribution holdout, sha256(case_id) % 10 == 0 (run_arm_lib's rule), so
  the weights never trained on it. For v3.0 --dev-corpus is the SFT corpus the parent
  was trained from; an RL child keeps the parent's case ids in its replay, so the same
  cases are held out of the child too.
- Sources default to the checkpoint's spec["sources"]; each source maps to one of
  cal-4b's groups (group_of below). Each group is capped at 1,500 questions, taken in
  file order, and a case is skipped if any of its questions has more options than the
  checkpoint's max_options.
- Leak guard: a held case is dropped when its state text also occurs in a trained
  (non-held) case of the candidate's own corpus (--trained-corpus, default
  --dev-corpus; --trained-sources, default spec["sources"]). An RL pool re-salted out
  of the parent's holdout is the case this catches. v3.0 dropped 534 cases here.

Writes calibration.safetensors + calibration.json into --out (the release directory,
next to tower.safetensors / scorer.safetensors / meta.json). The record names the
corpus and checkpoint by directory name only.

v4.0-VL adds two things, each behind its own flag, so a text release fits exactly
as before:

- --lineage: the leak guard covers every corpus of the checkpoint's lineage -- its
  own spec's corpus_dir + sources, then each init_parent's, read from the parent's
  meta.json -- not just one corpus, so a state trained anywhere upstream under
  another id cannot enter the DEV pool. An RL stage's cal_pool and dev_pool are
  never trained (fit_rl2 trains only its slice, plus replay when replay=true), so
  they are not trained states. Relative corpus_dir / init_parent paths (published
  meta.json files name them so) are resolved under --root.
- --vis-dev-root DIR: an image holdout. The held cases of DIR's vis_*.jsonl
  (vision_v1 for v4.0-VL; never an image eval set) join the pool as group "vis",
  weight --vis-weight (0.15), at most 2,400 questions, in sha256("rcvis:" + id)
  order. A held image case is dropped when its image set was trained (non-held)
  anywhere in the lineage. Images go through the base model's frozen ViT with the
  checkpoint's image-token budget, as served.

v4.0-VL's fit (8,464 DEV questions: 7,213 text + 1,251 image):

    python scripts/fit_release_calibration.py --ckpt asym-calA/s17 --root DATA \
        --dev-corpus DATA/v4-rep-b-s17ck --dev-sources <v4-rep-b-s17ck's spec sources> \
        --lineage --vis-dev-root DATA/vision_v1 --out RELEASE_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GROUP_OF = [("td_train", "td"), ("synth", "synth"), ("mc_replay", "mc"), ("st_kev_hard", "kev_hard"),
            ("st_kev_documents", "kev_docs"), ("st_kev_devtools", "kev_devtools"), ("st_nimble_train", "nimble_train"),
            ("cov_tasksource", "ts"), ("ds2_tasksource", "ts"), ("st_procedural", "synth"), ("cov_open_jev", "synth"),
            ("ds2_open_jev", "synth")]


def held(cid):
    return int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 10 == 0


def group_of(src):
    while src.startswith("rp_"):     # replay of a replay (continued-training chains): rp_rp_<source>
        src = src[3:]
    for p, g in GROUP_OF:
        if src.startswith(p):
            return g
    return "nimble_up" if src.startswith(("st_", "jsg_", "cov_")) else "other"


def trained_states_of(corpus: Path, sources) -> set:
    """States of every non-held case of the candidate's own corpus (image sources skipped)."""
    trained_states = set()
    for s_ in sources:
        f_ = corpus / f"{s_}.jsonl"
        if f_.exists() and not s_.startswith("vis_"):
            for line in open(f_):
                if line.strip():
                    j_ = json.loads(line)
                    if not held(j_["case_id"]):
                        trained_states.add(j_["state"])
    return trained_states


def hk(tag, s):
    return hashlib.sha256((tag + s).encode()).hexdigest()


def _at(root, p):
    p = Path(p)
    return p if p.is_absolute() or root is None else Path(root) / p


def lineage(spec: dict, root=None) -> list:
    """[(spec, corpus_dir, trained sources)] for the checkpoint and each init_parent,
    newest first. An rl2 stage's cal_pool / dev_pool are left out: never trained."""
    chain, sp_ = [], spec
    while sp_:
        r2 = (sp_.get("fit_extra") or {}).get("rl2") or {}
        untrained = {r2.get("cal_pool"), r2.get("dev_pool")} - {None, ""}
        chain.append((sp_, _at(root, sp_["corpus_dir"]),
                      [x for x in sp_["sources"].split(",") if x not in untrained]))
        ipp = (sp_.get("init_parent") or {}).get("path")
        pm = _at(root, ipp) / "meta.json" if ipp else None
        sp_ = json.loads(pm.read_text())["spec"] if pm is not None and pm.exists() else None
    return chain


def lineage_trained_states(chain) -> set:
    """States of every non-held case of every lineage corpus (image sources skipped)."""
    out = set()
    for _, corpus, srcs in chain:
        out |= trained_states_of(corpus, srcs)
    return out


def image_index(roots) -> dict:
    """case_id -> [image paths] over the vision roots' vis_*.jsonl files."""
    idx = {}
    for r in roots:
        for f in sorted(Path(r).glob("vis_*.jsonl")):
            for line in open(f):
                if line.strip():
                    j = json.loads(line)
                    idx[j["case_id"]] = [str(Path(r) / p) for p in (j.get("images") or [])]
    return idx


def lineage_trained_images(chain, img_index) -> set:
    """Image sets (file names, in order) of every non-held image case of the lineage."""
    out = set()
    for sp_, _, _ in chain:
        for s_ in sp_["sources"].split(","):
            f_ = Path(sp_["_corpus"]) / f"{s_}.jsonl"
            if f_.exists() and "vis" in s_:
                for line in open(f_):
                    if line.strip():
                        j_ = json.loads(line)
                        if not held(j_["case_id"]):
                            ims_ = j_.get("images") or img_index.get(j_["case_id"]) or []
                            if ims_:
                                out.add(tuple(Path(p).name for p in ims_))
    return out


def image_dev_pool(vis_root: Path, img_index: dict, trained_imgs: set, cap: int):
    """(case, [image paths], "vis") for the held image cases, <= cap questions."""
    from rsijev.contract import load_cases
    vrows = [c for f in sorted(Path(vis_root).glob("vis_*.jsonl")) for c in load_cases(str(f))
             if held(c.case_id)]
    n0 = len(vrows)
    vrows = [c for c in vrows
             if tuple(Path(p).name for p in img_index.get(c.case_id, [])) not in trained_imgs]
    vleak = n0 - len(vrows)
    vrows.sort(key=lambda c: hk("rcvis:", c.case_id))
    dev_v, nq = [], 0
    for c in vrows:
        if nq >= cap:
            break
        if c.case_id not in img_index:
            raise SystemExit(f"no images for {c.case_id}")
        dev_v.append((c, img_index[c.case_id], "vis"))
        nq += len(c.questions)
    return dev_v, nq, len(vrows), vleak


def collect_images(model, tok, prep, rows, enc_v, *, max_options, device, bs=8) -> dict:
    """calibrate.collect's outputs for image rows: raw logits, the scorer's decision
    state, mode, gold, in the evaluator's conditions (eval mode, canonical order)."""
    import torch
    from PIL import Image
    from rsijev.encode import unpermute_logits
    from rsijev.vision import encode_vision_question, vision_collate
    grab = {}
    hook = model.scorer.register_forward_pre_hook(
        lambda _m, _a, kw: grab.__setitem__("h", kw["decision_h"].detach().float()), with_kwargs=True)
    before = model.cal_mode
    model.cal_mode = "none"
    model.eval()
    Z, H, M, Y, GR, CID, GS = [], [], [], [], [], [], []
    flat = [(c, ims, g, q) for c, ims, g in rows for q in c.questions]
    try:
        with torch.no_grad():
            for i in range(0, len(flat), bs):
                ch = flat[i:i + bs]
                ex, cache = [], {}
                for c, ims, g, q in ch:
                    if c.case_id not in cache:
                        cache[c.case_id] = [Image.open(p).convert("RGB") for p in ims]
                    ex.append(encode_vision_question(tok, prep, c.state, cache[c.case_id], q, enc_v))
                bt = vision_collate(tok, ex, max_options, device=device)
                z = unpermute_logits(model(**bt).float(), bt["option_perm"], bt["option_mask"])
                Z.append(z)
                H.append(grab["h"])
                M.append(bt["mode_id"])
                for c, ims, g, q in ch:
                    gold = c.gold[q.key]
                    Y.append(max(range(len(gold)), key=gold.__getitem__))
                    GS.append(list(gold) + [0.0] * (max_options - len(gold)))
                    GR.append(g)
                    CID.append(c.case_id)
    finally:
        hook.remove()
        model.cal_mode = before
    return {"z": torch.cat(Z), "h": torch.cat(H), "mode": torch.cat(M),
            "y": torch.tensor(Y, device=device), "group": GR, "case_id": CID,
            "gold": torch.tensor(GS, device=device)}


def cat_rows(ds, device) -> dict:
    import torch
    ds = [d for d in ds if d]
    out = {k: torch.cat([d[k].to(device) for d in ds]) for k in ("z", "h", "mode", "y", "gold")}
    for k in ("group", "case_id"):
        out[k] = [x for d in ds for x in d[k]]
    return out


def dev_pool(corpus: Path, srcs, trained_states, max_options: int, cap: int):
    """(case, group) pairs of the held cases, <= cap questions per group, and the leak count."""
    from rsijev.contract import load_cases
    dev_t, per, leak = [], {}, 0
    for s in srcs:
        if s.startswith("vis_"):
            continue
        g = group_of(s)
        for c in load_cases(str(corpus / f"{s}.jsonl")):
            if held(c.case_id) and per.get(g, 0) < cap and all(len(q.options) <= max_options for q in c.questions):
                if c.state in trained_states:
                    leak += 1
                    continue
                dev_t.append((c, g))
                per[g] = per.get(g, 0) + len(c.questions)
    return dev_t, per, leak


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True, help="release checkpoint (meta.json, tower/scorer.safetensors)")
    ap.add_argument("--dev-corpus", required=True, type=Path, help="corpus whose holdout is the DEV pool")
    ap.add_argument("--dev-sources", default=None, help="comma list; default: the checkpoint's spec sources")
    ap.add_argument("--trained-corpus", default=None, type=Path,
                    help="the candidate's own training corpus, for the leak guard (default: --dev-corpus)")
    ap.add_argument("--trained-sources", default=None,
                    help="comma list for the leak guard; default: the checkpoint's spec sources")
    ap.add_argument("--out", required=True, type=Path, help="directory to write calibration.* into")
    ap.add_argument("--lineage", action="store_true",
                    help="leak guard over every corpus of the init_parent chain (v4.0-VL)")
    ap.add_argument("--root", default=None, type=Path,
                    help="directory that relative corpus_dir / init_parent / vision paths are under")
    ap.add_argument("--vis-dev-root", default=None, type=Path,
                    help="image holdout: the held cases of this root's vis_*.jsonl (v4.0-VL: vision_v1)")
    ap.add_argument("--vis-weight", type=float, default=0.15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", action="store_true", help="dry run: 20 questions per group")
    ap.add_argument("--fit-seed", type=int, default=None,
                    help="seed torch (CPU and CUDA) right before the fit; v5.0-VL's calibrator used 0. "
                         "Unset = no reseeding (how v4.0-VL's was fitted)")
    a = ap.parse_args()

    import torch
    import rsijev.calibrate as C
    from load_release import artifact_name, load_release
    from rsijev.calibrate import save_calibration

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    C.GROUP_WEIGHTS.setdefault("other", 0.05)
    dm, tok, enc, meta = load_release(a.ckpt, device)       # fp32 tower
    spec = meta["spec"]
    mo = spec["max_options"]
    srcs = a.dev_sources.split(",") if a.dev_sources else spec["sources"].split(",")
    chain = None
    if a.lineage:
        chain = lineage(spec, a.root)
        for sp_, corpus_, _ in chain:
            sp_["_corpus"] = str(corpus_)
        trained = lineage_trained_states(chain)
        print(f"leak guard: {len(trained)} trained states over {len(chain)} lineage corpora "
              f"{[c.name for _, c, _ in chain]}", flush=True)
    else:
        own_srcs = a.trained_sources.split(",") if a.trained_sources else spec["sources"].split(",")
        trained = trained_states_of(a.trained_corpus or a.dev_corpus, own_srcs)
    dev_t, per, leak = dev_pool(a.dev_corpus, srcs, trained, mo, 20 if a.limit else 1500)
    print(f"DEV: {len(dev_t)} held cases from {a.dev_corpus.name}: {per}; "
          f"dropped {leak} whose state the candidate trained on", flush=True)
    info = {"text_corpus": a.dev_corpus.name, "text_groups_q": per, "dropped_trained_state": leak}
    if chain is not None:
        info["leak_guard_lineage"] = [c.name for _, c, _ in chain]

    model = dm
    if a.fit_seed is not None:
        torch.manual_seed(a.fit_seed)
        torch.cuda.manual_seed_all(a.fit_seed)
        info["fit_seed"] = a.fit_seed
    if a.vis_dev_root is None:
        rep = C.calibrate(dm, tok, dev_t, enc, method="oof_head_scorefloor", max_options=mo, device=device)
    else:
        import dataclasses
        from rsijev.vision import (IMAGE_PAD, ImagePrep, VisionConfig, VisionDecisionModel,
                                   load_visual)
        if chain is None:
            raise SystemExit("--vis-dev-root needs --lineage (the image leak guard walks the lineage)")
        C.GROUP_WEIGHTS["vis"] = a.vis_weight
        # the vision block: the checkpoint's own, else its parent's (an RL child keeps it)
        vb = next(dict((sp_.get("fit_extra") or {}).get("vision"))
                  for sp_, _, _ in chain if (sp_.get("fit_extra") or {}).get("vision"))
        budget = int(vb.get("budget", 1024))
        model = VisionDecisionModel(dm.tower, dm.tower.config.hidden_size, dm.cfg,
                                    visual=load_visual(meta["base_model"]).to(device),
                                    image_token_id=tok.convert_tokens_to_ids(IMAGE_PAD),
                                    vcfg=VisionConfig(image_token_budget=budget))
        model.scorer.load_state_dict(dm.scorer.state_dict())
        model.to(device).eval()
        model.scorer.to(torch.float32)
        prep = ImagePrep(meta["base_model"], VisionConfig(image_token_budget=budget))
        enc_v = dataclasses.replace(enc, max_length=enc.max_length + budget, option_order="canonical")
        roots = vb.get("roots") or vb["root"]
        roots = [_at(a.root, r) for r in (roots if isinstance(roots, list) else [roots])]
        img = image_index(roots)
        trained_imgs = lineage_trained_images(chain, img)
        dev_v, nq, n_held, vleak = image_dev_pool(a.vis_dev_root, img, trained_imgs,
                                                  16 if a.limit else 2400)
        print(f"DEV images: {len(dev_v)} held cases, {nq} q (of {n_held} held); "
              f"dropped {vleak} whose image set was trained", flush=True)
        info.update({"vis_corpus": a.vis_dev_root.name, "vis_held_cases": n_held,
                     "vis_dropped_trained_images": vleak, "vis_dev_q": nq, "vis_weight": a.vis_weight})
        rows_t = C.collect(model, tok, dev_t, enc, max_options=mo, device=device) if dev_t else None
        rows_v = collect_images(model, tok, prep, dev_v, enc_v, max_options=mo, device=device)
        rows = cat_rows([rows_t, rows_v], device)
        orig = C.collect
        C.collect = lambda *x, **k: rows          # calibrate() fits on these rows, unchanged
        try:
            rep = C.calibrate(model, tok, [], enc, method="oof_head_scorefloor", max_options=mo,
                              device=device)
        finally:
            C.collect = orig
    rep.update({"cal_pool": info,
                "cal_fit_by": "scripts/fit_release_calibration.py",
                "cal_source_ckpt": artifact_name(a.ckpt)})
    save_calibration(model, a.out, rep)
    print("cal-4b:", {k: v for k, v in rep.items()
                      if k.startswith(("cal_cv_best", "cal_dev_w", "cal_T_head", "cal_n_dev"))}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
