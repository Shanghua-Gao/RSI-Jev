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
    src = src[3:] if src.startswith("rp_") else src
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
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", action="store_true", help="dry run: 20 questions per group")
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
    own_srcs = a.trained_sources.split(",") if a.trained_sources else spec["sources"].split(",")
    trained = trained_states_of(a.trained_corpus or a.dev_corpus, own_srcs)
    dev_t, per, leak = dev_pool(a.dev_corpus, srcs, trained, mo, 20 if a.limit else 1500)
    print(f"DEV: {len(dev_t)} held cases from {a.dev_corpus.name}: {per}; "
          f"dropped {leak} whose state the candidate trained on", flush=True)

    rep = C.calibrate(dm, tok, dev_t, enc, method="oof_head_scorefloor", max_options=mo, device=device)
    rep.update({"cal_pool": {"text_corpus": a.dev_corpus.name, "text_groups_q": per, "dropped_trained_state": leak},
                "cal_fit_by": "scripts/fit_release_calibration.py",
                "cal_source_ckpt": artifact_name(a.ckpt)})
    save_calibration(dm, a.out, rep)
    print("cal-4b:", {k: v for k, v in rep.items()
                      if k.startswith(("cal_cv_best", "cal_dev_w", "cal_T_head", "cal_n_dev"))}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
