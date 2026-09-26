"""Post-hoc calibration fitted on DEV data inside the model (cal-1 / cal-4).


The calibration pool is DEV only, never a suite or final-set TEST split:
  * td      20% of typed-decisions TRAIN cases (by a salted case-id hash; the x3
            copies share a case id, so a held-out case leaves training entirely)
  * synth   3% of synth, mc 5% of the general-MC replay (same hash rule)
  * caldev:<group>:*   suite TRAIN-split cases, decontaminated against the suite's
            TEST splits, restricted to a held-out dev partition
            (sha256('tool6-dev:'+id) % 10 == 0) that the training corpus excludes
            from its per-source caps.
None of these are trained on. Every group gets a weight that mirrors the suite
benchmark(s) it stands in for, so the fitted calibration targets suite_ece.

Methods (all argmax-preserving: one positive temperature per row):
  temp       one scalar temperature (cal-1 baseline)
  temp_mode  one temperature per mode (cal-1 variant; fitted and logged always)
  oof_head_scorefloor  (cal-4b) oof_head with log tau >= 0 on score rows, in the
             fit and at inference: score rows may soften but never sharpen, so the
             soft-gold score Brier the teacher defines is not traded for top-label ECE
  oof_head_joint (cal-4c) oof_head fitted on top-label BCE + lambda x soft-gold
             Brier on score rows (the Brier term weighted like the BCE, renormalised
             over score rows); no floor. lambda from FitConfig.cal_joint_lambda
  oof_head   a per-input temperature log tau = w . f(x) + b from a linear head on
             features of the same forward pass (arch.cal_features), fitted on the
             held-out ("out-of-fold") predictions of the trained model (cal-4)
The objective is the top-label one the suite reports: weighted binary
cross-entropy of the top-1 confidence against top-1 correctness.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch

from .arch import CAL_LOGT_CLAMP, CAL_PCA_DIM, cal_features
from .contract import Case, MODES
from .encode import EncodeConfig, collate, encode_question, iter_questions, unpermute_logits

# group -> weight; the suite benchmarks each group stands in for, by suite weight
GROUP_WEIGHTS = {
    "td": 0.20,            # typed_decisions
    "nimble_up": 0.25,     # nimble_public 0.15 + jev_style_panel 0.10 (public classification)
    "nimble_train": 0.05,  # nimble_holdout
    "mc": 0.10,            # mmlu_pro_1k
    "kev_hard": 0.16,      # kev_hard_v1 0.08 + kev_transfer_v4 0.08
    "kev_docs": 0.05,      # kev_documents_v1
    "kev_devtools": 0.05,  # kev_devtools_v1
    "synth": 0.14,         # jevbench_public 0.06 + procedural_test 0.04 + open_jev_ood 0.04
}


def _h(tag: str, cid: str) -> int:
    return int(hashlib.sha256((tag + cid).encode()).hexdigest(), 16)


def dev_group(c: Case, *, td_frac: float, synth_frac: float, mc_frac: float) -> str | None:
    """The calibration group of a case, or None if it is a training case."""
    if c.source.startswith("caldev:"):
        return c.source.split(":")[1]
    u = _h("cal-dev:", c.case_id) % 10000 / 10000
    if c.source == "typed_decisions_train":
        return "td" if u < td_frac else None
    if c.source == "synth":
        return "synth" if u < synth_frac else None
    if c.source.startswith("dc_"):
        return "mc" if u < mc_frac else None
    return None


def split_dev(cases: Sequence[Case], **fr) -> tuple[list[Case], list[tuple[Case, str]]]:
    """Dev integrity: a training case that duplicates a caldev case (same case id
    or same state) is withheld from SFT too. A corpus that also trains on the suite
    TRAIN splits (suite-train-1) holds the caldev cases under their original
    st_* source, and calibrating on rows the model was trained on is not DEV."""
    train, dev = [], []
    for c in cases:
        g = dev_group(c, **fr)
        if g is None:
            train.append(c)
        else:
            dev.append((c, g))
    cal_ids = {c.case_id for c, g in dev
               if c.source.startswith("caldev:")}
    cal_states = {c.state for c, g in dev if c.source.startswith("caldev:")}
    kept = [c for c in train if c.case_id not in cal_ids and c.state not in cal_states]
    split_dev.withheld_dups = len(train) - len(kept)
    return kept, dev


@torch.no_grad()
def collect(model, tokenizer, dev: Sequence[tuple[Case, str]], enc: EncodeConfig, *,
            max_options: int, device, batch_size: int = 32) -> dict:
    """Uncalibrated logits + decision hidden state for every dev question, in the
    evaluator's conditions: eval mode, no autocast, canonical option order."""
    enc_c = EncodeConfig(layout=enc.layout, option_pool=enc.option_pool, option_order="canonical")
    rows = [(c, q, g) for c, g in dev for q in c.questions]
    grab = {}

    def _hook(_m, _args, kwargs):
        grab["h"] = kwargs["decision_h"].detach().float()
    hook = model.scorer.register_forward_pre_hook(_hook, with_kwargs=True)
    was_training, mode_before = model.training, model.cal_mode
    model.eval()
    model.cal_mode = "none"
    Z, H, M, Y, W, G, CID, GS = [], [], [], [], [], [], [], []
    try:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            batch = collate(tokenizer, [encode_question(tokenizer, c.state, q, enc_c) for c, q, _ in chunk],
                            max_options=max_options, device=device)
            z = unpermute_logits(model(**batch).float(), batch["option_perm"], batch["option_mask"])
            Z.append(z.float())
            H.append(grab["h"])
            M.append(batch["mode_id"])
            for c, q, g in chunk:
                gold = c.gold[q.key]
                Y.append(max(range(len(gold)), key=gold.__getitem__))
                GS.append(list(gold) + [0.0] * (max_options - len(gold)))
                G.append(g)
                CID.append(c.case_id)
    finally:
        hook.remove()
        model.cal_mode = mode_before
        model.train(was_training)
    return {"z": torch.cat(Z), "h": torch.cat(H), "mode": torch.cat(M),
            "y": torch.tensor(Y, device=device), "group": G, "case_id": CID,
            "gold": torch.tensor(GS, device=device)}


def _weights(groups: list[str], device) -> torch.Tensor:
    n = {}
    for g in groups:
        n[g] = n.get(g, 0) + 1
    present = {g: GROUP_WEIGHTS.get(g, 0.0) for g in n}
    tot = sum(present.values()) or 1.0
    w = torch.tensor([present[g] / tot / n[g] for g in groups], device=device)
    return w / w.sum()


def _top_terms(z: torch.Tensor):
    """d = z - z_max (<= 0, -1e4 on padding) and a mask of the top entry.
    Top index uses the evaluator's rule: the FIRST maximal option."""
    finite = torch.isfinite(z)
    zz = z.masked_fill(~finite, -1e30)
    top = zz.argmax(-1)                                   # first max on ties (sorted scan)
    zmax = zz.gather(1, top[:, None])
    # padding at a large FINITE negative: -inf / tau has a nan gradient in log tau
    d = (z - zmax).masked_fill(~finite, -1e4)
    is_top = torch.zeros_like(finite)
    is_top.scatter_(1, top[:, None], True)
    return d, is_top, top


def _logp_top(d: torch.Tensor, is_top: torch.Tensor, log_t: torch.Tensor):
    """log p_top and log(1 - p_top) under softmax(z / tau), stable at p -> 1."""
    s = d / torch.exp(log_t)[:, None]
    s_all = torch.logsumexp(s, -1)
    s_rest = torch.logsumexp(s.masked_fill(is_top, float("-inf")), -1)
    return -s_all, s_rest - s_all


def _bce(d, is_top, correct, w, log_t):
    lp, lq = _logp_top(d, is_top, log_t)
    lq = torch.nan_to_num(lq, neginf=-1e4)            # a 1-option-mass row
    return -(w * (correct * lp + (1 - correct) * lq)).sum()


def _ece(conf: torch.Tensor, correct: torch.Tensor, w: torch.Tensor | None = None) -> float:
    if w is None:
        w = torch.ones_like(conf)
    w = w / w.sum()
    b = (conf * 15).long().clamp(max=14)
    e = 0.0
    for k in range(15):
        m = b == k
        if m.any():
            e += abs(float((w[m] * (correct[m] - conf[m])).sum()))
    return e


def _grid_logt(d, is_top, correct, w, joint=None) -> float:
    grid = torch.linspace(-2.5, 2.5, 501, device=d.device)
    best, arg = float("inf"), 0.0
    for lt in grid:
        v = float(_bce(d, is_top, correct, w, lt.expand(d.shape[0])) if joint is None
                  else _obj(d, is_top, correct, w, lt.expand(d.shape[0]), joint))
        if v < best:
            best, arg = v, float(lt)
    return arg


def _floor(lt, floor_mask):
    """log tau >= 0 on the masked rows (score rows under oof_head_scorefloor)."""
    if floor_mask is None:
        return lt
    return torch.where(floor_mask, lt.clamp_min(0.0), lt)


def _brier_term(d, lt, gold, sw):
    """Weighted soft-gold Brier of softmax(z / tau) on the rows with sw > 0."""
    if sw is None:
        return 0.0
    p = torch.softmax(d / torch.exp(lt)[:, None], -1)
    return (sw * ((p - gold) ** 2).sum(-1)).sum()


def _obj(d, is_top, correct, w, lt, joint):
    """BCE (weights w, normalised here) + lambda x Brier on score rows."""
    out = _bce(d, is_top, correct, w / w.sum(), lt)
    if joint is not None:
        lam_j, gold, smask = joint
        sw = w * smask
        if float(sw.sum()) > 0:
            out = out + lam_j * _brier_term(d, lt, gold, sw / sw.sum())
    return out


def _fit_head(f, d, is_top, correct, w, lam: float, b0: float, steps: int = 600, floor_mask=None,
              joint=None):
    wv = torch.zeros(f.shape[1], device=f.device, requires_grad=True)
    b = torch.tensor(b0, device=f.device, requires_grad=True)
    opt = torch.optim.Adam([wv, b], lr=0.03)
    for _ in range(steps):
        lt = _floor((f @ wv + b).clamp(-CAL_LOGT_CLAMP, CAL_LOGT_CLAMP), floor_mask)
        loss = _obj(d, is_top, correct, w, lt, joint) + lam * (wv ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return wv.detach(), b.detach()


def calibrate(model, tokenizer, dev, enc: EncodeConfig, *, method: str, max_options: int,
              device, lam_grid=(1e-4, 1e-3, 1e-2, 3e-2, 1e-1), folds: int = 5,
              joint_lambda: float = 0.0) -> dict:
    """Fit calibration on the dev pool, write it into the model's buffers, set
    model.cal_mode = method. Returns a flat dict of scalars for the record."""
    data = collect(model, tokenizer, dev, enc, max_options=max_options, device=device)
    z, h, mode, y, groups = data["z"], data["h"], data["mode"], data["y"], data["group"]
    d, is_top, top = _top_terms(z)
    correct = (top == y).float()
    w = _weights(groups, z.device)
    n = z.shape[0]
    out: dict = {"cal_method": method, "cal_n_dev_q": n}

    # global and per-mode temperatures (always fitted, logged for the per-mode variant)
    lt_g = _grid_logt(d, is_top, correct, w)
    out["cal_T_global"] = round(math.exp(lt_g), 4)
    lt_m = []
    for m in range(len(MODES)):
        sel = mode == m
        lt_m.append(_grid_logt(d[sel], is_top[sel], correct[sel], w[sel] / w[sel].sum())
                    if sel.any() else lt_g)
        out[f"cal_T_{MODES[m]}"] = round(math.exp(lt_m[-1]), 4)
    model.cal_logT.fill_(lt_g)
    model.cal_logT_mode.copy_(torch.tensor(lt_m, device=model.cal_logT_mode.device))

    # confidence-head features and PCA of the decision state (dev only)
    mean = h.mean(0)
    _, _, V = torch.pca_lowrank(h - mean, q=CAL_PCA_DIM, center=False)
    model.cal_pca_mean.copy_(mean)
    model.cal_pca_W.copy_(V[:, :CAL_PCA_DIM])
    f = cal_features(z, h, mode, model.cal_pca_mean, model.cal_pca_W)
    mu = (w[:, None] * f).sum(0)
    sd = ((w[:, None] * (f - mu) ** 2).sum(0)).sqrt().clamp_min(1e-4)
    fs = (f - mu) / sd
    model.cal_feat_mu.copy_(mu)
    model.cal_feat_sd.copy_(sd)

    if method in ("oof_head", "oof_head_scorefloor", "oof_head_joint"):
        fm = (mode == MODES.index("score")) if method == "oof_head_scorefloor" else None
        smask = (mode == MODES.index("score")).float()
        jt = (joint_lambda, data["gold"], smask) if method == "oof_head_joint" else None
        def J(idx):
            return None if jt is None else (jt[0], jt[1][idx], jt[2][idx])
        out["cal_joint_lambda"] = joint_lambda if jt is not None else 0.0
        # K-fold CV over CASES (a case's questions share a state) to pick the ridge
        # strength, and to check the head beats a single temperature at all.
        fold = torch.tensor([_h("cal-fold:", c) % folds for c in data["case_id"]], device=z.device)
        cv = {}
        for lam in ("temp", *lam_grid):
            tot = 0.0
            for k in range(folds):
                tr, te = fold != k, fold == k
                wt = w[tr] / w[tr].sum()
                if lam == "temp":
                    lt = torch.tensor(_grid_logt(d[tr], is_top[tr], correct[tr], wt, joint=J(tr)), device=z.device)
                    lte = _floor(lt.expand(int(te.sum())), None if fm is None else fm[te])
                else:
                    wv, b = _fit_head(fs[tr], d[tr], is_top[tr], correct[tr], wt, lam, lt_g,
                                      floor_mask=None if fm is None else fm[tr], joint=J(tr))
                    lte = _floor((fs[te] @ wv + b).clamp(-CAL_LOGT_CLAMP, CAL_LOGT_CLAMP),
                                 None if fm is None else fm[te])
                tot += (float(_bce(d[te], is_top[te], correct[te], w[te], lte)) if jt is None
                        else float(_obj(d[te], is_top[te], correct[te], w[te], lte, J(te))) * float(w[te].sum()))
            cv[lam] = tot
            out[f"cal_cv_{lam}"] = round(tot, 5)
        best = min(cv, key=cv.get)
        out["cal_cv_best"] = str(best)
        if best == "temp":
            model.cal_w.zero_()
            model.cal_b.fill_(lt_g if jt is None else _grid_logt(d, is_top, correct, w, joint=jt))
        else:
            wv, b = _fit_head(fs, d, is_top, correct, w, best, lt_g, floor_mask=fm, joint=jt)
            model.cal_w.copy_(wv)
            model.cal_b.fill_(float(b))
        lt_all = _floor((fs @ model.cal_w + model.cal_b).clamp(-CAL_LOGT_CLAMP, CAL_LOGT_CLAMP), fm)
        out["cal_T_head_min"] = round(float(torch.exp(lt_all).min()), 4)
        out["cal_T_head_max"] = round(float(torch.exp(lt_all).max()), 4)
    elif method == "temp":
        lt_all = torch.full((n,), lt_g, device=z.device)
    elif method == "temp_mode":
        lt_all = torch.tensor(lt_m, device=z.device)[mode]
    else:
        raise ValueError(f"unknown calibration method {method!r}")

    # in-sample dev diagnostics, per group: accuracy, mean confidence and ECE
    # before and after (fitted on these rows, so optimistic; the suite is the read)
    conf0 = torch.exp(_logp_top(d, is_top, torch.zeros(n, device=z.device))[0])
    conf1 = torch.exp(_logp_top(d, is_top, lt_all)[0])
    for g in sorted(set(groups)):
        sel = torch.tensor([x == g for x in groups], device=z.device)
        out[f"cal_dev_{g}_n"] = int(sel.sum())
        out[f"cal_dev_{g}_acc"] = round(float(correct[sel].mean()), 4)
        out[f"cal_dev_{g}_conf_raw"] = round(float(conf0[sel].mean()), 4)
        out[f"cal_dev_{g}_conf_cal"] = round(float(conf1[sel].mean()), 4)
        out[f"cal_dev_{g}_ece_raw"] = round(_ece(conf0[sel], correct[sel]), 4)
        out[f"cal_dev_{g}_ece_cal"] = round(_ece(conf1[sel], correct[sel]), 4)
        out[f"cal_dev_{g}_T_mean"] = round(float(torch.exp(lt_all[sel]).mean()), 4)
    sc = torch.tensor([g == "td" for g in groups], device=z.device) & (mode == MODES.index("score"))
    if sc.any():
        for tag, lt_ in (("raw", torch.zeros(n, device=z.device)), ("cal", lt_all)):
            p_ = torch.softmax(z[sc] / torch.exp(lt_[sc])[:, None], -1)
            out[f"cal_dev_td_score_brier_{tag}"] = round(float(((p_ - data["gold"][sc]) ** 2).sum(-1).mean()), 5)
    out["cal_dev_wbce_raw"] = round(float(_bce(d, is_top, correct, w, torch.zeros(n, device=z.device))), 5)
    out["cal_dev_wbce_cal"] = round(float(_bce(d, is_top, correct, w, lt_all)), 5)
    out["cal_dev_wece_raw"] = round(_ece(conf0, correct, w), 4)
    out["cal_dev_wece_cal"] = round(_ece(conf1, correct, w), 4)
    model.cal_mode = method
    return out


CAL_BUFFERS = ("cal_logT", "cal_logT_mode", "cal_pca_mean", "cal_pca_W", "cal_feat_mu",
               "cal_feat_sd", "cal_w", "cal_b")


def save_calibration(model, out_dir, report: dict) -> None:
    """calibration.safetensors (the fitted buffers) + calibration.json (cal_mode and
    the fit report), next to a release checkpoint."""
    import json
    from pathlib import Path
    from safetensors.torch import save_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: getattr(model, k).detach().float().contiguous().cpu() for k in CAL_BUFFERS},
              str(out / "calibration.safetensors"))
    (out / "calibration.json").write_text(json.dumps({"cal_mode": model.cal_mode, **report},
                                                     indent=1, default=str) + "\n")
    print(f"    saved calibration ({model.cal_mode}) -> {out}", flush=True)


def load_calibration(model, ckpt_dir) -> str:
    """Put a saved calibration into a DecisionModel built by load_release."""
    import json
    from pathlib import Path
    from safetensors.torch import load_file
    ck = Path(ckpt_dir)
    t = load_file(str(ck / "calibration.safetensors"))
    for k in CAL_BUFFERS:
        getattr(model, k).copy_(t[k].to(getattr(model, k).device))
    model.cal_mode = json.loads((ck / "calibration.json").read_text())["cal_mode"]
    return model.cal_mode
