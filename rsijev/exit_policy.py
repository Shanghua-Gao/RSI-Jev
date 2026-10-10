"""Choosing a multi-exit model's exit policy from per-exit dumps (v6.0-VL stages 5-6). CPU only.

Inputs are the per-exit logits written by scripts/dump_exits.py: {"z": {exit: (n, W)
canonical logits, -inf where masked}, "mode", "y", "tag", "case_id"}. Nothing here runs a
model. scripts/select_exit_policy.py is the command line.

Three steps, in the order v6.0-VL ran them:

  * seed_metric: one temperature per exit fitted by NLL on the checkpoint's own DEV
    "cal" half; the seed-selection metric is the main exit's cal-4b ECE on the "tau"
    half (lower wins; within .002 the first seed). Its main-exit temperature is the one
    the package's calibration.safetensors carries (cal_logT).
  * cascade selection: per-exit temperatures fitted on half A of a policy development set,
    weighted to the evaluation set's question-type mix (EVAL_CELLS); policy families
    (fixed exits, cascades C_t / C16_t, a learned router R, mixtures M, a per-type table
    TT) scored on half B; the binding policy is the best servable one (a fixed exit or a
    cascade with one tau) that keeps text, knowledge and image accuracy within .010 of
    always-last-exit and text ECE <= .07; ties within .002 go to the least compute.
  * auto_thresholds: per-exit thresholds for effort=auto on a 16 -> 20 -> 32 cascade
    (images always at the last exit). A grid of (tau16, tau20) is screened on half A
    (suite-like depth <= 24 layers, suite-like U and held-out accuracy within .003 of
    the last exit), the feasible point with the best Decision-Index-metric U on a
    shift-matched set wins (ties within .002: lower depth, then lower tau16, tau20),
    and it must be confirmed on half B (feasible there and better than the single-tau
    cascade at 0.59). With single=True the same rule runs over one tau shared by both
    early exits (the default for multi-question requests); when no member is confirmed
    it falls back to 0.95, the most conservative member, recorded as not confirmed
    (v6.1-VL).

Before a refit on a new checkpoint (v6.1-VL, a weight average of two fine-tunes), every
development row whose case id, or whose state and first question, appears in a training
corpus of either parent is dropped from every dump: overlap_case_ids finds them,
drop_case_ids removes them.

The halves of every development set are fixed by a hash of the case id
(sha256("policy3-half:" + case_id) % 2: 0 -> A, 1 -> B); the checkpoint's own DEV dump
maps its "cal" half to A and its "tau" half to B.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from . import adaptive as A

NM, W160 = 8, 160
# Question-type counts of the evaluation set: (mode id, option bucket 0: <=2, 1: 3-4,
# 2: 5-9, 3: >=10), counted on its 21,660 rows without labels.
EVAL_CELLS = {(0, 0): 801, (0, 1): 4932, (0, 2): 3680, (0, 3): 1594, (1, 0): 8123, (2, 1): 1352,
              (2, 2): 1070, (2, 3): 108}
TOL_COMP, ECE_MAX, TIE = 0.010, 0.07, 0.002
# The fifteen-benchmark suite's weights (for the one evaluation read).
SUITE_W = {"typed_decisions": 0.20, "nimble_public": 0.121, "mmlu_pro_1k": 0.081, "jev_style_panel": 0.081,
           "kev_transfer_v4": 0.065, "kev_hard_v1": 0.065, "jevbench_public": 0.049, "kev_documents_v1": 0.041,
           "kev_devtools_v1": 0.041, "nimble_holdout": 0.041, "procedural_test": 0.033, "open_jev_ood": 0.032,
           "tasksource_jev_test": 0.05, "semif_external": 0.05, "scienthoon_ood": 0.05}
# Decision Index benchmark id of each development-set slug that is not named "di<id>".
DIMAP = {"acos_v2": 38, "apibank": 3, "banking77_clean": 4, "bbh": 58, "bpomp": 20, "chess_sf": 31, "cladder": 44,
         "clinc_clean": 5, "contractnli_v2": 11, "crux_real": 43, "esci_v2": 37, "forecastbench": 48, "gsm8k": 30,
         "habermas_v2": 50, "hellaswag": 29, "home_sim": 9, "hover_v2": 61, "isarcasm_v2": 40, "mmlupro": 57,
         "newyorker": 64, "nli4ct_v2": 42, "phishnchips": 56, "pop909_real": 22, "ragtruth_v2": 59, "vast_v2": 41,
         "when2call_v2": 62, "winogrande": 28, "home_kitgen": 9, "apibank_kitcat": 3}
# Decision Index metric families (benchmark ids): case-level exact match, macro-F1, review F1.
CASE_EXACT, MACRO_F1, REVIEW_F1 = {1, 9, 33}, {4, 5, 10, 11, 12, 37, 39, 41, 42}, {38}
C_DEPTH = 0.001                       # J = U - C_DEPTH * (depth - 20)


def pad(z):
    z = z.float()
    return torch.nn.functional.pad(z, (0, W160 - z.shape[-1]), value=float("-inf")) if z.shape[-1] < W160 else z[..., :W160]


def clean(z):
    return z.float().masked_fill(~torch.isfinite(z.float()), -1e30)


def half_of(cid: str) -> str:
    return "A" if A.hash_int("policy3-half:", cid) % 2 == 0 else "B"


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- development rows a parent trained on
def row_keys(row: dict) -> tuple:
    """(case id, content key) of a corpus or development row: the content key hashes the
    state and the first question's instructions and options, so a row re-keyed under a new
    case id still matches."""
    st = json.dumps(row.get("state", {}), sort_keys=True)
    qs = row.get("questions") or []
    q0 = json.dumps([(q.get("instructions") or q.get("prompt") or "", q.get("options")) for q in qs][:1], sort_keys=True)
    return row.get("case_id"), hashlib.md5((st + q0).encode()).hexdigest()


def overlap_case_ids(dev_rows, corpus_rows, dev_case_ids=()) -> set:
    """Development case ids that a training corpus contains, by case id or by content.

    dev_rows: development rows (dicts with case_id, state, questions); dev_case_ids: further
    ids (a dump's own DEV rows, matched by id only); corpus_rows: training rows."""
    ids, by_key = set(dev_case_ids), defaultdict(list)
    for r in dev_rows:
        c, k = row_keys(r)
        if c is None:
            continue
        ids.add(c)
        by_key[k].append(c)
    hit = set()
    for r in corpus_rows:
        c, k = row_keys(r)
        if c in ids:
            hit.add(c)
        hit.update(by_key.get(k, ()))
    return hit


def jsonl_rows(root, skip=("dev", "held")):
    """Every dict row with a case_id under root (recursively); files whose name contains
    one of `skip` (a corpus's held-out files) are left out."""
    for f in sorted(Path(root).rglob("*.jsonl")):
        if any(s in f.name for s in skip):
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and "case_id" in r:
                yield r


def drop_case_ids(dump: dict, excluded) -> dict:
    """The dump without the rows whose case_id is excluded: every tensor, list and per-exit
    dict aligned with case_id is cut the same way. A dump without case ids (an image dump)
    is returned as it is."""
    if "case_id" not in dump:
        return dump
    ex = set(excluded)
    cid = list(dump["case_id"])
    n = len(cid)
    keep = [i for i, c in enumerate(cid) if c not in ex]
    ix = torch.tensor(keep, dtype=torch.long)

    def cut(v):
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n:
            return v[ix]
        if isinstance(v, list) and len(v) == n:
            return [v[i] for i in keep]
        if isinstance(v, dict) and v and all((torch.is_tensor(x) and x.dim() > 0 and x.shape[0] == n)
                                             or (isinstance(x, list) and len(x) == n) for x in v.values()):
            return {k: cut(x) for k, x in v.items()}
        return v
    return {k: cut(v) for k, v in dump.items()}


# ---------------------------------------------------------------- temperatures
def logp(z, logT):
    return torch.log_softmax(clean(z) / math.exp(logT), -1)


def fit_T(fn, y, w):
    """argmin over log T in -2..3 (step .01) of the w-weighted NLL; fn(logT) -> log-probs."""
    grid = [round(-2 + 0.01 * i, 2) for i in range(501)]
    return min(grid, key=lambda t: float(-(w * fn(t).gather(1, y[:, None]).squeeze(1)).sum()))


def fit_T_unweighted(z, y):
    """The plain (mean) NLL fit on the same grid."""
    def nll(t):
        p = torch.softmax(clean(z) / math.exp(t), -1)
        return float(-torch.log(p.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12)).mean())
    grid = [round(-2 + 0.01 * i, 2) for i in range(501)]
    return min(grid, key=nll)


def seed_metric(dev: dict) -> dict:
    """Per-exit temperature (NLL on the DEV "cal" half) and the seed-selection metric (the
    main exit's cal-4b ECE on the "tau" half). dev: dump_exits.py's dev_dump.pt."""
    exits = dev["exits"]
    main_L = exits[-1]
    cal_i = torch.tensor([t.endswith("|cal") for t in dev["tag"]]).nonzero().squeeze(1)
    tau_i = torch.tensor([t.endswith("|tau") for t in dev["tag"]]).nonzero().squeeze(1)
    out = {"exits": exits, "n_dev_cal": int(cal_i.numel()), "n_dev_tau": int(tau_i.numel()), "per_exit": {}}
    for L in exits:
        lt = fit_T_unweighted(dev["z"][L][cal_i], dev["y"][cal_i])
        ok = dev["correct"][L][tau_i]
        pT = torch.softmax(clean(dev["z"][L]) / math.exp(lt), -1).max(-1).values[tau_i]
        out["per_exit"][str(L)] = {"logT": lt, "T": round(math.exp(lt), 4),
                                   "dev_tau": {"temp_ece": round(A.ece(pT.float(), ok.float()), 4),
                                               "cal4b_ece": round(A.ece(dev["conf"][L][tau_i].float(), ok.float()), 4),
                                               "acc": round(float(ok.float().mean()), 4), "n": int(ok.numel())}}
    out["sel_metric"] = out["per_exit"][str(main_L)]["dev_tau"]["cal4b_ece"]
    return out


def pick_seed(metrics: dict, tie: float = 0.002) -> str:
    """The seed with the lower selection metric; a seed within `tie` of the first
    listed one does not displace it. The anchor stays on the first listed metric,
    so a clearly better seed later in the list is not blocked by an intermediate
    seed that only ties with the first."""
    names = list(metrics)
    best = names[0]
    floor = metrics[best]["sel_metric"] - tie
    for n in names[1:]:
        if metrics[n]["sel_metric"] < floor:
            best = n
    return best


# ---------------------------------------------------------------- development set v1
def load_v1(dev_dump, text_dump, vis_dump, vis_case_files) -> dict:
    """The policy development set: the checkpoint's DEV dump ("cal" -> A, "tau" -> B) plus
    the held-out proxy text rows (rows already in the DEV dump dropped) and the image slice.
    vis_case_files: the image dump's case files, in dump order (case ids by question)."""
    D, P, V = (load(x) if not isinstance(x, dict) else x for x in (dev_dump, text_dump, vis_dump))
    EX = sorted(int(x) for x in D["z"])
    LAST = EX[-1]
    assert sorted(int(x) for x in P["z"]) == EX and sorted(int(x) for x in V["z"]) == EX
    dc = set(D["case_id"])
    k = torch.tensor([c not in dc for c in P["case_id"]])     # proxy rows already in the dump DEV
    ndup = int((~k).sum())
    P = {"z": {L: P["z"][L][k] for L in EX}, "mode": P["mode"][k], "y": P["y"][k],
         "tag": [t for t, kk in zip(P["tag"], k.tolist()) if kk]}
    tx = {"z": {L: torch.cat([pad(D["z"][L]), pad(P["z"][L])]) for L in EX},
          "mode": torch.cat([D["mode"], P["mode"]]).long(), "y": torch.cat([D["y"], P["y"]]).long(),
          "half": [("A" if t.endswith("|cal") else "B") for t in D["tag"]] + [t.split("|")[1] for t in P["tag"]],
          "src": ["dev:" + t.split("|")[0] for t in D["tag"]] + ["pd:" + t.split("|")[0] for t in P["tag"]]}
    vcid = []
    for f in vis_case_files:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                vcid += [r["case_id"]] * len(r["questions"])
    assert len(vcid) == len(V["gold"]), (len(vcid), len(V["gold"]))
    vs = {"z": {L: pad(V["z"][L]) for L in EX}, "mode": V["mode"].long(), "y": V["gold"].long(),
          "half": [half_of(c) for c in vcid], "src": ["vis:" + b for b in V["bench"]]}
    v1 = {"EX": EX, "LAST": LAST, "tx": tx, "vs": vs, "ndup": ndup, "H": {}}
    for h in "AB":
        t, v = _sub(tx, h, EX), _sub(vs, h, EX)
        wt, tcells = tweights(t, LAST)
        kn = torch.tensor([s_ == "pd:mmlupro" for s_ in t["src"]])
        v1["H"][h] = {"t": t, "v": v, "wt": wt, "tcells": tcells, "wv": vweights(v), "kn": kn,
                      "wk": kn.float() / max(1, int(kn.sum()))}
    return v1


def _sub(d, h, EX):
    ix = torch.tensor([x == h for x in d["half"]]).nonzero().squeeze(1)
    return {"z": {L: d["z"][L][ix] for L in EX}, "mode": d["mode"][ix], "y": d["y"][ix],
            "src": [d["src"][i] for i in ix.tolist()]}


def cells(d, LAST):
    nop = torch.isfinite(d["z"][LAST]).sum(-1)
    ob = torch.bucketize(nop, torch.tensor([2, 4, 9]))
    return [(int(m), int(o)) for m, o in zip(d["mode"].clamp(0, NM - 1), ob)]


def tweights(d, LAST):
    """Row weights that give each question-type cell its evaluation-set share (no labels)."""
    c = cells(d, LAST)
    n = {}
    for x in c:
        n[x] = n.get(x, 0) + 1
    tot = sum(EVAL_CELLS[x] for x in n if x in EVAL_CELLS)
    w = torch.tensor([EVAL_CELLS.get(x, 0) / tot / n[x] for x in c])
    return w / w.sum(), {str(k): v for k, v in n.items()}


def vweights(d):
    """Equal weight per image source."""
    n = {}
    for s in d["src"]:
        n[s] = n.get(s, 0) + 1
    w = torch.tensor([1.0 / len(n) / n[s] for s in d["src"]])
    return w / w.sum()


def fit_temperatures(v1) -> dict:
    """{exit: logT} fitted on text half A, EVAL-type weighted NLL."""
    tA, wA = v1["H"]["A"]["t"], v1["H"]["A"]["wt"]
    return {L: fit_T(lambda t, L=L: logp(tA["z"][L], t), tA["y"], wA) for L in v1["EX"]}


def wece(conf, ok, w, bins=15):
    e = 0.0
    b = torch.clamp((conf * bins).long(), max=bins - 1)
    for k in range(bins):
        m = b == k
        if m.any():
            e += float(w[m].sum()) * abs(float((w[m] * ok[m]).sum() / w[m].sum()) - float((w[m] * conf[m]).sum() / w[m].sum()))
    return e / float(w.sum())


def cascade(P: dict, tau: float, exits: list, last: int) -> torch.Tensor:
    """Answer exit per row: the first of `exits` whose top probability (P[L], calibrated
    distributions) reaches tau, else `last`."""
    n = P[last].shape[0]
    ch = torch.full((n,), last)
    und = torch.ones(n, dtype=torch.bool)
    for L in exits:
        s = und & (P[L].max(-1).values >= tau)
        ch[s] = L
        und &= ~s
    return ch


# ---------------------------------------------------------------- cascade selection
def select_cascade(v1: dict, eval_dump=None, eval_vis_dump=None, seed: int = 17) -> dict:
    """Fit on half A, select on half B; with the evaluation dumps, ONE read of the binding
    policy, the unrestricted winner and the fixed exits (report only)."""
    torch.manual_seed(seed)
    EX, LAST, H = v1["EX"], v1["LAST"], v1["H"]
    tA, wA = H["A"]["t"], H["A"]["wt"]
    cov = {"eval_mass_covered": round(sum(EVAL_CELLS[tuple(int(x) for x in k.strip("()").split(","))] for k in H["B"]["tcells"]
                                          if tuple(int(x) for x in k.strip("()").split(",")) in EVAL_CELLS) / sum(EVAL_CELLS.values()), 4),
           "cells_B": H["B"]["tcells"], "kn_B": int(H["B"]["kn"].sum()), "n_text": {h: len(H[h]["t"]["y"]) for h in "AB"},
           "n_vis": {h: len(H[h]["v"]["y"]) for h in "AB"}, "dup_dropped": v1["ndup"]}
    T = fit_temperatures(v1)

    def view(d):
        Z = d["z"]
        Pp = {L: torch.softmax(clean(Z[L]) / math.exp(T[L]), -1) for L in EX}
        return {"Z": Z, "P": Pp, "am": {L: clean(Z[L]).argmax(-1) for L in EX}, "mode": d["mode"].clamp(0, NM - 1),
                "nop": torch.isfinite(Z[LAST]).sum(-1), "y": d.get("y")}

    VA = {h: {"t": view(H[h]["t"]), "v": view(H[h]["v"])} for h in "AB"}

    def feats(v):
        f = []
        for L in EX:
            p = v["P"][L]
            t2 = p.topk(2, -1).values
            ent = -(p * p.clamp_min(1e-12).log()).sum(-1)
            f += [t2[:, 0], t2[:, 0] - t2[:, 1], ent]
        for L in EX[:-1]:
            f.append((v["am"][L] == v["am"][LAST]).float())
        mo = torch.nn.functional.one_hot(v["mode"], NM).float()
        ob = torch.nn.functional.one_hot(torch.bucketize(v["nop"], torch.tensor([2, 4, 9])), 4).float()
        return torch.cat([torch.stack(f, -1), mo, ob], -1)

    FA = feats(VA["A"]["t"])
    MU, SD = FA.mean(0), FA.std(0).clamp_min(1e-6)

    def nz(F):
        return (F - MU) / SD
    RT = {}
    for L in EX:   # router R: per-exit logistic p(correct_L), text A weighted
        X = nz(FA)
        y = (VA["A"]["t"]["am"][L] == tA["y"]).float()
        m = torch.nn.Linear(X.shape[1], 1)
        opt = torch.optim.LBFGS(m.parameters(), max_iter=300)

        def cl(m=m, X=X, y=y, opt=opt):
            opt.zero_grad()
            l_ = (wA * torch.nn.functional.binary_cross_entropy_with_logits(m(X).squeeze(1), y, reduction="none")).sum() \
                + 1e-3 * m.weight.pow(2).sum()
            l_.backward()
            return l_
        opt.step(cl)
        RT[L] = m

    def fit_mix(Ls):
        a = torch.zeros(len(Ls), requires_grad=True)
        opt = torch.optim.Adam([a], lr=.05)
        PA = torch.stack([VA["A"]["t"]["P"][L] for L in Ls])
        y = tA["y"]
        for _ in range(400):
            w = torch.softmax(a, 0)
            p = (w[:, None, None] * PA).sum(0)
            l_ = -(wA * p.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12).log()).sum()
            opt.zero_grad()
            l_.backward()
            opt.step()
        w = torch.softmax(a, 0).detach()
        p = (w[:, None, None] * PA).sum(0)
        return w, fit_T(lambda t: torch.log_softmax(p.clamp_min(1e-12).log() / math.exp(t), -1), y, wA)

    MIX = {"M_all": (EX, *fit_mix(EX)), "M_deep": ([L for L in EX if L >= 16], *fit_mix([L for L in EX if L >= 16]))}
    # type table TT: per EVAL cell, the exit with the best A accuracy if its gain over LAST is >= 2 SE (n >= 50), else LAST
    cA = cells(H["A"]["t"], LAST)
    TAB = {}
    for c in set(cA):
        m = torch.tensor([x == c for x in cA])
        best = (LAST, 0.0)
        for L in EX[:-1]:
            d = ((VA["A"]["t"]["am"][L] == tA["y"]).float() - (VA["A"]["t"]["am"][LAST] == tA["y"]).float())[m]
            g = float(d.mean())
            se = float(d.std() / math.sqrt(len(d))) if len(d) > 1 else 1.0
            if len(d) >= 50 and g >= 2 * se and g > best[1]:
                best = (L, g)
        TAB[c] = best[0]

    def choose(v, pol):
        kind, arg = pol
        n = v["mode"].shape[0]
        if kind == "fixed":
            return torch.full((n,), arg)
        if kind == "C":
            t, dec = arg
            return cascade(v["P"], t, dec, LAST)
        if kind == "R":
            S = torch.stack([torch.sigmoid(RT[L](nz(feats(v))).squeeze(1)).detach() - arg * L / 32 for L in EX])
            return torch.tensor(EX)[S.argmax(0)]
        if kind == "TT":
            ob = torch.bucketize(v["nop"], torch.tensor([2, 4, 9]))
            return torch.tensor([TAB.get((int(m), int(o)), LAST) for m, o in zip(v["mode"], ob)])
        return None

    def evaluate(v, pol):
        """-> pred, conf, answer exit (compute = answer exit; mixture = LAST)"""
        if pol[0] == "M":
            Ls, w, tm = MIX[pol[1]]
            p = (w[:, None, None] * torch.stack([v["P"][L] for L in Ls])).sum(0)
            c, pred = torch.softmax(p.clamp_min(1e-12).log() / math.exp(tm), -1).max(-1)
            return pred, c, torch.full((len(pred),), LAST)
        ch = choose(v, pol)
        ar = torch.arange(len(ch))
        ix = [EX.index(int(x)) for x in ch]
        return (torch.stack([v["am"][L] for L in EX])[ix, ar], torch.stack([v["P"][L].max(-1).values for L in EX])[ix, ar], ch)

    POL = {f"fixed{L}": ("fixed", L) for L in EX}
    POL.update({f"R_lam{lam}": ("R", lam) for lam in (0, .005, .01, .02, .05, .1)})
    POL.update({k: ("M", k) for k in MIX})
    POL["TT"] = ("TT", None)
    for t in [round(.51 + .01 * i, 2) for i in range(49)]:
        POL[f"C_t{t}"] = ("C", (t, EX[:-1]))
        POL[f"C16_t{t}"] = ("C", (t, [L for L in EX[:-1] if L >= 16]))

    def servable(k):   # a fixed last exit, or a cascade with one tau and per-exit temperatures
        return k == f"fixed{LAST}" or k.startswith(("C_t", "C16_t"))
    dev = {}
    for k, pol in POL.items():
        r = {}
        for h in "AB":
            vt, vv = VA[h]["t"], VA[h]["v"]
            w = H[h]["wt"]
            pt, ct, cht = evaluate(vt, pol)
            okt = (pt == vt["y"]).float()
            pv, _, chv = evaluate(vv, pol)
            okv = (pv == vv["y"]).float()
            r[h] = {"TXT": round(float((w * okt).sum()), 4), "KN": round(float((H[h]["wk"] * okt).sum()), 4),
                    "VIS": round(float((H[h]["wv"] * okv).sum()), 4), "ECE_TXT": round(wece(ct, okt, w), 4),
                    "compute": round(float((w * cht.float()).sum()), 2),
                    "vis_compute": round(float((H[h]["wv"] * chv.float()).sum()), 2)}
            r[h]["U"] = round((r[h]["TXT"] + r[h]["KN"] + r[h]["VIS"]) / 3, 4)
        dev[k] = r
    ref = dev[f"fixed{LAST}"]["B"]

    def feasible(k):
        b = dev[k]["B"]
        return b["ECE_TXT"] <= ECE_MAX and all(b[c] >= ref[c] - TOL_COMP for c in ("TXT", "KN", "VIS"))

    def pick(keys):
        ok_ = [k for k in keys if feasible(k)]
        if not ok_:
            return f"fixed{LAST}"
        top = max(dev[k]["B"]["U"] for k in ok_)
        return min([k for k in ok_ if dev[k]["B"]["U"] >= top - TIE], key=lambda k: (dev[k]["B"]["compute"], k))

    BIND, WIN = pick([k for k in POL if servable(k)]), pick(list(POL))
    res = {"T": T, "coverage": cov, "table": {str(k): v for k, v in TAB.items()},
           "mix": {k: {"exits": v[0], "w": [round(float(x), 4) for x in v[1]], "logT": v[2]} for k, v in MIX.items()},
           "rule": {"TOL_COMP": TOL_COMP, "ECE_MAX": ECE_MAX, "TIE": TIE}, "dev": dev, "binding": BIND, "winner_any": WIN}
    pb = POL[BIND]
    res["serve"] = {"exits": EX, "tau": (pb[1][0] if pb[0] == "C" else 1.01), "logT": {str(L): T[L] for L in EX},
                    "skip_exit_logT": ({str(L): 10.0 for L in EX[:-1] if pb[0] == "C" and L not in pb[1][1]})}
    if eval_dump is not None:
        E = load(eval_dump) if not isinstance(eval_dump, dict) else eval_dump
        vE = view({"z": {L: pad(E["z"][L]) for L in EX}, "mode": E["mode"].long(), "y": E["y"].long()})
        tags = list(E["tag"])
        fin = torch.tensor([t.startswith("final.") for t in tags])
        TI = {t: torch.tensor([x == t for x in tags]) for t in set(tags)}
        VE = load(eval_vis_dump) if not isinstance(eval_vis_dump, dict) else eval_vis_dump
        vV = view({"z": {int(k): pad(v) for k, v in VE["z"].items()}, "mode": VE["mode"].long(), "y": VE["gold"].long()})
        ev = {}
        for k in [f"fixed{L}" for L in EX] + list(dict.fromkeys([BIND, WIN])):
            pred, conf, ch = evaluate(vE, POL[k])
            ok = (pred == vE["y"]).float()
            per = {t: float(ok[m].mean()) for t, m in TI.items()}
            sx = {s: per[s] for s in SUITE_W if s in per and s != "open_jev_ood"}
            pv, _, chv = evaluate(vV, POL[k])
            okv = (pv == vV["y"]).float()
            bench = VE["bench"]
            pbm = {b: round(float(okv[torch.tensor([x == b for x in bench])].mean()), 4) for b in sorted(set(bench))}
            ev[k] = {"suite_x_ood": round(sum(SUITE_W[s] * v for s, v in sx.items()) / sum(SUITE_W[s] for s in sx), 4),
                     "final": round(float(ok[fin].mean()), 4), "mmlu_pro": round(per.get("mmlu_pro_1k", float("nan")), 4),
                     "final_ece": round(A.ece(conf[fin], ok[fin]), 4), "suite_ece": round(A.ece(conf[~fin], ok[~fin]), 4),
                     "compute_all": round(float(ch.float().mean()), 2), "compute_final": round(float(ch[fin].float().mean()), 2),
                     "exit_share": {L: round(float((ch == L).float().mean()), 4) for L in EX},
                     "vision": round(sum(pbm.values()) / len(pbm), 4), "vision_per": pbm,
                     "vision_compute": round(float(chv.float().mean()), 2)}
        res["eval"] = ev
    return res


# ---------------------------------------------------------------- per-exit thresholds (effort=auto)
def bench_weights(diag: dict) -> dict:
    """Decision Index benchmark weights from a per-benchmark diagnostic of the checkpoint:
    {"bench": {"<id>": {"skill": {"exit32": s}, "DI_contrib": {"exit32": c}}}} ->
    {id: c / 100 / s} (the benchmark's share of the index per unit of skill; 0 when the
    skill is 0)."""
    WB = {}
    for b, o in diag["bench"].items():
        s = o["skill"]["exit32"]
        c = o.get("DI_contrib", {}).get("exit32")
        WB[int(b)] = (c / 100 / s) if (c and s) else 0.0
    return WB


class Thresholds:
    """The effort=auto threshold search. T: {exit: logT} (the cascade selection's); WB: bench_weights();
    CHANCE: {benchmark id: chance skill} from the Decision Index's index file."""

    G16 = [.59, .65, .70, .75, .80, .85, .90, .95]
    G20 = [.50, .59, .70, .80]
    # single=True: one tau at both early exits (the union of the two grids), and the
    # member taken when the A winner is not confirmed on B.
    G_SINGLE = [.50, .59, .65, .70, .75, .80, .85, .90, .95]
    FALLBACK = .95
    DCAP, TOL, TIE = 24.0, .003, .002

    def __init__(self, v1: dict, T: dict, WB: dict, CHANCE: dict, single: bool = False):
        self.v1, self.T, self.WB, self.CHANCE = v1, {int(k): float(v) for k, v in T.items()}, WB, CHANCE
        self.EX = v1["EX"]
        self.EXT = torch.tensor(self.EX, dtype=torch.float)
        self.single = single
        if single:
            self.grid = [{"kind": "pm", "tau": {16: t, 20: t}} for t in self.G_SINGLE]
        else:
            self.grid = [{"kind": "pm", "tau": {16: a, 20: b}} for a in self.G16 for b in self.G20]
        self.pols = {"fixed16": {"kind": "fixed", "L": 16}, "fixed20": {"kind": "fixed", "L": 20},
                     "fixed32": {"kind": "fixed", "L": 32}}
        self.pols.update({self.name(p): p for p in self.grid})
        self.c16 = "t16=0.59,t20=0.59"

    @staticmethod
    def name(p):
        return f"t16={p['tau'][16]},t20={p['tau'][20]}"

    # ---- per-row quantities
    def pack(self, d):
        EX, LAST = self.EX, self.EX[-1]
        Z = d["z"]
        K = torch.isfinite(Z[LAST]).sum(-1)
        Pp = {L: torch.softmax(clean(Z[L]) / math.exp(self.T[L]), -1) for L in EX}
        pm = torch.stack([Pp[L].max(-1).values for L in EX])
        pr = torch.stack([clean(Z[L]).argmax(-1) for L in EX])
        Kf = K.clamp_min(2).float()
        return {"ok": (pr == d["y"][None]).float(), "pm": pm, "pr": pr, "K": Kf,
                "kb": torch.bucketize(K, torch.tensor([2, 4, 10])), "n": pm.shape[1]}

    def route(self, pk, pol):
        """Answer exit (index into EX) per row: fixed, or the per-exit-tau cascade on the
        calibrated top probability, else the last exit."""
        EX = self.EX
        n = pk["n"]
        if pol["kind"] == "fixed":
            return torch.full((n,), EX.index(pol["L"]), dtype=torch.long)
        ch = torch.full((n,), EX.index(EX[-1]), dtype=torch.long)
        und = torch.ones(n, dtype=torch.bool)
        for L in (12, 16, 20):
            t = pol["tau"].get(L)
            if t is None:
                continue
            i = EX.index(L)
            s = und & (pk["pm"][i] >= float(t))
            ch[s] = i
            und &= ~s
        return ch

    # ---- Decision-Index-metric sets
    def load_di(self, dumps, pd_dirs, drop_slugs=()):
        meta = {}
        for d in pd_dirs:
            for f in sorted((Path(d) / "text").glob("*.jsonl")):
                for line in open(f):
                    if line.strip():
                        r = json.loads(line)
                        meta[r["case_id"]] = r["questions"]
        EX = self.EX
        zs, ys, bs, cids, labs, slugs, opts = {L: [] for L in EX}, [], [], [], [], [], []
        for p in dumps:
            X = load(p) if not isinstance(p, dict) else p
            keep = []
            seen = defaultdict(int)
            for i, (t, c) in enumerate(zip(X["tag"], X["case_id"])):
                s = t.split("|")[0]
                j = seen[c]
                seen[c] += 1
                if s in drop_slugs:
                    continue
                q = meta[c][j]
                keep.append(i)
                b = int(s[2:]) if s.startswith("di") and s[2:].isdigit() else DIMAP.get(s, -1)
                bs.append(b)
                cids.append(c)
                slugs.append(s)
                labs.append([json.dumps(q["criteria"].get(o, o), sort_keys=True, ensure_ascii=False) for o in q["options"]])
                opts.append(list(q["options"]))
            ix = torch.tensor(keep, dtype=torch.long)
            for L in EX:
                zs[L].append(pad(X["z"][L][ix]))
            ys.append(X["y"].long()[ix])
        d = {"z": {L: torch.cat(zs[L]) for L in EX}, "y": torch.cat(ys), "bench": torch.tensor(bs), "cid": cids,
             "lab": labs, "slug": slugs, "opts": opts}
        d["half"] = [half_of(c) for c in cids]
        return d

    def pack_di(self, d, sel=None):
        EX, WB = self.EX, self.WB
        ix = torch.arange(len(d["y"])) if sel is None else sel
        pk = self.pack({"z": {L: d["z"][L][ix] for L in EX}, "y": d["y"][ix]})
        pk["bench"] = d["bench"][ix]
        il = ix.tolist()
        pk["cid"] = [d["cid"][i] for i in il]
        pk["gold"] = d["y"][ix]
        bl = sorted({int(b) for b in pk["bench"].tolist() if WB.get(int(b), 0) > 0})
        pk["blist"] = bl
        pk["bix"] = {b: (pk["bench"] == b).nonzero().squeeze(1) for b in bl}
        pk["M"] = {}
        for b in bl:
            bix = pk["bix"][b]
            o = {}
            if b in CASE_EXACT or b in REVIEW_F1:
                cl = {}
                o["case"] = torch.tensor([cl.setdefault(pk["cid"][i], len(cl)) for i in bix.tolist()])
                o["nc"] = len(cl)
                if b in CASE_EXACT:
                    lk = torch.zeros(len(cl)).index_add_(0, o["case"], -torch.log(pk["K"][bix]))
                    o["chance"] = float(torch.exp(-lk).mean())
                else:
                    o["yes"] = torch.tensor([d["opts"][il[i]].index("yes") if "yes" in d["opts"][il[i]] else -1
                                             for i in bix.tolist()])
            if b in MACRO_F1:
                u = {}
                lab = [d["lab"][il[i]] for i in bix.tolist()]
                for ls in lab:
                    for x in ls:
                        u.setdefault(x, len(u))
                Wd = max(len(x) for x in lab)
                o["labmat"] = torch.tensor([[u[x] for x in ls] + [-1] * (Wd - len(ls)) for ls in lab])
                o["nl"] = len(u)
            pk["M"][b] = o
        return pk

    def bench_metric(self, pk, b, ch):
        """Decision-Index-metric skill and mean depth of benchmark b under exit choice ch."""
        bix = pk["bix"][b]
        c = ch[bix]
        ar = torch.arange(len(bix))
        pr = pk["pr"][c, bix]
        ok = pk["ok"][c, bix]
        o = pk["M"][b]
        if b in CASE_EXACT:
            bad = torch.zeros(o["nc"]).index_add_(0, o["case"], 1 - ok)
            raw = float((bad == 0).float().mean())
            ch_ = o["chance"]
        elif b in REVIEW_F1:
            yes = o["yes"]
            gy = pk["gold"][bix] == yes
            py = pr == yes
            G = torch.zeros(o["nc"]).index_add_(0, o["case"], gy.float())
            P_ = torch.zeros(o["nc"]).index_add_(0, o["case"], py.float())
            I_ = torch.zeros(o["nc"]).index_add_(0, o["case"], (gy & py).float())
            den = G + P_
            raw = float(torch.where(den > 0, 2 * I_ / den.clamp_min(1e-9), torch.ones_like(den)).mean())
            ch_ = self.CHANCE.get(b, 0.0)
        elif b in MACRO_F1:
            lm = o["labmat"]
            g = lm[ar, pk["gold"][bix]]
            p = lm[ar, pr]
            nl = o["nl"]
            tp = torch.bincount(g[g == p], minlength=nl).float()
            fp = torch.bincount(p[g != p], minlength=nl).float()
            fn = torch.bincount(g[g != p], minlength=nl).float()
            den = 2 * tp + fp + fn
            cls = torch.unique(lm[lm >= 0])
            raw = float(torch.where(den[cls] > 0, 2 * tp[cls] / den[cls].clamp_min(1e-9), torch.zeros_like(den[cls])).mean())
            ch_ = self.CHANCE.get(b, float((1 / pk["K"][bix]).mean()))
        else:
            raw = float(ok.mean())
            ch_ = self.CHANCE.get(b, float((1 / pk["K"][bix]).mean()))
        sk = min(1.0, max(0.0, (raw - ch_) / (1 - ch_))) if ch_ < 1 else raw
        return sk, float(self.EXT[c].mean())

    def m_sm(self, pk, ch):
        per = {b: self.bench_metric(pk, b, ch) for b in pk["blist"]}
        Wt = sum(self.WB[b] for b in per)
        U = sum(self.WB[b] * v[0] for b, v in per.items()) / Wt
        dep = sum(self.WB[b] * v[1] for b, v in per.items()) / Wt
        return {"U": U, "depth": dep, "J": U - C_DEPTH * (dep - 20)}, per

    # ---- the search
    def select(self, heldout: dict, shift_matched: dict) -> dict:
        """heldout / shift_matched: load_di() sets. A selects, B confirms."""
        H, EX = self.v1["H"], self.EX
        I32 = EX.index(EX[-1])

        def hsel(d, h):
            return torch.tensor([x == h for x in d["half"]]).nonzero().squeeze(1)
        res = {"T": {str(k): v for k, v in self.T.items()}, "n": {}, "dev": {}}
        D = {}
        for h in "AB":
            pk, pv = self.pack(H[h]["t"]), self.pack(H[h]["v"])
            s2, ss = hsel(heldout, h), hsel(shift_matched, h)
            D[h] = dict(pk=pk, pv=pv, v2=self.pack_di(heldout, s2), sm=self.pack_di(shift_matched, ss))
            D[h]["VIS32"] = float((H[h]["wv"] * pv["ok"][I32]).sum())
            res["n"][h] = {"v1_text": pk["n"], "v1_vis": pv["n"], "v2": D[h]["v2"]["n"], "sm": D[h]["sm"]["n"]}
        for k, p in self.pols.items():
            res["dev"][k] = {}
            for h in "AB":
                d = D[h]
                pk = d["pk"]
                w = H[h]["wt"]
                ch = self.route(pk, p)
                ok = pk["ok"][ch, torch.arange(pk["n"])]
                TXT, KN = float((w * ok).sum()), float((H[h]["wk"] * ok).sum())
                ch2 = self.route(d["v2"], p)
                ho = float(d["v2"]["ok"][ch2, torch.arange(d["v2"]["n"])].mean())
                sm, _ = self.m_sm(d["sm"], self.route(d["sm"], p))
                res["dev"][k][h] = {"suite_U": round((TXT + KN + d["VIS32"]) / 3, 4), "TXT": round(TXT, 4), "KN": round(KN, 4),
                                    "VIS": round(d["VIS32"], 4), "suite_depth": round(float((w * self.EXT[ch]).sum()), 2),
                                    "share16/20/32": [round(float(w[ch == EX.index(L)].sum()), 3) for L in (16, 20, 32)],
                                    "heldout_acc": round(ho, 4), "heldout_depth": round(float(self.EXT[ch2].mean()), 2),
                                    "sm_U": round(sm["U"], 4), "sm_depth": round(sm["depth"], 2)}

        def feas(k, h):
            r, f = res["dev"][k][h], res["dev"]["fixed32"][h]
            return {"C1": r["suite_depth"] <= self.DCAP, "C2": r["suite_U"] >= f["suite_U"] - self.TOL,
                    "C3": r["heldout_acc"] >= f["heldout_acc"] - self.TOL}
        for k in self.pols:
            for h in "AB":
                res["dev"][k][h]["feas"] = feas(k, h)
        FA = [self.name(p) for p in self.grid if all(res["dev"][self.name(p)]["A"]["feas"].values())]
        sel = {"feasible_A": FA}
        if FA:
            top = max(res["dev"][k]["A"]["sm_U"] for k in FA)
            tied = [k for k in FA if res["dev"][k]["A"]["sm_U"] >= top - self.TIE]
            win = min(tied, key=lambda k: (res["dev"][k]["A"]["suite_depth"], self.pols[k]["tau"][16], self.pols[k]["tau"][20]))
            b = res["dev"][win]["B"]
            sel.update(winner=win, tau={"16": self.pols[win]["tau"][16], "20": self.pols[win]["tau"][20]}, tied_A=tied,
                       B_feas=b["feas"], B_sm_U=b["sm_U"], B_sm_U_C16=res["dev"][self.c16]["B"]["sm_U"])
            sel["CONFIRMED"] = bool(win != self.c16 and all(b["feas"].values()) and b["sm_U"] > res["dev"][self.c16]["B"]["sm_U"])
        else:
            sel["CONFIRMED"] = False
        if self.single:
            sel["family"] = "single"
            if not sel["CONFIRMED"]:
                sel["fallback"] = True
                sel["tau"] = {"16": self.FALLBACK, "20": self.FALLBACK}
                fb = self.name({"tau": {16: self.FALLBACK, 20: self.FALLBACK}})
                sel["fallback_B"] = {k: v for k, v in res["dev"][fb]["B"].items() if k != "feas"}
        res["selection"] = sel
        return res

    def eval_read(self, sel: dict, eval_dump, eval_vis_dump) -> dict:
        """The one evaluation read of a confirmed winner, fixed32 and the single-tau cascade."""
        EX = self.EX
        E = load(eval_dump) if not isinstance(eval_dump, dict) else eval_dump
        Ed = {"z": {L: pad(E["z"][L]) for L in EX}, "y": E["y"].long()}
        pk = self.pack(Ed)
        tags = list(E["tag"])
        fin = torch.tensor([t.startswith("final.") for t in tags])
        TI = {t: torch.tensor([x == t for x in tags]) for t in set(tags)}
        VE = load(eval_vis_dump) if not isinstance(eval_vis_dump, dict) else eval_vis_dump
        zv = VE["z"][EX[-1]] if EX[-1] in VE["z"] else VE["z"][str(EX[-1])]
        okv = (clean(pad(zv)).argmax(-1) == VE["gold"].long()).float()
        bench = VE["bench"]
        pbm = {b: float(okv[torch.tensor([x == b for x in bench])].mean()) for b in sorted(set(bench))}
        vis32 = round(sum(pbm.values()) / len(pbm), 4)
        ev = {}
        for k in [sel["winner"], "fixed32", self.c16]:
            ch = self.route(pk, self.pols[k])
            ar = torch.arange(pk["n"])
            ok = pk["ok"][ch, ar]
            conf = pk["pm"][ch, ar]
            per = {t: float(ok[m].mean()) for t, m in TI.items()}
            sx = {s: per[s] for s in SUITE_W if s in per and s != "open_jev_ood"}
            ev[k] = {"suite_x_ood": round(sum(SUITE_W[s] * v for s, v in sx.items()) / sum(SUITE_W[s] for s in sx), 4),
                     "final": round(float(ok[fin].mean()), 4), "mmlu_pro": round(per.get("mmlu_pro_1k", float("nan")), 4),
                     "final_ece": round(A.ece(conf[fin], ok[fin]), 4), "suite_ece": round(A.ece(conf[~fin], ok[~fin]), 4),
                     "depth_all": round(float(self.EXT[ch].mean()), 2), "depth_final": round(float(self.EXT[ch][fin].mean()), 2),
                     "share16/20/32": [round(float((ch == EX.index(L)).float().mean()), 4) for L in (16, 20, 32)],
                     "vision_exit32": vis32}
        return ev
