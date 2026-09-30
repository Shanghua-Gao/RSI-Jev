"""Depth-dial generators: synthetic reasoning questions whose difficulty is one integer.

SYNTHETIC DATA. Every template and every name list is written in this file; no text is
taken from a benchmark. The three task FAMILIES do overlap BIG-Bench Hard's
web_of_lies (truth-value chains), tracking_shuffled_objects (pairwise swaps) and
logical_deduction (N-object orderings), so a model trained on these items is not
held out on those BBH tasks (the release card says so).

  lies   truth-value chains: a base fact, then d links that each copy or invert the
         previous one                                                       (k = 2)
  swap   object tracking: n holders, d pairwise exchanges, ask what one holder has at
         the end                                                            (k = n)
  order  N-object deduction: N objects in a row, relative clues until the order is
         unique                                                             (k = N)

Each task has several surface families, so a trained model cannot key on one phrasing.
The depth grid (CELLS) runs past the model's capacity on purpose: the calA-ce stage
teaches it where its accuracy falls to chance (scripts/build_depthdial.py).

Items are rsijev contract cases (choice mode, one question "answer", one-hot gold);
meta carries gen, depth, k, family, the true answer index y and the fold.

    python scripts/depthdial.py make --out gen_measure.jsonl --fold measure --n 200 --seed 1
    python scripts/depthdial.py make --out gen_probe.jsonl   --fold probe   --n 200 --seed 1
    python scripts/depthdial.py cells                        # print the depth grid

The folds used for calA-ce are listed in scripts/build_depthdial.py (FOLDS). The
training fold is made inside build_depthdial.py (fold "train", seed 2). The generators
are deterministic: the same fold, n and seed give the same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

NAMES = ["Arlo", "Bex", "Cato", "Dara", "Emrys", "Faye", "Gil", "Hana", "Ivo", "Juno", "Kei", "Lior", "Mira", "Nils",
         "Oona", "Pax", "Quill", "Rhea", "Soren", "Tamsin", "Ula", "Vero", "Wren", "Xan", "Yara", "Zev", "Ansel", "Brin",
         "Coral", "Dov", "Esme", "Fenn", "Greer", "Hollis", "Ines", "Jory", "Kit", "Lark", "Moss", "Nell"]
ITEMS = {
    "card": ["red card", "blue card", "green card", "gold card", "grey card", "violet card", "white card", "black card",
             "orange card"],
    "tool": ["hammer", "wrench", "chisel", "saw", "level", "trowel", "file", "clamp", "drill"],
    "fruit": ["plum", "fig", "lime", "pear", "kiwi", "date", "quince", "apricot", "lychee"],
}
SEATS = ["desk", "locker", "shelf", "crate", "tray", "drawer", "bin", "cubby", "slot"]
OBJ = {
    "vehicles": ["the van", "the bus", "the taxi", "the tractor", "the scooter", "the truck", "the tram", "the jeep"],
    "books": ["the atlas", "the diary", "the novel", "the manual", "the almanac", "the ledger", "the folio", "the primer"],
    "runners": NAMES[:8],
}

CELLS = {
    "lies": [(d, 2) for d in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)],
    "swap": [(d, n) for n in (3, 5) for d in (1, 2, 3, 4, 6, 8, 12)],
    "order": [(N, N) for N in (3, 4, 5, 6, 7)],
}


def _sid(*x):
    return hashlib.sha256("|".join(map(str, x)).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ lies
def gen_lies(rng: random.Random, d: int):
    fam = rng.choice(["people", "lamps", "notes"])
    ppl = rng.sample(NAMES, d + 1)
    val = rng.random() < 0.5                    # truth value of item 0
    vals = [val]
    lines = []
    if fam == "people":
        lines.append(f"In this group every person is either always honest or always dishonest. "
                     f"{ppl[0]} is {'honest' if val else 'dishonest'}.")
        for i in range(1, d + 1):
            says_honest = rng.random() < 0.5
            lines.append(f"{ppl[i]} states that {ppl[i-1]} is {'honest' if says_honest else 'dishonest'}.")
            vals.append(says_honest == vals[-1])
        q = f"Is {ppl[d]} honest?"
    elif fam == "lamps":
        lines.append(f"There are {d + 1} lamps in a row. Lamp 1 is {'on' if val else 'off'}.")
        for i in range(1, d + 1):
            same = rng.random() < 0.5
            lines.append(f"Lamp {i+1} is wired to lamp {i}: it is {'in the same state as' if same else 'in the opposite state to'} lamp {i}.")
            vals.append(vals[-1] if same else not vals[-1])
        q = f"Is lamp {d + 1} on?"
    else:
        tags = [f"note {c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:d + 1]]
        lines.append(f"Several notes were left on a table. {tags[0].capitalize()} is {'accurate' if val else 'inaccurate'}.")
        for i in range(1, d + 1):
            says_acc = rng.random() < 0.5
            lines.append(f"{tags[i].capitalize()} says that {tags[i-1]} is {'accurate' if says_acc else 'inaccurate'}.")
            vals.append(says_acc == vals[-1])
        q = f"Is {tags[d]} accurate?"
    opts = ["yes", "no"]
    return fam, "\n".join(lines), q, opts, 0 if vals[-1] else 1


# ------------------------------------------------------------------ swap
def gen_swap(rng: random.Random, d: int, n: int):
    fam = rng.choice(["people", "boxes"])
    kind = rng.choice(sorted(ITEMS))
    items = rng.sample(ITEMS[kind], n)
    if fam == "people":
        hold = rng.sample(NAMES, n)
        lines = ["At the start: " + "; ".join(f"{h} has the {it}" for h, it in zip(hold, items)) + "."]
        verbs = ["{a} and {b} trade what they hold.", "{a} swaps items with {b}.", "{b} and {a} exchange items."]
    else:
        hold = [f"the {s}" for s in rng.sample(SEATS, n)]
        lines = ["Initially: " + "; ".join(f"{h} contains the {it}" for h, it in zip(hold, items)) + "."]
        verbs = ["The contents of {a} and {b} are switched.", "Someone moves the item in {a} to {b} and the item in {b} to {a}."]
    cur = dict(zip(hold, items))
    for t in range(d):
        a, b = rng.sample(hold, 2)
        cur[a], cur[b] = cur[b], cur[a]
        lines.append(f"Step {t + 1}: " + rng.choice(verbs).format(a=a, b=b))
    who = rng.choice(hold)
    q = f"At the end, which item does {who} {'have' if fam == 'people' else 'contain'}?"
    opts = [f"the {it}" for it in items]
    return fam, "\n".join(lines), q, opts, items.index(cur[who])


# ------------------------------------------------------------------ order
def gen_order(rng: random.Random, N: int):
    fam = rng.choice(sorted(OBJ))
    objs = rng.sample(OBJ[fam], N)
    perm = objs[:]
    rng.shuffle(perm)                               # perm[i] = object at position i (0 = leftmost / first)
    pos = {o: i for i, o in enumerate(perm)}
    if fam == "runners":
        L = ("finished before", "finished after", "finished {k}", "finished {k} from last")
        where = "A race had {N} runners and no ties: {objs}."
        ask = "Who finished {k}?"
    else:
        L = ("is left of", "is right of", "is {k} from the left", "is {k} from the right")
        where = "{N} items stand in a single row: {objs}."
        ask = "Which item is {k} from the left?"
    ordn = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]

    def clue(rng):
        r = rng.random()
        if r < 0.12:                                 # absolute clue (rare)
            o = rng.choice(objs)
            if rng.random() < 0.5:
                return (f"{o} {L[2].format(k=ordn[pos[o]])}", lambda p, o=o, i=pos[o]: p.index(o) == i)
            j = N - 1 - pos[o]
            return (f"{o} {L[3].format(k=ordn[j])}", lambda p, o=o, j=j: p.index(o) == N - 1 - j)
        a, b = rng.sample(objs, 2)
        if r < 0.3:                                  # immediately adjacent
            if pos[a] + 1 == pos[b] or pos[b] + 1 == pos[a]:
                x, y = (a, b) if pos[a] < pos[b] else (b, a)
                rel = "directly before" if fam == "runners" else "immediately left of"
                return (f"{x} {'finished ' if fam == 'runners' else 'is '}{rel} {y}",
                        lambda p, x=x, y=y: p.index(x) + 1 == p.index(y))
        if pos[a] < pos[b]:
            return (f"{a} {L[0]} {b}", lambda p, a=a, b=b: p.index(a) < p.index(b))
        return (f"{a} {L[1]} {b}", lambda p, a=a, b=b: p.index(a) > p.index(b))

    cand = list(itertools.permutations(objs))
    clues, texts = [], []
    while len(cand) > 1:
        t, f = clue(rng)
        if t in texts:
            continue
        nxt = [p for p in cand if f(p)]
        if len(nxt) == len(cand):
            continue
        cand = nxt
        texts.append(t)
    assert list(cand[0]) == perm
    rng.shuffle(texts)
    k = rng.randrange(N)
    state = where.format(N=N, objs=", ".join(objs)) + "\n" + "\n".join(t[0].upper() + t[1:] + "." for t in texts)
    q = ask.format(k=ordn[k])
    return fam, state, q, objs, objs.index(perm[k])


def make_item(gen: str, depth: int, k: int, rng: random.Random, fold: str, i: int):
    if gen == "lies":
        fam, state, q, opts, y = gen_lies(rng, depth)
    elif gen == "swap":
        fam, state, q, opts, y = gen_swap(rng, depth, k)
    else:
        fam, state, q, opts, y = gen_order(rng, depth)
    cid = f"dd_{gen}_{fold}_{depth}_{k}_{i}_{_sid(state, q)}"
    return {"case_id": cid, "source": f"dd_{gen}", "state": state,
            "questions": [{"key": "answer", "mode": "choice", "instructions": q, "options": opts,
                           "criteria": {o: o for o in opts}}],
            "gold": {"answer": [1.0 if j == y else 0.0 for j in range(len(opts))]},
            "meta": {"gen": gen, "depth": depth, "k": k, "family": fam, "y": y, "fold": fold}}


def make(fold: str, n: int, seed: int):
    out = []
    for gen, cells in CELLS.items():
        for depth, k in cells:
            rng = random.Random(f"{seed}|{fold}|{gen}|{depth}|{k}")
            seen = set()
            i = 0
            while i < n:
                it = make_item(gen, depth, k, rng, fold, i)
                key = it["state"] + it["questions"][0]["instructions"]
                if key in seen:
                    continue
                seen.add(key)
                out.append(it)
                i += 1
    return out


def write(items, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Depth-dial synthetic reasoning generators.")
    ap.add_argument("cmd", choices=["make", "cells"])
    ap.add_argument("--out", help="output .jsonl (make)")
    ap.add_argument("--fold", default="measure")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "cells":
        print(json.dumps(CELLS))
    else:
        items = make(a.fold, a.n, a.seed)
        write(items, a.out)
        print(a.fold, len(items), "->", a.out)
