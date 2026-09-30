"""Continued-training corpora: a data arm and its replay-only control.

Every arm of one round trains the SAME number of steps S (batch B), so each
corpus holds exactly S*B questions (one epoch; the 10% in_distribution holdout
is taken by the trainer from these cases, identically across arms):

  data arm  = its new-data files (each file = one source, kept as its own corpus
              key so per-source attribution is possible) + the first
              S*B - N_new questions of the replay stream;
  control   = the first S*B questions of the SAME replay stream, no new data.

The replay stream is a fixed, seeded, source-stratified shuffle of the parent's
own training corpus (parent corpus dir restricted to the parent's sources), so
the control's replay is a superset of every data arm's replay. The comparison
data-arm vs control therefore isolates the new data at equal steps.

The ORDER of --parent-sources matters: the stratified interleave breaks ties by it.
Pass the parent's `sources` exactly as its training spec lists them (for a parent built
by this script, the "sources" field of its manifest.json, which is sorted).

Replay files are written as rp_<parent source>.jsonl (a parent that is itself a
continued-training corpus gives rp_rp_...); new files keep their file name. A
manifest.json lists counts and sha256 per file.

    python scripts/ct_build_corpus.py --parent-corpus DIR --parent-sources a,b,c \\
        --steps 1500 --batch 16 --out OUT [--new f1.jsonl f2.jsonl ...] [--seed 17]

scripts/build_stage_corpora.py calls this for every continued-training stage of v4.0-VL.
CPU only, standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def n_questions(line: str) -> int:
    return len(json.loads(line).get("questions") or [1])


def replay_stream(parent: Path, sources: list[str], seed: int) -> list[tuple[str, str]]:
    """(source, jsonl line) pairs: each source shuffled with its own seeded rng,
    then interleaved in proportion to its size (stratified), deterministically."""
    per = {}
    for s in sources:
        lines = [l for l in open(parent / f"{s}.jsonl") if l.strip()]
        random.Random(f"{seed}:{s}").shuffle(lines)
        per[s] = lines
    total = sum(len(v) for v in per.values())
    # stratified interleave: at position t take the source furthest behind its quota
    cur = {s: 0 for s in per}
    out = []
    for t in range(total):
        s = min(per, key=lambda k: (cur[k] + 1) / len(per[k]) if cur[k] < len(per[k]) else 9e9)
        out.append((s, per[s][cur[s]]))
        cur[s] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parent-corpus", required=True)
    ap.add_argument("--parent-sources", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--new", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    budget = a.steps * a.batch
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    new_q, files = 0, {}
    for f in map(Path, a.new):
        lines = [l for l in open(f) if l.strip()]
        (out / f.name).write_text("".join(lines))
        q = sum(n_questions(l) for l in lines)
        files[f.stem] = {"role": "new", "cases": len(lines), "questions": q}
        new_q += q
    if new_q >= budget:
        raise SystemExit(f"new data ({new_q} q) >= step budget ({budget} q): raise --steps")
    need = budget - new_q
    stream = replay_stream(Path(a.parent_corpus), a.parent_sources.split(","), a.seed)
    rp: dict[str, list[str]] = {}
    got = 0
    for s, line in stream:
        if got >= need:
            break
        rp.setdefault(s, []).append(line)
        got += n_questions(line)
    for s, lines in rp.items():
        (out / f"rp_{s}.jsonl").write_text("".join(lines))
        files[f"rp_{s}"] = {"role": "replay", "cases": len(lines),
                            "questions": sum(n_questions(l) for l in lines)}
    for k in files:
        files[k]["sha256"] = hashlib.sha256((out / f"{k}.jsonl").read_bytes()).hexdigest()
    man = {"steps": a.steps, "batch": a.batch, "budget_questions": budget,
           "new_questions": new_q, "replay_questions": got, "seed": a.seed,
           "parent_corpus": a.parent_corpus, "parent_sources": a.parent_sources,
           "sources": ",".join(sorted(files)), "files": files}
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(f"{out}: {new_q} new + {got} replay questions (budget {budget}); "
          f"sources={man['sources'][:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
