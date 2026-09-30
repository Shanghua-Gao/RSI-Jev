"""HTTP benchmark of one running /v1/systemone server: single requests and concurrency.

    python docs/assets/bench_http_gb10.py --port P --tag NAME --out results.json [--single] [--conc] [--showcase FILE]

--single   one client, sequential, keep-alive: p50 for 1/8/32 questions on the 80- and
           1,052-token documents
--showcase the showcase requests (a JSON list of {id, state, questions}), p50 of 9 after 3
--conc     C in {1, 8, 32} closed-loop clients sending back to back for --seconds after a
           warm-up; 1- and 8-question requests on the 80- and 1,052-token documents;
           throughput (completed req/s over the window) and p50/p95 latency

The method is the one the vLLM comparison used (speed/exp/vllm/bench_http.py): every
request starts with its own first line, so no request reads a document an earlier one
cached, and the 32 questions are distinct.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import statistics as st
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_gb10 import DOCS, QUESTIONS  # noqa: E402

POOL = [(f"q{i}", {**q, "instructions": q["instructions"] + (f" (item {i // 4})" if i >= 4 else "")})
        for i, q in enumerate(QUESTIONS.values())]
_ctr = itertools.count()


def body(doc: str, n: int) -> dict:
    return {"model": "jev-latest", "state": f"Ref {next(_ctr):08d}.\n" + DOCS[doc],
            "questions": dict(POOL[:n])}


async def one(client, url, b):
    t0 = time.perf_counter()
    r = await client.post(url, json=b)
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000, r.json()


async def single(client, url, reps):
    rows = []
    for doc in ("80", "1052"):
        for n in (1, 8, 32):
            for _ in range(3):
                await one(client, url, body(doc, n))
            xs = [(await one(client, url, body(doc, n)))[0] for _ in range(reps)]
            rows.append({"doc": doc, "q": n, "p50_ms": round(st.median(xs), 2)})
            print("single", rows[-1], flush=True)
    return rows


async def showcase(client, url, path):
    rows = []
    for e in json.loads(Path(path).read_text()):
        b = {"model": "jev-latest", "state": [{"role": "user", "content": e["state"]}],
             "questions": e["questions"]}
        for _ in range(3):
            await one(client, url, b)
        xs = sorted([(await one(client, url, b))[0] for _ in range(9)])
        _, js = await one(client, url, b)
        rows.append({"id": e["id"], "q": len(e["questions"]), "p50_ms": round(xs[4], 2),
                     "answers": js["answers"]})
        print("showcase", e["id"], rows[-1]["p50_ms"], flush=True)
    return rows


async def closed_loop(client, url, doc, n, conc, seconds):
    lat = []
    await asyncio.gather(*(one(client, url, body(doc, n)) for _ in range(conc)))
    stop = time.perf_counter() + seconds

    async def worker():
        while time.perf_counter() < stop:
            ms, _ = await one(client, url, body(doc, n))
            lat.append(ms)
    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(conc)))
    wall = time.perf_counter() - t0
    lat.sort()
    return {"conc": conc, "n": len(lat), "req_s": round(len(lat) / wall, 2),
            "p50_ms": round(lat[len(lat) // 2], 1),
            "p95_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--conc", action="store_true")
    ap.add_argument("--showcase", default=None)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--docs", default="80,1052")
    ap.add_argument("--qs", default="1,8")
    ap.add_argument("--clients", default="1,8,32")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/v1/systemone"
    res = {"tag": a.tag}
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=64)) as client:
        if a.showcase:
            res["showcase"] = await showcase(client, url, a.showcase)
        if a.single:
            res["single"] = await single(client, url, a.reps)
        if a.conc:
            rows = []
            for doc in a.docs.split(","):
                for n in map(int, a.qs.split(",")):
                    for c in map(int, a.clients.split(",")):
                        r = {"doc": doc, "q": n, **await closed_loop(client, url, doc, n, c, a.seconds)}
                        rows.append(r)
                        print("conc", r, flush=True)
            res["conc"] = rows
    p = Path(a.out)
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(res)
    p.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
