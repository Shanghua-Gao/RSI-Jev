"""Where one request's time goes on the served path, in-process and over HTTP.

    python docs/assets/profile_gb10.py --ckpt DIR --out profile.json --traces DIR [--http]

One model process. For each workload (1 question on the 80- and 1,052-token
documents, 32 questions on both) it records:

  wall      the served scorer end to end, CUDA-synchronised, p50 of --repeat;
  stages    the same pipeline re-run step by step with a synchronise between
            steps: request parse and validation, tokenisation (encode_question,
            per question), the shared-prefix check, collate (with the option
            first-token lookups it does, and the padded vs real tokens), the
            forward pass, unpermute/softmax, the per-row GPU->CPU copies,
            answer building and JSON serialisation;
  gpu       a torch.profiler trace of one scorer call with every tower module
            annotated, reduced to GPU kernel time per module type (Gated
            DeltaNet mixers, full-attention mixers, MLPs, RMSNorms, embedding,
            scorer, calibration, everything else) and the GPU-idle time inside
            the forward.

--http also serves the app in this process (uvicorn in a thread) and times the
same requests from a separate stdlib client process: client total, server total
(middleware entry to response), validation before the handler, the handler's
own prepare/infer/answer/serialise split.

The stage breakdown re-implements serve.infer's steps with the same functions in
the same order; `stages_total_ms` is reported next to `wall` so the two can be
compared.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics as st
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from bench_gb10 import DOCS, QUESTIONS  # noqa: E402

WORKLOADS = [("80", 1), ("1052", 1), ("80", 8), ("1052", 8), ("80", 32), ("1052", 32)]


def p50(xs):
    return round(st.median(xs), 3)


def sync():
    import torch
    torch.cuda.synchronize()


# ------------------------------------------------------------------ annotation
def annotate(model):
    """Wrap every tower module of interest in a record_function range."""
    import torch
    from rsijev.arch import _decoder_layers
    wrapped = []

    def wrap(mod, name):
        orig = mod.forward

        def fwd(*a, **kw):
            with torch.profiler.record_function(name):
                return orig(*a, **kw)
        mod.forward = fwd
        wrapped.append((mod, orig))

    tower = model.tower
    wrap(tower.embed_tokens, "tower.embed")
    wrap(tower.norm, "tower.final_norm")
    wrap(tower.rotary_emb, "tower.rotary")
    for layer in _decoder_layers(tower):
        if hasattr(layer, "linear_attn"):
            wrap(layer.linear_attn, "tower.deltanet")
        else:
            wrap(layer.self_attn, "tower.full_attn")
        wrap(layer.mlp, "tower.mlp")
        wrap(layer.input_layernorm, "tower.norm")
        wrap(layer.post_attention_layernorm, "tower.norm")
    wrap(model.tower, "tower")
    wrap(model.scorer, "scorer")
    orig_cal = model.cal_log_temperature

    def cal(*a, **kw):
        with torch.profiler.record_function("calibration"):
            return orig_cal(*a, **kw)
    model.cal_log_temperature = cal
    return wrapped


def gpu_breakdown(trace_path: Path, window: tuple[str, ...] = ("scorer_call",)) -> dict:
    """Kernel time per innermost annotation, from a chrome trace."""
    ev = json.loads(Path(trace_path).read_text())["traceEvents"]
    kernels = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    ann = [e for e in ev if e.get("cat") == "gpu_user_annotation"]
    cpu_ann = [e for e in ev if e.get("cat") == "user_annotation"]
    by = defaultdict(float)
    count = defaultdict(int)
    for k in kernels:
        t0, t1 = k["ts"], k["ts"] + k.get("dur", 0)
        best = None
        for a in ann:
            if a["name"] in window:
                continue
            if a["ts"] <= t0 and t1 <= a["ts"] + a["dur"] + 0.5:
                if best is None or a["dur"] < best["dur"]:
                    best = a
        name = best["name"] if best else "other"
        if name == "tower":
            name = "tower.other"
        by[name] += k.get("dur", 0) / 1000
        count[name] += 1
    busy = sum(by.values())
    # wall of the whole scorer call (CPU range) and span of GPU activity inside it
    call = [a for a in cpu_ann if a["name"] == "scorer_call"]
    wall = call[0]["dur"] / 1000 if call else None
    span = ((max(k["ts"] + k.get("dur", 0) for k in kernels) - min(k["ts"] for k in kernels)) / 1000
            if kernels else None)
    # kernel launches seen on the CPU side
    launches = sum(1 for e in ev if e.get("cat") == "cuda_runtime"
                   and ("LaunchKernel" in e.get("name", "") or "cuLaunch" in e.get("name", "")))
    return {"kernel_ms_by_module": {k: round(v, 3) for k, v in sorted(by.items(), key=lambda x: -x[1])},
            "kernels_by_module": dict(count), "kernel_ms_total": round(busy, 3),
            "scorer_call_wall_ms": round(wall, 3) if wall else None,
            "gpu_span_ms": round(span, 3) if span else None,
            "gpu_idle_in_span_ms": round(span - busy, 3) if span else None,
            "kernel_launches": launches, "kernels": len(kernels)}


# ------------------------------------------------------------------ stages
def stage_run(s, body: dict, batch_size: int = 16) -> dict:
    """The served pipeline, step by step, with a synchronise between steps."""
    import torch
    import torch.nn.functional as F
    from serve.app import SystemOneRequest, _wire_questions
    from serve.infer import _shared_prefix, _replicate, default_min_saved_tokens
    from serve.wire import state_to_text, to_answer
    from rsijev.encode import collate, encode_question, unpermute_logits, option_first_token_ids
    from rsijev.contract import Prediction
    T = {}
    raw = json.dumps(body).encode()

    t = time.perf_counter(); d = json.loads(raw); T["json_parse"] = time.perf_counter() - t
    t = time.perf_counter(); req = SystemOneRequest.model_validate(d); T["pydantic_validate"] = time.perf_counter() - t
    t = time.perf_counter(); qs = _wire_questions(req); state = state_to_text(req.state)
    T["wire_questions"] = time.perf_counter() - t

    tok, enc, model, device = s.tok, s.enc, s.model, s.device
    spec = s.meta["spec"]
    max_options = max(spec["max_options"], max(len(q.options) for q in qs))
    import serve.infer as infer
    new = hasattr(infer, "encode_questions")          # this tree has the one-pass encoder
    t = time.perf_counter()
    if new:
        encoded, lead = infer.encode_questions(tok, state, qs, enc)
    else:
        encoded, lead = [encode_question(tok, state, q, enc) for q in qs], None
    T["encode_question"] = time.perf_counter() - t
    t = time.perf_counter()
    prefix = _shared_prefix(tok, state, enc, encoded, **({"ids": lead} if new else {}))
    T["shared_prefix"] = time.perf_counter() - t
    use_cache = prefix is not None and len(qs) >= 2 and \
        (len(qs) - 1) * len(prefix) >= default_min_saved_tokens()
    npfx = len(prefix) if use_cache else 0
    rows = [{**e, "input_ids": e["input_ids"][npfx:],
             "option_index": [i - npfx for i in e["option_index"]],
             "option_span": [(a - npfx, b - npfx) for a, b in e["option_span"]],
             "decision_index": e["decision_index"] - npfx} for e in encoded] if npfx else encoded
    extra = defaultdict(float)
    real = padded = 0
    probs_all = []
    if use_cache:
        sync(); t = time.perf_counter()
        cache = model.encode_prefix(torch.tensor([prefix], dtype=torch.long, device=device))
        sync(); T["prefix_forward"] = time.perf_counter() - t
    for i in range(0, len(rows), batch_size):
        part = rows[i:i + batch_size]
        if not new:
            t = time.perf_counter()
            for e in part:
                option_first_token_ids(tok, e["options"])
            extra["collate.option_first_token_ids"] += time.perf_counter() - t
        t = time.perf_counter()
        batch = collate(tok, part, max_options=max_options, device=device,
                        **({"option_tokens": False} if new else {}))
        sync(); extra["collate"] += time.perf_counter() - t
        w = batch["input_ids"].shape[1]
        real += sum(len(e["input_ids"]) for e in part)
        padded += w * len(part)
        kw = {}
        if use_cache:
            t = time.perf_counter()
            batch["attention_mask"] = torch.cat(
                [torch.ones((len(part), npfx), dtype=batch["attention_mask"].dtype, device=device),
                 batch["attention_mask"]], dim=1)
            pos = (torch.arange(w, device=device) + npfx).unsqueeze(0).expand(len(part), w)
            kw = {"past_key_values": _replicate(cache, len(part), device), "position_ids": pos}
            sync(); extra["cache_replicate"] += time.perf_counter() - t
        t = time.perf_counter()
        logits = model(**batch, **kw)
        sync(); extra["forward"] += time.perf_counter() - t
        t = time.perf_counter()
        probs = F.softmax(unpermute_logits(logits, batch["option_perm"], batch["option_mask"]), dim=-1)
        sync(); extra["unpermute_softmax"] += time.perf_counter() - t
        t = time.perf_counter()
        if new:                                        # one copy per batch
            host = probs.float().cpu()
            for r, q in enumerate(qs[i:i + batch_size]):
                probs_all.append(list(Prediction(tuple(host[r, : len(q.options)].tolist())).probs))
        else:
            for r, q in enumerate(qs[i:i + batch_size]):
                probs_all.append(list(Prediction(tuple(probs[r, : len(q.options)].float().tolist())).probs))
        extra["per_row_tolist"] += time.perf_counter() - t
    T.update(extra)
    t = time.perf_counter()
    answers = {q.key: to_answer(q, p) for q, p in zip(qs, probs_all)}
    T["to_answer"] = time.perf_counter() - t
    out = {"model": req.model, "answers": answers,
           "usage": {"input_tokens": npfx + real, "output_tokens": len(qs)}}
    t = time.perf_counter()
    json.dumps(out, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode()
    T["json_serialize"] = time.perf_counter() - t
    try:
        import orjson
        t = time.perf_counter(); orjson.dumps(out); T["json_serialize_orjson"] = time.perf_counter() - t
    except ImportError:
        pass
    res = {k: v * 1000 for k, v in T.items()}
    res["tokens_real"] = real
    res["tokens_padded"] = padded
    res["prefix_tokens"] = npfx
    res["cached_path"] = bool(use_cache)
    res["forward_passes"] = (1 if use_cache else 0) + -(-len(rows) // batch_size)
    return res


def body_for(doc: str, n: int, i: int = 0) -> dict:
    return {"model": "jev-latest", "state": DOCS[doc],
            "questions": dict(list(QUESTIONS.items())[:n])}


# ------------------------------------------------------------------ HTTP
CLIENT = r'''
import http.client, json, sys, time, statistics as st
port, reps, bodies = int(sys.argv[1]), int(sys.argv[2]), json.loads(sys.stdin.read())
c = http.client.HTTPConnection("127.0.0.1", port)
out = {}
for name, body in bodies.items():
    raw = json.dumps(body).encode()
    xs, timing = [], []
    for i in range(reps + 5):
        t = time.perf_counter()
        c.request("POST", "/v1/systemone", body=raw, headers={"Content-Type": "application/json"})
        r = c.getresponse(); data = r.read()
        ms = (time.perf_counter() - t) * 1000
        assert r.status == 200, data[:300]
        if i >= 5:
            xs.append(ms); timing.append(r.getheader("x-profile"))
    out[name] = {"client_ms": xs, "server": timing}
print(json.dumps(out))
'''


def http_profile(s, scorer, reps: int, port: int, http_impl: str, loop: str) -> dict:
    import subprocess
    import uvicorn
    import serve.app as appmod
    from starlette.responses import JSONResponse as _JR

    rec = threading.local()
    marks: dict = {}

    new = hasattr(appmod, "finish")                 # async route: prepare / worker / finish
    orig_answer = appmod.answer_request
    orig_prepare = getattr(appmod, "prepare", None)
    orig_finish = getattr(appmod, "finish", None)
    resp_attr = "FastJSONResponse" if new else "JSONResponse"
    orig_resp = getattr(appmod, resp_attr)

    def timed_answer(*a, **kw):
        marks["handler_in"] = time.perf_counter()
        out = orig_answer(*a, **kw)
        marks["answer_out"] = time.perf_counter()
        marks["prepare_ms"], marks["infer_ms"] = out[2], out[3]
        return out

    def timed_prepare(*a, **kw):
        marks["handler_in"] = time.perf_counter()
        out = orig_prepare(*a, **kw)
        marks["prepared"] = time.perf_counter()
        marks["prepare_ms"] = (marks["prepared"] - marks["handler_in"]) * 1000
        return out

    def timed_finish(*a, **kw):
        t = time.perf_counter()
        marks["infer_ms"] = (t - marks["prepared"]) * 1000
        out = orig_finish(*a, **kw)
        marks["answer_out"] = time.perf_counter()
        return out
    if new:
        appmod.prepare, appmod.finish = timed_prepare, timed_finish
    else:
        appmod.answer_request = timed_answer

    class TimedJSON(orig_resp):
        def render(self, content):
            t = time.perf_counter()
            b = super().render(content)
            marks["render_ms"] = (time.perf_counter() - t) * 1000
            return b
    setattr(appmod, resp_attr, TimedJSON)
    app = appmod.create_app(scorer, served_model_name=s.name,
                            calibration=s.meta.get("calibration", "none"))

    @app.middleware("http")
    async def prof(request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        t1 = time.perf_counter()
        if request.url.path == "/v1/systemone":
            hin = marks.get("handler_in", t0)
            resp.headers["x-profile"] = json.dumps({
                "server_ms": (t1 - t0) * 1000,
                "before_handler_ms": (hin - t0) * 1000,
                "prepare_ms": marks.get("prepare_ms"), "infer_ms": marks.get("infer_ms"),
                "answer_ms": (marks["answer_out"] - hin) * 1000 - marks["prepare_ms"] - marks["infer_ms"],
                "render_ms": marks.get("render_ms"),
                "after_answer_ms": (t1 - marks["answer_out"]) * 1000})
        return resp

    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                         http=http_impl, loop=loop)
    server = uvicorn.Server(cfg)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    bodies = {f"{d}x{n}": body_for(d, n) for d, n in WORKLOADS}
    p = subprocess.run([sys.executable, "-c", CLIENT, str(port), str(reps)],
                       input=json.dumps(bodies), capture_output=True, text=True, timeout=900)
    server.should_exit = True
    th.join(timeout=10)
    appmod.answer_request = orig_answer
    if new:
        appmod.prepare, appmod.finish = orig_prepare, orig_finish
    setattr(appmod, resp_attr, orig_resp)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    raw = json.loads(p.stdout)
    res = {}
    for name, r in raw.items():
        srv = [json.loads(x) for x in r["server"]]
        keys = srv[0].keys()
        res[name] = {"client_p50_ms": p50(r["client_ms"]),
                     **{f"{k}_p50": p50([x[k] for x in srv]) for k in keys}}
        res[name]["outside_server_ms"] = round(res[name]["client_p50_ms"] - res[name]["server_ms_p50"], 3)
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--port", type=int, default=8831)
    ap.add_argument("--http-impl", default="h11")
    ap.add_argument("--loop", default="asyncio")
    a = ap.parse_args()
    import torch
    import transformers
    from serve.server import load_for_serving, make_scorer, warm_up
    from serve.wire import parse_questions
    s = load_for_serving(a.ckpt, device="cuda", dtype="bf16")
    scorer = make_scorer(s, 16)
    warm_up(scorer)
    traces = Path(a.traces); traces.mkdir(parents=True, exist_ok=True)
    res = {"torch": torch.__version__, "transformers": transformers.__version__,
           "gpu": torch.cuda.get_device_name(0), "applied": s.applied,
           "doc_tokens": {k: len(s.tok(v, add_special_tokens=False)["input_ids"]) for k, v in DOCS.items()},
           "workloads": {}}
    for doc, n in WORKLOADS:
        name = f"{doc}x{n}"
        body = body_for(doc, n)
        qs = parse_questions(body["questions"])
        state = body["state"]
        for _ in range(5):
            scorer(state, qs)
        xs = []
        for _ in range(a.repeat):
            sync(); t = time.perf_counter(); scorer(state, qs); sync()
            xs.append((time.perf_counter() - t) * 1000)
        stages = defaultdict(list)
        for _ in range(a.repeat):
            for k, v in stage_run(s, body).items():
                stages[k].append(v)
        stage = {k: (p50(v) if isinstance(v[0], float) else v[0]) for k, v in stages.items()}
        timed = [k for k in stage if k not in ("tokens_real", "tokens_padded", "prefix_tokens",
                                                "cached_path", "forward_passes",
                                                "collate.option_first_token_ids", "json_serialize_orjson")]
        # torch.profiler: one annotated call
        wrapped = annotate(s.model)
        from torch.profiler import ProfilerActivity, profile, record_function
        for _ in range(2):
            scorer(state, qs)
        sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function("scorer_call"):
                scorer(state, qs)
            sync()
        tp = traces / f"trace_{name}.json"
        prof.export_chrome_trace(str(tp))
        gpu = gpu_breakdown(tp)
        with open(tp, "rb") as f, gzip.open(str(tp) + ".gz", "wb") as g:
            g.write(f.read())
        tp.unlink()
        for mod, orig in wrapped:
            mod.forward = orig
        del s.model.cal_log_temperature
        res["workloads"][name] = {"questions": n, "doc_tokens": res["doc_tokens"][doc],
                                  "wall_p50_ms": p50(xs), "stages_ms": stage,
                                  "stages_total_ms": round(sum(stage[k] for k in timed), 3),
                                  "gpu": gpu}
        print(name, res["workloads"][name]["wall_p50_ms"], json.dumps(gpu["kernel_ms_by_module"]),
              flush=True)
    if a.http:
        res["http"] = {"impl": f"uvicorn http={a.http_impl} loop={a.loop}",
                       "results": http_profile(s, scorer, a.repeat, a.port, a.http_impl, a.loop)}
        print(json.dumps(res["http"], indent=1), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
