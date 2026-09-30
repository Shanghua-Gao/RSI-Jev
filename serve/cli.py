"""`rsi-jev`: the command installed with the package.

    rsi-jev serve shgao/rsi-jev-v3.0-qwen3.5-2b [--port 8000] [--profile agent|server]
    rsi-jev bench [v3.0-2b]
    rsi-jev download v3.0-2b         # fetch into the Hugging Face cache, print the path
    rsi-jev env                      # torch, device and kernels, without loading a model

Each subcommand is a thin front for code that also runs from a clone as
`python scripts/<name>.py`.
"""
from __future__ import annotations

import argparse


def _download(a: argparse.Namespace) -> int:
    from serve.release import resolve_ckpt
    print(resolve_ckpt(a.model, revision=a.revision))
    return 0


def _env(a: argparse.Namespace) -> int:
    import torch
    from serve.runtime import best_device, default_dtype_name, startup_lines
    device = a.device or best_device(torch)
    for line in startup_lines(torch=torch, device=device,
                              dtype_name=default_dtype_name(device)):
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from serve.bench import add_bench_args, run
    from serve.server import add_serve_args, serve
    ap = argparse.ArgumentParser(prog="rsi-jev", description="Run RSI-Jev decision models.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="serve a release behind the Jev-compatible HTTP API")
    add_serve_args(p, positional=True)
    p.set_defaults(func=serve)

    p = sub.add_parser("bench", help="time the served path on this machine")
    add_bench_args(p)
    p.set_defaults(func=run)

    p = sub.add_parser("download", help="download a release into the Hugging Face cache "
                                        "and print its local path")
    p.add_argument("model", help="Hugging Face repo id or alias, e.g. v3.0-2b")
    p.add_argument("--revision", default=None)
    p.set_defaults(func=_download)

    p = sub.add_parser("env", help="print torch, device and kernel status, no model loaded")
    p.add_argument("--device", default=None)
    p.set_defaults(func=_env)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
