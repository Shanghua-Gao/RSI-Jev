"""The background model behind `decide` (see serve/autostart.py).

    python -m serve.daemon fetch [--vision]    download the model
    python -m serve.daemon run [--vision]      serve it on ~/.rsi-jev/model.sock until idle
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import threading
import time
from pathlib import Path

from serve.autostart import home, model_ref

NOT_NEEDED = ["vocab.json", "merges.txt", "tokenizer.json", "video_preprocessor_config.json", "chat_template.jinja"]
VISION_FILE = "visual.safetensors"


def _write(name: str, data: dict) -> None:
    tmp = home() / f".{name}.tmp"
    tmp.write_text(json.dumps(data))
    os.replace(tmp, home() / name)


def fetch(vision: bool) -> Path:
    """The model on disk: without its vision tower (0.2 GB) unless images are wanted."""
    from serve.release import repo_id_for
    ref = model_ref()
    path = Path(ref).expanduser()
    if not path.is_dir():
        from huggingface_hub import snapshot_download
        repo = repo_id_for(ref)
        if repo is None:
            raise SystemExit(f"{ref!r} is not a model directory or a Hugging Face repo")
        path = Path(snapshot_download(repo, ignore_patterns=NOT_NEEDED + ([] if vision else [VISION_FILE])))
    _write("model.json", {"ref": ref, "path": str(path), "vision": (path / VISION_FILE).is_file()})
    return path


class IdleWatch:
    """ASGI wrapper recording when the last question ended. Health checks do not count."""

    def __init__(self, app):
        self.app, self.last, self.busy = app, time.monotonic(), 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path", "").startswith("/health"):
            return await self.app(scope, receive, send)
        self.busy += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.busy -= 1
            self.last = time.monotonic()

    def idle_for(self) -> float:
        return 0.0 if self.busy else time.monotonic() - self.last


def run(vision: bool) -> int:
    import uvicorn
    from serve.server import add_serve_args, build_mlx_app
    model = json.loads((home() / "model.json").read_text())
    ap = argparse.ArgumentParser()
    add_serve_args(ap, positional=True)
    a = ap.parse_args([model["path"], "--backend", "mlx", "--no-warmup", "--served-model-name", model["ref"]])
    a.vision = bool(vision)
    watch = IdleWatch(build_mlx_app(a, model["path"]))
    sock = home() / "model.sock"
    server = uvicorn.Server(uvicorn.Config(watch, uds=str(sock), log_level="warning"))
    limit = float(os.environ.get("RSIJEV_IDLE_MINUTES") or 10) * 60

    def exit_when_idle():
        while not server.should_exit:
            time.sleep(min(1.0, max(limit / 10, 0.05)))
            if watch.idle_for() > limit:
                server.should_exit = True

    if limit > 0:
        threading.Thread(target=exit_when_idle, daemon=True).start()
    _write("daemon.json", {"pid": os.getpid(), "vision": bool(vision), "model": model["ref"]})
    try:
        server.run()
    finally:
        for f in (sock, home() / "daemon.json"):
            with contextlib.suppress(OSError):
                f.unlink()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m serve.daemon")
    ap.add_argument("command", choices=("fetch", "run"))
    ap.add_argument("--vision", action="store_true")
    a = ap.parse_args()
    if a.command == "fetch":
        fetch(a.vision)
        return 0
    return run(a.vision)


if __name__ == "__main__":
    raise SystemExit(main())
