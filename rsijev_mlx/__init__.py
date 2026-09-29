"""RSI-Jev on MLX (Apple Silicon). See the "Apple Silicon (MLX)" section of README.md.

    from rsijev_mlx import load, decide
    model, tok, meta = load("shgao/rsi-jev-v3.0-qwen3.5-2b")
    decide(model, tok, meta, state, questions)
"""
from .infer import decide, score_questions  # noqa: F401
from .model import load  # noqa: F401
