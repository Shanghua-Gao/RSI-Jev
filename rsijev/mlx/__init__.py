"""RSI-Jev on MLX (Apple Silicon): the released decision models without PyTorch.

    python scripts/convert_mlx.py v6.1-vl-4b ./rsi-jev-v6.1-vl-4b-mlx-8bit --bits 8 --group 64
    rsi-jev serve ./rsi-jev-v6.1-vl-4b-mlx-8bit --backend mlx
    Decider("./rsi-jev-v6.1-vl-4b-mlx-8bit", backend="mlx")

  convert.py   release package -> MLX checkpoint (tower in MLX affine 8-bit g64, or bf16)
  quant.py     the affine quantizer (bit for bit mx.quantize) and MLX's packing
  text.py      the Qwen3.5 hybrid tower (gated DeltaNet + gated attention), staged per layer
  heads.py     option pooling, the option_xattn scorer, calibration
  vision.py    the Qwen3.5 vision tower, image preparation, M-RoPE positions
  model.py     MLXDecisionModel: fixed exit, every exit, the adaptive cascade, prefix cache
  collate.py   numpy batching
  serving.py   the serve/ API on MLX (rsi-jev serve --backend mlx, Decider(backend="mlx"))

Needs `mlx` (Apple Silicon; the Linux CPU/CUDA builds run it too, for development and
tests), numpy, transformers (tokenizer and image processor only), safetensors and Pillow
for images. torch is needed only to convert, not to serve.
"""
