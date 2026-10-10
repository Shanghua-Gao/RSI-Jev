# RSI-Jev on a Mac (MLX)

RSI-Jev v6.1-VL 4B runs on Apple silicon through [MLX](https://github.com/ml-explore/mlx). There
are two builds. Both have the same API, effort levels and image support as the PyTorch release;
they store the weights in fewer bits.

| build | download | decoder layers | same answer as the bf16 release, layers 16 / 20 / 32 | accuracy, layer 32 | Decision Index 0.2.1, 16k sample |
|---|---|---|---|---|---|
| **[8-bit](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b-mlx-8bit)** | 6.33 GB | 8-bit, round to nearest | 99.1 / 99.5 / 99.3% | 83.5% | 51.39 |
| [4-bit](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b-mlx-4bit) | 3.87 GB | 4-bit, GPTQ | 97.1 / 97.9 / 97.9% | 83.3% | 50.99 |
| [bf16 release](https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b) (reference) | 9.68 GB | bf16 | – | 83.7% | 51.24 |

Agreement and accuracy are on 2,200 evaluation questions run by MLX; the Decision Index reads
are on the same rows and GPU type for all three. The 8-bit build gives the release's answers:
its differences are at the level of rounding, on questions where the top two options nearly tie.
The 4-bit build changes about one answer in fifty and costs about half a point of accuracy.
**Pick 8-bit unless memory is tight.**

## Install and run

```bash
pip install "rsi-jev[mlx,vision] @ git+https://github.com/Shanghua-Gao/RSI-Jev"
rsi-jev serve v6.1-vl-4b-mlx-8bit --backend mlx          # or v6.1-vl-4b-mlx-4bit
```

The server speaks the same HTTP API as the PyTorch one ([`serve/README.md`](../serve/README.md)),
images and `effort` included. From Python:

```python
from serve.decider import Decider
d = Decider("v6.1-vl-4b-mlx-8bit", backend="mlx")
d.decide("I was charged twice. Please refund.",
         {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}})
```

To build an MLX checkpoint from a release package yourself:

```bash
python scripts/convert_mlx.py v6.1-vl-4b out-8bit --bits 8 --group 64
python scripts/mlx_verify.py out-8bit --pkg <the release directory>   # bit-exact check with MLX
```

## Speed and memory

Not yet measured on Apple silicon; numbers will be added after a run of `scripts/mac_check.sh` (it prints median latency per effort, peak memory and agreement with the reference outputs). On MLX's CUDA backend the 8-bit build used about 7 GB at 1k input tokens, 12.7 GB at 8k and 26 GB at 32k.

## What is quantized

| part | 8-bit build | 4-bit build |
|---|---|---|
| decoder layers 1–32, every linear layer | 8-bit, groups of 64, round to nearest | 4-bit, groups of 32, GPTQ |
| token embeddings | bf16 | 8-bit, groups of 64 |
| vision tower | bf16 | bf16 |
| decision heads (layers 16, 20, 32) | fp32 | bf16 weights, computed in fp32 |
| per-exit temperatures, `auto` thresholds | as released | as released |

The 4-bit codes come from GPTQ ([Frantar et al., 2022](https://arxiv.org/abs/2210.17323)) in
MLX's affine format. The vision tower stays bf16 in both: 4 bits there costs about 7 points of
image agreement for 0.5 GB.

## How the builds were checked

- **The stored weights are the measured ones.** The 8-bit codes, scales and biases equal
  `mx.quantize` of the release's bf16 weights, bit for bit. The 4-bit build stores precomputed
  GPTQ codes as they are; MLX's `mx.dequantize` of every matrix equals the weights the quality
  numbers were measured with, bit for bit (`scripts/mlx_verify.py`).
- **MLX itself.** Each build was loaded with the MLX backend on Linux and run on the 2,200
  evaluation questions (text, a 151-option intent set, 300 image questions) and 300 held-out
  questions, against the bf16 release served by PyTorch.
- **The served paths.** `low`, `medium` and `high` answer at layers 16, 20 and 32; `auto` stops
  where the offline cascade stops. `rsi-jev serve --backend mlx` answered text and image
  requests.
- **Apple silicon.** `scripts/mac_check.sh` runs 200 questions per effort on the Mac and compares
  the answers with the bf16 release and with the same build run by MLX on Linux.

## Limitations

- Image requests use the PyTorch release's image processor when `torchvision` is installed
  (`rsi-jev[vision]`); without it, a PIL resize is used and pixels differ slightly.
- On image questions MLX and PyTorch differ a little more than on text (mean probability
  difference 0.008 against 0.004), in both builds and in the unquantized one: the vision tower's
  arithmetic differs between the two frameworks.
- Questions with images always run all 32 layers, as in the PyTorch release.
