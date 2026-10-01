---
title: RSI-Jev v4.0-VL
emoji: 🖼️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
suggested_hardware: l4x1
license: mit
short_description: Typed questions about an image, answered with probabilities
models:
  - shgao/rsi-jev-v4.0-vl-qwen3.5-2b
---

# RSI-Jev v4.0-VL

Upload an image, type a yes/no or multiple-choice question, and get a probability for every
allowed answer. The app calls the package's own `Decider`, so each answer is what
`POST /v1/systemone` on `rsi-jev serve` returns for the same request.

- Code, docs and the release record: <https://github.com/Shanghua-Gao/RSI-Jev>
- Weights: `shgao/rsi-jev-v4.0-vl-qwen3.5-2b` (set `RSIJEV_MODEL` to use another checkpoint)
- Hardware: a GPU. The app loads the model at startup (it runs in bf16 on CUDA), so pick a
  GPU with native bf16, such as an L4.

## Example images

| file | source | license |
|---|---|---|
| `examples/stop.jpg` | [Stop sign us.jpg](https://commons.wikimedia.org/wiki/File:Stop_sign_us.jpg), Dori, via Wikimedia Commons (downscaled) | public domain |
| `examples/dog.jpg` | [Boxer dog posing on the grass.jpg](https://commons.wikimedia.org/wiki/File:Boxer_dog_posing_on_the_grass.jpg), Joselodos, via Wikimedia Commons (downscaled) | CC0 |
| `examples/capsules.jpg` | VisA dataset, Zou et al. 2022 (Amazon Science), capsules test image 045, resized | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `examples/checkout.png`, `examples/chart.png` | drawn for this project | CC0 |

## Run it locally

```bash
pip install -r requirements.txt gradio
python app.py                          # RSIJEV_MODEL=/path/to/checkpoint python app.py
```

`test_app.py` checks the app on CPU with a stub in place of the model.
