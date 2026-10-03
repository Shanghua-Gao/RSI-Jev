"""CPU check of the Space with a stub scorer in place of the model (no weights, no GPU).

    PYTHONPATH=. python -m pytest -q space/test_app.py      # from a clone, with gradio installed

The stub goes through the same Decider request path as the model (validation, image
decoding, the answer shapes); only the probabilities are fake.
"""
import sys
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app                                              # noqa: E402
from serve.decider import Decider                       # noqa: E402
from serve.images import image_limits                   # noqa: E402

seen = []


def stub(state, questions, images=None):
    seen.append((state, len(images or [])))
    return [[1.0 / len(q.options)] * (len(q.options) - 1) + [1.0 / len(q.options)]
            for q in questions], 123


@pytest.fixture(scope="module")
def decider():
    return Decider.from_scorer(stub, name="stub", images=image_limits(
        {"image_token_budget": 1024, "min_tokens_per_image": 64}))


def test_examples_are_there_and_answer(decider):
    answer = app.make_answer(decider)
    from PIL import Image
    for path, context, question, kind, options in app.EXAMPLES:
        assert path is None or Path(path).exists(), path
        image = Image.open(path) if path else None
        probs, line, out = answer(image, context, question, kind, options)
        assert abs(sum(probs.values()) - 1) < 1e-6
        assert out["answers"]["answer"]["type"] == ("noul" if kind == "yes/no" else "choice")
        if kind == "choice":
            assert list(probs) == list(app.parse_options(options))
        if path:
            assert seen[-1][1] == 1 and "<image>" in seen[-1][0]
        else:                                       # text only: no image, no placeholder
            assert seen[-1][1] == 0 and "<image>" not in seen[-1][0]


def test_bad_forms_are_refused(decider):
    answer = app.make_answer(decider)
    from PIL import Image
    im = Image.new("RGB", (32, 32))
    for args in [(None, "", "q?", "yes/no", ""), (None, "  ", "q?", "yes/no", ""), (im, "", " ", "yes/no", ""),
                 (im, "", "q?", "choice", "only"), (im, "", "q?", "choice", "a\na")]:
        with pytest.raises(gr.Error):
            answer(*args)


def test_the_app_serves_over_http(decider):
    client_mod = pytest.importorskip("gradio_client")
    demo = app.build_demo(decider)
    demo.queue().launch(prevent_thread_lock=True, server_port=7869, quiet=True)
    try:
        c = client_mod.Client("http://127.0.0.1:7869/", verbose=False)
        ex = next(e for e in app.EXAMPLES if e[0] and e[0].endswith("checkout.png"))
        probs, line, raw = c.predict(client_mod.handle_file(ex[0]), *ex[1:], api_name="/answer")
        assert raw["answers"]["answer"]["choice"] in ("paid", "declined", "pending")
        assert "input tokens" in line
        text = next(e for e in app.EXAMPLES if e[0] is None)
        probs, line, raw = c.predict(None, *text[1:], api_name="/answer")
        assert raw["answers"]["answer"]["type"] == "noul"
        assert {c["label"] for c in probs["confidences"]} == {"yes", "no"}
    finally:
        demo.close()
