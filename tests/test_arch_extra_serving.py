"""spec.arch_extra at serving: training-only keys are ignored, unknown keys refused."""
import pytest

from serve.release import TRAINING_ONLY_ARCH_KEYS, serving_arch_extra


def test_training_only_keys_dropped():
    spec = {"arch_extra": {"exit_layer": 32, "aux_exits": [12, 16, 20], "freeze_lower_n": 8, "xattn_combine": "mlp"}}
    assert serving_arch_extra(spec) == {"exit_layer": 32, "aux_exits": [12, 16, 20], "xattn_combine": "mlp"}
    assert "freeze_lower_n" in TRAINING_ONLY_ARCH_KEYS


def test_unknown_key_refused():
    with pytest.raises(RuntimeError, match="does not know"):
        serving_arch_extra({"arch_extra": {"exit_layer": 20, "made_up_key": 1}})


def test_empty():
    assert serving_arch_extra({}) == {} and serving_arch_extra({"arch_extra": None}) == {}
