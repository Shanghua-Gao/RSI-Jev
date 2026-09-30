from rsijev.fit import FitConfig, _lr_lambdas


def test_lower_layers_follow_the_tower_schedule():
    cfg = FitConfig(steps=100, warmup=0.1, base_schedule="cosine", head_schedule="cosine")
    groups = [{"name": "head"}, {"name": "mix"}, {"name": "base"}, {"name": "base_lower"}]
    head, mix, base, lower = _lr_lambdas(groups, cfg)
    for s in (0, 5, 10, 50, 99):
        assert lower(s) == base(s)
    assert base(99) < 0.01          # decayed, not held constant
    assert mix(99) == 1.0           # other groups only warm up
