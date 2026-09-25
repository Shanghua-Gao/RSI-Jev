def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: loads real weights; run explicitly, not in a quick sweep")
