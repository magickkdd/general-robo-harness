def pytest_configure(config):
    config.addinivalue_line("markers", "slow: builds a real sim env (needs vulkan)")
