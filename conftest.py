def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_llm: test needs a running local LLM server (e.g. sglang)",
    )
