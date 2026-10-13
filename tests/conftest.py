import os

import pytest
from dotenv import load_dotenv

load_dotenv(".env")


def _has_llm_key():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    return provider == "ollama"


def pytest_collection_modifyitems(config, items):
    if _has_llm_key():
        return
    skip = pytest.mark.skip(reason="no LLM API key set")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip)
