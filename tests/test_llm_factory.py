# tests/test_llm_factory.py
"""Every agent gets its LLM client from one factory with timeout and retries set."""

import pytest

from config import settings
from utils.llm import make_llm


@pytest.fixture
def fake_key(monkeypatch):
    """Constructing a client needs a key, but these tests never call the API."""
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key-not-real")


def test_factory_applies_timeout_and_retries(fake_key):
    llm = make_llm(temperature=0.2, max_tokens=123)
    assert llm.max_retries == settings.LLM_MAX_RETRIES
    assert llm.request_timeout == settings.LLM_TIMEOUT
    assert llm.max_tokens == 123
    assert llm.temperature == 0.2


def test_missing_key_gives_a_clear_setup_error(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        make_llm(temperature=0)


def test_router_budget_leaves_room_for_reasoning_tokens(fake_key):
    """Regression: max_tokens=10 made the reasoning model return '' every time."""
    from tools.csv_query_router import CSVQueryRouter
    assert CSVQueryRouter()._llm.max_tokens >= 256
