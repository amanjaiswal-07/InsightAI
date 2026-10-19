# tests/test_llm_factory.py
"""Every agent gets its LLM client from one factory with timeout and retries set."""

from config import settings
from utils.llm import make_llm


def test_factory_applies_timeout_and_retries():
    llm = make_llm(temperature=0.2, max_tokens=123)
    assert llm.max_retries == settings.LLM_MAX_RETRIES
    assert llm.request_timeout == settings.LLM_TIMEOUT
    assert llm.max_tokens == 123
    assert llm.temperature == 0.2


def test_router_budget_leaves_room_for_reasoning_tokens():
    """Regression: max_tokens=10 made the reasoning model return '' every time."""
    from tools.csv_query_router import CSVQueryRouter
    assert CSVQueryRouter()._llm.max_tokens >= 256
