# utils/llm.py
"""
Single factory for the Groq chat model used by every agent.

Centralising it means model name, timeout and retry policy are configured in
one place (.env) instead of in each agent. Retries matter on Groq's free tier:
a burst of calls hits the tokens-per-minute limit (HTTP 429), and the client
waits for the "retry after" interval and tries again instead of failing the
user's question.
"""

from typing import Optional

from langchain_groq import ChatGroq

from config import settings


def make_llm(temperature: float, max_tokens: Optional[int] = None) -> ChatGroq:
    """Return a configured ChatGroq client."""
    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a key "
            "from https://console.groq.com/keys"
        )
    kwargs = {
        "api_key":     settings.GROQ_API_KEY,
        "model":       settings.LLM_MODEL,
        "temperature": temperature,
        "timeout":     settings.LLM_TIMEOUT,
        "max_retries": settings.LLM_MAX_RETRIES,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatGroq(**kwargs)
