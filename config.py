# config.py
"""
Central configuration for InsightAI.

Every setting is read from environment variables (loaded from a local .env
file via python-dotenv) with a sensible default, so the app runs with only
GROQ_API_KEY set. See .env.example for the full list.

Two access styles are supported because both are used across the codebase:
    from config import settings          # settings.LLM_MODEL
    from config import LLM_MODEL         # module-level constant
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str, default: str) -> List[str]:
    return [v.strip() for v in os.getenv(name, default).split(",") if v.strip()]


@dataclass
class Settings:
    """Application settings, resolved once at import time."""

    # Database (uploaded CSVs become tables here; SQLite is the supported backend)
    DB_TYPE: str = os.getenv("DB_TYPE", "sqlite")
    DB_PATH: str = os.getenv("DB_PATH", "./data/business.db")
    CHAT_DB_PATH: str = os.getenv("CHAT_DB_PATH", "./data/chat_history.db")

    # LLM (Groq)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    # Seconds before a single LLM request is abandoned, and how many times a
    # failed request (429 rate limit, 5xx, timeout) is retried with backoff.
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "60"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "5"))

    # RAG
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_stores")

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS: List[str] = field(default_factory=lambda: _env_list("CORS_ORIGINS", "*"))
    # Comma-separated API keys accepted in the X-Api-Key header. Empty = auth disabled.
    API_KEYS: List[str] = field(default_factory=lambda: _env_list("API_KEYS", ""))

    # Optional Redis cache for chat history (falls back to in-process fakeredis)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/app.log")

    # CSV hybrid processing (CSV → SQL table + RAG narrative chunks)
    CSV_TABLE_PREFIX: str = os.getenv("CSV_TABLE_PREFIX", "upload_")
    ENABLE_HYBRID_CSV: bool = _env_bool("ENABLE_HYBRID_CSV", "True")


settings = Settings()

# ── Database configuration ────────────────────────────────────────────────────

DB_TYPE = settings.DB_TYPE
DB_PATH = settings.DB_PATH

# Postgres/MySQL connection settings are read by DatabaseManager. Note that the
# upload pipeline (table import/cleanup) is SQLite-specific; see README.
POSTGRES_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "business_db"),
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

MYSQL_CONFIG = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "port":     int(os.getenv("MYSQL_PORT", "3306")),
    "database": os.getenv("MYSQL_DB", "business_db"),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
}


def get_db_config():
    """Return the connection config for the active DB_TYPE."""
    if DB_TYPE == "sqlite":
        return {"type": "sqlite", "path": DB_PATH}
    if DB_TYPE == "postgresql":
        return {"type": "postgresql", **POSTGRES_CONFIG}
    if DB_TYPE == "mysql":
        return {"type": "mysql", **MYSQL_CONFIG}
    raise ValueError(f"Unsupported DB_TYPE: {DB_TYPE}")


# ── Module-level constants (for `from config import X`) ──────────────────────

GROQ_API_KEY      = settings.GROQ_API_KEY
LLM_MODEL         = settings.LLM_MODEL
TEMPERATURE       = settings.TEMPERATURE
MAX_TOKENS        = settings.MAX_TOKENS
EMBEDDING_MODEL   = settings.EMBEDDING_MODEL
VECTOR_STORE_PATH = settings.VECTOR_STORE_PATH
HOST              = settings.HOST
PORT              = settings.PORT
LOG_LEVEL         = settings.LOG_LEVEL
LOG_FILE          = settings.LOG_FILE
CSV_TABLE_PREFIX  = settings.CSV_TABLE_PREFIX
ENABLE_HYBRID_CSV = settings.ENABLE_HYBRID_CSV

# ── Make sure runtime folders exist ──────────────────────────────────────────

Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
Path(settings.CHAT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
if DB_TYPE == "sqlite":
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def print_config():
    """Print the active configuration (never prints secrets)."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Database:        {DB_TYPE} ({DB_PATH if DB_TYPE == 'sqlite' else get_db_config()['host']})")
    print(f"LLM model:       {LLM_MODEL}")
    print(f"Groq key set:    {bool(GROQ_API_KEY)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"CSV hybrid:      {ENABLE_HYBRID_CSV}")
    print(f"API auth:        {'on' if settings.API_KEYS else 'off'}")
    print(f"Server:          {HOST}:{PORT}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
