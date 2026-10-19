# auth.py
"""
API-key authentication and rate limiting shared by main.py.

- Rate limiting: slowapi keyed on the client IP. Limits are declared per
  endpoint in main.py with @limiter.limit("N/minute").
- API keys: set API_KEYS=key1,key2 in .env and send one in the X-Api-Key
  header. When API_KEYS is empty, auth is disabled (local development); a
  warning is logged at startup so this is never silent.
"""

from typing import Optional

from fastapi import Header, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import settings
from utils.logger import logger

limiter = Limiter(key_func=get_remote_address)

VALID_KEYS = set(settings.API_KEYS)
if not VALID_KEYS:
    logger.warning("⚠️  API_KEYS not set — API key auth is DISABLED (fine for local dev)")


async def verify_key(x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key")):
    """FastAPI dependency: raise 401 if auth is enabled and the key is missing or wrong."""
    if not VALID_KEYS:
        return
    if not x_api_key or x_api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
