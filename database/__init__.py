# database/__init__.py
"""
Database access: DatabaseManager (read-only query path for LLM SQL) and sample data.
"""

from .db_manager import DatabaseManager
from .sample_data import create_sample_database

__all__ = ['DatabaseManager', 'create_sample_database']