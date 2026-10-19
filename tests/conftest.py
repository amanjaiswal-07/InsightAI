# tests/conftest.py
"""
Shared fixtures for the offline test suite.

These tests never call Groq or download models: agents are built with
object.__new__ (skipping __init__, which would create LLM clients and load
embeddings) and the pieces a test needs are attached by hand.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Keep any runtime files the imported modules create out of the repo's data/
os.environ.setdefault("LOG_FILE", str(ROOT / "logs" / "test.log"))


@pytest.fixture
def db(tmp_path):
    """A DatabaseManager backed by a throwaway SQLite file."""
    from database.db_manager import DatabaseManager
    return DatabaseManager(config={"type": "sqlite", "path": str(tmp_path / "test.db")})


@pytest.fixture
def sales_csv(tmp_path):
    """A small monthly sales CSV with a date, a category and numeric columns."""
    path = tmp_path / "Sales Data.csv"
    path.write_text(
        "OrderID,Month,Region,Revenue,Units\n"
        "O1,2024-01-01,North,1000,10\n"
        "O2,2024-02-01,South,1500,12\n"
        "O3,2024-03-01,North,1200,11\n"
        "O4,2024-04-01,South,1800,15\n"
        "O5,2024-05-01,North,1700,14\n"
        "O6,2024-06-01,South,2100,18\n",
        encoding="utf-8",
    )
    return path


class FakeChain:
    """Stand-in for a LangChain runnable: returns canned outputs and records inputs."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, inputs):
        self.calls.append(inputs)
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
