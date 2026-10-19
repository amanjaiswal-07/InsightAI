# tests/test_sql_safety.py
"""LLM-generated SQL must be read-only and well-formed before it reaches the database."""

import pytest

from database.db_manager import _assert_read_only_sql
from agents.sql_analyst import SQLAnalystAgent


@pytest.mark.parametrize("sql", [
    "SELECT * FROM sales",
    "  select Region, SUM(Revenue) FROM sales GROUP BY Region",
    "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
    "SELECT REPLACE(Region, 'N', 'n') FROM sales",   # REPLACE() function is fine
    "SELECT * FROM sales;",                          # one trailing semicolon is fine
])
def test_read_only_queries_are_allowed(sql):
    _assert_read_only_sql(sql)


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales",
    "DELETE FROM sales",
    "SELECT 1; DROP TABLE sales",
    "WITH t AS (SELECT 1) INSERT INTO sales VALUES (1)",
    "REPLACE INTO sales VALUES (1)",
    "UPDATE sales SET Revenue = 0",
    "ATTACH DATABASE 'x.db' AS x",
    "PRAGMA writable_schema = 1",
])
def test_mutating_queries_are_rejected(sql):
    with pytest.raises(ValueError):
        _assert_read_only_sql(sql)


def test_database_rejects_mutation_even_through_execute_query(db):
    db.get_connection().execute("CREATE TABLE t (x INTEGER)").connection.close()
    with pytest.raises(ValueError):
        db.execute_query("DROP TABLE t")
    assert "t" in db.list_tables()


@pytest.mark.parametrize("raw, expected", [
    ("```sql\nSELECT * FROM t;\n```", "SELECT * FROM t"),
    ("SELECT MAX Revenue FROM t", "SELECT MAX(Revenue) FROM t"),
    ("SELECT COUNT(*) FROM t", "SELECT COUNT(*) FROM t"),
    ("SELECT Region, MIN Revenue FROM t", "SELECT Region, MIN(Revenue) FROM t"),
    # Regression: the alias/column "total" must not be mistaken for TOTAL()
    ("SELECT SUM(Revenue) AS total FROM t", "SELECT SUM(Revenue) AS total FROM t"),
    ("SELECT total FROM t ORDER BY total DESC", "SELECT total FROM t ORDER BY total DESC"),
])
def test_sanitize_sql(raw, expected):
    assert SQLAnalystAgent._sanitize_sql(raw) == expected


def test_not_available_sentinel_is_detected():
    rows = [{"message": "This information is not available in the uploaded data"}]
    assert SQLAnalystAgent._is_not_available_result(rows)
    assert not SQLAnalystAgent._is_not_available_result([{"total": 5}])
