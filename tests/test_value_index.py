# tests/test_value_index.py
"""ColumnValueIndex rewrites SQL string literals to the exact values stored in the table."""

import pandas as pd

from tools.column_value_index import ColumnValueIndex


def _index():
    df = pd.DataFrame({
        "Region": ["North", "South", "Regions"],    # a value that looks like a column name
        "Month": ["Jan", "Feb", "Mar"],
        "Revenue": [1, 2, 3],
    })
    idx = ColumnValueIndex()
    idx.build_from_dataframe(df, "t")
    return idx


def test_case_and_month_variants_are_normalised():
    idx = _index()
    sql = "SELECT * FROM t WHERE Region = 'north' AND Month = 'January'"
    assert idx.rewrite_sql(sql, "t") == "SELECT * FROM t WHERE Region = 'North' AND Month = 'Jan'"


def test_double_quoted_identifiers_are_never_rewritten():
    """Regression: "Region" (an identifier) used to be fuzzy-matched to the value 'Regions'."""
    idx = _index()
    sql = 'SELECT "Region", SUM("Revenue") FROM t GROUP BY "Region"'
    assert idx.rewrite_sql(sql, "t") == sql


def test_unknown_literals_and_numbers_are_left_alone():
    idx = _index()
    sql = "SELECT * FROM t WHERE Region = 'Antarctica' AND Revenue > '2'"
    assert idx.rewrite_sql(sql, "t") == sql
