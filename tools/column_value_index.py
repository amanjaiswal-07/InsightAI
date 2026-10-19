# tools/column_value_index.py
"""
Column value index for SQL literal normalisation.

Built from each uploaded table. When the LLM writes WHERE Region = 'north'
but the data says 'North' (or uses an abbreviation / month variant), the
literal is rewritten to the exact stored value before the query runs.
"""

import re
import difflib
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
from utils.logger import logger


# ================================================================
# UNIVERSAL ABBREVIATION EXPANDER
# ================================================================
# These are NOT domain-specific hardcodes.
# They are general English abbreviation rules that apply universally.
# "jan" is a prefix-abbreviation of "january" by these rules.
# "avg" is a standard English contraction of "average".
# The key insight: we expand the *query* to full form, then fuzzy-match
# against full-form versions of the *data values*.

_COMMON_CONTRACTIONS: Dict[str, str] = {
    # Statistical
    "avg": "average", "min": "minimum", "max": "maximum",
    "std": "standard deviation", "var": "variance",
    "cnt": "count", "num": "number", "pct": "percent",
    "qty": "quantity", "amt": "amount", "rev": "revenue",
    # Business
    "emp": "employee", "dept": "department", "mgr": "manager",
    "cust": "customer", "prod": "product", "trans": "transaction",
    "acct": "account", "addr": "address", "desc": "description",
    # Time
    "yr": "year", "mo": "month", "wk": "week", "dy": "day",
    "hr": "hour", "min": "minute", "sec": "second",
    "q1": "quarter 1", "q2": "quarter 2", "q3": "quarter 3", "q4": "quarter 4",
    "fy": "fiscal year", "ytd": "year to date", "mtd": "month to date",
}


def _normalize_token(s: str) -> str:
    """
    Produce a canonical form of a string for fuzzy matching.
    Lowercases, removes punctuation/underscores/extra spaces.
    'Avg_Deal_Size' → 'avg deal size'
    'January'       → 'january'
    'Q1-2024'       → 'q1 2024'
    """
    s = s.lower().strip()
    s = re.sub(r'[_\-/\\|]', ' ', s)      # separators → space
    s = re.sub(r'[^\w\s]', '', s)          # remove remaining punctuation
    s = re.sub(r'\s+', ' ', s).strip()     # collapse whitespace
    return s


def _expand_abbreviations(token: str) -> str:
    """
    Expand known abbreviations in a normalized token.
    Operates word-by-word so 'avg deal size' → 'average deal size'.
    """
    words = token.split()
    expanded = []
    for word in words:
        expanded.append(_COMMON_CONTRACTIONS.get(word, word))
    return ' '.join(expanded)


def _try_parse_as_month(value: str) -> Optional[str]:
    """
    Try to interpret a string as a month using datetime.strptime.
    Returns the full month name if successful, None otherwise.
    Handles: 'Jan', 'jan', 'JAN', 'January', '1', '01', '2024-01', etc.
    No hardcoded month lists — strptime does the work.
    """
    v = value.strip()
    formats_to_try = [
        '%b',        # Jan, Feb, ...
        '%B',        # January, February, ...
        '%m',        # 01, 02, ...
        '%Y-%m',     # 2024-01
        '%m/%Y',     # 01/2024
        '%b %Y',     # Jan 2024
        '%B %Y',     # January 2024
    ]
    for fmt in formats_to_try:
        try:
            dt = datetime.strptime(v, fmt)
            return dt.strftime('%B')   # always return full month name as canonical
        except ValueError:
            continue
    return None


class ColumnValueIndex:
    """
    Builds and queries a per-table index of actual column names and values.

    Usage:
        # At ingest time (once per table):
        idx = ColumnValueIndex()
        idx.build_from_dataframe(df, table_name="upload_abc_sales")

        # At query time:
        rewritten_sql = idx.rewrite_sql(sql, table_name="upload_abc_sales")

    The index stores:
        table_name → {
            "columns": { canonical_col_name → actual_col_name },
            "values":  { col_name → { canonical_value → actual_value } }
        }
    """

    def __init__(self):
        # { table_name: { "columns": {...}, "values": {...} } }
        self._index: Dict[str, Dict[str, Any]] = {}
        logger.info("📇 ColumnValueIndex initialized")

    # ================================================================
    # INDEX BUILDING (called at ingest time)
    # ================================================================

    def build_from_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Scan a DataFrame and build the lookup index for that table.
        Call this right after the CSV is imported into SQLite.

        Args:
            df:         The DataFrame (same one that was imported)
            table_name: The SQLite table name it was imported as
        """
        column_index: Dict[str, str] = {}
        value_index: Dict[str, Dict[str, str]] = {}

        # ── 1. Index column names ────────────────────────────────
        for col in df.columns:
            # Map multiple canonical forms → actual column name
            canonical_plain = _normalize_token(col)
            canonical_expanded = _expand_abbreviations(canonical_plain)

            column_index[canonical_plain] = col
            if canonical_expanded != canonical_plain:
                column_index[canonical_expanded] = col

            # Also map each individual word that appears in the column name
            for word in canonical_expanded.split():
                if len(word) > 2:   # skip tiny words
                    column_index.setdefault(word, col)

            # ── 2. Index unique values in this column ────────────
            col_vals: Dict[str, str] = {}

            # Only index string/object columns (not numerics)
            if df[col].dtype == object or df[col].dtype.name == 'string':
                unique_vals = df[col].dropna().astype(str).unique()

                for val in unique_vals:
                    actual = str(val)

                    # Canonical form 1: raw normalized
                    c1 = _normalize_token(actual)
                    col_vals[c1] = actual

                    # Canonical form 2: with abbreviation expansion
                    c2 = _expand_abbreviations(c1)
                    if c2 != c1:
                        col_vals[c2] = actual

                    # Canonical form 3: try month interpretation
                    month_full = _try_parse_as_month(actual)
                    if month_full:
                        # Store: "january" → "Jan" (or whatever actual is)
                        # Also store all prefix lengths ≥ 3
                        mf_lower = month_full.lower()
                        col_vals[mf_lower] = actual
                        # Prefixes: 'jan', 'janu', 'janua', etc.
                        for length in range(3, len(mf_lower)):
                            col_vals[mf_lower[:length]] = actual

                value_index[col] = col_vals

        self._index[table_name] = {
            "columns": column_index,
            "values": value_index,
        }

        logger.info(
            f"📇 Indexed '{table_name}': "
            f"{len(column_index)} column forms, "
            f"{sum(len(v) for v in value_index.values())} value forms"
        )

    def has_table(self, table_name: str) -> bool:
        return table_name in self._index

    def get_indexed_tables(self) -> List[str]:
        return list(self._index.keys())

    # ================================================================
    # VALUE LOOKUP
    # ================================================================

    def resolve_value(self, query_value: str, table_name: str,
                      column_hint: Optional[str] = None) -> str:
        """
        Given a string literal from the SQL query, return the actual
        value that exists in the database.

        Strategy:
          1. Exact match (already correct)
          2. Canonical match (normalized form maps to actual)
          3. Fuzzy match (difflib against all known values)

        Args:
            query_value:  The string as it appears in the SQL, e.g. 'January'
            table_name:   Which table to look in
            column_hint:  If we know which column (from context), prioritize it

        Returns:
            The actual database value, or query_value unchanged if no match found.
        """
        if table_name not in self._index:
            return query_value

        value_maps = self._index[table_name]["values"]
        canonical_query = _normalize_token(query_value)
        expanded_query = _expand_abbreviations(canonical_query)

        # Also try month parsing of the query value
        month_full = _try_parse_as_month(query_value)
        month_canonical = month_full.lower() if month_full else None

        # ── Priority 1: column-specific lookup ───────────────────
        cols_to_check = []
        if column_hint and column_hint in value_maps:
            cols_to_check.append(column_hint)
        # Then check all other columns
        cols_to_check += [c for c in value_maps if c != column_hint]

        for col in cols_to_check:
            vmap = value_maps[col]

            # Exact match
            if query_value in vmap.values():
                return query_value

            # Canonical match (plain)
            if canonical_query in vmap:
                resolved = vmap[canonical_query]
                if resolved != query_value:
                    logger.info(f"🔄 Value resolved: '{query_value}' → '{resolved}' (canonical)")
                return resolved

            # Canonical match (expanded abbreviations)
            if expanded_query in vmap:
                resolved = vmap[expanded_query]
                logger.info(f"🔄 Value resolved: '{query_value}' → '{resolved}' (abbreviation)")
                return resolved

            # Month canonical match
            if month_canonical and month_canonical in vmap:
                resolved = vmap[month_canonical]
                logger.info(f"🔄 Value resolved: '{query_value}' → '{resolved}' (month parse)")
                return resolved

        # ── Priority 2: fuzzy match across all value maps ────────
        all_actual_values = []
        for col in cols_to_check:
            all_actual_values.extend(value_maps[col].values())

        if all_actual_values:
            # Use difflib to find closest match
            candidates = difflib.get_close_matches(
                canonical_query,
                [_normalize_token(v) for v in all_actual_values],
                n=1,
                cutoff=0.75  # require 75% similarity
            )
            if candidates:
                # Map back from normalized candidate to actual value
                for v in all_actual_values:
                    if _normalize_token(v) == candidates[0]:
                        logger.info(f"🔄 Value resolved: '{query_value}' → '{v}' (fuzzy)")
                        return v

        # No match found — return unchanged
        logger.debug(f"⚠️ No resolution for value '{query_value}' in '{table_name}'")
        return query_value

    def resolve_column(self, query_col: str, table_name: str) -> str:
        """
        Given a column name (or user's natural language column reference),
        return the actual column name in the database.

        e.g. 'average deal size' → 'Avg_Deal_Size'
             'avg_deal_size'     → 'Avg_Deal_Size'
        """
        if table_name not in self._index:
            return query_col

        col_map = self._index[table_name]["columns"]
        canonical = _normalize_token(query_col)
        expanded = _expand_abbreviations(canonical)

        if canonical in col_map:
            resolved = col_map[canonical]
            if resolved != query_col:
                logger.info(f"🔄 Column resolved: '{query_col}' → '{resolved}'")
            return resolved

        if expanded in col_map:
            resolved = col_map[expanded]
            logger.info(f"🔄 Column resolved: '{query_col}' → '{resolved}' (expanded)")
            return resolved

        # Fuzzy fallback
        actual_cols = list(set(col_map.values()))
        candidates = difflib.get_close_matches(
            canonical,
            [_normalize_token(c) for c in actual_cols],
            n=1, cutoff=0.7
        )
        if candidates:
            for c in actual_cols:
                if _normalize_token(c) == candidates[0]:
                    logger.info(f"🔄 Column resolved: '{query_col}' → '{c}' (fuzzy)")
                    return c

        return query_col

    # ================================================================
    # SQL REWRITER (the main public interface)
    # ================================================================

    def rewrite_sql(self, sql: str, table_name: str) -> str:
        """
        Rewrite a SQL string so that all string literals match the actual
        values stored in the database, and column references match actual
        column names.

        This is called AFTER the LLM generates SQL, BEFORE execution.

        Args:
            sql:        The LLM-generated SQL string
            table_name: The primary table being queried

        Returns:
            Rewritten SQL with corrected string literals and column names.
        """
        if table_name not in self._index:
            return sql

        original_sql = sql

        # ── Step 1: Resolve string literals in WHERE / IN clauses ─
        # Matches single-quoted strings in SQL: 'January', 'jan', etc.
        def replace_literal(match: re.Match) -> str:
            value = match.group(1).replace("''", "'")

            # Skip numeric strings
            if re.match(r'^\d+(\.\d+)?$', value):
                return match.group(0)

            resolved = self.resolve_value(value, table_name)
            if resolved != value:
                return "'" + str(resolved).replace("'", "''") + "'"
            return match.group(0)

        # Only single-quoted strings are value literals in SQL. Double quotes
        # are identifiers ("Region"), which must never be swapped for a cell
        # value. '' is an escaped quote inside a literal.
        sql = re.sub(r"'((?:[^']|'')*)'", replace_literal, sql)

        if sql != original_sql:
            logger.info(f"✏️ SQL rewritten by ColumnValueIndex")
            logger.debug(f"   Before: {original_sql[:120]}")
            logger.debug(f"   After:  {sql[:120]}")

        return sql

    # ================================================================
    # DIAGNOSTICS
    # ================================================================

    def dump_index(self, table_name: str) -> Dict[str, Any]:
        """Return the full index for a table (for debugging/logging)."""
        return self._index.get(table_name, {})

    def get_actual_values(self, table_name: str, column: str) -> List[str]:
        """Return all known actual values for a column."""
        try:
            return list(set(self._index[table_name]["values"][column].values()))
        except KeyError:
            return []