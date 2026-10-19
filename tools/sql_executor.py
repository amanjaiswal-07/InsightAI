# tools/sql_executor.py
"""
SQL Executor - executes cleaned SQL against the database.
Handles value normalization via ColumnValueIndex.
Logs exactly once per step — no duplicates.
"""

import re
import pandas as pd
from typing import Dict, Any, Optional

from database.db_manager import DatabaseManager
from tools.column_value_index import ColumnValueIndex
from utils.logger import logger


class SQLExecutor:

    def __init__(self, db_manager: DatabaseManager):
        self.db          = db_manager
        self.value_index = ColumnValueIndex()
        logger.info("🔧 SQL Executor initialized")

    def index_table(self, df: pd.DataFrame, table_name: str) -> None:
        """Register a table's schema with the value index (call after CSV import)."""
        self.value_index.build_from_dataframe(df, table_name)

    def clean_sql(self, sql: str) -> str:
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        return sql.strip().rstrip(';')

    def _extract_primary_table(self, sql: str) -> Optional[str]:
        matches = re.findall(
            r'\b(?:FROM|JOIN|UPDATE|INTO)\s+([`"\[]?[\w]+[`"\]]?)',
            sql, flags=re.IGNORECASE,
        )
        return matches[0].strip('`"[]') if matches else None

    def execute_query(self, sql: str) -> Dict[str, Any]:
        """Clean → normalize → execute. Single log per step, no duplicates."""
        try:
            cleaned = self.clean_sql(sql)

            table_name = self._extract_primary_table(cleaned)
            if table_name and self.value_index.has_table(table_name):
                rewritten = self.value_index.rewrite_sql(cleaned, table_name)
                if rewritten != cleaned:
                    logger.info(f"✏️  SQL normalized for '{table_name}'")
                    cleaned = rewritten

            # FIX: removed duplicate 📝/🔍 log lines — caller (sql_analyst) logs SQL
            df = self.db.execute_query(cleaned)

            return {
                "success":      True,
                "sql":          cleaned,
                "row_count":    len(df),
                "columns":      list(df.columns),
                "data":         df.head(100).to_dict('records'),
                "data_preview": df.head(10).to_dict('records'),
                "truncated":    len(df) > 100,
            }

        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            return {"success": False, "error": str(e), "sql": sql}

    def format_results_as_text(self, result: Dict[str, Any]) -> str:
        if not result["success"]:
            return f"Error: {result['error']}"
        df = pd.DataFrame(result["data_preview"])
        if df.empty:
            return "No results found."
        if len(df) == 1 and len(df.columns) == 1:
            return f"Result: {df.iloc[0, 0]}"
        text = f"Found {result['row_count']} result(s):\n\n"
        for _, row in df.iterrows():
            text += "• " + ", ".join(f"{c}: {v}" for c, v in row.items()) + "\n"
        if result.get("truncated"):
            text += f"\n(showing first 10 of {result['row_count']} rows)"
        return text