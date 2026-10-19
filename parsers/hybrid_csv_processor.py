# parsers/hybrid_csv_processor.py
"""
Hybrid CSV/Excel ingestion.

Each uploaded table is processed two ways:
  1. Imported into SQLite as upload_<session>_<file> for exact SQL answers.
  2. Turned into natural-language chunks for RAG: one narrative per row,
     one summary per categorical group, and one dataset overview.
Narrative chunks rerank far better than raw CSV text, which lets the RAG
agent answer descriptive questions about tabular data.
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain.schema import Document

from database.db_manager import DatabaseManager
from parsers.document_parser import DocumentParser
from utils.logger import logger
from config import CSV_TABLE_PREFIX


# ── Column role detection constants ──────────────────────────────────
# Max unique values for a column to be treated as categorical (groupable)
MAX_CATEGORICAL_UNIQUE = 20
# Min rows in a group to generate a group summary chunk
MIN_GROUP_SIZE = 2


def _read_table(file_path: str) -> pd.DataFrame:
    """Read a .csv or .xlsx file into a DataFrame (extension match is case-insensitive)."""
    if Path(file_path).suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


class HybridCSVProcessor:

    def __init__(self, db_manager: DatabaseManager, sql_executor=None):
        self.db = db_manager
        self.doc_parser = DocumentParser()
        self.sql_executor = sql_executor
        logger.info("📊 Hybrid CSV Processor initialized")

    def is_structured_file(self, filename: str) -> bool:
        return filename.lower().endswith(('.csv', '.xlsx'))

    def process_file(self, file_path: str, conversation_id: str) -> Dict[str, Any]:
        filename = Path(file_path).name

        if not self.is_structured_file(filename):
            raise ValueError(f"File {filename} is not a structured data file")

        logger.info(f"📊 Processing structured file: {filename}")
        logger.info(f"   Mode: HYBRID (SQL + RAG)")

        sql_table = self._import_to_sql(file_path, conversation_id)
        rag_docs  = self._parse_to_rag(file_path)
        metadata  = self._extract_metadata(file_path)

        result = {
            "filename":     filename,
            "sql_table":    sql_table,
            "rag_documents": rag_docs,
            "metadata":     metadata,
            "capabilities": {
                "sql_calculations": True,
                "semantic_search":  True,
                "hybrid_analysis":  True,
            },
        }

        logger.info(f"""
                ✅ Hybrid Processing Complete: {filename}
                📊 SQL Table: {sql_table} ({metadata['row_count']} rows)
                📄 RAG Chunks: {len(rag_docs)} documents
                ✨ Capabilities: SQL calculations + Semantic understanding
                    """)

        return result

    # ================================================================
    # SQL IMPORT (unchanged)
    # ================================================================

    def _import_to_sql(self, file_path: str, conversation_id: str) -> str:
        try:
            df = _read_table(file_path)
        except Exception as e:
            logger.error(f"❌ Failed to read file: {e}")
            raise

        table_name = self._generate_table_name(file_path, conversation_id)

        try:
            conn = self.db.get_connection()
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            logger.info(f"   ✅ SQL Import: {table_name} ({len(df)} rows, {len(df.columns)} columns)")

            if self.sql_executor is not None:
                self.sql_executor.index_table(df, table_name)
                logger.info(f"   📇 Value index built for '{table_name}'")

            return table_name

        except Exception as e:
            logger.error(f"❌ SQL import failed: {e}")
            raise

    # ================================================================
    # NARRATIVE RAG PARSING (core fix)
    # ================================================================

    def _parse_to_rag(self, file_path: str) -> List[Document]:
        """
        Generate three types of semantic chunks from a CSV/Excel file.

        Type 1 — Row narratives: one prose sentence per row.
            "Patient P005 is a 72-year-old Male in Oncology diagnosed with
             Lung Cancer. Length of stay: 14 days. Treatment cost: $67,000.
             Severity score: 9. Readmitted: Yes. Status: Under Treatment."

        Type 2 — Group summaries: one chunk per categorical group value.
            "Oncology Department — 5 patients:
             - P005 (72M): Lung Cancer, 14 days, $67,000, severity 9
             ..."

        Type 3 — Dataset overview: one global summary chunk.
            "This dataset contains 40 patients across 7 departments.
             Age ranges 27–83. Average cost: $28,450. ..."

        These narrative chunks score -1 to -4 on the cross-encoder
        (vs -7 to -11 for raw tabular chunks), enabling RAG to answer
        semantic questions about CSV data correctly.
        """
        try:
            df = _read_table(file_path)
        except Exception as e:
            logger.error(f"❌ RAG parsing failed: {e}")
            raise

        filename    = Path(file_path).name
        col_roles   = self._detect_column_roles(df)
        all_docs    = []

        # ── Type 1: Row narratives ────────────────────────────────────
        row_docs = self._build_row_narratives(df, col_roles, filename)
        all_docs.extend(row_docs)
        logger.info(f"   📝 Row narratives: {len(row_docs)} chunks")

        # ── Type 2: Group summaries ───────────────────────────────────
        group_docs = self._build_group_summaries(df, col_roles, filename)
        all_docs.extend(group_docs)
        logger.info(f"   📊 Group summaries: {len(group_docs)} chunks")

        # ── Type 3: Dataset overview ──────────────────────────────────
        overview_doc = self._build_dataset_overview(df, col_roles, filename)
        all_docs.append(overview_doc)
        logger.info(f"   🌐 Dataset overview: 1 chunk")

        logger.info(f"   ✅ RAG Parsing: {len(all_docs)} document chunks")
        return all_docs

    # ================================================================
    # COLUMN ROLE DETECTION
    # ================================================================

    def _detect_column_roles(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Automatically classify each column into a role:
          'id'          → identifier (PatientID, EmployeeID, OrderID)
          'numeric'     → continuous number (Age, Cost, Score, Salary)
          'categorical' → low-cardinality text (Department, Gender, Status)
          'date'        → date/datetime column
          'text'        → high-cardinality text (Diagnosis, Notes)

        No hardcoding — uses dtype + cardinality + name heuristics.
        Works on any CSV schema.
        """
        roles = {}
        n_rows = len(df)

        for col in df.columns:
            col_lower = col.lower()

            # Date detection — check dtype first
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                roles[col] = 'date'
                continue

            # Try parsing as date by name heuristic
            if any(kw in col_lower for kw in ['date', 'time', 'dt', 'day', 'month', 'year']):
                roles[col] = 'date'
                continue

            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                roles[col] = 'numeric'
                continue

            # For object/string columns — use cardinality
            n_unique = df[col].nunique()

            # ID columns: high cardinality, name ends in ID/Code/No/Num
            if any(col_lower.endswith(s) for s in ['id', 'code', 'no', 'num', 'number', 'key']):
                roles[col] = 'id'
                continue

            # Categorical: low cardinality (≤20 unique or ≤10% of rows)
            if n_unique <= MAX_CATEGORICAL_UNIQUE or (n_rows > 0 and n_unique / n_rows <= 0.1):
                roles[col] = 'categorical'
                continue

            # Everything else is free text
            roles[col] = 'text'

        return roles

    # ================================================================
    # TYPE 1: ROW NARRATIVES
    # ================================================================

    def _build_row_narratives(
        self, df: pd.DataFrame, col_roles: Dict[str, str], filename: str
    ) -> List[Document]:
        """
        One narrative sentence per row. Automatically adapts to any schema.

        Strategy:
          - Find the subject (ID column or first column)
          - Build a natural sentence using column name + value pairs
          - Format numerics nicely (currency, percentage detection)
          - Skip NaN values silently
        """
        docs      = []
        id_cols   = [c for c, r in col_roles.items() if r == 'id']
        subject_col = id_cols[0] if id_cols else df.columns[0]

        # Infer a "record type" from the subject column name
        # "PatientID" → "Patient", "EmployeeID" → "Employee", etc.
        record_type = re.sub(
            r'[Ii][Dd]$|[Cc]ode$|[Nn]o$|[Nn]um(ber)?$', '',
            subject_col
        ).strip() or "Record"

        for _, row in df.iterrows():
            subject_val = row.get(subject_col, "Unknown")
            parts       = []

            for col in df.columns:
                if col == subject_col:
                    continue
                val = row.get(col)
                if pd.isna(val):
                    continue

                role      = col_roles.get(col, 'text')
                label     = self._humanize_col_name(col)
                formatted = self._format_value(col, val, role)
                parts.append(f"{label}: {formatted}")

            # Build narrative
            narrative = f"{record_type} {subject_val} — " + ". ".join(parts) + "."

            docs.append(Document(
                page_content=narrative,
                metadata={
                    "chunk_type":  "row_narrative",
                    "source":      filename,
                    "source_type": "csv",
                    "record_id":   str(subject_val),
                }
            ))

        return docs

    # ================================================================
    # TYPE 2: GROUP SUMMARIES
    # ================================================================

    def _build_group_summaries(
        self, df: pd.DataFrame, col_roles: Dict[str, str], filename: str
    ) -> List[Document]:
        """
        One summary chunk per group value for each categorical column.

        E.g. for Department column:
          "Oncology Department — 5 records:
           - P005 (72, Male): Lung Cancer, 14 days, $67,000, severity 9
           - P008 (44, Female): Breast Cancer, 14 days, $58,000, severity 8
           Average LengthOfStay_Days: 14.0. Average TreatmentCost_USD: 63800."
        """
        docs        = []
        id_cols     = [c for c, r in col_roles.items() if r == 'id']
        cat_cols    = [c for c, r in col_roles.items() if r == 'categorical']
        num_cols    = [c for c, r in col_roles.items() if r == 'numeric']
        subject_col = id_cols[0] if id_cols else df.columns[0]

        record_type = re.sub(
            r'[Ii][Dd]$|[Cc]ode$|[Nn]o$|[Nn]um(ber)?$', '',
            subject_col
        ).strip() or "Record"

        for group_col in cat_cols:
            # Skip columns with too many unique values (already filtered but double-check)
            if df[group_col].nunique() > MAX_CATEGORICAL_UNIQUE:
                continue

            for group_val, group_df in df.groupby(group_col):
                if len(group_df) < MIN_GROUP_SIZE:
                    continue

                group_label = self._humanize_col_name(group_col)
                lines       = [
                    f"{group_val} {group_label} — {len(group_df)} {record_type.lower()}s:"
                ]

                # List individual records (up to 15 to keep chunk size reasonable)
                for _, row in group_df.head(15).iterrows():
                    subject_val = row.get(subject_col, "?")
                    # Pick 3-4 most informative non-group, non-id columns
                    info_parts  = []
                    for col in df.columns:
                        if col in (subject_col, group_col):
                            continue
                        val = row.get(col)
                        if pd.isna(val):
                            continue
                        role = col_roles.get(col, 'text')
                        info_parts.append(self._format_value(col, val, role))
                        if len(info_parts) >= 5:   # cap at 5 details per row
                            break
                    detail = ", ".join(info_parts)
                    lines.append(f"  - {subject_val}: {detail}")

                if len(group_df) > 15:
                    lines.append(f"  ... and {len(group_df) - 15} more.")

                # Aggregate stats for numeric columns
                agg_parts = []
                for num_col in num_cols[:6]:   # cap at 6 numeric stats
                    try:
                        mean_val = group_df[num_col].mean()
                        if not pd.isna(mean_val):
                            label = self._humanize_col_name(num_col)
                            agg_parts.append(
                                f"Average {label}: {self._format_value(num_col, mean_val, 'numeric')}"
                            )
                    except Exception:
                        continue

                if agg_parts:
                    lines.append("Averages — " + ". ".join(agg_parts) + ".")

                content = "\n".join(lines)
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "chunk_type":  "group_summary",
                        "source":      filename,
                        "source_type": "csv",
                        "group_col":   group_col,
                        "group_val":   str(group_val),
                    }
                ))

        return docs

    # ================================================================
    # TYPE 3: DATASET OVERVIEW
    # ================================================================

    def _build_dataset_overview(
        self, df: pd.DataFrame, col_roles: Dict[str, str], filename: str
    ) -> Document:
        """
        One global overview chunk covering the whole dataset.
        Answers "summarize the data", "what are the key patterns", etc.
        """
        n_rows    = len(df)
        n_cols    = len(df.columns)
        cat_cols  = [c for c, r in col_roles.items() if r == 'categorical']
        num_cols  = [c for c, r in col_roles.items() if r == 'numeric']
        id_cols   = [c for c, r in col_roles.items() if r == 'id']

        subject_col = id_cols[0] if id_cols else df.columns[0]
        record_type = re.sub(
            r'[Ii][Dd]$|[Cc]ode$|[Nn]o$|[Nn]um(ber)?$', '',
            subject_col
        ).strip() or "Record"

        lines = [
            f"Dataset overview — {filename}",
            f"Contains {n_rows} {record_type.lower()}s with {n_cols} attributes.",
        ]

        # Categorical breakdowns
        for cat_col in cat_cols[:4]:   # cap at 4 categorical breakdowns
            counts   = df[cat_col].value_counts()
            label    = self._humanize_col_name(cat_col)
            breakdown = ", ".join(
                f"{val} ({cnt})" for val, cnt in counts.head(8).items()
            )
            lines.append(f"{label} breakdown: {breakdown}.")

        # Numeric ranges and averages
        for num_col in num_cols[:6]:   # cap at 6 numeric summaries
            try:
                col_data = df[num_col].dropna()
                if col_data.empty:
                    continue
                label    = self._humanize_col_name(num_col)
                min_val  = self._format_value(num_col, col_data.min(), 'numeric')
                max_val  = self._format_value(num_col, col_data.max(), 'numeric')
                avg_val  = self._format_value(num_col, col_data.mean(), 'numeric')
                lines.append(
                    f"{label}: ranges {min_val}–{max_val}, average {avg_val}."
                )
            except Exception:
                continue

        content = "\n".join(lines)
        return Document(
            page_content=content,
            metadata={
                "chunk_type":  "dataset_overview",
                "source":      filename,
                "source_type": "csv",
            }
        )

    # ================================================================
    # FORMATTING HELPERS
    # ================================================================

    def _humanize_col_name(self, col: str) -> str:
        """
        Convert column name to readable label.
        'TreatmentCost_USD' → 'Treatment Cost USD'
        'LengthOfStay_Days' → 'Length Of Stay Days'
        'PerformanceScore'  → 'Performance Score'
        """
        # Split on underscores and camelCase
        s = re.sub(r'([a-z])([A-Z])', r'\1 \2', col)  # camelCase split
        s = s.replace('_', ' ')
        return s.strip()

    def _format_value(self, col: str, val: Any, role: str) -> str:
        """
        Format a value based on column name hints and role.
        Detects currency, percentage, and count columns automatically.
        """
        if pd.isna(val):
            return "N/A"

        col_lower = col.lower()

        if role == 'numeric':
            try:
                num = float(val)
                # Currency detection
                if any(kw in col_lower for kw in ['cost', 'salary', 'revenue', 'price',
                                                    'income', 'spend', 'budget', 'usd',
                                                    'amount', 'fee', 'pay', 'lpa']):
                    if 'lpa' in col_lower:
                        return f"{num:,.1f} LPA"
                    return f"${num:,.0f}" if num >= 100 else f"${num:.2f}"
                # Percentage detection
                if any(kw in col_lower for kw in ['pct', 'percent', 'rate', 'ratio']):
                    return f"{num:.1f}%"
                # Integer-like floats
                if num == int(num):
                    return str(int(num))
                return f"{num:.2f}"
            except (ValueError, TypeError):
                return str(val)

        return str(val)

    # ================================================================
    # METADATA & UTILITIES (unchanged)
    # ================================================================

    @staticmethod
    def session_table_prefix(conversation_id: str) -> str:
        """
        The exact table-name prefix used for every CSV table belonging to a
        session. Shared by table creation (_generate_table_name) and lookup
        (get_uploaded_tables) so the two can never disagree.

        conversation_id is usually a UUID (hyphens only after position 8, so
        [:8] happens to dodge them) but is not guaranteed to be one —
        sanitizing it the same way as an uploaded filename ensures no
        non-alphanumeric character (hyphens included) can ever produce an
        unquoted SQL identifier SQLite fails to parse
        (e.g. "near '-': syntax error").
        """
        id_sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', conversation_id[:8]).lower()
        return f"{CSV_TABLE_PREFIX}{id_sanitized}_"

    def _generate_table_name(self, file_path: str, conversation_id: str) -> str:
        base_name  = Path(file_path).stem
        sanitized  = re.sub(r'[^a-zA-Z0-9_]', '_', base_name).lower()
        table_name = f"{self.session_table_prefix(conversation_id)}{sanitized}"
        if len(table_name) > 63:
            table_name = table_name[:63]
        return table_name

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        try:
            df = _read_table(file_path)
            return {
                "row_count":    len(df),
                "column_count": len(df.columns),
                "columns":      list(df.columns),
                "dtypes":       {col: str(dtype) for col, dtype in df.dtypes.items()},
                "file_size_mb": Path(file_path).stat().st_size / (1024 * 1024),
            }
        except Exception as e:
            logger.warning(f"⚠️ Metadata extraction failed: {e}")
            return {}

    def get_uploaded_tables(self, conversation_id: str) -> List[str]:
        all_tables    = self.db.list_tables()
        prefix        = self.session_table_prefix(conversation_id)
        session_tables = [t for t in all_tables if t.startswith(prefix)]
        logger.info(f"📊 Found {len(session_tables)} uploaded tables for session")
        return session_tables

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        try:
            sample_df = self.db.get_table_sample(table_name, limit=5)
            count_df  = self.db.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
            row_count = count_df.iloc[0]['count']
            return {
                "table_name":  table_name,
                "row_count":   row_count,
                "columns":     list(sample_df.columns),
                "sample_data": sample_df.to_dict('records'),
            }
        except Exception as e:
            logger.error(f"❌ Failed to get table info: {e}")
            return {}


# ── Module-level convenience function (unchanged signature) ──────────
def process_csv_hybrid(
    file_path: str,
    conversation_id: str,
    db_manager: DatabaseManager,
) -> Dict[str, Any]:
    processor = HybridCSVProcessor(db_manager)
    return processor.process_file(file_path, conversation_id)