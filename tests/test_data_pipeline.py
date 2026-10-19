# tests/test_data_pipeline.py
"""
CSV ingestion (CSV → SQL table + RAG narrative chunks), forecast data
preparation, upload validation and RAG session restore.
"""

import pandas as pd
import pytest

from agents.forecast_agent import ForecastAgent
from agents.session_rag import SessionAwareRAGAgent
from parsers.hybrid_csv_processor import HybridCSVProcessor
from utils.uploads import is_valid_conversation_id, safe_upload_name


# ── Hybrid CSV processing ────────────────────────────────────────────────────

def test_csv_becomes_session_scoped_table_and_narratives(db, sales_csv):
    result = HybridCSVProcessor(db).process_file(str(sales_csv), "abcd-1234-xyz")

    table = result["sql_table"]
    assert table == "upload_abcd_123_sales_data"          # sanitized id + file stem
    assert db.execute_query(f'SELECT COUNT(*) AS n FROM "{table}"')["n"][0] == 6

    kinds = {d.metadata["chunk_type"] for d in result["rag_documents"]}
    assert kinds == {"row_narrative", "group_summary", "dataset_overview"}
    assert result["metadata"]["columns"] == ["OrderID", "Month", "Region", "Revenue", "Units"]


def test_uppercase_extension_is_read_as_csv(db, sales_csv, tmp_path):
    upper = tmp_path / "SALES.CSV"
    upper.write_bytes(sales_csv.read_bytes())
    assert HybridCSVProcessor(db).process_file(str(upper), "c1")["metadata"]["row_count"] == 6


def test_session_tables_are_isolated(db, sales_csv):
    proc = HybridCSVProcessor(db)
    proc.process_file(str(sales_csv), "sessionA1")
    proc.process_file(str(sales_csv), "sessionB2")
    assert proc.get_uploaded_tables("sessionA1") == ["upload_sessiona_sales_data"]


# ── Forecast agent data preparation ─────────────────────────────────────────

def _forecast_agent(db):
    agent = object.__new__(ForecastAgent)
    agent.db_manager = db
    agent.llm = None
    return agent


def test_forecast_only_reads_its_own_session(db, sales_csv):
    HybridCSVProcessor(db).process_file(str(sales_csv), "sessionB2")
    agent = _forecast_agent(db)
    assert agent._load_session_csv("sessionA1") is None    # no cross-session fallback
    assert len(agent._load_session_csv("sessionB2")) == 6


def test_forecast_column_detection():
    agent = _forecast_agent(None)
    df = pd.DataFrame({
        "LengthOfStay_Days": [3, 5, 7],                    # numeric: must not be the date
        "AdmissionDate": ["2024-01-01", "2024-02-01", "2024-03-01"],
        "TreatmentCost": [100.0, 200.0, 300.0],
    })
    assert agent._detect_date_column(df) == "AdmissionDate"
    assert agent._detect_value_column(df, "forecast treatmentcost") == "TreatmentCost"


@pytest.mark.parametrize("dates, freq", [
    (["2024-01-01", "2024-01-02", "2024-01-03"], "D"),
    (["2024-01-01", "2024-01-08", "2024-01-15"], "W"),
    (["2024-01-01", "2024-02-01", "2024-03-01"], "MS"),
    (["2024-01-01", "2024-04-01", "2024-07-01"], "QS"),
])
def test_frequency_inference(dates, freq):
    assert _forecast_agent(None)._infer_frequency(pd.Series(pd.to_datetime(dates))) == freq


# ── Upload validation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("report.pdf", "report.pdf"),
    ("Data.XLSX", "Data.XLSX"),
    ("../../etc/passwd.txt", "passwd.txt"),
    ("..\\..\\Windows\\evil.csv", "evil.csv"),
    ("script.py", None),
    ("legacy.xls", None),
    ("", None),
    (None, None),
])
def test_safe_upload_name(raw, expected):
    assert safe_upload_name(raw) == expected


@pytest.mark.parametrize("cid, ok", [
    ("3f2b8c1e-1234-4abc-9def-001122334455", True),
    ("../secrets", False),
    ("a/b", False),
    ("", False),
    (None, False),
])
def test_conversation_id_validation(cid, ok):
    assert is_valid_conversation_id(cid) is ok


# ── RAG session restore after restart ────────────────────────────────────────

def test_reloaded_session_recovers_chunks_files_and_tables(db, sales_csv):
    """Regression: a session reloaded from disk had no chunks, so BM25 crashed."""
    from langchain.schema import Document

    proc = HybridCSVProcessor(db)
    table = proc.process_file(str(sales_csv), "conv5678")["sql_table"]

    docs = {
        "1": Document(page_content="Budget approved for AI integration", metadata={"source": "plan.docx", "source_type": "document"}),
        "2": Document(page_content="Order O1 revenue 1000", metadata={"source": "Sales Data.csv", "source_type": "csv", "columns": ["Revenue"]}),
    }
    fake_store = type("Store", (), {"docstore": type("DS", (), {"_dict": docs})()})()

    rag = object.__new__(SessionAwareRAGAgent)
    rag.csv_processor = proc
    session = rag._create_empty_session()
    session["vectorstore"] = fake_store

    rag._restore_session_state(session, "conv5678")

    assert len(session["documents"]) == 2
    assert session["loaded_files"] == ["plan.docx", "Sales Data.csv"]
    assert session["csv_tables"] == [table]
    assert "revenue" in session["source_vocab"]["csv"]
