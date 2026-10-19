# tests/test_sql_analyst.py
"""
SQLAnalystAgent.analyze() end to end on a real SQLite table, with the LLM
chains replaced by fakes so the test is deterministic and offline.
"""

from agents.sql_analyst import SQLAnalystAgent
from parsers.hybrid_csv_processor import HybridCSVProcessor
from tools.language_detector import LanguageDetector
from tools.sql_executor import SQLExecutor
from utils.plot_builder import PlotBuilder
from tests.conftest import FakeChain


def _agent(db, sql_outputs, explanation="EXPLAINED", fix_outputs=("SELECT 1",)):
    agent = object.__new__(SQLAnalystAgent)
    agent.db = db
    agent.language_detector = LanguageDetector()
    agent.sql_executor = SQLExecutor(db)
    agent.plot_builder = PlotBuilder()
    agent._schema_cache = {}
    agent.sql_chain = FakeChain(*sql_outputs)
    agent.sql_fix_chain = FakeChain(*fix_outputs)
    agent.explanation_chain = FakeChain(explanation)
    return agent


def _load(db, sales_csv, cid="conv1234"):
    return HybridCSVProcessor(db).process_file(str(sales_csv), cid)["sql_table"]


def test_single_value_count_uses_fast_path(db, sales_csv):
    table = _load(db, sales_csv)
    agent = _agent(db, [f'SELECT COUNT(*) FROM "{table}"'])

    result = agent.analyze("How many orders are there?", [table], "conv1234")

    assert result["success"]
    assert result["explanation"] == "Total: 6 orders"
    assert agent.explanation_chain.calls == []   # no LLM needed for one number


def test_grouped_count_is_explained_not_summed(db, sales_csv):
    """Regression: 'how many ... per X' used to collapse every row into one total."""
    table = _load(db, sales_csv)
    agent = _agent(db, [f'SELECT Region, COUNT(*) AS n FROM "{table}" GROUP BY Region'])

    result = agent.analyze("How many orders per region?", [table], "conv1234")

    assert result["explanation"] == "EXPLAINED"
    assert len(agent.explanation_chain.calls) == 1
    assert {r["Region"]: r["n"] for r in result["data"]} == {"North": 3, "South": 3}


def test_failed_sql_is_retried_once_with_the_error(db, sales_csv):
    table = _load(db, sales_csv)
    agent = _agent(
        db,
        [f'SELECT Revenu FROM "{table}"'],                      # typo -> no such column
        fix_outputs=[f'SELECT SUM(Revenue) AS total FROM "{table}"'],
    )

    result = agent.analyze("What is the total revenue overall?", [table], "conv1234")

    assert result["success"]
    assert result["sql_query"].startswith("SELECT SUM(Revenue)")
    fix_call = agent.sql_fix_chain.calls[0]
    assert "no such column" in fix_call["error"].lower()
    assert "Revenu" in fix_call["failed_sql"]


def test_error_is_returned_when_retry_also_fails(db, sales_csv):
    table = _load(db, sales_csv)
    agent = _agent(db, ['SELECT nope FROM "missing"'], fix_outputs=['SELECT nope FROM "missing"'])

    result = agent.analyze("Show the revenue values please", [table], "conv1234")

    assert result["success"] is False
    assert "no such table" in result["error"].lower()


def test_no_tables_returns_upload_hint(db):
    agent = _agent(db, ["SELECT 1"])
    result = agent.analyze("total revenue", [], "conv1234")
    assert "upload a CSV" in result["explanation"]


def test_chart_request_produces_plot_spec(db, sales_csv):
    table = _load(db, sales_csv)
    agent = _agent(db, [f'SELECT Region, SUM(Revenue) AS revenue FROM "{table}" GROUP BY Region'])

    result = agent.analyze("Show a bar chart of revenue by region", [table], "conv1234")

    assert result["plot"]["type"] == "bar"
