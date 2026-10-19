# tests/test_routing.py
"""
The orchestrator's rule-based routing cascade, and the A2A dispatch path.
The LLM fallback classifier is stubbed so these tests are deterministic.
"""

import asyncio

import pytest

import agents.orchestrator as orch_mod
from agents.orchestrator import OrchestratorAgent
from protocols import A2AAgent, A2ARegistry


class _StubRAG:
    """Just enough of SessionAwareRAGAgent for _classify_query."""

    def __init__(self, docs=False, csv=False, files=()):
        self.status = {"vectorstore_ready": docs, "has_csv_data": csv, "loaded_files": list(files)}

    def get_session_status(self, _cid):
        return self.status


class _StubLLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def invoke(self, _prompt):
        self.calls += 1
        return type("Msg", (), {"content": self.answer})()


def _orchestrator(rag, llm_answer="SQL"):
    o = object.__new__(OrchestratorAgent)
    A2AAgent.__init__(o, "orchestrator", "test", "test")
    o.rag_agent = rag
    o.csv_router = None
    o.routing_llm = _StubLLM(llm_answer)
    o.routing_prompt = orch_mod.PromptTemplate(
        input_variables=["question", "document_files"], template="{question} {document_files}"
    )
    return o


@pytest.mark.parametrize("question", ["hi there", "Thanks!", "what can you do", "what is machine learning"])
def test_greetings_and_general_knowledge_route_general(question):
    o = _orchestrator(_StubRAG(docs=True, csv=True, files=["a.csv"]))
    assert o._classify_query(question, "c1") == "GENERAL"


def test_no_uploads_routes_general():
    assert _orchestrator(_StubRAG())._classify_query("total revenue by region", "c1") == "GENERAL"


def test_documents_only_routes_document():
    o = _orchestrator(_StubRAG(docs=True, files=["report.pdf"]))
    assert o._classify_query("what budget was approved", "c1") == "DOCUMENT"


def test_forecast_keyword_with_csv_routes_forecast(monkeypatch):
    monkeypatch.setattr(orch_mod, "PROPHET_AVAILABLE", True)
    o = _orchestrator(_StubRAG(docs=True, csv=True, files=["sales.csv"]))
    assert o._classify_query("forecast revenue for the next 6 months", "c1") == "FORECAST"


def test_historical_projection_is_not_a_forecast(monkeypatch):
    monkeypatch.setattr(orch_mod, "PROPHET_AVAILABLE", True)
    o = _orchestrator(_StubRAG(docs=True, csv=True, files=["sales.csv"]), llm_answer="SQL")
    assert o._classify_query("what was last year's projection", "c1") != "FORECAST"


def test_chart_request_on_csv_routes_sql():
    o = _orchestrator(_StubRAG(docs=True, csv=True, files=["sales.csv"]))
    assert o._classify_query("bar chart of revenue by region", "c1") == "SQL"


def test_causal_question_on_csv_routes_hybrid():
    o = _orchestrator(_StubRAG(docs=True, csv=True, files=["sales.csv"]))
    assert o._classify_query("why did revenue drop in the south region", "c1") == "HYBRID"


def test_llm_fallback_is_used_and_parsed():
    o = _orchestrator(_StubRAG(docs=True, csv=True, files=["sales.csv"]), llm_answer=" document ")
    assert o._classify_query("tell me about the regional managers", "c1") == "DOCUMENT"
    assert o.routing_llm.calls == 1


def test_forecast_tab_column_reaches_forecast_agent():
    """Regression: value_col from the Forecast tab used to be dropped before the agent."""
    captured = {}

    class _StubForecast:
        async def forecast(self, question, conversation_id, periods, value_col=None):
            captured.update(question=question, cid=conversation_id, periods=periods, value_col=value_col)
            return {"success": True, "explanation": "ok"}

    o = _orchestrator(_StubRAG(csv=True, files=["sales.csv"]))
    o.forecast_agent = _StubForecast()
    o.registry = A2ARegistry()
    o.registry.register_agent(o._wrap_forecast_agent())
    o._classify_query = lambda q, c: "FORECAST"

    result = asyncio.run(o.route_query("forecast Units for next 3 periods", "conv1", value_col="Units"))

    assert result["routing"] == "forecast"
    assert captured == {"question": "forecast Units for next 3 periods", "cid": "conv1",
                        "periods": 3, "value_col": "Units"}
