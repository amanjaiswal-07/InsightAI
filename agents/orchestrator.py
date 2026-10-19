# agents/orchestrator.py
"""
Smart Query Orchestrator

Routes every query through a 6-priority classification cascade before
dispatching to the correct agent via the A2A registry. No agent is ever
called directly — all calls go through registry.route_message().

Routing priorities:
  -1   General / conversational         → GENERAL   (LLM direct)
  0.5  Forecast keywords + CSV present  → FORECAST  (A2A → forecast_agent)
  0    No uploads                        → GENERAL   (LLM direct)
  1    Text/PDF/DOCX only                → DOCUMENT  (A2A → rag_agent)
  1.5  Chart/visualisation request       → SQL       (A2A → sql_agent)
  2    Both file types + vocab match     → SQL / DOCUMENT / HYBRID
  3    Explicit comparison keywords      → HYBRID
  4    CSVQueryRouter                    → SQL / DOCUMENT / HYBRID
  5    LLM fallback classifier           → SQL / DOCUMENT / HYBRID
"""

from typing import Dict, Any
import asyncio
import re
from collections import Counter

from langchain.prompts import PromptTemplate

from protocols import A2AAgent, AgentCapability, MessageType, A2AMessage, A2ARegistry
from agents.sql_analyst import SQLAnalystAgent
from agents.session_rag import SessionAwareRAGAgent
from agents.forecast_agent import ForecastAgent, PROPHET_AVAILABLE
from database.db_manager import DatabaseManager
from config import settings, ENABLE_HYBRID_CSV
from utils.llm import make_llm
from utils.logger import logger, log_section

if ENABLE_HYBRID_CSV:
    try:
        from tools.csv_query_router import CSVQueryRouter
        CSV_ROUTER_AVAILABLE = True
    except ImportError:
        logger.warning("⚠️  CSV Router not available — hybrid CSV disabled")
        CSV_ROUTER_AVAILABLE = False
else:
    CSV_ROUTER_AVAILABLE = False


class OrchestratorAgent(A2AAgent):

    def __init__(self, db_manager: DatabaseManager):
        super().__init__(
            agent_id="orchestrator",
            name="Smart Query Orchestrator",
            description="Routes queries to GENERAL, SQL, RAG, HYBRID, or FORECAST via A2A",
        )

        # A2A registry — all inter-agent calls go through here
        self.registry  = A2ARegistry()

        # Core agents
        self.sql_agent = SQLAnalystAgent(db_manager)
        self.rag_agent = SessionAwareRAGAgent(
            db_manager=db_manager,
            sql_executor=self.sql_agent.sql_executor,
        )
        self.forecast_agent = ForecastAgent(db_manager)

        # CSV hybrid router (optional feature flag)
        self.csv_router = CSVQueryRouter() if CSV_ROUTER_AVAILABLE else None
        if self.csv_router:
            logger.info("✅  CSV Hybrid Router enabled")

        # Wrap each agent in an A2A-compatible peer and register in registry
        self.registry.register_agent(self._wrap_sql_agent())
        self.registry.register_agent(self._wrap_rag_agent())
        self.registry.register_agent(self._wrap_forecast_agent())
        logger.info(
            f"✅  A2A registry: {list(self.registry.agents.keys())} registered"
        )

        # General chat LLM (warmer temperature for conversational responses)
        self.llm = make_llm(temperature=0.7)

        # Deterministic routing classifier (temperature=0 for consistent routing)
        self.routing_llm = make_llm(temperature=0)

        # Synthesis LLM for HYBRID responses that need both sources merged
        self.synthesis_llm = make_llm(temperature=0.1)

        self.routing_prompt = PromptTemplate(
            input_variables=["question", "document_files"],
            template="""You are a query router for an analytics system.
The user has uploaded CSV/structured files: {document_files}

ROUTING RULES:
1. SQL      → precise calculations, counts, aggregations, rankings on uploaded CSV data
2. DOCUMENT → summarization, explanation, context from uploaded CSV (semantic questions)
3. HYBRID   → needs BOTH a calculation AND an explanation/reason

Reply with ONLY one word: SQL, DOCUMENT, or HYBRID

Question: {question}""",
        )

        self._register_capabilities()
        self._register_handlers()
        logger.info("🎯  Smart Hybrid Orchestrator ready")

    # ── A2A agent wrappers ────────────────────────────────────────────────────

    def _wrap_sql_agent(self) -> A2AAgent:
        """Wrap SQLAnalystAgent as an A2A peer with execute_sql capability."""
        agent = A2AAgent(
            agent_id="sql_agent",
            name="SQL Analyst",
            description="Natural language to SQL on uploaded CSVs",
        )
        agent.register_capability(AgentCapability(
            name="execute_sql",
            description="Execute SQL queries from natural language",
            input_schema={"type": "object", "properties": {"question": {"type": "string"}}},
            output_schema={"type": "object"},
        ))

        async def handle_query(message: A2AMessage) -> Dict[str, Any]:
            # analyze() makes blocking LLM + DB calls; run it off the event loop
            return await asyncio.to_thread(
                self.sql_agent.analyze,
                question=message.payload.get("question"),
                allowed_tables=message.payload.get("allowed_tables"),
                conversation_id=message.payload.get("conversation_id"),
            )

        agent.register_handler(MessageType.QUERY, handle_query)
        return agent

    def _wrap_rag_agent(self) -> A2AAgent:
        """Wrap SessionAwareRAGAgent as an A2A peer with query_documents capability."""
        agent = A2AAgent(
            agent_id="rag_agent",
            name="Document RAG Agent",
            description="Hybrid BM25+FAISS retrieval over session documents",
        )
        agent.register_capability(AgentCapability(
            name="query_documents",
            description="Semantic + keyword search over uploaded documents",
            input_schema={"type": "object", "properties": {
                "question":        {"type": "string"},
                "conversation_id": {"type": "string"},
            }},
            output_schema={"type": "object"},
        ))

        async def handle_query(message: A2AMessage) -> Dict[str, Any]:
            # Retrieval, reranking and generation are blocking; keep the loop free
            return await asyncio.to_thread(
                self.rag_agent.query,
                message.payload.get("conversation_id"),
                message.payload.get("question"),
                message.payload.get("language"),
            )

        agent.register_handler(MessageType.QUERY, handle_query)
        return agent

    def _wrap_forecast_agent(self) -> A2AAgent:
        """
        Wrap ForecastAgent as an A2A peer with forecast + anomaly_detection capabilities.
        Parses period count from the question before delegating to forecast_agent.forecast().
        """
        agent = A2AAgent(
            agent_id="forecast_agent",
            name="Prophet Forecast Agent",
            description="Time series forecasting and anomaly detection from uploaded CSV",
        )
        agent.register_capability(AgentCapability(
            name="forecast",
            description="Forecast numeric time series using Prophet with 95% confidence intervals",
            input_schema={"type": "object", "properties": {
                "question":        {"type": "string"},
                "conversation_id": {"type": "string"},
                "periods":         {"type": "integer"},
            }},
            output_schema={"type": "object"},
        ))
        agent.register_capability(AgentCapability(
            name="anomaly_detection",
            description="Detect anomalies in time series data using Prophet confidence bounds",
            input_schema={"type": "object", "properties": {
                "question":        {"type": "string"},
                "conversation_id": {"type": "string"},
            }},
            output_schema={"type": "object"},
        ))

        async def handle_forecast(message: A2AMessage) -> Dict[str, Any]:
            question = message.payload.get("question", "")
            periods  = message.payload.get("periods", 6)

            # Parse period count from question text (e.g. "next 12 months" → 12)
            m = re.search(r'(\d+)\s*(month|week|quarter|year|period)', question.lower())
            if m:
                periods = min(max(int(m.group(1)), 1), 36)  # clamp to [1, 36]

            return await self.forecast_agent.forecast(
                question=question,
                conversation_id=message.payload.get("conversation_id", ""),
                periods=periods,
                value_col=message.payload.get("value_col"),
            )

        agent.register_handler(MessageType.QUERY, handle_forecast)
        return agent

    # ── Capability / handler registration ────────────────────────────────────

    def _register_capabilities(self):
        self.register_capability(AgentCapability(
            name="route_query",
            description="Route queries to GENERAL, SQL, RAG, HYBRID, or FORECAST",
            input_schema={"type": "object", "properties": {
                "question":        {"type": "string"},
                "conversation_id": {"type": "string"},
            }},
            output_schema={"type": "object"},
        ))

    def _register_handlers(self):
        self.register_handler(MessageType.QUERY, self._handle_query_message)

    async def _handle_query_message(self, message: A2AMessage) -> Dict[str, Any]:
        question        = message.payload.get("question")
        conversation_id = message.payload.get("conversation_id")
        if not question:
            return {"error": "No question provided"}
        return await self.route_query(question, conversation_id)

    # ── Vocab-based source detection ──────────────────────────────────────────

    def _detect_source_hint(self, question: str, conversation_id: str) -> str:
        """
        Score the question against per-source unique vocabularies built at ingest.

        Returns one of:
          "csv"      — question terms match CSV-unique vocab
          "document" — question terms match document-unique vocab
          "tie"      — ambiguous; caller uses rerank dominance as tiebreaker
          None       — vocab not built yet

        Intent/aggregation words are stripped before scoring to prevent
        UI-side keywords ("chart", "count") from polluting source signals.
        """
        session = self.rag_agent.get_or_create_session(conversation_id)
        vocab   = session.get("source_vocab")
        if not vocab:
            return None

        INTENT_WORDS = {
            'chart', 'plot', 'graph', 'pie', 'bar', 'line', 'show', 'give',
            'display', 'visualize', 'grouped', 'group', 'number', 'count',
            'total', 'average', 'sum', 'list', 'tell', 'what', 'how', 'many',
            'much', 'all', 'get',
        }

        q_words  = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower())) - INTENT_WORDS
        csv_hits = len(q_words & vocab["csv"])
        doc_hits = len(q_words & vocab["document"])

        logger.info(f"🎯  Vocab hits — CSV: {csv_hits}, DOC: {doc_hits}")

        if csv_hits > doc_hits:   return "csv"
        if doc_hits > csv_hits:   return "document"
        return "tie"

    def _rerank_dominant_source(self, conversation_id: str, question: str) -> str:
        """
        Run a quick retrieval and check which source dominates the top-3 results.
        Used as tiebreaker when vocab scoring is ambiguous.

        Returns "csv", "document", or "mixed".
        Requires 2/3 dominance to be confident; otherwise returns "mixed".
        """
        try:
            session = self.rag_agent.get_or_create_session(conversation_id)
            if not session["vectorstore"]:
                return "mixed"

            retriever = session["vectorstore"].as_retriever(search_kwargs={"k": 6})
            docs      = retriever.invoke(question)
            sources   = [d.metadata.get("source_type", "unknown") for d in docs[:3]]
            counts    = Counter(sources)
            top       = counts.most_common(1)[0]

            logger.info(f"🔍  Rerank dominance: {dict(counts)}")
            return top[0] if top[1] >= 2 else "mixed"

        except Exception as e:
            logger.warning(f"⚠️  Rerank dominance check failed: {e}")
            return "mixed"

    # ── Core classification ───────────────────────────────────────────────────

    def _classify_query(self, question: str, conversation_id: str = None) -> str:
        session_has_docs = False
        session_has_csv  = False
        document_files   = []

        if conversation_id:
            status           = self.rag_agent.get_session_status(conversation_id)
            session_has_docs = status["vectorstore_ready"]
            session_has_csv  = status.get("has_csv_data", False)
            document_files   = status.get("loaded_files", [])

            logger.info(
                f"📊  Session — docs: {session_has_docs}, "
                f"csv: {session_has_csv}, files: {document_files}"
            )

        question_lower = question.lower()

        # ── Priority -1: General / conversational queries ─────────────────────
        # FIX 1: Runs BEFORE forecast and CSV router — prevents greetings routing
        # to document/SQL even when uploads exist.
        # FIX 2: Curly-quote bug fixed — two clean ASCII patterns replace the
        # broken r'^(how are you|how\u2019s it going)\b' single pattern.
        _GEN_EXACT = [
            r'^(hi|hello|hey|greetings|howdy)',
            r'^(how are you)\b',
            r'^hello.*how are you',
            r'^(what can you do|what do you do|what are your capabilities)',
            r'^(who are you|what are you|tell me about yourself)',
            r'^(thanks?|thank you|thx|cheers|bye|goodbye)',
            r'^(can you help|how does this work|how do i use this)',
        ]
        _GEN_TOPICS = [
            'machine learning', 'artificial intelligence', 'deep learning',
            'neural network', 'what is nlp', 'what is ai', 'what is gpt',
            'what is llm', 'what is rag', 'what is python', 'what is sql',
            'large language model',
        ]
        _GEN_STARTERS = [
            'explain what ', 'explain how ', 'what is a ', 'what is an ',
            'how does ', 'describe what ',
        ]
        _DATA_SIGS = [
            'total', 'average', 'count', 'sum', 'top', 'max', 'min',
            'revenue', 'sales', 'chart', 'plot', 'forecast', 'trend',
            'data', 'dataset', 'csv', 'column', 'row', 'table', 'report',
        ]
        q_lc = question_lower.strip()
        _gen = False
        if any(re.search(p, q_lc) for p in _GEN_EXACT):
            _gen = True
        elif any(t in q_lc for t in _GEN_TOPICS):
            _gen = True
        elif any(q_lc.startswith(s) for s in _GEN_STARTERS):
            if not any(s in q_lc for s in _DATA_SIGS):
                _gen = True
        if _gen:
            logger.info("💬  Route: GENERAL (conversational/knowledge query)")
            return "GENERAL"

        # ── Priority 0.5: Forecast keywords + CSV present ─────────────────────
        # Guard: historical queries ("last year's projections") are NOT forecasts
        _HIST_SIGS = [
            'last year', 'last quarter', 'previous', 'past year',
            'unaudited', 'internal projection', 'historical', 'prior year', ' ago',
        ]
        _is_hist = any(s in question_lower for s in _HIST_SIGS)
        FORECAST_KEYWORDS = [
            'forecast', 'predict', 'next month', 'next quarter', 'next year',
            'next 3', 'next 6', 'next 12',
            'trend ahead', 'upcoming periods',
            'anomaly', 'anomalies', 'outlier', 'detect anomal', 'time series',
        ]
        # "projection" only triggers forecast if it is about the future
        if 'projection' in question_lower and not _is_hist:
            FORECAST_KEYWORDS.append('projection')
        if session_has_csv and not _is_hist and any(kw in question_lower for kw in FORECAST_KEYWORDS):
            if PROPHET_AVAILABLE:
                logger.info("📈  FORECAST: keyword match + CSV present")
                return "FORECAST"
            logger.info("⚠️   Forecast keyword matched but Prophet not installed")

        # ── Priority 0: No uploads ────────────────────────────────────────────
        if not session_has_docs and not session_has_csv:
            logger.info("💬  No uploads → GENERAL")
            return "GENERAL"

        # ── Priority 1: Text/doc files only ───────────────────────────────────
        if session_has_docs and not session_has_csv:
            logger.info("📄  Text-only session → DOCUMENT")
            return "DOCUMENT"

        # ── Priority 1.5: Chart/visualisation requests ────────────────────────
        # Must be checked before vocab scoring — chart keywords pollute doc vocab.
        CHART_KEYWORDS = [
            'chart', 'plot', 'graph', 'pie', 'bar chart',
            'line chart', 'histogram', 'visualize', 'visualization',
        ]
        if session_has_csv and any(kw in question_lower for kw in CHART_KEYWORDS):
            logger.info("📊  Chart request → SQL")
            return "SQL"

        # ── Priority 2: Both file types — vocab-based source detection ────────
        csv_only_session = session_has_csv and set(
            f.split('.')[-1].lower() for f in document_files
        ).issubset({'csv', 'xlsx', 'xls'})

        if not csv_only_session and session_has_docs and session_has_csv:
            hint = self._detect_source_hint(question, conversation_id)

            COMPARISON_KEYWORDS    = ['compare', 'why', 'explain', 'causes', 'factors']
            HYBRID_OVERRIDE_KEYWORDS = [
                'compare', 'match', 'vs', 'versus', 'based on', 'at risk',
                'most likely', 'which employees', 'who are', 'actual', 'real data',
                'from the data', 'in the data', 'cross', 'combined', 'together', 'both',
            ]

            if hint == "csv":
                if any(kw in question_lower for kw in COMPARISON_KEYWORDS):
                    logger.info("🔀  Vocab→CSV but comparison keyword → HYBRID")
                    return "HYBRID"
                logger.info("📊  Vocab routing → SQL")
                return "SQL"

            elif hint == "document":
                if any(kw in question_lower for kw in HYBRID_OVERRIDE_KEYWORDS):
                    logger.info("🔀  Vocab→DOC but cross-source indicator → HYBRID")
                    return "HYBRID"
                logger.info("📄  Vocab routing → DOCUMENT")
                return "DOCUMENT"

            elif hint == "tie":
                logger.info("⚖️   Vocab tie → checking rerank dominance")
                dominant = self._rerank_dominant_source(conversation_id, question)
                if dominant == "csv":
                    logger.info("📊  Rerank dominance → SQL")
                    return "SQL"
                if dominant == "document":
                    logger.info("📄  Rerank dominance → DOCUMENT")
                    return "DOCUMENT"
                logger.info("🔀  Rerank mixed → HYBRID")
                return "HYBRID"

        # ── Priority 3: Explicit hybrid indicators ────────────────────────────
        COMPARISON_KW = [
            'compare', 'correlation', 'relationship', 'why', 'causes',
            'factors', 'difference between', 'what led to', 'how come',
        ]
        DEFINITION_KW = ['explain what', 'what does', 'what is', 'define', 'meaning of']
        DATA_KW       = ['show', 'list', 'top', 'count', 'calculate', 'numbers', 'data']

        if any(kw in question_lower for kw in COMPARISON_KW):
            logger.info("🔀  HYBRID: comparison/causal keyword")
            return "HYBRID"

        # FIX 3: pure_metric guard — "what is the maximum value in the dataset"
        # should route SQL, not HYBRID. Guard must live here in the active class.
        _pure_metric = any(m in question_lower for m in [
            'maximum', 'minimum', 'average', 'total', 'sum', 'count',
            'max value', 'min value', 'how many', 'how much',
            'highest value', 'lowest value',
        ])
        if (any(kw in question_lower for kw in DEFINITION_KW) and
                any(kw in question_lower for kw in DATA_KW) and not _pure_metric):
            logger.info("🔀  HYBRID: definition + data keywords")
            return "HYBRID"

        # ── Priority 4: CSV query router ──────────────────────────────────────
        if self.csv_router:
            route   = self.csv_router.route(question, has_csv_data=True)
            mapping = {"SQL": "SQL", "RAG": "DOCUMENT", "HYBRID": "HYBRID"}
            if route in mapping:
                logger.info(f"📊  CSV Router → {mapping[route]}")
                return mapping[route]

        # ── Priority 5: LLM fallback classifier ──────────────────────────────
        try:
            prompt   = self.routing_prompt.format(
                question=question,
                document_files=", ".join(document_files) if document_files else "None",
            )
            response = self.routing_llm.invoke(prompt).content.strip().upper()
            logger.info(f"🧭  LLM classification: {response}")
            if "HYBRID"   in response: return "HYBRID"
            if "DOCUMENT" in response: return "DOCUMENT"
            if "SQL"      in response: return "SQL"
        except Exception as e:
            logger.warning(f"⚠️  LLM classification error: {e}")

        logger.info("📊  Default → SQL")
        return "SQL"

    # ── Main routing method ───────────────────────────────────────────────────

    async def route_query(
        self,
        question: str,
        conversation_id: str = None,
        value_col: str = None,
    ) -> Dict[str, Any]:
        """
        Classify the question and dispatch it to the right agent over A2A.

        value_col is only used by the FORECAST route: the column the user picked
        in the Forecast tab. When None, the forecast agent infers it.
        """
        log_section("ORCHESTRATING QUERY")
        logger.info(f"❓  Question: {question}")
        logger.info(f"💬  Session:  {conversation_id or 'None'}")

        # Classification may call the LLM fallback, so run it in a worker thread
        classification = await asyncio.to_thread(self._classify_query, question, conversation_id)

        try:
            # ── FORECAST ─────────────────────────────────────────────────────
            if classification == "FORECAST":
                logger.info("📤  A2A → forecast_agent")
                message  = await self.send_message(
                    to_agent="forecast_agent",
                    message_type=MessageType.QUERY,
                    payload={
                        "question":        question,
                        "conversation_id": conversation_id,
                        "value_col":       value_col,
                    },
                )
                response = await self.registry.route_message(message)
                result   = response.payload
                result.update({
                    "routing":        "forecast",
                    "agent":          "forecast_agent",
                    "routing_reason": "Time series forecast via A2A → Prophet",
                })
                return result

            # ── GENERAL ──────────────────────────────────────────────────────
            elif classification == "GENERAL":
                logger.info("💬  GENERAL → LLM direct")
                system_prompt = (
                    "You are an AI Business Analyst assistant. "
                    "Help users analyse documents and CSV data. "
                    "Answer conversational questions naturally and helpfully. "
                    "If asked who you are: explain you are an AI analyst that can analyse "
                    "uploaded documents (PDF, TXT, DOCX) and CSV/Excel files. "
                    "Keep answers concise and friendly."
                )
                response = await asyncio.to_thread(
                    self.llm.invoke, f"{system_prompt}\n\nUser: {question}"
                )
                return {
                    "success":        True,
                    "answer":         response.content.strip(),
                    "routing":        "general",
                    "agent":          "llm",
                    "routing_reason": "Conversational or general knowledge query",
                }

            # ── DOCUMENT ─────────────────────────────────────────────────────
            elif classification == "DOCUMENT":
                logger.info("📤  A2A → rag_agent")
                message  = await self.send_message(
                    to_agent="rag_agent",
                    message_type=MessageType.QUERY,
                    payload={"question": question, "conversation_id": conversation_id},
                )
                response = await self.registry.route_message(message)
                result   = response.payload
                result.update({
                    "routing":        "document",
                    "agent":          "rag_agent",
                    "routing_reason": "Document/semantic query",
                })
                return result

            # ── SQL ───────────────────────────────────────────────────────────
            elif classification == "SQL":
                logger.info("📤  A2A → sql_agent")
                session_tables = self.rag_agent.get_csv_tables(conversation_id)

                if not session_tables:
                    logger.warning("⚠️  SQL route but no CSV tables — falling back to GENERAL")
                    response = await asyncio.to_thread(self.llm.invoke, question)
                    return {
                        "success":        True,
                        "answer":         response.content.strip(),
                        "routing":        "general",
                        "agent":          "llm",
                        "routing_reason": "SQL requested but no CSV uploaded",
                    }

                message  = await self.send_message(
                    to_agent="sql_agent",
                    message_type=MessageType.QUERY,
                    payload={"question": question, "allowed_tables": session_tables, "conversation_id": conversation_id},
                )
                response = await self.registry.route_message(message)
                result   = response.payload
                result.update({
                    "routing":        "sql",
                    "agent":          "sql_agent",
                    "routing_reason": "Calculation on uploaded CSV",
                })
                return result

            # ── HYBRID ────────────────────────────────────────────────────────
            else:
                logger.info("📤  A2A → sql_agent + rag_agent (HYBRID)")
                session_tables = self.rag_agent.get_csv_tables(conversation_id)

                sql_msg = await self.send_message(
                    to_agent="sql_agent",
                    message_type=MessageType.QUERY,
                    payload={"question": question, "allowed_tables": session_tables, "conversation_id": conversation_id},
                )
                rag_msg = await self.send_message(
                    to_agent="rag_agent",
                    message_type=MessageType.QUERY,
                    payload={"question": question, "conversation_id": conversation_id},
                )

                # The two agents are independent, so run them concurrently
                sql_response, rag_response = await asyncio.gather(
                    self.registry.route_message(sql_msg),
                    self.registry.route_message(rag_msg),
                )

                sql_payload = sql_response.payload
                rag_payload = rag_response.payload
                sql_text    = sql_payload.get("explanation", "")
                rag_text    = rag_payload.get("answer", "")

                sql_empty = (
                    not sql_text
                    or "not available" in sql_text.lower()
                    or sql_payload.get("row_count", 1) == 0
                )
                rag_empty = rag_payload.get("not_found", False)

                if sql_empty and rag_empty:
                    combined = "This information is not available in the uploaded files."
                elif sql_empty:
                    combined = (
                        await asyncio.to_thread(
                            self._synthesize_partial,
                            question, rag_text,
                            "The uploaded CSV data does not contain the specific "
                            "columns needed to answer this numerically.",
                        )
                        if rag_text.strip()
                        else rag_text.strip()
                    )
                elif rag_empty:
                    combined = sql_text.strip()
                else:
                    combined = await asyncio.to_thread(
                        self._synthesize_hybrid, question, sql_text, rag_text
                    )

                return {
                    "success":        True,
                    "question":       question,
                    "answer":         combined,
                    "explanation":    combined,
                    "sql_result":     sql_payload,
                    "rag_result":     rag_payload,
                    "routing":        "hybrid",
                    "agent":          "both",
                    "routing_reason": "Needs both calculation and context",
                }

        except Exception as e:
            logger.error(f"❌  Routing error: {e}")
            import traceback; traceback.print_exc()
            return {"success": False, "error": str(e), "question": question}

        finally:
            logger.info(f"✅  Routed as: {classification}")

    # ── Hybrid synthesis helpers ──────────────────────────────────────────────

    def _synthesize_partial(self, question: str, rag_text: str, sql_note: str) -> str:
        """
        Called when SQL has no data but RAG does.
        Produces an honest partial answer that acknowledges the data gap.
        """
        try:
            prompt = (
                f"You are a data analyst. Answer using only the document context below.\n\n"
                f"RULES:\n"
                f"1. Use ONLY the document context.\n"
                f"2. Acknowledge that CSV data could not provide the numerical comparison.\n"
                f"3. Be honest about what cannot be verified.\n"
                f"4. Under 100 words.\n\n"
                f"Document Context: {rag_text.strip()}\n"
                f"CSV Note: {sql_note}\n"
                f"Question: {question}\n\nAnswer:"
            )
            return self.synthesis_llm.invoke(prompt).content.strip()
        except Exception:
            return rag_text.strip()

    def _synthesize_hybrid(self, question: str, sql_text: str, rag_text: str) -> str:
        """
        Merge SQL numbers + RAG context into a single coherent answer.
        Uses LLM synthesis rather than string concatenation.
        """
        try:
            prompt = (
                f"You are a data analyst synthesising two sources.\n\n"
                f"RULES:\n"
                f"1. Use ONLY the facts below — no outside knowledge.\n"
                f"2. Combine numbers from SQL with context from Document into ONE answer.\n"
                f"3. If sources contradict, prefer the more specific one.\n"
                f"4. Under 120 words.\n\n"
                f"SQL Data:\n{sql_text.strip()}\n\n"
                f"Document Context:\n{rag_text.strip()}\n\n"
                f"Question: {question}\n\nSynthesised Answer:"
            )
            return self.synthesis_llm.invoke(prompt).content.strip()
        except Exception as e:
            logger.warning(f"⚠️  Synthesis failed: {e} — using concatenation")
            return (
                f"**Data Analysis:**\n{sql_text.strip()}\n\n"
                f"**Document Context:**\n{rag_text.strip()}"
            )

    # ── System status ─────────────────────────────────────────────────────────

    def get_system_status(self) -> Dict[str, Any]:
        rag_stats = self.rag_agent.get_global_stats()
        return {
            "orchestrator": {
                "agent_id":           self.agent_id,
                "status":             "active",
                "csv_hybrid_enabled": CSV_ROUTER_AVAILABLE,
            },
            "sql_agent": {
                "status":   "active",
                "database": settings.DB_PATH,
            },
            "rag_agent": {
                "status":          "active",
                "total_sessions":  rag_stats["total_sessions"],
                "active_sessions": rag_stats["active_sessions"],
                "total_vectors":   rag_stats["total_vectors"],
            },
            "forecast_agent": {
                "status":            "active",
                "prophet_available": PROPHET_AVAILABLE,
            },
            "a2a_registry": {
                "registered_agents": len(self.registry.agents),
                "agent_ids":         list(self.registry.agents.keys()),
            },
        }