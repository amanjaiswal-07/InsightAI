# main.py
"""
InsightAI API server (FastAPI).

Wires the agents together and exposes them over HTTP:

  Conversations
    POST   /conversations                         create a session
    GET    /conversations                         list a user's sessions
    GET    /conversations/{cid}                   session + visible messages
    DELETE /conversations/{cid}                   delete session, vectors, SQL tables
    GET    /conversations/{cid}/documents         files loaded into the session
    GET    /conversations/{cid}/processing-status background indexing state
    GET    /conversations/{cid}/schema            classified columns of the session's CSVs
    GET    /conversations/{cid}/dashboard         saved dashboard state
    POST   /conversations/{cid}/dashboard         save dashboard state

  Queries (API key + rate limit protected)
    POST   /query          JSON answer, saved to chat history
    POST   /query/stream   Server-Sent Events answer for the chat UI
    POST   /query/silent   JSON answer, NOT saved (dashboard / forecast tabs, benchmark)
    POST   /upload         save files, then index them in the background

  Ops / integrations
    GET    /health, /status, /api/metrics, /analytics
    POST   /mcp/initialize, GET /mcp/tools/list, POST /mcp/tools/call, GET /mcp/info
    GET    /a2a/registry

If the React app has been built (frontend/dist), it is served from "/".
"""

import asyncio
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from agents import OrchestratorAgent
from auth import limiter, verify_key
from chat.chat_history import ChatHistoryManager
from config import settings
from database import DatabaseManager
from mcp import MCPServer, MCPTool
from parsers.hybrid_csv_processor import HybridCSVProcessor
from utils.logger import log_section, logger
from utils.uploads import ALLOWED_UPLOAD_EXTENSIONS, is_valid_conversation_id, safe_upload_name

APP_VERSION = "3.4.0"
FRONTEND_DIST = Path("frontend/dist")
UPLOAD_ROOT = Path("./uploads")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))


# ── Core singletons ───────────────────────────────────────────────────────────

db_manager   = DatabaseManager()
orchestrator = OrchestratorAgent(db_manager)
chat_manager = ChatHistoryManager(
    db_path=settings.CHAT_DB_PATH,
    redis_host=settings.REDIS_HOST,
    redis_port=settings.REDIS_PORT,
)
mcp = MCPServer(name="insightai", version=APP_VERSION)

# Background indexing state keyed by conversation_id (in-process only)
_processing_state: dict = {}


# ── Startup helpers ───────────────────────────────────────────────────────────

def _drop_tables(tables: list):
    conn   = db_manager.get_connection()
    cursor = conn.cursor()
    for t in tables:
        cursor.execute(f'DROP TABLE IF EXISTS "{t}"')
        logger.info(f"🧹 Dropped table: {t}")
    conn.commit()
    conn.close()


def _cleanup_orphan_tables():
    """
    Drop upload_ tables whose conversation no longer exists.

    Tables of live conversations are kept so a session's CSV data survives a
    server restart (the RAG agent re-attaches them when the session reloads).
    """
    try:
        live_prefixes = {
            HybridCSVProcessor.session_table_prefix(cid)
            for cid in chat_manager.list_conversation_ids()
        }
        orphans = [
            t for t in db_manager.list_tables()
            if t.startswith(settings.CSV_TABLE_PREFIX)
            and not any(t.startswith(p) for p in live_prefixes)
        ]
        if orphans:
            _drop_tables(orphans)
            logger.info(f"🧹 Dropped {len(orphans)} orphan table(s)")
        else:
            logger.info("✅ No orphan tables")
    except Exception as e:
        logger.warning(f"⚠️  Orphan cleanup failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_section("INSIGHTAI STARTING")
    _cleanup_orphan_tables()
    logger.info(
        f"🚀  {settings.HOST}:{settings.PORT} | "
        f"MCP: {len(mcp.tools)} tools | "
        f"A2A: {len(orchestrator.registry.agents)} agents"
    )
    log_section("READY")
    yield
    logger.info("🛑  Shutdown complete")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="InsightAI", version=APP_VERSION, lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# No cookies are used (auth is a header), so credentials are not allowed;
# that keeps a wildcard origin list valid and safe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    value_col: Optional[str] = None  # explicit column chosen in the Forecast tab

class ConversationCreate(BaseModel):
    user_id: str
    title: Optional[str] = "New Chat"

class DashboardState(BaseModel):
    state: dict = {}

class MCPCallRequest(BaseModel):
    name: str
    arguments: dict = {}


def _get_user_id(x: Optional[str]) -> str:
    return x or "default_user"


def _answer_text(result: dict) -> str:
    """Text shown to the user: the explanation, else the answer, else the error."""
    text = (result.get("explanation") or result.get("answer") or "").strip()
    if not text and result.get("error"):
        text = _user_facing_error(str(result["error"]))
    return text


def _user_facing_error(error: str) -> str:
    """Turn an internal/provider error into a short message (details stay in the log)."""
    low = error.lower()
    if "rate limit" in low or "rate_limit" in low or "error code: 429" in low:
        return ("The language model is rate-limited right now (Groq free tier allows a "
                "limited number of tokens per minute). Please wait a few seconds and try again.")
    if "api key" in low or "error code: 401" in low:
        return "The language model rejected the request: check GROQ_API_KEY on the server."
    first_line = error.splitlines()[0] if error else "unknown error"
    return f"Sorry, I couldn't answer that: {first_line[:200]}"


def _save_assistant_message(cid: str, result: dict, content: str) -> None:
    """Persist the assistant reply, including its chart so it re-renders on reload."""
    chat_manager.add_message(
        conversation_id=cid,
        role="assistant",
        content=content,
        routing=result.get("routing"),
        plot=result.get("plot"),
        metadata={
            "sql_query": result.get("sql_query"),
            "sources":   result.get("sources", []),
        },
    )


def _require_cid(cid: Optional[str]) -> str:
    if not is_valid_conversation_id(cid):
        raise HTTPException(400, "A valid conversation id is required")
    return cid


# ── Background file processing ────────────────────────────────────────────────

async def _process_files_background(
    conversation_id: str,
    file_paths: List[str],
    file_names: List[str],
):
    _processing_state[conversation_id] = {
        "status": "processing", "files": file_names, "error": None
    }
    try:
        logger.info(f"📤  Background indexing: {file_names}")
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            orchestrator.rag_agent.load_documents,
            conversation_id,
            file_paths,
        )
        if not result.get("success"):
            raise RuntimeError(result.get("message", "No documents could be loaded"))
        _processing_state[conversation_id] = {
            "status": "ready", "files": file_names, "error": None
        }
        orchestrator.sql_agent.invalidate_schema_cache(conversation_id)
        logger.info(f"✅  Indexing complete: {conversation_id}")
    except Exception as e:
        _processing_state[conversation_id] = {
            "status": "error", "files": file_names, "error": str(e)
        }
        logger.error(f"❌  Indexing error: {e}")


# ── MCP tool handlers ─────────────────────────────────────────────────────────

class MCPToolHandlers:
    """
    Handlers for the MCP tools. Each one goes through orchestrator.route_query(),
    which classifies the question and dispatches it over the A2A registry.
    """

    def __init__(self, orch: OrchestratorAgent):
        self.orchestrator: OrchestratorAgent = orch

    async def handle_smart_query(self, args: dict) -> dict:
        """Route any question through the orchestrator."""
        question = args.get("question")
        conv_id  = args.get("conversation_id")
        if not question:
            return {"content": [{"type": "text", "text": "No question provided"}], "isError": True}
        result = await self.orchestrator.route_query(question, conv_id)
        return {
            "content": [{"type": "text", "text": _answer_text(result) or "No answer generated."}],
            "isError": not result.get("success", True),
        }

    async def handle_get_status(self, args: dict) -> dict:
        """Return current system and A2A agent status."""
        s    = self.orchestrator.get_system_status()
        text = (
            f"A2A agents: {s['a2a_registry']['registered_agents']} registered "
            f"({', '.join(s['a2a_registry']['agent_ids'])})\n"
            f"Sessions: {s['rag_agent']['total_sessions']} | "
            f"Vectors: {s['rag_agent']['total_vectors']} | "
            f"Prophet: {s['forecast_agent']['prophet_available']}"
        )
        return {"content": [{"type": "text", "text": text}], "isError": False}

    async def handle_forecast_query(self, args: dict) -> dict:
        """Forecast a metric from the session's uploaded CSV (requires conversation_id)."""
        question = args.get("question", "forecast the main metric for next 6 months")
        conv_id  = args.get("conversation_id", "")
        if not conv_id:
            return {
                "content": [{"type": "text", "text": "conversation_id is required"}],
                "isError": True,
            }
        result = await self.orchestrator.route_query(question, conv_id)
        return {
            "content": [{"type": "text", "text": _answer_text(result) or "Forecast failed"}],
            "isError": not result.get("success", False),
        }


mcp_handlers = MCPToolHandlers(orchestrator)

mcp.register_tool(MCPTool(
    name="smart_query",
    description="Route any business question to SQL, RAG, Hybrid, Forecast, or General automatically",
    inputSchema={
        "type": "object",
        "properties": {
            "question":        {"type": "string", "description": "Natural language question"},
            "conversation_id": {"type": "string", "description": "Active session ID"},
        },
        "required": ["question"],
    },
    handler=mcp_handlers.handle_smart_query,
))

mcp.register_tool(MCPTool(
    name="get_status",
    description="Return system status: agents, sessions, vector counts",
    inputSchema={"type": "object", "properties": {}},
    handler=mcp_handlers.handle_get_status,
))

mcp.register_tool(MCPTool(
    name="forecast_query",
    description=(
        "Forecast future values and detect anomalies from an uploaded CSV "
        "(Prophet + LLM narrative). Requires a conversation_id with a CSV uploaded."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "question":        {"type": "string", "description": "e.g. 'forecast revenue for next 6 months'"},
            "conversation_id": {"type": "string", "description": "Active session ID with uploaded CSV"},
        },
        "required": ["question", "conversation_id"],
    },
    handler=mcp_handlers.handle_forecast_query,
))


# ── Basic routes ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "InsightAI", "version": APP_VERSION}


@app.get("/ui")
async def serve_ui():
    """Kept for old bookmarks: the UI now lives at '/'."""
    return RedirectResponse("/")


# ── Conversation management ───────────────────────────────────────────────────

@app.post("/conversations")
async def create_conversation(
    request: ConversationCreate,
    x_user_id: Optional[str] = Header(None),
):
    user_id = x_user_id or request.user_id
    cid     = chat_manager.create_conversation(user_id, request.title)
    orchestrator.rag_agent.get_or_create_session(cid)
    logger.info(f"🆕  New conversation: {cid}")
    return {"success": True, "conversation_id": cid, "user_id": user_id}


@app.get("/conversations")
async def get_conversations(
    x_user_id: Optional[str] = Header(None),
    limit: int = 20,
):
    convs = chat_manager.get_user_conversations(_get_user_id(x_user_id), limit)
    for c in convs:
        s = orchestrator.rag_agent.get_session_status(c["conversation_id"])
        c["documents"] = s["total_documents"]
        c["vectors"]   = s["total_vectors"]
    return {"success": True, "conversations": convs, "count": len(convs)}


@app.get("/conversations/{cid}")
async def get_conversation(cid: str, limit: int = 50):
    conv = chat_manager.get_conversation(cid, limit)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    conv["session"] = orchestrator.rag_agent.get_session_status(cid)
    return {"success": True, "conversation": conv}


@app.delete("/conversations/{cid}")
async def delete_conversation(cid: str):
    chat_manager.delete_conversation(cid)
    orchestrator.rag_agent.delete_session(cid)
    orchestrator.sql_agent.invalidate_schema_cache(cid)
    _processing_state.pop(cid, None)
    try:
        prefix  = HybridCSVProcessor.session_table_prefix(cid)
        to_drop = [t for t in db_manager.list_tables() if t.startswith(prefix)]
        if to_drop:
            _drop_tables(to_drop)
    except Exception as e:
        logger.warning(f"⚠️  Table cleanup on delete failed: {e}")
    if is_valid_conversation_id(cid):
        shutil.rmtree(UPLOAD_ROOT / cid, ignore_errors=True)
    return {"success": True}


@app.get("/conversations/{cid}/documents")
async def get_documents(cid: str):
    s = orchestrator.rag_agent.get_session_status(cid)
    return {
        "success":         True,
        "conversation_id": cid,
        "documents":       s["loaded_files"],
        "total_documents": s["total_documents"],
        "total_vectors":   s["total_vectors"],
    }


@app.get("/conversations/{cid}/schema")
async def get_schema(cid: str):
    """
    Return classified columns for all CSV tables in this session.
    Every column is tagged as: date | numeric | categorical | id.
    Used by the Forecast tab (numeric_columns) and the Dashboard (all_columns).
    """
    import pandas as pd

    DATE_KEYWORDS = {"date", "time", "month", "year", "week", "day", "created", "updated", "period", "timestamp", "dt"}
    ID_KEYWORDS   = {"id", "key", "code", "uuid", "ref", "index", "no", "num", "number", "serial"}
    NUMERIC_TYPES = {"INT", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL", "NUMBER", "BIGINT", "SMALLINT"}

    try:
        csv_tables   = orchestrator.rag_agent.get_csv_tables(cid)
        schema       = db_manager.get_schema() if csv_tables else {}
        all_columns  = []
        numeric_cols = []
        date_cols    = []
        categorical  = []

        for table in csv_tables:
            cols = schema.get(table, [])

            # A few sample rows help classify TEXT columns that hold numbers/dates
            try:
                sample_df = db_manager.execute_query(f'SELECT * FROM "{table}" LIMIT 5')
            except Exception:
                sample_df = pd.DataFrame()

            for col in cols:
                name       = col["name"]
                col_type   = (col["type"] or "").upper()
                name_low   = name.lower().replace("_", " ").replace("-", " ")
                name_words = set(name_low.split())

                is_id   = bool(name_words & ID_KEYWORDS) or name_low.endswith(" id")
                is_date = (
                    bool(name_words & DATE_KEYWORDS)
                    or any(t in col_type for t in ("DATE", "TIME", "TIMESTAMP"))
                )
                is_num  = any(t in col_type for t in NUMERIC_TYPES)

                if not is_num and not is_date and not sample_df.empty and name in sample_df.columns:
                    series = sample_df[name].dropna()
                    if len(series):
                        if pd.to_numeric(series, errors="coerce").notna().sum() == len(series):
                            is_num = True
                        else:
                            try:
                                pd.to_datetime(series, errors="raise")
                                is_date = True
                            except Exception:
                                pass

                if is_id:
                    role = "id"
                elif is_date:
                    role = "date"
                    date_cols.append(name)
                elif is_num:
                    role = "numeric"
                    numeric_cols.append(name)
                else:
                    role = "categorical"
                    categorical.append(name)

                all_columns.append({
                    "name":        name,
                    "type":        col["type"] or "TEXT",
                    "role":        role,
                    "table":       table,
                    "primary_key": col.get("primary_key", False),
                })

        return {
            "success":             True,
            "conversation_id":     cid,
            "tables":              list(csv_tables),
            "all_columns":         all_columns,
            "numeric_columns":     list(dict.fromkeys(numeric_cols)),
            "date_columns":        list(dict.fromkeys(date_cols)),
            "categorical_columns": list(dict.fromkeys(categorical)),
        }
    except Exception as e:
        logger.error(f"❌ Schema endpoint error: {e}")
        return {"success": False, "numeric_columns": [], "all_columns": [], "error": str(e)}


@app.get("/conversations/{cid}/dashboard")
async def get_dashboard(cid: str):
    """Load persisted dashboard state for a session."""
    try:
        return {"success": True, "state": chat_manager.get_dashboard(cid)}
    except Exception as e:
        logger.error(f"❌ Dashboard load error: {e}")
        return {"success": False, "state": None, "error": str(e)}


@app.post("/conversations/{cid}/dashboard")
async def save_dashboard(cid: str, body: DashboardState):
    """Persist dashboard state for a session."""
    try:
        chat_manager.save_dashboard(cid, body.state)
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ Dashboard save error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/conversations/{cid}/processing-status")
async def processing_status(cid: str):
    state = _processing_state.get(cid)
    if not state:
        s = orchestrator.rag_agent.get_session_status(cid)
        if s.get("total_documents", 0) > 0:
            return {"status": "ready", "files": s.get("loaded_files", []), "error": None}
        return {"status": "unknown", "files": [], "error": None}
    return {"status": state["status"], "files": state["files"], "error": state["error"]}


# ── Query endpoints ───────────────────────────────────────────────────────────
# Note: slowapi requires the starlette Request parameter to be named exactly
# "request", so the JSON body is named "body".

@app.post("/query", dependencies=[Depends(verify_key)])
@limiter.limit("30/minute")
async def query(
    request: Request,
    body: QueryRequest,
    x_user_id: Optional[str] = Header(None),
):
    """Synchronous query. Returns data, plot spec, explanation, routing, sql_query, sources."""
    try:
        uid = _get_user_id(x_user_id)
        cid = body.conversation_id or chat_manager.create_conversation(uid)
        orchestrator.rag_agent.get_or_create_session(cid)

        chat_manager.add_message(conversation_id=cid, role="user", content=body.question)
        result = await orchestrator.route_query(body.question, cid, value_col=body.value_col)
        _save_assistant_message(cid, result, _answer_text(result))
        result["conversation_id"] = cid
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"❌  Query error: {e}")
        raise HTTPException(500, str(e))


@app.post("/query/silent", dependencies=[Depends(verify_key)])
@limiter.limit("60/minute")
async def query_silent(
    request: Request,
    body: QueryRequest,
):
    """
    Same pipeline as /query but saves NO chat history. Used by the Dashboard
    and Forecast tabs (and scripts/benchmark.py) so their background queries
    never appear in the user's chat thread.
    """
    cid = body.conversation_id
    if not cid:
        raise HTTPException(400, "conversation_id required for silent query")
    try:
        orchestrator.rag_agent.get_or_create_session(cid)
        result = await orchestrator.route_query(body.question, cid, value_col=body.value_col)
        result["conversation_id"] = cid
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"❌ Silent query error: {e}")
        raise HTTPException(500, str(e))


@app.post("/query/stream", dependencies=[Depends(verify_key)])
@limiter.limit("20/minute")
async def query_stream(
    request: Request,
    body: QueryRequest,
    x_user_id: Optional[str] = Header(None),
):
    """
    SSE streaming endpoint for the Chat UI.

    Event sequence:
      event: meta   → JSON with routing, plot, data, sql_query, sources
      event: token  → one word chunk of the explanation text
      event: done   → end of stream signal
      event: error  → on exception
    """
    uid = _get_user_id(x_user_id)
    cid = body.conversation_id or chat_manager.create_conversation(uid)
    orchestrator.rag_agent.get_or_create_session(cid)
    chat_manager.add_message(conversation_id=cid, role="user", content=body.question)

    async def event_generator():
        try:
            result = await orchestrator.route_query(body.question, cid, value_col=body.value_col)

            # Non-text fields first so the UI can render the routing tag and
            # chart before the explanation starts streaming.
            meta = {
                "success":         result.get("success", True),
                "routing":         result.get("routing", ""),
                "plot":            result.get("plot"),
                "data":            result.get("data", []),
                "sql_query":       result.get("sql_query"),
                "sources":         result.get("sources", []),
                "conversation_id": cid,
            }
            yield f"event: meta\ndata: {json.dumps(meta, default=str)}\n\n"

            # Persist before streaming: if the client disconnects mid-stream the
            # generator is cancelled, and the answer must not be lost.
            explanation = _answer_text(result)
            _save_assistant_message(cid, result, explanation)

            # The agents return the full answer at once; re-stream it word by
            # word so the UI feels responsive without per-agent token streaming.
            words = explanation.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"event: token\ndata: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.018)

            yield f"event: done\ndata: {json.dumps({'conversation_id': cid})}\n\n"

        except Exception as e:
            logger.error(f"❌  Stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/upload", dependencies=[Depends(verify_key)])
@limiter.limit("10/minute")
async def upload(
    request: Request,
    files: List[UploadFile] = File(...),
    conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
):
    cid = _require_cid(conversation_id)

    # Validate everything before writing anything
    names = []
    for f in files:
        name = safe_upload_name(f.filename)        # strips any client-side path
        if not name:
            raise HTTPException(
                400,
                f"Unsupported file '{f.filename}'. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
            )
        names.append(name)

    upload_dir = UPLOAD_ROOT / cid
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    paths = []
    for f, name in zip(files, names):
        dest    = upload_dir / name
        written = 0
        with open(dest, "wb") as buf:
            while chunk := f.file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    buf.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"'{name}' is larger than {MAX_UPLOAD_MB} MB")
                buf.write(chunk)
        paths.append(str(dest))
        logger.info(f"✅  Saved: {name} ({written / 1024:.0f} KB)")

    asyncio.create_task(_process_files_background(cid, paths, names))
    return JSONResponse({"success": True, "loaded_files": names, "processing": True})


# ── Status / metrics ──────────────────────────────────────────────────────────

@app.get("/status")
async def status():
    return JSONResponse(orchestrator.get_system_status())


@app.get("/api/metrics")
async def api_metrics():
    sys = orchestrator.get_system_status()
    return {
        "success":        True,
        "sql_agent":      sys["sql_agent"]["status"],
        "rag_agent":      sys["rag_agent"]["status"],
        "total_sessions": sys["rag_agent"]["total_sessions"],
        "total_vectors":  sys["rag_agent"]["total_vectors"],
        "a2a_agents":     sys["a2a_registry"]["registered_agents"],
    }


@app.get("/analytics")
async def analytics(x_user_id: Optional[str] = Header(None)):
    return {
        "success":   True,
        "analytics": chat_manager.get_analytics(_get_user_id(x_user_id)),
    }


# ── MCP endpoints ─────────────────────────────────────────────────────────────

@app.post("/mcp/initialize")
async def mcp_initialize(req: dict):
    return await mcp.handle_initialize(req)


@app.get("/mcp/tools/list")
async def mcp_tools_list():
    return await mcp.handle_list_tools()


@app.post("/mcp/tools/call", dependencies=[Depends(verify_key)])
async def mcp_tools_call(body: MCPCallRequest):
    return JSONResponse(await mcp.handle_call_tool(body.name, body.arguments))


@app.get("/mcp/info")
async def mcp_info():
    return mcp.get_server_info()


# ── A2A registry ──────────────────────────────────────────────────────────────

@app.get("/a2a/registry")
async def a2a_registry():
    """Live snapshot of the A2A agent registry (agents, capabilities, peers)."""
    registered = {
        agent_id: {
            "agent_id":        agent.agent_id,
            "name":            agent.name,
            "description":     agent.description,
            "capabilities":    list(agent.capabilities.keys()),
            "connected_peers": list(agent.connected_agents.keys()),
        }
        for agent_id, agent in orchestrator.registry.agents.items()
    }
    return {
        "status":           "active",
        "total_agents":     len(registered),
        "agents":           registered,
        "protocol_version": "a2a-v1",
        "mcp_tools":        list(mcp.tools.keys()),
    }


# ── Built React frontend (production / single-container deploy) ───────────────
# Registered last so every API route above takes precedence over the SPA.

if (FRONTEND_DIST / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        candidate = (FRONTEND_DIST / full_path).resolve()
        if full_path and candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)          # favicon.svg etc.
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    async def root():
        return {
            "service": "InsightAI",
            "version": APP_VERSION,
            "hint":    "Build the UI with `npm run build` in frontend/, or run `npm run dev` on :3000",
        }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info")
