# InsightAI

**Ask business questions about your own spreadsheets and documents in plain language, and get answers backed by exact SQL, cited document passages, charts and forecasts.**

InsightAI is a full-stack AI analytics assistant. Upload CSV/Excel files and PDF/DOCX/TXT documents into a session, then ask questions in English or another language. An orchestrator decides whether each question needs a SQL calculation, a document search (RAG), both, a time-series forecast, or just a conversational reply, and sends it to the right agent.

![CI](https://github.com/amanjaiswal-07/InsightAI/actions/workflows/ci.yml/badge.svg)

**Contents:** [Screenshots](#screenshots) · [Why](#why-this-exists) · [Features](#key-features) · [Tech stack](#tech-stack) · [Architecture](#architecture) · [How it works](#how-it-works) · [Project structure](#project-structure) · [Getting started](#getting-started) · [Deployment](#deployment) · [Using it](#using-it) · [Configuration](#configuration) · [API](#api) · [Data model](#data-model) · [Testing and evaluation](#testing-and-evaluation) · [Limitations](#known-limitations) · [Future work](#future-improvements) · [Author](#author)

## Screenshots

These are captured from the app running locally on the files in [`sample_data/`](sample_data). There is no hosted demo yet; see [Getting started](#getting-started) to run it yourself.

**SQL answer with chart:** "Show a bar chart of total revenue by region" is routed to the SQL agent, answered from the uploaded CSV, and charted.

![SQL answer with bar chart](docs/screenshots/chat-sql-chart.jpg)

**Document answer with source:** a question about an uploaded Word file is routed to RAG and answered with a citation.

![Document answer citing the source file](docs/screenshots/chat-document-answer.jpg)

**Hallucination guard and hybrid answer:** a question the documents can't answer is refused instead of guessed. A "why + numbers" question combines SQL figures with document context.

![Refused out-of-scope question and a hybrid SQL plus document answer](docs/screenshots/chat-refusal-and-hybrid.jpg)

**Forecast tab:** Prophet forecast of a chosen column, with the AI-written summary and flagged anomalies.

![Prophet forecast with actual vs forecast series](docs/screenshots/forecast-tab.jpg)

---

## Why this exists

Business questions usually span two kinds of data:

- **Structured data** (sales exports, HR sheets), where answers must be *exact*. An LLM reading raw CSV text will guess at totals and averages.
- **Unstructured documents** (plans, reviews, reports), where answers need *retrieval and citations*, and the model must refuse when the document doesn't say.

A plain chatbot handles neither reliably. InsightAI routes each question to the method that can answer it correctly: generated SQL for numbers, hybrid retrieval with reranking for documents, and Prophet for forecasts. It refuses instead of inventing an answer when nothing relevant is found.

## Key features

| Feature | How it works |
|---|---|
| **Natural language → SQL** | The LLM writes SQLite for the session's uploaded tables. The real column values are injected into the prompt so `WHERE` clauses match the data exactly. If the query fails, the database error is sent back to the LLM for **one self-correction retry**. |
| **Read-only SQL guard** | All LLM SQL runs through a statement allow-list (single `SELECT`/`WITH`, no `INSERT`/`DROP`/`ATTACH`/...) **and** a read-only SQLite connection. |
| **Hybrid RAG** | BM25 (keyword) + FAISS (semantic) ensemble → cross-encoder reranking → top-3 chunks → grounded answer with source file names. |
| **Hallucination guard** | If the reranker scores for the best chunks are below a threshold, the answer is *"This information is not available in the uploaded documents."* instead of a guess. A query-expansion retry runs first for low-scoring questions. |
| **CSV → SQL + RAG** | Each CSV becomes both a SQL table *and* narrative text chunks (one per row, one per category group, one dataset overview), so it can answer exact *and* descriptive questions. |
| **Routing** | Rule cascade (greetings, forecast keywords, chart requests, vocabulary overlap with each source, causal keywords) with an LLM classifier as the last fallback. |
| **Hybrid answers** | SQL and RAG agents run concurrently; the LLM merges the numbers and the context into one answer. |
| **Forecasting** | Prophet with 95% intervals, automatic date/value column detection, anomaly flags for points outside the interval, and an LLM-written business summary. |
| **Charts** | Bar / line / pie specs generated from SQL results and rendered with Chart.js. Charts can be pinned to per-session dashboards. |
| **Multilingual** | Language detection (script-based for CJK/Arabic/Hindi, `langdetect` otherwise); answers come back in the question's language. |
| **Reliable LLM calls** | One client factory (`utils/llm.py`) gives every agent the same timeout and retry-with-backoff policy, so a Groq 429 or a transient 5xx is retried instead of failing the question. |
| **Streaming chat** | Server-Sent Events: routing tag and chart arrive first, then the answer text. |
| **Integrations** | Minimal MCP-style tool endpoints (`/mcp/*`) and an agent-to-agent message registry (`/a2a/registry`). |
| **Evaluated** | An offline pytest suite (CI) plus a live evaluation harness that scores routing, SQL exactness, RAG faithfulness, refusals and language detection. |

## Tech stack

| Layer | Technology |
|---|---|
| Backend API | Python 3.12, FastAPI, Uvicorn, slowapi (rate limits, API keys) |
| LLM | Groq API (`openai/gpt-oss-120b` by default) via LangChain |
| Retrieval | FAISS, rank_bm25, `sentence-transformers/all-MiniLM-L6-v2` embeddings, `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker |
| Data | pandas, SQLite, openpyxl, pypdf, python-docx |
| Forecasting | Prophet |
| Frontend | React 19, Vite, Tailwind CSS 4, Chart.js, marked |
| Testing / CI | pytest, GitHub Actions, custom LLM evaluation harness |

## Architecture

```mermaid
flowchart LR
    UI[React UI] -- REST + SSE --> API[FastAPI main.py]
    API -- upload --> ING[Ingestion]
    ING -- CSV / XLSX --> HCP[HybridCSVProcessor]
    HCP --> SQLDB[(SQLite tables<br/>upload_session_file)]
    HCP --> CH[Narrative chunks]
    ING -- PDF / DOCX / TXT --> DP[DocumentParser] --> CH
    CH --> VS[(FAISS index per session<br/>+ BM25 corpus)]

    API -- question --> ORC{Orchestrator<br/>routing cascade}
    ORC -- GENERAL --> LLM[Groq LLM]
    ORC -- SQL --> SQLA[SQL Analyst] --> SQLDB
    ORC -- DOCUMENT --> RAG[RAG Agent] --> VS
    ORC -- HYBRID --> SQLA & RAG
    ORC -- FORECAST --> FC[Prophet Forecast Agent] --> SQLDB
    API --- HIST[(Chat history + dashboards<br/>SQLite)]
```

Agents never call each other directly: the orchestrator wraps each one as an A2A peer and sends messages through `protocols/a2a.py`'s registry.

## How it works

### End-to-end workflow

From uploading files to a streamed answer:

```mermaid
sequenceDiagram
    actor User
    participant UI as React UI
    participant API as FastAPI
    participant IDX as Indexer (background)
    participant ORC as Orchestrator
    participant AG as Agent(s)
    participant LLM as Groq LLM

    User->>UI: Upload CSV / XLSX / PDF / DOCX / TXT
    UI->>API: POST /upload
    API-->>UI: accepted, processing
    API->>IDX: index files
    IDX->>IDX: CSV to SQLite table + narrative chunks
    IDX->>IDX: documents to chunks, embeddings into FAISS
    UI->>API: poll /processing-status
    API-->>UI: ready

    User->>UI: Ask a question
    UI->>API: POST /query/stream
    API->>ORC: route_query()
    ORC->>ORC: rule cascade, LLM fallback
    ORC->>AG: A2A message (SQL, RAG, Forecast or both)
    AG->>LLM: write SQL / answer from context / narrative
    LLM-->>AG: text
    AG-->>ORC: answer, rows, chart spec, sources
    ORC-->>API: result
    API-->>UI: SSE meta (routing, chart, SQL, sources)
    API->>API: save reply + chart to chat history
    API-->>UI: SSE tokens, then done
```

### Query routing

`agents/orchestrator.py` tries the cheapest rules first and only asks the LLM when the rules can't decide (simplified):

```mermaid
flowchart TD
    Q[Question] --> G{Greeting or general knowledge?}
    G -- yes --> GEN([GENERAL: LLM directly])
    G -- no --> F{Forecast wording and a CSV uploaded?}
    F -- yes --> FC([FORECAST: Prophet agent])
    F -- no --> U{What has been uploaded?}
    U -- nothing --> GEN
    U -- documents only --> DOC([DOCUMENT: RAG agent])
    U -- CSV --> C{Chart request?}
    C -- yes --> SQL([SQL: SQL agent])
    C -- no --> M{CSV and documents both present?}
    M -- yes --> V{Vocabulary vote, rerank tie-break}
    V -- CSV terms --> SQL
    V -- document terms --> DOC
    V -- mixed or cross-source wording --> HY([HYBRID: SQL + RAG])
    M -- no --> K{Causal or comparison wording?}
    K -- yes --> HY
    K -- no --> R[Regex router, then LLM classifier]
    R --> SQL
    R --> DOC
    R --> HY
```

### SQL agent: generation, validation, self-correction

```mermaid
flowchart TD
    Q[Question] --> S[Schema + real column values]
    S --> L1[LLM writes SQLite]
    L1 --> SAN[Sanitise: strip fences, fix bare aggregates]
    SAN --> NORM[Normalise literals to stored values]
    NORM --> GUARD{Read-only guard}
    GUARD --> EX[Execute on read-only connection]
    EX --> OK{Succeeded?}
    OK -- no --> FIX[LLM fixes SQL using the error, once] --> GUARD
    OK -- yes --> ONE{Single number?}
    ONE -- yes --> FAST[Direct answer, no extra LLM call]
    ONE -- no --> EXP[LLM explanation + chart spec]
```

A failed retry is returned as a clear error rather than looping.

### Ingestion and RAG pipeline

```mermaid
flowchart LR
    subgraph Ingest
        CSV[CSV / XLSX] --> TBL[(SQLite table)]
        CSV --> NAR[Row narratives, group summaries, dataset overview]
        DOCS[PDF / DOCX / TXT] --> PAR[Chunks, headings kept with their content]
        NAR --> EMB[MiniLM embeddings] --> FAISS[(FAISS)]
        PAR --> EMB
        NAR --> BM[(BM25 corpus)]
        PAR --> BM
    end
    subgraph Answer
        QU[Question] --> HYB[BM25 + FAISS, top 10]
        HYB --> RR[Cross-encoder rerank, top 3]
        RR --> LOW{Top score low?}
        LOW -- yes --> EXP[LLM query expansion, re-retrieve, rerank] --> TH
        LOW -- no --> TH{Relevant enough?}
        TH -- no --> NA([Not available in the uploaded documents])
        TH -- yes --> ANS([Answer from top 3 chunks + source file])
    end
```

The **forecast agent** loads only the session's own table, detects the date and value columns, infers the frequency (daily, weekly, monthly, quarterly), fits Prophet with 95% intervals, flags points outside the interval as anomalies, and asks the LLM for a short business summary. A template summary is used if the LLM is unavailable.

## Project structure

```
InsightAI/
├── main.py                  FastAPI app: all HTTP routes, upload handling, serves the built UI
├── config.py                Settings from environment / .env
├── auth.py                  API-key dependency + rate limiter
├── agents/
│   ├── orchestrator.py      Routing cascade + A2A dispatch + hybrid synthesis
│   ├── sql_analyst.py       NL → SQL, self-correction retry, explanation, chart spec
│   ├── session_rag.py       Per-session BM25 + FAISS + cross-encoder RAG with refusal guard
│   └── forecast_agent.py    Prophet forecasting + anomaly detection
├── parsers/
│   ├── hybrid_csv_processor.py   CSV/XLSX → SQL table + narrative RAG chunks
│   └── document_parser.py        PDF / DOCX / TXT → LangChain documents
├── database/
│   ├── db_manager.py        SQLite access, read-only query guard
│   └── sample_data.py       Sample business DB used by the evaluation suite
├── tools/                   SQL executor, value normaliser, CSV router, language detector
├── protocols/a2a.py         Agent-to-agent message registry
├── mcp/mcp_server.py        MCP-style tool server behind /mcp/*
├── chat/chat_history.py     Conversations, messages, dashboards (SQLite + Redis/fakeredis cache)
├── utils/                   LLM client factory, logger, plot-spec builder, upload validation
├── frontend/                React app (Chat, Dashboard and Forecast tabs)
├── tests/                   Offline pytest suite (no network, no model downloads)
├── evaluation/              Live LLM evaluation harness + latest report
├── scripts/benchmark.py     HTTP benchmark against a running server
├── sample_data/             Example CSV / XLSX / DOCX / TXT files to try
├── Dockerfile               Single container: API + built UI (any Docker host)
└── .github/workflows/ci.yml Tests + frontend build on every push
```

## Getting started

### Prerequisites

- Python **3.11 or 3.12** (the pinned `numpy`/LangChain versions do not support 3.13 yet)
- Node.js 20+
- A free Groq API key: <https://console.groq.com/keys>

The first start downloads two small Hugging Face models (about 200 MB total).

### 1. Backend

```bash
git clone https://github.com/amanjaiswal-07/InsightAI.git
cd InsightAI
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux), then:

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY`. Then start the API on <http://localhost:8000> (interactive docs at `/docs`):

```bash
python main.py
```

### 2. Frontend (development)

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>. The Vite dev server proxies API calls to port 8000.

### Single-server mode

Build the UI once and FastAPI serves it at <http://localhost:8000>:

```bash
cd frontend && npm run build
```

### Docker

```bash
docker build -t insightai .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key insightai
```

## Deployment

The repo includes a `Dockerfile` that builds the React UI and serves it from FastAPI on port 7860, so the whole app is one container. Give it at least about 2 GB of RAM: torch, the two models and Prophet do not fit in 512 MB free tiers. Any host that runs a Docker image works, for example Google Cloud Run or a small VM.

### Hugging Face Spaces

As of October 2026, Hugging Face requires a paid plan (PRO) for Docker Spaces; only static Spaces are free.

1. Create a Space at <https://huggingface.co/new-space> and choose **Docker** (blank template).
2. In the Space's **Settings → Variables and secrets**, add the secret `GROQ_API_KEY`. Optionally add `API_KEYS` as well.
3. Push the code to the Space's git remote. Hugging Face rejects binary files that are not in Git LFS, so track the sample `.docx`/`.xlsx` files with `git lfs track "*.docx" "*.xlsx"` in a fresh snapshot of the repo first:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-user>/<space-name>
   ```

   ```bash
   git push space main
   ```

4. The Space builds the image (about 10 minutes) and serves the app at `https://<your-user>-<space-name>.hf.space`.

The Space's README needs Hugging Face front matter (`sdk: docker`, `app_port: 7860`); add it in the Space if the build asks for it.

Notes:

- Every visitor uses **your** Groq quota.
- Uploaded files and sessions are wiped whenever the Space restarts.

## Using it

1. Click **New session** and upload files, for example everything in `sample_data/`.
2. Wait for the "processing" indicator to finish (CSV import, chunking and embeddings).
3. Ask questions. Some examples with the sample files:
   - `What is the total revenue by region?` → **SQL**
   - `Show a bar chart of units sold by product` → **SQL** + chart
   - `What budget was approved for AI integration?` → **DOCUMENT** with source
   - `Why did profit change and what do the numbers show?` → **HYBRID**
   - `What is the CEO's phone number?` → refused (not in the documents)
4. **Forecast tab:** choose a numeric column and horizon, then run a Prophet forecast (`sample_data/forecast.csv` works well).
5. **Dashboard tab:** pin charts from the chat into named dashboards.

## Configuration

All settings live in `.env` (see [`.env.example`](.env.example)). Only `GROQ_API_KEY` is required.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | **Required.** Groq API key |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Any Groq chat model |
| `API_KEYS` | empty | Comma-separated keys for the `X-Api-Key` header. Empty disables auth (local dev). |
| `CORS_ORIGINS` | `*` | Allowed browser origins |
| `MAX_UPLOAD_MB` | `25` | Per-file upload limit |
| `DB_PATH` / `CHAT_DB_PATH` / `VECTOR_STORE_PATH` | `./data/...` | Where SQLite DBs and FAISS indexes are stored |
| `EMBEDDING_MODEL` / `RERANKER_MODEL` | MiniLM models | Hugging Face model ids |
| `REDIS_HOST` / `REDIS_PORT` | `localhost:6379` | Optional chat-history cache. Falls back to in-process fakeredis. |
| `LLM_TIMEOUT` / `LLM_MAX_RETRIES` | `60` / `5` | Per-request timeout and retries (429 / 5xx / timeout) |
| `LOG_LEVEL` / `LOG_FILE` | `INFO` / `./logs/app.log` | Logging |

Frontend variables (`frontend/.env`): `VITE_API_BASE_URL` (empty means same origin) and `VITE_API_KEY` (only if `API_KEYS` is set; it ends up in the browser bundle, so it is not a secret).

## API

Interactive OpenAPI docs: `http://localhost:8000/docs`.

| Method | Path | Description |
|---|---|---|
| `POST` | `/conversations` | Create a session `{user_id, title}` |
| `GET` | `/conversations` | List sessions for `X-User-ID` |
| `GET` / `DELETE` | `/conversations/{id}` | Get messages / delete session, vectors, tables and uploads |
| `POST` | `/upload` | Multipart `files`, header `X-Conversation-ID`. Indexing runs in the background. |
| `GET` | `/conversations/{id}/processing-status` | `processing` / `ready` / `error` |
| `GET` | `/conversations/{id}/schema` | Uploaded columns classified as date / numeric / categorical / id |
| `POST` | `/query` | JSON answer, saved to history |
| `POST` | `/query/stream` | SSE answer (`meta`, `token`, `done`, `error` events) |
| `POST` | `/query/silent` | JSON answer, not saved (dashboard / forecast tabs) |
| `GET` / `POST` | `/conversations/{id}/dashboard` | Load / save dashboard state |
| `GET` | `/health`, `/status`, `/api/metrics` | Health and agent status |
| `GET` | `/mcp/tools/list` · `POST /mcp/tools/call` | MCP-style tools: `smart_query`, `forecast_query`, `get_status` |
| `GET` | `/a2a/registry` | Registered agents and their capabilities |

Example (after uploading a CSV to session `$CID`):

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"What is the total revenue by region?\", \"conversation_id\": \"$CID\"}"
```

The response includes `routing`, `sql_query`, `data` (rows), `explanation`, and a `plot` spec when a chart was requested.

## Data model

- **Uploaded tables** (`DB_PATH`): one table per uploaded CSV/XLSX, named `upload_<first 8 chars of session id>_<file name>`. Tables are scoped to their session. They survive restarts and are dropped when the session is deleted, or at startup if the session no longer exists.
- **Chat history** (`CHAT_DB_PATH`): `conversations(conversation_id, user_id, title, ...)`, `messages(message_id, conversation_id, role, content, routing, metadata JSON, is_visible, ...)`, `dashboards(conversation_id, state JSON, ...)`. Charts are stored in message metadata so they re-render when a session is reopened.
- **Vectors** (`VECTOR_STORE_PATH/<session id>/`): FAISS index plus docstore. On reload the chunks, file list and CSV tables are rebuilt from it.

## Testing and evaluation

### Offline tests (no API key, no network)

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover:

- the read-only SQL guard and SQL sanitising;
- the SQL agent end to end on real SQLite tables with fake LLM chains (including the self-correction retry and the single-value vs grouped-result handling);
- the routing cascade and the A2A forecast path;
- CSV ingestion and session isolation;
- forecast column and frequency detection;
- upload filename and session-id validation;
- DOCX chunking (headings kept with their content);
- SQL literal normalisation (identifiers never rewritten);
- the LLM client factory settings;
- RAG session restore after a restart.

CI runs these tests plus the frontend lint and build on every push.

### Live LLM evaluation

`evaluation/eval_suite.py` runs the real agents with real Groq calls, in-process. It needs `GROQ_API_KEY`.

```bash
python evaluation/eval_suite.py
python evaluation/eval_suite.py --skip-rag
```

The second command skips document ingestion and runs faster.

| Suite | What it checks |
|---|---|
| Routing accuracy | 15 labeled questions across SQL / DOCUMENT / HYBRID / FORECAST / GENERAL, each in a correctly-scoped session |
| SQL correctness | Executed answer vs ground truth computed independently in Python from `database/sample_data.py` |
| RAG faithfulness | Answers about the two sample `.docx` files must contain facts taken from those files |
| RAG refusal | Out-of-scope questions (phone numbers, live stock prices, ...) must be refused |
| Language detection | 9 languages |

**Latest run (2026-10-20, `openai/gpt-oss-120b` on Groq): 40/42 (95.2%)**

| Suite | Result | Avg latency |
|---|---|---|
| Routing accuracy | 13/15 (86.7%) | 2.9 s |
| SQL correctness | 8/8 (100%) | 8.5 s |
| RAG faithfulness | 5/5 (100%) | 1.3 s |
| RAG refusal | 5/5 (100%) | 3.5 s |
| Language detection | 9/9 (100%) | under 0.01 s |

The two misses are both expected-`HYBRID` questions. "Explain the sales trend and show the underlying numbers" went to `SQL`, and "Why did engineering perform well, and what were the department's total sales?" went to `DOCUMENT`. They are reported as-is rather than tuned away; the full per-case detail is in [`evaluation/eval_report.txt`](evaluation/eval_report.txt).

These are small hand-labeled sets (42 cases), so treat them as regression checks rather than a benchmark. Of four full runs during this cleanup, the other three scored 39/42 or 40/42. Every 39/42 run lost its extra case to a Groq 429 rate-limit error, which is what led to the retry policy in `utils/llm.py`.

Results are written to `evaluation/eval_report.txt` and `evaluation/eval_results.json`. LLM output varies a little between runs.

`scripts/benchmark.py` measures latency and routing over HTTP against a running server (`BENCHMARK_API_KEY` / `BENCHMARK_BASE_URL` env vars).

## Known limitations

- **SQLite only for uploads.** `DatabaseManager` can connect to Postgres/MySQL, but the CSV import, cleanup and schema code use SQLite syntax, and the drivers are not in `requirements.txt`.
- **Single process.** Upload-processing status and the session cache live in memory, so run one worker.
- **Groq rate limits.** On the free tier, `openai/gpt-oss-120b` allows 8,000 tokens per minute. The SQL-generation call alone sends about 1,700 tokens (schema, sample values and few-shot examples), so only a few questions per minute fit. Past that, the UI shows a "rate-limited, try again" message. A paid tier or a smaller `LLM_MODEL` raises the ceiling.
- **Routing is heuristic.** Ambiguous questions that mix calculation and explanation can land on `SQL` or `DOCUMENT` instead of `HYBRID`.
- **Streaming is simulated.** Agents produce the full answer first; `/query/stream` then streams it word by word.
- **`.xls` (legacy Excel)** is not accepted; save as `.xlsx` or `.csv`.
- **Uploads are not encrypted** and there are no user accounts: `X-User-ID` only separates session lists and is not authentication.

## Future improvements

These are *not* implemented yet:

- True token streaming from the LLM in each agent
- Postgres support for uploaded tables (dialect-aware import and cleanup)
- User authentication and per-user storage quotas
- Evaluation tracked over time in CI with a mocked or cached LLM
- Smaller frontend bundle (code-split Chart.js and marked)

## License

[MIT](LICENSE)

## Author

**Aman Jaiswal**

- LinkedIn: [linkedin.com/in/aman-jaiswal-aa31b51b5](https://www.linkedin.com/in/aman-jaiswal-aa31b51b5)
- GitHub: [@amanjaiswal-07](https://github.com/amanjaiswal-07)
