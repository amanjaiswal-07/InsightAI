# agents/session_rag.py
"""
Session-Aware RAG Agent
=======================
Three-stage hybrid retrieval: BM25 (lexical) + FAISS (semantic) → ensemble
→ cross-encoder reranking with dual-threshold hallucination guards.

Key design decisions:
  - Every chunk tagged with source_type ("csv" or "document") at ingest time
  - Per-source vocabulary built after ingest to enable orchestrator routing
  - CSV column names pinned into csv vocab to prevent overlap with doc text
  - Rerank threshold guard: returns "not available" rather than hallucinating
  - Lazy query expansion: only fires when top rerank score is poor
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil
import re
from collections import Counter, OrderedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from pydantic import ConfigDict

from parsers.document_parser import DocumentParser
from parsers.hybrid_csv_processor import HybridCSVProcessor
from database.db_manager import DatabaseManager
from tools.language_detector import LanguageDetector
from config import settings, ENABLE_HYBRID_CSV
from utils.llm import make_llm
from utils.logger import logger, log_section


class _LRUSessionCache(OrderedDict):
    """
    Bounded in-memory cache of loaded FAISS sessions, evicted least-recently-used.

    Each session's FAISS index is also persisted to disk (get_or_create_session
    reloads via FAISS.load_local on a miss), so evicting an entry here only
    means the next access pays a reload cost — it never loses data. Without
    this bound, session_stores grows for the life of the process: every
    conversation ever opened keeps its FAISS index resident in memory until
    the user explicitly deletes that conversation.
    """

    def __init__(self, max_size: int = 50):
        super().__init__()
        self.max_size = max_size

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.max_size:
            evicted_id, _ = self.popitem(last=False)
            logger.info(f"🧹  Evicted session from memory (LRU): {evicted_id}")


# ── Pydantic v2 compatible pre-retrieved retriever ───────────────────────────

class PreRetrievedRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    docs: List[Document]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self.docs


# ── Rerank threshold ──────────────────────────────────────────────────────────
# If the top reranked chunk scores below this, nothing relevant was found.
# Returning "not available" is safer than hallucinating an answer.
RERANK_RELEVANCE_THRESHOLD = -10.0  # base threshold (single-doc sessions)


class SessionAwareRAGAgent:
    """
    Production RAG with BM25+FAISS ensemble retrieval, cross-encoder reranking,
    source-type tagging, dynamic vocabulary building, and anti-hallucination guards.
    """

    def __init__(
        self,
        base_vector_store_path: str = settings.VECTOR_STORE_PATH,
        db_manager: DatabaseManager = None,
        sql_executor=None,
    ):
        self.base_vector_store_path = Path(base_vector_store_path)
        self.base_vector_store_path.mkdir(parents=True, exist_ok=True)

        self.parser            = DocumentParser()
        self.language_detector = LanguageDetector()

        if ENABLE_HYBRID_CSV and db_manager:
            self.csv_processor = HybridCSVProcessor(db_manager, sql_executor=sql_executor)
            logger.info("✅  Hybrid CSV processing enabled")
        else:
            self.csv_processor = None

        # Larger chunks for structured data, smaller for prose
        self.document_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.structured_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000, chunk_overlap=50,
            separators=["\n\n", "\n"],
        )

        logger.info("📄  Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )

        logger.info("🎯  Loading cross-encoder for reranking...")
        self.cross_encoder = CrossEncoder(settings.RERANKER_MODEL)

        self.INITIAL_RETRIEVAL_K = 10
        self.RERANK_TOP_K        = 3
        self.BM25_WEIGHT         = 0.5
        self.FAISS_WEIGHT        = 0.5

        logger.info(
            f"📊  Hybrid config: BM25({self.BM25_WEIGHT}) + "
            f"FAISS({self.FAISS_WEIGHT}) → rerank@{self.RERANK_TOP_K}"
        )

        self.llm = make_llm(temperature=0.1)

        self.session_stores: Dict[str, Dict[str, Any]] = _LRUSessionCache(max_size=50)
        logger.info("🤖  RAG Agent initialized")

    # ── Session management ────────────────────────────────────────────────────

    def _get_session_path(self, conversation_id: str) -> Path:
        return self.base_vector_store_path / (conversation_id or "default_session")

    def _is_structured_file(self, filename: str) -> bool:
        return filename.lower().endswith(('.csv', '.xlsx'))

    def _create_empty_session(self) -> Dict[str, Any]:
        return {
            "vectorstore":     None,
            "documents":       [],
            "loaded_files":    [],
            "csv_tables":      [],
            "total_chunks":    0,
            "total_documents": 0,
            "source_vocab":    None,   # populated by _build_source_vocabulary()
        }

    def get_or_create_session(self, conversation_id: str) -> Dict[str, Any]:
        if conversation_id not in self.session_stores:
            session_path = self._get_session_path(conversation_id)

            if session_path.exists() and (session_path / "index.faiss").exists():
                try:
                    logger.info(f"📂  Loading session: {conversation_id}")
                    vectorstore = FAISS.load_local(
                        str(session_path), self.embeddings,
                        allow_dangerous_deserialization=True,
                    )
                    session = self._create_empty_session()
                    session["vectorstore"]  = vectorstore
                    session["total_chunks"] = vectorstore.index.ntotal
                    self._restore_session_state(session, conversation_id)
                    self.session_stores[conversation_id] = session
                    logger.info(f"✅  Loaded: {vectorstore.index.ntotal} vectors")
                except Exception as e:
                    logger.warning(f"⚠️  Load error: {e} — creating fresh session")
                    self.session_stores[conversation_id] = self._create_empty_session()
            else:
                logger.info(f"🆕  New session: {conversation_id}")
                self.session_stores[conversation_id] = self._create_empty_session()

        return self.session_stores[conversation_id]

    def _restore_session_state(self, session: Dict[str, Any], conversation_id: str) -> None:
        """
        Rebuild the in-memory parts of a session that was reloaded from disk
        (after a server restart or an LRU eviction).

        Only the FAISS index is persisted, but BM25 needs the raw chunks, the
        UI needs the file list, and the SQL route needs the session's CSV
        tables. The chunks (with their metadata) live in the FAISS docstore,
        and the tables are still in SQLite, so everything can be recovered.
        """
        docs = list(session["vectorstore"].docstore._dict.values())
        session["documents"] = docs
        session["loaded_files"] = list(dict.fromkeys(
            d.metadata.get("source") for d in docs if d.metadata.get("source")
        ))
        session["total_documents"] = len(session["loaded_files"])

        if self.csv_processor:
            tables = self.csv_processor.get_uploaded_tables(conversation_id)
            session["csv_tables"] = tables
            # Rebuild the value-normalisation index used by the SQL executor
            executor = self.csv_processor.sql_executor
            if executor is not None:
                for table in tables:
                    try:
                        executor.index_table(
                            self.csv_processor.db.execute_query(f'SELECT * FROM "{table}"'), table
                        )
                    except Exception as e:
                        logger.warning(f"⚠️  Could not re-index {table}: {e}")

        if docs:
            self._build_source_vocabulary(session)
        logger.info(
            f"♻️  Restored session: {len(docs)} chunks, "
            f"{len(session['loaded_files'])} files, {len(session['csv_tables'])} tables"
        )

    # ── Document loading ──────────────────────────────────────────────────────

    def load_documents(self, conversation_id: str, file_paths: List[str]) -> Dict[str, Any]:
        log_section(f"LOADING DOCUMENTS: {conversation_id}")
        session = self.get_or_create_session(conversation_id)
        regular_docs, structured_docs = [], []
        loaded_files_info = []

        for file_path in file_paths:
            if not Path(file_path).exists():
                logger.error(f"❌  Not found: {file_path}")
                continue

            filename = Path(file_path).name
            ext      = Path(file_path).suffix.lower()

            try:
                if self._is_structured_file(filename) and self.csv_processor:
                    logger.info(f"📊  HYBRID ingest: {filename}")
                    result = self.csv_processor.process_file(file_path, conversation_id)

                    # Read column names so they can be pinned in vocab later
                    col_names = result.get("metadata", {}).get("columns", [])

                    for doc in result['rag_documents']:
                        doc.metadata["source_type"] = "csv"
                        doc.metadata["source"]      = filename
                        doc.metadata["columns"]     = col_names

                    structured_docs.extend(result['rag_documents'])
                    session["loaded_files"].append(filename)
                    session["csv_tables"].append(result['sql_table'])
                    loaded_files_info.append({
                        "filename":   filename,
                        "format":     ext,
                        "type":       "hybrid",
                        "sql_table":  result['sql_table'],
                        "rag_chunks": len(result['rag_documents']),
                    })
                    logger.info(f"✅  HYBRID: {len(result['rag_documents'])} chunks")

                else:
                    logger.info(f"📄  DOCUMENT ingest: {filename}")
                    if ext == '.pdf':
                        docs = self.parser.parse_pdf(file_path)
                    elif ext == '.docx':
                        docs = self.parser.parse_docx(file_path)
                    elif ext == '.txt':
                        docs = self.parser.parse_txt(file_path)
                    else:
                        logger.warning(f"⚠️  Unsupported format: {ext}")
                        continue

                    for doc in docs:
                        doc.metadata["source_type"] = "document"
                        doc.metadata["source"]      = filename

                    regular_docs.extend(docs)
                    session["loaded_files"].append(filename)
                    loaded_files_info.append({
                        "filename": filename,
                        "format":   ext,
                        "type":     "document",
                        "chunks":   len(docs),
                    })
                    logger.info(f"✅  Parsed: {len(docs)} chunks")

            except Exception as e:
                logger.error(f"❌  Error processing {filename}: {e}")

        if not regular_docs and not structured_docs:
            return {"success": False, "message": "No documents loaded"}

        logger.info("✂️  Splitting chunks...")
        split_docs = []

        if regular_docs:
            chunks = self.document_splitter.split_documents(regular_docs)
            for chunk in chunks:
                chunk.metadata.setdefault("source_type", "document")
            split_docs.extend(chunks)
            logger.info(f"📄  Document chunks: {len(chunks)}")

        if structured_docs:
            chunks = self.structured_splitter.split_documents(structured_docs)
            for chunk in chunks:
                chunk.metadata.setdefault("source_type", "csv")
            split_docs.extend(chunks)
            logger.info(f"📊  Structured chunks: {len(chunks)}")

        MAX_CHUNKS = 5000
        if len(split_docs) > MAX_CHUNKS:
            logger.warning(f"⚠️  Capping at {MAX_CHUNKS} chunks (was {len(split_docs)})")
            split_docs = split_docs[:MAX_CHUNKS]

        logger.info(f"✅  Total chunks: {len(split_docs)}")
        session["documents"].extend(split_docs)

        # Build / extend vector store in batches
        batch_size = 500
        logger.info("🔢  Building vector store...")
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i + batch_size]
            if session["vectorstore"] is None:
                session["vectorstore"] = FAISS.from_documents(batch, self.embeddings)
            else:
                session["vectorstore"].add_documents(batch)
            logger.info(f"   ✅  Batch {i // batch_size + 1}")

        total_vectors              = session["vectorstore"].index.ntotal
        session["total_chunks"]    = total_vectors
        session["total_documents"] = len(session["loaded_files"])

        session_path = self._get_session_path(conversation_id)
        session_path.mkdir(parents=True, exist_ok=True)
        session["vectorstore"].save_local(str(session_path))
        logger.info(f"💾  Saved: {total_vectors} vectors")

        # Build per-source vocabulary for orchestrator routing
        self._build_source_vocabulary(session)

        return {
            "success":         True,
            "conversation_id": conversation_id,
            "files_loaded":    loaded_files_info,
            "total_chunks":    len(split_docs),
            "total_vectors":   total_vectors,
        }

    # ── Dynamic vocabulary builder ────────────────────────────────────────────

    def _build_source_vocabulary(self, session: dict) -> None:
        """
        Build per-source unique vocabularies from chunk content.

        Words that appear in BOTH sources are considered ambiguous and excluded.
        CSV column names are pinned directly into csv vocab so queries that
        mention column names (e.g. "AttendancePct by department") always route
        to SQL even if those terms also appear contextually in document text.

        Result stored in session["source_vocab"]:
          "csv"      → terms exclusive to CSV chunks
          "document" → terms exclusive to document chunks
          "overlap"  → ambiguous terms (for debugging)
        """
        STOPWORDS = {
            'the', 'a', 'an', 'is', 'in', 'of', 'to', 'and', 'or', 'for', 'with',
            'from', 'at', 'by', 'on', 'are', 'was', 'be', 'this', 'that', 'it',
            'as', 'we', 'our', 'has', 'have', 'had', 'not', 'but', 'if', 'so',
            'its', 'can', 'all', 'any', 'one', 'two', 'per', 'will', 'been',
            'more', 'also', 'than', 'when', 'what', 'how', 'which', 'who',
            'they', 'them', 'their', 'there', 'here', 'each', 'both', 'some',
            'into', 'over', 'after', 'about', 'up', 'out', 'do', 'did',
            'were', 'would', 'could', 'should', 'may', 'might', 'shall',
            'row', 'file', 'column', 'data', 'value', 'table', 'page',
            'source', 'type', 'total', 'number', 'name', 'date', 'time',
        }

        csv_vocab = Counter()
        doc_vocab = Counter()

        for doc in session.get("documents", []):
            source_type = doc.metadata.get("source_type", "")
            words = re.findall(r'\b[a-zA-Z]{3,}\b', doc.page_content.lower())
            filtered = [w for w in words if w not in STOPWORDS]
            if source_type == "csv":
                csv_vocab.update(filtered)
            elif source_type == "document":
                doc_vocab.update(filtered)

        csv_terms = set(csv_vocab.keys())
        doc_terms = set(doc_vocab.keys())

        # Pin column name tokens directly into csv_terms.
        # This ensures queries referencing column names always route to SQL
        # even when the same words appear contextually in document text.
        pinned = set()
        for doc in session.get("documents", []):
            if doc.metadata.get("source_type") == "csv":
                for col in doc.metadata.get("columns", []):
                    tokens = re.findall(r'[a-zA-Z]{3,}', col.lower())
                    csv_terms.update(tokens)
                    pinned.update(tokens)

        # Compute overlap but exclude pinned tokens (CSV always wins)
        overlap    = (csv_terms & doc_terms) - pinned
        unique_csv = csv_terms - overlap
        unique_doc = doc_terms - overlap

        session["source_vocab"] = {
            "csv":      unique_csv,
            "document": unique_doc,
            "overlap":  overlap,
        }

        logger.info(
            f"📚  Vocab built — CSV unique: {len(unique_csv)}, "
            f"DOC unique: {len(unique_doc)}, "
            f"Overlap (ambiguous): {len(overlap)}"
        )

    # ── Hybrid retrieval ──────────────────────────────────────────────────────

    def _create_hybrid_retriever(self, session: Dict[str, Any]):
        faiss_only = session["vectorstore"].as_retriever(
            search_kwargs={"k": self.INITIAL_RETRIEVAL_K}
        )
        # BM25 cannot be built from an empty corpus; fall back to FAISS alone
        if not session["documents"]:
            return faiss_only
        bm25   = BM25Retriever.from_documents(session["documents"])
        bm25.k = self.INITIAL_RETRIEVAL_K
        faiss  = session["vectorstore"].as_retriever(
            search_kwargs={"k": self.INITIAL_RETRIEVAL_K}
        )
        return EnsembleRetriever(
            retrievers=[bm25, faiss],
            weights=[self.BM25_WEIGHT, self.FAISS_WEIGHT],
        )

    def _expand_query(self, question: str) -> str:
        """
        Expand a low-scoring query with synonyms and document-style phrasing.
        Only triggered when the top rerank score falls below EXPANSION_TRIGGER.
        Combined with the original question to preserve intent.
        """
        try:
            prompt = (
                "A user asked this question but the document uses different terminology.\n"
                "Generate 6-10 keywords/phrases covering:\n"
                "1. Direct synonyms of the question terms\n"
                "2. How a formal business document would phrase this topic\n"
                "3. Related section headers a document might use\n"
                "Output ONLY the keywords, comma-separated, no explanation.\n\n"
                f"Question: {question}\n\nKeywords:"
            )
            expanded = self.llm.invoke(prompt).content.strip()
            logger.info(f"🔄  Query expanded: {expanded}")
            return f"{question} {expanded}"
        except Exception as e:
            logger.warning(f"⚠️  Query expansion failed: {e}")
            return question

    def _rerank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        pairs  = [[query, doc.page_content] for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        reranked = []
        for doc, score in ranked[:self.RERANK_TOP_K]:
            doc.metadata['rerank_score'] = float(score)
            reranked.append(doc)

        score_str = [f'{d.metadata["rerank_score"]:.3f}' for d in reranked]
        logger.info(f"🎯  Reranked scores: {score_str}")
        return reranked

    # ── Main query pipeline ───────────────────────────────────────────────────

    def query(
        self,
        conversation_id: str,
        question: str,
        language: str = None,
    ) -> Dict[str, Any]:
        """
        Three-stage retrieval pipeline with anti-hallucination guards:
          1. Hybrid (BM25 + FAISS) → 10 candidates
          2. Cross-encoder rerank  → top 3
          3. Dual-threshold guard  → return "not available" if irrelevant
          4. LLM generation with top-3 context only
        """
        session = self.get_or_create_session(conversation_id)

        if not session["vectorstore"]:
            return {
                "success": False,
                "error":   "No documents loaded",
                "answer":  "Please upload documents first.",
            }

        log_section(f"QUERYING: {conversation_id}")
        logger.info(f"❓  {question}")

        try:
            if not language:
                language = self.language_detector.detect_language(question)

            has_csv = self.has_csv_data(conversation_id)

            # Stage 1: hybrid retrieval
            logger.info("🔍  Stage 1: BM25 + FAISS ensemble")
            retriever    = self._create_hybrid_retriever(session)
            initial_docs = retriever.invoke(question)

            # Deduplicate by content prefix
            seen, unique_docs = set(), []
            for doc in initial_docs:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(doc)
            initial_docs = unique_docs[:self.INITIAL_RETRIEVAL_K]
            logger.info(f"   {len(initial_docs)} unique candidates")

            # Stage 2: cross-encoder rerank
            logger.info("🎯  Stage 2: Cross-encoder reranking")
            reranked = self._rerank_documents(question, initial_docs)

            # Lazy query expansion — only when top score is poor.
            # Handles inferential queries that don't match document phrasing.
            EXPANSION_TRIGGER = -8.0
            if reranked and reranked[0].metadata.get("rerank_score", 0) < EXPANSION_TRIGGER:
                logger.info(
                    f"🔄  Low rerank score "
                    f"({reranked[0].metadata['rerank_score']:.3f}) — expanding query"
                )
                expanded_q    = self._expand_query(question)
                expanded_docs = retriever.invoke(expanded_q)

                seen_keys = {doc.page_content[:100] for doc in initial_docs}
                for doc in expanded_docs:
                    key = doc.page_content[:100]
                    if key not in seen_keys:
                        seen_keys.add(key)
                        initial_docs.append(doc)

                reranked = self._rerank_documents(question, initial_docs)
                logger.info("✅  Re-ranked after expansion")

            # Dual-threshold hallucination guard.
            # Both top and average must be below threshold to block —
            # prevents false negatives while blocking truly irrelevant queries.
            if reranked:
                all_scores = [d.metadata.get("rerank_score", 0.0) for d in reranked]
                top_score  = all_scores[0]
                avg_score  = sum(all_scores) / len(all_scores)

                # Dynamic threshold: relax slightly for multi-doc sessions.
                # More docs = noisier vector space = scores naturally lower.
                session_info   = self.get_session_status(conversation_id)
                doc_count      = session_info.get("total_documents", 1)
                # Relax by 0.5 per extra doc beyond the first, max relax = -2.0
                relax          = min(2.0, max(0.0, (doc_count - 1) * 0.5))
                effective_threshold = RERANK_RELEVANCE_THRESHOLD - relax

                logger.info(
                    f"📊  Rerank — top: {top_score:.3f}, "
                    f"avg: {avg_score:.3f}, threshold: {effective_threshold:.1f} "
                    f"(docs: {doc_count}, relax: -{relax:.1f})"
                )

                if top_score < effective_threshold and avg_score < effective_threshold:
                    logger.info("🚫  Both scores below threshold — not available")
                    return {
                        "success":          True,
                        "answer":           "This information is not available in the uploaded documents.",
                        "sources":          [],
                        "language":         language,
                        "source_count":     0,
                        "conversation_id":  conversation_id,
                        "retrieval_method": "hybrid_bm25_faiss_reranking",
                        "not_found":        True,
                    }

            # Stage 3: generate answer from top-3 context
            answer  = self._generate_answer(question, reranked, has_csv, language)

            # The LLM itself may refuse (guardrail prompt). Don't attach
            # "sources" to a non-answer: that would look like a citation.
            if "not available in the uploaded" in answer.lower():
                return {
                    "success":          True,
                    "answer":           answer,
                    "sources":          [],
                    "language":         language,
                    "source_count":     0,
                    "conversation_id":  conversation_id,
                    "retrieval_method": "hybrid_bm25_faiss_reranking",
                    "not_found":        True,
                }

            sources = []
            for doc in reranked:
                src = {
                    "source":          doc.metadata.get("source", "unknown"),
                    "type":            doc.metadata.get("source_type", "unknown"),
                    "preview":         doc.page_content[:150] + "...",
                    "relevance_score": doc.metadata.get("rerank_score", 0.0),
                }
                if "page" in doc.metadata:
                    src["page"] = doc.metadata["page"]
                sources.append(src)

            source_files = list({s["source"] for s in sources})
            if source_files and "Source" not in answer:
                suffix = (
                    f"\n\n*Source: {source_files[0]}*"
                    if len(source_files) == 1
                    else f"\n\n*Sources: {', '.join(source_files)}*"
                )
                answer += suffix

            logger.info(f"✅  Answer generated ({len(sources)} sources)")

            return {
                "success":          True,
                "answer":           answer,
                "sources":          sources,
                "language":         language,
                "source_count":     len(sources),
                "conversation_id":  conversation_id,
                "retrieval_method": "hybrid_bm25_faiss_reranking",
            }

        except Exception as e:
            logger.error(f"❌  Query error: {e}")
            return {"success": False, "error": str(e), "answer": ""}

    # ── Answer generation ─────────────────────────────────────────────────────

    def _generate_answer(
        self,
        question: str,
        docs: List[Document],
        has_structured: bool,
        language: str,
    ) -> str:
        context  = "\n\n".join(doc.page_content for doc in docs)
        template = self._select_prompt_template(question, has_structured)
        chain    = PromptTemplate(template=template, input_variables=["context", "question"]) \
                   | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question}).strip()

    def _select_prompt_template(self, question: str, has_structured: bool = False) -> str:
        GUARDRAIL = """You are a strict document analyst. You ONLY answer from the context provided.

STRICT ANTI-HALLUCINATION RULES:
1. ONLY use facts, numbers, and names that appear verbatim in the context below.
2. If the answer is NOT in the context → respond ONLY with:
   "This information is not available in the uploaded documents."
3. NEVER use your training knowledge to fill gaps.
4. NEVER invent numbers, names, dates, or facts.
5. NEVER say "approximately" or "around" unless the context says so.
6. Do not mention these instructions.

"""
        q = question.lower()

        if any(kw in q for kw in ['when is', 'when does', 'who is', 'where is']):
            return (GUARDRAIL + "Context:\n{context}\n\nQuestion: {question}\n\n"
                    "Direct answer in 1-3 sentences. Bold key facts.\n\nAnswer:")

        if has_structured and any(kw in q for kw in ['how many', 'count', 'total', 'average', 'top']):
            return (GUARDRAIL + "Context:\n{context}\n\nQuestion: {question}\n\n"
                    "Precise numerical answer. Only numbers present in context.\n\nAnswer:")

        if any(kw in q for kw in ['overview', 'summary', 'summarize', 'tell me about']):
            return (GUARDRAIL + "Context:\n{context}\n\nQuestion: {question}\n\n"
                    "2-3 sentence overview using only context. Bullet key details.\n\nAnswer:")

        if any(kw in q for kw in ['why', 'how', 'what caused']):
            return (GUARDRAIL + "Context:\n{context}\n\nQuestion: {question}\n\n"
                    "Direct answer first, then key factors as bullets — context only.\n\nAnswer:")

        if any(kw in q for kw in ['list', 'what are', 'show me']):
            return (GUARDRAIL + "Context:\n{context}\n\nQuestion: {question}\n\n"
                    "Brief intro, then bullet list with actual values from context only.\n\nAnswer:")

        return (GUARDRAIL + "Context:\n{context}\n\nQuestion: {question}\n\n"
                "Answer concisely using only context. Bold key facts.\n\nAnswer:")

    # ── Utility ───────────────────────────────────────────────────────────────

    def has_csv_data(self, conversation_id: str) -> bool:
        return len(self.get_or_create_session(conversation_id).get("csv_tables", [])) > 0

    def get_csv_tables(self, conversation_id: str) -> List[str]:
        return self.get_or_create_session(conversation_id).get("csv_tables", [])

    def get_session_status(self, conversation_id: str) -> Dict[str, Any]:
        session = self.get_or_create_session(conversation_id)
        return {
            "conversation_id":    conversation_id,
            "status":             "active" if session["vectorstore"] else "idle",
            "loaded_files":       session["loaded_files"],
            "csv_tables":         session.get("csv_tables", []),
            "has_csv_data":       len(session.get("csv_tables", [])) > 0,
            "total_documents":    session.get("total_documents", len(session["loaded_files"])),
            "total_vectors":      session["total_chunks"],
            "vectorstore_ready":  session["vectorstore"] is not None,
            "source_vocab_ready": session.get("source_vocab") is not None,
            "retrieval_config": {
                "method":           "hybrid_bm25_faiss_reranking",
                "initial_k":        self.INITIAL_RETRIEVAL_K,
                "rerank_k":         self.RERANK_TOP_K,
                "bm25_weight":      self.BM25_WEIGHT,
                "faiss_weight":     self.FAISS_WEIGHT,
                "rerank_threshold": RERANK_RELEVANCE_THRESHOLD,
            },
        }

    def delete_session(self, conversation_id: str):
        try:
            self.session_stores.pop(conversation_id, None)
            path = self._get_session_path(conversation_id)
            if path.exists():
                shutil.rmtree(path)
                logger.info(f"🗑️  Deleted session: {conversation_id}")
        except Exception as e:
            logger.error(f"❌  Delete error: {e}")

    def get_global_stats(self) -> Dict[str, Any]:
        total   = len(self.session_stores)
        active  = sum(1 for s in self.session_stores.values() if s["vectorstore"])
        vectors = sum(s["total_chunks"] for s in self.session_stores.values())
        return {
            "total_sessions":      total,
            "active_sessions":     active,
            "total_vectors":       vectors,
            "base_path":           str(self.base_vector_store_path),
            "retrieval_method":    "hybrid_bm25_faiss_reranking",
            "initial_retrieval_k": self.INITIAL_RETRIEVAL_K,
            "rerank_k":            self.RERANK_TOP_K,
        }