"""
evaluation/eval_suite.py — End-to-end evaluation suite for the AI Business Analyst.

Runs fully in-process (imports the real orchestrator/agents directly, no HTTP
server or manual conversation setup required) and non-interactively (no
input() prompts — safe for CI or `python evaluation/eval_suite.py`).

Four evals, each against real data and real LLM calls (Groq):

  1. Routing accuracy   — does the orchestrator classify a question as the
                           correct engine (SQL / DOCUMENT / HYBRID / FORECAST
                           / GENERAL)? Labeled set, scored as accuracy +
                           a confusion matrix.

  2. SQL correctness    — does the generated SQL, executed against the
                           project's own sample_data.py business database,
                           return the numerically correct answer? This is
                           exact-match against ground truth computed
                           independently in Python (not via the LLM), the
                           gap flagged as missing in the repo's prior eval
                           scripts (bechmark.py only does keyword matching).

  3. RAG faithfulness   — real .docx files from sample_data/ are ingested
     & refusal            through the actual hybrid RAG pipeline, then:
                             (a) answerable questions are checked against
                                 facts independently extracted from the
                                 source document (faithfulness), and
                             (b) out-of-scope questions are checked for a
                                 correct refusal instead of a hallucinated
                                 answer (the dual-threshold guard).

  4. Language detection — does tools/language_detector.py correctly detect
                           the language of non-English questions.

Usage:
    python evaluation/eval_suite.py
    python evaluation/eval_suite.py --skip-rag     # faster, no doc ingestion
    python evaluation/eval_suite.py --json-only    # suppress console report

Output:
    evaluation/eval_results.json   — machine-readable results for every case
    evaluation/eval_report.txt     — human-readable summary report
"""

import argparse
import asyncio
import json
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import DatabaseManager
from database.sample_data import create_sample_database
from agents.orchestrator import OrchestratorAgent
from tools.language_detector import LanguageDetector
from utils.logger import logger


# ══════════════════════════════════════════════════════════════════════════
#  1. ROUTING ACCURACY — labeled set, ground truth is the intended engine
# ══════════════════════════════════════════════════════════════════════════

ROUTING_CASES = [
    # (question, expected_routing, requires_csv, requires_docs)
    ("How many employees are in the database?", "sql", True, False),
    ("What is the average salary of engineers?", "sql", True, False),
    ("Count products in the Electronics category", "sql", True, False),
    ("Show total revenue by month", "sql", True, False),
    ("Who is the top sales rep by total_amount?", "sql", True, False),

    ("Summarize the strategic planning document", "document", False, True),
    ("What are the action items from the meeting?", "document", False, True),
    ("What budget was approved for AI integration?", "document", False, True),
    ("What does the performance review say about the sales department?", "document", False, True),

    ("Explain the sales trend and show the underlying numbers", "hybrid", True, True),
    ("Why did engineering perform well, and what were the department's total sales?", "hybrid", True, True),

    ("Forecast sales for the next 3 months", "forecast", True, False),
    ("Predict revenue trend going forward", "forecast", True, False),

    ("Hello, how are you?", "general", False, False),
    ("What is machine learning?", "general", False, False),
]

# Orchestrator emits "document"/"rag" interchangeably depending on code path —
# normalize both to "document" for scoring so a correct classification isn't
# marked wrong over a label spelling difference.
ROUTING_ALIASES = {"rag": "document"}


def normalize_routing(label: Optional[str]) -> str:
    label = (label or "").lower().strip()
    return ROUTING_ALIASES.get(label, label)


# ══════════════════════════════════════════════════════════════════════════
#  2. SQL CORRECTNESS — exact-match against ground truth computed in Python
#     from database/sample_data.py's fixed, known rows (not from the LLM).
# ══════════════════════════════════════════════════════════════════════════

# Questions answerable from the employees export (10 rows, all of
# sample_data.py's employees — see _employees_csv_path).
SQL_CASES_EMPLOYEES = [
    # (question, expected_value, tolerance) — tolerance 0 means exact match,
    # else relative tolerance for float comparisons.
    ("How many employees are in the database?", 10, 0),
    ("How many employees are in the Engineering department?", 4, 0),
    ("What is the average salary of employees in Engineering?", (120000 + 110000 + 95000 + 92000) / 4, 0.01),
    ("How many departments are represented among the employees?", 5, 0),
    ("What is the highest salary in the company?", 120000, 0),
]

# Questions answerable from the flattened sales export (8 rows — see
# _sample_csv_path). Ground truth computed independently from
# sample_data.py's fixed sales_data rows.
SQL_CASES_SALES = [
    ("What is the total quantity sold across all sales?", 10 + 50 + 25 + 5 + 30 + 8 + 20 + 12, 0),
    ("What is the total revenue across all sales?", 12999.90 + 1499.50 + 3249.75 + 6499.95 + 1499.70 + 3199.92 + 2599.80 + 15599.88, 0.001),
    ("How many sales transactions are there in total?", 8, 0),
]


def _sql_ground_truth_note(question: str) -> str:
    """Human-readable note on how each ground-truth value was derived, for the report."""
    return "computed directly from database/sample_data.py's fixed rows, independent of the LLM"


# ══════════════════════════════════════════════════════════════════════════
#  3. RAG FAITHFULNESS + REFUSAL
# ══════════════════════════════════════════════════════════════════════════

RAG_DOCS = [
    PROJECT_ROOT / "sample_data" / "Strategic_Planning_Jan2025.docx",
    PROJECT_ROOT / "sample_data" / "Performance_Review_Q4.docx",
]

# Answerable questions: (question, list of required substrings — ANY match on
# each inner list counts as that fact being present; ALL facts must be found
# for the case to pass). Facts were extracted by reading the source .docx
# directly (see conversation record), not guessed.
RAG_FAITHFUL_CASES = [
    (
        "What budget was approved for AI integration in the strategic plan?",
        [["500", "$500"]],
    ),
    (
        "By when should Alice finalize the AI architecture?",
        [["jan 20", "january 20"]],
    ),
    (
        "What is the target Monthly Recurring Revenue by June?",
        [["750", "$750"]],
    ),
    (
        "What was the quarterly target achievement percentage for the sales department?",
        [["115"]],
    ),
    (
        "What was the average performance rating for the engineering department?",
        [["4.3"]],
    ),
]

# Out-of-scope questions: nothing in sample_data/ can answer these truthfully.
# A correct system should refuse / say it doesn't have this information,
# not fabricate an answer. This directly exercises the dual-threshold
# rerank-score refusal guard in agents/session_rag.py.
RAG_REFUSAL_CASES = [
    "What is the CEO's home phone number?",
    "What were Apple's Q4 2024 earnings per share?",
    "Who won the 2024 US presidential election?",
    "What is the exact current stock price of this company?",
    "What is my social security number?",
]

REFUSAL_SIGNALS = [
    "not available", "don't have", "do not have", "no information",
    "cannot find", "can't find", "not found", "not mentioned",
    "outside the scope", "not in the", "unable to answer",
    "i don't know", "no data", "not contain",
]


# ══════════════════════════════════════════════════════════════════════════
#  4. LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════

LANGUAGE_CASES = [
    ("How many employees work here?", "en"),
    ("¿Cuántos empleados trabajan aquí?", "es"),
    ("Combien d'employés travaillent ici?", "fr"),
    ("Wie viele Mitarbeiter arbeiten hier?", "de"),
    ("यहाँ कितने कर्मचारी काम करते हैं?", "hi"),
    ("这里有多少员工？", "zh-cn"),
    ("ここに何人の従業員がいますか？", "ja"),
    ("كم عدد الموظفين الذين يعملون هنا؟", "ar"),
    ("Quantos funcionários trabalham aqui?", "pt"),
]


# ══════════════════════════════════════════════════════════════════════════
#  Result bookkeeping
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CaseResult:
    question: str
    expected: Any
    actual: Any
    passed: bool
    latency_s: float
    detail: str = ""


@dataclass
class SuiteResult:
    name: str
    cases: list = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)

    @property
    def avg_latency(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.latency_s for c in self.cases) / len(self.cases)


# ══════════════════════════════════════════════════════════════════════════
#  Eval runner
# ══════════════════════════════════════════════════════════════════════════

class EvalSuite:
    def __init__(self, skip_rag: bool = False):
        self.skip_rag = skip_rag
        self.db_manager = DatabaseManager()
        self.orchestrator = OrchestratorAgent(self.db_manager)
        self.language_detector = LanguageDetector()

        # Isolated conversation IDs for this run, cleaned up at the end —
        # eval data must never pollute a real user's sessions.
        self.sql_conv_id = f"eval-sql-{uuid.uuid4().hex[:8]}"
        self.rag_conv_id = f"eval-rag-{uuid.uuid4().hex[:8]}"
        self.hybrid_conv_id = f"eval-hybrid-{uuid.uuid4().hex[:8]}"
        self._routing_extra_cids: list[str] = []  # populated by run_routing()
        self._upload_dirs_to_clean: list[Path] = []

    # ── Setup ──────────────────────────────────────────────────────────

    def _ensure_sample_db(self):
        # Always run it: DatabaseManager creates an empty business.db on
        # startup, so "file exists" does not mean the sample tables exist.
        # create_sample_database is idempotent (CREATE IF NOT EXISTS / INSERT OR IGNORE).
        db_path = PROJECT_ROOT / "data" / "business.db"
        create_sample_database(str(db_path))

    def _sample_csv_path(self) -> Path:
        """
        Export database/sample_data.py's tables as a single CSV so the eval
        exercises the SAME ingestion path (hybrid_csv_processor -> SQL import
        + RAG chunking) a real user's upload would go through, rather than
        querying business.db directly.
        """
        import pandas as pd
        import sqlite3

        db_path = PROJECT_ROOT / "data" / "business.db"
        conn = sqlite3.connect(str(db_path))
        # A single flattened sales-with-context table gives the SQL agent
        # everything it needs (employee/department/product/customer names)
        # in one uploadable CSV, mirroring how a real user would export a
        # business report rather than raw normalized tables.
        df = pd.read_sql_query(
            """
            SELECT s.sale_id, s.quantity, s.sale_date, s.total_amount,
                   e.first_name || ' ' || e.last_name AS employee_name,
                   d.department_name,
                   p.product_name, p.category,
                   c.company_name, c.country
            FROM sales s
            JOIN employees e ON s.employee_id = e.employee_id
            JOIN departments d ON e.department_id = d.department_id
            JOIN products p ON s.product_id = p.product_id
            JOIN customers c ON s.customer_id = c.customer_id
            """,
            conn,
        )
        conn.close()

        out_dir = PROJECT_ROOT / "evaluation" / "_tmp"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "eval_sales_export.csv"
        df.to_csv(out_path, index=False)
        return out_path

    def _employees_csv_path(self) -> Path:
        """
        A direct export of the raw `employees` table (joined to department
        name), separate from the flattened sales report above. SQL_CASES'
        headcount/salary questions have ground truth computed from ALL 10
        employees in sample_data.py — the flattened sales CSV only contains
        the 2 employees who happened to have a sale recorded, so it can't
        truthfully answer "how many employees are there" at all. Uploading
        the actual employees table (as a user reasonably would for HR-style
        questions) keeps the ground truth honest.
        """
        import pandas as pd
        import sqlite3

        db_path = PROJECT_ROOT / "data" / "business.db"
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query(
            """
            SELECT e.employee_id, e.first_name, e.last_name, e.job_title,
                   e.salary, e.hire_date, d.department_name
            FROM employees e
            JOIN departments d ON e.department_id = d.department_id
            """,
            conn,
        )
        conn.close()

        out_dir = PROJECT_ROOT / "evaluation" / "_tmp"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "eval_employees_export.csv"
        df.to_csv(out_path, index=False)
        return out_path

    async def _ingest_csv(self, conversation_id: str, csv_path: Path) -> list[str]:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.orchestrator.rag_agent.load_documents, conversation_id, [str(csv_path)]
        )
        return self.orchestrator.rag_agent.get_csv_tables(conversation_id)

    async def _ingest_docs(self, conversation_id: str, doc_paths: list[Path]):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self.orchestrator.rag_agent.load_documents, conversation_id,
            [str(p) for p in doc_paths],
        )

    # ── 1. Routing ─────────────────────────────────────────────────────

    async def run_routing(self, sample_csv: Path) -> SuiteResult:
        result = SuiteResult(name="Routing Accuracy")
        cid = f"eval-routing-{uuid.uuid4().hex[:8]}"

        # Give each category its own session scoped to exactly what that
        # category needs. Loading CSV+docs into one shared session for
        # every case would make every SQL-only question ambiguous by
        # construction (the orchestrator's vocabulary/source-disambiguation
        # logic correctly kicks in whenever both source types are present,
        # which is real, intended behavior — but it means a single shared
        # session can't fairly test "is a pure-SQL question routed as SQL"
        # in isolation). Hybrid cases get their own session with both.
        csv_cid = f"{cid}-csv"
        docs_cid = f"{cid}-docs"
        hybrid_cid = f"{cid}-hybrid"
        await self._ingest_csv(csv_cid, sample_csv)
        if not self.skip_rag:
            await self._ingest_docs(docs_cid, RAG_DOCS)
            await self._ingest_csv(hybrid_cid, sample_csv)
            await self._ingest_docs(hybrid_cid, RAG_DOCS)

        self._routing_extra_cids = [csv_cid, docs_cid, hybrid_cid]

        for question, expected, requires_csv, requires_docs in ROUTING_CASES:
            if requires_docs and self.skip_rag:
                continue
            case_cid = (
                hybrid_cid if (requires_csv and requires_docs)
                else docs_cid if requires_docs
                else csv_cid if requires_csv
                else csv_cid  # general/no-source questions — session choice is irrelevant
            )
            t0 = time.time()
            try:
                res = await self.orchestrator.route_query(question, case_cid)
                actual = normalize_routing(res.get("routing"))
                passed = actual == expected
                detail = f"routing_reason={res.get('routing_reason', '')}"
            except Exception as e:
                actual, passed, detail = "error", False, str(e)
            result.cases.append(CaseResult(question, expected, actual, passed, time.time() - t0, detail))

        return result

    # ── 2. SQL correctness ────────────────────────────────────────────

    async def run_sql_correctness(self, sample_csv: Path) -> SuiteResult:
        result = SuiteResult(name="SQL Correctness")

        # Two separate conversations, each with the ONE CSV that can
        # actually answer its question set — otherwise "how many employees
        # in total" asked against a sales-only export would be an unfair
        # test (that export only contains employees who made a sale).
        emp_cid = f"{self.sql_conv_id}-emp"
        sales_cid = self.sql_conv_id
        await self._ingest_csv(emp_cid, self._employees_csv_path())
        await self._ingest_csv(sales_cid, sample_csv)

        for cid, cases in [(emp_cid, SQL_CASES_EMPLOYEES), (sales_cid, SQL_CASES_SALES)]:
            for question, expected, tolerance in cases:
                t0 = time.time()
                try:
                    res = await self.orchestrator.route_query(question, cid)
                    actual = self._extract_numeric(res)
                    passed = self._numeric_match(actual, expected, tolerance)
                    detail = f"sql={res.get('sql_query', '')!r}"
                except Exception as e:
                    actual, passed, detail = None, False, str(e)
                result.cases.append(CaseResult(question, expected, actual, passed, time.time() - t0, detail))

        return result

    @staticmethod
    def _extract_numeric(route_result: dict) -> Optional[float]:
        """Pull the single numeric value out of a SQL query result, if there is one."""
        data = route_result.get("data") or []
        if not data:
            return None
        row = data[0]
        if not isinstance(row, dict):
            return None
        for v in row.values():
            if isinstance(v, (int, float)):
                return float(v)
            try:
                return float(str(v).replace(",", "").replace("$", ""))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _numeric_match(actual: Optional[float], expected: float, tolerance: float) -> bool:
        if actual is None:
            return False
        if tolerance == 0:
            return round(actual) == round(expected)
        return abs(actual - expected) <= abs(expected) * tolerance

    # ── 3. RAG faithfulness + refusal ─────────────────────────────────

    async def run_rag(self) -> tuple[SuiteResult, SuiteResult]:
        faithful = SuiteResult(name="RAG Faithfulness")
        refusal = SuiteResult(name="RAG Refusal (hallucination guard)")

        if self.skip_rag:
            return faithful, refusal

        cid = self.rag_conv_id
        await self._ingest_docs(cid, RAG_DOCS)

        for question, fact_groups in RAG_FAITHFUL_CASES:
            t0 = time.time()
            try:
                res = await self.orchestrator.route_query(question, cid)
                raw_text = (res.get("explanation") or res.get("answer") or "").lower()
                # Normalize all whitespace variants (regular space, non-
                # breaking space U+00A0, narrow no-break space U+202F — the
                # LLM's markdown formatting sometimes uses these between a
                # bolded label and its value) to a plain space before
                # substring matching, so formatting differences can't fail
                # an otherwise-correct, faithful answer.
                text = re.sub(r"[\s  ]+", " ", raw_text)
                found = [any(kw in text for kw in group) for group in fact_groups]
                passed = all(found)
                detail = f"missing_facts={[fact_groups[i] for i,f in enumerate(found) if not f]}" if not passed else "all facts present"
            except Exception as e:
                text, passed, detail = "", False, str(e)
            faithful.cases.append(CaseResult(question, fact_groups, text[:200], passed, time.time() - t0, detail))

        for question in RAG_REFUSAL_CASES:
            t0 = time.time()
            try:
                res = await self.orchestrator.route_query(question, cid)
                text = (res.get("explanation") or res.get("answer") or "").lower()
                refused = res.get("not_found") or any(sig in text for sig in REFUSAL_SIGNALS)
                passed = bool(refused)
                detail = "correctly refused" if passed else "answered without refusal — possible hallucination"
            except Exception as e:
                text, passed, detail = "", False, str(e)
            refusal.cases.append(CaseResult(question, "should refuse", text[:200], passed, time.time() - t0, detail))

        return faithful, refusal

    # ── 4. Language detection ─────────────────────────────────────────

    def run_language_detection(self) -> SuiteResult:
        result = SuiteResult(name="Language Detection")
        for text, expected in LANGUAGE_CASES:
            t0 = time.time()
            try:
                detected = self.language_detector.detect_language(text)
                passed = detected == expected
            except Exception as e:
                detected, passed = None, False
            result.cases.append(CaseResult(text, expected, detected, passed, time.time() - t0))
        return result

    # ── Cleanup ────────────────────────────────────────────────────────

    def cleanup(self):
        from parsers.hybrid_csv_processor import HybridCSVProcessor

        conv_ids = [
            self.sql_conv_id, f"{self.sql_conv_id}-emp",
            self.rag_conv_id, self.hybrid_conv_id,
            *self._routing_extra_cids,
        ]
        for cid in conv_ids:
            try:
                self.orchestrator.rag_agent.delete_session(cid)
            except Exception:
                pass
            # Also drop any SQL tables this eval run created in the shared
            # business.db — otherwise every run leaves orphaned
            # upload_eval-*_... tables behind, and get_schema() must
            # introspect ALL tables in the DB on every future query
            # (including other, unrelated sessions'), so leftover junk here
            # silently slows down and can even break every subsequent run.
            try:
                prefix = HybridCSVProcessor.session_table_prefix(cid)
                to_drop = [t for t in self.db_manager.list_tables() if t.startswith(prefix)]
                if to_drop:
                    conn = self.db_manager.get_connection()
                    cur = conn.cursor()
                    for t in to_drop:
                        cur.execute(f'DROP TABLE IF EXISTS "{t}"')
                    conn.commit()
                    conn.close()
            except Exception:
                pass
        tmp_dir = PROJECT_ROOT / "evaluation" / "_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        for d in self._upload_dirs_to_clean:
            shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════════

def build_confusion_matrix(routing_result: SuiteResult) -> dict:
    labels = sorted({c.expected for c in routing_result.cases} | {c.actual for c in routing_result.cases})
    matrix = {expected: {actual: 0 for actual in labels} for expected in labels}
    for c in routing_result.cases:
        matrix[c.expected][c.actual] += 1
    return matrix


def print_suite(result: SuiteResult):
    print(f"\n{'─' * 70}")
    print(f"  {result.name}  —  {result.accuracy:.0%} ({sum(c.passed for c in result.cases)}/{len(result.cases)})"
          f"  avg latency {result.avg_latency:.2f}s")
    print("─" * 70)
    for c in result.cases:
        mark = "✓" if c.passed else "✗"
        q_short = c.question if len(c.question) <= 60 else c.question[:57] + "..."
        print(f"  {mark} {q_short}")
        if not c.passed:
            print(f"      expected={c.expected!r}  actual={c.actual!r}  {c.detail}")


def write_report(suites: list[SuiteResult], confusion: dict, out_path: Path):
    lines = []
    lines.append("=" * 70)
    lines.append("AI BUSINESS ANALYST — EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    overall_total = sum(len(s.cases) for s in suites)
    overall_passed = sum(sum(c.passed for c in s.cases) for s in suites)
    lines.append(f"Overall: {overall_passed}/{overall_total} ({overall_passed/max(overall_total,1):.1%})")
    lines.append("")

    for s in suites:
        lines.append(f"{s.name}: {s.accuracy:.1%} ({sum(c.passed for c in s.cases)}/{len(s.cases)}) "
                      f"— avg latency {s.avg_latency:.2f}s")
    lines.append("")

    if confusion:
        lines.append("Routing confusion matrix (rows=expected, cols=actual):")
        labels = sorted(confusion.keys())
        header = "expected\\actual".ljust(16) + "".join(l.ljust(12) for l in labels)
        lines.append(header)
        for expected in labels:
            row = expected.ljust(16) + "".join(str(confusion[expected].get(actual, 0)).ljust(12) for actual in labels)
            lines.append(row)
        lines.append("")

    for s in suites:
        lines.append("-" * 70)
        lines.append(s.name)
        lines.append("-" * 70)
        for c in s.cases:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"[{mark}] {c.question}")
            if not c.passed:
                lines.append(f"       expected: {c.expected}")
                lines.append(f"       actual:   {c.actual}")
                if c.detail:
                    lines.append(f"       detail:   {c.detail}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(suites: list[SuiteResult], confusion: dict, out_path: Path):
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_accuracy": (
            sum(sum(c.passed for c in s.cases) for s in suites) / max(sum(len(s.cases) for s in suites), 1)
        ),
        "suites": [
            {
                "name": s.name,
                "accuracy": s.accuracy,
                "avg_latency_s": s.avg_latency,
                "cases": [
                    {
                        "question": c.question,
                        "expected": c.expected if not isinstance(c.expected, list) else str(c.expected),
                        "actual": c.actual,
                        "passed": c.passed,
                        "latency_s": round(c.latency_s, 3),
                        "detail": c.detail,
                    }
                    for c in s.cases
                ],
            }
            for s in suites
        ],
        "routing_confusion_matrix": confusion,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="AI Business Analyst evaluation suite")
    parser.add_argument("--skip-rag", action="store_true", help="Skip document ingestion + RAG evals (faster)")
    parser.add_argument("--json-only", action="store_true", help="Suppress console report, only write files")
    args = parser.parse_args()

    suite = EvalSuite(skip_rag=args.skip_rag)
    suite._ensure_sample_db()

    if not args.json_only:
        print("\n" + "=" * 70)
        print("  AI BUSINESS ANALYST — EVALUATION SUITE")
        print("=" * 70)

    try:
        sample_csv = suite._sample_csv_path()

        routing_result = await suite.run_routing(sample_csv)
        sql_result = await suite.run_sql_correctness(sample_csv)
        rag_faithful, rag_refusal = await suite.run_rag()
        lang_result = suite.run_language_detection()

        all_suites = [routing_result, sql_result, rag_faithful, rag_refusal, lang_result]
        all_suites = [s for s in all_suites if s.cases]  # drop empty (e.g. skipped RAG)

        confusion = build_confusion_matrix(routing_result)

        if not args.json_only:
            for s in all_suites:
                print_suite(s)

            overall_total = sum(len(s.cases) for s in all_suites)
            overall_passed = sum(sum(c.passed for c in s.cases) for s in all_suites)
            print(f"\n{'=' * 70}")
            print(f"  OVERALL: {overall_passed}/{overall_total} ({overall_passed/max(overall_total,1):.1%})")
            print("=" * 70)

        out_dir = PROJECT_ROOT / "evaluation"
        out_dir.mkdir(exist_ok=True)
        write_report(all_suites, confusion, out_dir / "eval_report.txt")
        write_json(all_suites, confusion, out_dir / "eval_results.json")

        if not args.json_only:
            print(f"\nReport written to: evaluation/eval_report.txt")
            print(f"JSON written to:   evaluation/eval_results.json\n")

    finally:
        suite.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
