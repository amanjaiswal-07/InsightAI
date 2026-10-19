# agents/sql_analyst.py
"""
SQL Analyst Agent

Converts natural language questions to SQL, executes them, and generates
business-friendly explanations.

Anti-hallucination measures:
  - Zero-row guard: returns "not available" immediately on empty results
  - Not-available sentinel guard: intercepts SQL placeholder rows before LLM
  - Short-query language detection guard: defaults to English for <4 words
  - Tightened explanation prompt with strict rules and few-shot examples
  - Injected actual column values into schema so LLM uses exact spelling
"""

import re
from typing import Dict, Any, List, Optional

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from database.db_manager import DatabaseManager
from tools.language_detector import LanguageDetector
from tools.sql_executor import SQLExecutor
from config import settings
from utils.llm import make_llm
from utils.logger import logger, log_section
from utils.plot_builder import PlotBuilder


# Phrases that indicate SQL returned a "not available" sentinel row.
# Matched before sending to LLM to prevent hallucinated rephrasing.
_NOT_AVAILABLE_PATTERNS = [
    "not available in the uploaded",
    "information is not available",
    "no data found",
    "no results",
]


class SQLAnalystAgent:

    MONTH_ORDER = {
        'january': 1,   'february': 2,  'march': 3,    'april': 4,
        'may': 5,        'june': 6,      'july': 7,     'august': 8,
        'september': 9,  'october': 10,  'november': 11, 'december': 12,
        'jan': 1,  'feb': 2,  'mar': 3,  'apr': 4,
        'jun': 6,  'jul': 7,  'aug': 8,  'sep': 9,
        'oct': 10, 'nov': 11, 'dec': 12,
        '01': 1, '02': 2, '03': 3, '04': 4,  '05': 5,  '06': 6,
        '07': 7, '08': 8, '09': 9, '10': 10, '11': 11, '12': 12,
        '1': 1,  '2': 2,  '3': 3,  '4': 4,  '5': 5,   '6': 6,
        '7': 7,  '8': 8,  '9': 9,
    }

    ENTITY_MAP = {
        'repo': 'repositories', 'repository': 'repositories',
        'customer': 'customers', 'employee': 'employees',
        'department': 'departments', 'sale': 'sales',
        'issue': 'issues', 'user': 'users', 'order': 'orders',
        'product': 'products', 'transaction': 'transactions',
        'patient': 'patients', 'visit': 'visits', 'record': 'records',
        'file': 'files', 'document': 'documents',
    }

    _AGGREGATION_EXACT = re.compile(
        r'\b(how many|count of|number of|sum of|overall total)\b',
        re.IGNORECASE,
    )

    def __init__(self, db_manager: DatabaseManager):
        self.db                = db_manager
        self.language_detector = LanguageDetector()
        self.sql_executor      = SQLExecutor(db_manager)
        self.plot_builder      = PlotBuilder()
        self._schema_cache: dict = {}   # cache key: "conv_id|table1,table2" → schema str

        self.llm = make_llm(temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS)

        sql_prompt = PromptTemplate(
            input_variables=["schema", "question", "allowed_tables"],
            template="""You are an expert SQLite query writer. Convert the question to a single valid SQL query.

DATABASE SCHEMA:
{schema}

ALLOWED TABLES (use ONLY these):
{allowed_tables}

STRICT SYNTAX RULES:
1. Aggregate functions MUST use parentheses:
   CORRECT → SELECT MAX(col), MIN(col), AVG(col), SUM(col), COUNT(col)
   WRONG   → SELECT MAX col, MIN col

2. To fetch the row WHERE a max/min occurs (no grouping), use a subquery:
   SELECT col1, col2 FROM table_name
   WHERE numeric_col = (SELECT MAX(numeric_col) FROM table_name)
   Use this ONLY when NOT grouping. Never combine with GROUP BY.

3. "highest/lowest AVERAGE per group" → GROUP BY + ORDER BY, never a subquery:
   CORRECT → SELECT group_col, AVG(val_col) as avg_val
             FROM table GROUP BY group_col ORDER BY avg_val DESC LIMIT 1

4. "highest/lowest TOTAL per group" → GROUP BY + SUM + ORDER BY:
   SELECT group_col, SUM(val_col) as total FROM table
   GROUP BY group_col ORDER BY total DESC LIMIT 1

5. Date/month extraction: STRFTIME('%Y-%m', date_col) AS month

6. Column names with special characters must be quoted: "egfr_mL_min_1.73m2"

7. "X as a number" means SELECT AVG(X) — never COUNT unless asked explicitly.

8. Return ONLY the SQL — no markdown, no explanation, no code fences.

9. Add LIMIT 50 for multi-row results. Never UNION between different upload tables.

10. EVERY SELECT must include a FROM clause using the exact table name from ALLOWED TABLES.
    Even single-value aggregates must have FROM the_table_name. No bare SELECTs without FROM.

11. Round decimal results: ROUND(AVG(col), 2), ROUND(SUM(col), 2)

12. For "who is X" questions: SELECT all meaningful columns (ID, Role, Department).

13. For "list all X" questions: SELECT identifier AND at least one descriptive column.

14. For correlation questions ("does X affect Y", "do high X have more Y"):
    Bucket the independent variable with CASE WHEN, then AVG the dependent variable per bucket.

15. If the question cannot be answered from the allowed tables, return exactly:
    SELECT 'This information is not available in the uploaded data' AS message

--- FEW-SHOT EXAMPLES ---

Schema: employees(EmployeeID TEXT, Department TEXT, Salary_LPA REAL, PerformanceScore REAL, Promoted TEXT, TenureYears REAL)

Q: Which department has the highest average salary?
SQL:
SELECT Department, ROUND(AVG(Salary_LPA), 2) AS avg_salary
FROM employees
GROUP BY Department
ORDER BY avg_salary DESC
LIMIT 1

Q: Who is the highest paid employee?
SQL:
SELECT EmployeeID, Department, Salary_LPA
FROM employees
WHERE Salary_LPA = (SELECT MAX(Salary_LPA) FROM employees)

Q: Do high performers get promoted more?
SQL:
SELECT
  CASE WHEN PerformanceScore >= 4.5 THEN 'High (4.5+)'
       WHEN PerformanceScore >= 3.5 THEN 'Mid (3.5-4.5)'
       ELSE 'Low (<3.5)' END AS performance_band,
  ROUND(100.0 * SUM(CASE WHEN Promoted = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS promotion_rate_pct,
  COUNT(*) AS employee_count
FROM employees
GROUP BY performance_band
ORDER BY MIN(PerformanceScore) DESC

Q: What is the attrition rate?
SQL:
SELECT 'This information is not available in the uploaded data' AS message

--- END EXAMPLES ---

QUESTION: {question}

SQL:""",
        )

        explanation_prompt = PromptTemplate(
            input_variables=["question", "sql", "results", "language", "language_name"],
            template="""You are a professional data analyst. Answer ONLY what was asked.

STRICT ANTI-HALLUCINATION RULES:
1. Answer the exact question — nothing more, nothing less.
2. ONLY use values present in the data. Never invent or estimate.
3. Single value result → one clean sentence.
4. Multiple rows → one-line summary, then bullets: "• Label: Value"
5. For "who is X" — always show: ID - Role (Department).
6. For list questions — show ID + Role or Name, not just IDs.
7. Round decimals to 2 places.
8. No markdown (**, ##). Plain text only. Under 100 words.
9. Do not mention SQL, databases, tables, or technical terms.
10. NEVER infer or guess from unrelated columns. The provided Data contains the exact answer.
11. For correlation results (age groups, performance bands) →
    lead with the trend, then list the buckets as bullets, end with a one-line conclusion.
12. NEVER output ASCII text charts, asterisks (*), progress bars, or pseudo-graphs in your text. A real graphical chart canvas is rendered by the user interface.

--- FEW-SHOT EXAMPLES ---

Q: What is the average salary?
Data: [{{"ROUND(AVG(Salary_LPA), 2)": 11.45}}]
Answer: The average salary across all employees is 11.45 LPA.

Q: Which department has the highest average salary?
Data: [{{"Department": "Technology & Engineering", "avg_salary": 17.0}}]
Answer: Technology & Engineering has the highest average salary at 17.0 LPA.

Q: Are older patients staying longer?
Data: [{{"age_group": "Under 40", "avg_stay_days": 3.5}}, {{"age_group": "40-59", "avg_stay_days": 7.2}}, {{"age_group": "60+", "avg_stay_days": 10.8}}]
Answer: Yes, older patients do stay longer on average.
• Under 40: 3.5 days
• 40-59: 7.2 days
• 60+: 10.8 days
Patients aged 60+ stay about 3x longer than those under 40.

Q: What is the attrition rate?
Data: [{{"message": "This information is not available in the uploaded data"}}]
Answer: This information is not available in the uploaded data.

--- END EXAMPLES ---

Question: {question}
Data: {results}

Answer in {language_name}:""",
        )

        sql_fix_prompt = PromptTemplate(
            input_variables=["schema", "allowed_tables", "question", "failed_sql", "error"],
            template="""You wrote a SQLite query that failed. Fix it.

DATABASE SCHEMA:
{schema}

ALLOWED TABLES (use ONLY these):
{allowed_tables}

QUESTION: {question}

FAILED SQL:
{failed_sql}

DATABASE ERROR:
{error}

Rules: a single read-only SELECT (or WITH ... SELECT) statement, exact column
names from the schema (quote names with special characters), every SELECT has
a FROM clause. If the question cannot be answered from these tables, return:
SELECT 'This information is not available in the uploaded data' AS message

Return ONLY the corrected SQL, no markdown, no explanation.

SQL:""",
        )

        parser = StrOutputParser()
        self.sql_chain         = sql_prompt         | self.llm | parser
        self.sql_fix_chain     = sql_fix_prompt     | self.llm | parser
        self.explanation_chain = explanation_prompt  | self.llm | parser

        logger.info("✅  SQL Analyst Agent ready")

    # ── SQL sanitisation ──────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_sql(sql: str) -> str:
        """Strip markdown fences, trailing semicolons, and fix bare aggregate functions."""
        sql = re.sub(r'^```(?:sql)?\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'\s*```$', '', sql)
        sql = sql.strip().rstrip(';')
        # Fix "SELECT MAX col" → "SELECT MAX(col)". Only in select-list position
        # (after SELECT or a comma) and never when the next word is a keyword,
        # so aliases/columns named "total" ("AS total FROM") are left alone.
        sql = re.sub(
            r'(\bSELECT\s+|,\s*)(MAX|MIN|AVG|SUM|COUNT|TOTAL)\s+'
            r'(?!\(|(?:FROM|AS|WHERE|GROUP|ORDER|LIMIT|HAVING|AND|OR|ON|JOIN)\b)'
            r'([a-zA-Z_][a-zA-Z0-9_\."]*)',
            r'\1\2(\3)', sql, flags=re.IGNORECASE,
        )
        return sql.strip()

    @staticmethod
    def _is_not_available_result(data_preview: List[dict]) -> bool:
        """
        Return True if SQL produced a "not available" sentinel row.
        Intercepting before the LLM prevents hallucinated rephrasing of the sentinel.
        """
        if not data_preview:
            return True
        for val in data_preview[0].values():
            if any(p in str(val).lower() for p in _NOT_AVAILABLE_PATTERNS):
                return True
        return False

    # ── Smart chronological sort ──────────────────────────────────────────────

    def _smart_sort_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Sort month-column results into calendar order."""
        if not result.get('success') or not result.get('data_preview'):
            return result
        try:
            df = pd.DataFrame(result['data_preview'])
            for col in df.columns:
                if 'month' not in col.lower() or df.empty:
                    continue
                sample = str(df[col].iloc[0]).lower().strip()
                if sample not in self.MONTH_ORDER:
                    continue
                df['_sort'] = df[col].str.lower().str.strip().map(self.MONTH_ORDER)
                df = df.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
                result['data_preview'] = df.to_dict('records')
                logger.info(f"  Sorted '{col}' chronologically")
                break
        except Exception as e:
            logger.warning(f"⚠️  Smart sort skipped: {e}")
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_simple_aggregation(self, question: str) -> bool:
        return bool(self._AGGREGATION_EXACT.search(question))

    @staticmethod
    def _single_numeric_value(result: Dict[str, Any]) -> Optional[float]:
        """Return the value if the result is exactly one row with one numeric cell."""
        rows = result.get('data_preview') or []
        if result.get('row_count') != 1 or len(rows) != 1 or len(rows[0]) != 1:
            return None
        value = next(iter(rows[0].values()))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value

    def _extract_entity(self, question: str, result: Dict[str, Any]) -> str:
        q = question.lower()
        for kw, entity in self.ENTITY_MAP.items():
            if kw in q:
                return entity
        first_row = result.get('data_preview', [{}])[0]
        if first_row:
            col = list(first_row.keys())[0]
            return re.sub(r'(COUNT|SUM|AVG|MAX|MIN)\s*\(|\)', '', col,
                          flags=re.IGNORECASE).replace('_', ' ').strip()
        return 'items'

    # ── Main analysis method ──────────────────────────────────────────────────

    def analyze(
        self,
        question: str,
        allowed_tables: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        log_section("ANALYZING QUESTION")
        logger.info(f"❓  Question: {question}")

        # Hard guard: no CSV uploaded
        if not allowed_tables:
            logger.warning("⚠️  SQL analyze called with no allowed_tables — aborting")
            return {
                'success':     True,
                'question':    question,
                'explanation': "Please upload a CSV or Excel file to run data queries.",
                'answer':      "Please upload a CSV or Excel file to run data queries.",
                'data':        [],
                'row_count':   0,
            }

        try:
            # Language detection guard: langdetect is unreliable on <4 Latin words.
            # Short queries like "total patients" default to English.
            # Exception: CJK/Arabic/Hindi text must always be detected by script
            # since they don't have word spaces so word count is always 1.
            words = question.strip().split()
            if len(words) < 4 and not self.language_detector.is_cjk_text(question):
                lang_code = 'en'
                lang_name = 'English'
                logger.info("🌍  Language: English (short query — skipped detect)")
            else:
                lang_code = self.language_detector.detect_language(question)
                lang_name = self.language_detector.get_language_name(lang_code)
                logger.info(f"🌍  Language: {lang_name} ({lang_code})")

            schema     = self._get_filtered_schema(allowed_tables, conversation_id)
            tables_str = ', '.join(allowed_tables)

            logger.info("🔧  Generating SQL...")
            raw_sql   = self.sql_chain.invoke({
                'schema': schema, 'question': question, 'allowed_tables': tables_str,
            })
            sql_query = self._sanitize_sql(raw_sql)
            logger.info(f"📝  SQL: {sql_query}")

            logger.info("⚙️   Executing...")
            result = self.sql_executor.execute_query(sql_query)

            # Self-correction: if the generated SQL fails (bad column name,
            # syntax, blocked statement), show the LLM its query and the
            # database error and let it try exactly once more.
            if not result['success']:
                logger.warning(f"🔁  SQL failed ({result['error']}) — retrying once with the error")
                fixed_sql = self._sanitize_sql(self.sql_fix_chain.invoke({
                    'schema':         schema,
                    'allowed_tables': tables_str,
                    'question':       question,
                    'failed_sql':     sql_query,
                    'error':          result['error'],
                }))
                logger.info(f"📝  Retry SQL: {fixed_sql}")
                retry = self.sql_executor.execute_query(fixed_sql)
                if retry['success']:
                    sql_query, result = fixed_sql, retry

            if not result['success']:
                logger.error(f"❌  SQL error: {result['error']}")
                return {
                    'success':  False,
                    'error':    result['error'],
                    'question': question,
                    'language': lang_code,
                }

            logger.info(f"✅  {result['row_count']} rows returned")

            # Zero-row guard
            if result['row_count'] == 0:
                return {
                    'success':     True,
                    'question':    question,
                    'explanation': "This information is not available in the uploaded data.",
                    'answer':      "This information is not available in the uploaded data.",
                    'data':        [],
                    'row_count':   0,
                    'language':    lang_code,
                }

            # Not-available sentinel guard — intercept before LLM
            if self._is_not_available_result(result.get('data_preview', [])):
                logger.info("🚫  SQL sentinel detected — skipping LLM")
                return {
                    'success':     True,
                    'question':    question,
                    'explanation': "This information is not available in the uploaded data.",
                    'answer':      "This information is not available in the uploaded data.",
                    'data':        result.get('data_preview', []),
                    'row_count':   result['row_count'],
                    'language':    lang_code,
                }

            result = self._smart_sort_results(result)

            # Fast path for simple count/sum queries — skip the explanation LLM,
            # but only when the result is a single number. Grouped results
            # ("how many employees per department") go to the LLM so the
            # per-group breakdown is explained instead of summed into one total.
            if self._is_simple_aggregation(question):
                single = self._single_numeric_value(result)
                if single is not None:
                    value = int(single) if float(single).is_integer() else round(single, 2)
                    explanation = (
                        f"Total: {value:,} "
                        f"{self._extract_entity(question, result)}"
                    )
                    return self._build_response(
                        question, lang_code, lang_name, sql_query, result, explanation
                    )

            results_text = self.sql_executor.format_results_as_text(result)
            logger.info("💬  Generating explanation...")
            explanation = self.explanation_chain.invoke({
                'question':     question,
                'sql':          sql_query,
                'results':      results_text,
                'language':     lang_code,
                'language_name': lang_name,
            }).strip()

            logger.info("✅  Analysis complete")
            return self._build_response(
                question, lang_code, lang_name, sql_query, result, explanation
            )

        except Exception as e:
            logger.error(f"❌  Analysis error: {e}")
            return {'success': False, 'error': str(e), 'question': question}

    # ── Schema builder ────────────────────────────────────────────────────────

    def _get_filtered_schema(self, allowed_tables: List[str], conversation_id: Optional[str] = None) -> str:
        """
        Build a schema string restricted to allowed_tables.
        Injects actual distinct values for text columns so the LLM uses exact
        spellings in WHERE clauses (e.g. "Technology & Engineering" not "Technology").

        Cache is session-scoped: key = conversation_id + sorted table names.
        Clearing one session's cache never affects another session.
        """
        table_key = ",".join(sorted(allowed_tables))
        session   = conversation_id if conversation_id else "global"
        cache_key = session + "|" + table_key
        if cache_key in self._schema_cache:
            logger.info("📋  Schema cache hit — skipping SELECT DISTINCT queries")
            return self._schema_cache[cache_key]

        full_schema = self.db.get_schema()
        filtered    = {t: cols for t, cols in full_schema.items() if t in allowed_tables}

        if not filtered:
            return "No schema available for the specified tables."

        text = "DATABASE SCHEMA (uploaded tables only):\n\n"
        for table, columns in filtered.items():
            text += f"Table: {table}\nColumns:\n"
            for col in columns:
                nullable = "NULL" if not col["notnull"] else "NOT NULL"
                pk       = " PRIMARY KEY" if col["primary_key"] else ""
                text    += f'  - "{col["name"]}" ({col["type"]}) {nullable}{pk}\n'

            # Inject actual distinct values for text/string columns
            try:
                text += "  ACTUAL VALUES (use EXACT spelling in WHERE clauses):\n"
                for col in columns:
                    if col["type"].upper() in ("TEXT", "VARCHAR", "CHAR", "STRING", ""):
                        rows = self.db.execute_query(
                            f'SELECT DISTINCT "{col["name"]}" FROM "{table}" '
                            f'WHERE "{col["name"]}" IS NOT NULL LIMIT 20'
                        )
                        if not rows.empty:
                            vals = rows.iloc[:, 0].tolist()
                            text += f'    {col["name"]}: {vals}\n'
            except Exception as e:
                logger.warning(f"⚠️  Could not fetch distinct values: {e}")

            text += "\n"

        self._schema_cache[cache_key] = text
        logger.info(f"📋  Schema cached for session '{conversation_id}', tables: {allowed_tables}")
        return text

    def invalidate_schema_cache(self, conversation_id: Optional[str] = None) -> None:
        """
        Call this after a new CSV is uploaded so the next query rebuilds the schema.
        Pass conversation_id to clear only that session's cache entries.
        Pass None to clear all (e.g. on server restart).
        Session-scoped: clearing session A never evicts session B's cache.
        """
        if conversation_id is None:
            self._schema_cache.clear()
            logger.info("📋  Schema cache cleared (all sessions)")
        else:
            # Remove only keys belonging to this conversation
            prefix = conversation_id + "|"
            keys_to_drop = [k for k in self._schema_cache if k.startswith(prefix)]
            for k in keys_to_drop:
                del self._schema_cache[k]
            logger.info(f"📋  Schema cache cleared for session: {conversation_id}")

    def _build_response(
        self,
        question:    str,
        lang_code:   str,
        lang_name:   str,
        sql_query:   str,
        result:      Dict[str, Any],
        explanation: str,
    ) -> Dict[str, Any]:
        response = {
            'success':       True,
            'question':      question,
            'language':      lang_code,
            'language_name': lang_name,
            'sql_query':     sql_query,
            'row_count':     result['row_count'],
            'data':          result['data_preview'],
            'explanation':   explanation,
            'raw_results':   result,
        }
        if self.plot_builder.needs_plot(question):
            plot_spec = self.plot_builder.build_plot_spec(result, question)
            if plot_spec:
                response['plot'] = plot_spec
                logger.info(f"📊  Chart: {plot_spec['type']}")
        return response

    def get_database_info(self) -> Dict[str, Any]:
        tables = self.db.list_tables()
        return {'tables': tables, 'schema': self.db.get_schema(), 'total_tables': len(tables)}