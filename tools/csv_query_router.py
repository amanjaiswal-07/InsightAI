'''Routes the queries to SQL, RAG or HYBRID based on usecases:
- Precise calculations → SQL
- Deep understanding → RAG
- Both → HYBRID
'''

import re
from typing import Literal
from utils.llm import make_llm
from utils.logger import logger


QueryRoute = Literal["SQL", "RAG", "HYBRID"]


class CSVQueryRouter:


    # ── Fast-path: unambiguous SQL signals ───────────────────────────
    # Only patterns where intent is 100% clear regardless of context.
    # Keep this list SHORT — false positives here skip LLM entirely.
    _FAST_SQL = re.compile(
        r'\b(how many|top \d+|bottom \d+|count|sum of|total|'
        r'average .{1,20} by|group by|filter by|sort by|'
        r'list all|show all|give me all)\b',
        re.IGNORECASE,
    )

    # ── Fast-path: unambiguous chart signals ─────────────────────────
    # Charts are always SQL — they need actual numeric data to render.
    _FAST_CHART = re.compile(
        r'\b(bar chart|pie chart|line chart|histogram|scatter|'
        r'plot|graph|visualize|visualization)\b',
        re.IGNORECASE,
    )

    def __init__(self):
        # Deterministic routing. The answer is one word, but gpt-oss is a
        # reasoning model whose hidden reasoning counts toward max_tokens: a
        # tight cap (it used to be 10) returns an empty string every time.
        self._llm = make_llm(temperature=0, max_tokens=512)
        logger.info("🧭 CSV Query Router initialized")

    def route(self, question: str, has_csv_data: bool = False) -> QueryRoute:
        '''Route to SQL, RAG, or HYBRID.'''

        if not question or len(question.strip()) < 3:
            logger.warning("⚠️ Query too short — defaulting to SQL")
            return "SQL"

        if not has_csv_data:
            return "RAG"

        q = question.strip()

        # ── Fast path: chart ─────────────────────────────────────────
        if self._FAST_CHART.search(q):
            logger.info("📊 CSV Route: SQL (chart/visualization)")
            return "SQL"

        # ── Fast path: unambiguous SQL ────────────────────────────────
        if self._FAST_SQL.search(q):
            logger.info("📊 CSV Route: SQL (fast-path pattern)")
            return "SQL"

        # ── Slow path: LLM decides ────────────────────────────────────
        return self._llm_classify(q)

    def _llm_classify(self, question: str) -> QueryRoute:

        prompt = f"""You are routing a data question to the correct handler.

SQL   → needs precise numbers: counts, averages, rankings, filters, aggregations, correlations between columns (e.g. "do older patients stay longer", "does X affect Y")
RAG   → needs interpretation of patterns already visible in data: summaries, overviews, "what trends do you see", "describe the data"
HYBRID → needs BOTH a calculation AND explanation: "why did X change", "which dept should we hire and why", "explain the drop"

Reply with ONLY one word: SQL, RAG, or HYBRID

Question: {question}
Category:"""

        try:
            response = self._llm.invoke(prompt).content.strip().upper()
            logger.info(f"🤖 LLM Route: {response}")

            if "HYBRID" in response:
                return "HYBRID"
            if "RAG" in response:
                return "RAG"
            return "SQL"   # default to SQL for structured data

        except Exception as e:
            logger.warning(f"⚠️ LLM routing failed: {e} — defaulting to SQL")
            return "SQL"

    def explain_route(self, question: str, route: QueryRoute) -> str:
        explanations = {
            "SQL":    "Requires precise calculations or aggregations from the CSV.",
            "RAG":    "Requires semantic understanding or pattern interpretation.",
            "HYBRID": "Requires both precise calculations and contextual explanation.",
        }
        return explanations.get(route, "Unknown routing decision")