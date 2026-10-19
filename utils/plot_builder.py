# utils/plot_builder.py
"""
Plot Builder: turns SQL results into chart specs (bar / line / pie) for the UI.
"""

from typing import Dict, Any, List, Optional
from utils.logger import logger


class PlotBuilder:
    """Build chart specifications from query results"""

    # Keywords that indicate need for plotting
    PLOT_KEYWORDS = [
        "chart", "plot", "graph", "visualize", "show",
        "trend", "over time", "breakdown", "distribution",
        "compare", "comparison"
    ]

    # CHART TYPE KEYWORDS - Priority order
    CHART_TYPE_KEYWORDS = {
        "bar": ["bar chart", "bar graph", "bar"],
        "line": ["line chart", "line graph", "line", "trend"],
        "pie": ["pie chart", "pie graph", "pie", "donut"],
    }

    def __init__(self):
        logger.info("📊 Plot Builder initialized")

    def needs_plot(self, question: str) -> bool:
        """Check if question requests a chart"""
        q = question.lower()
        return any(keyword in q for keyword in self.PLOT_KEYWORDS)

    def build_plot_spec(
        self,
        sql_result: Dict[str, Any],
        question: str
    ) -> Optional[Dict[str, Any]]:
        """Build chart specification from SQL result"""

        if not sql_result.get("success"):
            logger.warning("⚠️ Cannot plot failed query")
            return None

        rows = sql_result.get("data_preview", [])

        if not rows or len(rows) < 2:
            logger.warning("⚠️ Insufficient data for plotting (need 2+ rows)")
            return None

        keys = list(rows[0].keys())
        if len(keys) < 1:
            logger.warning("⚠️ No columns found")
            return None

        # Smart column detection: find which column is numeric and which is categorical
        col1 = keys[0]
        col2 = keys[1] if len(keys) > 1 else keys[0]

        try:
            col1_vals = [row[col1] for row in rows]
            col2_vals = [row[col2] for row in rows]

            col1_is_num = all(isinstance(v, (int, float)) or self._is_numeric_string(v) for v in col1_vals if v is not None)
            col2_is_num = all(isinstance(v, (int, float)) or self._is_numeric_string(v) for v in col2_vals if v is not None)

            if col1_is_num and not col2_is_num:
                x_col, y_col = col2, col1
            else:
                x_col, y_col = col1, col2

            x_values = [str(row[x_col]) if row[x_col] is not None else "" for row in rows]
            y_values = [row[y_col] for row in rows]

            if not all(isinstance(v, (int, float)) or self._is_numeric_string(v) for v in y_values if v is not None):
                logger.warning("⚠️ Y-axis values are not numeric")
                return None

            y_values = [float(v) if v is not None and isinstance(v, str) else (v or 0.0) for v in y_values]

        except Exception as e:
            logger.error(f"❌ Failed to extract values: {e}")
            return None

        # FIXED: Detect chart type from question FIRST
        chart_type = self._infer_chart_type(question, x_col, y_col)

        plot_spec = {
            "type": chart_type,
            "x": x_values,
            "y": y_values,
            "x_label": x_col,
            "y_label": y_col,
            "title": self._generate_title(question, x_col, y_col)
        }

        logger.info(f"✅ Plot spec created: {chart_type} chart with {len(x_values)} points")

        return plot_spec

    def _infer_chart_type(self, question: str, x_col: str, y_col: str) -> str:
        """
        FIXED: Infer chart type with PRIORITY for user's explicit request

        Priority:
        1. User explicitly says "bar chart" → bar
        2. User explicitly says "line chart" → line
        3. User explicitly says "pie chart" → pie
        4. Time-based data → line (default for trends)
        5. Categorical data → bar (default)
        """
        q = question.lower()
        x_lower = x_col.lower()

        # PRIORITY 1: Explicit user request (HIGHEST PRIORITY)
        # Check in order: bar → line → pie
        for chart_type, keywords in self.CHART_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in q:
                    logger.info(f"📊 Chart type: {chart_type} (explicit user request: '{keyword}')")
                    return chart_type

        # PRIORITY 2: Distribution/breakdown → pie
        if any(kw in q for kw in ["distribution", "breakdown", "share", "proportion"]):
            logger.info("📊 Chart type: pie (distribution query)")
            return "pie"

        # PRIORITY 3: Time-based data → line
        if any(time_kw in x_lower for time_kw in ["month", "year", "quarter", "date", "time", "week", "day"]):
            logger.info("📊 Chart type: line (time-based data)")
            return "line"

        # PRIORITY 4: Default to bar
        logger.info("📊 Chart type: bar (default)")
        return "bar"

    def _generate_title(self, question: str, x_col: str, y_col: str) -> str:
        """Generate chart title"""
        return f"{y_col} by {x_col}"

    def _is_numeric_string(self, value: Any) -> bool:
        """Check if string can be converted to number"""
        if not isinstance(value, str):
            return False
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False


# Convenience function
def create_plot_spec(sql_result: Dict, question: str) -> Optional[Dict]:
    """Create plot specification from SQL result and question"""
    builder = PlotBuilder()
    return builder.build_plot_spec(sql_result, question)