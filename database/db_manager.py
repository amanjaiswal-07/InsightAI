# database/db_manager.py
"""
Database manager used by every agent.

SQLite is the supported backend (uploaded CSVs become tables here).
execute_query() is the path LLM-generated SQL takes, so it enforces two
independent guards: a statement allow-list (single read-only SELECT/WITH)
and a read-only connection (SQLite mode=ro).
"""

import re
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from utils.logger import logger


# Statements permitted through execute_query(). LLM-generated SQL is never
# trusted to self-police — this allow-list is the first of two independent
# guards (the second is opening the DB connection itself in read-only mode).
_READ_ONLY_STATEMENT_RE = re.compile(r"^\s*(SELECT|WITH|EXPLAIN|PRAGMA\s+table_info)\b", re.IGNORECASE)

# Any of these appearing as a standalone SQL keyword anywhere in the query
# is grounds for rejection, even if the statement starts with SELECT/WITH —
# guards against smuggled mutations via CTEs, subqueries, or stacked
# statements (e.g. "SELECT 1; DROP TABLE x").
_FORBIDDEN_KEYWORDS_RE = re.compile(
    # REPLACE is only blocked as a statement ("REPLACE INTO"); the REPLACE()
    # string function is a legitimate read-only expression.
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE(?!\s*\()|ATTACH|DETACH|"
    r"GRANT|REVOKE|VACUUM|REINDEX|PRAGMA(?!\s+table_info))\b",
    re.IGNORECASE,
)


def _assert_read_only_sql(query: str) -> None:
    """Raise ValueError if `query` is anything other than a single read-only statement."""
    stripped = query.strip().rstrip(";")
    if ";" in stripped:
        raise ValueError("Multiple/stacked SQL statements are not allowed")
    if not _READ_ONLY_STATEMENT_RE.match(stripped):
        raise ValueError("Only SELECT/WITH/EXPLAIN queries are allowed")
    if _FORBIDDEN_KEYWORDS_RE.search(stripped):
        raise ValueError("Query contains a disallowed (non-read-only) SQL keyword")


class DatabaseManager:
    """Universal database manager"""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            from config import get_db_config
            config = get_db_config()

        self.config = config
        self.db_type = config["type"]

        if self.db_type == "sqlite":
            self._init_sqlite(config["path"])
        elif self.db_type == "postgresql":
            self._init_postgresql(config)
        elif self.db_type == "mysql":
            self._init_mysql(config)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

        logger.info(f"📊 Database initialized: {self.db_type}")

    def _init_sqlite(self, db_path: str):
        import sqlite3
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.connection_string = db_path
        conn = sqlite3.connect(db_path)
        conn.close()
        logger.info(f"   Path: {db_path}")

    def _init_postgresql(self, config: Dict):
        try:
            import psycopg2
        except ImportError:
            raise ImportError("PostgreSQL requires psycopg2-binary: pip install psycopg2-binary")

        self.connection_string = (
            f"postgresql://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['database']}"
        )

        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        conn.close()
        logger.info(f"   Host: {config['host']}:{config['port']}")

    def _init_mysql(self, config: Dict):
        try:
            import pymysql
        except ImportError:
            raise ImportError("MySQL requires pymysql: pip install pymysql")

        self.connection_string = (
            f"mysql+pymysql://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['database']}"
        )

        conn = pymysql.connect(
            host=config['host'],
            port=config['port'],
            database=config['database'],
            user=config['user'],
            password=config['password']
        )
        conn.close()
        logger.info(f"   Host: {config['host']}:{config['port']}")

    def get_connection(self):
        """Get database connection"""
        if self.db_type == "sqlite":
            import sqlite3
            return sqlite3.connect(self.db_path)
        elif self.db_type == "postgresql":
            import psycopg2
            config = self.config
            return psycopg2.connect(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password']
            )
        elif self.db_type == "mysql":
            import pymysql
            config = self.config
            return pymysql.connect(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password']
            )

    def get_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """Get database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            if self.db_type == "sqlite":
                return self._get_sqlite_schema(cursor)
            elif self.db_type == "postgresql":
                return self._get_postgres_schema(cursor)
            elif self.db_type == "mysql":
                return self._get_mysql_schema(cursor)
        finally:
            conn.close()

    def _get_sqlite_schema(self, cursor) -> Dict:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table in tables:
            # Table name must be quoted — PRAGMA table_info(x) does not
            # accept bind parameters, and an unquoted identifier containing
            # any non-alphanumeric character (a hyphen, for example) is a
            # syntax error rather than a valid table reference.
            cursor.execute(f'PRAGMA table_info("{table}");')
            columns = cursor.fetchall()
            schema[table] = [
                {
                    "name": col[1],
                    "type": col[2],
                    "notnull": bool(col[3]),
                    "primary_key": bool(col[5])
                }
                for col in columns
            ]
        return schema

    def _get_postgres_schema(self, cursor) -> Dict:
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table in tables:
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table}'
            """)
            columns = cursor.fetchall()
            schema[table] = [
                {
                    "name": col[0],
                    "type": col[1],
                    "notnull": col[2] == 'NO',
                    "primary_key": False
                }
                for col in columns
            ]
        return schema

    def _get_mysql_schema(self, cursor) -> Dict:
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table in tables:
            cursor.execute(f"DESCRIBE {table}")
            columns = cursor.fetchall()
            schema[table] = [
                {
                    "name": col[0],
                    "type": col[1],
                    "notnull": col[2] == 'NO',
                    "primary_key": col[3] == 'PRI'
                }
                for col in columns
            ]
        return schema

    def get_schema_text(self) -> str:
        """Get schema as formatted text"""
        schema = self.get_schema()

        schema_text = f"DATABASE SCHEMA ({self.db_type.upper()}):\n\n"

        for table, columns in schema.items():
            schema_text += f"Table: {table}\n"
            schema_text += "Columns:\n"

            for col in columns:
                nullable = "NULL" if not col["notnull"] else "NOT NULL"
                pk = " PRIMARY KEY" if col["primary_key"] else ""
                schema_text += f"  - {col['name']} ({col['type']}) {nullable}{pk}\n"

            schema_text += "\n"

        return schema_text

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a read-only SQL query (the path LLM-generated SQL runs through).

        Two independent guards, since the SQL text here originates from an LLM
        and must never be trusted to only contain SELECT statements on its own:
          1. Statement allow-list — reject anything that isn't a single
             read-only SELECT/WITH/EXPLAIN/PRAGMA-table-info statement before
             it ever reaches the database driver.
          2. A connection opened read-only at the OS/driver level (SQLite URI
             mode=ro; Postgres/MySQL default-transaction READ ONLY) as
             defense-in-depth, so even a guard bypass can't mutate data.
        """
        _assert_read_only_sql(query)
        logger.info(f"🔍 Executing query: {query[:100]}...")

        try:
            if self.db_type == "sqlite":
                import sqlite3
                conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            else:
                from sqlalchemy import create_engine
                engine = create_engine(
                    self.connection_string,
                    execution_options={"postgresql_readonly": True} if self.db_type == "postgresql" else {},
                )
                conn = engine.connect()
                if self.db_type == "mysql":
                    conn.execute("SET SESSION TRANSACTION READ ONLY")

            df = pd.read_sql_query(query, conn)
            conn.close()

            logger.info(f"✅ Query returned {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"❌ Query error: {e}")
            raise

    def get_table_sample(self, table: str, limit: int = 5) -> pd.DataFrame:
        """Get sample rows"""
        query = f'SELECT * FROM "{table}" LIMIT {int(limit)}'
        return self.execute_query(query)

    def list_tables(self) -> List[str]:
        """List all tables"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            if self.db_type == "sqlite":
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            elif self.db_type == "postgresql":
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                """)
            elif self.db_type == "mysql":
                cursor.execute("SHOW TABLES")

            tables = [row[0] for row in cursor.fetchall()]
            return tables
        finally:
            conn.close()

    def get_db_info(self) -> Dict[str, Any]:
        """Get database information"""
        return {
            "type": self.db_type,
            "tables": self.list_tables(),
            "connection": self.connection_string if self.db_type != "sqlite" else self.db_path
        }


if __name__ == "__main__":
    print("\n🧪 Testing Database Connection...\n")
    try:
        db = DatabaseManager()
        print(f"✅ Connected to {db.db_type}")
        print(f"\nTables: {db.list_tables()}")
        print(f"\nSchema preview:")
        print(db.get_schema_text()[:500] + "...")
    except Exception as e:
        print(f"❌ Connection failed: {e}")