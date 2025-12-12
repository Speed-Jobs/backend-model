"""Database Query Tool

Tool for executing SQL queries and statistical analysis.
"""

from typing import List, Dict, Any
from sqlalchemy import create_engine, text
from app.core.config import settings


class DatabaseQueryTool:
    """Tool for database query operations"""

    def __init__(self):
        db_url = (
            f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        self.engine = create_engine(db_url, pool_pre_ping=True)

    def execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return results

        Args:
            sql_query: SQL query to execute

        Returns:
            List of dictionaries representing query results
        """
        print(f"[DatabaseQueryTool] Executing query")

        try:
            stats_data = []
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                columns = list(result.keys())
                for row in result:
                    stats_data.append(dict(zip(columns, row)))

            print(f"[DatabaseQueryTool] Query executed: {len(stats_data)} rows returned")
            return stats_data

        except Exception as e:
            print(f"[DatabaseQueryTool] Error: {e}")
            raise
