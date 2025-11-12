import logging
from contextlib import contextmanager
import mysql.connector
from dotenv import load_dotenv
import os

# 로깅 설정 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


class DatabaseManager:
    """
    DB connection 관리
    """

    def __init__(self):
        """DB 설정 초기화."""
        self.db_config = {
            'host': os.getenv("DB_HOST"),
            'user': os.getenv("DB_USER"),
            'password': os.getenv("DB_PASSWORD"),
            'database': os.getenv("DB_NAME"),
            'port': int(os.getenv("DB_PORT", 3306)),
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci',
            'autocommit': False
        }
        self._connection = None
        logger.info("DatabaseManager initialized")

    def get_connection(self):
        """DB connection 반환"""
        try:
            if self._connection is None or not self._connection.is_connected():
                self._connection = mysql.connector.connect(**self.db_config)
                logger.info("Database connection established")
            return self._connection
        except Exception as e:
            import traceback
            logger.error(f"Failed to connect to database: {e}")
            logger.error("Traceback: %s", traceback.format_exc())
            raise

    @contextmanager
    def get_cursor(self, dict_cursor: bool = True):
        """Get database cursor with context management."""
        conn = self.get_connection()
        cursor = None
        try:
            if dict_cursor:
                cursor = conn.cursor(dictionary=True)
            else:
                cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception as e:
            import traceback
            conn.rollback()
            logger.error(f"Database operation failed: {e}")
            logger.error("Traceback: %s", traceback.format_exc())
            raise
        finally:
            if cursor:
                cursor.close()

    def close(self):
        """명시적으로 연결 종료"""
        if self._connection and self._connection.is_connected():
            self._connection.close()
            logger.info("Database connection closed")