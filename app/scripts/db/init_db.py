"""
Database Table Initialization Script (RECOMMENDED VERSION with URL hash)
Creates all MySQL tables based on the data model with proper dependency order
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Optional
from contextlib import contextmanager
from app.db.config.db_connection import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('db_init.log')
    ]
)
logger = logging.getLogger(__name__)


class TableInitializer:
    """Database table initialization manager"""
    
    def __init__(self):
        self.db = DatabaseManager()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
    
    @contextmanager
    def _transaction(self):
        """Context manager for database transactions"""
        cursor = None
        try:
            with self.db.get_cursor(dict_cursor=False) as cursor:
                yield cursor
                cursor.execute("COMMIT")
        except Exception as e:
            if cursor:
                cursor.execute("ROLLBACK")
            logger.error(f"Transaction failed: {e}")
            raise
    
    def create_all_tables(self) -> bool:
        """
        Create all tables in proper dependency order
        Returns: True if successful, False otherwise
        """
        try:
            with self._transaction() as cursor:
                logger.info("Starting table creation...")
                
                # Create tables in dependency order
                self._create_position_table(cursor)
                self._create_company_table(cursor)
                self._create_industry_table(cursor)
                self._create_skill_table(cursor)
                self._create_post_table(cursor)
                self._create_post_skill_table(cursor)
                self._create_industry_skill_table(cursor)
                self._create_position_skill_table(cursor)
                self._create_dashboard_stat_table(cursor)
                
                logger.info("✓ All tables created successfully!")
                return True
                
        except Exception as e:
            logger.error(f"✗ Failed to create tables: {e}")
            return False
    
    def _create_position_table(self, cursor):
        """Create position table (no dependencies)"""
        logger.info("Creating position table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS `position` (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Position ID',
                name VARCHAR(255) NOT NULL COMMENT 'Position name',
                description TEXT COMMENT 'Position description',
                skillset TEXT COMMENT 'Position skillset',
                category ENUM('TECH', 'BIZ', 'BIZ_SUPPORTING') COMMENT 'Position category',
                is_deleted BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Soft delete flag',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created timestamp',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated timestamp',
                INDEX idx_name (name(191))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Job positions';
        """)
    
    def _create_company_table(self, cursor):
        """Create company table (no dependencies)"""
        logger.info("Creating company table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS company (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Company ID',
                name VARCHAR(255) NOT NULL COMMENT 'Company name',
                description TEXT COMMENT 'Company description',
                domain VARCHAR(255) COMMENT 'Company domain',
                location VARCHAR(255) COMMENT 'Company location',
                founded_year INT COMMENT 'Year founded',
                size VARCHAR(50) COMMENT 'Company size',
                logo VARCHAR(500) COMMENT 'Logo image URL',
                is_competitor BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Competitor flag',
                is_deleted BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Soft delete flag',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created timestamp',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated timestamp',
                INDEX idx_name (name(191)),
                INDEX idx_domain (domain(191))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Companies';
        """)
    
    def _create_industry_table(self, cursor):
        """Create industry table (depends on position)"""
        logger.info("Creating industry table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS industry (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Industry ID',
                name VARCHAR(255) NOT NULL COMMENT 'Industry name',
                description TEXT COMMENT 'Industry description',
                skillset TEXT COMMENT 'Industry skillset',
                position_id INT COMMENT 'Related position ID',
                is_deleted BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Soft delete flag',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created timestamp',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated timestamp',
                INDEX idx_name (name(191)),
                INDEX idx_position_id (position_id),
                FOREIGN KEY (position_id) REFERENCES `position`(id) ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Industries';
        """)
    
    def _create_skill_table(self, cursor):
        """Create skill table (no dependencies)"""
        logger.info("Creating skill table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Skill ID',
                name VARCHAR(255) NOT NULL UNIQUE COMMENT 'Skill name',
                is_deleted BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Soft delete flag',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created timestamp',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated timestamp',
                INDEX idx_name (name(191))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Skills and technologies';
        """)
    
    def _create_post_table(self, cursor):
        """Create post table (depends on company, industry)"""
        logger.info("Creating post table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS post (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Post ID',
                title VARCHAR(500) NOT NULL COMMENT 'Job posting title',
                employment_type VARCHAR(50) COMMENT 'Employment type',
                experience VARCHAR(100) COMMENT 'Required experience',
                work_type VARCHAR(50) COMMENT 'Work type',
                description TEXT COMMENT 'Job description',
                meta_data JSON COMMENT 'Additional metadata',
                posted_at DATETIME COMMENT 'Posted date',
                close_at DATETIME COMMENT 'Closing date',
                crawled_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Crawled timestamp',
                source_url VARCHAR(1000) NOT NULL COMMENT 'Original URL',
                source_url_hash CHAR(64) GENERATED ALWAYS AS (SHA2(source_url, 256)) STORED COMMENT 'SHA256 hash of source_url for duplicate detection',
                screenshot_url VARCHAR(1000) COMMENT 'Screenshot URL',
                company_id INT NOT NULL COMMENT 'Company ID',
                industry_id INT COMMENT 'Industry ID',
                is_deleted BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Soft delete flag',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created timestamp',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated timestamp',
                INDEX idx_title (title(191)),
                INDEX idx_posted_at (posted_at),
                INDEX idx_crawled_at (crawled_at),
                INDEX idx_company_id (company_id),
                INDEX idx_industry_id (industry_id),
                INDEX idx_source_url_prefix (source_url(255)),
                UNIQUE KEY idx_source_url_hash (source_url_hash),
                FULLTEXT INDEX ft_idx_title_description (title, description),
                FOREIGN KEY (company_id) REFERENCES company(id) ON DELETE CASCADE,
                FOREIGN KEY (industry_id) REFERENCES industry(id) ON DELETE SET NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Job postings';
        """)
    
    def _create_post_skill_table(self, cursor):
        """Create post_skill junction table"""
        logger.info("Creating post_skill table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS post_skill (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Mapping ID',
                skill_id INT NOT NULL COMMENT 'Skill ID',
                post_id INT NOT NULL COMMENT 'Post ID',
                is_deleted BOOLEAN NOT NULL DEFAULT FALSE COMMENT 'Soft delete flag',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created timestamp',
                modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Modified timestamp',
                INDEX idx_skill_id (skill_id),
                INDEX idx_post_id (post_id),
                UNIQUE KEY unique_post_skill (post_id, skill_id),
                FOREIGN KEY (skill_id) REFERENCES skill(id) ON DELETE CASCADE,
                FOREIGN KEY (post_id) REFERENCES post(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Post-Skill mapping';
        """)
    
    def _create_industry_skill_table(self, cursor):
        """Create industry_skill junction table"""
        logger.info("Creating industry_skill table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS industry_skill (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Mapping ID',
                skill_id INT NOT NULL COMMENT 'Skill ID',
                industry_id INT NOT NULL COMMENT 'Industry ID',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created timestamp',
                INDEX idx_skill_id (skill_id),
                INDEX idx_industry_id (industry_id),
                UNIQUE KEY unique_industry_skill (industry_id, skill_id),
                FOREIGN KEY (skill_id) REFERENCES skill(id) ON DELETE CASCADE,
                FOREIGN KEY (industry_id) REFERENCES industry(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Industry-Skill mapping';
        """)
    
    def _create_position_skill_table(self, cursor):
        """Create position_skill junction table"""
        logger.info("Creating position_skill table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_skill (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Mapping ID',
                position_id INT NOT NULL COMMENT 'Position ID',
                skill_id INT NOT NULL COMMENT 'Skill ID',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created timestamp',
                INDEX idx_position_id (position_id),
                INDEX idx_skill_id (skill_id),
                UNIQUE KEY unique_position_skill (position_id, skill_id),
                FOREIGN KEY (position_id) REFERENCES `position`(id) ON DELETE CASCADE,
                FOREIGN KEY (skill_id) REFERENCES skill(id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Position-Skill mapping';
        """)
    
    def _create_dashboard_stat_table(self, cursor):
        """Create dashboard_stat table"""
        logger.info("Creating dashboard_stat table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dashboard_stat (
                id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Stat ID',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created timestamp',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Updated timestamp'
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='Dashboard statistics';
        """)
    
    def drop_all_tables(self) -> bool:
        """
        Drop all tables in reverse dependency order
        Returns: True if successful, False otherwise
        """
        try:
            with self._transaction() as cursor:
                logger.warning("Dropping all tables...")
                
                # Drop in reverse dependency order
                tables = [
                    'dashboard_stat',
                    'position_skill',
                    'industry_skill',
                    'post_skill',
                    'post',
                    'skill',
                    'industry',
                    'company',
                    'position'
                ]
                
                for table in tables:
                    logger.info(f"Dropping table: {table}")
                    cursor.execute(f"DROP TABLE IF EXISTS {table}")
                
                logger.info("✓ All tables dropped successfully!")
                return True
                
        except Exception as e:
            logger.error(f"✗ Failed to drop tables: {e}")
            return False
    
    def show_tables(self) -> bool:
        """
        Display all tables and their structures
        Returns: True if successful, False otherwise
        """
        try:
            with self.db.get_cursor(dict_cursor=True) as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                
                if not tables:
                    logger.info("No tables found in database")
                    return True
                
                logger.info(f"\n{'='*80}")
                logger.info("Database Tables:")
                logger.info(f"{'='*80}\n")
                
                for table in tables:
                    table_name = list(table.values())[0]
                    
                    # Get table info
                    cursor.execute(f"SHOW TABLE STATUS LIKE '{table_name}'")
                    table_info = cursor.fetchone()
                    
                    logger.info(f"Table: {table_name}")
                    if table_info and 'Comment' in table_info:
                        logger.info(f"Comment: {table_info['Comment']}")
                    
                    # Get column information
                    cursor.execute(f"DESCRIBE {table_name}")
                    columns = cursor.fetchall()
                    
                    logger.info(f"\n{'Field':<20} {'Type':<25} {'Null':<6} {'Key':<6} {'Extra':<15}")
                    logger.info(f"{'-'*80}")
                    
                    for col in columns:
                        logger.info(
                            f"{col['Field']:<20} "
                            f"{col['Type']:<25} "
                            f"{col['Null']:<6} "
                            f"{col['Key']:<6} "
                            f"{col.get('Extra', ''):<15}"
                        )
                    
                    logger.info(f"\n{'-'*80}\n")
                
                return True
                
        except Exception as e:
            logger.error(f"✗ Failed to show tables: {e}")
            return False


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Database Table Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.scripts.init_db create      # Create all tables
  python -m app.scripts.init_db drop        # Drop all tables
  python -m app.scripts.init_db recreate    # Drop and recreate all tables
  python -m app.scripts.init_db show        # Show all tables and structures
        """
    )
    
    parser.add_argument(
        'action',
        choices=['create', 'drop', 'recreate', 'show'],
        help='Action to perform on database tables'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute requested action
    with TableInitializer() as initializer:
        if args.action == 'create':
            success = initializer.create_all_tables()
        elif args.action == 'drop':
            success = initializer.drop_all_tables()
        elif args.action == 'recreate':
            success = initializer.drop_all_tables() and initializer.create_all_tables()
        elif args.action == 'show':
            success = initializer.show_tables()
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()