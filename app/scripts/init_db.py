"""
테이블 생성 스크립트
models 폴더의 모든 모델을 기반으로 MySQL 테이블 생성
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from app.db.db_connection import DatabaseManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_tables():
    """모든 테이블 생성 (외래키 순서 고려)"""
    db = DatabaseManager()
    
    try:
        with db.get_cursor(dict_cursor=False) as cursor:
            logger.info("Starting table creation...")
            
            # 1. Position 테이블 (의존성 없음)
            logger.info("Creating position table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS position (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '아이디',
                    name VARCHAR(255) NOT NULL COMMENT '이름',
                    description TEXT COMMENT '설명',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일',
                    INDEX idx_name (name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='직무';
            """)
            
            # 2. Company 테이블 (의존성 없음)
            logger.info("Creating company table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS company (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '아이디',
                    name VARCHAR(255) NOT NULL COMMENT '이름',
                    description TEXT COMMENT '소개',
                    domain VARCHAR(255) COMMENT '도메인',
                    location VARCHAR(255) COMMENT '위치',
                    founded_year INT COMMENT '설립연도',
                    size VARCHAR(50) COMMENT '규모',
                    logo VARCHAR(500) COMMENT '로고_이미지',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일',
                    INDEX idx_name (name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='회사';
            """)
            
            # 3. Industry 테이블 (position_id 외래키)
            logger.info("Creating industry table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS industry (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '아이디',
                    name VARCHAR(255) NOT NULL COMMENT '이름',
                    description TEXT COMMENT '설명',
                    position_id INT COMMENT '직무id',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일',
                    INDEX idx_name (name),
                    INDEX idx_position_id (position_id),
                    FOREIGN KEY (position_id) REFERENCES position(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='산업군';
            """)
            
            # 4. Skill 테이블 (의존성 없음)
            logger.info("Creating skill table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS skill (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '아이디',
                    name VARCHAR(255) NOT NULL UNIQUE COMMENT '이름',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일',
                    INDEX idx_name (name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='기술스택';
            """)
            
            # 5. Post 테이블 (company_id, industry_id 외래키)
            logger.info("Creating post table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS post (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '아이디',
                    title VARCHAR(500) NOT NULL COMMENT '제목',
                    employment_type VARCHAR(50) COMMENT '채용형태',
                    experience VARCHAR(100) COMMENT '경력',
                    work_type VARCHAR(50) COMMENT '근무형태',
                    description TEXT COMMENT '공고 상세설명',
                    meta_data JSON COMMENT '메타데이터 (job_category, preferred_qualifications 등)',
                    posted_at DATETIME COMMENT '게시시작',
                    close_at DATETIME COMMENT '종료시작',
                    crawled_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '크롤링시작',
                    source_url VARCHAR(1000) NOT NULL UNIQUE COMMENT '원문url',
                    screenshot_url VARCHAR(1000) COMMENT '스크린샷_url',
                    company_id INT NOT NULL COMMENT '회사id',
                    industry_id INT COMMENT '산업id',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일',
                    INDEX idx_title (title),
                    INDEX idx_posted_at (posted_at),
                    INDEX idx_crawled_at (crawled_at),
                    INDEX idx_company_id (company_id),
                    INDEX idx_industry_id (industry_id),
                    INDEX idx_source_url (source_url),
                    FOREIGN KEY (company_id) REFERENCES company(id) ON DELETE CASCADE,
                    FOREIGN KEY (industry_id) REFERENCES industry(id) ON DELETE SET NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='채용공고';
            """)
            
            # 6. PostSkill 중간 테이블 (post_id, skill_id 외래키)
            logger.info("Creating post_skill table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS post_skill (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '아이디',
                    skill_id INT NOT NULL COMMENT '기술id',
                    post_id INT NOT NULL COMMENT '공고id',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일',
                    INDEX idx_skill_id (skill_id),
                    INDEX idx_post_id (post_id),
                    UNIQUE KEY unique_post_skill (post_id, skill_id),
                    FOREIGN KEY (skill_id) REFERENCES skill(id) ON DELETE CASCADE,
                    FOREIGN KEY (post_id) REFERENCES post(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='공고-기술 매핑';
            """)
            
            # 7. IndustrySkill 중간 테이블 (industry_id, skill_id 외래키)
            logger.info("Creating industry_skill table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS industry_skill (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '아이디',
                    skill_id INT NOT NULL COMMENT '기술id',
                    industry_id INT NOT NULL COMMENT '산업id',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일',
                    INDEX idx_skill_id (skill_id),
                    INDEX idx_industry_id (industry_id),
                    UNIQUE KEY unique_industry_skill (industry_id, skill_id),
                    FOREIGN KEY (skill_id) REFERENCES skill(id) ON DELETE CASCADE,
                    FOREIGN KEY (industry_id) REFERENCES industry(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='산업-기술 매핑';
            """)
            
            # 8. PositionSkill 중간 테이블 (position_id, skill_id 외래키)
            logger.info("Creating position_skill table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS position_skill (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '아이디',
                    position_id INT NOT NULL COMMENT '직무id',
                    skill_id INT NOT NULL COMMENT '기술id',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일',
                    INDEX idx_position_id (position_id),
                    INDEX idx_skill_id (skill_id),
                    UNIQUE KEY unique_position_skill (position_id, skill_id),
                    FOREIGN KEY (position_id) REFERENCES position(id) ON DELETE CASCADE,
                    FOREIGN KEY (skill_id) REFERENCES skill(id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='직무-기술 매핑';
            """)
            
            # 9. DashboardStat 테이블 (의존성 없음, 빈 테이블)
            logger.info("Creating dashboard_stat table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dashboard_stat (
                    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '아이디',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '생성일',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '수정일'
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='대시보드 통계';
            """)
            
            logger.info("All tables created successfully!")
            
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise
    finally:
        db.close()


def drop_tables():
    """모든 테이블 삭제 (개발용) - 외래키 순서 역순"""
    db = DatabaseManager()
    
    try:
        with db.get_cursor(dict_cursor=False) as cursor:
            logger.warning("Dropping all tables...")
            
            # 외래키 제약 때문에 생성 역순으로 삭제
            cursor.execute("DROP TABLE IF EXISTS dashboard_stat")
            cursor.execute("DROP TABLE IF EXISTS position_skill")
            cursor.execute("DROP TABLE IF EXISTS industry_skill")
            cursor.execute("DROP TABLE IF EXISTS post_skill")
            cursor.execute("DROP TABLE IF EXISTS post")
            cursor.execute("DROP TABLE IF EXISTS skill")
            cursor.execute("DROP TABLE IF EXISTS industry")
            cursor.execute("DROP TABLE IF EXISTS company")
            cursor.execute("DROP TABLE IF EXISTS position")
            
            logger.info("All tables dropped successfully!")
            
    except Exception as e:
        logger.error(f"Failed to drop tables: {e}")
        raise
    finally:
        db.close()


def show_tables():
    """생성된 테이블 목록 확인"""
    db = DatabaseManager()
    
    try:
        with db.get_cursor(dict_cursor=True) as cursor:
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            logger.info("Current tables:")
            for table in tables:
                table_name = list(table.values())[0]
                logger.info(f"  - {table_name}")
                
                # 테이블 구조 확인
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                for col in columns:
                    logger.info(f"      {col['Field']} {col['Type']} {col['Null']} {col['Key']}")
                    
    except Exception as e:
        logger.error(f"Failed to show tables: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Database table management')
    parser.add_argument('action', choices=['create', 'drop', 'recreate', 'show'], 
                        help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'create':
        create_tables()
    elif args.action == 'drop':
        drop_tables()
    elif args.action == 'recreate':
        drop_tables()
        create_tables()
    elif args.action == 'show':
        show_tables()