from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd
import numpy as np

# 경쟁사 목록 상수
COMPETITOR_COMPANIES = [
    '현대', 'Coupang', '한화', '카카오', 'LINE', 
    'NAVER', '토스', '비바리퍼블리카', '우아한형제들', '배달의민족'
]

def get_competitors_skill_diversity_all(db: Session) -> List[Dict]:
    """전체 경쟁사별 스킬 다양성 조회"""
    
    # WHERE 조건 동적 생성
    where_conditions = " OR ".join([f"c.name LIKE :company_{i}" for i in range(len(COMPETITOR_COMPANIES))])
    
    query = text(f"""
        SELECT 
            c.name AS company,
            COUNT(DISTINCT s.id) AS skills
        FROM company c
        INNER JOIN post p ON c.id = p.company_id
        INNER JOIN post_skill ps ON p.id = ps.post_id
        INNER JOIN skill s ON ps.skill_id = s.id
        WHERE {where_conditions}
        GROUP BY c.id, c.name
        ORDER BY skills DESC, c.name
    """)
    
    # 파라미터 바인딩
    params = {f"company_{i}": f"%{company}%" for i, company in enumerate(COMPETITOR_COMPANIES)}
    
    result = db.execute(query, params)
    return [{"company": row.company, "skills": row.skills} for row in result]


def get_competitors_skill_diversity_by_year(db: Session, year: int) -> List[Dict]:
    """연도별 경쟁사별 스킬 다양성 조회"""
    
    where_conditions = " OR ".join([f"c.name LIKE :company_{i}" for i in range(len(COMPETITOR_COMPANIES))])
    
    query = text(f"""
        SELECT 
            c.name AS company,
            COUNT(DISTINCT s.id) AS skills
        FROM company c
        INNER JOIN post p ON c.id = p.company_id
        INNER JOIN post_skill ps ON p.id = ps.post_id
        INNER JOIN skill s ON ps.skill_id = s.id
        WHERE ({where_conditions})
          AND YEAR(p.posted_at) = :year
        GROUP BY c.id, c.name
        ORDER BY skills DESC, c.name
    """)
    
    params = {f"company_{i}": f"%{company}%" for i, company in enumerate(COMPETITOR_COMPANIES)}
    params['year'] = year
    
    result = db.execute(query, params)
    return [{"company": row.company, "skills": row.skills} for row in result]


def get_competitors_posts_with_skills(
    db: Session, 
    company_name: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 100
) -> List[Dict]:
    """경쟁사별 공고 및 스킬 상세 조회"""
    
    where_conditions = " OR ".join([f"c.name LIKE :company_{i}" for i in range(len(COMPETITOR_COMPANIES))])
    
    additional_where = []
    params = {f"company_{i}": f"%{company}%" for i, company in enumerate(COMPETITOR_COMPANIES)}
    
    if company_name:
        additional_where.append("c.name LIKE :target_company")
        params['target_company'] = f"%{company_name}%"
    
    if year:
        additional_where.append("YEAR(p.posted_at) = :year")
        params['year'] = year
    
    where_clause = where_conditions
    if additional_where:
        where_clause += " AND " + " AND ".join(additional_where)
    
    query = text(f"""
        SELECT
            p.id AS post_id,
            p.title AS post_title,
            p.posted_at,
            p.close_at,
            p.crawled_at,
            c.id AS company_id,
            c.name AS company_name,
            GROUP_CONCAT(s.name ORDER BY s.name SEPARATOR ', ') AS skills
        FROM company c
        INNER JOIN post p ON c.id = p.company_id
        INNER JOIN post_skill ps ON p.id = ps.post_id
        INNER JOIN skill s ON ps.skill_id = s.id
        WHERE {where_clause}
        GROUP BY p.id, p.title, p.posted_at, p.close_at, p.crawled_at, c.id, c.name
        ORDER BY c.name, p.posted_at DESC
        LIMIT :limit
    """)
    
    params['limit'] = limit
    
    result = db.execute(query, params)
    return [dict(row._mapping) for row in result]


def get_company_skill_trends(
    db: Session,
    company_id: str,
    year: int,
    top_n: int = 10
) -> Dict:
    """회사별 상위 스킬 분기별 트렌드 조회"""
    
    # company_id가 숫자인지 확인 (ID인지 이름인지)
    try:
        company_id_int = int(company_id)
        where_clause = "c.id = :company_id"
        params = {"company_id": company_id_int}
        company_where_clause = "id = :company_id"
        company_params = {"company_id": company_id_int}
    except ValueError:
        # 문자열이면 회사명으로 검색
        where_clause = "c.name = :company_name"
        params = {"company_name": company_id}
        company_where_clause = "name = :company_name"
        company_params = {"company_name": company_id}
    
    # 회사명 먼저 조회
    company_query = text(f"SELECT name FROM company WHERE {company_where_clause} LIMIT 1")
    company_result = db.execute(company_query, company_params)
    company_row = company_result.first()
    company_name = company_row.name if company_row else company_id
    
    # 해당 연도의 모든 공고와 스킬 데이터 조회
    query = text(f"""
        SELECT
            p.id AS post_id,
            p.posted_at,
            s.name AS skill_name
        FROM company c
        INNER JOIN post p ON c.id = p.company_id
        INNER JOIN post_skill ps ON p.id = ps.post_id
        INNER JOIN skill s ON ps.skill_id = s.id
        WHERE {where_clause}
          AND YEAR(p.posted_at) = :year
          AND p.posted_at IS NOT NULL
        ORDER BY p.posted_at, s.name
    """)
    
    params['year'] = year
    
    result = db.execute(query, params)
    rows = [dict(row._mapping) for row in result]
    
    if not rows:
        return {
            "company": company_name,
            "year": year,
            "trends": []
        }
    
    # DataFrame으로 변환
    df = pd.DataFrame(rows)
    df['posted_at'] = pd.to_datetime(df['posted_at'])
    
    # 전체 기간에서 상위 N개 스킬 찾기
    top_skills = df['skill_name'].value_counts().head(top_n).index.tolist()
    
    # 분기 계산 함수
    def get_quarter(date):
        if pd.isna(date):
            return None
        return f"{date.year} Q{(date.month - 1) // 3 + 1}"
    
    df['quarter'] = df['posted_at'].apply(get_quarter)
    
    # 현재 분기와 이전 분기만 필터링
    if len(df) > 0:
        quarters = sorted(df['quarter'].dropna().unique())
        if len(quarters) >= 2:
            # 최근 2개 분기만 선택
            quarters = quarters[-2:]
        elif len(quarters) == 1:
            quarters = quarters
        else:
            quarters = []
    else:
        quarters = []
    
    # 분기별 스킬별 공고 수 집계
    trends = []
    for quarter in quarters:
        quarter_df = df[df['quarter'] == quarter]
        skill_counts = {}
        
        # 상위 N개 스킬에 대해 카운트 (없으면 0)
        for skill in top_skills:
            count = len(quarter_df[quarter_df['skill_name'] == skill])
            skill_counts[skill] = count
        
        trends.append({
            "quarter": quarter,
            "skills": skill_counts
        })
    
    return {
        "company": company_name,
        "year": year,
        "trends": trends
    }