from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd


# 경쟁사 목록 상수
COMPETITOR_COMPANIES = [
    '현대오토에버', 'Coupang', '한화시스템템', '카카오', 'LINE', 
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
        GROUP BY c.id,
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
    year: Optional[int] = None,
    top_n: int = 10
) -> Dict:
    """회사별 상위 스킬 분기별 트렌드 조회
    
    Args:
        db: 데이터베이스 세션
        company_id: 회사 ID 또는 회사명
        year: 조회 연도 (None일 경우 현재 연도 기준 근 5개년)
        top_n: 상위 N개 스킬
    """
    
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
    
    # year가 None이면 현재 연도 기준 근 5개년
    if year is None:
        current_year = datetime.now().year
        years = list(range(current_year - 4, current_year + 1))  # 5개년 (예: 2021~2025)
        # SQL injection 방지를 위해 숫자만 사용하므로 안전하게 포맷팅
        years_str = ','.join(map(str, years))
        year_condition = f"YEAR(p.posted_at) IN ({years_str})"
    else:
        years = [year]
        year_condition = "YEAR(p.posted_at) = :year"
        params['year'] = year
    
    # 해당 연도(들)의 모든 공고와 스킬 데이터 조회
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
          AND {year_condition}
          AND p.posted_at IS NOT NULL
        ORDER BY p.posted_at, s.name
    """)
    
    result = db.execute(query, params)
    rows = [dict(row._mapping) for row in result]
    
    if not rows:
        return {
            "company": company_name,
            "year": year,
            "years": years if year is None else None,
            "trends": [],
            "skill_frequencies": {} if year is None else None
        }
    
    # DataFrame으로 변환
    df = pd.DataFrame(rows)
    df['posted_at'] = pd.to_datetime(df['posted_at'])
    
    # year가 None이면 (근 5개년 조회 시) 각 연도별 상위 N개 스킬 빈도수 반환
    if year is None:
        # 연도별 스킬 빈도수 계산
        yearly_skill_frequencies = {}
        
        for y in years:
            year_df = df[df['posted_at'].dt.year == y]
            
            if len(year_df) > 0:
                # 해당 연도의 상위 N개 스킬 빈도수 계산
                skill_counts = year_df['skill_name'].value_counts().head(top_n)
                yearly_skill_frequencies[str(y)] = skill_counts.to_dict()
            else:
                # 해당 연도에 데이터가 없으면 빈 딕셔너리
                yearly_skill_frequencies[str(y)] = {}
        
        return {
            "company": company_name,
            "year": year,
            "years": years,
            "trends": [],
            "skill_frequencies": yearly_skill_frequencies
        }
    
    # year가 지정된 경우 현재 분기와 직전 분기의 스킬 빈도수 반환
    # 분기 계산 함수
    def get_quarter(date):
        if pd.isna(date):
            return None
        return (date.year, (date.month - 1) // 3 + 1)  # (연도, 분기) 튜플로 반환
    
    # 현재 분기와 직전 분기 계산
    current_date = datetime.now()
    current_year = current_date.year
    current_quarter = (current_date.month - 1) // 3 + 1
    current_q_tuple = (current_year, current_quarter)
    
    # 직전 분기 계산
    if current_quarter == 1:
        previous_q_tuple = (current_year - 1, 4)
    else:
        previous_q_tuple = (current_year, current_quarter - 1)
    
    # 분기를 튜플로 추가
    df['quarter_tuple'] = df['posted_at'].apply(get_quarter)
    
    # 현재 분기와 직전 분기만 필터링
    target_quarters = [current_q_tuple, previous_q_tuple]
    df_filtered = df[df['quarter_tuple'].isin(target_quarters)].copy()
    
    if len(df_filtered) == 0:
        return {
            "company": company_name,
            "year": year,
            "years": None,
            "trends": [],
            "skill_frequencies": None
        }
    
    # 분기를 문자열 형식으로 변환
    def quarter_to_str(q_tuple):
        if pd.isna(q_tuple) or q_tuple is None:
            return None
        return f"{q_tuple[0]} Q{q_tuple[1]}"
    
    df_filtered['quarter'] = df_filtered['quarter_tuple'].apply(quarter_to_str)
    
    # 각 분기별로 상위 top_n개 스킬의 빈도수 계산
    trends = []
    for q_tuple in [current_q_tuple, previous_q_tuple]:
        quarter_str = quarter_to_str(q_tuple)
        quarter_df = df_filtered[df_filtered['quarter_tuple'] == q_tuple]
        
        if len(quarter_df) > 0:
            # 해당 분기의 상위 top_n개 스킬 빈도수 계산
            skill_counts_series = quarter_df['skill_name'].value_counts().head(top_n)
            skill_counts = skill_counts_series.to_dict()
            
            trends.append({
                "quarter": quarter_str,
                "skills": skill_counts
            })
        else:
            # 데이터가 없어도 분기 정보는 반환
            trends.append({
                "quarter": quarter_str,
                "skills": {}
            })
    
    return {
        "company": company_name,
        "year": year,
        "years": None,
        "trends": trends,
        "skill_frequencies": None
    }