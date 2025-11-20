from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import pandas as pd
# v2

# 경쟁사 그룹 및 키워드 매핑
COMPETITOR_GROUPS: Dict[str, List[str]] = {
    "토스": ["토스%", "토스뱅크%", "토스증권%", "비바리퍼블리카%", "AICC%"],
    "카카오": ["카카오%"],
    "한화시스템": ["한화시스템%", "한화시스템템%", "한화시스템/ICT%", "한화시스템·ICT%"],
    "현대오토에버": ["현대오토에버%"],
    "우아한형제들": ["우아한%", "배달의민족", "배민"],
    "쿠팡": ["쿠팡%", "Coupang%"],
    "라인": ["LINE%", "라인%"],
    "네이버": ["NAVER%", "네이버%"],
    "LG CNS": ["LG_CNS%", "LG CNS%"],
}

EFFECTIVE_POSTED_AT_SQL = "COALESCE(p.posted_at, p.crawled_at)"


def _normalize_like_pattern(keyword: str) -> str:
    """와일드카드가 없다면 자동으로 부분 일치 패턴으로 변환"""
    if any(char in keyword for char in ("%", "_")):
        return keyword
    return f"%{keyword}%"


def _build_like_clauses(
    keywords: List[str],
    alias: str,
    prefix: str
) -> Tuple[List[str], Dict[str, str]]:
    """키워드 리스트를 LIKE 절과 파라미터로 변환"""
    clauses: List[str] = []
    params: Dict[str, str] = {}
    for idx, keyword in enumerate(keywords):
        param_name = f"{prefix}_{idx}"
        clauses.append(f"{alias} LIKE :{param_name}")
        params[param_name] = _normalize_like_pattern(keyword)
    return clauses, params


def _build_competitor_group_case(alias: str = "c.name") -> Tuple[str, str, Dict[str, str]]:
    """경쟁사 그룹 CASE 절과 기본 WHERE 절 생성"""
    where_fragments: List[str] = []
    case_fragments: List[str] = []
    params: Dict[str, str] = {}
    for group_idx, (group_name, keywords) in enumerate(COMPETITOR_GROUPS.items()):
        clauses, clause_params = _build_like_clauses(
            keywords, alias, prefix=f"group_{group_idx}"
        )
        if not clauses:
            continue
        condition_sql = " OR ".join(clauses)
        where_fragments.append(f"({condition_sql})")
        case_fragments.append(f"WHEN {condition_sql} THEN '{group_name}'")
        params.update(clause_params)
    if not where_fragments:
        raise ValueError("COMPETITOR_GROUPS가 비어 있어 그룹 조건을 생성할 수 없습니다.")
    where_clause = " OR ".join(where_fragments)
    case_expression = f"CASE {' '.join(case_fragments)} ELSE {alias} END"
    return where_clause, case_expression, params


def _strip_wildcards(keyword: str) -> str:
    """LIKE 패턴 문자열에서 와일드카드를 제거"""
    return keyword.replace("%", "").replace("_", "").strip()


def _normalize_for_compare(value: str) -> str:
    """비교를 위한 문자열 정규화"""
    return value.replace(" ", "").lower()


def _get_group_by_company_name(company_name: str) -> Optional[str]:
    """회사명이 속한 그룹 키 반환"""
    normalized_name = _normalize_for_compare(company_name)
    for group_name, keywords in COMPETITOR_GROUPS.items():
        if _normalize_for_compare(group_name) == normalized_name:
            return group_name
        for keyword in keywords:
            base_keyword = _strip_wildcards(keyword)
            if base_keyword and _normalize_for_compare(base_keyword) in normalized_name:
                return group_name
    return None


def _build_group_filter_clause(
    group_name: str,
    alias: str = "c.name",
    prefix: str = "target_group"
) -> Tuple[str, Dict[str, str]]:
    """그룹명을 기반으로 WHERE 절 생성"""
    keywords = COMPETITOR_GROUPS.get(group_name)
    if not keywords:
        return "", {}
    clauses, clause_params = _build_like_clauses(
        keywords, alias, prefix=f"{prefix}_{group_name}"
    )
    return f"({' OR '.join(clauses)})", clause_params


def _build_company_condition_from_input(
    company_name: str,
    alias: str = "c.name",
    prefix: str = "target_company"
) -> Tuple[str, Dict[str, str]]:
    """입력된 회사명(또는 그룹명)에 맞는 WHERE 절 생성"""
    group_name = _get_group_by_company_name(company_name)
    if group_name:
        return _build_group_filter_clause(group_name, alias=alias, prefix=prefix)
    clauses, clause_params = _build_like_clauses(
        [company_name], alias, prefix=prefix
    )
    return f"({' OR '.join(clauses)})", clause_params

def get_competitors_skill_diversity_all(db: Session) -> List[Dict]:
    """전체 경쟁사별 스킬 다양성 조회 (그룹 단위 집계)"""
    
    where_clause, group_case_expr, params = _build_competitor_group_case()
    
    query = text(f"""
        SELECT 
            {group_case_expr} AS company,
            COUNT(DISTINCT s.id) AS skills
        FROM company c
        INNER JOIN post p ON c.id = p.company_id
        INNER JOIN post_skill ps ON p.id = ps.post_id
        INNER JOIN skill s ON ps.skill_id = s.id
        WHERE {where_clause}
        GROUP BY company
        ORDER BY skills DESC, company
    """)
    
    result = db.execute(query, params)
    return [{"company": row.company, "skills": row.skills} for row in result]


def get_competitors_skill_diversity_by_year(db: Session, year: int) -> List[Dict]:
    """연도별 경쟁사별 스킬 다양성 조회 (그룹 단위 집계)"""
    
    where_clause, group_case_expr, params = _build_competitor_group_case()
    params['year'] = year
    
    query = text(f"""
        SELECT 
            {group_case_expr} AS company,
            COUNT(DISTINCT s.id) AS skills
        FROM company c
        INNER JOIN post p ON c.id = p.company_id
        INNER JOIN post_skill ps ON p.id = ps.post_id
        INNER JOIN skill s ON ps.skill_id = s.id
        WHERE ({where_clause})
          AND YEAR({EFFECTIVE_POSTED_AT_SQL}) = :year
        GROUP BY company
        ORDER BY skills DESC, company
    """)
    
    result = db.execute(query, params)
    return [{"company": row.company, "skills": row.skills} for row in result]


def get_competitors_posts_with_skills(
    db: Session, 
    company_name: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 100
) -> List[Dict]:
    """경쟁사별 공고 및 스킬 상세 조회"""
    
    base_where_clause, _, params = _build_competitor_group_case()
    
    additional_where = []
    
    if company_name:
        company_clause, clause_params = _build_company_condition_from_input(
            company_name, alias="c.name", prefix="target_company"
        )
        additional_where.append(company_clause)
        params.update(clause_params)
    
    if year:
        additional_where.append(f"YEAR({EFFECTIVE_POSTED_AT_SQL}) = :year")
        params['year'] = year
    
    where_clause = base_where_clause
    if additional_where:
        where_clause = f"({where_clause}) AND " + " AND ".join(additional_where)
    
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
        ORDER BY c.name, {EFFECTIVE_POSTED_AT_SQL} DESC
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
    group_name = None
    try:
        company_id_int = int(company_id)
        where_clause = "c.id = :company_id"
        params = {"company_id": company_id_int}
        company_where_clause = "id = :company_id"
        company_params = {"company_id": company_id_int}
    except ValueError:
        group_name = _get_group_by_company_name(company_id)
        if group_name:
            where_clause, params = _build_group_filter_clause(
                group_name, alias="c.name", prefix="trend_group"
            )
            company_where_clause = None
            company_params = None
        else:
            # 문자열이면 회사명으로 검색
            where_clause = "c.name = :company_name"
            params = {"company_name": company_id}
            company_where_clause = "name = :company_name"
            company_params = {"company_name": company_id}
    
    # 회사명 먼저 조회 (그룹 매칭 시에는 그룹명 사용)
    if group_name:
        company_name = group_name
    else:
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
        year_condition = f"YEAR({EFFECTIVE_POSTED_AT_SQL}) IN ({years_str})"
    else:
        years = [year]
        year_condition = f"YEAR({EFFECTIVE_POSTED_AT_SQL}) = :year"
        params['year'] = year
    
    # 해당 연도(들)의 모든 공고와 스킬 데이터 조회
    query = text(f"""
        SELECT
            p.id AS post_id,
            {EFFECTIVE_POSTED_AT_SQL} AS effective_posted_at,
            s.name AS skill_name
        FROM company c
        INNER JOIN post p ON c.id = p.company_id
        INNER JOIN post_skill ps ON p.id = ps.post_id
        INNER JOIN skill s ON ps.skill_id = s.id
        WHERE {where_clause}
          AND {year_condition}
          AND {EFFECTIVE_POSTED_AT_SQL} IS NOT NULL
        ORDER BY {EFFECTIVE_POSTED_AT_SQL}, s.name
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
    df['effective_posted_at'] = pd.to_datetime(df['effective_posted_at'])
    
    # year가 None이면 (근 5개년 조회 시) 각 연도별 상위 N개 스킬 빈도수 반환
    if year is None:
        # 연도별 스킬 빈도수 계산
        yearly_skill_frequencies = {}
        
        for y in years:
            year_df = df[df['effective_posted_at'].dt.year == y]
            
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
    df['quarter_tuple'] = df['effective_posted_at'].apply(get_quarter)
    
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