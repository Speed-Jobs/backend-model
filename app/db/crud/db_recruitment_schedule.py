"""
CRUD operations for RecruitmentSchedule model
"""
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_

from app.models.recruitment_schedule import RecruitmentSchedule
from app.models.company import Company
from app.models.post import Post
from app.models.industry import Industry
from app.config.company_groups import COMPANY_GROUPS, get_company_patterns

def get_recruitment_schedules(
    db: Session,
    company_patterns: Optional[List[str]] = None,
    experience: Optional[str] = None
) -> List[RecruitmentSchedule]:
    """
    채용 일정을 조회합니다.

    Args:
        db: Database session
        company_patterns: 회사명 패턴 리스트 (None이면 전체, 예: ["토스%", "카카오%"])
        experience: 경험 유형 ("신입" 또는 "경력", None이면 전체)

    Returns:
        List of RecruitmentSchedule objects
    """
    query = db.query(RecruitmentSchedule)\
        .options(
            joinedload(RecruitmentSchedule.company),
            joinedload(RecruitmentSchedule.post)
        )

    # company_patterns 필터 추가
    if company_patterns:
        query = query.join(Company, RecruitmentSchedule.company_id == Company.id)\
            .filter(or_(*[Company.name.like(pattern) for pattern in company_patterns]))

    # experience 필터 추가
    if experience:
        query = query.join(RecruitmentSchedule.post)\
            .filter(RecruitmentSchedule.post.has(experience=experience))

    return query.all()


def get_competitor_companies(
    db: Session,
    company_keywords: Optional[List[str]] = None
) -> List[Tuple[int, str]]:
    """
    경쟁사 회사 ID와 이름 조회

    Args:
        db: Database session
        company_keywords: 조회할 회사 키워드 리스트 (None이면 전체 경쟁사)

    Returns:
        List of (company_id, company_name) tuples
    """
    like_conditions = []

    if company_keywords:
        # 지정된 키워드만 조회
        for keyword in company_keywords:
            patterns = get_company_patterns(keyword)
            for pattern in patterns:
                like_conditions.append(Company.name.like(pattern))
    else:
        # 전체 경쟁사 조회
        for patterns in COMPANY_GROUPS.values():
            for pattern in patterns:
                like_conditions.append(Company.name.like(pattern))

    if not like_conditions:
        return []

    return db.query(Company.id, Company.name)\
        .filter(or_(*like_conditions))\
        .order_by(Company.id)\
        .all()


def get_competition_intensity_analysis(
    db: Session,
    start_date: str,
    end_date: str,
    type_filter: Optional[str] = None
) -> dict:
    """
    경쟁 강도 분석 - 날짜별 채용 중인 경쟁사 수 계산

    Args:
        db: Database session
        start_date: 분석 시작일 (YYYY-MM-DD)
        end_date: 분석 종료일 (YYYY-MM-DD)
        type_filter: 채용 유형 필터 ("신입", "경력", None)

    Returns:
        {
            "status": 200,
            "code": "SUCCESS",
            "message": "경쟁 강도 분석 성공",
            "data": {
                "period": {"start_date": "...", "end_date": "..."},
                "max_overlaps": int,
                "daily_intensity": [
                    {
                        "date": "...",
                        "overlap_count": int,
                        "companies": [
                            {"company_id": int, "company_name": str}
                        ]
                    }
                ]
            }
        }
    """
    from sqlalchemy import text
    from collections import defaultdict

    # 1. 경쟁사 패턴 생성
    company_conditions = []
    for patterns in COMPANY_GROUPS.values():
        for pattern in patterns:
            escaped_pattern = pattern.replace("'", "''")
            company_conditions.append(f"c.name LIKE '{escaped_pattern}'")

    if not company_conditions:
        return {
            "status": 200,
            "code": "SUCCESS",
            "message": "경쟁 강도 분석 성공",
            "data": {
                "period": {"start_date": start_date, "end_date": end_date},
                "max_overlaps": 0,
                "daily_intensity": []
            }
        }

    company_where = " OR ".join(company_conditions)

    # 2. 신입/경력 필터링 조건 생성
    experience_filter = ""
    if type_filter == "신입":
        experience_filter = "AND p.experience = '신입'"
    elif type_filter == "경력":
        experience_filter = "AND p.experience = '경력'"

    # 3. Raw SQL 쿼리 실행
    sql_query = text(f"""
        SELECT
            dr.date,
            rs.company_id,
            rs.company_name
        FROM (
            SELECT DATE_ADD(:start_date, INTERVAL n DAY) AS date
            FROM (
                SELECT a.N + b.N * 10 AS n
                FROM
                    (SELECT 0 AS N UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
                     UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) a,
                    (SELECT 0 AS N UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
                     UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) b
            ) numbers
            WHERE DATE_ADD(:start_date, INTERVAL n DAY) <= :end_date
        ) dr
        INNER JOIN (
            SELECT
                rs.schedule_id,
                rs.company_id,
                c.name AS company_name,
                LEAST(
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.application_date, '$[0][0]')), '9999-12-31'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.document_screening_date, '$[0][0]')), '9999-12-31'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.first_interview, '$[0][0]')), '9999-12-31'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.second_interview, '$[0][0]')), '9999-12-31'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.join_date, '$[0][0]')), '9999-12-31')
                ) AS start_date,
                GREATEST(
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.application_date, '$[0][1]')), '0000-01-01'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.document_screening_date, '$[0][1]')), '0000-01-01'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.first_interview, '$[0][1]')), '0000-01-01'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.second_interview, '$[0][1]')), '0000-01-01'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.join_date, '$[0][1]')), '0000-01-01')
                ) AS end_date
            FROM recruit_schedule rs
            INNER JOIN company c ON rs.company_id = c.id
            INNER JOIN post p ON rs.post_id = p.id
            WHERE
                ({company_where})
                AND p.experience IS NOT NULL
                {experience_filter}
        ) rs
            ON dr.date BETWEEN rs.start_date AND rs.end_date
            AND rs.start_date != '9999-12-31'
            AND rs.end_date != '0000-01-01'
            AND rs.end_date >= :start_date
            AND rs.start_date <= :end_date
        ORDER BY dr.date, rs.company_id
    """)

    # 4. SQL 실행
    result = db.execute(sql_query, {"start_date": start_date, "end_date": end_date})
    rows = result.fetchall()

    # 5. 결과 변환 - 날짜별로 그룹화하여 회사 리스트 생성
    daily_companies = defaultdict(set)
    company_info = {}

    for row in rows:
        date_str = str(row.date)
        company_id = row.company_id
        company_name = row.company_name

        daily_companies[date_str].add(company_id)
        company_info[company_id] = company_name

    # 6. 날짜별 결과 리스트 생성
    daily_intensity = []
    max_overlaps = 0

    for date_str in sorted(daily_companies.keys()):
        company_ids = sorted(daily_companies[date_str])
        overlap_count = len(company_ids)
        max_overlaps = max(max_overlaps, overlap_count)

        companies = [
            {
                "company_id": company_id,
                "company_name": company_info[company_id]
            }
            for company_id in company_ids
        ]

        daily_intensity.append({
            "date": date_str,
            "overlap_count": overlap_count,
            "companies": companies
        })

    # 7. 결과 반환
    return {
        "status": 200,
        "code": "SUCCESS",
        "message": "경쟁 강도 분석 성공",
        "data": {
            "period": {"start_date": start_date, "end_date": end_date},
            "max_overlaps": max_overlaps,
            "daily_intensity": daily_intensity
        }
    }
