"""
YoY Overheat Index 관련 DB CRUD
"""
from datetime import date
from typing import List, Tuple, Optional

from sqlalchemy import func, or_, extract, and_
from sqlalchemy.orm import Session

from app.models.recruitment_schedule import RecruitmentSchedule
from app.models.company import Company
from app.models.industry import Industry
from app.models.position import Position


def get_monthly_recruit_counts(
    db: Session,
    year: int,
    month: int,
    window_months: int,
    company_patterns: Optional[List[str]] = None,
) -> List[Tuple[int, int, int, str, int, str, int]]:
    """
    지정된 연도/월로부터 window_months만큼의 채용 일정 건수 집계

    Args:
        db: DB 세션
        year: 조회 연도
        month: 조회 월 (1~12)
        window_months: 집계할 개월 수 (1 또는 3)
        company_patterns: 회사명 패턴 리스트 (get_company_patterns()로 변환된 값)

    Returns:
        List of (year, month, position_id, position_name, industry_id, industry_name, count)

    recruit_schedule 테이블의 application_date JSON 컬럼:
        - 형식: [[start, end], ...] (예: [["2024-12-01", "2024-12-15"]])
        - 첫 번째 기간의 시작일을 기준으로 연/월 추출
    """
    # application_date JSON의 첫 번째 요소 [0][0] (시작일)을 추출
    # MySQL JSON_EXTRACT: application_date->'$[0][0]' → "2024-12-01" (문자열)
    # STR_TO_DATE로 날짜 변환 후 YEAR(), MONTH() 사용

    app_date_str = func.json_unquote(
        func.json_extract(RecruitmentSchedule.application_date, "$[0][0]")
    )
    app_year = extract("year", func.str_to_date(app_date_str, "%Y-%m-%d"))
    app_month = extract("month", func.str_to_date(app_date_str, "%Y-%m-%d"))

    # 윈도우 기간 계산
    target_months = []
    for i in range(window_months):
        target_year = year
        target_month = month - i
        while target_month < 1:
            target_month += 12
            target_year -= 1
        target_months.append((target_year, target_month))

    # 연/월 필터 조건 생성
    date_filters = [
        and_(app_year == ty, app_month == tm)
        for ty, tm in target_months
    ]

    query = (
        db.query(
            app_year.label("year"),
            app_month.label("month"),
            Position.id.label("position_id"),
            Position.name.label("position_name"),
            Industry.id.label("industry_id"),
            Industry.name.label("industry_name"),
            func.count(RecruitmentSchedule.schedule_id).label("count"),
        )
        .join(Industry, RecruitmentSchedule.industry_id == Industry.id)
        .join(Position, Industry.position_id == Position.id)
        .filter(
            RecruitmentSchedule.application_date.isnot(None),
            or_(*date_filters),
        )
    )

    # 회사 필터링
    if company_patterns:
        query = query.join(Company, RecruitmentSchedule.company_id == Company.id)
        query = query.filter(or_(*[Company.name.like(pattern) for pattern in company_patterns]))

    query = query.group_by(
        app_year,
        app_month,
        Position.id,
        Position.name,
        Industry.id,
        Industry.name,
    )

    return query.all()


def get_recruit_counts_for_period(
    db: Session,
    start_date: date,
    end_date: date,
    company_patterns: Optional[List[str]] = None,
) -> List[Tuple[int, str, int, str, int]]:
    """
    지정된 기간의 채용 일정 건수 집계 (직무/산업별)

    Args:
        db: DB 세션
        start_date: 시작일
        end_date: 종료일
        company_patterns: 회사명 패턴 리스트 (get_company_patterns()로 변환된 값)

    Returns:
        List of (position_id, position_name, industry_id, industry_name, count)
    """
    app_date_str = func.json_unquote(
        func.json_extract(RecruitmentSchedule.application_date, "$[0][0]")
    )
    app_date = func.str_to_date(app_date_str, "%Y-%m-%d")

    query = (
        db.query(
            Position.id.label("position_id"),
            Position.name.label("position_name"),
            Industry.id.label("industry_id"),
            Industry.name.label("industry_name"),
            func.count(RecruitmentSchedule.schedule_id).label("count"),
        )
        .join(Industry, RecruitmentSchedule.industry_id == Industry.id)
        .join(Position, Industry.position_id == Position.id)
        .filter(
            RecruitmentSchedule.application_date.isnot(None),
            app_date >= start_date,
            app_date <= end_date,
        )
    )

    if company_patterns:
        query = query.join(Company, RecruitmentSchedule.company_id == Company.id)
        query = query.filter(or_(*[Company.name.like(pattern) for pattern in company_patterns]))

    query = query.group_by(
        Position.id,
        Position.name,
        Industry.id,
        Industry.name,
    )

    return query.all()
