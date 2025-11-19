"""
주요 회사별 채용 활동 Service
"""
from datetime import date, datetime, timedelta
from typing import Optional, List, Tuple, Dict
from sqlalchemy.orm import Session
from collections import defaultdict

from app.db.crud import db_competitor_recruit_counter
from app.schemas.schemas_competitor_recruit_counter import (
    RecruitmentActivityData,
    CompanyInfo,
    ActivityItem
)


# 기본 8개 주요 경쟁사 키워드 (시작 일치 검색)
DEFAULT_COMPANY_KEYWORDS = ["토스", "한화", "라인", "네이버", "카카오", "LG", "현대오토에버", "우아한"]

# 회사명 → 영문 key 매핑
COMPANY_NAME_TO_KEY = {
    "토스": "toss",
    "한화": "hanwha",
    "라인": "line",
    "네이버": "naver",
    "카카오": "kakao",
    "LG": "lg",
    "LG CNS": "lgcns",
    "현대오토에버": "hyundai_autoever",
    "우아한": "woowahan",
    "우아한형제들": "woowahan",
    "Toss": "toss",
    "Line": "line",
    "Hanwha": "hanwha",
    "Kakao": "kakao",
    "Naver": "naver",
    "Hyundai Autoever": "hyundai_autoever",
    "Woowahan": "woowahan",
}


def _get_default_period(timeframe: str) -> Tuple[date, date]:
    """자동 조회 기간 계산"""
    today = date.today()
    
    if timeframe == "daily":
        # 최근 30일 (1달)
        start_date = today - timedelta(days=29)
        end_date = today
    elif timeframe == "weekly":
        # 최근 12주
        start_date = today - timedelta(weeks=12)
        end_date = today
    elif timeframe == "monthly":
        # 최근 11개월
        start_date = today - timedelta(days=330)  # 11개월 ≈ 330일
        end_date = today
    else:
        # 기본값: 최근 30일
        start_date = today - timedelta(days=29)
        end_date = today
    
    return start_date, end_date


def _generate_company_key(name: str) -> str:
    """회사명을 영문 key로 변환"""
    # 매핑 테이블에서 찾기
    if name in COMPANY_NAME_TO_KEY:
        return COMPANY_NAME_TO_KEY[name]
    
    # 매핑 없으면 소문자로 변환
    return name.lower().replace(" ", "_")


def _format_period_daily(dt: date) -> str:
    """일간 period 포맷"""
    return f"{dt.month}/{dt.day}"


def _format_period_weekly(year: int, week: int) -> str:
    """주간 period 포맷"""
    first_day = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
    month = first_day.month
    week_of_month = (first_day.day - 1) // 7 + 1
    return f"{month}월 {week_of_month}주"


def _format_period_monthly(year: int, month: int) -> str:
    """월간 period 포맷"""
    return f"{year}-{month:02d}"


def get_companies_recruitment_activity(
    db: Session,
    timeframe: str,
    company_keywords: Optional[List[str]] = None
) -> RecruitmentActivityData:
    """
    주요 회사별 채용 활동 조회
    
    Args:
        db: 데이터베이스 세션
        timeframe: 시간 단위 (daily, weekly, monthly)
        company_keywords: 조회할 회사명 키워드 리스트 (회사명 시작 기준 매칭, 없으면 기본 8개)
    
    Returns:
        RecruitmentActivityData: 회사별 채용 활동 데이터
    
    Note:
        날짜는 자동으로 계산됩니다:
        - daily: 최근 30일 (오늘부터 29일 전까지)
        - weekly: 최근 12주
        - monthly: 최근 11개월
    """
    from sqlalchemy import or_
    
    # 기본값 설정
    if not company_keywords:
        company_keywords = DEFAULT_COMPANY_KEYWORDS
    
    # 기간 자동 계산
    start_date, end_date = _get_default_period(timeframe)
    
    # 회사 정보 조회 (CRUD 호출)
    companies_query = db_competitor_recruit_counter.get_companies_by_keywords(db, company_keywords)
    
    companies_info = []
    company_id_to_key = {}
    company_ids = []  # 조회된 회사들의 ID 리스트
    
    for row in companies_query:
        company_id, company_name = row[0], row[1]
        key = _generate_company_key(company_name)
        companies_info.append(
            CompanyInfo(
                id=company_id,
                name=company_name,
                key=key
            )
        )
        company_id_to_key[company_id] = key
        company_ids.append(company_id)
    
    # 회사가 없으면 빈 결과 반환
    if not company_ids:
        return RecruitmentActivityData(
            timeframe=timeframe,
            companies=companies_info,
            activities=[]
        )
    
    # timeframe별 쿼리
    if timeframe == "daily":
        # 일간 CRUD 호출
        results = db_competitor_recruit_counter.get_companies_recruitment_daily(
            db, company_ids, start_date, end_date
        )
        
        # 데이터 구조화
        period_counts = defaultdict(dict)
        for row in results:
            period = _format_period_daily(row[0])  # date
            company_key = company_id_to_key.get(row[1], f"company_{row[1]}")  # company_id
            period_counts[period][company_key] = row[2]  # count
        
        activities = [
            ActivityItem(period=period, counts=counts)
            for period, counts in sorted(period_counts.items())
        ]
    
    elif timeframe == "weekly":
        # 주간 CRUD 호출
        results = db_competitor_recruit_counter.get_companies_recruitment_weekly(
            db, company_ids, start_date, end_date
        )
        
        period_counts = defaultdict(dict)
        for row in results:
            period = _format_period_weekly(row[0], row[1])  # year, week
            company_key = company_id_to_key.get(row[2], f"company_{row[2]}")  # company_id
            period_counts[period][company_key] = row[3]  # count
        
        activities = [
            ActivityItem(period=period, counts=counts)
            for period, counts in sorted(period_counts.items())
        ]
    
    elif timeframe == "monthly":
        # 월간 CRUD 호출
        results = db_competitor_recruit_counter.get_companies_recruitment_monthly(
            db, company_ids, start_date, end_date
        )
        
        period_counts = defaultdict(dict)
        for row in results:
            period = _format_period_monthly(row[0], row[1])  # year, month
            company_key = company_id_to_key.get(row[2], f"company_{row[2]}")  # company_id
            period_counts[period][company_key] = row[3]  # count
        
        activities = [
            ActivityItem(period=period, counts=counts)
            for period, counts in sorted(period_counts.items())
        ]
    
    else:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    
    return RecruitmentActivityData(
        timeframe=timeframe,
        companies=companies_info,
        activities=activities
    )

