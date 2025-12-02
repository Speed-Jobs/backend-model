"""
채용 공고 수 추이 Service
"""
from datetime import date, datetime, timedelta
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from collections import defaultdict

from app.db.crud import db_recruit_counter
from app.schemas.schemas_recruit_counter import (
    JobPostingsTrendData, 
    TrendItem, 
    PeriodInfo, 
    TopCompanyInfo,
    SelectedCompanyInfo
)


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


def _format_period_daily(dt: date) -> str:
    """일간 period 포맷: '11/1'"""
    return f"{dt.month}/{dt.day}"


def _format_period_weekly(year: int, week: int) -> str:
    """주간 period 포맷: '9월 1주'"""
    # week를 월로 변환 (대략적)
    first_day = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
    month = first_day.month
    week_of_month = (first_day.day - 1) // 7 + 1
    return f"{month}월 {week_of_month}주"


def _format_period_monthly(year: int, month: int) -> str:
    """월간 period 포맷: '2025-01'"""
    return f"{year}-{month:02d}"


def get_job_postings_trend(
    db: Session,
    timeframe: str
) -> JobPostingsTrendData:
    """
    채용 공고 수 추이 조회
    
    Args:
        db: 데이터베이스 세션
        timeframe: 시간 단위 (daily, weekly, monthly)
    
    Returns:
        JobPostingsTrendData: 추이 데이터
    
    Note:
        날짜는 자동으로 계산됩니다:
        - daily: 최근 30일 (오늘부터 29일 전까지)
        - weekly: 최근 12주
        - monthly: 최근 11개월
    """
    # 기간 자동 계산
    start_date, end_date = _get_default_period(timeframe)
    
    # timeframe별 쿼리
    if timeframe == "daily":
        # 일간 CRUD 호출
        results = db_recruit_counter.get_job_postings_daily(db, start_date, end_date)
        trends = [
            TrendItem(
                period=_format_period_daily(row[0]),  # date
                count=row[1]  # count
            )
            for row in results
        ]
    
    elif timeframe == "weekly":
        # 주간 CRUD 호출
        results = db_recruit_counter.get_job_postings_weekly(db, start_date, end_date)
        trends = [
            TrendItem(
                period=_format_period_weekly(row[0], row[1]),  # year, week
                count=row[2]  # count
            )
            for row in results
        ]
    
    elif timeframe == "monthly":
        # 월간 CRUD 호출
        results = db_recruit_counter.get_job_postings_monthly(db, start_date, end_date)
        trends = [
            TrendItem(
                period=_format_period_monthly(row[0], row[1]),  # year, month
                count=row[2]  # count
            )
            for row in results
        ]
    
    else:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    
    return JobPostingsTrendData(
        timeframe=timeframe,
        period=PeriodInfo(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        ),
        trends=trends
    )


def get_job_postings_trend_with_companies(
    db: Session,
    timeframe: str,
    company_keyword: Optional[str] = None
) -> JobPostingsTrendData:
    """
    채용 공고 수 추이 조회 (전체 추이 + 상위 5개 회사 또는 특정 회사)
    
    Args:
        db: 데이터베이스 세션
        timeframe: 시간 단위 (daily, weekly, monthly)
        company_keyword: 회사 키워드 (None이면 전체 모드, 상위 5개 회사 포함)
                        지정하면 해당 회사 1개만 조회 (인사이트는 추후 추가)
    
    Returns:
        JobPostingsTrendData: 추이 데이터 (전체 모드일 때 top_companies 포함)
    
    Note:
        날짜는 자동으로 계산됩니다:
        - daily: 최근 30일 (오늘부터 29일 전까지)
        - weekly: 최근 12주
        - monthly: 최근 11개월
    """
    # 기간 자동 계산
    start_date, end_date = _get_default_period(timeframe)
    
    # 전체 추이 데이터 조회
    if timeframe == "daily":
        results = db_recruit_counter.get_job_postings_daily(db, start_date, end_date)
        trends = [
            TrendItem(
                period=_format_period_daily(row[0]),
                count=row[1]
            )
            for row in results
        ]
    elif timeframe == "weekly":
        results = db_recruit_counter.get_job_postings_weekly(db, start_date, end_date)
        trends = [
            TrendItem(
                period=_format_period_weekly(row[0], row[1]),
                count=row[2]
            )
            for row in results
        ]
    elif timeframe == "monthly":
        results = db_recruit_counter.get_job_postings_monthly(db, start_date, end_date)
        trends = [
            TrendItem(
                period=_format_period_monthly(row[0], row[1]),
                count=row[2]
            )
            for row in results
        ]
    else:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    
    # 회사 선택 모드에 따라 처리
    top_companies = []
    selected_company = None
    
    if company_keyword is None:
        # 전체 모드: 경쟁사 그룹에 속한 상위 5개 회사 조회
        top_companies_results = db_recruit_counter.get_top_competitor_companies_by_postings(
            db, start_date, end_date, top_n=9
        )
        top_companies = [
            TopCompanyInfo(
                company_id=row[0],
                company_name=row[1],
                total_count=row[2]
            )
            for row in top_companies_results
        ]
    else:
        # 특정 회사 선택 모드: 해당 회사 정보 조회
        company_result = db_recruit_counter.get_company_by_keyword(
            db, company_keyword, start_date, end_date
        )
        if company_result:
            selected_company = SelectedCompanyInfo(
                company_id=company_result[0],
                company_name=company_result[1],
                total_count=company_result[2]
            )
    
    return JobPostingsTrendData(
        timeframe=timeframe,
        period=PeriodInfo(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        ),
        trends=trends,
        top_companies=top_companies,
        selected_company=selected_company
    )

