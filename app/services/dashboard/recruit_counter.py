"""
채용 공고 수 추이 Service
"""
from datetime import date, datetime, timedelta
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from collections import defaultdict

from app.models.post import Post
from app.schemas.schemas_recruit_counter import JobPostingsTrendData, TrendItem, PeriodInfo


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
        # 일간: DATE(posted_at)로 그룹핑
        query = db.query(
            func.date(Post.posted_at).label('date'),
            func.count(Post.id).label('count')
        ).filter(
            Post.posted_at >= start_date,
            Post.posted_at <= end_date
        ).group_by(
            func.date(Post.posted_at)
        ).order_by(
            func.date(Post.posted_at)
        )
        
        results = query.all()
        trends = [
            TrendItem(
                period=_format_period_daily(row.date),
                count=row.count
            )
            for row in results
        ]
    
    elif timeframe == "weekly":
        # 주간: YEARWEEK로 그룹핑
        query = db.query(
            func.year(Post.posted_at).label('year'),
            func.week(Post.posted_at, 1).label('week'),
            func.count(Post.id).label('count')
        ).filter(
            Post.posted_at >= start_date,
            Post.posted_at <= end_date
        ).group_by(
            func.year(Post.posted_at),
            func.week(Post.posted_at, 1)
        ).order_by(
            func.year(Post.posted_at),
            func.week(Post.posted_at, 1)
        )
        
        results = query.all()
        trends = [
            TrendItem(
                period=_format_period_weekly(row.year, row.week),
                count=row.count
            )
            for row in results
        ]
    
    elif timeframe == "monthly":
        # 월간: YEAR-MONTH로 그룹핑
        query = db.query(
            func.year(Post.posted_at).label('year'),
            func.month(Post.posted_at).label('month'),
            func.count(Post.id).label('count')
        ).filter(
            Post.posted_at >= start_date,
            Post.posted_at <= end_date
        ).group_by(
            func.year(Post.posted_at),
            func.month(Post.posted_at)
        ).order_by(
            func.year(Post.posted_at),
            func.month(Post.posted_at)
        )
        
        results = query.all()
        trends = [
            TrendItem(
                period=_format_period_monthly(row.year, row.month),
                count=row.count
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

