"""
채용 공고 수 추이 관련 DB CRUD
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date
from typing import List, Tuple

from app.models.post import Post


def get_job_postings_daily(
    db: Session,
    start_date: date,
    end_date: date
) -> List[Tuple[date, int]]:
    """일간 채용 공고 수 조회 (posted_at NULL이면 crawled_at 사용)"""
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    query = db.query(
        func.date(effective_date).label('date'),
        func.count(Post.id).label('count')
    ).filter(
        effective_date >= start_date,
        effective_date <= end_date
    ).group_by(
        func.date(effective_date)
    ).order_by(
        func.date(effective_date)
    )
    
    return query.all()


def get_job_postings_weekly(
    db: Session,
    start_date: date,
    end_date: date
) -> List[Tuple[int, int, int]]:
    """주간 채용 공고 수 조회 (year, week, count) (posted_at NULL이면 crawled_at 사용)"""
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    query = db.query(
        func.year(effective_date).label('year'),
        func.week(effective_date, 1).label('week'),
        func.count(Post.id).label('count')
    ).filter(
        effective_date >= start_date,
        effective_date <= end_date
    ).group_by(
        func.year(effective_date),
        func.week(effective_date, 1)
    ).order_by(
        func.year(effective_date),
        func.week(effective_date, 1)
    )
    
    return query.all()


def get_job_postings_monthly(
    db: Session,
    start_date: date,
    end_date: date
) -> List[Tuple[int, int, int]]:
    """월간 채용 공고 수 조회 (year, month, count) (posted_at NULL이면 crawled_at 사용)"""
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    query = db.query(
        func.year(effective_date).label('year'),
        func.month(effective_date).label('month'),
        func.count(Post.id).label('count')
    ).filter(
        effective_date >= start_date,
        effective_date <= end_date
    ).group_by(
        func.year(effective_date),
        func.month(effective_date)
    ).order_by(
        func.year(effective_date),
        func.month(effective_date)
    )
    
    return query.all()

