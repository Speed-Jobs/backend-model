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
    """일간 채용 공고 수 조회"""
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
    
    return query.all()


def get_job_postings_weekly(
    db: Session,
    start_date: date,
    end_date: date
) -> List[Tuple[int, int, int]]:
    """주간 채용 공고 수 조회 (year, week, count)"""
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
    
    return query.all()


def get_job_postings_monthly(
    db: Session,
    start_date: date,
    end_date: date
) -> List[Tuple[int, int, int]]:
    """월간 채용 공고 수 조회 (year, month, count)"""
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
    
    return query.all()

