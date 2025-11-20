"""
주요 회사별 채용 활동 관련 DB CRUD
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from datetime import date
from typing import List, Tuple, Optional

from app.models.post import Post
from app.models.company import Company


def get_companies_by_keywords(
    db: Session,
    keywords: List[str]
) -> List[Tuple[int, str]]:
    """키워드로 회사 검색 (id, name 반환)"""
    keyword_filters = [
        Company.name.like(f'{keyword}%') 
        for keyword in keywords
    ]
    
    companies_query = db.query(
        Company.id,
        Company.name
    ).filter(
        or_(*keyword_filters)
    ).all()
    
    return companies_query


def get_companies_recruitment_daily(
    db: Session,
    company_ids: List[int],
    start_date: date,
    end_date: date
) -> List[Tuple[date, int, int]]:
    """일간 회사별 채용 공고 수 조회 (date, company_id, count) (posted_at NULL이면 crawled_at 사용)"""
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    query = db.query(
        func.date(effective_date).label('date'),
        Post.company_id,
        func.count(Post.id).label('count')
    ).filter(
        Post.company_id.in_(company_ids),
        effective_date >= start_date,
        effective_date <= end_date
    ).group_by(
        func.date(effective_date),
        Post.company_id
    ).order_by(
        func.date(effective_date),
        Post.company_id
    )
    
    return query.all()


def get_companies_recruitment_weekly(
    db: Session,
    company_ids: List[int],
    start_date: date,
    end_date: date
) -> List[Tuple[int, int, int, int]]:
    """주간 회사별 채용 공고 수 조회 (year, week, company_id, count) (posted_at NULL이면 crawled_at 사용)"""
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    query = db.query(
        func.year(effective_date).label('year'),
        func.week(effective_date, 1).label('week'),
        Post.company_id,
        func.count(Post.id).label('count')
    ).filter(
        Post.company_id.in_(company_ids),
        effective_date >= start_date,
        effective_date <= end_date
    ).group_by(
        func.year(effective_date),
        func.week(effective_date, 1),
        Post.company_id
    ).order_by(
        func.year(effective_date),
        func.week(effective_date, 1),
        Post.company_id
    )
    
    return query.all()


def get_companies_recruitment_monthly(
    db: Session,
    company_ids: List[int],
    start_date: date,
    end_date: date
) -> List[Tuple[int, int, int, int]]:
    """월간 회사별 채용 공고 수 조회 (year, month, company_id, count) (posted_at NULL이면 crawled_at 사용)"""
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    query = db.query(
        func.year(effective_date).label('year'),
        func.month(effective_date).label('month'),
        Post.company_id,
        func.count(Post.id).label('count')
    ).filter(
        Post.company_id.in_(company_ids),
        effective_date >= start_date,
        effective_date <= end_date
    ).group_by(
        func.year(effective_date),
        func.month(effective_date),
        Post.company_id
    ).order_by(
        func.year(effective_date),
        func.month(effective_date),
        Post.company_id
    )
    
    return query.all()

