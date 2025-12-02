"""
채용 공고 수 추이 관련 DB CRUD - 최종 버전 (weekly, monthly만)
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import date
from typing import List, Optional, Tuple

from app.models.post import Post
from app.models.company import Company


def get_job_postings_weekly(
    db: Session,
    start_date: date,
    end_date: date
) -> List[Tuple[int, int, int]]:
    """주간 채용 공고 수 조회 (year, week, count)"""
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    return db.query(
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
    ).all()


def get_job_postings_monthly(
    db: Session,
    start_date: date,
    end_date: date
) -> List[Tuple[int, int, int]]:
    """월간 채용 공고 수 조회 (year, month, count)"""
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    return db.query(
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
    ).all()


def get_companies_by_keyword(
    db: Session,
    keyword: str,
    start_date: date,
    end_date: date
) -> List[Tuple[int, str, int]]:
    """
    키워드를 포함하는 모든 회사 조회
    
    Args:
        db: 데이터베이스 세션
        keyword: 회사명 키워드 (대소문자 무시)
        start_date: 시작일
        end_date: 종료일
    
    Returns:
        List[Tuple[int, str, int]]: (company_id, company_name, total_count)
                                     공고 수가 많은 순으로 정렬
    
    Examples:
        >>> get_companies_by_keyword(db, "네이버", start, end)
        [(25, "NAVER Cloud", 24), (30, "NAVER", 8), ...]
    """
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    return db.query(
        Company.id,
        Company.name,
        func.count(Post.id).label('total_count')
    ).join(
        Post, Company.id == Post.company_id
    ).filter(
        func.upper(Company.name).like(f'%{keyword.upper()}%'),
        effective_date >= start_date,
        effective_date <= end_date
    ).group_by(
        Company.id,
        Company.name
    ).order_by(
        func.count(Post.id).desc()
    ).all()


def get_company_by_keyword(
    db: Session,
    keyword: str,
    start_date: date,
    end_date: date
) -> Optional[Tuple[int, str, int]]:
    """
    [DEPRECATED] 키워드로 회사 조회 및 기간 내 공고 수 반환
    대신 get_companies_by_keyword 사용 권장
    """
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    query = db.query(
        Company.id,
        Company.name,
        func.count(Post.id).label('total_count')
    ).join(
        Post, Company.id == Post.company_id
    ).filter(
        Company.name.like(f'{keyword}%'),
        effective_date >= start_date,
        effective_date <= end_date
    ).group_by(
        Company.id,
        Company.name
    ).order_by(
        func.count(Post.id).desc()
    ).limit(1)
    
    result = query.first()
    return result