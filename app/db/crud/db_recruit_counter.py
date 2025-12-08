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
    그룹명인 경우 COMPANY_GROUPS의 모든 키워드로 검색
    
    Args:
        db: 데이터베이스 세션
        keyword: 회사명 키워드 (대소문자 무시)
        start_date: 시작일
        end_date: 종료일
    
    Returns:
        List[Tuple[int, str, int]]: (company_id, company_name, total_count)
                                     company_name은 입력받은 keyword로 반환됨
                                     공고 수가 많은 순으로 정렬
    
    Examples:
        >>> get_companies_by_keyword(db, "line", start, end)
        [(25, "line", 24), (30, "line", 8), ...]
    """
    from app.config.company_groups import COMPANY_GROUPS
    from sqlalchemy import or_
    
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    # 그룹명인지 확인
    if keyword in COMPANY_GROUPS:
        # 그룹의 모든 키워드로 검색
        keywords = COMPANY_GROUPS[keyword]
        keyword_filters = []
        for kw in keywords:
            # % 제거하고 앞뒤에 % 추가
            clean_kw = kw.rstrip('%').lstrip('%')
            keyword_filters.append(func.upper(Company.name).like(f'%{clean_kw.upper()}%'))
        filter_condition = or_(*keyword_filters)
    else:
        # 단일 키워드 검색
        filter_condition = func.upper(Company.name).like(f'%{keyword.upper()}%')
    
    # 쿼리 실행
    results = db.query(
        Company.id,
        Company.name,
        func.count(Post.id).label('total_count')
    ).join(
        Post, Company.id == Post.company_id
    ).filter(
        filter_condition,
        effective_date >= start_date,
        effective_date <= end_date
    ).group_by(
        Company.id,
        Company.name
    ).order_by(
        func.count(Post.id).desc()
    ).all()
    
    # company_name을 입력받은 keyword로 변환하여 반환
    return [(row[0], keyword, row[2]) for row in results]


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