"""
HHI Concentration Index 관련 DB CRUD
"""
from datetime import date
from typing import List, Tuple, Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.models.post import Post
from app.models.company import Company
from app.models.industry import Industry
from app.models.position import Position


def get_position_recruit_counts(
    db: Session,
    start_date: date,
    end_date: date,
    company_patterns: Optional[List[str]] = None,
) -> List[Tuple[int, str, int]]:
    """
    지정된 기간의 직무별 채용 공고 건수 집계 (HHI 계산용)

    Post 테이블을 사용하여 posted_at (없으면 crawled_at) 기준으로 집계합니다.

    Args:
        db: DB 세션
        start_date: 시작일
        end_date: 종료일
        company_patterns: 회사명 패턴 리스트 (get_company_patterns()로 변환된 값)

    Returns:
        List of (position_id, position_name, count)
    """
    # posted_at이 없으면 crawled_at 사용
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)

    query = (
        db.query(
            Position.id.label("position_id"),
            Position.name.label("position_name"),
            func.count(Post.id).label("count"),  # Post.id를 카운트
        )
        .join(Industry, Post.industry_id == Industry.id)
        .join(Position, Industry.position_id == Position.id)
        .filter(
            Post.is_deleted == False,  # 삭제되지 않은 공고만
            effective_date >= start_date,
            effective_date <= end_date,
        )
    )

    if company_patterns:
        query = query.join(Company, Post.company_id == Company.id)
        query = query.filter(or_(*[Company.name.like(pattern) for pattern in company_patterns]))

    query = query.group_by(
        Position.id,
        Position.name,
    )

    return query.all()


def get_industry_recruit_counts(
    db: Session,
    start_date: date,
    end_date: date,
    company_patterns: Optional[List[str]] = None,
) -> List[Tuple[int, str, int]]:
    """
    지정된 기간의 산업별 채용 공고 건수 집계 (HHI 계산용)

    Post 테이블을 사용하여 posted_at (없으면 crawled_at) 기준으로 집계합니다.

    Args:
        db: DB 세션
        start_date: 시작일
        end_date: 종료일
        company_patterns: 회사명 패턴 리스트 (get_company_patterns()로 변환된 값)

    Returns:
        List of (industry_id, industry_name, count)
    """
    # posted_at이 없으면 crawled_at 사용
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)

    query = (
        db.query(
            Industry.id.label("industry_id"),
            Industry.name.label("industry_name"),
            func.count(Post.id).label("count"),  # Post.id를 카운트
        )
        .join(Industry, Post.industry_id == Industry.id)
        .filter(
            Post.is_deleted == False,  # 삭제되지 않은 공고만
            effective_date >= start_date,
            effective_date <= end_date,
        )
    )

    if company_patterns:
        query = query.join(Company, Post.company_id == Company.id)
        query = query.filter(or_(*[Company.name.like(pattern) for pattern in company_patterns]))

    query = query.group_by(
        Industry.id,
        Industry.name,
    )

    return query.all()
