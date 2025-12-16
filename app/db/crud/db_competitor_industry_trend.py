"""
직군별 통계 관련 DB CRUD
"""
from datetime import date
from typing import List, Tuple, Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.models.post import Post
from app.models.company import Company
from app.models.industry import Industry
from app.models.position import Position


def get_job_role_counts(
    db: Session,
    start_date: date,
    end_date: date,
    company_patterns: Optional[List[str]] = None,
) -> List[Tuple[int, str, int, str, int]]:
    """
    기간 내 직군(포지션) / 산업(Industry)별 공고 수 집계

    Args:
        company_patterns: 회사명 패턴 리스트 (예: ["토스%", "토스뱅크%", ...])
                         COMPANY_GROUPS에서 변환된 패턴 리스트 사용

    Returns:
        List of (position_id, position_name, industry_id, industry_name, count)
    """
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)

    query = (
        db.query(
            Position.id.label("position_id"),
            Position.name.label("position_name"),
            Industry.id.label("industry_id"),
            Industry.name.label("industry_name"),
            func.count(Post.id).label("count"),
        )
        .join(Industry, Post.industry_id == Industry.id)
        .join(Position, Industry.position_id == Position.id)
        .join(Company, Post.company_id == Company.id)
        .filter(
            effective_date >= start_date,
            effective_date <= end_date,
        )
    )

    if company_patterns:
        # 여러 패턴을 OR 조건으로 검색
        query = query.filter(or_(*[Company.name.like(pattern) for pattern in company_patterns]))

    query = query.group_by(
        Position.id,
        Position.name,
        Industry.id,
        Industry.name,
    )

    return query.all()


