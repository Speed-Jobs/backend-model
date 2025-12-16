"""
직군별 통계 관련 DB CRUD
"""
from datetime import date
from typing import List, Tuple, Optional

from sqlalchemy import func, or_, case, and_
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


def get_job_role_counts_optimized(
    db: Session,
    current_start: date,
    current_end: date,
    previous_start: date,
    previous_end: date,
    category: str,
    company_patterns: Optional[List[str]] = None,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    현재/이전 기간의 직군별 통계를 한 번의 쿼리로 조회 (성능 최적화)
    
    Args:
        category: Tech, Biz, BizSupporting 중 하나
        
    Returns:
        (current_data, previous_data)
        각각 List of (position_name, industry_name, count)
    """
    from app.models.position import PositionCategory
    
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)
    
    # 카테고리 매핑 (서비스 레이어의 문자열 -> DB Enum)
    category_map = {
        "Tech": PositionCategory.TECH,
        "Biz": PositionCategory.BIZ,
        "BizSupporting": PositionCategory.BIZ_SUPPORTING,
    }
    db_category = category_map.get(category)
    
    # 현재 기간 카운트
    current_count = func.sum(
        case(
            (and_(effective_date >= current_start, effective_date <= current_end), 1),
            else_=0
        )
    ).label("current_count")
    
    # 이전 기간 카운트
    previous_count = func.sum(
        case(
            (and_(effective_date >= previous_start, effective_date <= previous_end), 1),
            else_=0
        )
    ).label("previous_count")
    
    query = (
        db.query(
            Position.name.label("position_name"),
            Industry.name.label("industry_name"),
            current_count,
            previous_count,
        )
        .join(Industry, Post.industry_id == Industry.id)
        .join(Position, Industry.position_id == Position.id)
        .join(Company, Post.company_id == Company.id)
        .filter(
            or_(
                and_(effective_date >= current_start, effective_date <= current_end),
                and_(effective_date >= previous_start, effective_date <= previous_end),
            ),
            Position.category == db_category,  # DB의 category 컬럼 직접 사용 (매우 빠름!)
        )
    )
    
    if company_patterns:
        query = query.filter(or_(*[Company.name.like(pattern) for pattern in company_patterns]))
    
    query = query.group_by(
        Position.name,
        Industry.name,
    )
    
    # 결과를 분리
    results = query.all()
    current_data = [(r.position_name, r.industry_name, r.current_count) for r in results if r.current_count > 0]
    previous_data = [(r.position_name, r.industry_name, r.previous_count) for r in results if r.previous_count > 0]
    
    return current_data, previous_data


