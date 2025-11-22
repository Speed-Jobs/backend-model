"""
CRUD operations for Post model
"""
from typing import List, Optional
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_

from app.models.company import Company
from app.models.position import Position
from app.models.industry import Industry
from app.models.post import Post
from app.models.skill import Skill
from app.models.post_skill import PostSkill
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill

def get_posts(
    db: Session,
    skip: int = 0,
    limit: int = 30
) -> List[Post]:
    """
    Get posts with pagination

    Args:
        db: Database session
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return

    Returns:
        List of Post objects
    """
    return db.query(Post)\
        .options(joinedload(Post.company))\
        .offset(skip)\
        .limit(limit)\
        .all()


def get_post_by_id(db: Session, post_id: int) -> Optional[Post]:
    """
    Get a single post by ID

    Args:
        db: Database session
        post_id: Post ID

    Returns:
        Post object or None if not found
    """

    return db.query(Post)\
        .options(joinedload(Post.company))\
        .filter(Post.id == post_id)\
        .first()


def get_posts_by_company_id(
    db: Session,
    company_id: int,
    skip: int = 0,
    limit: int = 30
) -> List[Post]:
    """
    Get posts by company ID

    Args:
        db: Database session
        company_id: Company ID
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List of Post objects
    """
    return db.query(Post)\
        .options(joinedload(Post.company))\
        .filter(Post.company_id == company_id)\
        .offset(skip)\
        .limit(limit)\
        .all()


def get_posts_with_skills(
    db: Session,
    skip: int = 0,
    limit: int = 30
) -> List[Post]:
    """
    Get posts with their related skills

    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List of Post objects with skills loaded
    """
    posts = db.query(Post)\
        .options(joinedload(Post.company))\
        .offset(skip)\
        .limit(limit)\
        .all()

    # Load skills for each post
    for post in posts:
        post_skills = db.query(Skill)\
            .join(PostSkill, Skill.id == PostSkill.skill_id)\
            .filter(PostSkill.post_id == post.id)\
            .all()
        post.skills = post_skills

    return posts


def count_posts(db: Session) -> int:
    """
    Count total number of posts

    Args:
        db: Database session

    Returns:
        Total count of posts
    """
    return db.query(Post).count()


def get_posts_by_skill_name(
    db: Session,
    skill_name: str,
    skip: int = 0,
    limit: int = 30
) -> List[Post]:
    """
    Get posts that require a specific skill

    Args:
        db: Database session
        skill_name: Name of the skill
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List of Post objects
    """
    return db.query(Post)\
        .join(PostSkill, Post.id == PostSkill.post_id)\
        .join(Skill, PostSkill.skill_id == Skill.id)\
        .filter(Skill.name == skill_name)\
        .options(joinedload(Post.company))\
        .offset(skip)\
        .limit(limit)\
        .all()


def get_posts_by_competitor_groups(
    db: Session,
    company_groups: Optional[List[str]] = None
) -> List[Post]:
    """
    경쟁사 그룹별 Post 조회 (스킬 포함)
    
    Args:
        db: Database session
        company_groups: 회사 그룹 리스트 (None이면 전체 9개 그룹)
                       예: ["토스", "카카오", "네이버", "쿠팡", "라인"]
    
    Returns:
        List of Post objects with company and skills loaded
    """
    from app.db.crud.db_competitors_skills import COMPETITOR_GROUPS
    
    # company_groups가 없으면 전체 그룹
    if company_groups is None:
        company_groups = list(COMPETITOR_GROUPS.keys())
    
    # 모든 그룹의 조건 생성
    or_conditions = []
    
    for group_name in company_groups:
        if group_name in COMPETITOR_GROUPS:
            keywords = COMPETITOR_GROUPS[group_name]
            for keyword in keywords:
                # % 제거하고 LIKE 패턴 생성
                pattern = keyword.replace("%", "")
                or_conditions.append(Company.name.like(f"{pattern}%"))
    
    if not or_conditions:
        return []
    
    # Post 조회 (회사, 스킬 포함)
    posts = (
        db.query(Post)
        .join(Company, Post.company_id == Company.id)
        .filter(or_(*or_conditions))
        .options(
            joinedload(Post.company),
            joinedload(Post.post_skills).joinedload(PostSkill.skill),
        )
        .all()
    )
    
    return posts