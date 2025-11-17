"""
CRUD operations for Post model
"""
from typing import List, Optional
from sqlalchemy.orm import Session, joinedload

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
