"""
CRUD operations for RecruitmentSchedule model
"""
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_

from app.models.recruitment_schedule import RecruitmentSchedule
from app.models.company import Company
from app.config.company_groups import COMPANY_GROUPS, get_company_patterns

def get_recruitment_schedules_by_company(
    db: Session,
    company_id: int,
    experience: Optional[str] = None
) -> List[RecruitmentSchedule]:
    """
    특정 회사의 채용 일정을 조회합니다.
    
    Args:
        db: Database session
        company_id: 회사 ID
        experience: 경험 유형 ("신입" 또는 "경력", None이면 전체)
        
    Returns:
        List of RecruitmentSchedule objects
    """
    query = db.query(RecruitmentSchedule)\
        .options(
            joinedload(RecruitmentSchedule.company),
            joinedload(RecruitmentSchedule.post)
        )\
        .filter(RecruitmentSchedule.company_id == company_id)
    
    # experience 필터 추가
    if experience:
        query = query.join(RecruitmentSchedule.post)\
            .filter(RecruitmentSchedule.post.has(experience=experience))
    
    return query.all()

def get_recruitment_schedules(
    db: Session,
    company_ids: Optional[List[int]] = None,
    experience: Optional[str] = None
) -> List[RecruitmentSchedule]:
    """
    채용 일정을 조회합니다.
    
    Args:
        db: Database session
        company_ids: 회사 ID 리스트 (None이면 전체)
        experience: 경험 유형 ("신입" 또는 "경력", None이면 전체)
        
    Returns:
        List of RecruitmentSchedule objects
    """
    query = db.query(RecruitmentSchedule)\
        .options(
            joinedload(RecruitmentSchedule.company),
            joinedload(RecruitmentSchedule.post)
        )
    
    # company_ids 필터 추가
    if company_ids:
        query = query.filter(RecruitmentSchedule.company_id.in_(company_ids))
    
    # experience 필터 추가
    if experience:
        query = query.join(RecruitmentSchedule.post)\
            .filter(RecruitmentSchedule.post.has(experience=experience))
    
    return query.all()


def get_competitor_companies(
    db: Session,
    company_keywords: Optional[List[str]] = None
) -> List[Tuple[int, str]]:
    """
    경쟁사 회사 ID와 이름 조회
    
    Args:
        db: Database session
        company_keywords: 조회할 회사 키워드 리스트 (None이면 전체 경쟁사)
        
    Returns:
        List of (company_id, company_name) tuples
    """
    like_conditions = []
    
    if company_keywords:
        # 지정된 키워드만 조회
        for keyword in company_keywords:
            patterns = get_company_patterns(keyword)
            for pattern in patterns:
                like_conditions.append(Company.name.like(pattern))
    else:
        # 전체 경쟁사 조회
        for patterns in COMPANY_GROUPS.values():
            for pattern in patterns:
                like_conditions.append(Company.name.like(pattern))
    
    if not like_conditions:
        return []
    
    return db.query(Company.id, Company.name)\
        .filter(or_(*like_conditions))\
        .order_by(Company.id)\
        .all()
