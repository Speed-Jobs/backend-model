"""
CRUD operations for RecruitmentSchedule model
"""
from typing import List, Optional

from sqlalchemy.orm import Session, joinedload

from app.models.recruitment_schedule import RecruitmentSchedule

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

