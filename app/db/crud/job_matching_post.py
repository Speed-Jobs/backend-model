"""
Job Matching 관련 Post CRUD operations

Job Matching 전용 Post 조회 함수들
- 스킬 정보 포함 조회
- 회사명 필터링 지원
- 경쟁사 그룹 필터링 지원
"""
from typing import List, Optional
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_

from app.models.company import Company
from app.models.post import Post
from app.models.post_skill import PostSkill


def get_post_by_id_with_skills(
    db: Session, 
    post_id: int
) -> Optional[Post]:
    """
    ID로 Post 조회 (회사, 스킬 포함 - eager load)
    
    Args:
        db: Database session
        post_id: Post ID
        
    Returns:
        Post object with company and skills loaded, or None
    """
    return (
        db.query(Post)
        .options(
            joinedload(Post.company),
            joinedload(Post.post_skills).joinedload(PostSkill.skill),
        )
        .filter(Post.id == post_id)
        .first()
    )


def get_posts_with_filters(
    db: Session,
    limit: int = 10,
    offset: int = 0,
    company_id: Optional[int] = None,
    company_name: Optional[str] = None,
) -> List[Post]:
    """
    필터링된 Post 목록 조회 (회사, 스킬 포함)
    
    Args:
        db: Database session
        limit: 조회할 개수
        offset: 시작 위치
        company_id: 회사 ID 필터 (우선순위 높음)
        company_name: 회사명 필터 (부분 일치, 그룹명 지원)
                     예: "라인", "토스", "카카오", "한화시스템"
    
    Returns:
        List of Post objects with company and skills loaded
    """
    # 기본 쿼리 (Company join - 회사명 필터링을 위해 필요)
    query = (
        db.query(Post)
        .join(Company, Post.company_id == Company.id)
        .options(
            joinedload(Post.company),
            joinedload(Post.post_skills).joinedload(PostSkill.skill),
        )
    )
    
    # 회사 필터 적용 (company_id가 우선)
    if company_id is not None:
        query = query.filter(Post.company_id == company_id)
    elif company_name is not None:
        # 회사명 필터링 (그룹명도 지원 - 대시보드와 동일한 로직)
        from app.config.company_groups import COMPANY_GROUPS
        
        # 입력된 회사명이 경쟁사 그룹에 속하는지 확인
        company_name_normalized = company_name.lower().strip().replace(" ", "")
        group_name = None
        
        # COMPANY_GROUPS에서 매칭되는 그룹 찾기
        for group_key, patterns in COMPANY_GROUPS.items():
            # 그룹 키와 직접 매칭
            group_key_normalized = group_key.lower().replace(" ", "")
            if group_key_normalized == company_name_normalized:
                group_name = group_key
                break
            
            # 패턴과 매칭 (양방향 확인)
            for pattern in patterns:
                # % 제거하고 비교
                clean_pattern = pattern.replace("%", "").replace("_", "").strip().lower()
                if clean_pattern:
                    # 양방향 매칭: 패턴이 입력에 포함되거나, 입력이 패턴에 포함되는 경우
                    if clean_pattern in company_name_normalized or company_name_normalized in clean_pattern:
                        group_name = group_key
                        break
            
            if group_name:
                break
        
        if group_name and group_name in COMPANY_GROUPS:
            # 그룹명인 경우: 해당 그룹의 모든 키워드로 OR 조건 생성
            # 예: "라인" 입력 시 → "LINE%", "라인%" 모두 검색 (LINE PAY, 라인프레쉬 등 포함)
            keywords = COMPANY_GROUPS[group_name]
            or_conditions = []
            for keyword in keywords:
                # % 제거하고 시작 일치 패턴 생성
                pattern = keyword.replace("%", "")
                or_conditions.append(Company.name.like(f"{pattern}%"))
            query = query.filter(or_(*or_conditions))
        else:
            # 단일 회사명인 경우: 시작 일치로 검색
            # 예: "한화시스템" 입력 시 → "한화시스템%" (한화시스템/ICT 포함)
            query = query.filter(Company.name.like(f"{company_name}%"))
    
    # 페이징 적용
    return query.offset(offset).limit(limit).all()


def get_posts_by_competitor_groups(
    db: Session,
    company_groups: Optional[List[str]] = None
) -> List[Post]:
    """
    경쟁사 그룹별 Post 조회 (스킬 포함)
    
    학습 데이터 로드용으로 사용됩니다.
    
    Args:
        db: Database session
        company_groups: 회사 그룹 리스트 (None이면 전체 9개 그룹)
                       예: ["토스", "카카오", "네이버", "쿠팡", "라인"]
    
    Returns:
        List of Post objects with company and skills loaded
    """
    from app.config.company_groups import COMPANY_GROUPS
    
    # company_groups가 없으면 전체 그룹
    if company_groups is None:
        company_groups = list(COMPANY_GROUPS.keys())
    
    # 모든 그룹의 조건 생성
    or_conditions = []
    
    for group_name in company_groups:
        if group_name in COMPANY_GROUPS:
            keywords = COMPANY_GROUPS[group_name]
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

