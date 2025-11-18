"""
Job Matching Router

API 엔드포인트:
- GET /job-matching/posts/{post_id}: 특정 채용공고 매칭
- GET /job-matching/posts: 여러 채용공고 배치 매칭
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from app.db.base import get_db
from app.models.post import Post
from app.models.post_skill import PostSkill
from app.core.job_matching.model_loader import get_job_matching_system
from app.core.job_matching.job_matching_system import JobMatchingSystem
from app.services.job_matching.job_matching_service import JobMatchingService
from app.core.job_matching.config import PPR_TOP_N, FINAL_TOP_K
from app.schemas.job_output_schema import (
    SingleJobMatchingResponse,
    BatchJobMatchingResponse,
)

router = APIRouter(
    prefix="/job-matching",
    tags=["job-matching"],
)


@router.get("/posts/{post_id}", response_model=SingleJobMatchingResponse)
def match_single_post(
    post_id: int,
    db: Session = Depends(get_db),
    ppr_top_n: int = Query(default=PPR_TOP_N, ge=1, le=50, description="PPR로 상위 N개 직무 추출"),
    final_top_k: int = Query(default=FINAL_TOP_K, ge=1, le=10, description="최종 반환할 매칭 결과 개수"),
):
    """
    특정 채용공고에 대한 직무 매칭 수행
    
    Args:
        post_id: 채용공고 ID
        db: 데이터베이스 세션
        ppr_top_n: PPR로 상위 N개 직무 추출 (기본값: 20)
        final_top_k: 최종 반환할 매칭 결과 개수 (기본값: 2)
        
    Returns:
        SingleJobMatchingResponse: 매칭 결과
        
    Raises:
        HTTPException 404: 채용공고를 찾을 수 없을 때
        HTTPException 400: 스킬 정보가 없을 때
        HTTPException 500: 매칭 처리 중 오류가 발생했을 때
    """
    # DB에서 Post 조회 (relationship eager load)
    post = (
        db.query(Post)
        .options(
            joinedload(Post.company),
            joinedload(Post.post_skills).joinedload(PostSkill.skill),
        )
        .filter(Post.id == post_id)
        .first()
    )
    
    if not post:
        raise HTTPException(status_code=404, detail=f"Post ID {post_id}를 찾을 수 없습니다.")
    
    try:
        # JobMatchingSystem 및 Service 초기화
        system = get_job_matching_system()
        service = JobMatchingService(system)
        
        # Service에서 매칭 수행
        result = service.match_post(post, ppr_top_n=ppr_top_n, final_top_k=final_top_k)
        return result
        
    except ValueError as e:
        # 스킬이 없는 경우
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # 매칭 처리 중 오류
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예상치 못한 오류가 발생했습니다: {str(e)}")


@router.get("/posts", response_model=BatchJobMatchingResponse)
def match_multiple_posts(
    db: Session = Depends(get_db),
    limit: int = Query(default=10, ge=1, le=100, description="조회할 채용공고 개수"),
    offset: int = Query(default=0, ge=0, description="시작 위치"),
    company_id: Optional[int] = Query(default=None, description="회사 ID 필터"),
    ppr_top_n: int = Query(default=PPR_TOP_N, ge=1, le=50, description="PPR로 상위 N개 직무 추출"),
    final_top_k: int = Query(default=FINAL_TOP_K, ge=1, le=10, description="최종 반환할 매칭 결과 개수"),
):
    """
    여러 채용공고에 대한 배치 직무 매칭 수행
    
    Args:
        db: 데이터베이스 세션
        limit: 조회할 채용공고 개수 (기본값: 10, 최대: 100)
        offset: 시작 위치 (기본값: 0)
        company_id: 회사 ID 필터 (선택적)
        ppr_top_n: PPR로 상위 N개 직무 추출 (기본값: 20)
        final_top_k: 최종 반환할 매칭 결과 개수 (기본값: 2)
        
    Returns:
        BatchJobMatchingResponse: 배치 매칭 결과
    """
    # DB에서 Post 목록 조회 (relationship eager load)
    query = (
        db.query(Post)
        .options(
            joinedload(Post.company),
            joinedload(Post.post_skills).joinedload(PostSkill.skill),
        )
    )
    
    # 회사 필터 적용
    if company_id is not None:
        query = query.filter(Post.company_id == company_id)
    
    # 페이징 적용
    posts = query.offset(offset).limit(limit).all()
    
    if not posts:
        # 빈 결과 반환
        return BatchJobMatchingResponse(
            total_count=0,
            success_count=0,
            results=[],
        )
    
    try:
        # JobMatchingSystem 및 Service 초기화
        system = get_job_matching_system()
        service = JobMatchingService(system)
        
        # Service에서 배치 매칭 수행
        result = service.match_posts(posts, ppr_top_n=ppr_top_n, final_top_k=final_top_k)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"배치 매칭 처리 중 오류가 발생했습니다: {str(e)}")

