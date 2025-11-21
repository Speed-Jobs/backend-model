"""
Job Matching Service Layer

비즈니스 로직 처리:
- DB Post 모델 → NewJobPosting 변환
- JobMatchingSystem 호출
- JobMatchResult → Schema 변환
"""
from typing import List, Optional
import logging

from sqlalchemy.orm import Session

from app.models.post import Post
from app.core.job_matching.job_matching_system import (
    JobMatchingSystem,
    NewJobPosting,
    JobMatchResult as InternalJobMatchResult,
)
from app.schemas.job_output_schema import (
    JobMatchResult,
    SingleJobMatchingResponse,
    BatchJobMatchingResponse,
)

logger = logging.getLogger(__name__)


class JobMatchingService:
    """Job Matching 비즈니스 로직 처리"""

    def __init__(self, job_matching_system: JobMatchingSystem):
        self.system = job_matching_system

    def post_to_new_job_posting(self, post: Post) -> NewJobPosting:
        """
        DB Post 모델을 NewJobPosting으로 변환
        
        Args:
            post: DB Post 모델 인스턴스
            
        Returns:
            NewJobPosting: JobMatchingSystem이 필요로 하는 형식
        """
        # PostSkill 관계에서 Skill.name 리스트 추출
        skills = []
        if post.post_skills:
            skills = [ps.skill.name for ps in post.post_skills if ps.skill]

        # Company 관계에서 company name 추출
        company_name = post.company.name if post.company else "Unknown"

        return NewJobPosting(
            posting_id=str(post.id),
            company=company_name,
            title=post.title,
            skills=skills,
            url=post.source_url or "",
            description=post.description or "",
        )

    def internal_result_to_schema(
        self, result: InternalJobMatchResult
    ) -> JobMatchResult:
        """
        내부 JobMatchResult (dataclass)를 Pydantic Schema로 변환
        
        Args:
            result: InternalJobMatchResult (dataclass)
            
        Returns:
            JobMatchResult: Pydantic Schema
        """
        return JobMatchResult(
            job_name=result.job_name,
            industry=result.industry,
            final_score=result.final_score,
            jaccard_score=result.jaccard_score,
            embedding_score=result.sbert_score,  # sbert_score를 embedding_score로 매핑
            pagerank_score=result.pagerank_score,
            matching_skills=result.matching_skills,
            job_definition=result.job_definition,
            reason=result.reason,
        )

    def match_post(
        self,
        post: Post,
        ppr_top_n: int = 20,
        final_top_k: int = 2,
    ) -> SingleJobMatchingResponse:
        """
        단일 Post에 대한 매칭 수행
        
        Args:
            post: DB Post 모델 인스턴스
            ppr_top_n: PPR로 상위 N개 직무 추출
            final_top_k: 최종 반환할 매칭 결과 개수
            
        Returns:
            SingleJobMatchingResponse: 매칭 결과 응답
            
        Raises:
            ValueError: 스킬이 없는 경우
            RuntimeError: 매칭 실패 시
        """
        # 1. Post → NewJobPosting 변환
        new_posting = self.post_to_new_job_posting(post)

        # 스킬이 없으면 매칭 불가
        if not new_posting.skills:
            raise ValueError("스킬 정보가 없어 매칭할 수 없습니다.")

        try:
            # 2. JobMatchingSystem 호출
            internal_results = self.system.match_new_job(
                new_posting,
                ppr_top_n=ppr_top_n,
                final_top_k=final_top_k,
            )

            # 3. InternalJobMatchResult → Schema 변환
            matches = [
                self.internal_result_to_schema(result) for result in internal_results
            ]

            # 4. SingleJobMatchingResponse 생성
            return SingleJobMatchingResponse(
                posting_id=str(post.id),
                company=new_posting.company,
                title=new_posting.title,
                url=new_posting.url,
                matches=matches,
            )

        except Exception as e:
            logger.error(f"Post {post.id} 매칭 실패: {e}", exc_info=True)
            raise RuntimeError(f"매칭 처리 중 오류가 발생했습니다: {str(e)}")

    def match_posts(
        self,
        posts: List[Post],
        ppr_top_n: int = 20,
        final_top_k: int = 2,
    ) -> BatchJobMatchingResponse:
        """
        여러 Post에 대한 배치 매칭 수행
        
        Args:
            posts: DB Post 모델 인스턴스 리스트
            ppr_top_n: PPR로 상위 N개 직무 추출
            final_top_k: 최종 반환할 매칭 결과 개수
            
        Returns:
            BatchJobMatchingResponse: 배치 매칭 결과 응답
        """
        total_count = len(posts)
        success_count = 0
        results = []

        for post in posts:
            try:
                result = self.match_post(post, ppr_top_n=ppr_top_n, final_top_k=final_top_k)
                results.append(result)
                success_count += 1
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Post {post.id} 매칭 실패: {e}")
                # 실패한 경우 빈 결과로 추가하지 않음 (옵션)
                # 또는 빈 matches로 추가할 수도 있음
                continue
            except Exception as e:
                logger.error(f"Post {post.id} 매칭 중 예상치 못한 오류: {e}", exc_info=True)
                continue

        return BatchJobMatchingResponse(
            total_count=total_count,
            success_count=success_count,
            results=results,
        )


def get_job_matching_service(
    job_matching_system: JobMatchingSystem,
) -> JobMatchingService:
    """
    JobMatchingService 인스턴스를 생성하여 반환
    
    Args:
        job_matching_system: JobMatchingSystem 인스턴스 (dependency injection)
        
    Returns:
        JobMatchingService: Service 인스턴스
    """
    return JobMatchingService(job_matching_system)

