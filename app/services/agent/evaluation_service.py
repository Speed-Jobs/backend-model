"""
Evaluation Service
채용공고 평가 서비스 로직
"""

from typing import Dict, Any
import asyncio

from app.db.config.base import SessionLocal
from app.models.post import Post
from app.models.company import Company
from app.utils.agents.evaluation.modules.module_readability import collect_readability_data
from app.utils.agents.evaluation.modules.module_specificity import collect_specificity_data
from app.utils.agents.evaluation.modules.module_attractiveness import collect_attractiveness_data
from app.schemas.agent import EvaluationResponse


async def evaluate_post_by_id(post_id: int) -> EvaluationResponse:
    """
    특정 채용공고를 평가합니다.
    
    Args:
        post_id: 채용공고 ID
        
    Returns:
        EvaluationResponse: 평가 결과 (가독성, 구체성, 매력도)
        
    Raises:
        ValueError: 채용공고를 찾을 수 없는 경우
        Exception: 평가 중 오류 발생
    """
    db = SessionLocal()
    
    try:
        # 1. 채용공고 조회
        post = db.query(Post).filter(Post.id == post_id).first()
        
        if not post:
            raise ValueError(f"Post with id={post_id} not found")
        
        if not post.description:
            raise ValueError(f"Post id={post_id} has no description")
        
        # 2. 회사 정보 조회
        company_name = post.company.name if post.company else "알 수 없음"
        
        # 3. 채용공고 평가 실행 (병렬 처리)
        print(f"[Evaluation Service] Evaluating post_id={post.id}, company={company_name}")
        
        readability_result, specificity_result, attractiveness_result = await asyncio.gather(
            collect_readability_data(
                job_description=post.description,
                company_name=company_name,
                title=post.title
            ),
            collect_specificity_data(
                job_description=post.description,
                company_name=company_name,
                title=post.title
            ),
            collect_attractiveness_data(
                job_description=post.description,
                company_name=company_name,
                title=post.title
            ),
            return_exceptions=True
        )
        
        # 4. 예외 처리
        if isinstance(readability_result, Exception):
            print(f"[Evaluation Service] Readability evaluation failed: {readability_result}")
            raise readability_result
        
        if isinstance(specificity_result, Exception):
            print(f"[Evaluation Service] Specificity evaluation failed: {specificity_result}")
            raise specificity_result
        
        if isinstance(attractiveness_result, Exception):
            print(f"[Evaluation Service] Attractiveness evaluation failed: {attractiveness_result}")
            raise attractiveness_result
        
        # 5. EvaluationResponse 반환
        print(f"[Evaluation Service] Evaluation completed successfully")
        
        return EvaluationResponse(
            readability=readability_result,
            specificity=specificity_result,
            attractiveness=attractiveness_result
        )
        
    finally:
        db.close()


def evaluate_post_by_id_sync(post_id: int) -> EvaluationResponse:
    """
    동기 버전: 특정 채용공고를 평가합니다.
    
    Args:
        post_id: 채용공고 ID
        
    Returns:
        EvaluationResponse: 평가 결과
    """
    return asyncio.run(evaluate_post_by_id(post_id))


async def evaluate_two_posts(post_id_1: int, post_id_2: int) -> Dict[str, EvaluationResponse]:
    """
    2개의 채용공고를 병렬로 평가합니다.
    
    Args:
        post_id_1: 첫 번째 채용공고 ID
        post_id_2: 두 번째 채용공고 ID
        
    Returns:
        Dict[str, EvaluationResponse]: 두 개의 평가 결과
            {
                "post_1": EvaluationResponse,
                "post_2": EvaluationResponse
            }
    """
    print(f"[Evaluation Service] Evaluating two posts: post_id_1={post_id_1}, post_id_2={post_id_2}")
    
    # 2개의 채용공고를 병렬로 평가
    result_1, result_2 = await asyncio.gather(
        evaluate_post_by_id(post_id_1),
        evaluate_post_by_id(post_id_2),
        return_exceptions=True
    )
    
    # 예외 처리
    if isinstance(result_1, Exception):
        raise ValueError(f"Post {post_id_1} evaluation failed: {str(result_1)}")
    
    if isinstance(result_2, Exception):
        raise ValueError(f"Post {post_id_2} evaluation failed: {str(result_2)}")
    
    return {
        "post_1": result_1,
        "post_2": result_2
    }


async def evaluate_pdf_files(pdf_content_1: str, pdf_content_2: str, 
                             filename_1: str = "file_1.pdf", 
                             filename_2: str = "file_2.pdf") -> Dict[str, EvaluationResponse]:
    """
    2개의 PDF 채용공고를 병렬로 평가합니다.
    
    Args:
        pdf_content_1: 첫 번째 PDF의 텍스트 내용
        pdf_content_2: 두 번째 PDF의 텍스트 내용
        filename_1: 첫 번째 파일명
        filename_2: 두 번째 파일명
        
    Returns:
        Dict[str, EvaluationResponse]: 두 개의 평가 결과
            {
                "file_1": EvaluationResponse,
                "file_2": EvaluationResponse
            }
    """
    print(f"[Evaluation Service] Evaluating two PDF files: {filename_1}, {filename_2}")
    
    # 2개의 PDF를 병렬로 평가
    result_1, result_2 = await asyncio.gather(
        asyncio.gather(
            collect_readability_data(
                job_description=pdf_content_1,
                company_name="",
                title=filename_1
            ),
            collect_specificity_data(
                job_description=pdf_content_1,
                company_name="",
                title=filename_1
            ),
            collect_attractiveness_data(
                job_description=pdf_content_1,
                company_name="",
                title=filename_1
            )
        ),
        asyncio.gather(
            collect_readability_data(
                job_description=pdf_content_2,
                company_name="",
                title=filename_2
            ),
            collect_specificity_data(
                job_description=pdf_content_2,
                company_name="",
                title=filename_2
            ),
            collect_attractiveness_data(
                job_description=pdf_content_2,
                company_name="",
                title=filename_2
            )
        ),
        return_exceptions=True
    )
    
    # 예외 처리 및 결과 생성
    if isinstance(result_1, Exception):
        raise ValueError(f"File {filename_1} evaluation failed: {str(result_1)}")
    
    if isinstance(result_2, Exception):
        raise ValueError(f"File {filename_2} evaluation failed: {str(result_2)}")
    
    # 결과 언패킹
    readability_1, specificity_1, attractiveness_1 = result_1
    readability_2, specificity_2, attractiveness_2 = result_2
    
    return {
        "file_1": EvaluationResponse(
            readability=readability_1,
            specificity=specificity_1,
            attractiveness=attractiveness_1
        ),
        "file_2": EvaluationResponse(
            readability=readability_2,
            specificity=specificity_2,
            attractiveness=attractiveness_2
        )
    }

