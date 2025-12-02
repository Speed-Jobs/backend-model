"""
Evaluation Service
채용공고 평가 서비스 로직
"""

from typing import Dict, Any
import asyncio
import json
import os
from pathlib import Path

from app.db.config.base import SessionLocal
from app.models.post import Post
from app.models.company import Company
from app.utils.agents.evaluation.modules.module_readability import collect_readability_data
from app.utils.agents.evaluation.modules.module_specificity import collect_specificity_data
from app.utils.agents.evaluation.modules.module_attractiveness import collect_attractiveness_data
from app.utils.agents.evaluation.report_saver import save_raw_evaluation_data
from app.schemas.agent import EvaluationResponse
from app.core.agents.job_posting.job_posting_agent import generate_improved_job_posting_async


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


async def evaluate_two_posts(sk_ax_post: int, competitor_post: int) -> Dict[str, EvaluationResponse]:
    """
    2개의 채용공고를 병렬로 평가합니다.
    
    Args:
        sk_ax_post: SK AX 채용공고 ID
        competitor_post: 경쟁사 채용공고 ID
        
    Returns:
        Dict[str, EvaluationResponse]: 두 개의 평가 결과
            {
                "sk_ax": EvaluationResponse,
                "competitor": EvaluationResponse
            }
    """
    print(f"[Evaluation Service] Evaluating two posts: sk_ax_post={sk_ax_post}, competitor_post={competitor_post}")
    
    # 2개의 채용공고를 병렬로 평가
    sk_ax_result, competitor_result = await asyncio.gather(
        evaluate_post_by_id(sk_ax_post),
        evaluate_post_by_id(competitor_post),
        return_exceptions=True
    )
    
    # 예외 처리
    if isinstance(sk_ax_result, Exception):
        raise ValueError(f"SK AX Post {sk_ax_post} evaluation failed: {str(sk_ax_result)}")
    
    if isinstance(competitor_result, Exception):
        raise ValueError(f"Competitor Post {competitor_post} evaluation failed: {str(competitor_result)}")
    
    # 결과를 파일로 저장 (보고서 생성용)
    db = SessionLocal()
    try:
        # SK AX Post 정보 조회
        sk_ax_post_data = db.query(Post).filter(Post.id == sk_ax_post).first()
        if sk_ax_post_data:
            save_raw_evaluation_data(
                post_id=sk_ax_post,
                title=sk_ax_post_data.title or "Unknown",
                company=sk_ax_post_data.company.name if sk_ax_post_data.company else "Unknown",
                raw_results=sk_ax_result.dict(),
                company_id=sk_ax_post_data.company_id
            )
            print(f"[Evaluation Service] Saved SK AX evaluation data for post_id={sk_ax_post}")
        
        # Competitor Post 정보 조회
        competitor_post_data = db.query(Post).filter(Post.id == competitor_post).first()
        if competitor_post_data:
            save_raw_evaluation_data(
                post_id=competitor_post,
                title=competitor_post_data.title or "Unknown",
                company=competitor_post_data.company.name if competitor_post_data.company else "Unknown",
                raw_results=competitor_result.dict(),
                company_id=competitor_post_data.company_id
            )
            print(f"[Evaluation Service] Saved Competitor evaluation data for post_id={competitor_post}")
    finally:
        db.close()
    
    return {
        "sk_ax": sk_ax_result,
        "competitor": competitor_result
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


async def generate_report(post_id: int) -> Dict[str, Any]:
    """
    특정 post_id의 평가 데이터를 기반으로 개선된 채용공고 보고서를 생성합니다.
    
    Args:
        post_id: 채용공고 ID
        
    Returns:
        Dict[str, Any]: 보고서 생성 결과
            - status: 생성 상태 (success/error)
            - improved_posting: 개선된 채용공고 텍스트
            
    Raises:
        ValueError: 평가 데이터를 찾을 수 없음
        Exception: 보고서 생성 실패
    """
    data_dir = Path("data/report")
    
    # 디렉토리 존재 확인
    if not data_dir.exists():
        raise ValueError(f"평가 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
    
    # 모든 JSON 파일을 읽어서 metadata.post_id가 일치하는 파일 찾기
    matching_files = []
    for json_file in data_dir.glob("post_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata = data.get('metadata', {})
                if metadata.get('post_id') == post_id:
                    matching_files.append((json_file, json_file.stat().st_mtime))
        except Exception as e:
            print(f"[Generate Report] Failed to read {json_file}: {e}")
            continue
    
    # 일치하는 파일이 없는 경우
    if not matching_files:
        raise ValueError(
            f"post_id {post_id}에 대한 평가 데이터를 찾을 수 없습니다. "
            f"먼저 /evaluation/compare API를 호출하여 평가를 진행해주세요."
        )
    
    # 가장 최근 파일 선택 (mtime 기준 내림차순 정렬)
    matching_files.sort(key=lambda x: x[1], reverse=True)
    latest_file_path = matching_files[0][0]
    json_filename = latest_file_path.name
    
    print(f"[Generate Report] Using evaluation file: {json_filename}")
    
    # 보고서 생성
    result = await generate_improved_job_posting_async(
        json_filename=json_filename,
        llm_model="gpt-4o"
    )
    
    # 생성 실패 시
    if result.get("status") == "error":
        raise Exception(result.get("message", "보고서 생성 실패"))
    
    # 성공 시 JSON 파일 삭제
    try:
        os.remove(latest_file_path)
        print(f"[Generate Report] Deleted evaluation file: {json_filename}")
    except Exception as e:
        print(f"[Generate Report] Failed to delete JSON file: {e}")
        # 삭제 실패해도 보고서는 반환
    
    return {
        "status": result.get("status", "success"),
        "data": result.get("data"),  # JobPostingDetailReport 데이터
        "message": result.get("message", "채용공고 정보 추출 완료")
    }

