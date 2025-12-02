"""
Evaluation API Router
채용공고 평가 결과 조회 API
"""

from fastapi import APIRouter, HTTPException, Path as PathParam, UploadFile, File
from pydantic import BaseModel
from typing import Dict
import PyPDF2
import io

from app.schemas.agent import EvaluationResponse
from app.services.agent.evaluation_service import (
    evaluate_post_by_id,
    evaluate_two_posts,
    evaluate_pdf_files
)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


# Request Schema
class TwoPostsRequest(BaseModel):
    post_id_1: int
    post_id_2: int


@router.get(
    "/post/{post_id}",
    response_model=EvaluationResponse,
    summary="채용공고 평가",
    description="특정 채용공고를 실시간으로 평가하여 결과를 반환합니다.",
)
async def get_evaluation_by_post(
    post_id: int = PathParam(..., description="채용공고 ID", ge=1)
) -> EvaluationResponse:
    """
    채용공고 ID로 평가
    
    이 API는:
    1. DB에서 해당 post_id의 채용공고를 조회
    2. 평가 모듈(가독성, 구체성, 매력도)을 병렬로 실행
    3. 평가 결과를 반환
    
    Args:
        post_id: 채용공고 ID
        
    Returns:
        EvaluationResponse: 가독성, 구체성, 매력도 평가 결과
        
    Raises:
        404: 해당 채용공고를 찾을 수 없음
        500: 서버 내부 오류
    """
    try:
        # Service 레이어에서 평가 실행
        result = await evaluate_post_by_id(post_id)
        return result
        
    except ValueError as e:
        # 채용공고를 찾을 수 없는 경우
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        # 기타 서버 오류
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.post(
    "/posts",
    response_model=Dict[str, EvaluationResponse],
    summary="2개 채용공고 비교 평가",
    description="2개의 채용공고를 병렬로 평가하여 비교 결과를 반환합니다.",
)
async def evaluate_two_posts_api(
    request: TwoPostsRequest
) -> Dict[str, EvaluationResponse]:
    """
    2개의 채용공고 ID로 비교 평가
    
    이 API는:
    1. DB에서 2개의 채용공고를 조회
    2. 각 공고를 병렬로 평가 (가독성, 구체성, 매력도)
    3. 두 개의 평가 결과를 반환
    
    Args:
        request: TwoPostsRequest
            - post_id_1: 첫 번째 채용공고 ID
            - post_id_2: 두 번째 채용공고 ID
        
    Returns:
        Dict[str, EvaluationResponse]: 두 개의 평가 결과
            - post_1: 첫 번째 공고 평가 결과
            - post_2: 두 번째 공고 평가 결과
        
    Raises:
        404: 채용공고를 찾을 수 없음
        500: 서버 내부 오류
    """
    try:
        result = await evaluate_two_posts(request.post_id_1, request.post_id_2)
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.post(
    "/pdfs",
    response_model=Dict[str, EvaluationResponse],
    summary="2개 PDF 채용공고 평가",
    description="2개의 PDF 파일을 업로드하여 병렬로 평가합니다.",
)
async def evaluate_two_pdfs_api(
    file_1: UploadFile = File(..., description="첫 번째 PDF 파일"),
    file_2: UploadFile = File(..., description="두 번째 PDF 파일")
) -> Dict[str, EvaluationResponse]:
    """
    2개의 PDF 파일로 비교 평가
    
    이 API는:
    1. 2개의 PDF 파일에서 텍스트 추출
    2. 각 텍스트를 병렬로 평가 (가독성, 구체성, 매력도)
    3. 두 개의 평가 결과를 반환
    
    Args:
        file_1: 첫 번째 PDF 파일
        file_2: 두 번째 PDF 파일
        
    Returns:
        Dict[str, EvaluationResponse]: 두 개의 평가 결과
            - file_1: 첫 번째 파일 평가 결과
            - file_2: 두 번째 파일 평가 결과
        
    Raises:
        400: PDF 파일이 아니거나 텍스트 추출 실패
        500: 서버 내부 오류
    """
    try:
        # PDF 파일 검증
        if not file_1.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"File 1 must be a PDF file, got: {file_1.filename}"
            )
        
        if not file_2.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"File 2 must be a PDF file, got: {file_2.filename}"
            )
        
        # PDF에서 텍스트 추출
        content_1 = await file_1.read()
        content_2 = await file_2.read()
        
        # PyPDF2로 텍스트 추출
        pdf_reader_1 = PyPDF2.PdfReader(io.BytesIO(content_1))
        pdf_reader_2 = PyPDF2.PdfReader(io.BytesIO(content_2))
        
        # 모든 페이지의 텍스트 추출
        text_1 = ""
        for page in pdf_reader_1.pages:
            text_1 += page.extract_text()
        
        text_2 = ""
        for page in pdf_reader_2.pages:
            text_2 += page.extract_text()
        
        # 텍스트가 비어있는지 확인
        if not text_1.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from file: {file_1.filename}"
            )
        
        if not text_2.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from file: {file_2.filename}"
            )
        
        # 평가 실행
        result = await evaluate_pdf_files(
            pdf_content_1=text_1,
            pdf_content_2=text_2,
            filename_1=file_1.filename,
            filename_2=file_2.filename
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF evaluation failed: {str(e)}"
        )

