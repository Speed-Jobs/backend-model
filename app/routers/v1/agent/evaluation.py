"""
Evaluation API Router
채용공고 평가 결과 조회 API
"""

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from typing import Dict
import PyPDF2
import io

from app.schemas.agent import (
    EvaluationResponse, 
    TwoPostsRequest, 
    ReportGenerationResponse,
    JobPostingDetailReportResponse
)
from app.services.agent.evaluation_service import (
    evaluate_two_posts,
    evaluate_pdf_files,
    generate_report
)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get(
    "/compare",
    response_model=Dict[str, EvaluationResponse],
    summary="2개 채용공고 비교 평가",
    description="SK AX 채용공고와 경쟁사 채용공고를 병렬로 평가하여 비교 결과를 반환합니다.",
)
async def evaluate_two_posts_api(
    sk_ax_post: int = Query(..., description="SK AX 채용공고 ID", ge=1, example=123),
    competitor_post: int = Query(..., description="경쟁사 채용공고 ID", ge=1, example=456)
) -> Dict[str, EvaluationResponse]:
    """
    2개의 채용공고 ID로 비교 평가
    
    이 API는:
    1. DB에서 2개의 채용공고를 조회 (SK AX vs 경쟁사)
    2. 각 공고를 병렬로 평가 (가독성, 구체성, 매력도)
    3. 두 개의 평가 결과를 반환
    
    Args:
        sk_ax_post: SK AX 채용공고 ID
        competitor_post: 경쟁사 채용공고 ID
        
    Returns:
        Dict[str, EvaluationResponse]: 두 개의 평가 결과
            - sk_ax: SK AX 공고 평가 결과
            - competitor: 경쟁사 공고 평가 결과
        
    Raises:
        404: 채용공고를 찾을 수 없음
        500: 서버 내부 오류
        
    Example:
        GET /api/v1/evaluation/compare?sk_ax_post=123&competitor_post=456
    """
    try:
        result = await evaluate_two_posts(sk_ax_post, competitor_post)
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


@router.post(
    "/reports/{post_id}",
    response_model=JobPostingDetailReportResponse,
    summary="평가 기반 보고서 생성",
    description="평가 데이터를 기반으로 구조화된 채용공고 상세 정보를 생성합니다.",
)
async def generate_report_api(
    post_id: int
) -> JobPostingDetailReportResponse:
    """
    특정 post_id의 평가 데이터를 기반으로 채용공고 상세 정보를 구조화하여 반환합니다.
    
    이 API는:
    1. post_id로 저장된 JSON 파일 검색 (가장 최근 파일)
    2. job_posting_generator를 사용하여 채용공고 정보를 20개 필드로 구조화
    3. 사용한 JSON 파일 자동 삭제
    
    Args:
        post_id: 채용공고 ID
        
    Returns:
        JobPostingDetailReportResponse: 구조화된 채용공고 상세 정보
            - status: 응답 상태 (success/error)
            - data: JobPostingDetailReport (20개 필드)
            - message: 응답 메시지
        
    Raises:
        404: 평가 데이터를 찾을 수 없음
        500: 서버 내부 오류
        
    Example:
        POST /api/v1/evaluation/reports/123
        
        Response:
        {
            "status": "success",
            "data": {
                "company_name": "현대오토에버",
                "position": "백엔드 개발자",
                "employment_type": "정규직",
                ...
            },
            "message": "채용공고 정보 추출 완료"
        }
    """
    try:
        # Service layer에서 보고서 생성
        result = await generate_report(post_id)
        
        # 응답 반환
        return JobPostingDetailReportResponse(
            status=result.get("status", "success"),
            data=result.get("data"),
            message=result.get("message", "채용공고 정보 추출 완료")
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"보고서 생성 중 오류 발생: {str(e)}"
        )

