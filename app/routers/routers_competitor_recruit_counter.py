"""
주요 회사별 채용 활동 API Router
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import date
from typing import Optional, List

from app.db.config.base import get_db
from app.services.dashboard.competitor_recruit_counter import get_companies_recruitment_activity
from app.schemas.schemas_recruit_counter import DashBoardResponse


router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["Dashboard"]
)


@router.get(
    "/companies/recruitment-activity",
    response_model=DashBoardResponse,
    summary="주요 회사별 채용 활동 조회",
    description="여러 회사의 채용 활동을 일간/주간/월간 단위로 비교 조회합니다."
)
def get_companies_recruitment_activity_endpoint(
    timeframe: str = Query(
        ...,
        description="시간 단위",
        enum=["daily", "weekly", "monthly"],
        example="daily"
    ),
    company_keywords: Optional[str] = Query(
        None,
        description="조회할 회사명 키워드 (쉼표로 구분, 시작 일치). 예: '토스,한화,라인'",
        example="토스,한화,라인,네이버,카카오,LG,현대오토에버,우아한"
    ),
    db: Session = Depends(get_db)
):
    """
    주요 회사별 채용 활동 조회
    
    - **timeframe**: daily (일간), weekly (주간), monthly (월간)
    - **company_keywords**: 조회할 회사명 키워드 (쉼표로 구분, 회사명 시작 기준 매칭)
      - "Line" 검색 시: Line Pay, Line Taiwan 등 Line으로 시작하는 회사 포함
      - "한화" 검색 시: 한화생명, 한화손해보험 등 한화로 시작하는 회사 포함
    
    company_keywords를 지정하지 않으면 기본 키워드 (토스, 한화, 라인, 네이버, 카카오, LG, 현대오토에버, 우아한)를 조회합니다.
    
    날짜는 자동으로 계산됩니다:
    - daily: 오늘부터 30일 전까지 (1달)
    - weekly: 오늘부터 12주 전까지
    - monthly: 오늘부터 11개월 전까지
    """
    try:
        # company_keywords 파싱
        keywords_list: Optional[List[str]] = None
        if company_keywords:
            keywords_list = [keyword.strip() for keyword in company_keywords.split(",")]
        
        # Service 호출 (날짜 자동 계산)
        activity_data = get_companies_recruitment_activity(
            db=db,
            timeframe=timeframe,
            company_keywords=keywords_list
        )
        
        # 응답 메시지 생성
        timeframe_kr = {
            "daily": "일간",
            "weekly": "주간",
            "monthly": "월간"
        }
        message = f"{timeframe_kr.get(timeframe, timeframe)} 주요 회사별 채용 활동 조회 성공"
        
        return DashBoardResponse(
            status=200,
            code="SUCCESS",
            message=message,
            data=activity_data.dict()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

