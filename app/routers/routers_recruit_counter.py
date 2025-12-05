"""
채용 공고 수 추이 API Router
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import date
from typing import Optional

from app.db.config.base import get_db,get_db_readonly
from app.services.dashboard.recruit_counter import get_job_postings_trend
from app.schemas.schemas_recruit_counter import DashBoardResponse, JobPostingsTrendData


router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["Dashboard"]
)


@router.get(
    "/job-postings-trend",
    response_model=DashBoardResponse,
    summary="채용 공고 수 추이 조회",
    description="전체 채용 공고 수의 일간/주간/월간 추이를 조회합니다."
)
def get_job_postings_trend_endpoint(
    timeframe: str = Query(
        ...,
        description="시간 단위",
        enum=["daily", "weekly", "monthly"],
        example="daily"
    ),
    db: Session = Depends(get_db_readonly)
):
    """
    채용 공고 수 추이 조회
    
    - **timeframe**: daily (일간), weekly (주간), monthly (월간)
    
    날짜는 자동으로 계산됩니다:
    - daily: 오늘부터 30일 전까지 (1달)
    - weekly: 오늘부터 12주 전까지
    - monthly: 오늘부터 11개월 전까지
    """
    try:
        # Service 호출 (날짜 자동 계산)
        trend_data = get_job_postings_trend(
            db=db,
            timeframe=timeframe
        )
        
        # 응답 메시지 생성
        timeframe_kr = {
            "daily": "일간",
            "weekly": "주간",
            "monthly": "월간"
        }
        message = f"{timeframe_kr.get(timeframe, timeframe)} 채용 공고 수 추이 조회 성공"
        
        return DashBoardResponse(
            status=200,
            code="SUCCESS",
            message=message,
            data=trend_data.dict()
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

