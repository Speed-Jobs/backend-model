"""직군별 통계 API Router"""
from typing import Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.config.base import get_db,get_db_readonly
from app.schemas.schemas_competitor_industry_trend import JobRoleStatisticsResponse
from app.schemas.schemas_competitor_industry_trend import (
    JobRoleStatisticsResponse,
    JobRoleStatisticsWithInsightsResponse,
    JobRoleStatisticsWithInsightsData,
)
from app.services.dashboard.competitor_industry_trend import (
    get_job_role_statistics,
)
from app.core.agents.dashboard.job_role.job_role_insight_agent import (
    generate_job_role_insight_async,
)


router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["직군별 통계"],
)


@router.get(
    "/job-role-statistics",
    response_model=JobRoleStatisticsWithInsightsResponse,
    summary="직군별 통계 조회 (인사이트 포함)",
    description=(
        "지정된 기간(timeframe)과 카테고리에 따른 직군별 채용 공고 통계를 조회합니다.\n"
        "현재 기간과 이전 기간(동기간)의 데이터를 함께 반환하여 비교 분석이 가능합니다.\n"
        "include_insights=true로 설정하면 AI 기반 인사이트도 함께 제공됩니다."
    ),
)
async def get_job_role_statistics_endpoint(
    timeframe: Literal["monthly_same_period", "quarterly_same_period"] = Query(
        "monthly_same_period",
        description=(
            "통계 기간 단위 (동기간 기준 비교)\n"
            "- monthly_same_period: 동기간 대비 월별 (이번 달 vs 지난 달 동일 일자 기준)\n"
            "- quarterly_same_period: 동기간 대비 분기별 (이번 분기 vs 직전 분기 동일 일자 기준)"
        ),
    ),
    category: Literal["Tech", "Biz", "BizSupporting"] = Query(
        "Tech",
        description=(
            "직군 카테고리\n"
            "- Tech: 기술 직군 (Software Development 등)\n"
            "- Biz: 비즈니스 직군 (Sales, Consulting 등)\n"
            "- BizSupporting: 비즈니스 지원 직군 (Biz. Supporting 등)"
        ),
    ),
    start_date: Optional[str] = Query(
        None,
        description=(
            "현재 기간 시작일 (YYYY-MM-DD)\n"
            "미지정 시 monthly_same_period 는 이번 달 1일, "
            "quarterly_same_period 는 현재 분기 첫 날을 사용합니다."
        ),
        example="2024-12-01",
    ),
    end_date: Optional[str] = Query(
        None,
        description=(
            "현재 기간 종료일 (YYYY-MM-DD)\n"
            "미지정 시 오늘 날짜를 사용합니다."
        ),
        example="2024-12-15",
    ),
    company: Optional[str] = Query(
        None,
        description="특정 회사명 필터 (부분 일치). 지정하지 않으면 전체 회사 기준으로 집계합니다.",
        example="토스",
    ),
    include_insights: bool = Query(
        False,
        description="AI 기반 인사이트 포함 여부. true로 설정하면 직군별 트렌드 분석 및 인사이트를 제공합니다.",
    ),
    db: Session = Depends(get_db_readonly),
) -> JobRoleStatisticsResponse:

    """
    직군별 채용 공고 통계를 조회합니다.

    - timeframe: monthly_same_period / quarterly_same_period
    - category: Tech / Biz / BizSupporting
    - start_date, end_date: 미지정 시 자동 계산 규칙을 따릅니다.
    - company: 특정 회사명 부분 일치 필터 (예: "토스", "네이버")
    - include_insights: AI 기반 인사이트 포함 여부
    """
    try:
        # 1. 통계 데이터 조회
        statistics_data = get_job_role_statistics(
            db=db,
            timeframe=timeframe,
            category=category,
            start_date=start_date,
            end_date=end_date,
            company=company,
        )

        # 2. 인사이트 생성 (include_insights=true인 경우)
        insights_data = None
        if include_insights:
            insight_result = await generate_job_role_insight_async(
                timeframe=timeframe,
                category=category,
                start_date=start_date,
                end_date=end_date,
                company=company,
            )
            
            if insight_result["status"] == "success":
                # Pydantic 모델을 dict로 변환
                insights_data = insight_result["data"].model_dump() if hasattr(insight_result["data"], "model_dump") else insight_result["data"]
            # 인사이트 생성 실패해도 통계 데이터는 반환

        message = "직군별 통계 조회 성공"
        if include_insights and insights_data:
            message += " (인사이트 포함)"
        
        return JobRoleStatisticsWithInsightsResponse(
            status=200,
            code="SUCCESS",
            message=message,
            data=JobRoleStatisticsWithInsightsData(
                statistics=statistics_data,
                insights=insights_data,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


