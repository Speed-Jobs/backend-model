"""
직군별 통계 API Schemas
"""
from typing import List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.schemas.schemas_job_role_insight import JobRoleInsightData


class PeriodSummary(BaseModel):
    """기간 요약 정보"""

    start_date: str = Field(..., description="해당 기간 시작일 (YYYY-MM-DD)")
    end_date: str = Field(..., description="해당 기간 종료일 (YYYY-MM-DD)")
    total_count: int = Field(..., description="해당 기간 전체 공고 수", ge=0)


class IndustryStatistic(BaseModel):
    """직군 내 산업(세부 직무) 통계"""

    name: str = Field(..., description="산업/세부 직무 이름 (예: Front-end Development)")
    current_count: int = Field(..., description="현재 기간 공고 수", ge=0)
    previous_count: int = Field(..., description="이전 기간 공고 수", ge=0)


class JobRoleStatistic(BaseModel):
    """직군별 통계"""

    name: str = Field(..., description="직군 이름 (예: Software Development)")
    current_count: int = Field(..., description="현재 기간 해당 직군 공고 수", ge=0)
    current_percentage: float = Field(
        ...,
        description="현재 기간 전체 대비 비율(%)",
        ge=0.0,
    )
    previous_count: int = Field(..., description="이전 기간 해당 직군 공고 수", ge=0)
    previous_percentage: float = Field(
        ...,
        description="이전 기간 전체 대비 비율(%)",
        ge=0.0,
    )
    change_rate: float = Field(
        ...,
        description="현재 기간 비율 - 이전 기간 비율 (퍼센트 포인트)",
    )
    industries: List[IndustryStatistic] = Field(
        default_factory=list,
        description="직군 내 산업(세부 직무)별 공고 수",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Software Development",
                "current_count": 45,
                "current_percentage": 22.96,
                "previous_count": 38,
                "previous_percentage": 21.35,
                "change_rate": 1.61,
                "industries": [
                    {"name": "Front-end Development", "current_count": 15, "previous_count": 12},
                    {"name": "Back-end Development", "current_count": 20, "previous_count": 18},
                    {"name": "Mobile Development", "current_count": 10, "previous_count": 8},
                ],
            }
        }


class JobRoleStatisticsData(BaseModel):
    """직군별 통계 응답 데이터"""

    timeframe: str = Field(
        ...,
        description="통계 기간 단위 (monthly_same_period / quarterly_same_period)",
    )
    category: str = Field(
        ...,
        description="직군 카테고리 (Tech / Biz / BizSupporting)",
    )
    current_period: PeriodSummary = Field(..., description="현재 기간 요약 정보")
    previous_period: PeriodSummary = Field(..., description="이전 기간 요약 정보")
    statistics: List[JobRoleStatistic] = Field(
        default_factory=list,
        description="직군별 통계 목록",
    )


class JobRoleStatisticsResponse(BaseModel):
    """직군별 통계 조회 응답 래퍼"""

    status: int = Field(200, description="HTTP 상태 코드")
    code: str = Field("SUCCESS", description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    data: JobRoleStatisticsData


class JobRoleStatisticsWithInsightsData(BaseModel):
    """직군별 통계 + 인사이트 응답 데이터"""

    statistics: JobRoleStatisticsData = Field(..., description="직군별 통계 데이터")
    insights: Optional[dict] = Field(None, description="직군별 인사이트 데이터 (include_insights=true인 경우)")


class JobRoleStatisticsWithInsightsResponse(BaseModel):
    """직군별 통계 + 인사이트 조회 응답 래퍼"""

    status: int = Field(200, description="HTTP 상태 코드")
    code: str = Field("SUCCESS", description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    data: JobRoleStatisticsWithInsightsData


