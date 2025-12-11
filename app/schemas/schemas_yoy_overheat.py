"""
YoY Overheat Index API Schemas

YoY (Year-over-Year) Overheat Index:
채용 시장의 전년 대비 과열도를 측정하는 지표입니다.
- 공식: YoY(t) = min(100, max(0, Ct/Bt-1 × 50))
- 해석:
  * 50: 작년과 동일한 수준 (기준선)
  * 50 초과: 작년보다 증가 → 과열 (채용 확대)
  * 50 미만: 작년보다 감소 → 냉각 (채용 축소)
  * 100: 2배 이상 증가 (극심한 과열)
  * 0: 채용 없음 (극심한 냉각)
"""
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field

# 레거시 스키마 제거됨 (더 이상 사용되지 않음)
# - YoYScoreByPosition: PositionYoYData로 대체됨
# - YoYScoreByIndustry: IndustryYoYData로 대체됨


class OverallYoYData(BaseModel):
    """전체 시장 YoY 분석 데이터"""

    analysis_type: Literal["overall"] = Field(
        default="overall",
        description="분석 유형"
    )
    overall_yoy_score: float = Field(
        ...,
        description=(
            "전체 평균 YoY 점수 (0~100)\n"
            "모든 직무/산업을 통합한 시장 전체 과열도\n"
            "- 50 초과: 시장 과열 (채용 확대)\n"
            "- 50: 작년과 동일\n"
            "- 50 미만: 시장 냉각 (채용 축소)"
        ),
        ge=0.0,
        le=100.0,
    )
    overall_trend: str = Field(
        ...,
        description="전체 시장 트렌드 (과열/기준/냉각)",
    )
    overall_current_count: int = Field(..., description="현재 기간 전체 채용 공고 수", ge=0)
    overall_previous_count: int = Field(..., description="작년 동기간 전체 채용 공고 수", ge=0)


class PositionYoYData(BaseModel):
    """직군별 YoY 분석 데이터"""

    analysis_type: Literal["position"] = Field(
        default="position",
        description="분석 유형"
    )
    position_id: int = Field(..., description="직군 ID")
    position_name: str = Field(..., description="직군명")
    position_yoy_score: float = Field(
        ...,
        description="직군별 YoY 점수 (0~100)",
        ge=0.0,
        le=100.0,
    )
    position_trend: str = Field(..., description="직군 트렌드 (과열/기준/냉각)")
    position_current_count: int = Field(..., description="현재 기간 직군 채용 공고 수", ge=0)
    position_previous_count: int = Field(..., description="작년 동기간 직군 채용 공고 수", ge=0)


class IndustryYoYData(BaseModel):
    """산업별 YoY 분석 데이터"""

    analysis_type: Literal["industry"] = Field(
        default="industry",
        description="분석 유형"
    )
    industry_id: int = Field(..., description="산업 ID")
    industry_name: str = Field(..., description="산업명")
    industry_yoy_score: float = Field(
        ...,
        description="산업별 YoY 점수 (0~100)",
        ge=0.0,
        le=100.0,
    )
    industry_trend: str = Field(..., description="산업 트렌드 (과열/기준/냉각)")
    industry_current_count: int = Field(..., description="현재 기간 산업 채용 공고 수", ge=0)
    industry_previous_count: int = Field(..., description="작년 동기간 산업 채용 공고 수", ge=0)


class CombinedYoYData(BaseModel):
    """통합 YoY 인사이트 분석 데이터 (Total + Position + Industry)"""

    analysis_type: Literal["combined"] = Field(
        default="combined",
        description="분석 유형"
    )
    
    # Total 인사이트 (항상 포함)
    total_insight: OverallYoYData = Field(
        ...,
        description="전체 시장 YoY 인사이트"
    )
    
    # Position 인사이트 (position_name 지정 시)
    position_insight: Optional[PositionYoYData] = Field(
        None,
        description="직군별 YoY 인사이트 (position_name 지정 시)"
    )
    
    # Industry 인사이트 (position_name + industry_name 지정 시)
    industry_insight: Optional[IndustryYoYData] = Field(
        None,
        description="산업별 YoY 인사이트 (position_name + industry_name 지정 시)"
    )


# 레거시 스키마 제거됨 (더 이상 사용되지 않음)
# - YoYOverheatData: analyze_combined_yoy_insights()로 대체됨
# - YoYScoreByPosition, YoYScoreByIndustry: CombinedYoYData로 대체됨


# 통합 인사이트용 Union 타입
YoYAnalysisData = Union[
    OverallYoYData,
    PositionYoYData,
    IndustryYoYData,
    CombinedYoYData,
]


class YoYOverheatResponse(BaseModel):
    """YoY Overheat 조회 응답"""

    status: int = Field(200, description="HTTP 상태 코드")
    code: str = Field("SUCCESS", description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    data: YoYAnalysisData = Field(..., description="YoY 과열도 분석 데이터")
