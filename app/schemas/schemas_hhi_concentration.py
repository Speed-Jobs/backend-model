"""
HHI Concentration Index API Schemas

HHI (Herfindahl-Hirschman Index):
채용 시장의 직무별 집중도를 측정하는 지표입니다.
- 공식: HHI = Σ(si²) where si = 직무i의 점유율
- 해석:
  * 0 ~ 0.15: "분산" - 다양한 직무에 채용이 골고루 분포 (경쟁 다양화)
  * 0.15 ~ 0.25: "부분 집중" - 일부 직무에 채용이 집중되는 경향
  * 0.25+: "쏠림" - 특정 직무에 채용이 과도하게 집중 (독점적 경향)
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class HHIScore(BaseModel):
    """HHI 점수 및 해석"""

    hhi_value: float = Field(
        ...,
        description=(
            "HHI 지수 (0~1)\n"
            "시장 집중도를 나타내는 수치. 낮을수록 분산, 높을수록 집중"
        ),
        ge=0.0,
        le=1.0,
    )
    level: str = Field(
        ...,
        description=(
            "집중도 수준\n"
            "- 분산: HHI < 0.15 (다양한 직무 골고루 분포)\n"
            "- 부분집중: 0.15 ≤ HHI < 0.25\n"
            "- 쏠림: HHI ≥ 0.25 (특정 직무 독점)"
        ),
    )
    interpretation: str = Field(
        ...,
        description="집중도에 대한 해석 텍스트 (한글)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "hhi_value": 0.28,
                "level": "쏠림",
                "interpretation": "SW개발 직무에 채용이 과도하게 집중되어 있으며, 시장 다양화가 필요한 상황입니다.",
            }
        }


class PositionConcentration(BaseModel):
    """직무별 점유율 정보"""

    position_id: int = Field(..., description="직무 ID")
    position_name: str = Field(..., description="직무명")
    count: int = Field(..., description="채용 공고 수", ge=0)
    share_percentage: float = Field(
        ...,
        description="전체 대비 점유율 (%)",
        ge=0.0,
        le=100.0,
    )
    rank: int = Field(..., description="점유율 순위 (1위부터)", ge=1)

    class Config:
        json_schema_extra = {
            "example": {
                "position_id": 1,
                "position_name": "Software Development",
                "count": 450,
                "share_percentage": 45.2,
                "rank": 1,
            }
        }


class IndustryConcentration(BaseModel):
    """산업별 점유율 정보"""

    industry_id: int = Field(..., description="산업 ID")
    industry_name: str = Field(..., description="산업명")
    count: int = Field(..., description="채용 공고 수", ge=0)
    share_percentage: float = Field(
        ...,
        description="전체 대비 점유율 (%)",
        ge=0.0,
        le=100.0,
    )
    rank: int = Field(..., description="점유율 순위 (1위부터)", ge=1)

    class Config:
        json_schema_extra = {
            "example": {
                "industry_id": 10,
                "industry_name": "Front-end Development",
                "count": 180,
                "share_percentage": 18.1,
                "rank": 1,
            }
        }


class HHIConcentrationInsightData(BaseModel):
    """HHI Concentration 인사이트 데이터"""

    period: str = Field(..., description="조회 기간 (YYYY-MM 또는 YYYY-MM-DD ~ YYYY-MM-DD)")
    window_type: str = Field(
        ...,
        description=(
            "분석 기간 윈도우 타입\n"
            "- 1month: 단일 월 분석\n"
            "- period: 사용자 지정 기간"
        ),
    )
    total_count: int = Field(..., description="전체 채용 공고 수", ge=0)

    overall_hhi: HHIScore = Field(
        ...,
        description="전체 시장 HHI (직무+산업 통합)",
    )
    position_hhi: HHIScore = Field(
        ...,
        description="직무별 HHI (Position 레벨 집중도)",
    )
    industry_hhi: HHIScore = Field(
        ...,
        description="산업별 HHI (Industry 레벨 집중도)",
    )

    top_positions: List[PositionConcentration] = Field(
        default_factory=list,
        description="점유율 상위 직무 목록 (최대 5개)",
    )
    top_industries: List[IndustryConcentration] = Field(
        default_factory=list,
        description="점유율 상위 산업 목록 (최대 5개)",
    )

    insights: str = Field(
        ...,
        description=(
            "종합 시사점 및 권장사항 (한글 텍스트)\n"
            "- 집중도 트렌드 분석\n"
            "- 시장 다양화/집중화 경향\n"
            "- 채용 전략 권장사항"
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "period": "2024-12",
                "window_type": "1month",
                "total_count": 1000,
                "overall_hhi": {
                    "hhi_value": 0.28,
                    "level": "쏠림",
                    "interpretation": "SW개발 직무에 채용이 과도하게 집중",
                },
                "position_hhi": {
                    "hhi_value": 0.26,
                    "level": "쏠림",
                    "interpretation": "SW개발, 데이터 분석 직무가 전체의 65% 차지",
                },
                "industry_hhi": {
                    "hhi_value": 0.18,
                    "level": "부분집중",
                    "interpretation": "프론트엔드, 백엔드 개발 분야 집중",
                },
                "top_positions": [
                    {
                        "position_id": 1,
                        "position_name": "Software Development",
                        "count": 450,
                        "share_percentage": 45.2,
                        "rank": 1,
                    }
                ],
                "top_industries": [
                    {
                        "industry_id": 10,
                        "industry_name": "Front-end Development",
                        "count": 180,
                        "share_percentage": 18.1,
                        "rank": 1,
                    }
                ],
                "insights": "SW개발 직무 쏠림 현상이 심화되고 있습니다. 데이터 직군이 전월 대비 12% 성장하며 다양화 조짐을 보이나, 여전히 시장 집중도가 높은 상황입니다. 비즈니스 직군 채용 확대를 통해 포트폴리오 다양화가 필요합니다.",
            }
        }


class HHIConcentrationInsightResponse(BaseModel):
    """HHI Concentration 인사이트 조회 응답"""

    status: int = Field(200, description="HTTP 상태 코드")
    code: str = Field("SUCCESS", description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    data: HHIConcentrationInsightData = Field(..., description="HHI 집중도 인사이트 데이터")
