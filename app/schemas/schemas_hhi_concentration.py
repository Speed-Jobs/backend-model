"""
HHI Concentration Index API Schemas

HHI (Herfindahl-Hirschman Index):
채용 시장의 직무별 집중도를 측정하는 지표입니다.
- 공식: HHI = Σ(si²) where si = 직무i의 점유율
- 해석:
  * 0 ~ 0.15: "분산" - 다양한 직무에 채용이 골고루 분포 (경쟁 다양화)
  * 0.15 ~ 0.25: "부분 집중" - 일부 직무에 채용이 집중되는 경향
  * 0.25+: "쏠림" - 특정 직무에 채용이 과도하게 집중 (독점적 경향)

Analysis Types:
- overall: 전체 시장 분석 (직군별 HHI)
- position: 특정 직군 내 산업별 분석
- industry: 특정 산업 분석 (순위, 점유율, 대안 추천)
"""
from typing import List, Optional, Literal
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


class PeriodInfo(BaseModel):
    """분석 기간 정보"""

    start: str = Field(..., description="시작일 (YYYY-MM-DD)")
    end: str = Field(..., description="종료일 (YYYY-MM-DD)")

    class Config:
        json_schema_extra = {
            "example": {
                "start": "2025-07-01",
                "end": "2025-09-30"
            }
        }


class HHIInterpretation(BaseModel):
    """HHI 해석 정보 (간소화 버전)"""

    level: str = Field(
        ...,
        description="집중도 수준: 분산 / 부분집중 / 쏠림"
    )
    difficulty: str = Field(
        ...,
        description="경쟁 난이도: 낮음 / 보통 / 높음"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "level": "분산",
                "difficulty": "낮음"
            }
        }


class AlternativeIndustry(BaseModel):
    """대안 산업 추천 정보"""

    industry_id: int = Field(..., description="산업 ID")
    industry_name: str = Field(..., description="산업명")
    skill_similarity: float = Field(..., description="스킬 유사도 (0~1)", ge=0.0, le=1.0)
    share_percentage: float = Field(..., description="해당 산업 점유율 (%)", ge=0.0, le=100.0)

    class Config:
        json_schema_extra = {
            "example": {
                "industry_id": 15,
                "industry_name": "Back-end Development",
                "skill_similarity": 0.85,
                "share_percentage": 12.3
            }
        }


# ===== 시나리오별 데이터 모델 =====

class OverallAnalysisData(BaseModel):
    """전체 시장 분석 데이터 (HHI + YoY 통합)"""

    analysis_type: Literal["overall"] = Field(
        default="overall",
        description="분석 유형"
    )
    period: PeriodInfo = Field(..., description="분석 기간 (3개월 고정)")
    total_posts: int = Field(..., description="전체 채용 공고 수", ge=0)

    # HHI 분석
    hhi: float = Field(
        ...,
        description="HHI 지수 (0~1, 시각화용)",
        ge=0.0,
        le=1.0
    )
    interpretation: HHIInterpretation = Field(..., description="집중도 해석")

    top_positions: List[PositionConcentration] = Field(
        default_factory=list,
        description="점유율 상위 직군 목록 (최대 5개)"
    )

    # YoY Overheat 분석
    yoy_overheat_score: float = Field(
        ...,
        description="전년 대비 과열도 지수 (0~100)",
        ge=0.0,
        le=100.0
    )
    yoy_trend: str = Field(..., description="YoY 트렌드: 과열/기준/냉각")
    yoy_current_count: int = Field(..., description="현재 기간 채용 공고 수", ge=0)
    yoy_previous_count: int = Field(..., description="작년 동기간 채용 공고 수", ge=0)

    # 통합 인사이트
    insights: List[str] = Field(
        default_factory=list,
        description="통합 인사이트 (HHI, CR₂, Entropy, YoY 기반). include_insights=false면 빈 배열"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_type": "overall",
                "period": {
                    "start": "2025-07-01",
                    "end": "2025-09-30"
                },
                "total_posts": 1194,
                "hhi": 0.1459,
                "interpretation": {
                    "level": "분산",
                    "difficulty": "낮음"
                },
                "top_positions": [
                    {
                        "position_id": 1,
                        "position_name": "Software Development",
                        "count": 315,
                        "share_percentage": 26.4,
                        "rank": 1
                    }
                ],
                "insights": [
                    "다양한 직군에서 고르게 채용이 진행되고 있어 시장 경쟁이 분산되어 있습니다 (HHI 0.15)",
                    "상위 2개 직군이 전체의 42.2%를 차지하고 있어 일부 집중 경향이 있습니다",
                    "직군 구성이 매우 다양하여 지원자 입장에서 다양한 선택지를 확보할 수 있습니다"
                ]
            }
        }


class PositionAnalysisData(BaseModel):
    """특정 직군 내 산업별 분석 데이터 (HHI + YoY 통합)"""

    analysis_type: Literal["position"] = Field(
        default="position",
        description="분석 유형"
    )
    period: PeriodInfo = Field(..., description="분석 기간 (3개월 고정)")

    position_id: int = Field(..., description="분석 대상 직군 ID")
    position_name: str = Field(..., description="분석 대상 직군명")
    total_posts: int = Field(..., description="해당 직군 채용 공고 수", ge=0)

    # HHI 분석
    hhi: float = Field(
        ...,
        description="해당 직군 내 산업별 HHI 지수 (0~1, 시각화용)",
        ge=0.0,
        le=1.0
    )
    interpretation: HHIInterpretation = Field(..., description="집중도 해석")

    top_industries: List[IndustryConcentration] = Field(
        default_factory=list,
        description="해당 직군 내 점유율 상위 산업 목록 (최대 5개)"
    )

    # YoY Overheat 분석
    yoy_overheat_score: float = Field(
        ...,
        description="전년 대비 과열도 지수 (0~100)",
        ge=0.0,
        le=100.0
    )
    yoy_trend: str = Field(..., description="YoY 트렌드: 과열/기준/냉각")
    yoy_current_count: int = Field(..., description="현재 기간 채용 공고 수", ge=0)
    yoy_previous_count: int = Field(..., description="작년 동기간 채용 공고 수", ge=0)

    # 통합 인사이트
    insights: List[str] = Field(
        default_factory=list,
        description="통합 인사이트 (HHI, CR₂, Entropy, YoY 기반). include_insights=false면 빈 배열"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_type": "position",
                "period": {
                    "start": "2025-07-01",
                    "end": "2025-09-30"
                },
                "position_id": 1,
                "position_name": "Software Development",
                "total_posts": 315,
                "hhi": 0.2156,
                "interpretation": {
                    "level": "부분집중",
                    "difficulty": "보통"
                },
                "top_industries": [
                    {
                        "industry_id": 10,
                        "industry_name": "Front-end Development",
                        "count": 92,
                        "share_percentage": 29.2,
                        "rank": 1
                    }
                ],
                "insights": [
                    "'Software Development' 직군 내에서 산업별 채용이 일부 집중되어 있습니다 (HHI 0.22)",
                    "상위 2개 산업이 전체의 51.7%를 차지하며 Front-end와 Back-end 개발이 주류를 이루고 있습니다",
                    "Full-stack Development와 DevOps 분야로 다양화가 진행 중입니다"
                ]
            }
        }


class IndustryAnalysisData(BaseModel):
    """특정 산업 분석 데이터 (HHI + YoY 통합)"""

    analysis_type: Literal["industry"] = Field(
        default="industry",
        description="분석 유형"
    )
    period: PeriodInfo = Field(..., description="분석 기간 (3개월 고정)")

    position_id: int = Field(..., description="소속 직군 ID")
    position_name: str = Field(..., description="소속 직군명")
    industry_id: int = Field(..., description="분석 대상 산업 ID")
    industry_name: str = Field(..., description="분석 대상 산업명")

    posts_count: int = Field(..., description="해당 산업 채용 공고 수", ge=0)
    rank: int = Field(..., description="직군 내 순위", ge=1)
    share_percentage: float = Field(
        ...,
        description="직군 내 점유율 (%)",
        ge=0.0,
        le=100.0
    )

    alternative_industries: List[AlternativeIndustry] = Field(
        default_factory=list,
        description="스킬 유사도 기반 대안 산업 추천 (최대 3개)"
    )

    # YoY Overheat 분석
    yoy_overheat_score: float = Field(
        ...,
        description="전년 대비 과열도 지수 (0~100)",
        ge=0.0,
        le=100.0
    )
    yoy_trend: str = Field(..., description="YoY 트렌드: 과열/기준/냉각")
    yoy_current_count: int = Field(..., description="현재 기간 채용 공고 수", ge=0)
    yoy_previous_count: int = Field(..., description="작년 동기간 채용 공고 수", ge=0)

    # 통합 인사이트
    insights: List[str] = Field(
        default_factory=list,
        description="통합 인사이트 (순위, 점유율, 경쟁 난이도, 대안 추천, YoY 기반). include_insights=false면 빈 배열"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_type": "industry",
                "period": {
                    "start": "2025-07-01",
                    "end": "2025-09-30"
                },
                "position_id": 1,
                "position_name": "Software Development",
                "industry_id": 10,
                "industry_name": "Front-end Development",
                "posts_count": 92,
                "rank": 1,
                "share_percentage": 29.2,
                "alternative_industries": [
                    {
                        "industry_id": 15,
                        "industry_name": "Back-end Development",
                        "skill_similarity": 0.78,
                        "share_percentage": 22.5
                    }
                ],
                "insights": [
                    "'Software Development' 직군 내에서 'Front-end Development' 산업은 1위로 29.2%의 점유율을 차지하고 있습니다",
                    "높은 점유율로 인해 경쟁이 치열할 수 있습니다",
                    "대안으로 'Back-end Development' (유사도 78%, 점유율 22.5%) 산업을 고려해보세요"
                ]
            }
        }


class CombinedIndustryAnalysisData(BaseModel):
    """통합 인사이트 분석 데이터 (Total + Position + Industry)"""

    analysis_type: Literal["combined"] = Field(
        default="combined",
        description="분석 유형"
    )
    period: PeriodInfo = Field(..., description="분석 기간 (3개월 고정)")

    # Total 시장 인사이트 (간소화 버전)
    total_insight: OverallAnalysisData = Field(
        ...,
        description="전체 시장 HHI 인사이트"
    )

    # Position 인사이트 (선택)
    position_insight: Optional[PositionAnalysisData] = Field(
        None,
        description="직군별 HHI 인사이트 (position_name이 지정된 경우에만 포함)"
    )

    # Industry 인사이트 (선택)
    industry_insight: Optional[IndustryAnalysisData] = Field(
        None,
        description="산업별 상세 인사이트 (industry_name이 지정된 경우에만 포함)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_type": "combined",
                "period": {
                    "start": "2025-07-01",
                    "end": "2025-09-30"
                },
                "total_insight": {
                    "analysis_type": "overall",
                    "period": {
                        "start": "2025-07-01",
                        "end": "2025-09-30"
                    },
                    "total_posts": 1194,
                    "hhi": 0.1459,
                    "interpretation": {
                        "level": "분산",
                        "difficulty": "낮음"
                    },
                    "top_positions": [],
                    "insights": []
                },
                "position_insight": {
                    "analysis_type": "position",
                    "period": {
                        "start": "2025-07-01",
                        "end": "2025-09-30"
                    },
                    "position_id": 1,
                    "position_name": "Software Development",
                    "total_posts": 315,
                    "hhi": 0.2156,
                    "interpretation": {
                        "level": "부분집중",
                        "difficulty": "보통"
                    },
                    "top_industries": [],
                    "insights": []
                },
                "industry_insight": {
                    "analysis_type": "industry",
                    "period": {
                        "start": "2025-07-01",
                        "end": "2025-09-30"
                    },
                    "position_id": 1,
                    "position_name": "Software Development",
                    "industry_id": 10,
                    "industry_name": "Front-end Development",
                    "posts_count": 92,
                    "rank": 1,
                    "share_percentage": 29.2,
                    "alternative_industries": [],
                    "insights": []
                }
            }
        }


# 레거시 모델 제거됨 (더 이상 사용되지 않음)
# - HHIConcentrationInsightData: analyze_combined_insights()로 대체됨
# - HHIConcentrationInsightResponse: HHIAnalysisResponse로 대체됨

# ===== 새로운 API 응답 모델 =====

from typing import Union

HHIAnalysisData = Union[
    OverallAnalysisData, 
    PositionAnalysisData, 
    IndustryAnalysisData,
    CombinedIndustryAnalysisData
]


class HHIAnalysisResponse(BaseModel):
    """HHI 집중도 분석 응답 (신규)"""

    status: int = Field(200, description="HTTP 상태 코드")
    code: str = Field("SUCCESS", description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    data: HHIAnalysisData = Field(..., description="HHI 분석 데이터 (시나리오별 상이)")

    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "code": "SUCCESS",
                "message": "HHI 집중도 분석 완료",
                "data": {
                    "analysis_type": "overall",
                    "period": {
                        "start": "2025-07-01",
                        "end": "2025-09-30"
                    },
                    "total_posts": 1194,
                    "hhi": 0.1459,
                    "interpretation": {
                        "level": "분산",
                        "difficulty": "낮음"
                    },
                    "top_positions": [],
                    "insights": []
                }
            }
        }
