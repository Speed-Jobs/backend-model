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
from typing import List
from pydantic import BaseModel, Field


class YoYScoreByPosition(BaseModel):
    """직무별 YoY 과열도 점수"""

    position_id: int = Field(..., description="직무 ID")
    position_name: str = Field(..., description="직무명 (예: Software Development)")
    yoy_score: float = Field(
        ...,
        description=(
            "YoY 과열도 점수 (0~100)\n"
            "- 0: 채용 없음 (극심한 냉각)\n"
            "- 25: 작년 대비 절반 (냉각)\n"
            "- 50: 작년과 동일 (기준선)\n"
            "- 75: 작년 대비 1.5배 (과열)\n"
            "- 100: 작년 대비 2배 이상 (극심한 과열)"
        ),
        ge=0.0,
        le=100.0,
    )
    current_count: int = Field(..., description="현재 기간 채용 공고 수", ge=0)
    previous_year_count: int = Field(..., description="작년 동기간 채용 공고 수", ge=0)
    trend: str = Field(
        ...,
        description=(
            "트렌드 해석\n"
            "- 과열: YoY > 50 (채용 확대)\n"
            "- 기준: YoY = 50 (작년 동일)\n"
            "- 냉각: YoY < 50 (채용 축소)"
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "position_id": 1,
                "position_name": "Software Development",
                "yoy_score": 75.0,
                "current_count": 150,
                "previous_year_count": 100,
                "trend": "과열",
            }
        }


class YoYScoreByIndustry(BaseModel):
    """산업별 YoY 과열도 점수"""

    industry_id: int = Field(..., description="산업 ID")
    industry_name: str = Field(..., description="산업명 (예: Front-end Development)")
    yoy_score: float = Field(
        ...,
        description=(
            "YoY 과열도 점수 (0~100)\n"
            "- 0: 채용 없음 (극심한 냉각)\n"
            "- 25: 작년 대비 절반 (냉각)\n"
            "- 50: 작년과 동일 (기준선)\n"
            "- 75: 작년 대비 1.5배 (과열)\n"
            "- 100: 작년 대비 2배 이상 (극심한 과열)"
        ),
        ge=0.0,
        le=100.0,
    )
    current_count: int = Field(..., description="현재 기간 채용 공고 수", ge=0)
    previous_year_count: int = Field(..., description="작년 동기간 채용 공고 수", ge=0)
    trend: str = Field(
        ...,
        description=(
            "트렌드 해석\n"
            "- 과열: YoY > 50 (채용 확대)\n"
            "- 기준: YoY = 50 (작년 동일)\n"
            "- 냉각: YoY < 50 (채용 축소)"
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "industry_id": 10,
                "industry_name": "Front-end Development",
                "yoy_score": 82.5,
                "current_count": 65,
                "previous_year_count": 40,
                "trend": "과열",
            }
        }


class YoYOverheatData(BaseModel):
    """YoY Overheat 분석 데이터"""

    year: int = Field(..., description="조회 연도", ge=2020)
    month: int = Field(..., description="조회 월 (1~12)", ge=1, le=12)
    window_type: str = Field(
        ...,
        description=(
            "분석 기간 윈도우 타입\n"
            "- 1month: 단일 월 분석\n"
            "- 3month: 3개월 이동평균 (현재월, -1개월, -2개월)"
        ),
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
    by_position: List[YoYScoreByPosition] = Field(
        default_factory=list,
        description="직무별 YoY 점수 목록 (점수 높은 순 정렬)",
    )
    by_industry: List[YoYScoreByIndustry] = Field(
        default_factory=list,
        description="산업별 YoY 점수 목록 (점수 높은 순 정렬)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2024,
                "month": 12,
                "window_type": "3month",
                "overall_yoy_score": 68.5,
                "overall_trend": "과열",
                "by_position": [
                    {
                        "position_id": 1,
                        "position_name": "Software Development",
                        "yoy_score": 75.0,
                        "current_count": 150,
                        "previous_year_count": 100,
                        "trend": "과열",
                    }
                ],
                "by_industry": [
                    {
                        "industry_id": 10,
                        "industry_name": "Front-end Development",
                        "yoy_score": 82.5,
                        "current_count": 65,
                        "previous_year_count": 40,
                        "trend": "과열",
                    }
                ],
            }
        }


class YoYOverheatResponse(BaseModel):
    """YoY Overheat 조회 응답"""

    status: int = Field(200, description="HTTP 상태 코드")
    code: str = Field("SUCCESS", description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    data: YoYOverheatData = Field(..., description="YoY 과열도 분석 데이터")
