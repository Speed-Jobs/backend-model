from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class MonthlySkillTrend(BaseModel):
    """
    월별 스킬 트렌드 한 행

    - Swagger/OpenAPI 상에서는 month 필드 외에 스킬명이 키가 되는 동적 필드를 허용해야 하므로
      additionalProperties 구조를 사용하기 위해 extra 허용 모드로 둡니다.
    """

    month: str = Field(
        ...,
        description="월 (YYYY-MM 형식)",
        pattern=r"^\d{4}-\d{2}$",
    )

    class Config:
        extra = "allow"  # 스킬명(key)들을 동적으로 허용


class SkillTrendData(BaseModel):
    """
    상위 스킬 연도별 트렌드 데이터

    - 전체/회사별 공용 구조
    """

    company_name: Optional[str] = Field(
        None, description="회사 이름 (회사별 조회 시에만 포함)"
    )
    year: Optional[str] = Field(
        None,
        description="조회 연도 (단일 연도 조회 시, YYYY 형식)",
        pattern=r"^\d{4}$",
    )
    years: Optional[List[str]] = Field(
        None,
        description="조회 연도 목록 (다중 연도 조회 시, 예: ['2022', '2023', '2024'])",
    )
    skills: List[str] = Field(
        ..., description="상위 스킬 목록 (공고 수 기준 내림차순 정렬)"
    )
    trends: List[MonthlySkillTrend] = Field(
        ..., description="월별 스킬 트렌드 데이터"
    )


class SkillTrendResponse(BaseModel):
    """스킬 트렌드 조회 응답"""

    status: int = Field(200, description="HTTP 상태 코드", example=200)
    code: str = Field("SUCCESS", description="응답 코드", example="SUCCESS")
    message: str = Field(..., description="응답 메시지", example="스킬 트렌드 조회 성공")
    data: SkillTrendData

