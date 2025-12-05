from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class RelatedSkillStatistics(BaseModel):
    """관련 스킬 통계 정보 (스킬 클라우드용)"""

    name: str = Field(..., description="관련 스킬 이름", example="react")
    count: int = Field(
        ...,
        description="해당 스킬이 요구된 채용 공고 수",
        ge=0,
        example=245,
    )
    percentage: float = Field(
        ...,
        description="전체 공고 대비 비율 (%)",
        ge=0,
        le=100,
        example=19.6,
    )
    change: float = Field(
        ...,
        description="이전 기간 대비 변화율 (%)",
        example=3.5,
    )


class SkillStatistics(BaseModel):
    """개별 스킬 통계 정보"""

    name: str = Field(..., description="스킬 이름", example="typescript")
    count: int = Field(
        ...,
        description="해당 스킬이 요구된 채용 공고 수",
        ge=0,
        example=286,
    )
    percentage: float = Field(
        ...,
        description="전체 공고 대비 비율 (%)",
        ge=0,
        le=100,
        example=22.9,
    )
    change: float = Field(
        ...,
        description="이전 기간 대비 변화율 (%)",
        example=5.2,
    )
    relatedSkills: List[RelatedSkillStatistics] = Field(
        ...,
        description="관련 스킬 목록 (Node2Vec 스킬 연관성 모델 기반 추천 + 통계 정보 포함)",
    )


class SkillStatisticsPeriod(BaseModel):
    """스킬 통계 조회 기간"""

    start_date: date = Field(..., description="시작 날짜 (YYYY-MM-DD)", example="2024-01-01")
    end_date: date = Field(..., description="종료 날짜 (YYYY-MM-DD)", example="2024-12-31")


class SkillStatisticsData(BaseModel):
    """스킬 클라우드용 통계 데이터"""

    period: SkillStatisticsPeriod
    company: Optional[str] = Field(
        None,
        description="회사 이름 (전체 조회 시 null)",
        example="토스",
    )
    total_job_postings: int = Field(
        ...,
        description="전체 채용 공고 수",
        ge=0,
        example=1250,
    )
    skills: List[SkillStatistics] = Field(
        ..., description="스킬 통계 목록 (공고 수 기준 내림차순 정렬)"
    )


class SkillStatisticsResponse(BaseModel):
    """스킬 클라우드 통계 조회 응답"""

    status: int = Field(200, description="HTTP 상태 코드", example=200)
    code: str = Field(
        "SUCCESS",
        description="응답 코드",
        example="SUCCESS",
    )
    message: str = Field(..., description="응답 메시지", example="스킬 통계 조회 성공")
    data: SkillStatisticsData


