from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, conint, validator


class SimilarSkillItem(BaseModel):
    skill: str = Field(..., description="추천된 스킬 명")
    score: float = Field(..., ge=-1.0, le=1.0, description="모델이 계산한 유사도 점수")


class SimilarSkillRequest(BaseModel):
    keywords: List[str] = Field(
        ..., min_items=1, description="유사 스킬을 조회할 키워드 목록"
    )
    top_n: conint(ge=1, le=50) = Field(
        10, description="가져올 유사 스킬 개수 (1~50)"
    )

    @validator("keywords", each_item=True)
    def strip_keyword(cls, v: str) -> str:
        value = v.strip()
        if not value:
            raise ValueError("공백만 있는 스킬 키워드는 허용되지 않습니다.")
        return value


class SimilarSkillResponse(BaseModel):
    input_keywords: List[str]
    matched_keywords: List[str]
    results: List[SimilarSkillItem]
    top_n: int
    count: int

