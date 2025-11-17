from __future__ import annotations

from functools import lru_cache
from typing import List

from fastapi import APIRouter, HTTPException

from app.schemas import (
    SimilarSkillItem,
    SimilarSkillRequest,
    SimilarSkillResponse,
)
from app.services.skill_match.skill_match import (
    MODEL_PATH,
    get_similar_skills,
    load_skill_association_model,
)


router = APIRouter(
    prefix="/skill-match",
    tags=["skill-match"],
)


@lru_cache()
def _get_skill_model():
    return load_skill_association_model(MODEL_PATH)


def _filter_existing_keywords(keywords: List[str]) -> List[str]:
    model = _get_skill_model()
    return [keyword for keyword in keywords if keyword in model.wv]


@router.post(
    "/similar-skills",
    response_model=SimilarSkillResponse,
    summary="입력 스킬과 유사한 스킬 추천",
)
def fetch_similar_skills(payload: SimilarSkillRequest) -> SimilarSkillResponse:
    model = _get_skill_model()
    matched_keywords = _filter_existing_keywords(payload.keywords)

    if not matched_keywords:
        raise HTTPException(
            status_code=404,
            detail="입력한 스킬을 모델에서 찾을 수 없습니다.",
        )

    similar_skills = get_similar_skills(
        model=model,
        keywords=matched_keywords,
        top_n=payload.top_n,
    )

    results = [
        SimilarSkillItem(skill=skill, score=float(score))
        for skill, score in similar_skills
    ]

    return SimilarSkillResponse(
        input_keywords=payload.keywords,
        matched_keywords=matched_keywords,
        results=results,
        top_n=payload.top_n,
        count=len(results),
    )

