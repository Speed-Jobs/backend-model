from fastapi import APIRouter, Depends, Query, HTTPException, Path
from sqlalchemy.orm import Session
from typing import Optional
from app.db import get_db
from app.schemas import schemas_competitors_skills
from app.services.dashboard.competitors_skills import competitors_skills_service

router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["경쟁사 분석"]
)

@router.get(
    "/skills/diversity",
    response_model=schemas_competitors_skills.SkillDiversityResponse,
    summary="회사별 스킬 다양성 조회",
    description="경쟁사별 요구하는 고유 스킬 개수를 조회합니다."
)
async def get_skill_diversity(
    year: Optional[int] = Query(None, description="조회 연도 (미입력시 전체)", ge=2020, le=2030),
    db: Session = Depends(get_db)
):
    """전체 또는 연도별 회사별 스킬 다양성 조회"""
    
    try:
        data = competitors_skills_service.get_skill_diversity(db, year)
        
        message = f"회사별 스킬 다양성 조회 성공 ({'전체' if year is None else f'{year}년'})"
        
        return schemas_competitors_skills.SkillDiversityResponse(
            status=200,
            code="SUCCESS",
            message=message,
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 중 오류 발생: {str(e)}")


@router.get(
    "/posts",
    response_model=schemas_competitors_skills.PostsWithSkillsResponse,
    summary="경쟁사 공고 및 스킬 조회",
    description="경쟁사별 채용 공고와 요구 스킬을 상세 조회합니다."
)
async def get_posts_with_skills(
    company_name: Optional[str] = Query(None, description="회사명 필터 (부분 일치)"),
    year: Optional[int] = Query(None, description="연도 필터", ge=2020, le=2030),
    limit: int = Query(100, description="조회 제한", ge=1, le=500),
    db: Session = Depends(get_db)
):
    """경쟁사 공고 및 스킬 상세 조회"""
    
    try:
        data = competitors_skills_service.get_posts_with_skills(
            db=db,
            company_name=company_name,
            year=year,
            limit=limit
        )
        
        filters = []
        if company_name:
            filters.append(f"회사: {company_name}")
        if year:
            filters.append(f"연도: {year}년")
        
        filter_text = f" ({', '.join(filters)})" if filters else ""
        message = f"경쟁사 공고 조회 성공{filter_text}"
        
        return schemas_competitors_skills.PostsWithSkillsResponse(
            status=200,
            code="SUCCESS",
            message=message,
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 중 오류 발생: {str(e)}")


@router.get(
    "/companies/{companyId}/skill-trends",
    response_model=schemas_competitors_skills.SkillTrendResponse,
    summary="회사별 상위 스킬 분기별 트렌드 조회",
    description="특정 회사의 상위 스킬들이 분기별로 어떻게 변하는지 추이를 조회합니다. 현재 분기와 이전 분기의 데이터만 반환됩니다."
)
async def get_company_skill_trends(
    companyId: str = Path(..., description="회사 ID 또는 회사명 (한글)"),
    year: Optional[int] = Query(None, description="연도 (기본값: 현재 연도)", ge=2021, le=2025),
    top_n: int = Query(10, description="상위 N개 스킬 (기본값: 10)", ge=1, le=20),
    db: Session = Depends(get_db)
):
    """회사별 상위 스킬 분기별 트렌드 조회"""
    
    from datetime import datetime
    
    # year가 없으면 현재 연도 사용
    if year is None:
        year = datetime.now().year
    
    try:
        data = competitors_skills_service.get_skill_trends(
            db=db,
            company_id=companyId,
            year=year,
            top_n=top_n
        )
        
        message = f"회사별 스킬 트렌드 조회 성공"
        
        return schemas_competitors_skills.SkillTrendResponse(
            status=200,
            code="SUCCESS",
            message=message,
            data=data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 중 오류 발생: {str(e)}")