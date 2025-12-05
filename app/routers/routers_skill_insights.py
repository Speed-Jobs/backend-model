"""
스킬 트렌드 API Router
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime

from app.db.config.base import get_db,get_db_readonly
from app.services.dashboard.skill_insights import skill_insights_service
from app.schemas import schemas_skill_insights


router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["Skill Trends"]
)


@router.get(
    "/skill-trends",
    response_model=schemas_skill_insights.SkillTrendResponse,
    summary="전체 상위 스킬 연도별 트렌드 조회",
    description="""
    전체 채용 공고에서 연도별로 상위 N개 스킬의 월별 트렌드 데이터를 조회합니다.
    - year 파라미터 미입력: 최근 5개년의 연도별 스킬 빈도수를 반환합니다.
    - year 파라미터 입력: 해당 연도의 분기별 데이터와 전년도 동기간 분기별 데이터를 비교하여 반환합니다.
    """
)
async def get_skill_trends(
    year: Optional[str] = Query(
        None,
        description="조회할 연도 (예: 2021, 2022, 2023, 2024, 2025). 미입력 시 최근 5개년 데이터 반환",
        pattern=r"^\d{4}$",
        example="2024"
    ),
    top_n: int = Query(
        10,
        description="상위 N개 스킬 조회 (기본값: 10)",
        ge=1,
        le=50,
        example=10
    ),
    db: Session = Depends(get_db_readonly)
):
    """
    스킬 트렌드 조회
    
    - **year**: 조회할 연도 (YYYY 형식). 미입력 시 최근 5개년 데이터 반환
    - **top_n**: 상위 N개 스킬 조회 (1-50)
    
    Returns:
        SkillTrendResponse: 스킬 트렌드 데이터
    """
    try:
        # 연도 검증
        if year:
            try:
                year_int = int(year)
                current_year = datetime.now().year
                if year_int < 2020 or year_int > current_year:
                    raise HTTPException(
                        status_code=400,
                        detail=f"연도는 2020년부터 {current_year}년까지 조회 가능합니다."
                    )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="연도는 YYYY 형식의 숫자여야 합니다."
                )
        
        # 서비스 호출
        data = skill_insights_service.get_skill_trends(
            db=db,
            year=year,
            top_n=top_n
        )
        
        # 응답 메시지 생성
        if year:
            message = f"스킬 트렌드 조회 성공 ({year}년)"
        else:
            message = "스킬 트렌드 조회 성공 (최근 5개년)"
        
        return schemas_skill_insights.SkillTrendResponse(
            status=200,
            code="SUCCESS",
            message=message,
            data=data
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"스킬 트렌드 조회 중 오류 발생: {str(e)}"
        )

