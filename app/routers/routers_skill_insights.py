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


@router.get(
    "/companies/{company_name}/skill-trends",
    response_model=schemas_skill_insights.SkillTrendResponse,
    summary="회사별 상위 스킬 연도별 트렌드 조회",
    description="""
    특정 회사의 채용 공고에서 연도별로 상위 N개 스킬의 월별 트렌드 데이터를 조회합니다.
    - year 파라미터 미입력: 최근 5개년의 연도별 스킬 빈도수를 반환합니다.
    - year 파라미터 입력: 해당 연도의 분기별 데이터와 전년도 동기간 분기별 데이터를 비교하여 반환합니다.
    회사 키워드는 COMPANY_GROUPS 딕셔너리를 참조합니다 (예: "toss" 입력 시 "토스", "토스뱅크", "토스증권", "비바리퍼블리카", "AICC" 등 모두 검색).
    """
)
async def get_company_skill_trends(
    company_name: str,
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
    회사별 스킬 트렌드 조회
    
    - **company_name**: 회사 키워드 (예: "toss", "kakao", "hanwha") - COMPANY_GROUPS 딕셔너리 참조
    - **year**: 조회할 연도 (YYYY 형식). 미입력 시 최근 5개년 데이터 반환
    - **top_n**: 상위 N개 스킬 조회 (1-50)
    
    Returns:
        SkillTrendResponse: 회사별 스킬 트렌드 데이터
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
        data = skill_insights_service.get_company_skill_trends(
            db=db,
            company_keyword=company_name,
            year=year,
            top_n=top_n
        )
        
        # 데이터가 없는 경우 확인
        if (data.years and len(data.years) == 0) or (data.skills and len(data.skills) == 0):
            raise HTTPException(
                status_code=404,
                detail=f"'{company_name}' 회사의 스킬 트렌드 데이터를 찾을 수 없습니다."
            )
        
        # 응답 메시지 생성
        if year:
            message = f"회사별 스킬 트렌드 조회 성공 ({company_name}, {year}년)"
        else:
            message = f"회사별 스킬 트렌드 조회 성공 ({company_name}, 최근 5개년)"
        
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
            detail=f"회사별 스킬 트렌드 조회 중 오류 발생: {str(e)}"
        )


@router.get(
    "/skills/statistics",
    response_model=schemas_skill_insights.SkillStatisticsResponse,
    summary="스킬 클라우드 통계 조회",
    description="""
    지정된 기간과 회사에 대한 스킬 통계 데이터를 조회합니다.
    스킬별 공고 수, 비율, 변화율, 관련 스킬 정보를 제공합니다.
    스킬 클라우드 시각화에 사용됩니다.
    start_date와 end_date를 지정하지 않으면 현재 연도 1월 1일부터 현재 날짜까지의 데이터를 반환합니다.
    """
)
async def get_skills_statistics(
    start_date: Optional[str] = Query(
        None,
        description="시작 날짜 (YYYY-MM-DD 형식). 미입력 시 현재 연도 1월 1일",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        example="2024-01-01"
    ),
    end_date: Optional[str] = Query(
        None,
        description="종료 날짜 (YYYY-MM-DD 형식). 미입력 시 현재 날짜",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        example="2024-12-31"
    ),
    company: Optional[str] = Query(
        None,
        description="회사 키워드 필터 (선택사항, 예: 'toss', 'kakao', 'hanwha'). COMPANY_GROUPS에서 매핑된 여러 회사를 통합 집계",
        example="toss"
    ),
    limit: int = Query(
        20,
        description="반환할 상위 스킬 개수 (기본값: 20)",
        ge=1,
        le=100,
        example=20
    ),
    db: Session = Depends(get_db_readonly)
):
    """
    스킬 클라우드 통계 조회
    
    - **start_date**: 시작 날짜 (YYYY-MM-DD 형식). 미입력 시 현재 연도 1월 1일
    - **end_date**: 종료 날짜 (YYYY-MM-DD 형식). 미입력 시 현재 날짜
    - **company**: 회사 키워드 필터 (선택사항, 예: 'toss', 'kakao', 'hanwha')
      - COMPANY_GROUPS에서 매핑된 여러 회사를 통합 집계
      - 예: 'toss' → "토스", "토스뱅크", "토스증권", "비바리퍼블리카" 등 모두 포함
    - **limit**: 반환할 상위 스킬 개수 (1-100)
    
    Returns:
        SkillStatisticsResponse: 스킬 통계 데이터 (관련 스킬 및 유사도 포함)
    """
    try:
        # 날짜 검증
        if start_date:
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용해주세요."
                )
        
        if end_date:
            try:
                datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식을 사용해주세요."
                )
        
        # 서비스 호출
        data = skill_insights_service.get_skill_statistics(
            db=db,
            start_date=start_date,
            end_date=end_date,
            company=company,
            limit=limit
        )
        
        # 응답 메시지 생성
        if start_date and end_date:
            message = "스킬 통계 조회 성공"
        else:
            message = "스킬 통계 조회 성공 (현재 연도 1월 1일 ~ 현재 날짜)"
        
        return schemas_skill_insights.SkillStatisticsResponse(
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
            detail=f"스킬 통계 조회 중 오류 발생: {str(e)}"
        )

