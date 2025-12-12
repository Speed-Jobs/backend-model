"""
Recruitment Schedule Router
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from app.db.config.base import get_db
from app.services.dashboard.recruitment_schedule import (
    get_all_recruitment_schedules,
    get_competition_intensity
)
from app.config.company_groups import COMPANY_GROUPS

router = APIRouter(
    prefix="/recruitment-schedule",
    tags=["recruitment-schedule"]
)


@router.get("/companies")
def get_recruitment_schedules(
    type: str = Query(
        ...,
        description="채용 유형 (Entry-level/Experienced, 신입/경력)",
        example="Entry-level",
    ),
    data_type: str = Query(default="actual", description="데이터 유형 (actual: 실제 공고, predicted: 예측치, all: 전체)", example="actual"),
    start_date: str = Query(..., description="시작일 (YYYY-MM-DD 형식)", example="2025-01-01"),
    end_date: str = Query(..., description="종료일 (YYYY-MM-DD 형식)", example="2025-12-31"),
    company_keywords: Optional[str] = Query(None, description="회사 키워드 리스트 (쉼표로 구분, 예: 'toss,kakao,hanwha')", example="toss,kakao"),
    db: Session = Depends(get_db)
):
    """
    전체 회사의 채용 일정을 조회합니다.

    - **type**: 신입 또는 경력
    - **data_type**: actual(실제 공고), predicted(예측치), all(전체)
    - **start_date**: 조회 시작일
    - **end_date**: 조회 종료일
    - **company_keywords**: (선택) 특정 회사들만 조회 (쉼표로 구분, 예: "toss,kakao")
    """
    # type 검증 + 노멀라이즈 (영어만 허용 → 내부 한글로 매핑)
    allowed_types = ["entry-level", "experienced", "Entry-level", "Experienced"]
    if type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="type은 Entry-level 또는 Experienced여야 합니다."
        )
    type_map = {
        "entry-level": "신입",
        "experienced": "경력",
        "Entry-level": "신입",
        "Experienced": "경력",
    }
    normalized_type = type_map.get(type, type)

    # data_type 검증
    if data_type not in ["actual", "predicted", "all"]:
        raise HTTPException(
            status_code=400,
            detail="data_type은 'actual', 'predicted', 'all' 중 하나여야 합니다."
        )

    # company_keywords 파싱
    parsed_company_keywords = None
    if company_keywords:
        parsed_company_keywords = [keyword.strip() for keyword in company_keywords.split(",")]

    # 서비스 호출
    result = get_all_recruitment_schedules(
        db=db,
        type_filter=normalized_type,
        start_date=start_date,
        end_date=end_date,
        company_keywords=parsed_company_keywords,
        data_type=data_type
    )

    # 에러 처리
    if result["status"] != 200:
        raise HTTPException(
            status_code=result["status"],
            detail=result["message"]
        )

    return result


@router.get("/competition-intensity")
def get_recruitment_competition_intensity(
    start_date: str = Query(..., description="시작일 (YYYY-MM-DD 형식)", example="2025-01-01"),
    end_date: str = Query(..., description="종료일 (YYYY-MM-DD 형식)", example="2025-01-31"),
    type: Optional[str] = Query(
        None,
        description="채용 유형 (Entry-level/Experienced, 신입/경력)",
        example="Entry-level",
    ),
    include_insights: bool = Query(
        False,
        description="Rule-Based 인사이트 포함 여부 (true일 경우 통계 분석, 추천 날짜, 경쟁사 패턴, 최적 시기 포함)",
        example=False,
    ),
    db: Session = Depends(get_db)
):
    """
    날짜별 경쟁 강도 분석

    특정 기간 내에 각 날짜별로 동시에 채용 중인 경쟁사 수를 계산합니다.

    - **start_date**: 분석 시작일
    - **end_date**: 분석 종료일
    - **type**: (선택) 신입 또는 경력 (미지정 시 전체)
    - **include_insights**: (선택) Rule-Based 인사이트 포함 여부 (기본값: false)

    Returns:
        - **period**: 분석 기간
        - **max_overlaps**: 최대 겹침 수
        - **daily_intensity**: 날짜별 경쟁 강도
            - **date**: 날짜
            - **overlap_count**: 겹침 수
            - **companies**: 채용 중인 회사 목록
        - **insights**: (include_insights=true일 때만) Rule-Based 분석 결과
            - **statistics**: 통계 분석 (평균, 중앙값, 분포 등)
            - **recommended_dates**: 추천 날짜 (best_dates, worst_dates)
            - **company_patterns**: 경쟁사 패턴 (가장 활발한 회사 TOP 5)
            - **optimal_period**: 최적 시기 추천 (연속 저경쟁 기간)
    """
    # type 검증 + 노멀라이즈 (영어만 허용 → 내부 한글로 매핑)
    if type:
        allowed_types = ["entry-level", "experienced", "Entry-level", "Experienced"]
        if type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="type은 Entry-level 또는 Experienced여야 합니다."
            )
        type_map = {
            "entry-level": "신입",
            "experienced": "경력",
            "Entry-level": "신입",
            "Experienced": "경력",
        }
        type = type_map.get(type, type)

    # 서비스 호출
    result = get_competition_intensity(
        db=db,
        start_date=start_date,
        end_date=end_date,
        type_filter=type,
        include_insights=include_insights
    )

    # 에러 처리
    if result["status"] != 200:
        raise HTTPException(
            status_code=result["status"],
            detail=result["message"]
        )

    return result

