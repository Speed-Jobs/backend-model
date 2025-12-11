"""
Recruitment Schedule Router
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from app.db.config.base import get_db
from app.services.dashboard.recruitment_schedule import (
    get_company_recruitment_schedule,
    get_all_recruitment_schedules,
    get_competition_intensity
)
from app.config.company_groups import COMPANY_GROUPS

router = APIRouter(
    prefix="/recruitment-schedule",
    tags=["recruitment-schedule"]
)


@router.get("/companies/{company_keyword}")
def get_recruitment_schedule_by_company(
    company_keyword: str,
    type: str = Query(..., description="채용 유형 (신입/경력)", example="신입"),
    data_type: str = Query(default="actual", description="데이터 유형 (actual: 실제 공고, predicted: 예측치, all: 전체)", example="actual"),
    start_date: str = Query(..., description="시작일 (YYYY-MM-DD 형식)", example="2025-01-01"),
    end_date: str = Query(..., description="종료일 (YYYY-MM-DD 형식)", example="2025-12-31"),
    position_ids: Optional[str] = Query(None, description="직군 ID 리스트 (쉼표로 구분)", example="1,2,3"),
    db: Session = Depends(get_db)
):
    """
    특정 회사의 채용 일정을 상세 조회합니다.

    - **company_keyword**: 회사 키워드 (예: 'toss', 'kakao', 'hanwha')
    - **type**: 신입 또는 경력
    - **data_type**: actual(실제 공고), predicted(예측치), all(전체)
    - **start_date**: 조회 시작일
    - **end_date**: 조회 종료일
    """
    # type 검증
    if type not in ["신입", "경력"]:
        raise HTTPException(
            status_code=400,
            detail="type은 '신입' 또는 '경력'이어야 합니다."
        )

    # data_type 검증
    if data_type not in ["actual", "predicted", "all"]:
        raise HTTPException(
            status_code=400,
            detail="data_type은 'actual', 'predicted', 'all' 중 하나여야 합니다."
        )

    # position_ids 파싱
    parsed_position_ids = None
    if position_ids:
        try:
            parsed_position_ids = [int(id.strip()) for id in position_ids.split(",")]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="position_ids는 숫자를 쉼표로 구분한 형식이어야 합니다. (예: 1,2,3)"
            )

    # 서비스 호출
    result = get_company_recruitment_schedule(
        db=db,
        company_keyword=company_keyword,
        type_filter=type,
        start_date=start_date,
        end_date=end_date,
        data_type=data_type,
        position_ids=parsed_position_ids
    )
    
    # 에러 처리
    if result["status"] != 200:
        raise HTTPException(
            status_code=result["status"],
            detail=result["message"]
        )
    
    return result


@router.get("/companies")
def get_recruitment_schedules(
    type: str = Query(..., description="채용 유형 (신입/경력)", example="신입"),
    data_type: str = Query(default="actual", description="데이터 유형 (actual: 실제 공고, predicted: 예측치, all: 전체)", example="actual"),
    start_date: str = Query(..., description="시작일 (YYYY-MM-DD 형식)", example="2025-01-01"),
    end_date: str = Query(..., description="종료일 (YYYY-MM-DD 형식)", example="2025-12-31"),
    company_keywords: Optional[str] = Query(None, description="회사 키워드 리스트 (쉼표로 구분, 예: 'toss,kakao,hanwha')", example="toss,kakao"),
    position_ids: Optional[str] = Query(None, description="직군 ID 리스트 (쉼표로 구분)", example="1,2,3"),
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
    # type 검증
    if type not in ["신입", "경력"]:
        raise HTTPException(
            status_code=400,
            detail="type은 '신입' 또는 '경력'이어야 합니다."
        )

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

    # position_ids 파싱
    parsed_position_ids = None
    if position_ids:
        try:
            parsed_position_ids = [int(id.strip()) for id in position_ids.split(",")]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="position_ids는 숫자를 쉼표로 구분한 형식이어야 합니다. (예: 1,2,3)"
            )

    # 서비스 호출
    result = get_all_recruitment_schedules(
        db=db,
        type_filter=type,
        start_date=start_date,
        end_date=end_date,
        company_keywords=parsed_company_keywords,
        data_type=data_type,
        position_ids=parsed_position_ids
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
    type: Optional[str] = Query(None, description="채용 유형 (신입/경력)", example="신입"),
    db: Session = Depends(get_db)
):
    """
    날짜별 경쟁 강도 분석

    특정 기간 내에 각 날짜별로 동시에 채용 중인 경쟁사 수를 계산합니다.

    - **start_date**: 분석 시작일
    - **end_date**: 분석 종료일
    - **type**: (선택) 신입 또는 경력 (미지정 시 전체)

    Returns:
        - **period**: 분석 기간
        - **max_overlaps**: 최대 겹침 수
        - **daily_intensity**: 날짜별 경쟁 강도
            - **date**: 날짜
            - **overlap_count**: 겹침 수
            - **companies**: 채용 중인 회사 목록
    """
    # type 검증
    if type and type not in ["신입", "경력"]:
        raise HTTPException(
            status_code=400,
            detail="type은 '신입' 또는 '경력'이어야 합니다."
        )

    # 서비스 호출
    result = get_competition_intensity(
        db=db,
        start_date=start_date,
        end_date=end_date,
        type_filter=type
    )

    # 에러 처리
    if result["status"] != 200:
        raise HTTPException(
            status_code=result["status"],
            detail=result["message"]
        )

    return result

