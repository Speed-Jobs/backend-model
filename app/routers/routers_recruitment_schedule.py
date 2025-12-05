"""
Recruitment Schedule Router
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
from app.db.config.base import get_db
from app.services.dashboard.recruitment_schedule import (
    get_company_recruitment_schedule,
    get_all_recruitment_schedules
)
from app.config.company_groups import COMPANY_GROUPS

router = APIRouter(
    prefix="/recruitment-schedule",
    tags=["recruitment-schedule"]
)


@router.get("/companies/{companyId}")
def get_recruitment_schedule_by_company(
    companyId: int,
    type: str = Query(..., description="채용 유형 (신입/경력)", example="신입"),
    data_type: str = Query(default="actual", description="데이터 유형 (actual: 실제 공고, predicted: 예측치, all: 전체)", example="actual"),
    start_date: str = Query(..., description="시작일 (YYYY-MM-DD 형식)", example="2025-01-01"),
    end_date: str = Query(..., description="종료일 (YYYY-MM-DD 형식)", example="2025-12-31"),
    db: Session = Depends(get_db)
):
    """
    특정 회사의 채용 일정을 상세 조회합니다.
    
    - **companyId**: 회사 ID
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
    
    # 서비스 호출
    result = get_company_recruitment_schedule(
        db=db,
        company_id=companyId,
        type_filter=type,
        start_date=start_date,
        end_date=end_date,
        data_type=data_type
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
    company_ids: Optional[str] = Query(None, description="회사 ID 리스트 (쉼표로 구분)", example="1,2,3"),
    db: Session = Depends(get_db)
):
    """
    전체 회사의 채용 일정을 조회합니다.
    
    - **type**: 신입 또는 경력
    - **data_type**: actual(실제 공고), predicted(예측치), all(전체)
    - **start_date**: 조회 시작일
    - **end_date**: 조회 종료일
    - **company_ids**: (선택) 특정 회사들만 조회 (쉼표로 구분, 예: "1,2,3")
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
    
    # company_ids 파싱
    parsed_company_ids = None
    if company_ids:
        try:
            parsed_company_ids = [int(id.strip()) for id in company_ids.split(",")]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="company_ids는 숫자를 쉼표로 구분한 형식이어야 합니다. (예: 1,2,3)"
            )
    
    # 서비스 호출
    result = get_all_recruitment_schedules(
        db=db,
        type_filter=type,
        start_date=start_date,
        end_date=end_date,
        company_ids=parsed_company_ids,
        data_type=data_type
    )
    
    # 에러 처리
    if result["status"] != 200:
        raise HTTPException(
            status_code=result["status"],
            detail=result["message"]
        )
    
    return result

