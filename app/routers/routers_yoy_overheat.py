"""
YoY Overheat Index API Router

YoY (Year-over-Year) Overheat Index:
채용 시장의 전년 대비 과열도를 측정하는 지표입니다.
- 50 초과: 작년보다 증가 → 과열 (채용 확대)
- 50: 작년과 동일 (기준선)
- 50 미만: 작년보다 감소 → 냉각 (채용 축소)
"""
from typing import Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.config.base import get_db_readonly
from app.schemas.schemas_yoy_overheat import YoYOverheatResponse
from app.services.dashboard.yoy_overheat import calculate_yoy_overheat


router = APIRouter(
    prefix="/api/v1/position-competition",
    tags=["직무 경쟁력 분석"],
)


@router.get(
    "/yoy-overheat",
    response_model=YoYOverheatResponse,
    summary="YoY 과열도 지수 조회",
    description=(
        "채용 시장의 전년 대비 과열도를 측정하는 YoY (Year-over-Year) Overheat Index를 조회합니다.\n\n"
        "**YoY 지수 해석:**\n"
        "- **0**: 채용 없음 (극심한 냉각)\n"
        "- **25**: 작년 대비 절반 수준 (냉각)\n"
        "- **50**: 작년과 동일 (기준선)\n"
        "- **75**: 작년 대비 1.5배 (과열)\n"
        "- **100**: 작년 대비 2배 이상 (극심한 과열)\n\n"
        "**트렌드 분류:**\n"
        "- **과열**: YoY > 50 (채용 확대 중)\n"
        "- **기준**: YoY = 50 (작년 동일)\n"
        "- **냉각**: YoY < 50 (채용 축소 중)\n\n"
        "전체 시장, 직무별, 산업별 YoY 점수를 제공하여 시장 과열도를 다각도로 분석할 수 있습니다."
    ),
)
def get_yoy_overheat_index(
    year: int = Query(
        ...,
        description="조회 연도 (예: 2024)",
        ge=2020,
        example=2024,
    ),
    month: int = Query(
        ...,
        description="조회 월 (1~12)",
        ge=1,
        le=12,
        example=12,
    ),
    window_type: Literal["1month", "3month"] = Query(
        "1month",
        description=(
            "분석 기간 윈도우 타입\n"
            "- **1month**: 단일 월 분석\n"
            "- **3month**: 3개월 이동평균 (현재월, -1개월, -2개월)\n\n"
            "3개월 윈도우는 단기 변동성을 완화하여 트렌드를 명확히 보여줍니다."
        ),
    ),
    company: Optional[str] = Query(
        None,
        description=(
            "특정 회사명 필터 (부분 일치)\n\n"
            "**지원 키워드:**\n"
            "- toss, kakao, naver, line, coupang, woowahan, lg cns, hanwha, hyundai autoever\n\n"
            "미지정 시 전체 시장 기준으로 집계합니다."
        ),
        example="toss",
    ),
    db: Session = Depends(get_db_readonly),
) -> YoYOverheatResponse:
    """
    채용 시장의 전년 대비 과열도를 측정하는 YoY Overheat Index를 조회합니다.

    - **year, month**: 분석 대상 연도/월
    - **window_type**: 1month (단일월) / 3month (3개월 평균)
    - **company**: 특정 회사 필터 (예: "toss", "kakao")

    **응답 데이터:**
    - overall_yoy_score: 전체 시장 평균 YoY 점수
    - by_position: 직무별 YoY 점수 (점수 높은 순 정렬)
    - by_industry: 산업별 YoY 점수 (점수 높은 순 정렬)
    """
    try:
        data = calculate_yoy_overheat(
            db=db,
            year=year,
            month=month,
            window_type=window_type,
            company=company,
        )

        return YoYOverheatResponse(
            status=200,
            code="SUCCESS",
            message=f"{year}년 {month}월 YoY 과열도 지수 조회 성공",
            data=data,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")
