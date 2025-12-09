"""
HHI Concentration Index API Router

HHI (Herfindahl-Hirschman Index):
채용 시장의 직무별 집중도를 측정하는 지표입니다.
- 0 ~ 0.15: "분산" (다양한 직무 골고루 분포)
- 0.15 ~ 0.25: "부분집중"
- 0.25+: "쏠림" (특정 직무 독점)
"""
from typing import Optional, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.config.base import get_db_readonly
from app.schemas.schemas_hhi_concentration import HHIConcentrationInsightResponse
from app.services.dashboard.hhi_concentration import generate_hhi_concentration_insight


router = APIRouter(
    prefix="/api/v1/position-competition",
    tags=["직무 경쟁력 분석"],
)


@router.get(
    "/hhi-concentration",
    response_model=HHIConcentrationInsightResponse,
    summary="HHI 집중도 인사이트 조회",
    description=(
        "채용 시장의 직무별 집중도를 측정하는 HHI (Herfindahl-Hirschman Index) 인사이트를 조회합니다.\n\n"
        "**HHI 지수 해석:**\n"
        "- **0 ~ 0.15**: 분산 - 다양한 직무에 채용이 골고루 분포 (시장 경쟁 다양화)\n"
        "- **0.15 ~ 0.25**: 부분집중 - 일부 직무에 채용이 집중되는 경향\n"
        "- **0.25 이상**: 쏠림 - 특정 직무에 채용이 과도하게 집중 (시장 독점 경향)\n\n"
        "**인사이트 제공 내용:**\n"
        "- 전체/직무별/산업별 HHI 지수 및 해석\n"
        "- 점유율 상위 직무/산업 목록 (Top 5)\n"
        "- 시장 트렌드 분석 및 채용 전략 권장사항\n\n"
        "대시보드에 표시할 수 있도록 인사이트 텍스트 중심으로 응답합니다."
    ),
)
def get_hhi_concentration_insight(
    year: Optional[int] = Query(
        None,
        description=(
            "조회 연도 (예: 2024)\n\n"
            "**window_type='1month'인 경우 필수**"
        ),
        ge=2020,
        example=2024,
    ),
    month: Optional[int] = Query(
        None,
        description=(
            "조회 월 (1~12)\n\n"
            "**window_type='1month'인 경우 필수**"
        ),
        ge=1,
        le=12,
        example=12,
    ),
    window_type: Literal["1month", "period"] = Query(
        "1month",
        description=(
            "분석 기간 윈도우 타입\n"
            "- **1month**: 단일 월 분석 (year, month 필수)\n"
            "- **period**: 사용자 지정 기간 (start_date, end_date 필수)\n\n"
            "월별 추적이 필요한 경우 '1month', 특정 기간 분석이 필요한 경우 'period'를 사용하세요."
        ),
    ),
    start_date: Optional[str] = Query(
        None,
        description=(
            "시작일 (YYYY-MM-DD)\n\n"
            "**window_type='period'인 경우 필수**"
        ),
        example="2024-10-01",
    ),
    end_date: Optional[str] = Query(
        None,
        description=(
            "종료일 (YYYY-MM-DD)\n\n"
            "**window_type='period'인 경우 필수**"
        ),
        example="2024-12-31",
    ),
    company: Optional[str] = Query(
        None,
        description=(
            "특정 회사명 필터 (부분 일치)\n\n"
            "**지원 키워드:**\n"
            "- toss, kakao, naver, line, coupang, woowahan, lg cns, hanwha, hyundai autoever\n\n"
            "미지정 시 전체 시장 기준으로 집계합니다."
        ),
        example="kakao",
    ),
    db: Session = Depends(get_db_readonly),
) -> HHIConcentrationInsightResponse:
    """
    채용 시장의 직무별 집중도를 측정하는 HHI Concentration 인사이트를 조회합니다.

    **파라미터 조합:**
    1. **window_type='1month'**: year + month 필수
    2. **window_type='period'**: start_date + end_date 필수

    **응답 데이터:**
    - overall_hhi, position_hhi, industry_hhi: HHI 지수 및 해석
    - top_positions, top_industries: 점유율 상위 직무/산업 (Top 5)
    - insights: 종합 시사점 및 채용 전략 권장사항 (텍스트)

    **활용 예시:**
    - 대시보드에 인사이트 텍스트 표시
    - 시장 집중도 트렌드 모니터링
    - 포트폴리오 다양화 전략 수립
    """
    try:
        # 파라미터 검증
        if window_type == "1month":
            if not year or not month:
                raise ValueError("window_type='1month'인 경우 year와 month가 필수입니다.")
        elif window_type == "period":
            if not start_date or not end_date:
                raise ValueError("window_type='period'인 경우 start_date와 end_date가 필수입니다.")

        data = generate_hhi_concentration_insight(
            db=db,
            year=year,
            month=month,
            window_type=window_type,
            start_date_str=start_date,
            end_date_str=end_date,
            company=company,
        )

        period_desc = data.period
        return HHIConcentrationInsightResponse(
            status=200,
            code="SUCCESS",
            message=f"{period_desc} HHI 집중도 인사이트 조회 성공",
            data=data,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")
