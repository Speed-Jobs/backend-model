"""
YoY Overheat Index API Router

YoY (Year-over-Year) Overheat Index:
채용 시장의 전년 대비 과열도를 측정하는 지표입니다.
- 50 초과: 작년보다 증가 → 과열 (채용 확대)
- 50: 작년과 동일 (기준선)
- 50 미만: 작년보다 감소 → 냉각 (채용 축소)
"""
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.config.base import get_db_readonly
from app.schemas.schemas_yoy_overheat import YoYOverheatResponse
from app.services.dashboard.yoy_overheat import analyze_combined_yoy_insights
from app.utils.position_industry_loader import POSITION_NAMES, INDUSTRY_NAMES


router = APIRouter(
    prefix="/api/v1/position-competition",
    tags=["직무 경쟁력 분석"],
)


@router.get(
    "/yoy-overheat",
    response_model=YoYOverheatResponse,
    summary="YoY 과열도 지수 조회 (3개월 고정)",
    description=(
        "채용 시장의 전년 대비 과열도를 측정하는 YoY (Year-over-Year) Overheat Index를 조회합니다.\\n\\n"
        "**분석 기간:** 종료일 기준 과거 3개월 고정 (종료일 - 3개월 ~ 종료일)\\n\\n"
        "**통합 인사이트 제공:**\\n"
        "- **항상 Total 인사이트 포함**\\n"
        "- **position_name 지정 시:** Total + Position 인사이트\\n"
        "- **position_name + industry_name 지정 시:** Total + Position + Industry 인사이트\\n\\n"
        "**YoY 지수 해석:**\\n"
        "- **0**: 채용 없음 (극심한 냉각)\\n"
        "- **25**: 작년 대비 절반 수준 (냉각)\\n"
        "- **50**: 작년과 동일 (기준선)\\n"
        "- **75**: 작년 대비 1.5배 (과열)\\n"
        "- **100**: 작년 대비 2배 이상 (극심한 과열)\\n\\n"
        "**트렌드 분류:**\\n"
        "- **과열**: YoY > 50 (채용 확대 중)\\n"
        "- **기준**: YoY = 50 (작년 동일)\\n"
        "- **냉각**: YoY < 50 (채용 축소 중)\\n\\n"
        "**참고:**\\n"
        "- position_name, industry_name만 사용 가능합니다 (ID는 사용 불가).\\n"
        "- industry_name을 지정하려면 position_name이 필수입니다."
    ),
)
def get_yoy_overheat_index(
    start_date: str = Query(
        ...,
        description="종료일 (YYYY-MM-DD)\\n\\n시작일은 자동으로 종료일 - 3개월로 계산됩니다. (과거 3개월 분석)",
        example="2025-12-01",
        regex=r"^\d{4}-\d{2}-\d{2}$",
    ),
    position_name: Optional[str] = Query(
        None,
        description=(
            "직군명 (선택)\\n\\n"
            "- **미지정 시:** 전체 시장 분석 (시나리오 1)\\n"
            "- **지정 시:** 해당 직군 내 산업별 분석 (시나리오 2)\\n"
            "- **industry_name과 함께 지정 시:** 통합 인사이트 (Total + Position + Industry, 시나리오 3)"
        ),
        example="Software Development",
        enum=POSITION_NAMES if POSITION_NAMES else None,
    ),
    industry_name: Optional[str] = Query(
        None,
        description=(
            "산업명 (선택)\\n\\n"
            "**주의:** industry_name을 지정하려면 position_name이 필수입니다.\\n"
            "지정 시 Total, Position, Industry 인사이트를 모두 제공합니다 (시나리오 3)."
        ),
        example="Front-end Development",
        enum=INDUSTRY_NAMES if INDUSTRY_NAMES else None,
    ),
    db: Session = Depends(get_db_readonly),
) -> YoYOverheatResponse:
    """
    채용 시장의 전년 대비 과열도를 측정하는 YoY Overheat Index를 조회합니다.

    - **start_date**: 분석 종료일 (YYYY-MM-DD). 시작일은 자동으로 종료일 - 3개월로 계산됩니다.
    - **position_name**: 직군명 (선택)
    - **industry_name**: 산업명 (선택, position_name 필수)

    **응답 데이터:**
    - total_insight: 전체 시장 YoY 인사이트 (항상 포함)
    - position_insight: 직군별 YoY 인사이트 (position_name 지정 시)
    - industry_insight: 산업별 YoY 인사이트 (position_name + industry_name 지정 시)
    """
    try:
        from app.models.position import Position
        from app.models.industry import Industry

        # 파라미터 검증
        if industry_name and not position_name:
            raise ValueError("industry_name을 지정하려면 position_name이 필수입니다")

        # Name → ID 변환
        resolved_position_id = None
        resolved_industry_id = None

        if position_name:
            position = db.query(Position).filter(Position.name == position_name).first()
            if not position:
                raise ValueError(f"직군명 '{position_name}'을 찾을 수 없습니다")
            resolved_position_id = position.id

        if industry_name:
            if not resolved_position_id:
                raise ValueError("industry_name을 사용하려면 position_name이 필요합니다")

            industry = db.query(Industry).filter(
                Industry.name == industry_name,
                Industry.position_id == resolved_position_id
            ).first()
            if not industry:
                raise ValueError(f"'{position_name}' 직군 내에 산업명 '{industry_name}'을 찾을 수 없습니다")
            resolved_industry_id = industry.id

        # 항상 통합 인사이트 반환 (Total + Position + Industry)
        data = analyze_combined_yoy_insights(
            db=db,
            start_date_str=start_date,
            position_id=resolved_position_id,
            industry_id=resolved_industry_id,
        )

        # 메시지 생성
        if resolved_industry_id:
            message = f"'{data.industry_insight.industry_name}' 산업 통합 YoY 인사이트 완료"
        elif resolved_position_id:
            message = f"'{data.position_insight.position_name}' 직군 통합 YoY 인사이트 완료"
        else:
            message = f"전체 시장 YoY 과열도 지수 조회 성공"

        return YoYOverheatResponse(
            status=200,
            code="SUCCESS",
            message=message,
            data=data,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")
