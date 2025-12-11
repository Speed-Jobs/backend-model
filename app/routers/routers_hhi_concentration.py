"""
HHI Concentration Index API Router

HHI (Herfindahl-Hirschman Index):
채용 시장의 직무별 집중도를 측정하는 지표입니다.
- 0 ~ 0.15: "분산" (다양한 직무 골고루 분포)
- 0.15 ~ 0.25: "부분집중"
- 0.25+: "쏠림" (특정 직무 독점)
"""
from typing import Optional, Literal, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.config.base import get_db_readonly
from app.schemas.schemas_hhi_concentration import (
    HHIAnalysisResponse,
)
from app.services.dashboard.hhi_concentration import (
    analyze_overall_market,
    analyze_position_industries,
    analyze_specific_industry,
    analyze_combined_insights,
)
from app.utils.position_industry_loader import POSITION_NAMES, INDUSTRY_NAMES


router = APIRouter(
    prefix="/api/v1/position-competition",
    tags=["직무 경쟁력 분석"],
)


@router.get(
    "/hhi-analysis",
    response_model=HHIAnalysisResponse,
    summary="HHI 집중도 분석 (신규, 3개월 고정)",
    description=(
        "채용 시장의 직무별 집중도를 측정하는 HHI 분석을 제공합니다.\\n\\n"
        "**분석 기간:** 종료일 기준 과거 3개월 고정 (종료일 - 3개월 ~ 종료일)\\n\\n"
        "**통합 인사이트 제공:**\\n"
        "- **항상 Total 인사이트 포함**\\n"
        "- **position_name 지정 시:** Total + Position 인사이트\\n"
        "- **position_name + industry_name 지정 시:** Total + Position + Industry 인사이트\\n\\n"
        "**응답 구조:**\\n"
        "- HHI 지수 (시각화용)\\n"
        "- 해석 (집중도 수준, 경쟁 난이도)\\n"
        "- 인사이트 (HHI, CR₂, Entropy 기반 생성)\\n\\n"
        "**참고:**\\n"
        "- CR₂와 Entropy는 내부적으로 계산되어 인사이트 생성에 활용되며, 응답에는 포함되지 않습니다.\\n"
        "- position_name, industry_name만 사용 가능합니다 (ID는 사용 불가).\\n"
        "- industry_name을 지정하려면 position_name이 필수입니다."
    ),
)
def get_hhi_analysis(
    start_date: str = Query(
        ...,
        description="종료일 (YYYY-MM-DD)\\n\\n시작일은 자동으로 종료일 - 3개월로 계산됩니다. (과거 3개월 분석)",
        example="2025-09-01",
        regex=r"^\d{4}-\d{2}-\d{2}$",
    ),
    position_name: Optional[str] = Query(
        None,
        description=(
            "직군명 (선택)\\n\\n"
            "- **미지정 시:** 전체 시장 분석 (시나리오 1)\\n"
            "- **지정 시:** 해당 직군 내 산업별 분석 (시나리오 2)\\n"
            "- **industry_name과 함께 지정 시:** 통합 인사이트 (Total + Position + Industry, 시나리오 3)\\n\\n"
            "**사용 가능한 직군명:** /positions 엔드포인트에서 조회 가능"
        ),
        example="Software Development",
        enum=POSITION_NAMES if POSITION_NAMES else None,  # ✅ Swagger UI에서 dropdown으로 표시
    ),
    industry_name: Optional[str] = Query(
        None,
        description=(
            "산업명 (선택)\\n\\n"
            "**주의:** industry_name을 지정하려면 position_name이 필수입니다.\\n"
            "지정 시 Total, Position, Industry 인사이트를 모두 제공합니다 (시나리오 3)."
        ),
        example="Front-end Development",
        enum=INDUSTRY_NAMES if INDUSTRY_NAMES else None,  # ✅ Swagger UI에서 dropdown으로 표시
    ),
    db: Session = Depends(get_db_readonly),
) -> HHIAnalysisResponse:
    """
    HHI 집중도 분석 API (신규)

    **시나리오 1: 전체 시장 분석**
    - GET /api/v1/position-competition/hhi-analysis?start_date=2025-09-01
    - 모든 직군의 채용 집중도 분석

    **시나리오 2: 직군별 분석**
    - GET /api/v1/position-competition/hhi-analysis?start_date=2025-09-01&position_name=Software Development
    - 특정 직군 내 산업별 채용 집중도 분석

    **시나리오 3: 통합 인사이트 (Total + Position + Industry)**
    - GET /api/v1/position-competition/hhi-analysis?start_date=2025-09-01&position_name=Software Development&industry_name=Front-end Development
    - Total 시장, Position, Industry 인사이트를 모두 제공
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
        data = analyze_combined_insights(
            db=db,
            start_date_str=start_date,
            position_id=resolved_position_id,
            industry_id=resolved_industry_id,
        )
        
        # 메시지 생성
        if resolved_industry_id:
            message = f"{data.period.start} ~ {data.period.end} '{data.industry_insight.industry_name}' 산업 통합 인사이트 완료"
        elif resolved_position_id:
            message = f"{data.period.start} ~ {data.period.end} '{data.position_insight.position_name}' 직군 통합 인사이트 완료"
        else:
            message = f"{data.period.start} ~ {data.period.end} 전체 시장 HHI 분석 완료"

        return HHIAnalysisResponse(
            status=200,
            code="SUCCESS",
            message=message,
            data=data,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")


