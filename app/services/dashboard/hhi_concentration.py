"""
HHI Concentration Index Service
"""
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.db.crud import db_hhi_concentration
from app.config.company_groups import get_company_patterns
from app.schemas.schemas_hhi_concentration import (
    HHIConcentrationInsightData,
    HHIScore,
    PositionConcentration,
    IndustryConcentration,
)


def _calculate_hhi(counts: List[int]) -> float:
    """
    HHI 지수 계산

    공식: HHI = Σ(si²) where si = 항목i의 점유율

    Args:
        counts: 항목별 건수 리스트

    Returns:
        HHI 지수 (0~1)
    """
    total = sum(counts)
    if total == 0:
        return 0.0

    hhi = sum((count / total) ** 2 for count in counts)
    return hhi


def _interpret_hhi(hhi_value: float) -> Tuple[str, str]:
    """
    HHI 값을 수준과 해석으로 변환

    Args:
        hhi_value: HHI 지수 (0~1)

    Returns:
        (level, interpretation)
        - level: "분산" / "부분집중" / "쏠림"
        - interpretation: 해석 텍스트
    """
    if hhi_value < 0.15:
        return "분산", "다양한 직무에 채용이 골고루 분포되어 있습니다. 시장 경쟁이 다양화되어 있는 상태입니다."
    elif hhi_value < 0.25:
        return "부분집중", "일부 직무에 채용이 집중되는 경향이 있습니다. 시장 다양화가 일부 진행 중입니다."
    else:
        return "쏠림", "특정 직무에 채용이 과도하게 집중되어 있습니다. 시장 다양화가 필요한 상황입니다."


def _generate_insights(
    position_data: List[Tuple[int, str, int]],
    industry_data: List[Tuple[int, str, int]],
    position_hhi_value: float,
    industry_hhi_value: float,
) -> str:
    """
    종합 인사이트 텍스트 생성

    Args:
        position_data: 직무별 데이터 (position_id, position_name, count)
        industry_data: 산업별 데이터 (industry_id, industry_name, count)
        position_hhi_value: 직무별 HHI
        industry_hhi_value: 산업별 HHI

    Returns:
        종합 시사점 텍스트
    """
    total_positions = sum(count for _, _, count in position_data)
    total_industries = sum(count for _, _, count in industry_data)

    insights_parts = []

    # 상위 직무 분석
    if position_data:
        top_position = position_data[0]
        top_share = (top_position[2] / total_positions * 100) if total_positions > 0 else 0
        insights_parts.append(
            f"{top_position[1]} 직무가 전체 채용의 {top_share:.1f}%를 차지하고 있습니다."
        )

    # 집중도 해석
    position_level, _ = _interpret_hhi(position_hhi_value)
    if position_level == "쏠림":
        insights_parts.append(
            "직무별 채용이 특정 분야에 과도하게 집중되어 있어, 포트폴리오 다양화가 권장됩니다."
        )
    elif position_level == "분산":
        insights_parts.append(
            "직무별 채용이 다양하게 분포되어 있어, 균형 잡힌 인재 확보가 가능한 상태입니다."
        )
    else:
        insights_parts.append(
            "일부 직무에 채용이 집중되는 경향이 있으나, 전반적으로 다양화가 진행 중입니다."
        )

    # 산업별 분석
    if industry_data:
        top_industry = industry_data[0]
        top_ind_share = (top_industry[2] / total_industries * 100) if total_industries > 0 else 0
        if top_ind_share > 30:
            insights_parts.append(
                f"{top_industry[1]} 분야가 {top_ind_share:.1f}%로 가장 높은 비중을 차지하고 있습니다."
            )

    # 권장사항
    if position_hhi_value >= 0.25:
        insights_parts.append(
            "시장 쏠림 현상을 완화하기 위해 비즈니스 직군 및 지원 직군 채용 확대를 고려해보세요."
        )

    return " ".join(insights_parts)


def generate_hhi_concentration_insight(
    db: Session,
    year: Optional[int],
    month: Optional[int],
    window_type: str,
    start_date_str: Optional[str],
    end_date_str: Optional[str],
    company: Optional[str],
) -> HHIConcentrationInsightData:
    """
    HHI Concentration 인사이트 생성

    Args:
        db: DB 세션
        year: 조회 연도 (window_type="1month"인 경우 필수)
        month: 조회 월 (window_type="1month"인 경우 필수)
        window_type: "1month" (단일월) 또는 "period" (사용자 지정 기간)
        start_date_str: 시작일 (YYYY-MM-DD, window_type="period"인 경우 필수)
        end_date_str: 종료일 (YYYY-MM-DD, window_type="period"인 경우 필수)
        company: 회사명 키워드 (None이면 전체)

    Returns:
        HHIConcentrationInsightData
    """
    if window_type not in ["1month", "period"]:
        raise ValueError("window_type은 '1month' 또는 'period'여야 합니다.")

    # 기간 계산
    if window_type == "1month":
        if not year or not month:
            raise ValueError("window_type='1month'인 경우 year와 month가 필수입니다.")

        start_date = date(year, month, 1)
        # 다음 달 1일 - 1일 = 이번 달 말일
        if month == 12:
            end_date = date(year, 12, 31)
        else:
            end_date = date(year, month + 1, 1)
            from datetime import timedelta
            end_date = end_date - timedelta(days=1)

        period_str = f"{year}-{month:02d}"

    else:  # period
        if not start_date_str or not end_date_str:
            raise ValueError("window_type='period'인 경우 start_date와 end_date가 필수입니다.")

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        period_str = f"{start_date_str} ~ {end_date_str}"

    # 회사 패턴 변환
    company_patterns = None
    if company:
        company_patterns = get_company_patterns(company)

    # DB 조회
    position_rows = db_hhi_concentration.get_position_recruit_counts(
        db=db,
        start_date=start_date,
        end_date=end_date,
        company_patterns=company_patterns,
    )

    industry_rows = db_hhi_concentration.get_industry_recruit_counts(
        db=db,
        start_date=start_date,
        end_date=end_date,
        company_patterns=company_patterns,
    )

    # 데이터 추출
    position_counts = [count for _, _, count in position_rows]
    industry_counts = [count for _, _, count in industry_rows]
    total_count = sum(position_counts)

    # HHI 계산
    position_hhi_value = _calculate_hhi(position_counts)
    industry_hhi_value = _calculate_hhi(industry_counts)

    # 전체 HHI (position + industry 통합)
    overall_counts = position_counts + industry_counts
    overall_hhi_value = _calculate_hhi(overall_counts) if overall_counts else 0.0

    # 해석
    overall_level, overall_interp = _interpret_hhi(overall_hhi_value)
    position_level, position_interp = _interpret_hhi(position_hhi_value)
    industry_level, industry_interp = _interpret_hhi(industry_hhi_value)

    # 상위 직무/산업 (최대 5개)
    position_rows_sorted = sorted(position_rows, key=lambda x: x[2], reverse=True)[:5]
    industry_rows_sorted = sorted(industry_rows, key=lambda x: x[2], reverse=True)[:5]

    top_positions = [
        PositionConcentration(
            position_id=pos_id,
            position_name=pos_name,
            count=count,
            share_percentage=round((count / total_count * 100) if total_count > 0 else 0, 2),
            rank=idx + 1,
        )
        for idx, (pos_id, pos_name, count) in enumerate(position_rows_sorted)
    ]

    top_industries = [
        IndustryConcentration(
            industry_id=ind_id,
            industry_name=ind_name,
            count=count,
            share_percentage=round((count / total_count * 100) if total_count > 0 else 0, 2),
            rank=idx + 1,
        )
        for idx, (ind_id, ind_name, count) in enumerate(industry_rows_sorted)
    ]

    # 인사이트 생성
    insights = _generate_insights(
        position_data=position_rows,
        industry_data=industry_rows,
        position_hhi_value=position_hhi_value,
        industry_hhi_value=industry_hhi_value,
    )

    return HHIConcentrationInsightData(
        period=period_str,
        window_type=window_type,
        total_count=total_count,
        overall_hhi=HHIScore(
            hhi_value=round(overall_hhi_value, 4),
            level=overall_level,
            interpretation=overall_interp,
        ),
        position_hhi=HHIScore(
            hhi_value=round(position_hhi_value, 4),
            level=position_level,
            interpretation=position_interp,
        ),
        industry_hhi=HHIScore(
            hhi_value=round(industry_hhi_value, 4),
            level=industry_level,
            interpretation=industry_interp,
        ),
        top_positions=top_positions,
        top_industries=top_industries,
        insights=insights,
    )
