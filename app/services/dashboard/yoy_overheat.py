"""
YoY Overheat Index Service
"""
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.crud import db_yoy_overheat
from app.schemas.schemas_yoy_overheat import (
    OverallYoYData,
    PositionYoYData,
    IndustryYoYData,
    CombinedYoYData,
)
from app.utils.date_calculator import calculate_3month_period, calculate_previous_year_period


def _calculate_yoy_score(current_count: int, previous_count: int) -> float:
    """
    YoY 점수 계산

    공식: YoY(t) = min(100, max(0, Ct/Bt-1 × 50))

    Args:
        current_count: 현재 기간 건수
        previous_count: 이전 기간 (작년 동기) 건수

    Returns:
        YoY 점수 (0~100)
        - 0: 채용 없음
        - 50: 작년과 동일
        - 100: 2배 이상
    """
    if previous_count == 0:
        # 작년에 없었는데 올해 생긴 경우 → 100 (신규 과열)
        return 100.0 if current_count > 0 else 0.0

    if current_count == 0:
        # 작년에 있었는데 올해 없는 경우 → 0 (극심한 냉각)
        return 0.0

    ratio = current_count / previous_count
    yoy = ratio * 50
    return min(100.0, max(0.0, yoy))


def _get_trend(yoy_score: float) -> str:
    """
    YoY 점수를 트렌드로 변환

    Args:
        yoy_score: YoY 점수 (0~100)

    Returns:
        트렌드 ("과열" / "기준" / "냉각")
    """
    if yoy_score > 50:
        return "과열"
    elif yoy_score == 50:
        return "기준"
    else:
        return "냉각"


def analyze_overall_yoy(
    db: Session,
    start_date_str: str,
) -> OverallYoYData:
    """
    전체 시장 YoY 분석

    Args:
        db: DB 세션
        start_date_str: 종료일 (YYYY-MM-DD)

    Returns:
        OverallYoYData
    """
    # 기간 계산 (3개월 고정, 과거 3개월)
    start_date, end_date = calculate_3month_period(start_date_str)

    # 작년 동기 기간 계산 (정확히 1년 전)
    previous_start_date, previous_end_date = calculate_previous_year_period(start_date, end_date)

    # 현재 기간 데이터 조회
    current_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=start_date,
        end_date=end_date,
        company_patterns=None,
    )

    # 이전 기간 (작년 동기) 데이터 조회
    previous_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=previous_start_date,
        end_date=previous_end_date,
        company_patterns=None,
    )

    # 전체 합계 계산
    overall_current = sum(count for _, _, _, _, count in current_rows)
    overall_previous = sum(count for _, _, _, _, count in previous_rows)
    overall_yoy_score = _calculate_yoy_score(overall_current, overall_previous)

    return OverallYoYData(
        analysis_type="overall",
        overall_yoy_score=round(overall_yoy_score, 2),
        overall_trend=_get_trend(overall_yoy_score),
        overall_current_count=overall_current,
        overall_previous_count=overall_previous,
    )


def analyze_position_yoy(
    db: Session,
    start_date_str: str,
    position_id: int,
) -> PositionYoYData:
    """
    특정 직군 YoY 분석

    Args:
        db: DB 세션
        start_date_str: 종료일 (YYYY-MM-DD)
        position_id: 직군 ID

    Returns:
        PositionYoYData
    """
    from app.models.position import Position

    # 기간 계산 (3개월 고정, 과거 3개월)
    start_date, end_date = calculate_3month_period(start_date_str)

    # 작년 동기 기간 계산 (정확히 1년 전)
    previous_start_date, previous_end_date = calculate_previous_year_period(start_date, end_date)

    # 직군 정보 조회
    position = db.query(Position).filter(Position.id == position_id).first()
    if not position:
        raise ValueError(f"직군 ID {position_id}를 찾을 수 없습니다")

    # 현재 기간 데이터 조회
    current_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=start_date,
        end_date=end_date,
        company_patterns=None,
    )

    # 이전 기간 (작년 동기) 데이터 조회
    previous_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=previous_start_date,
        end_date=previous_end_date,
        company_patterns=None,
    )

    # 해당 직군 데이터만 필터링
    position_current = sum(
        count for pos_id, _, _, _, count in current_rows
        if pos_id == position_id
    )
    position_previous = sum(
        count for pos_id, _, _, _, count in previous_rows
        if pos_id == position_id
    )

    position_yoy_score = _calculate_yoy_score(position_current, position_previous)

    return PositionYoYData(
        analysis_type="position",
        position_id=position_id,
        position_name=position.name,
        position_yoy_score=round(position_yoy_score, 2),
        position_trend=_get_trend(position_yoy_score),
        position_current_count=position_current,
        position_previous_count=position_previous,
    )


def analyze_industry_yoy(
    db: Session,
    start_date_str: str,
    position_id: int,
    industry_id: int,
) -> IndustryYoYData:
    """
    특정 산업 YoY 분석

    Args:
        db: DB 세션
        start_date_str: 종료일 (YYYY-MM-DD)
        position_id: 직군 ID
        industry_id: 산업 ID

    Returns:
        IndustryYoYData
    """
    from app.models.industry import Industry

    # 기간 계산 (3개월 고정, 과거 3개월)
    start_date, end_date = calculate_3month_period(start_date_str)

    # 작년 동기 기간 계산 (정확히 1년 전)
    previous_start_date, previous_end_date = calculate_previous_year_period(start_date, end_date)

    # 산업 정보 조회
    industry = db.query(Industry).filter(Industry.id == industry_id).first()
    if not industry:
        raise ValueError(f"산업 ID {industry_id}를 찾을 수 없습니다")

    if industry.position_id != position_id:
        raise ValueError(f"산업 '{industry.name}'은 직군 ID {position_id}에 속하지 않습니다")

    # 현재 기간 데이터 조회
    current_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=start_date,
        end_date=end_date,
        company_patterns=None,
    )

    # 이전 기간 (작년 동기) 데이터 조회
    previous_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=previous_start_date,
        end_date=previous_end_date,
        company_patterns=None,
    )

    # 해당 산업 데이터만 필터링
    industry_current = sum(
        count for _, _, ind_id, _, count in current_rows
        if ind_id == industry_id
    )
    industry_previous = sum(
        count for _, _, ind_id, _, count in previous_rows
        if ind_id == industry_id
    )

    industry_yoy_score = _calculate_yoy_score(industry_current, industry_previous)

    return IndustryYoYData(
        analysis_type="industry",
        industry_id=industry_id,
        industry_name=industry.name,
        industry_yoy_score=round(industry_yoy_score, 2),
        industry_trend=_get_trend(industry_yoy_score),
        industry_current_count=industry_current,
        industry_previous_count=industry_previous,
    )


def analyze_combined_yoy_insights(
    db: Session,
    start_date_str: str,
    position_id: Optional[int] = None,
    industry_id: Optional[int] = None,
) -> CombinedYoYData:
    """
    통합 YoY 인사이트 분석 (Total + Position + Industry)

    항상 Total 인사이트를 포함하고, position_id와 industry_id가 있으면 해당 인사이트도 포함합니다.

    Args:
        db: DB 세션
        start_date_str: 종료일 (YYYY-MM-DD)
        position_id: 직군 ID (선택)
        industry_id: 산업 ID (선택, position_id가 필수)

    Returns:
        CombinedYoYData
    """
    # 1. Total 시장 분석 (항상 포함)
    total_insight = analyze_overall_yoy(
        db=db,
        start_date_str=start_date_str,
    )

    # 2. Position 분석 (position_id가 있으면 포함)
    position_insight = None
    if position_id:
        position_insight = analyze_position_yoy(
            db=db,
            start_date_str=start_date_str,
            position_id=position_id,
        )

    # 3. Industry 분석 (industry_id가 있으면 포함)
    industry_insight = None
    if industry_id and position_id:
        industry_insight = analyze_industry_yoy(
            db=db,
            start_date_str=start_date_str,
            position_id=position_id,
            industry_id=industry_id,
        )

    return CombinedYoYData(
        analysis_type="combined",
        total_insight=total_insight,
        position_insight=position_insight,
        industry_insight=industry_insight,
    )
