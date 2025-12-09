"""
YoY Overheat Index Service
"""
from datetime import date
from typing import Dict, List, Optional
from collections import defaultdict

from sqlalchemy.orm import Session

from app.db.crud import db_yoy_overheat
from app.config.company_groups import get_company_patterns
from app.schemas.schemas_yoy_overheat import (
    YoYOverheatData,
    YoYScoreByPosition,
    YoYScoreByIndustry,
)


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


def calculate_yoy_overheat(
    db: Session,
    year: int,
    month: int,
    window_type: str,
    company: Optional[str],
) -> YoYOverheatData:
    """
    YoY Overheat 점수 계산

    Args:
        db: DB 세션
        year: 조회 연도
        month: 조회 월 (1~12)
        window_type: "1month" (단일월) 또는 "3month" (3개월 평균)
        company: 회사명 키워드 (None이면 전체)

    Returns:
        YoYOverheatData
    """
    if window_type not in ["1month", "3month"]:
        raise ValueError("window_type은 '1month' 또는 '3month'여야 합니다.")

    window_months = 1 if window_type == "1month" else 3

    # 회사 패턴 변환
    company_patterns = None
    if company:
        company_patterns = get_company_patterns(company)

    # 현재 기간 데이터 조회
    current_rows = db_yoy_overheat.get_monthly_recruit_counts(
        db=db,
        year=year,
        month=month,
        window_months=window_months,
        company_patterns=company_patterns,
    )

    # 이전 기간 (작년 동기) 데이터 조회
    previous_year = year - 1
    previous_rows = db_yoy_overheat.get_monthly_recruit_counts(
        db=db,
        year=previous_year,
        month=month,
        window_months=window_months,
        company_patterns=company_patterns,
    )

    # 데이터 집계 (position_id, industry_id 기준)
    current_data: Dict[tuple, int] = defaultdict(int)
    previous_data: Dict[tuple, int] = defaultdict(int)

    position_names: Dict[int, str] = {}
    industry_names: Dict[int, str] = {}

    for year_val, month_val, pos_id, pos_name, ind_id, ind_name, count in current_rows:
        key = (pos_id, ind_id)
        current_data[key] += count
        position_names[pos_id] = pos_name
        industry_names[ind_id] = ind_name

    for year_val, month_val, pos_id, pos_name, ind_id, ind_name, count in previous_rows:
        key = (pos_id, ind_id)
        previous_data[key] += count
        position_names[pos_id] = pos_name
        industry_names[ind_id] = ind_name

    # 전체 키 합집합
    all_keys = set(current_data.keys()) | set(previous_data.keys())

    # Position별 집계
    position_totals_current: Dict[int, int] = defaultdict(int)
    position_totals_previous: Dict[int, int] = defaultdict(int)

    # Industry별 집계
    industry_totals_current: Dict[int, int] = defaultdict(int)
    industry_totals_previous: Dict[int, int] = defaultdict(int)

    for pos_id, ind_id in all_keys:
        curr = current_data.get((pos_id, ind_id), 0)
        prev = previous_data.get((pos_id, ind_id), 0)

        position_totals_current[pos_id] += curr
        position_totals_previous[pos_id] += prev

        industry_totals_current[ind_id] += curr
        industry_totals_previous[ind_id] += prev

    # 전체 합계
    overall_current = sum(position_totals_current.values())
    overall_previous = sum(position_totals_previous.values())
    overall_yoy_score = _calculate_yoy_score(overall_current, overall_previous)

    # Position별 YoY 점수 계산
    by_position: List[YoYScoreByPosition] = []
    for pos_id in set(position_totals_current.keys()) | set(position_totals_previous.keys()):
        curr = position_totals_current.get(pos_id, 0)
        prev = position_totals_previous.get(pos_id, 0)
        yoy = _calculate_yoy_score(curr, prev)
        trend = _get_trend(yoy)

        by_position.append(
            YoYScoreByPosition(
                position_id=pos_id,
                position_name=position_names.get(pos_id, "Unknown"),
                yoy_score=round(yoy, 2),
                current_count=curr,
                previous_year_count=prev,
                trend=trend,
            )
        )

    # Industry별 YoY 점수 계산
    by_industry: List[YoYScoreByIndustry] = []
    for ind_id in set(industry_totals_current.keys()) | set(industry_totals_previous.keys()):
        curr = industry_totals_current.get(ind_id, 0)
        prev = industry_totals_previous.get(ind_id, 0)
        yoy = _calculate_yoy_score(curr, prev)
        trend = _get_trend(yoy)

        by_industry.append(
            YoYScoreByIndustry(
                industry_id=ind_id,
                industry_name=industry_names.get(ind_id, "Unknown"),
                yoy_score=round(yoy, 2),
                current_count=curr,
                previous_year_count=prev,
                trend=trend,
            )
        )

    # 점수 높은 순 정렬
    by_position.sort(key=lambda x: x.yoy_score, reverse=True)
    by_industry.sort(key=lambda x: x.yoy_score, reverse=True)

    return YoYOverheatData(
        year=year,
        month=month,
        window_type=window_type,
        overall_yoy_score=round(overall_yoy_score, 2),
        overall_trend=_get_trend(overall_yoy_score),
        by_position=by_position,
        by_industry=by_industry,
    )
