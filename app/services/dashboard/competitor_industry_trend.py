"""
직군별 통계 Service
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.db.crud import db_competitor_industry_trend
from app.config.company_groups import get_company_patterns
from app.schemas.schemas_competitor_industry_trend import (
    JobRoleStatisticsData,
    JobRoleStatistic,
    IndustryStatistic,
    PeriodSummary,
)


@dataclass
class _PeriodRange:
    current_start: date
    current_end: date
    previous_start: date
    previous_end: date


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _shift_months(d: date, months: int) -> date:
    """월 단위 이동 (연/월 보정 및 말일 처리 포함)"""
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    # 말일 보정
    for day in (31, 30, 29, 28):
        try:
            return date(year, month, min(d.day, day))
        except ValueError:
            continue
    # fallback
    return date(year, month, 1)


def _get_quarter_start(d: date) -> date:
    if d.month <= 3:
        return date(d.year, 1, 1)
    if d.month <= 6:
        return date(d.year, 4, 1)
    if d.month <= 9:
        return date(d.year, 7, 1)
    return date(d.year, 10, 1)


def _calculate_periods(
    timeframe: str,
    start_date_str: Optional[str],
    end_date_str: Optional[str],
) -> _PeriodRange:
    """현재/이전 기간 계산 (동기간 기준)"""
    today = date.today()
    start_date = _parse_date(start_date_str)
    end_date = _parse_date(end_date_str) or today

    if timeframe == "monthly_same_period":
        if not start_date:
            # 이번 달 1일 ~ 오늘 / 지난 달 1일 ~ 지난 달 동일 일자
            current_start = date(today.year, today.month, 1)
            current_end = end_date
        else:
            current_start = start_date
            current_end = end_date

        previous_start = _shift_months(current_start, -1)
        previous_end = _shift_months(current_end, -1)

    elif timeframe == "quarterly_same_period":
        if not start_date:
            q_start = _get_quarter_start(today)
            current_start = q_start
            current_end = end_date
        else:
            current_start = start_date
            current_end = end_date

        previous_start = _shift_months(current_start, -3)
        previous_end = _shift_months(current_end, -3)

    else:
        raise ValueError(f"지원하지 않는 timeframe 입니다: {timeframe}")

    # 날짜 정합성 보장
    if current_start > current_end:
        raise ValueError("current 기간의 start_date가 end_date보다 뒤입니다.")
    if previous_start > previous_end:
        raise ValueError("previous 기간의 start_date가 end_date보다 뒤입니다.")

    return _PeriodRange(
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
    )


def _classify_category(position_name: str) -> str:
    """포지션 이름을 Tech / Biz / BizSupporting 으로 분류"""
    name = (position_name or "").lower()

    tech_keywords = [
        "개발",
        "developer",
        "engineer",
        "engineering",
        "software",
        "data",
        "ai",
        "ml",
        "infra",
        "platform",
        "devops",
        "backend",
        "front",
        "mobile",
    ]
    biz_keywords = [
        "sales",
        "영업",
        "사업",
        "biz",
        "business",
        "consult",
        "컨설턴트",
        "전략",
        "기획",
        "마케팅",
        "pm",
        "product manager",
        "Domain Expert"," Consulting"
    ]
    support_keywords = [
        "Biz. Supporting",
        "support",
    ]

    if any(k in name for k in tech_keywords):
        return "Tech"
    if any(k in name for k in biz_keywords):
        return "Biz"
    if any(k in name for k in support_keywords):
        return "BizSupporting"
    # 기본은 Tech로 두지 않고 기타로 분류해서 제외
    return "Other"


def _aggregate_period_result(
    rows: List[Tuple[int, str, int, str, int]],
    category: str,
) -> Tuple[Dict[str, Dict[str, int]], int]:
    """
    한 기간(current 또는 previous)에 대한 집계.

    Returns:
        (job_role_map, total_count)
        job_role_map: {job_role_name: {"total": int, "industries": {industry_name: int}}}
    """
    job_roles: Dict[str, Dict[str, object]] = {}
    total = 0

    for position_id, position_name, industry_id, industry_name, count in rows:
        cat = _classify_category(position_name)
        if cat != category:
            continue

        job = job_roles.setdefault(
            position_name,
            {"total": 0, "industries": {}},
        )
        job["total"] = int(job["total"]) + int(count)
        industries: Dict[str, int] = job["industries"]  # type: ignore
        industries[industry_name] = industries.get(industry_name, 0) + int(count)
        total += int(count)

    # 타입 정리
    job_roles_clean: Dict[str, Dict[str, int]] = {}
    for name, data in job_roles.items():
        job_roles_clean[name] = {
            "total": int(data["total"]),  # type: ignore
        }
        # 산업별 카운트는 별도에서 사용
        job_roles_clean[name + "::__industries__"] = data["industries"]  # type: ignore

    return job_roles_clean, total


def _aggregate_optimized_result(
    rows: List[Tuple[str, str, int]],
) -> Tuple[Dict[str, Dict[str, int]], int]:
    """
    최적화된 쿼리 결과 집계 (카테고리 필터링은 SQL에서 완료됨)
    
    Args:
        rows: List of (position_name, industry_name, count)
    
    Returns:
        (job_role_map, total_count)
        job_role_map: {job_role_name: {"total": int, "industries": {industry_name: int}}}
    """
    job_roles: Dict[str, Dict[str, object]] = {}
    total = 0

    for position_name, industry_name, count in rows:
        job = job_roles.setdefault(
            position_name,
            {"total": 0, "industries": {}},
        )
        job["total"] = int(job["total"]) + int(count)
        industries: Dict[str, int] = job["industries"]  # type: ignore
        industries[industry_name] = industries.get(industry_name, 0) + int(count)
        total += int(count)

    # 타입 정리
    job_roles_clean: Dict[str, Dict[str, int]] = {}
    for name, data in job_roles.items():
        job_roles_clean[name] = {
            "total": int(data["total"]),  # type: ignore
        }
        job_roles_clean[name + "::__industries__"] = data["industries"]  # type: ignore

    return job_roles_clean, total


def get_job_role_statistics(
    db: Session,
    timeframe: str,
    category: str,
    start_date: Optional[str],
    end_date: Optional[str],
    company: Optional[str],
) -> JobRoleStatisticsData:
    """
    직군별 통계 조회 서비스 (최적화됨)
    - 단일 쿼리로 현재/이전 기간 동시 조회
    - SQL에서 카테고리 필터링 수행
    """
    if category not in {"Tech", "Biz", "BizSupporting"}:
        raise ValueError("category 는 Tech, Biz, BizSupporting 중 하나여야 합니다.")

    periods = _calculate_periods(timeframe, start_date, end_date)

    # 회사 키워드를 패턴 리스트로 변환
    company_patterns = None
    if company:
        company_patterns = get_company_patterns(company)

    # 최적화된 단일 쿼리로 현재/이전 기간 동시 조회
    current_rows, previous_rows = db_competitor_industry_trend.get_job_role_counts_optimized(
        db=db,
        current_start=periods.current_start,
        current_end=periods.current_end,
        previous_start=periods.previous_start,
        previous_end=periods.previous_end,
        category=category,
        company_patterns=company_patterns,
    )

    current_map_raw, current_total = _aggregate_optimized_result(current_rows)
    previous_map_raw, previous_total = _aggregate_optimized_result(previous_rows)

    # job_role 별 union
    job_role_names = set()
    for key in current_map_raw.keys():
        if not key.endswith("::__industries__"):
            job_role_names.add(key)
    for key in previous_map_raw.keys():
        if not key.endswith("::__industries__"):
            job_role_names.add(key)

    statistics: List[JobRoleStatistic] = []

    for job_name in sorted(job_role_names):
        current_total_role = current_map_raw.get(job_name, {}).get("total", 0)
        previous_total_role = previous_map_raw.get(job_name, {}).get("total", 0)

        # 퍼센트 계산
        current_pct = (current_total_role / current_total * 100.0) if current_total > 0 else 0.0
        previous_pct = (previous_total_role / previous_total * 100.0) if previous_total > 0 else 0.0
        change_rate = round(current_pct - previous_pct, 2)

        # 산업(세부 직무)별
        cur_industries_map: Dict[str, int] = current_map_raw.get(
            job_name + "::__industries__", {}
        )  # type: ignore
        prev_industries_map: Dict[str, int] = previous_map_raw.get(
            job_name + "::__industries__", {}
        )  # type: ignore

        all_industries = set(cur_industries_map.keys()) | set(prev_industries_map.keys())
        industries_stats: List[IndustryStatistic] = []
        for ind_name in sorted(all_industries):
            industries_stats.append(
                IndustryStatistic(
                    name=ind_name,
                    current_count=int(cur_industries_map.get(ind_name, 0)),
                    previous_count=int(prev_industries_map.get(ind_name, 0)),
                )
            )

        statistics.append(
            JobRoleStatistic(
                name=job_name,
                current_count=current_total_role,
                current_percentage=round(current_pct, 2),
                previous_count=previous_total_role,
                previous_percentage=round(previous_pct, 2),
                change_rate=change_rate,
                industries=industries_stats,
            )
        )

    return JobRoleStatisticsData(
        timeframe=timeframe,
        category=category,
        current_period=PeriodSummary(
            start_date=periods.current_start.strftime("%Y-%m-%d"),
            end_date=periods.current_end.strftime("%Y-%m-%d"),
            total_count=current_total,
        ),
        previous_period=PeriodSummary(
            start_date=periods.previous_start.strftime("%Y-%m-%d"),
            end_date=periods.previous_end.strftime("%Y-%m-%d"),
            total_count=previous_total,
        ),
        statistics=statistics,
    )


