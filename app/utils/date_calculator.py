"""
날짜 계산 유틸리티 함수
"""
from datetime import date, datetime, timedelta
from typing import Tuple


def calculate_3month_period(end_date_str: str) -> Tuple[date, date]:
    """
    3개월 기간 계산 (과거 3개월)

    Args:
        end_date_str: 종료일 (YYYY-MM-DD)

    Returns:
        (start_date, end_date)
    """
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    start_date = end_date - timedelta(days=90)  # 3개월 전부터
    return start_date, end_date


def calculate_previous_year_period(start_date: date, end_date: date) -> Tuple[date, date]:
    """
    작년 동기 기간 계산 (정확히 1년 전)

    Args:
        start_date: 현재 기간 시작일
        end_date: 현재 기간 종료일

    Returns:
        (previous_start_date, previous_end_date)
    """
    previous_end_date = date(end_date.year - 1, end_date.month, end_date.day)
    previous_start_date = date(start_date.year - 1, start_date.month, start_date.day)
    return previous_start_date, previous_end_date

