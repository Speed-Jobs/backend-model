"""
주요 회사별 채용 활동 Service
"""
from datetime import date, datetime, timedelta
from typing import Optional, List, Tuple, Dict
from sqlalchemy.orm import Session
from collections import defaultdict
import re

from app.db.crud import db_competitor_recruit_counter
from app.schemas.schemas_competitor_recruit_counter import (
    RecruitmentActivityData,
    CompanyInfo,
    ActivityItem
)


# 기본 5개 주요 경쟁사 키워드 (그룹 대표)
DEFAULT_COMPANY_KEYWORDS = ["토스", "카카오", "한화시스템", "현대오토에버", "우아한", "LG_CNS", "네이버", "쿠팡", "LINE"]


# 키워드별 정규식 패턴 (계열사 자동 매칭)
COMPANY_KEYWORD_PATTERNS = {
    "toss": r"^토스",           # 토스로 시작 (토스뱅크, 토스증권 등)
    "kakao": r"^카카오",        # 카카오로 시작
    "hanwha": r"^한화시스템",         # 한화로 시작 (한화시스템, 한화손해보험 등)
    "hyundai_autoever": r"^현대오토에버",
    "woowahan": r"^우아한"      # 우아한으로 시작
}

# 예외 케이스 하드코딩 (정규식으로 매칭 안 되는 경우)
SPECIAL_COMPANY_MAPPING = {
    "AICC": "toss",
    "비바리퍼블리카": "toss",
    "배달의민족": "woowahan",
    "배민": "woowahan"
}

# 그룹 키 → 표시명
GROUP_DISPLAY_NAMES = {
    "toss": "토스",
    "kakao": "카카오",
    "hanwha": "한화",
    "hyundai_autoever": "현대오토에버",
    "woowahan": "우아한형제들"
}


def _get_default_period(timeframe: str) -> Tuple[date, date]:
    """자동 조회 기간 계산"""
    today = date.today()
    
    if timeframe == "daily":
        # 최근 30일 (1달)
        start_date = today - timedelta(days=29)
        end_date = today
    elif timeframe == "weekly":
        # 최근 12주
        start_date = today - timedelta(weeks=12)
        end_date = today
    elif timeframe == "monthly":
        # 최근 11개월
        start_date = today - timedelta(days=330)  # 11개월 ≈ 330일
        end_date = today
    else:
        # 기본값: 최근 30일
        start_date = today - timedelta(days=29)
        end_date = today
    
    return start_date, end_date


def _map_company_to_group(company_name: str) -> Optional[str]:
    """
    회사명을 그룹 키로 매핑
    
    Args:
        company_name: 회사명 (예: 토스증권, 한화시스템/ICT)
    
    Returns:
        그룹 키 (예: toss, hanwha) 또는 None (매칭 안 됨)
    """
    # 1. 먼저 예외 케이스 체크
    if company_name in SPECIAL_COMPANY_MAPPING:
        return SPECIAL_COMPANY_MAPPING[company_name]
    
    # 2. 정규식 패턴 매칭
    for group_key, pattern in COMPANY_KEYWORD_PATTERNS.items():
        if re.match(pattern, company_name):
            return group_key
    
    # 3. 매칭 실패
    return None


def _format_period_daily(dt: date) -> str:
    """일간 period 포맷"""
    return f"{dt.month}/{dt.day}"


def _format_period_weekly(year: int, week: int) -> str:
    """주간 period 포맷"""
    first_day = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
    month = first_day.month
    week_of_month = (first_day.day - 1) // 7 + 1
    return f"{month}월 {week_of_month}주"


def _format_period_monthly(year: int, month: int) -> str:
    """월간 period 포맷"""
    return f"{year}-{month:02d}"


def get_companies_recruitment_activity(
    db: Session,
    timeframe: str,
    company_keywords: Optional[List[str]] = None
) -> RecruitmentActivityData:
    """
    주요 회사별 채용 활동 조회
    
    Args:
        db: 데이터베이스 세션
        timeframe: 시간 단위 (daily, weekly, monthly)
        company_keywords: 조회할 회사명 키워드 리스트 (회사명 시작 기준 매칭, 없으면 기본 8개)
    
    Returns:
        RecruitmentActivityData: 회사별 채용 활동 데이터
    
    Note:
        날짜는 자동으로 계산됩니다:
        - daily: 최근 30일 (오늘부터 29일 전까지)
        - weekly: 최근 12주
        - monthly: 최근 11개월
    """
    from sqlalchemy import or_
    
    # 기본값 설정
    if not company_keywords:
        company_keywords = DEFAULT_COMPANY_KEYWORDS
    
    # 기간 자동 계산
    start_date, end_date = _get_default_period(timeframe)
    
    # 회사 정보 조회 (CRUD 호출)
    companies_query = db_competitor_recruit_counter.get_companies_by_keywords(db, company_keywords)
    
    # 그룹별로 회사 매핑
    company_id_to_group = {}  # company_id → group_key
    group_to_company_id = {}  # group_key → 대표 company_id (첫 번째 등장)
    company_ids = []  # 조회된 회사들의 ID 리스트
    
    for row in companies_query:
        company_id, company_name = row[0], row[1]
        group_key = _map_company_to_group(company_name)
        
        if group_key:
            company_id_to_group[company_id] = group_key
            company_ids.append(company_id)
            
            # 그룹의 첫 번째 회사를 대표로 저장
            if group_key not in group_to_company_id:
                group_to_company_id[group_key] = company_id
    
    # companies_info: 5개 그룹 대표만 반환
    companies_info = [
        CompanyInfo(
            id=group_to_company_id.get(group_key, 0),
            name=GROUP_DISPLAY_NAMES[group_key],
            key=group_key
        )
        for group_key in ["toss", "kakao", "hanwha", "hyundai_autoever", "woowahan"]
        if group_key in group_to_company_id  # 실제 데이터가 있는 그룹만
    ]
    
    # 회사가 없으면 빈 결과 반환
    if not company_ids:
        return RecruitmentActivityData(
            timeframe=timeframe,
            companies=companies_info,
            activities=[]
        )
    
    # timeframe별 쿼리
    if timeframe == "daily":
        # 일간 CRUD 호출
        results = db_competitor_recruit_counter.get_companies_recruitment_daily(
            db, company_ids, start_date, end_date
        )
        
        # 데이터 구조화 (그룹별 합산)
        period_counts = defaultdict(lambda: defaultdict(int))
        for row in results:
            period = _format_period_daily(row[0])  # date
            company_id = row[1]
            count = row[2]
            
            # 그룹 키로 변환 후 합산
            group_key = company_id_to_group.get(company_id)
            if group_key:
                period_counts[period][group_key] += count
        
        activities = [
            ActivityItem(period=period, counts=dict(counts))
            for period, counts in sorted(period_counts.items())
        ]
    
    elif timeframe == "weekly":
        # 주간 CRUD 호출
        results = db_competitor_recruit_counter.get_companies_recruitment_weekly(
            db, company_ids, start_date, end_date
        )
        
        # 데이터 구조화 (그룹별 합산)
        period_counts = defaultdict(lambda: defaultdict(int))
        for row in results:
            period = _format_period_weekly(row[0], row[1])  # year, week
            company_id = row[2]
            count = row[3]
            
            # 그룹 키로 변환 후 합산
            group_key = company_id_to_group.get(company_id)
            if group_key:
                period_counts[period][group_key] += count
        
        activities = [
            ActivityItem(period=period, counts=dict(counts))
            for period, counts in sorted(period_counts.items())
        ]
    
    elif timeframe == "monthly":
        # 월간 CRUD 호출
        results = db_competitor_recruit_counter.get_companies_recruitment_monthly(
            db, company_ids, start_date, end_date
        )
        
        # 데이터 구조화 (그룹별 합산)
        period_counts = defaultdict(lambda: defaultdict(int))
        for row in results:
            period = _format_period_monthly(row[0], row[1])  # year, month
            company_id = row[2]
            count = row[3]
            
            # 그룹 키로 변환 후 합산
            group_key = company_id_to_group.get(company_id)
            if group_key:
                period_counts[period][group_key] += count
        
        activities = [
            ActivityItem(period=period, counts=dict(counts))
            for period, counts in sorted(period_counts.items())
        ]
    
    else:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    
    return RecruitmentActivityData(
        timeframe=timeframe,
        companies=companies_info,
        activities=activities
    )

