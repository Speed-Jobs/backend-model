"""
주요 회사별 채용 활동 Service
"""
from datetime import date, datetime, timedelta
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from collections import defaultdict

from app.db.crud import db_competitor_recruit_counter
from app.schemas.schemas_competitor_recruit_counter import (
    RecruitmentActivityData,
    CompanyInfo,
    ActivityItem
)
from app.config.company_groups import COMPANY_GROUPS, get_company_patterns

# 표시명 매핑 (로컬 정의)
COMPANY_KEY_TO_DISPLAY_NAME = {
    "toss": "토스",
    "kakao": "카카오",
    "hanwha": "한화시스템",
    "hyundai autoever": "현대오토에버",
    "woowahan": "우아한형제들",
    "coupang": "쿠팡",
    "line": "라인",
    "naver": "네이버",
    "lg cns": "LG CNS",
}


def _calculate_period(timeframe: str) -> Tuple[date, date]:
    """조회 기간 계산"""
    today = date.today()
    
    if timeframe == "daily":
        return today - timedelta(days=29), today
    elif timeframe == "weekly":
        return today - timedelta(weeks=12), today
    elif timeframe == "monthly":
        return today - timedelta(days=330), today
    else:
        return today - timedelta(days=29), today


def _get_company_key(company_name: str) -> Optional[str]:
    """회사명을 그룹 키로 매핑 (COMPANY_GROUPS 패턴 기반)"""
    if not company_name:
        return None
    
    company_name_upper = company_name.upper().strip()
    company_name_stripped = company_name.strip()
    
    # COMPANY_GROUPS의 각 그룹과 패턴을 확인
    for group_key, patterns in COMPANY_GROUPS.items():
        for pattern in patterns:
            # 패턴에서 % 제거
            clean_pattern = pattern.replace("%", "").strip()
            # 대소문자 무시 매칭
            if clean_pattern.upper() in company_name_upper or clean_pattern in company_name_stripped:
                return group_key
    
    return None


def _format_weekly(year: int, week: int) -> str:
    """주간 포맷: 'N월 N주'"""
    try:
        first_day = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
        month = first_day.month
        week_of_month = (first_day.day - 1) // 7 + 1
        return f"{month}월 {week_of_month}주"
    except:
        return f"{year}년 {week}주"


def _get_display_key(company_key: str) -> str:
    """그룹 키를 표시명 기반 키로 변환 (한글은 그대로, 영어는 소문자+언더스코어)"""
    display_name = COMPANY_KEY_TO_DISPLAY_NAME.get(company_key, company_key)
    # 영어는 소문자로 변환하고 공백을 언더스코어로, 한글은 그대로
    if display_name.isascii():
        return display_name.lower().replace(' ', '_')
    else:
        return display_name


def get_companies_recruitment_activity(
    db: Session,
    timeframe: str,
    company_keywords: Optional[List[str]] = None
) -> RecruitmentActivityData:
    """
    주요 회사별 채용 활동 조회
    
    Args:
        db: DB 세션
        timeframe: "daily", "weekly", "monthly"
        company_keywords: 조회할 회사 키워드 리스트 (None이면 전체 9개 회사)
    
    Returns:
        RecruitmentActivityData
    """
    # 1. 기간 계산
    start_date, end_date = _calculate_period(timeframe)
    
    # 2. 조회할 회사 그룹 결정
    all_groups = list(COMPANY_GROUPS.keys())
    
    # company_keywords가 있으면 매칭되는 그룹만 선택
    target_groups = set()
    if company_keywords:
        # 키워드로부터 그룹 키 찾기 (get_company_patterns 사용)
        for keyword in company_keywords:
            keyword_normalized = keyword.lower().strip()
            # COMPANY_GROUPS에서 직접 매칭
            if keyword_normalized in COMPANY_GROUPS:
                target_groups.add(keyword_normalized)
            else:
                # 패턴 기반 매칭 시도
                patterns = get_company_patterns(keyword)
                for pattern in patterns:
                    clean_pattern = pattern.replace("%", "").strip()
                    # 키워드가 패턴에 포함되는지 확인
                    if clean_pattern.lower() in keyword_normalized or keyword_normalized in clean_pattern.lower():
                        # 패턴이 속한 그룹 찾기
                        for group_key, group_patterns in COMPANY_GROUPS.items():
                            if pattern in group_patterns:
                                target_groups.add(group_key)
                                break
    else:
        # company_keywords가 없으면 전체 그룹
        target_groups = set(all_groups)
    
    # 3. DB에서 모든 회사 조회
    from app.models.company import Company
    all_companies = db.query(Company.id, Company.name).all()
    
    # 4. 회사명을 그룹 키로 매핑
    company_id_to_key = {}  # company_id -> key
    key_to_company_id = {}  # key -> 대표 company_id
    
    for company_id, company_name in all_companies:
        if not company_name:
            continue
        
        company_key = _get_company_key(company_name)
        if company_key and company_key in target_groups:  # target_groups에 있는 것만
            # company_id를 그룹 키로 매핑
            company_id_to_key[company_id] = company_key
            # 대표 company_id 설정 (없으면 첫 번째로 발견된 것)
            if company_key not in key_to_company_id:
                key_to_company_id[company_key] = company_id
    
    # 4. companies 리스트 생성 (target_groups에 있는 것만 포함)
    companies_list = []
    for key in sorted(target_groups):  # all_groups 대신 target_groups 사용
        companies_list.append(CompanyInfo(
            id=key_to_company_id.get(key, 0),
            name=COMPANY_KEY_TO_DISPLAY_NAME.get(key, key),
            key=key
        ))
    
    # 5. 모든 회사 ID 수집
    all_company_ids = list(company_id_to_key.keys())
    if not all_company_ids:
        return RecruitmentActivityData(
            timeframe=timeframe,
            companies=companies_list,
            activities=[]
        )
    
    # 6. timeframe별 데이터 조회 및 집계
    activities = []
    
    if timeframe == "daily":
        results = db_competitor_recruit_counter.get_companies_recruitment_daily(
            db, all_company_ids, start_date, end_date
        )
        
        period_counts = defaultdict(lambda: defaultdict(int))
        for result_date, company_id, count in results:
            company_key = company_id_to_key.get(company_id)
            if company_key:
                period = f"{result_date.month}/{result_date.day}"
                display_key = _get_display_key(company_key)
                period_counts[period][display_key] += count
        
        activities = [
            ActivityItem(period=period, counts=dict(counts))
            for period, counts in sorted(period_counts.items())
        ]
    
    elif timeframe == "weekly":
        results = db_competitor_recruit_counter.get_companies_recruitment_weekly(
            db, all_company_ids, start_date, end_date
        )
        
        period_counts = defaultdict(lambda: defaultdict(int))
        for year, week, company_id, count in results:
            company_key = company_id_to_key.get(company_id)
            if company_key:
                period = _format_weekly(year, week)
                display_key = _get_display_key(company_key)
                period_counts[period][display_key] += count
        
        activities = [
            ActivityItem(period=period, counts=dict(counts))
            for period, counts in sorted(period_counts.items())
        ]
    
    elif timeframe == "monthly":
        results = db_competitor_recruit_counter.get_companies_recruitment_monthly(
            db, all_company_ids, start_date, end_date
        )
        
        period_counts = defaultdict(lambda: defaultdict(int))
        for year, month, company_id, count in results:
            company_key = company_id_to_key.get(company_id)
            if company_key:
                period = f"{year}-{month:02d}"
                display_key = _get_display_key(company_key)
                period_counts[period][display_key] += count
        
        activities = [
            ActivityItem(period=period, counts=dict(counts))
            for period, counts in sorted(period_counts.items())
        ]
    
    else:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    
    return RecruitmentActivityData(
        timeframe=timeframe,
        companies=companies_list,
        activities=activities
    )