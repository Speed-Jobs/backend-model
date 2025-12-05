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


# 회사 키 → 표시명 매핑
COMPANY_KEY_TO_NAME = {
    "toss": "토스",
    "kakao": "카카오",
    "hanwha": "한화시스템",
    "hyundai_autoever": "현대오토에버",
    "woowahan": "우아한형제들",
    "lg": "LG_CNS",
    "naver": "네이버",
    "coupang": "쿠팡",
    "line": "LINE"
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
    """회사명을 그룹 키로 매핑"""
    if not company_name:
        return None
    
    name_upper = company_name.upper().strip()
    company_name = company_name.strip()
    
    # 토스 그룹
    if "토스" in company_name or name_upper == "AICC" or "비바리퍼블리카" in company_name:
        return "toss"
    
    # 카카오 그룹
    if "카카오" in company_name:
        return "kakao"
    
    # 한화 그룹 (한화시스템으로 고정, 한화손해보험 제외)
    if "한화시스템" in company_name:
        return "hanwha"
    
    # 현대오토에버 그룹
    if "현대오토에버" in company_name:
        return "hyundai_autoever"
    
    # 우아한 그룹
    if "우아한" in company_name or company_name in ["배달의민족", "배민"]:
        return "woowahan"
    
    # LG 그룹
    if "LG" in name_upper:
        return "lg"
    
    # 네이버 그룹
    if "NAVER" in name_upper or "네이버" in company_name:
        return "naver"
    
    # 쿠팡 그룹
    if "COUPANG" in name_upper or "쿠팡" in company_name:
        return "coupang"
    
    # LINE 그룹
    if "LINE" in name_upper in name_upper:
        return "line"
    
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
    display_name = COMPANY_KEY_TO_NAME.get(company_key, company_key)
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
    all_groups = ["coupang", "hanwha", "hyundai_autoever", "kakao", "lg", "line", "naver", "toss", "woowahan"]
    
    # company_keywords가 있으면 매칭되는 그룹만 선택
    target_groups = set()
    if company_keywords:
        # 키워드로부터 그룹 키 찾기
        for keyword in company_keywords:
            keyword_upper = keyword.upper().strip()
            keyword_stripped = keyword.strip()
            
            # 각 키워드가 어떤 그룹에 매칭되는지 확인
            if "토스" in keyword_stripped or keyword_upper == "AICC" or "비바리퍼블리카" in keyword_stripped:
                target_groups.add("toss")
            elif "카카오" in keyword_stripped:
                target_groups.add("kakao")
            elif "한화시스템" in keyword_stripped or "한화" in keyword_stripped:
                target_groups.add("hanwha")
            elif "현대오토에버" in keyword_stripped:
                target_groups.add("hyundai_autoever")
            elif "우아한" in keyword_stripped or keyword_stripped in ["배달의민족", "배민"]:
                target_groups.add("woowahan")
            elif "LG" in keyword_upper or "lg" in keyword_upper.lower():
                target_groups.add("lg")
            elif "NAVER" in keyword_upper or "네이버" in keyword_stripped:
                target_groups.add("naver")
            elif "COUPANG" in keyword_upper or "쿠팡" in keyword_stripped:
                target_groups.add("coupang")
            elif "LINE" in keyword_upper or "라인" in keyword_stripped or "IPX" in keyword_upper:
                target_groups.add("line")
    else:
        # company_keywords가 없으면 전체 9개 그룹
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
            name=COMPANY_KEY_TO_NAME[key],
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