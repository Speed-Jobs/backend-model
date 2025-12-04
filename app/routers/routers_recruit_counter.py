"""
채용 공고 수 추이 API Router
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import date, timedelta
from typing import Optional

from app.db.config.base import get_db
from app.db.crud import db_competitor_recruit_counter, db_recruit_counter
from app.services.dashboard.company_insight import get_company_insight
from app.schemas.schemas_recruit_counter import DashBoardResponse

COMPANY_KEYWORD_TO_GROUP = {
    "toss": "토스",
    "kakao": "카카오",
    "hanwha": "한화시스템",
    "hyundai_autoever": "현대오토에버",
    "woowahan": "우아한형제들",
    "coupang": "쿠팡",
    "line": "라인",
    "naver": "NAVER",
    "lg_cns": "LG CNS",  # lg → lg_cns로 변경
}

def resolve_company_keyword(keyword: str) -> Optional[str]:
    keyword_lower = keyword.lower().strip()
    # lg_cns 또는 lg cns 모두 지원
    if keyword_lower == "lg cns":
        keyword_lower = "lg_cns"
    if keyword_lower in COMPANY_KEYWORD_TO_GROUP:
        return COMPANY_KEYWORD_TO_GROUP[keyword_lower]
    return keyword


router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["Dashboard"]
)


def _get_default_period(timeframe: str) -> tuple[date, date]:
    """조회 기간 계산"""
    today = date.today()
    
    if timeframe == "weekly":
        return today - timedelta(weeks=12), today
    elif timeframe == "monthly":
        return today - timedelta(days=330), today
    else:
        return today - timedelta(weeks=12), today


def _format_period_weekly(year: int, week: int) -> str:
    from datetime import datetime
    first_day = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
    month = first_day.month
    week_of_month = (first_day.day - 1) // 7 + 1
    return f"{month}월 {week_of_month}주"


def _format_period_monthly(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


@router.get(
    "/job-postings-trend",
    response_model=DashBoardResponse,
    summary="채용 공고 추이 조회",
    description="전체 또는 특정 회사의 채용 공고 추이를 조회합니다."
)
async def get_job_postings_trend(
    timeframe: str = Query(..., enum=["weekly", "monthly"], description="시간 단위"),
    company_keyword: Optional[str] = Query(
        None, 
        description="""회사명 키워드 (영어 소문자만 사용 가능, 없으면 전체 조회)
        
        지원 키워드:
        - toss: 토스 계열사
        - kakao: 카카오 계열사
        - hanwha: 한화시스템 계열사
        - hyundai_autoever: 현대오토에버
        - woowahan: 우아한형제들 계열사
        - coupang: 쿠팡 계열사
        - line: 라인 계열사
        - naver: 네이버 계열사
        - lg_cns: LG CNS 계열사
        
        예시: 'toss', 'kakao', 'naver', 'lg_cns'""",
        example="toss"
    ),
    include_insight: bool = Query(False, description="인사이트 포함 여부 (특정 회사만)"),
    db: Session = Depends(get_db)
):
    """
    채용 공고 추이 조회
    
    - timeframe: weekly (12주), monthly (11개월)
    - company_keyword: None이면 전체, 있으면 특정 회사
    - include_insight: True면 인사이트 포함 (특정 회사만)
    """
    try:
        start, end = _get_default_period(timeframe)
        
        # 1. 전체 추이 데이터
        period_data = []
        total_count = 0
        
        if timeframe == "weekly":
            results = db_recruit_counter.get_job_postings_weekly(db, start, end)
            period_data = [{"period": _format_period_weekly(row[0], row[1]), "count": row[2]} for row in results]
            total_count = sum(row[2] for row in results)
        elif timeframe == "monthly":
            results = db_recruit_counter.get_job_postings_monthly(db, start, end)
            period_data = [{"period": _format_period_monthly(row[0], row[1]), "count": row[2]} for row in results]
            total_count = sum(row[2] for row in results)
        
        response_data = {
            "timeframe": timeframe,
            "period": {"start_date": start.isoformat(), "end_date": end.isoformat()},
            "trends": period_data,
            "total_count": total_count
        }
        
        # 2. 특정 회사 데이터
        if company_keyword:
            resolved_keyword = resolve_company_keyword(company_keyword)
            companies = db_recruit_counter.get_companies_by_keyword(db, resolved_keyword, start, end)
            
            if not companies:
                raise HTTPException(status_code=404, detail=f"회사를 찾을 수 없습니다: {company_keyword}")
            
            # 합산
            company_total = sum(count for _, _, count in companies)
            # 첫 번째 회사 정보는 참고용으로만 사용
            company_id, company_name, _ = companies[0] if companies else (None, None, 0)
            company_ids = [cid for cid, _, _ in companies]
            
            # 기간별 데이터
            company_trends = []
            if timeframe == "weekly":
                results = db_competitor_recruit_counter.get_companies_recruitment_weekly(db, company_ids, start, end)
                # period별로 합산
                from collections import defaultdict
                period_counts = defaultdict(int)
                for row in results:
                    period = _format_period_weekly(row[0], row[1])
                    period_counts[period] += row[3]
                company_trends = [{"period": period, "count": count} for period, count in sorted(period_counts.items())]
            elif timeframe == "monthly":
                results = db_competitor_recruit_counter.get_companies_recruitment_monthly(db, company_ids, start, end)
                # period별로 합산
                from collections import defaultdict
                period_counts = defaultdict(int)
                for row in results:
                    period = _format_period_monthly(row[0], row[1])
                    period_counts[period] += row[3]
                company_trends = [{"period": period, "count": count} for period, count in sorted(period_counts.items())]
            
            response_data["selected_company"] = {
                "company_id": company_id,
                "company_name": company_name,
                "total_count": company_total,
                "trends": company_trends
            }
            
            # 3. 인사이트 추가
            if include_insight:
                try:
                    insight = await get_company_insight(db, company_keyword, timeframe)
                    response_data["insight"] = insight.model_dump()
                except Exception as e:
                    response_data["insight_error"] = f"인사이트 생성 실패: {str(e)}"
        
        # 응답 메시지
        timeframe_kr = {"weekly": "주간", "monthly": "월간"}
        if company_keyword:
            if include_insight:
                message = f"{timeframe_kr[timeframe]} 채용 공고 추이 조회 성공 (회사: {company_keyword}, 인사이트 포함)"
            else:
                message = f"{timeframe_kr[timeframe]} 채용 공고 추이 조회 성공 (회사: {company_keyword})"
        else:
            message = f"{timeframe_kr[timeframe]} 채용 공고 추이 조회 성공 (전체)"
        
        return DashBoardResponse(
            status=200,
            code="SUCCESS",
            message=message,
            data=response_data
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")