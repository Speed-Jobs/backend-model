"""
Dashboard Tools - 최종 버전 (weekly, monthly만)
"""

from typing import Dict, Any, List, Optional
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session
import json

from app.db.crud import db_recruit_counter, db_competitor_recruit_counter
from app.db.config.base import SessionLocal
from app.config.company_groups import COMPANY_GROUPS


def _get_default_period(timeframe: str) -> tuple[date, date]:
    """자동 조회 기간 계산"""
    today = date.today()
    
    if timeframe == "weekly":
        return today - timedelta(weeks=12), today
    elif timeframe == "monthly":
        return today - timedelta(days=330), today
    else:
        return today - timedelta(weeks=12), today


def _format_period_weekly(year: int, week: int) -> str:
    first_day = datetime.strptime(f'{year}-W{week:02d}-1', '%Y-W%W-%w')
    month = first_day.month
    week_of_month = (first_day.day - 1) // 7 + 1
    return f"{month}월 {week_of_month}주"


def _format_period_monthly(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def get_company_recruitment_data(
    company_keyword: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """키워드를 포함하는 모든 회사의 채용 데이터 조회 (합산)"""
    db = SessionLocal()
    try:
        db.connection(execution_options={"isolation_level": "AUTOCOMMIT"})
        
        # 기간 계산
        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            start, end = _get_default_period(timeframe)
        
        # 키워드를 포함하는 모든 회사 조회
        companies = db_recruit_counter.get_companies_by_keyword(db, company_keyword, start, end)
        
        if not companies:
            return json.dumps({"error": f"회사를 찾을 수 없습니다: {company_keyword}"}, ensure_ascii=False)
        
        # 합산
        total_count = sum(count for _, _, count in companies)
        representative_id, representative_name, _ = companies[0]
        company_ids = [cid for cid, _, _ in companies]
        
        # 기간별 데이터 조회
        period_data = []
        if timeframe == "weekly":
            results = db_competitor_recruit_counter.get_companies_recruitment_weekly(db, company_ids, start, end)
            period_data = [{"period": _format_period_weekly(row[0], row[1]), "count": row[3]} for row in results]
        elif timeframe == "monthly":
            results = db_competitor_recruit_counter.get_companies_recruitment_monthly(db, company_ids, start, end)
            period_data = [{"period": _format_period_monthly(row[0], row[1]), "count": row[3]} for row in results]
        
        return json.dumps({
            "company_id": representative_id,
            "company_name": representative_name,
            "total_count": total_count,
            "timeframe": timeframe,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "period_data": period_data,
            "included_companies": [
                {"company_id": cid, "company_name": cname, "count": count}
                for cid, cname, count in companies
            ]
        }, ensure_ascii=False)
    finally:
        db.close()


def get_competitors_recruitment_data(
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """경쟁사 채용 데이터 조회 (고정 리스트)"""
    db = SessionLocal()
    try:
        db.connection(execution_options={"isolation_level": "AUTOCOMMIT"})
        
        # 기간 계산
        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            start, end = _get_default_period(timeframe)
        
        # 각 경쟁사별로 조회 (COMPANY_GROUPS의 모든 키 사용)
        competitors = []
        for keyword in COMPANY_GROUPS.keys():
            companies = db_recruit_counter.get_companies_by_keyword(db, keyword, start, end)
            
            if companies:
                total_count = sum(count for _, _, count in companies)
                representative_id, representative_name, _ = companies[0]
                
                competitors.append({
                    "company_id": representative_id,
                    "company_name": representative_name,
                    "total_count": total_count
                })
        
        # 순위 매기기
        competitors.sort(key=lambda x: x["total_count"], reverse=True)
        for rank, comp in enumerate(competitors, 1):
            comp["rank"] = rank
        
        return json.dumps({
            "timeframe": timeframe,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "competitors": competitors
        }, ensure_ascii=False)
    finally:
        db.close()


def get_total_recruitment_data(
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> str:
    """전체 채용 데이터 조회"""
    db = SessionLocal()
    try:
        db.connection(execution_options={"isolation_level": "AUTOCOMMIT"})
        
        # 기간 계산
        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        else:
            start, end = _get_default_period(timeframe)
        
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
        
        return json.dumps({
            "timeframe": timeframe,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "total_count": total_count,
            "period_data": period_data
        }, ensure_ascii=False)
    finally:
        db.close()