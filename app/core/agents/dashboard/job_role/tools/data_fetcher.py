"""
Job Role Insight Tools - Data Fetcher
기존 get_job_role_statistics 서비스를 사용하여 직군별 통계 데이터를 조회합니다.
"""

from typing import Optional
from sqlalchemy.orm import Session
import json

from app.db.config.base import SessionLocal
from app.services.dashboard.competitor_industry_trend import get_job_role_statistics


def get_job_role_statistics_data(
    timeframe: str,
    category: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    company: Optional[str] = None
) -> str:
    """
    직군별 통계 데이터 조회 (기존 서비스 사용)
    
    Args:
        timeframe: 시간 단위 ("monthly_same_period" 또는 "quarterly_same_period")
        category: 직군 카테고리 ("Tech", "Biz", "BizSupporting")
        start_date: 시작일 (YYYY-MM-DD 형식, 선택사항)
        end_date: 종료일 (YYYY-MM-DD 형식, 선택사항)
        company: 회사명 필터 (부분 일치, 선택사항)
    
    Returns:
        JSON 형식의 직군별 통계 데이터
    """
    db = SessionLocal()
    try:
        db.connection(execution_options={"isolation_level": "AUTOCOMMIT"})
        
        # 기존 서비스 호출
        statistics_data = get_job_role_statistics(
            db=db,
            timeframe=timeframe,
            category=category,
            start_date=start_date,
            end_date=end_date,
            company=company
        )
        
        # Pydantic 모델을 dict로 변환
        result = {
            "timeframe": statistics_data.timeframe,
            "category": statistics_data.category,
            "current_period": {
                "start_date": statistics_data.current_period.start_date,
                "end_date": statistics_data.current_period.end_date,
                "total_count": statistics_data.current_period.total_count
            },
            "previous_period": {
                "start_date": statistics_data.previous_period.start_date,
                "end_date": statistics_data.previous_period.end_date,
                "total_count": statistics_data.previous_period.total_count
            },
            "company_filter": company,
            "statistics": [
                {
                    "name": stat.name,
                    "current_count": stat.current_count,
                    "current_percentage": stat.current_percentage,
                    "previous_count": stat.previous_count,
                    "previous_percentage": stat.previous_percentage,
                    "change_rate": stat.change_rate,
                    "industries": [
                        {
                            "name": ind.name,
                            "current_count": ind.current_count,
                            "previous_count": ind.previous_count
                        }
                        for ind in stat.industries
                    ]
                }
                for stat in statistics_data.statistics
            ]
        }
        
        return json.dumps(result, ensure_ascii=False)
    finally:
        db.close()

