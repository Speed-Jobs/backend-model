"""
회사 채용 인사이트 Service - 최종 버전 (weekly, monthly만)
"""

from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import date, datetime, timedelta
import logging

from app.core.agents.dashboard.company_insight_agent import generate_company_insight_async
from app.schemas.schemas_company_insight import CompanyInsightData
from app.schemas.schemas_recruit_counter import PeriodInfo
from app.db.crud import db_recruit_counter

logger = logging.getLogger(__name__)


def _get_default_period(timeframe: str) -> tuple[date, date]:
    """조회 기간 계산"""
    today = date.today()
    
    if timeframe == "weekly":
        return today - timedelta(weeks=12), today
    elif timeframe == "monthly":
        return today - timedelta(days=330), today
    else:
        return today - timedelta(weeks=12), today


async def get_company_insight(
    db: Session,
    company_keyword: str,
    timeframe: str = "weekly",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> CompanyInsightData:
    """
    회사 채용 인사이트 조회
    
    Args:
        db: 데이터베이스 세션
        company_keyword: 회사명 키워드
        timeframe: 시간 단위 (weekly, monthly)
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
    
    Returns:
        CompanyInsightData: 회사 인사이트
    """
    logger.info(f"인사이트 조회 시작 - 회사: {company_keyword}, 기간: {timeframe}")
    
    # 기간 계산
    if start_date and end_date:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        start, end = _get_default_period(timeframe)
    
    # 회사 정보 조회 (키워드를 포함하는 모든 회사 합산)
    companies = db_recruit_counter.get_companies_by_keyword(db, company_keyword, start, end)
    
    if not companies:
        raise ValueError(f"회사를 찾을 수 없습니다: {company_keyword}")
    
    # 합산
    total_count = sum(count for _, _, count in companies)
    company_id, company_name, _ = companies[0]
    
    logger.info(f"회사 정보: {company_name} (ID: {company_id}), 총 {total_count}건")
    
    # Agent로 인사이트 생성
    result = await generate_company_insight_async(
        company_keyword=company_keyword,
        timeframe=timeframe,
        start_date=start_date or start.isoformat(),
        end_date=end_date or end.isoformat(),
        llm_model="gpt-4o-mini"
    )
    
    if result["status"] == "error":
        raise ValueError(f"인사이트 생성 실패: {result['error']}")
    
    insight_data = result["data"]
    
    # dict로 변환
    if isinstance(insight_data, CompanyInsightData):
        insight_dict = insight_data.model_dump()
    else:
        insight_dict = insight_data
    
    # 기본 정보 보완
    insight_dict["company_name"] = company_name
    insight_dict["company_id"] = company_id
    insight_dict["total_postings"] = total_count
    insight_dict["timeframe"] = timeframe
    insight_dict["period"] = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat()
    }
    
    days = (end - start).days + 1
    if days > 0:
        insight_dict["average_daily_postings"] = round(total_count / days, 2)
    
    # 경쟁사 비교에서 market_share 계산
    if "competitor_comparison" in insight_dict:
        competitors = insight_dict["competitor_comparison"]
        competitor_total = sum(c.get("total_count", 0) for c in competitors)
        
        if competitor_total > 0:
            for comp in competitors:
                comp["market_share"] = round((comp.get("total_count", 0) / competitor_total) * 100, 2)
        
        # 순위 할당
        competitors.sort(key=lambda x: x.get("total_count", 0), reverse=True)
        for rank, comp in enumerate(competitors, 1):
            comp["rank"] = rank
        
        # market_rank 찾기
        for comp in competitors:
            if comp.get("company_name") == company_name:
                insight_dict["market_rank"] = comp.get("rank", 1)
                break
    
    # key_findings_evidence의 data type evidence 보완
    if "key_findings_evidence" in insight_dict:
        for finding_evidences in insight_dict["key_findings_evidence"]:
            for evidence in finding_evidences:
                if evidence.get("type") == "data":
                    # 기본값 설정
                    if not evidence.get("date"):
                        evidence["date"] = f"{start.isoformat()} ~ {end.isoformat()}"
                    if not evidence.get("value"):
                        evidence["value"] = str(total_count)
    
    logger.info(f"인사이트 생성 완료 - {company_name}")
    
    return CompanyInsightData(**insight_dict)