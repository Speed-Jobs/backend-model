"""
직군별 인사이트 Agent
직군별 채용 공고 통계를 분석하여 인사이트를 생성합니다.
"""

from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
import os
import json
import logging
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tools
from app.core.agents.dashboard.job_role.tools import (
    get_job_role_statistics_data,
    search_job_role_news,
)

# Schemas
from app.schemas.schemas_job_role_insight import JobRoleInsightData

load_dotenv(override=True)


def _identify_trending_job_roles(statistics_data: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """
    통계 데이터에서 트렌딩 직군 식별
    
    Returns:
        {
            "top": [상위 직군들 (전체)],
            "growing": [성장 중인 직군들 (전체)],
            "declining": [감소 중인 직군들 (전체)]
        }
    """
    stats = statistics_data.get("statistics", [])
    
    # 상위 직군들 (현재 비율 기준, 전체)
    top = sorted(stats, key=lambda x: x.get("current_percentage", 0), reverse=True)
    
    # 성장 중 (변화율 상위, 전체)
    growing = sorted(stats, key=lambda x: x.get("change_rate", 0), reverse=True)
    growing = [g for g in growing if g.get("change_rate", 0) > 0]
    
    # 감소 중 (변화율 하위, 전체)
    declining = sorted(stats, key=lambda x: x.get("change_rate", 0))
    declining = [d for d in declining if d.get("change_rate", 0) < 0]
    
    return {
        "top": top,
        "growing": growing,
        "declining": declining
    }


async def generate_job_role_insight_async(
    timeframe: str,
    category: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    company: Optional[str] = None,
    llm_model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    직군별 인사이트 생성
    
    Args:
        timeframe: 시간 단위 ("monthly_same_period" 또는 "quarterly_same_period")
        category: 직군 카테고리 ("Tech", "Biz", "BizSupporting")
        start_date: 시작일 (YYYY-MM-DD 형식, 선택사항)
        end_date: 종료일 (YYYY-MM-DD 형식, 선택사항)
        company: 회사명 필터 (부분 일치, 선택사항)
        llm_model: 사용할 LLM 모델
    
    Returns:
        Dict: 인사이트 생성 결과
    """
    try:
        logger.info(f"인사이트 생성 시작 - 카테고리: {category}, 기간: {timeframe}")
        
        # 1. 데이터 수집
        logger.info(f"[1/2] 직군별 통계 데이터 조회 중...")
        statistics_data_str = get_job_role_statistics_data(
            timeframe=timeframe,
            category=category,
            start_date=start_date,
            end_date=end_date,
            company=company
        )
        statistics_data = json.loads(statistics_data_str)
        logger.info(f"✓ 통계 데이터 조회 완료 - {len(statistics_data.get('statistics', []))}개 직군")
        
        # 트렌딩 직군 식별
        trending = _identify_trending_job_roles(statistics_data)
        top_job_roles = trending["top"]
        growing_job_roles = trending["growing"]
        
        # 2. 뉴스 검색 (상위 직군 및 성장 중인 직군에 대해)
        logger.info(f"[2/2] 직군 관련 뉴스 검색 중...")
        news_data_map = {}
        
        # 상위 3개와 성장 중인 상위 3개 직군에 대해 뉴스 검색
        job_roles_to_search = (top_job_roles[:3] + growing_job_roles[:3])
        unique_job_roles = {jr["name"]: jr for jr in job_roles_to_search}.values()
        
        for job_role in unique_job_roles:
            job_role_name = job_role["name"]
            logger.info(f"  - {job_role_name} 관련 뉴스 검색 중...")
            news_data_str = await search_job_role_news(
                job_role_name=job_role_name,
                start_date=start_date or statistics_data["current_period"]["start_date"],
                end_date=end_date or statistics_data["current_period"]["end_date"],
                keywords="채용 인재",
                max_results=20
            )
            news_data = json.loads(news_data_str)
            news_data_map[job_role_name] = news_data
            logger.info(f"  ✓ {job_role_name} 뉴스 검색 완료 - {news_data.get('news_count', 0)}개")
        
        logger.info(f"✓ 뉴스 검색 완료")
        
        # 3. LLM으로 분석
        logger.info("LLM 분석 시작...")
        llm = ChatOpenAI(
            model=llm_model,
            temperature=0.3,
            max_tokens=16384,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # structured_output을 사용하여 스키마에 맞는 출력 보장
        structured_llm = llm.with_structured_output(JobRoleInsightData)
        
        # 분석 프롬프트
        analysis_prompt = f"""당신은 채용 시장 분석 전문가입니다. 다음 직군별 통계 데이터를 분석하여 인사이트를 제공하세요.

## 데이터

### 직군별 통계 데이터
{json.dumps(statistics_data, ensure_ascii=False, indent=2)}

### 뉴스 데이터
{json.dumps(news_data_map, ensure_ascii=False, indent=2)}

## 필수 필드 (서버에서 자동 설정됨)
- timeframe, category, current_period, previous_period, company_filter는 서버에서 프로그램적으로 설정되므로,
  LLM이 이 필드들을 생성할 필요가 없습니다. null로 설정하거나 무시하세요.

## KEY_FINDINGS 생성 규칙

1. **상위 직군 분석**: "{{직군명}}이 전체의 {{비율}}%를 차지하며 가장 많은 공고를 보유하고 있습니다."
   - evidence: type="data", source="직군별 통계 데이터", value="{{비율}}%"

2. **변화율 분석**: "{{직군명}}은 직전 대비 {{변화율}}%p {{증가/감소}}되었습니다."
   - evidence: type="data", source="직군별 통계 데이터", value="{{변화율}}%p"

3. **뉴스 기반 원인**: "이는, [뉴스에서 확인된 구체적 사건/발표/트렌드]와 관련될 가능성이 있습니다."
   - **중요**: 뉴스를 언급한 경우 반드시 해당 뉴스를 evidence에 포함해야 합니다!
   - evidence: type="news", source=뉴스제목, link=뉴스URL, date=뉴스발행일
   - 뉴스가 없으면: "구체적인 외부 원인은 확인되지 않았습니다." (evidence에 뉴스 포함하지 않음)

**CRITICAL**: key_findings에서 뉴스를 언급했다면, key_findings_evidence의 해당 finding에 반드시 뉴스 evidence를 포함하세요!

## JOB_ROLE_INSIGHTS 생성 규칙

주요 직군들에 대해 인사이트 생성 (상위 직군, 성장 중인 직군, 또는 중요한 변화가 있는 직군):
- job_role_name: 직군명
- insight: "{{직군명}}이 전체의 {{비율}}%를 차지하며..." 형식
- change_description: "{{직군명}}은 직전 대비 {{변화율}}%p {{증가/감소}}되었습니다." (변화가 있는 경우만)
- external_factors: 뉴스 기반 원인 설명 (뉴스가 있는 경우)
- evidence: 해당 직군의 데이터 및 뉴스 근거

## SUMMARY 생성 규칙

전체 카테고리({category})의 종합적인 요약:
- 주요 트렌드 (어떤 직군이 성장/감소하는지)
- 시장 변화 (전체적인 변화율)
- 주요 발견 사항 요약

**중요**: summary에서 뉴스를 언급한 경우, summary_evidence에 반드시 해당 뉴스를 포함하세요!
- summary_evidence는 summary에서 언급된 뉴스나 데이터에 대한 evidence 배열입니다
- 뉴스를 언급했다면: type="news", source=뉴스제목, link=뉴스URL, date=뉴스발행일
- 데이터를 언급했다면: type="data", source="직군별 통계 데이터", value="해당값"

## EVIDENCE 필수 규칙

1. **뉴스를 언급했다면 반드시 evidence에 포함**: summary나 key_findings에서 뉴스를 언급했는데 evidence에 뉴스가 없으면 안 됩니다!
2. **뉴스 evidence 형식**: 
   - type: "news"
   - source: 뉴스 제목 (news_data의 title)
   - link: 뉴스 URL (news_data의 link)
   - date: 뉴스 발행일 (news_data의 pub_date)
   - data_description: null
   - value: null
3. **데이터 evidence 형식**:
   - type: "data"
   - source: "직군별 통계 데이터"
   - link: null
   - date: null
   - data_description: 데이터 설명 (선택사항)
   - value: 데이터 값 (예: "70.99%", "37.65%p")

모든 숫자는 실제 데이터 기반으로 계산하고, 직군명은 statistics 데이터의 name을 정확히 사용하세요."""

        insight_data = await structured_llm.ainvoke(analysis_prompt)
        logger.info("LLM 분석 완료")
        
        # 프로그램적으로 필수 필드 설정 (LLM이 생성하지 않았으므로 여기서 설정)
        from app.schemas.schemas_competitor_industry_trend import PeriodSummary
        
        insight_data.timeframe = timeframe
        insight_data.category = category
        insight_data.current_period = PeriodSummary(
            start_date=statistics_data["current_period"]["start_date"],
            end_date=statistics_data["current_period"]["end_date"],
            total_count=statistics_data["current_period"]["total_count"]
        )
        insight_data.previous_period = PeriodSummary(
            start_date=statistics_data["previous_period"]["start_date"],
            end_date=statistics_data["previous_period"]["end_date"],
            total_count=statistics_data["previous_period"]["total_count"]
        )
        insight_data.company_filter = company
        
        return {
            "status": "success",
            "data": insight_data,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"인사이트 생성 실패: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "data": None,
            "error": str(e)
        }

