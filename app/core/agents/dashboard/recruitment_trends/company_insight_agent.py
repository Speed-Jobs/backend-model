"""
회사 채용 인사이트 Agent - 최종 버전 (weekly, monthly만)
"""

from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
import os
import json
import logging
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tools
from app.core.agents.dashboard.recruitment_trends.tools import (
    get_company_recruitment_data,
    get_competitors_recruitment_data,
    get_total_recruitment_data,
    search_naver_news,
)

# Schemas
from app.schemas.schemas_company_insight import CompanyInsightData

load_dotenv(override=True)


async def generate_company_insight_async(
    company_keyword: str,
    timeframe: str = "weekly",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    llm_model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Agent 없이 직접 툴을 호출하여 인사이트 생성
    
    Args:
        company_keyword: 회사명 키워드
        timeframe: 시간 단위 ("weekly", "monthly")
        start_date: 시작일 (YYYY-MM-DD 형식)
        end_date: 종료일 (YYYY-MM-DD 형식)
        llm_model: 사용할 LLM 모델
    
    Returns:
        Dict: 인사이트 생성 결과
    """
    try:
        logger.info(f"인사이트 생성 시작 - 회사: {company_keyword}, 기간: {timeframe}")
        
        # 1. 데이터 수집 - 직접 툴 호출
        logger.info(f"[1/4] {company_keyword} 채용 데이터 조회 중...")
        company_data_str = get_company_recruitment_data(
            company_keyword=company_keyword,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        company_data = json.loads(company_data_str)
        logger.info(f"✓ {company_keyword} 데이터 조회 완료")
        
        logger.info(f"[2/4] 경쟁사 데이터 조회 중...")
        competitors_data_str = get_competitors_recruitment_data(
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        competitors_data = json.loads(competitors_data_str)
        logger.info(f"✓ 경쟁사 데이터 조회 완료 - {len(competitors_data.get('competitors', []))}개 회사")
        
        logger.info(f"[3/4] 전체 시장 데이터 조회 중...")
        total_data_str = get_total_recruitment_data(
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        total_data = json.loads(total_data_str)
        logger.info(f"✓ 전체 시장 데이터 조회 완료")
        
        logger.info(f"[4/4] {company_keyword} 관련 뉴스 검색 중...")
        news_data_str = await search_naver_news(
            company_name=company_keyword,
            start_date=start_date,
            end_date=end_date,
            keywords="채용",
            max_results=30
        )
        news_data = json.loads(news_data_str)
        logger.info(f"✓ 뉴스 검색 완료 - {news_data.get('news_count', 0)}개")

        # 뉴스 데이터만 간소화 (상위 10개, 제목과 링크만)
        news_for_llm = {
            "company_name": news_data.get("company_name"),
            "news_count": news_data.get("news_count", 0),
            "news": [
                {
                    "title": n.get("title"),
                    "link": n.get("link")
                }
                for n in news_data.get("news", [])[:10]
            ]
        }

        # 2. LLM으로 분석 (with_structured_output 사용)
        logger.info("LLM 분석 시작...")
        llm = ChatOpenAI(
            model=llm_model,
            temperature=0.3,
            max_tokens=8192,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # structured_output을 사용하여 스키마에 맞는 출력 보장
        structured_llm = llm.with_structured_output(CompanyInsightData)

        # 분석 프롬프트
        analysis_prompt = f"""당신은 채용 시장 분석 전문가입니다. 다음 데이터를 분석하여 인사이트를 제공하세요.

## 데이터

### {company_keyword} 채용 데이터
{json.dumps(company_data, ensure_ascii=False, indent=2)}

### 경쟁사 데이터
{json.dumps(competitors_data, ensure_ascii=False, indent=2)}

### 전체 시장 데이터
{json.dumps(total_data, ensure_ascii=False, indent=2)}

### 뉴스 데이터 (상위 10개, 제목과 링크만)
{json.dumps(news_for_llm, ensure_ascii=False, indent=2)}

## 필수 필드
- company_name: "{company_data.get('company_name')}"  ← 반드시 이 값 사용
- company_id: {company_data.get('company_id', 0)}
- timeframe: "{timeframe}"
- total_postings: {company_data.get('total_count', 0)}

## CRITICAL: company_name 사용 규칙
- company_name은 위 "필수 필드"에 명시된 값을 **정확히** 사용하세요
- key_findings, summary 등 모든 곳에서 동일한 company_name을 사용하세요
- 절대 검색 키워드("{company_keyword}")를 사용하지 마세요

## KEY_FINDINGS 생성 규칙 (반드시 이 순서로)

1. **경쟁사 비교**: "{company_data.get('company_name')}는 경쟁사 중 X위로 N건의 채용공고를 게시했으며, 이는 1위 회사명(M건) 대비 Y% 수준입니다."
   - evidence: type="data", source="경쟁사 비교 데이터"

2. **뉴스 기반 원인**: "이는 [뉴스에서 확인된 구체적 사건/발표]와 관련이 있습니다."
   - evidence: type="news", source=뉴스제목, link=뉴스URL
   - 뉴스가 없으면: "구체적인 외부 원인은 확인되지 않았습니다."

3. **트렌드 변화** (선택): "기간 동안 N% 증가/감소 추세를 보였습니다."
   - evidence: type="data", source="트렌드 분석 데이터"

## KEY_FINDINGS_EVIDENCE 규칙
- key_findings와 동일한 개수의 배열 (각 finding마다 evidence 배열)
- data type: date=null, value=null (서버에서 자동 채움)
- news type: link 필수, data_description=null

## 뉴스 사용 규칙
- news_evidence: 최대 5개, 관련성 높은 순
- 뉴스 없으면: news_evidence=[], confidence="low"
- 뉴스 있으면: confidence="high" 또는 "medium"

## EVIDENCE 필수 필드
- type: "news" 또는 "data"
- source: 뉴스 제목 또는 데이터 소스
- link: 뉴스면 URL, 데이터면 null
- data_description: 데이터면 설명, 뉴스면 null

모든 숫자는 실제 데이터 기반으로 계산하고, 회사명은 company_data의 company_name을 정확히 사용하세요."""

        insight_data = await structured_llm.ainvoke(analysis_prompt)
        logger.info("LLM 분석 완료")
        
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