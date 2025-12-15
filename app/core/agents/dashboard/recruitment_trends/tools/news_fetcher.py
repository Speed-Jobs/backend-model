"""
Dashboard Agent Tools: 네이버 뉴스 검색 도구
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import List, Dict, Optional
import requests
import os
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class NaverNewsAPI:
    """네이버 뉴스 검색 API 클라이언트"""
    
    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.base_url = "https://openapi.naver.com/v1/search/news.json"
    
    def search_news(
        self,
        query: str,
        display: int = 100,
        start: int = 1,
        sort: str = "sim"
    ) -> Optional[Dict]:
        """
        뉴스 검색 API 호출
        
        Args:
            query: 검색어
            display: 한 번에 가져올 결과 수 (최대 100)
            start: 검색 시작 위치 (1~1000)
            sort: 정렬 방식 (sim: 정확도순, date: 날짜순)
        
        Returns:
            API 응답 딕셔너리 또는 None
        """
        if not self.client_id or not self.client_secret:
            return None
        
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        
        params = {
            "query": query,
            "display": min(display, 100),  # 최대 100개
            "start": start,
            "sort": sort
        }
        
        try:
            response = requests.get(self.base_url, headers=headers, params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None


# 전역 인스턴스
_news_api = NaverNewsAPI()


async def _filter_news_by_relevance(
    news_items: List[Dict],
    company_name: str,
    max_results: int = 30
) -> List[Dict]:
    """
    LLM을 사용하여 뉴스 중에서 회사명과 채용 관련성이 높은 뉴스만 필터링
    
    Args:
        news_items: 필터링할 뉴스 목록
        company_name: 회사명
        max_results: 최대 반환 개수
    
    Returns:
        필터링된 뉴스 목록
    """
    if not news_items:
        return []
    
    try:
        logger.info(f"[Naver News Filter] LLM 필터링 시작 - {len(news_items)}개 뉴스, 회사: {company_name}")
        
        # LLM 초기화
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=4096,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 뉴스 데이터를 간단한 형식으로 변환 (LLM에 전달하기 쉽게)
        news_summaries = []
        for idx, item in enumerate(news_items):
            title = item.get("title", "").replace("<b>", "").replace("</b>", "")
            description = item.get("description", "").replace("<b>", "").replace("</b>", "")
            news_summaries.append({
                "index": idx,
                "title": title,
                "description": description[:200]  # 설명은 200자로 제한
            })
        
        # LLM 필터링 프롬프트
        filter_prompt = f"""당신은 뉴스 필터링 전문가입니다. 다음 뉴스 목록에서 다음 조건을 모두 만족하는 뉴스만 선택하세요:

**필터링 조건:**
1. 회사명 "{company_name}"과 직접적으로 관련된 뉴스여야 합니다.
   - 회사명이 명시적으로 언급되거나
   - 회사의 사업, 서비스, 조직 등과 관련된 내용이어야 합니다.
   - 단순히 같은 업계나 경쟁사만 언급된 것은 제외합니다.

2. 채용에 영향을 끼치는 요소와 관련된 뉴스여야 합니다.
   - 직접적 채용 관련: 채용 공고, 채용 확대, 인력 충원, 조직 확장, 신규 사업부 신설, 인재 영입, 채용 전략 등
   - 간접적 채용 영향 요소: 신규 사업 런칭, 사업 확장, 투자 유치, M&A, 신규 서비스 출시, 조직 개편, 경영진 변경 등 (채용 증가로 이어질 수 있는 요소)
   - 단순히 회사 소식만 있고 채용과 무관한 뉴스는 제외합니다.

**뉴스 목록:**
{json.dumps(news_summaries, ensure_ascii=False, indent=2)}

**출력 형식:**
다음 JSON 형식으로 관련성 높은 뉴스의 index만 반환하세요:
{{
  "relevant_indices": [0, 2, 5, ...]
}}

**중요:**
- 조건을 모두 만족하는 뉴스만 선택하세요.
- 관련성이 낮은 뉴스는 제외하세요.
- 최대 {max_results}개까지만 선택하세요.
- JSON만 출력하세요 (설명 없이)."""

        response = await llm.ainvoke(filter_prompt)
        response_text = response.content.strip()
        
        # JSON 추출
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # JSON 파싱
        filter_result = json.loads(response_text)
        relevant_indices = filter_result.get("relevant_indices", [])
        
        # 인덱스로 필터링된 뉴스 추출
        filtered_news = []
        for idx in relevant_indices:
            if 0 <= idx < len(news_items):
                filtered_news.append(news_items[idx])
        
        logger.info(f"[Naver News Filter] LLM 필터링 완료 - {len(filtered_news)}개 뉴스 선택됨 (전체 {len(news_items)}개 중)")
        
        return filtered_news
        
    except Exception as e:
        logger.warning(f"[Naver News Filter] LLM 필터링 중 오류 발생: {str(e)}, 원본 뉴스 반환")
        import traceback
        logger.debug(f"[Naver News Filter] Traceback: {traceback.format_exc()}")
        # 오류 발생 시 원본 뉴스 반환 (필터링 실패해도 서비스는 계속)
        return news_items[:max_results]


async def search_naver_news(
    company_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = 30,
    keywords: Optional[str] = None
) -> str:
    """
    네이버 뉴스에서 특정 회사 관련 뉴스를 검색합니다.

    Args:
        company_name: 회사명 (예: "토스", "네이버")
        start_date: 시작일 (YYYY-MM-DD 형식, None이면 최근 30일)
        end_date: 종료일 (YYYY-MM-DD 형식, None이면 오늘)
        max_results: 최대 검색 결과 수 (기본값: 30, LLM 필터링 없이 그대로 반환)
        keywords: 추가 검색 키워드 (예: "채용", "투자", "사업 확장")

    Returns:
        JSON 형식의 뉴스 검색 결과 (제목, 링크, 발행일, 요약 등)
    """
    if not _news_api.client_id or not _news_api.client_secret:
        import json
        return json.dumps({
            "error": "네이버 뉴스 API 인증 정보가 설정되지 않았습니다. NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET 환경 변수를 설정해주세요."
        }, ensure_ascii=False)

    # 검색어 구성
    search_query = company_name
    if keywords:
        search_query = f"{company_name} {keywords}"

    try:
        logger.info(f"[Naver News] 검색 시작 - 회사: {company_name}, 검색어: {search_query}, max_results: {max_results}")

        # API를 한 번만 호출하여 max_results 개수만 수집 (LLM 필터링 없이 원본 반환)
        result = _news_api.search_news(
            query=search_query,
            display=max_results,
            start=1,
            sort="sim"
        )

        if not result or "items" not in result:
            logger.warning(f"[Naver News] API 호출 실패 또는 결과 없음")
            import json
            return json.dumps({
                "company_name": company_name,
                "query": search_query,
                "news_count": 0,
                "news": []
            }, ensure_ascii=False)

        all_items = result.get("items", [])
        logger.info(f"[Naver News] {len(all_items)}개 뉴스 수집 완료")

        if not all_items:
            logger.warning(f"[Naver News] 수집된 뉴스 없음 - 회사: {company_name}")
            import json
            return json.dumps({
                "company_name": company_name,
                "query": search_query,
                "news_count": 0,
                "news": []
            }, ensure_ascii=False)

        # 날짜 필터링 (필요한 경우)
        filtered_items = []
        if start_date or end_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

            for item in all_items:
                try:
                    # 네이버 뉴스 날짜 형식: "Mon, 01 Jan 2024 00:00:00 +0900"
                    pub_date_str = item.get("pubDate", "")
                    if pub_date_str:
                        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
                        pub_date = datetime.strptime(pub_date_str.split(" +")[0], "%a, %d %b %Y %H:%M:%S")
                        if start_dt and pub_date < start_dt:
                            continue
                        if pub_date > end_dt:
                            continue
                except:
                    pass  # 날짜 파싱 실패 시 포함
                filtered_items.append(item)
        else:
            filtered_items = all_items

        logger.info(f"[Naver News] 날짜 필터링 완료 - {len(filtered_items)}개 뉴스")

        # 결과 포맷팅 (LLM 필터링 없이 그대로 반환)
        news_list = []
        for item in filtered_items[:max_results]:
            news_list.append({
                "title": item.get("title", "").replace("<b>", "").replace("</b>", ""),
                "link": item.get("link", ""),
                "description": item.get("description", "").replace("<b>", "").replace("</b>", ""),
                "pub_date": item.get("pubDate", "")
            })

        logger.info(f"[Naver News] 최종 반환 - {len(news_list)}개 뉴스 (LLM 필터링 없음)")

        import json
        result_json = json.dumps({
            "company_name": company_name,
            "query": search_query,
            "news_count": len(news_list),
            "news": news_list
        }, ensure_ascii=False)

        logger.info(f"[Naver News] 반환 JSON 길이: {len(result_json)}")
        return result_json
    except Exception as e:
        import json
        return json.dumps({
            "error": f"뉴스 검색 중 오류 발생: {str(e)}"
        }, ensure_ascii=False)

