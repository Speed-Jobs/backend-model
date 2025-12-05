"""
Job Role Insight Tools: 네이버 뉴스 검색 도구 (직군 관련)
"""

from langchain_openai import ChatOpenAI
from typing import List, Dict, Optional
import requests
import os
import json
import logging
from datetime import datetime
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
    job_role_name: str,
    max_results: int = 30
) -> List[Dict]:
    """
    LLM을 사용하여 뉴스 중에서 직군과 채용 관련성이 높은 뉴스만 필터링
    
    Args:
        news_items: 필터링할 뉴스 목록
        job_role_name: 직군 이름 (예: "AI", "Software Development")
        max_results: 최대 반환 개수
    
    Returns:
        필터링된 뉴스 목록
    """
    if not news_items:
        return []
    
    try:
        logger.info(f"[Naver News Filter] LLM 필터링 시작 - {len(news_items)}개 뉴스, 직군: {job_role_name}")
        
        # LLM 초기화
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=4096,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 뉴스 데이터를 간단한 형식으로 변환
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
1. 직군 "{job_role_name}"과 직접적으로 관련된 뉴스여야 합니다.
   - 직군명이 명시적으로 언급되거나
   - 해당 직군의 기술, 트렌드, 시장 동향 등과 관련된 내용이어야 합니다.

2. 채용에 영향을 끼치는 요소와 관련된 뉴스여야 합니다.
   - 직접적 채용 관련: 해당 직군 채용 확대, 인력 수요 증가, 기술 인재 부족, 신규 직군 신설 등
   - 간접적 채용 영향 요소: 기술 발전, 산업 성장, 신규 서비스 출시, 투자 유치, 정부 정책 등 (채용 증가로 이어질 수 있는 요소)
   - 단순히 기술 소식만 있고 채용과 무관한 뉴스는 제외합니다.

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


async def search_job_role_news(
    job_role_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = 30,
    keywords: Optional[str] = None
) -> str:
    """
    네이버 뉴스에서 특정 직군 관련 뉴스를 검색합니다.
    
    Args:
        job_role_name: 직군 이름 (예: "AI", "Software Development", "Data Science")
        start_date: 시작일 (YYYY-MM-DD 형식, None이면 최근 30일)
        end_date: 종료일 (YYYY-MM-DD 형식, None이면 오늘)
        max_results: 최대 검색 결과 수 (기본값: 30, 최종 반환 개수)
        keywords: 추가 검색 키워드 (예: "채용", "인재", "시장")
    
    Returns:
        JSON 형식의 뉴스 검색 결과
    """
    if not _news_api.client_id or not _news_api.client_secret:
        return json.dumps({
            "error": "네이버 뉴스 API 인증 정보가 설정되지 않았습니다. NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET 환경 변수를 설정해주세요."
        }, ensure_ascii=False)
    
    # 검색어 구성
    search_query = job_role_name
    if keywords:
        search_query = f"{job_role_name} {keywords}"
    
    try:
        logger.info(f"[Naver News] 검색 시작 - 직군: {job_role_name}, 검색어: {search_query}, max_results: {max_results}")
        
        # API를 100개씩 5번 호출하여 총 500개 수집
        display_per_call = 100
        num_calls = 5
        all_items = []
        
        # 여러 번 API 호출하여 뉴스 수집
        for call_num in range(num_calls):
            start_pos = call_num * display_per_call + 1
            
            logger.info(f"[Naver News] API 호출 {call_num + 1}/{num_calls} - start: {start_pos}")
            
            result = _news_api.search_news(
                query=search_query,
                display=display_per_call,
                start=start_pos,
                sort="sim"
            )
            
            if not result or "items" not in result:
                logger.warning(f"[Naver News] API 호출 {call_num + 1} 실패 또는 결과 없음")
                break
            
            items = result.get("items", [])
            if not items:
                logger.info(f"[Naver News] API 호출 {call_num + 1} - 결과 없음, 중단")
                break
            
            logger.info(f"[Naver News] API 호출 {call_num + 1} 성공 - {len(items)}개 뉴스 수집")
            all_items.extend(items)
            
            # API 호출 제한 고려
            import time
            if call_num < num_calls - 1:
                time.sleep(0.1)
        
        logger.info(f"[Naver News] 총 {len(all_items)}개 뉴스 수집 완료")
        
        if not all_items:
            logger.warning(f"[Naver News] 수집된 뉴스 없음 - 직군: {job_role_name}")
            return json.dumps({
                "job_role_name": job_role_name,
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
                    pub_date_str = item.get("pubDate", "")
                    if pub_date_str:
                        pub_date = datetime.strptime(pub_date_str.split(" +")[0], "%a, %d %b %Y %H:%M:%S")
                        if start_dt and pub_date < start_dt:
                            continue
                        if pub_date > end_dt:
                            continue
                except:
                    pass
                filtered_items.append(item)
        else:
            filtered_items = all_items
        
        logger.info(f"[Naver News] 날짜 필터링 완료 - {len(filtered_items)}개 뉴스")
        
        # LLM을 사용한 직군 및 채용 관련성 필터링
        if filtered_items and job_role_name:
            logger.info(f"[Naver News] LLM 필터링 시작 - 직군: {job_role_name}, 뉴스 수: {len(filtered_items)}")
            filtered_items = await _filter_news_by_relevance(
                news_items=filtered_items,
                job_role_name=job_role_name,
                max_results=max_results
            )
            logger.info(f"[Naver News] LLM 필터링 완료 - {len(filtered_items)}개 뉴스")
        
        # 결과 포맷팅
        news_list = []
        for item in filtered_items[:max_results]:
            news_list.append({
                "title": item.get("title", "").replace("<b>", "").replace("</b>", ""),
                "link": item.get("link", ""),
                "description": item.get("description", "").replace("<b>", "").replace("</b>", ""),
                "pub_date": item.get("pubDate", "")
            })
        
        logger.info(f"[Naver News] 최종 반환 - {len(news_list)}개 뉴스")
        
        result_json = json.dumps({
            "job_role_name": job_role_name,
            "query": search_query,
            "news_count": len(news_list),
            "news": news_list
        }, ensure_ascii=False)
        
        return result_json
    except Exception as e:
        logger.error(f"[Naver News] 뉴스 검색 중 오류 발생: {str(e)}")
        return json.dumps({
            "error": f"뉴스 검색 중 오류 발생: {str(e)}"
        }, ensure_ascii=False)

