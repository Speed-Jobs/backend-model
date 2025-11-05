"""
LG 채용공고 비동기 통합 크롤러 (성능 최적화 버전)
1. API로 공고 리스트 수집
2. 비동기로 여러 상세 페이지를 동시 크롤링
3. 병렬 LLM 처리로 속도 향상
"""

import json
import re
import os
import asyncio
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.async_api import async_playwright, Browser, Page

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None


class LGCareerAsyncCrawler:
    def __init__(self, max_concurrent: int = 5):
        """
        비동기 크롤러 초기화
        
        Args:
            max_concurrent: 동시 크롤링 최대 개수 (기본: 5개)
        """
        load_dotenv()
        
        # API 설정
        self.base_url = "https://api.careers.lg.com"
        self.list_endpoint = "/rmk/job/retrieveJobNoticesList"
        self.detail_url_template = "https://careers.lg.com/apply/detail?id={}"
        
        # 최소한의 헤더만 사용
        self.headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'referer': 'https://careers.lg.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # 동시 실행 제어
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # OpenAI 클라이언트 초기화
        self.openai_client = self._get_openai_client()
    
    def _get_openai_client(self) -> Optional[Any]:
        """OpenAI 비동기 클라이언트 초기화"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or AsyncOpenAI is None:
            print("⚠️ OpenAI API 키가 없습니다. 상세 정보 추출이 제한됩니다.")
            return None
        return AsyncOpenAI(api_key=api_key)
    
    # ==================== 1단계: API로 기본 정보 수집 ====================
    
    def get_job_list_from_api(self) -> List[Dict]:
        """API에서 채용공고 리스트 가져오기 (동기)"""
        url = f"{self.base_url}{self.list_endpoint}"
        
        payload = {
            "lnbSearch": "",
            "hashTagText": "",
            "recDate": "CREATION_DATE",
            "order": "DESC",
            "careerList": [],
            "companyCodeList": [],
            "desireLocList": [],
            "jobGroupList": []
        }
        
        try:
            print("📡 [1/4] API 호출 중...")
            response = requests.post(url, headers=self.headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                job_list = data.get('data', {}).get('jobNoticeList', [])
                
                jobs = []
                for job in job_list:
                    job_id = job.get('jobNoticeId')
                    
                    # API에서 URL 필드가 있으면 사용, 없으면 템플릿으로 생성
                    job_url = job.get('jobNoticeUrl') or job.get('url')
                    if not job_url:
                        job_url = self.detail_url_template.format(job_id)
                    
                    job_info = {
                        'id': job_id,
                        'title': job.get('jobNoticeName'),
                        'company': job.get('companyName'),
                        'career_type': job.get('careerTypeName'),
                        'job_group': job.get('jobGroupName'),
                        'status': job.get('noticeStatusName'),
                        'deadline': job.get('recEndDateTime'),
                        'url': job_url
                    }
                    jobs.append(job_info)
                
                print(f"✅ API에서 {len(jobs)}개 공고 수집 완료")
                return jobs
            else:
                print(f"❌ API 호출 실패: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ API 요청 오류: {e}")
            return []
    
    # ==================== 2단계: 비동기 크롤링 ====================
    
    def _clean_html_text(self, html: str) -> str:
        """HTML에서 텍스트 추출 및 정제"""
        soup = BeautifulSoup(html, "html.parser")
        
        # 스크립트/스타일 제거
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        
        text = soup.get_text("\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t\x0b\x0c\r]+", " ", text)
        
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)
    
    async def _extract_detail_with_llm(self, raw_text: str, basic_info: Dict, retry_count: int = 3) -> Dict:
        """LLM을 사용하여 상세 정보 추출 (비동기)"""
        if self.openai_client is None:
            return {}
        
        system_prompt = """
너는 채용공고에서 정보를 추출하는 에이전트야.

**핵심 규칙: 모든 내용은 원본 텍스트를 그대로 추출해야 해. 절대 요약하거나 재작성하지 마!**

아래 JSON 형식으로 정보를 추출해:

{
  "description": string,           // 주요 업무/담당업무/업무내용 섹션의 원문 그대로
  "requirements": string,          // 자격요건/필수요건/지원자격 섹션의 원문 그대로
  "preferred": string,             // 우대사항/우대조건 섹션의 원문 그대로
  "benefits": string,              // 복리후생/근무조건/혜택 섹션의 원문 그대로
  "process": string,               // 전형절차/채용절차/전형단계 섹션의 원문 그대로
  "location": string,              // 근무지/근무지역 (간단한 위치 정보)
  "contact": string                // 담당자/문의처/연락처
}

**추출 방법**:
1. 각 항목에 해당하는 섹션을 찾아서 내용을 **있는 그대로 복사**
2. 불릿 포인트(•, -, 1. 등)와 줄바꿈도 **원본 그대로 유지**
3. 요약하거나 의역하지 말고 **전체 내용을 다 포함**
4. 해당 섹션이 없으면 null 처리
5. 여러 문단이면 모두 포함

**예시**:
원문: "• Python 개발 경험 3년 이상\n• Django/FastAPI 프레임워크 사용 경험"
추출: "• Python 개발 경험 3년 이상\n• Django/FastAPI 프레임워크 사용 경험"  (그대로!)
        """.strip()
        
        user_prompt = f"""
채용공고 제목: {basic_info.get('title', 'N/A')}
회사: {basic_info.get('company', 'N/A')}

아래는 채용공고 페이지의 전체 텍스트야. 각 섹션을 찾아서 원문 그대로 추출해줘:

{raw_text[:8000]}
        """.strip()
        
        for attempt in range(retry_count):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # 더 정확한 추출을 위해 낮춤
                    max_tokens=3000  # 원문 그대로 추출하므로 토큰 증가
                )
                
                content = response.choices[0].message.content
                
                # JSON 추출
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    detail_info = json.loads(json_match.group(0))
                    return detail_info
                
            except Exception as e:
                if attempt == retry_count - 1:
                    print(f"  ⚠️ LLM 추출 실패 (최종): {e}")
                else:
                    await asyncio.sleep(1)  # 재시도 전 대기
        
        return {}
    
    async def crawl_single_job(self, browser: Browser, job: Dict, index: int, total: int) -> Dict:
        """개별 채용공고 크롤링 (비동기)"""
        async with self.semaphore:  # 동시 실행 수 제한
            job_id = str(job['id'])
            url = job['url']
            
            print(f"[{index}/{total}] 🔍 {job['title']}")
            
            try:
                # 새 페이지 생성
                page: Page = await browser.new_page()
                
                # 페이지 접속
                await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                await page.wait_for_timeout(2000)
                
                # HTML 수집
                html = await page.content()
                clean_text = self._clean_html_text(html)
                
                # 페이지 닫기
                await page.close()
                
                # LLM으로 상세 정보 추출 (병렬 처리)
                detail_info = await self._extract_detail_with_llm(clean_text, job)
                
                # 기본 정보와 상세 정보 병합
                merged = {**job, **detail_info}
                
                print(f"[{index}/{total}] ✅ 완료: {job_id}")
                return merged
                
            except Exception as e:
                print(f"[{index}/{total}] ❌ 실패 ({job_id}): {str(e)[:100]}")
                return job  # 실패시 기본 정보만 반환
    
    async def crawl_all_async(self, jobs: List[Dict]) -> List[Dict]:
        """모든 공고를 비동기로 크롤링"""
        print(f"\n🔍 [2/4] 비동기 크롤링 시작 ({len(jobs)}개, 동시 {self.max_concurrent}개)\n")
        
        async with async_playwright() as p:
            # 브라우저 실행 (재사용)
            browser: Browser = await p.chromium.launch(headless=True)
            
            # 모든 크롤링 작업을 비동기로 실행
            tasks = [
                self.crawl_single_job(browser, job, idx + 1, len(jobs))
                for idx, job in enumerate(jobs)
            ]
            
            # 모든 작업 동시 실행
            results = await asyncio.gather(*tasks)
            
            await browser.close()
        
        return results
    
    # ==================== 3단계: 통합 실행 ====================
    
    def crawl_all(self, max_jobs: Optional[int] = None) -> List[Dict]:
        """전체 크롤링 파이프라인 실행"""
        print("="*80)
        print("🚀 LG 채용공고 비동기 크롤러 시작")
        print("="*80 + "\n")
        
        # 1단계: API로 기본 정보 수집
        basic_jobs = self.get_job_list_from_api()
        
        if not basic_jobs:
            print("❌ API에서 데이터를 가져오지 못했습니다.")
            return []
        
        # 최대 개수 제한
        if max_jobs:
            basic_jobs = basic_jobs[:max_jobs]
            print(f"📊 최대 {max_jobs}개로 제한\n")
        
        # 2단계: 비동기 크롤링 실행
        detailed_jobs = asyncio.run(self.crawl_all_async(basic_jobs))
        
        print(f"\n✅ [3/4] 전체 크롤링 완료: {len(detailed_jobs)}개")
        return detailed_jobs
    
    # ==================== 4단계: 결과 저장 ====================
    
    def save_results(self, jobs: List[Dict], output_dir: str = "output"):
        """결과를 JSON과 CSV로 저장"""
        print(f"\n💾 [4/4] 결과 저장 중...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 저장
        json_file = output_path / f"lg_jobs_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
        print(f"  ✅ JSON 저장: {json_file}")
        
        # CSV 저장
        try:
            import csv
            csv_file = output_path / f"lg_jobs_{timestamp}.csv"
            
            if jobs:
                # 모든 키 수집
                all_keys = set()
                for job in jobs:
                    all_keys.update(job.keys())
                
                with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    writer.writerows(jobs)
                
                print(f"  ✅ CSV 저장: {csv_file}")
        except Exception as e:
            print(f"  ⚠️ CSV 저장 실패: {e}")
        
        print("\n" + "="*80)
        print(f"🎉 모든 작업 완료! (동시 처리: {self.max_concurrent}개)")
        print("="*80)
    
    def print_summary(self, jobs: List[Dict]):
        """결과 요약 출력"""
        print("\n" + "="*80)
        print("📊 크롤링 결과 요약")
        print("="*80)
        
        for idx, job in enumerate(jobs[:10], 1):  # 처음 10개만
            print(f"\n{idx}. {job.get('title', 'N/A')}")
            print(f"   회사: {job.get('company', 'N/A')}")
            print(f"   직군: {job.get('job_group', 'N/A')}")
            print(f"   마감: {job.get('deadline', 'N/A')}")
            
            if job.get('description'):
                desc = job['description'][:80] + "..." if len(job.get('description', '')) > 80 else job.get('description', '')
                print(f"   📝 {desc}")
        
        if len(jobs) > 10:
            print(f"\n... 외 {len(jobs) - 10}개 (전체 결과는 파일 참조)")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LG 채용공고 비동기 크롤러")
    parser.add_argument("--max-jobs", type=int, help="최대 크롤링 개수")
    parser.add_argument("--concurrent", type=int, default=30, help="동시 크롤링 개수 (기본: 5)")
    parser.add_argument("--output-dir", default="output", help="출력 폴더")
    parser.add_argument("--no-summary", action="store_true", help="요약 출력 생략")
    args = parser.parse_args()
    
    # 크롤러 실행
    crawler = LGCareerAsyncCrawler(max_concurrent=args.concurrent)
    jobs = crawler.crawl_all(max_jobs=args.max_jobs)
    
    if jobs:
        # 결과 저장
        crawler.save_results(jobs, output_dir=args.output_dir)
        
        # 요약 출력
        if not args.no_summary:
            crawler.print_summary(jobs)
    else:
        print("❌ 크롤링된 데이터가 없습니다.")


if __name__ == "__main__":
    main()