"""
현대오토에버 채용공고 크롤러
1. HTML 파싱으로 공고 URL 리스트 수집
2. Async Playwright로 병렬 크롤링 (단일 브라우저 + 여러 페이지)
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
from playwright.async_api import async_playwright, BrowserContext, Page
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from app.services import resolve_dir, get_output_dir, get_img_dir
except ModuleNotFoundError:
    import sys
    _p = Path(__file__).resolve().parents[4]
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
    from app.services import resolve_dir, get_output_dir, get_img_dir

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None


def load_env() -> None:
    """Load environment variables from .env with fallbacks.

    Order: nearest discoverable .env from CWD → fproject/.env → backend-model/.env
    """
    try:
        from dotenv import find_dotenv
        found = find_dotenv(usecwd=True)
        if found:
            load_dotenv(found, override=False)
    except Exception:
        pass

    try:
        proj_env = Path(__file__).resolve().parents[5] / ".env"
        if proj_env.exists():
            load_dotenv(dotenv_path=proj_env, override=False)
    except Exception:
        pass

    try:
        backend_env = Path(__file__).resolve().parents[4] / ".env"
        if backend_env.exists():
            load_dotenv(dotenv_path=backend_env, override=False)
    except Exception:
        pass


load_env()


def get_openai_client() -> Optional[Any]:
    """OpenAI 비동기 클라이언트 초기화"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if AsyncOpenAI is None:
        return None
    return AsyncOpenAI(api_key=api_key)


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def summarize_with_llm(raw_text: str, url: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """LLM을 사용하여 채용공고에서 정보 추출 (description 제외) - 비동기"""
    client = get_openai_client()
    if client is None:
        return {}

    system_prompt = """
당신은 채용 공고 웹페이지에서 구조화된 정보를 추출하는 전문가입니다.
주어진 HTML 콘텐츠에서 다음 필드들을 정확하게 추출하여 JSON 형식으로 반환하세요.

# 추출할 필드
- title: 공고 제목
- location: 근무 위치
- employment_type: 고용 형태 (정규직, 계약직, 파트타임 등)
- experience: 경력 요구사항 (신입, 경력, 경력무관, 인턴 등)
- crawl_date: 크롤링 날짜 (YYYY-MM-DD 형식)
- posted_date: 공고 게시일 (YYYY-MM-DD 형식, 상시채용인 경우 크롤링 날짜와 동일)
- expired_date: 공고 마감일 (YYYY-MM-DD 형식, 없으면 null)
- meta_data: 위 필드 외 추가 정보를 담은 JSON 객체 (예: 직군, 연봉정보, 복리후생, 우대사항, 기술스택 등)

※ company, description, html, url은 별도로 처리되므로 추출하지 않습니다.

# 중요 지침
1. 날짜는 반드시 YYYY-MM-DD 형식으로 통일
2. 정보가 없는 경우 null 반환 (빈 문자열 X)
3. meta_data는 의미있는 키 이름으로 구조화 (영문 snake_case 사용)
4. 모든 텍스트는 공백 정리 및 정규화

---
# Example
{
    "title": "백엔드 개발자 (Python/Django)",
    "location": "서울 강남구",
    "employment_type": "정규직",
    "experience": "경력 3~5년",
    "crawl_date": "2025-11-05",
    "posted_date": "2025-10-28",
    "expired_date": "2025-11-30",
    "meta_data": {
        "job_category": "IT/개발",
        "tech_stack": ["Python", "Django", "PostgreSQL"],
        "benefits": "4대보험, 연차, 재택근무"
    }
}
---
"""
    
    user_prompt = (
        f"오늘 날짜는 {datetime.now().strftime('%Y-%m-%d')}이고, 이 날짜를 crawl_date로 사용해. "
        f"공고 URL: {url}\n\n"
        f"아래 텍스트에서 정보를 추출해줘:\n\n{raw_text[:8000]}"
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=3000,
    )
    
    content = response.choices[0].message.content if response and response.choices else "{}"
    
    # JSON만 남도록 추출
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    json_text = json_match.group(0) if json_match else content
    
    try:
        data = json.loads(json_text)
        return data
    except Exception:
        return {}


async def crawl_single_job(context: BrowserContext, job: Dict, index: int, total: int, semaphore: asyncio.Semaphore, screenshot_dir: Path = None) -> Optional[Dict[str, Any]]:
    """개별 채용공고 크롤링 (비동기) - context 공유, 각자 page 생성, semaphore로 동시 실행 제한"""
    # Semaphore로 동시 실행 개수 제한
    async with semaphore:
        page = None
        try:
            job_url = job['url']

            # 새 페이지 생성 (context는 공유)
            page = await context.new_page()

            print(f"  [{index}/{total}] 상세 페이지 로딩 중...")
            await page.goto(job_url, timeout=30000)
            await page.wait_for_timeout(2000)

            today = datetime.now().strftime('%Y-%m-%d')

            # 기본 정보 설정
            job_info = {
                "title": None,
                "company": "현대오토에버",
                "location": None,
                "employment_type": None,
                "experience": None,
                "crawl_date": today,
                "posted_date": None,
                "expired_date": None,
                "description": None,
                "url": job_url,
                "meta_data": {},
                "screenshots": {},  # 스크린샷 경로 저장용
            }

            # description 추출 - CSS Selector 사용
            try:
                description_element = await page.query_selector("div.sc-2109ef67-5.cSmBgl")
                if description_element:
                    description_text = await description_element.inner_text()
                    if description_text:
                        job_info["description"] = description_text
                        print(f"  [{index}/{total}] description 추출 완료")
                else:
                    # 백업: body 전체 사용
                    print(f"  [{index}/{total}] description selector를 찾지 못함, body 전체 사용")
                    job_info["description"] = await page.inner_text("body")
            except Exception as e:
                print(f"  [{index}/{total}] description 추출 실패, body 전체 사용: {e}")
                job_info["description"] = await page.inner_text("body")

            # 전체 페이지 스크린샷 저장
            if screenshot_dir:
                try:
                    # URL에서 job ID 추출하여 파일명에 사용
                    job_id = job.get('id', f"job_{index}")

                    screenshot_filename = f"hyundai_autoever_job_{job_id}.png"
                    screenshot_path = screenshot_dir / screenshot_filename

                    # 전체 페이지 스크린샷
                    await page.screenshot(path=str(screenshot_path), full_page=True)

                    job_info["screenshots"]["combined"] = str(screenshot_path)
                    print(f"  [{index}/{total}] 전체 페이지 스크린샷 저장: {screenshot_filename}")
                except Exception as e:
                    print(f"  [{index}/{total}] 스크린샷 저장 실패: {e}")

            # LLM으로 나머지 필드 파싱 (description 제외)
            try:
                full_text = await page.inner_text("body")
                parsed_data = await summarize_with_llm(full_text, job_url)
                
                if parsed_data:
                    # description을 제외한 필드만 업데이트
                    for key in ["title", "location", "employment_type", "experience",
                               "posted_date", "expired_date", "meta_data"]:
                        if key in parsed_data and parsed_data[key] is not None:
                            job_info[key] = parsed_data[key]
                    print(f"  [{index}/{total}] LLM 파싱 완료")
            except Exception as e:
                print(f"  [{index}/{total}] LLM 파싱 실패 (description은 저장됨): {e}")

            # 페이지 닫기
            await page.close()
            
            print(f"  [{index}/{total}] ✅ 완료: {job_info.get('title', 'N/A')}")
            return job_info

        except Exception as e:
            print(f"  [{index}/{total}] ❌ 상세 정보 추출 실패: {e}")
            # 페이지가 열려있으면 닫기
            if page:
                try:
                    await page.close()
                except:
                    pass
            return None


class HyundaiAutoeverCrawler:
    def __init__(self, max_concurrent: int = 30):
        """
        크롤러 초기화
        
        Args:
            max_concurrent: 동시 크롤링 최대 개수 (기본: 30)
        """
        load_dotenv()
        
        # 기본 URL 설정
        self.base_url = "https://career.hyundai-autoever.com"
        self.list_url = f"{self.base_url}/ko/apply"
        
        # 헤더 설정
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        }
        
        # 동시 실행 제어
        self.max_concurrent = max_concurrent
    
    # ==================== 1단계: HTML 파싱으로 URL 수집 ====================
    
    def get_job_list(self) -> List[Dict]:
        """HTML 파싱으로 채용공고 URL 리스트 가져오기"""
        print("📡 [1/3] 채용공고 목록 수집 중...")
        
        try:
            response = requests.get(self.list_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # CSS 선택자로 /ko/o/로 시작하는 모든 링크 선택
            job_links = soup.select('a[href^="/ko/o/"]')
            
            # URL 중복 제거를 위한 set
            seen_urls = set()
            jobs = []
            
            for link in job_links:
                href = link.get('href')
                full_url = self.base_url + href
                
                # 중복 제거
                if full_url not in seen_urls:
                    seen_urls.add(full_url)
                    
                    # URL에서 ID 추출
                    job_id = href.split('/')[-1]
                    
                    # 링크 텍스트에서 제목 추출 (있으면)
                    title = link.get_text(strip=True) or f"공고_{job_id}"
                    
                    job_info = {
                        'id': job_id,
                        'title': title,
                        'url': full_url
                    }
                    jobs.append(job_info)
            
            print(f"✅ {len(jobs)}개 채용공고 URL 수집 완료")
            return jobs
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 요청 중 오류 발생: {e}")
            return []
        except Exception as e:
            print(f"❌ 파싱 중 오류 발생: {e}")
            return []
    
    # ==================== 2단계: 비동기 병렬 크롤링 ====================

    async def crawl_details_async(self, jobs: List[Dict], screenshot_dir: Path = None) -> List[Dict]:
        """비동기로 모든 공고의 상세 정보 크롤링 (단일 브라우저 + 여러 페이지 + Semaphore)"""
        if not jobs:
            return []

        print(f"\n🔍 [2/3] 비동기 병렬 크롤링 시작 ({len(jobs)}개, 최대 동시 {self.max_concurrent}개)\n")

        async with async_playwright() as p:
            # 단일 브라우저 인스턴스
            browser = await p.chromium.launch(headless=True)
            # 단일 context
            context = await browser.new_context()

            # Semaphore 생성 (최대 동시 실행 수 제한)
            semaphore = asyncio.Semaphore(self.max_concurrent)

            # 모든 작업을 asyncio.gather로 동시 실행 (Semaphore가 제어)
            tasks = [
                crawl_single_job(context, job, idx + 1, len(jobs), semaphore, screenshot_dir)
                for idx, job in enumerate(jobs)
            ]

            # 모든 작업 실행
            results = await asyncio.gather(*tasks)

            # None이 아닌 결과만 필터링
            detailed_jobs = [job for job in results if job is not None]

            await context.close()
            await browser.close()

        print(f"\n✅ [2/3] 비동기 크롤링 완료: {len(detailed_jobs)}/{len(jobs)}개 성공")
        return detailed_jobs

    def crawl_details(self, jobs: List[Dict], screenshot_dir: Path = None) -> List[Dict]:
        """동기 래퍼: 비동기 크롤링 실행"""
        return asyncio.run(self.crawl_details_async(jobs, screenshot_dir))
    
    # ==================== 3단계: 결과 저장 ====================
    
    def save_results(self, jobs: List[Dict], output_dir: str = "output"):
        """결과를 JSON과 CSV로 저장"""
        print(f"\n💾 [3/3] 결과 저장 중...")
        
        output_path = resolve_dir(Path(output_dir), get_output_dir())
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 저장
        json_file = output_path / f"hyundai_autoever_jobs_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
        print(f"  ✅ JSON 저장: {json_file}")
        
        # CSV 저장
        try:
            import csv
            csv_file = output_path / f"hyundai_autoever_jobs_{timestamp}.csv"

            if jobs:
                # CSV에 필요한 필드만 포함 (html 제외)
                simple_jobs = []
                for job in jobs:
                    simple_job = job.copy()
                    # html 필드 제거
                    simple_job.pop('html', None)
                    # meta_data를 문자열로 변환
                    if 'meta_data' in simple_job:
                        simple_job['meta_data'] = json.dumps(simple_job['meta_data'], ensure_ascii=False)
                    # screenshots를 문자열로 변환
                    if 'screenshots' in simple_job:
                        simple_job['screenshots'] = json.dumps(simple_job['screenshots'], ensure_ascii=False)
                    simple_jobs.append(simple_job)
                
                all_keys = set()
                for job in simple_jobs:
                    all_keys.update(job.keys())
                
                with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                    writer.writeheader()
                    writer.writerows(simple_jobs)
                
                print(f"  ✅ CSV 저장: {csv_file}")
        except Exception as e:
            print(f"  ⚠️ CSV 저장 실패: {e}")
        
        print("\n" + "="*80)
        print(f"🎉 모든 작업 완료! (비동기 동시 처리: 최대 {self.max_concurrent}개)")
        print("="*80)


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="현대오토에버 채용공고 크롤러 (Async)")
    parser.add_argument("--max-jobs", type=int, help="최대 크롤링 개수 (테스트용)")
    parser.add_argument("--max-concurrent", type=int, default=30, help="최대 동시 크롤링 개수 (기본: 30)")
    parser.add_argument("--output-dir", default="../../output", help="출력 폴더 (기본: ../../output)")
    parser.add_argument("--screenshot-dir", default="../../img", help="스크린샷 폴더 (기본: ../../img)")
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 현대오토에버 채용공고 크롤러 시작 (Async Playwright)")
    print("="*80 + "\n")
    
    # 크롤러 초기화
    crawler = HyundaiAutoeverCrawler(max_concurrent=args.max_concurrent)
    
    # 1단계: URL 수집
    basic_jobs = crawler.get_job_list()
    
    if not basic_jobs:
        print("❌ 채용공고 목록을 가져오지 못했습니다.")
        return
    
    # 최대 개수 제한 (테스트용)
    if args.max_jobs:
        basic_jobs = basic_jobs[:args.max_jobs]
        print(f"📊 테스트: 최대 {args.max_jobs}개로 제한\n")

    # 스크린샷 디렉토리 준비
    screenshot_dir = resolve_dir(Path(args.screenshot_dir), get_img_dir())
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # 2단계: 비동기 병렬 크롤링으로 상세 정보 수집
    detailed_jobs = crawler.crawl_details(basic_jobs, screenshot_dir=screenshot_dir)
    
    if not detailed_jobs:
        print("❌ 상세 정보를 수집하지 못했습니다.")
        return
    
    # 3단계: 결과 저장
    crawler.save_results(detailed_jobs, output_dir=args.output_dir)
    
    # 요약 출력
    print("\n" + "="*80)
    print("📊 크롤링 결과 요약 (처음 3개)")
    print("="*80)
    
    for idx, job in enumerate(detailed_jobs[:3], 1):
        print(f"\n{idx}. {job.get('title', 'N/A')}")
        print(f"   회사: {job.get('company', 'N/A')}")
        print(f"   위치: {job.get('location', 'N/A')}")
        print(f"   경력: {job.get('experience', 'N/A')}")
        print(f"   마감: {job.get('expired_date', 'N/A')}")
        print(f"   URL: {job.get('url', 'N/A')}")
    
    if len(detailed_jobs) > 3:
        print(f"\n... 외 {len(detailed_jobs) - 3}개 (전체 결과는 파일 참조)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
import sys as _sys
from pathlib import Path as _P0
_backend_root = _P0(__file__).resolve().parents[4]
if str(_backend_root) not in _sys.path:
    _sys.path.append(str(_backend_root))
import sys as _sys
from pathlib import Path as _PX
_backend_root = _PX(__file__).resolve().parents[4]
if str(_backend_root) not in _sys.path:
    _sys.path.insert(0, str(_backend_root))
