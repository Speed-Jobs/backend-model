"""
LG 채용공고 크롤러 (리팩토링)
1. API로 공고 URL 리스트 수집
2. Async Playwright로 병렬 크롤링 (단일 브라우저 + 여러 페이지)

주요 개선사항:
- AsyncOpenAI context manager 사용으로 확실한 cleanup
- Playwright 리소스 관리 강화
- 에러 핸들링 개선
"""

import json
import re
import os
import asyncio
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv
try:
    from dotenv import find_dotenv  # type: ignore
except Exception:
    find_dotenv = None  # type: ignore
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

try:
    from app.utils.s3_uploader import upload_screenshot_to_s3
except ModuleNotFoundError:
    import sys
    _p = Path(__file__).resolve().parents[4]
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
    from app.utils.s3_uploader import upload_screenshot_to_s3


def load_env() -> None:
    """Load environment variables from .env with fallbacks."""
    try:
        if find_dotenv is not None:
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


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
async def summarize_with_llm(raw_text: str, url: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """LLM을 사용하여 채용공고에서 정보 추출 - AsyncOpenAI context manager 사용"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or AsyncOpenAI is None:
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
- meta_data: 위 필드 외 추가 정보를 담은 JSON 객체

※ company, description, html, url은 별도로 처리되므로 추출하지 않습니다.

# 중요 지침
1. 날짜는 반드시 YYYY-MM-DD 형식으로 통일
2. 정보가 없는 경우 null 반환 (빈 문자열 X)
3. ⭐ **meta_data는 반드시 한국어 키로만 구성** (예: "직무분야", "우대사항", "복리후생")
4. ⭐ **meta_data에는 위의 기본 필드(title, company, location, employment_type, experience, crawl_date, posted_date, expired_date, description)와 중복되는 정보를 절대 포함하지 말 것**
5. ⭐ meta_data에는 오직 추가적인 보조 정보만 포함 (예: 자격요건, 우대사항, 복리후생, 담당업무, 학력요건, 전형절차 등)
6. ⭐ meta_data에는 기술스택/소프트스킬을 넣지 않는다 (예: Python, Django, AWS, Docker 등 제외)
7. ⭐ meta_data 키는 절대 영어를 사용하지 말고 반드시 한국어만 사용할 것
8. 모든 텍스트는 공백 정리 및 정규화

---
# Example (한국어 키 필수!)
{
    "title": "백엔드 개발자 (Python/Django)",
    "location": "서울 강남구",
    "employment_type": "정규직",
    "experience": "경력 3~5년",
    "crawl_date": "2025-12-13",
    "posted_date": "2025-11-28",
    "expired_date": "2025-12-31",
    "meta_data": {
        "직무분야": "IT/개발",
        "우대사항": ["AWS 경험", "Docker/K8s 사용 경험", "MSA 아키텍처 이해"],
        "복리후생": ["건강검진", "자기계발비 지원", "유연근무제"],
        "학력요건": "학사 이상",
        "전형절차": "서류전형 > 1차 면접 > 2차 면접 > 최종합격",
        "담당업무": "백엔드 API 설계 및 개발, 데이터베이스 설계",
        "자격요건": "Python 3년 이상 경험"
    }
}
---

⚠️ 경고: meta_data의 키는 반드시 한국어여야 합니다!
❌ 잘못된 예: "job_category", "tech_stack", "benefits"
✅ 올바른 예: "직무분야", "우대사항", "복리후생"
"""
    
    user_prompt = (
        f"오늘 날짜는 {datetime.now().strftime('%Y-%m-%d')}이고, 이 날짜를 crawl_date로 사용해. "
        f"공고 URL: {url}\n\n"
        f"아래 텍스트에서 정보를 추출해줘:\n\n{raw_text[:8000]}"
    )

    # AsyncOpenAI를 async with로 사용하여 확실한 cleanup
    async with AsyncOpenAI(api_key=api_key) as client:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=8000,
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


async def crawl_single_job(
    context: BrowserContext, 
    job: Dict, 
    index: int, 
    total: int, 
    semaphore: asyncio.Semaphore, 
    screenshot_dir: Path = None
) -> Optional[Dict[str, Any]]:
    """개별 채용공고 크롤링 (비동기) - context 공유, 각자 page 생성, semaphore로 동시 실행 제한"""
    page = None
    
    # Semaphore로 동시 실행 개수 제한
    async with semaphore:
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
                "company": "LG CNS",
                "location": None,
                "employment_type": None,
                "experience": None,
                "crawl_date": today,
                "posted_date": None,
                "expired_date": None,
                "description": None,
                "url": job_url,
                "meta_data": {},
                "screenshots": {},
            }

            # description 추출 - 두 개의 CSS Selector를 순서대로 합치기
            try:
                description_parts = []
                
                # 첫 번째: MuiCollapse-root
                collapse_element = await page.query_selector("div.MuiCollapse-root.MuiCollapse-vertical.MuiCollapse-entered.css-c4sutr")
                if collapse_element:
                    text1 = await collapse_element.inner_text()
                    if text1:
                        description_parts.append(text1.strip())
                
                # 두 번째: MuiBox-root css-h38rax
                box_element = await page.query_selector("div.MuiBox-root.css-h38rax")
                if box_element:
                    text2 = await box_element.inner_text()
                    if text2:
                        description_parts.append(text2.strip())
                
                # 두 텍스트 합치기
                if description_parts:
                    job_info["description"] = "\n\n".join(description_parts)
                    print(f"  [{index}/{total}] description 추출 완료 ({len(description_parts)}개 섹션)")
                else:
                    print(f"  [{index}/{total}] description selector를 찾지 못함, body 전체 사용")
                    job_info["description"] = await page.inner_text("body")
            except Exception as e:
                print(f"  [{index}/{total}] description 추출 실패, body 전체 사용: {e}")
                job_info["description"] = await page.inner_text("body")

            # 스크린샷 S3 업로드
            if screenshot_dir:
                try:
                    job_id = job.get('id', f"job_{index}")
                    screenshot_filename = f"lg_cns_job_{job_id}.png"

                    # S3에 직접 업로드
                    screenshot_bytes = await page.screenshot(full_page=True)
                    s3_url = upload_screenshot_to_s3(screenshot_bytes, screenshot_filename)

                    if s3_url:
                        job_info["screenshots"]["combined"] = s3_url
                        print(f"  [{index}/{total}] S3 업로드 성공: {s3_url}")
                    else:
                        print(f"  [{index}/{total}] S3 업로드 실패")
                except Exception as e:
                    print(f"  [{index}/{total}] 스크린샷 처리 실패: {e}")

            # LLM으로 나머지 필드 파싱 (company, description, html, url 제외)
            try:
                # LLM에 전달할 텍스트 추출
                info_box = await page.query_selector("div.MuiBox-root.css-fflez4")
                if info_box:
                    llm_text = await info_box.inner_text()
                else:
                    llm_text = await page.inner_text("body")
                
                parsed_data = await summarize_with_llm(llm_text, job_url)
                
                if parsed_data:
                    for key in ["title", "location", "employment_type", "experience",
                               "posted_date", "expired_date", "meta_data"]:
                        if key in parsed_data and parsed_data[key] is not None:
                            job_info[key] = parsed_data[key]
                    print(f"  [{index}/{total}] LLM 파싱 완료")
            except Exception as e:
                print(f"  [{index}/{total}] LLM 파싱 실패 (description은 저장됨): {e}")

            print(f"  [{index}/{total}] ✅ 완료: {job_info.get('title', 'N/A')}")
            return job_info

        except Exception as e:
            print(f"  [{index}/{total}] ❌ 상세 정보 추출 실패: {e}")
            return None
            
        finally:
            # 페이지 확실하게 닫기
            if page:
                try:
                    await page.close()
                except Exception as e:
                    print(f"  [{index}/{total}] 페이지 닫기 실패: {e}")


class LGCareerCrawler:
    def __init__(self, max_concurrent: int = 30):
        """
        크롤러 초기화
        
        Args:
            max_concurrent: 동시 크롤링 최대 개수 (기본: 30)
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
    
    def get_job_list_from_api(self) -> List[Dict]:
        """API에서 채용공고 리스트 가져오기"""
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
            print("📡 [1/3] API 호출 중...")
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
                    
                    # 기본 정보 (API에서 가져온 것)
                    job_info = {
                        'id': job_id,
                        'url': job_url,
                        'api_title': job.get('jobNoticeName'),
                        'api_company': job.get('companyName'),
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

    async def crawl_details_async(self, jobs: List[Dict], screenshot_dir: Path = None) -> List[Dict]:
        """비동기로 모든 공고의 상세 정보 크롤링"""
        if not jobs:
            return []

        print(f"\n🔍 [2/3] 비동기 병렬 크롤링 시작 ({len(jobs)}개, 최대 동시 {self.max_concurrent}개)\n")

        browser = None
        context = None
        
        try:
            async with async_playwright() as p:
                # 단일 브라우저 인스턴스
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()

                # Semaphore 생성 (최대 동시 실행 수 제한)
                semaphore = asyncio.Semaphore(self.max_concurrent)

                # 모든 작업을 asyncio.gather로 동시 실행
                tasks = [
                    crawl_single_job(context, job, idx + 1, len(jobs), semaphore, screenshot_dir)
                    for idx, job in enumerate(jobs)
                ]

                # 모든 작업 실행
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 성공한 결과만 필터링 (예외 제외)
                detailed_jobs = [
                    job for job in results 
                    if job is not None and not isinstance(job, Exception)
                ]

                print(f"\n✅ [2/3] 비동기 크롤링 완료: {len(detailed_jobs)}/{len(jobs)}개 성공")
                return detailed_jobs
                
        except Exception as e:
            print(f"❌ 크롤링 중 에러 발생: {e}")
            return []
            
        finally:
            # 리소스 확실하게 정리
            if context:
                try:
                    await context.close()
                except Exception as e:
                    print(f"⚠️ Context 닫기 실패: {e}")
            
            if browser:
                try:
                    await browser.close()
                except Exception as e:
                    print(f"⚠️ Browser 닫기 실패: {e}")

    def crawl_details(self, jobs: List[Dict], screenshot_dir: Path = None) -> List[Dict]:
        """동기 래퍼: 비동기 크롤링 실행"""
        return asyncio.run(self.crawl_details_async(jobs, screenshot_dir))

    def save_results(self, jobs: List[Dict], output_dir: str = "../../output"):
        """결과를 JSON과 CSV로 저장"""
        print(f"\n💾 [3/3] 결과 저장 중...")
        
        output_path = resolve_dir(Path(output_dir), get_output_dir())
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 저장
        json_file = output_path / f"lg_jobs.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, ensure_ascii=False, indent=2)
        print(f"  ✅ JSON 저장: {json_file}")
        
        # CSV 저장
        try:
            import csv
            csv_file = output_path / f"lg_jobs.csv"

            if jobs:
                simple_jobs = []
                for job in jobs:
                    simple_job = job.copy()
                    simple_job.pop('html', None)
                    
                    if 'meta_data' in simple_job:
                        simple_job['meta_data'] = json.dumps(simple_job['meta_data'], ensure_ascii=False)
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
        print(f"🎉 LG 크롤링 완료! (비동기 동시 처리: 최대 {self.max_concurrent}개)")
        print("="*80)


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="LG 채용공고 크롤러 (Refactored)")
    parser.add_argument("--max-jobs", type=int, help="최대 크롤링 개수 (테스트용)")
    parser.add_argument("--max-concurrent", type=int, default=30, help="최대 동시 크롤링 개수 (기본: 30)")
    parser.add_argument("--output-dir", default="../../output", help="출력 폴더 (기본: ../../output)")
    parser.add_argument("--screenshot-dir", default="../../img", help="스크린샷 폴더 (기본: ../../img)")
    args = parser.parse_args()

    print("="*80)
    print("🚀 LG 채용공고 크롤러 시작 (Refactored)")
    print("="*80 + "\n")

    crawler = LGCareerCrawler(max_concurrent=args.max_concurrent)

    # 1단계: API로 URL 수집
    basic_jobs = crawler.get_job_list_from_api()

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

    # 2단계: 비동기 병렬 크롤링
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