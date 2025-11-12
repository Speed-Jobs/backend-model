import argparse
import json
import os
import re
import time
import random
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
try:
    from dotenv import find_dotenv  # type: ignore
except Exception:
    find_dotenv = None  # type: ignore
from tenacity import retry, stop_after_attempt, wait_exponential
from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext
try:
    from app.services import resolve_dir, get_output_dir, get_img_dir
except ModuleNotFoundError:
    import sys
    _p = Path(__file__).resolve().parents[4]
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
    from app.services import resolve_dir, get_output_dir, get_img_dir
import concurrent.futures
from threading import Lock

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


def is_wsl() -> bool:
    """WSL 환경인지 확인"""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False


# WSL 환경 감지
IS_WSL = is_wsl()
HEADLESS_MODE = True if IS_WSL else os.getenv('HEADLESS', 'true').lower() == 'true'

if IS_WSL:
    print("[INFO] WSL 환경 감지 - headless 모드로 실행됩니다")


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

def ensure_dir(path: Path) -> None:
    """디렉토리 생성"""
    path.mkdir(parents=True, exist_ok=True)

def get_openai_client() -> Optional[Any]:
    """OpenAI 클라이언트 생성"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def summarize_with_llm(raw_text: str, model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    """LLM을 사용한 공고 정보 추출"""
    client = get_openai_client()
    if client is None:
        return []
    
    system_prompt = (
        "당신은 채용 공고 웹페이지에서 구조화된 정보를 추출하는 전문가입니다.\n"
        "주어진 HTML 콘텐츠에서 다음 필드들을 정확하게 추출하여 JSON 형식으로 반환하세요.\n\n"
        "# 추출할 필드\n"
        "- title: 공고 제목\n"
        "- company: 회사명 (기본값: 'Coupang')\n"
        "- location: 근무 위치\n"
        "- employment_type: 고용 형태 (정규직, 계약직, 파트타임 등)\n"
        "- experience: 경력 요구사항 (신입, 경력, 경력무관, 인턴 등)\n"
        "- crawl_date: 크롤링 날짜 (YYYY-MM-DD 형식)\n"
        "- posted_date: 공고 게시일 (YYYY-MM-DD 형식, 없으면 null)\n"
        "- expired_date: 공고 마감일 (YYYY-MM-DD 형식, 없으면 null)\n"
        "- description: 공고 상세 내용 (직무 설명, 자격 요건, 우대 사항 등)\n"
        "- meta_data: 기타 정보 (딕셔너리 형태, 예: {\"job_category\": \"개발\"})\n\n"
        "# 출력 형식\n"
        "반드시 JSON 리스트 형식으로 반환하세요. 다른 설명 없이 JSON만 출력하세요.\n"
    )
    
    user_prompt = (
        f"오늘 날짜는 {datetime.now().strftime('%Y-%m-%d')}이고, 이 날짜를 crawl_date로 사용해. "
        f"공고들을 위 스키마에 맞춰 리스트로 정리해줘.\n\n" + raw_text
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=4000,
        )
        content = response.choices[0].message.content if response and response.choices else "[]"

        # JSON만 남도록 트리밍
        json_text_match = re.search(r"(\[.*\])", content, re.DOTALL)
        json_text = json_text_match.group(1) if json_text_match else content
        
        data = json.loads(json_text)
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"  [LLM] 파싱 실패: {e}")
    
    return []

def extract_job_detail_from_url(job_url: str, job_index: int, screenshot_dir: Path = None) -> Optional[Dict[str, Any]]:
    """URL로 직접 접속하여 상세 정보 추출 (병렬 처리용 - 독립 브라우저)"""
    browser = None
    context = None
    page = None
    playwright = None
    
    try:
        # 요청 간격 랜덤화 (robots.txt crawl-delay 준수: 1-2초)
        time.sleep(random.uniform(1.0, 2.5))

        # 각 스레드에서 독립적인 playwright와 브라우저 인스턴스 생성
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(
            headless=HEADLESS_MODE,  # WSL 환경 자동 감지
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
            ]
        )
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='ko-KR',
            timezone_id='Asia/Seoul',
        )
        page = context.new_page()

        # 자동화 감지 방지 스크립트 추가
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };
        """)

        print(f"  [{job_index}] 상세 페이지 로딩...")
        page.goto(job_url, timeout=60000, wait_until="domcontentloaded")

        # Cloudflare Challenge 대기
        try:
            page.wait_for_load_state("networkidle", timeout=20000)
            print(f"  [{job_index}] 네트워크 안정화 완료")
        except Exception as e:
            print(f"  [{job_index}] 네트워크 안정화 타임아웃 (무시): {e}")

        page.wait_for_timeout(3000)
        print(f"  [{job_index}] 페이지 로딩 완료")

        today = datetime.now().strftime('%Y-%m-%d')

        job_info = {
            "title": None,
            "company": "Coupang",
            "location": None,
            "employment_type": None,
            "experience": None,
            "crawl_date": today,
            "posted_date": None,
            "expired_date": None,
            "description": None,
            "url": job_url,
            "meta_data": "{}",
            "screenshots": {},
        }

        # description 추출 - 우선순위에 따라 시도
        print(f"  [{job_index}] description 추출 중...")
        full_text = page.inner_text("body")
        description_text = None

        # 우선순위 셀렉터 목록
        selectors = [
            ("article.cms-content", "article.cms-content"),
            ("article", "article"),
            ("div.main-col article", "div.main-col > article"),
            ("main", "main"),
            ("div[class*='detail']", "div containing 'detail'"),
        ]

        for selector, name in selectors:
            try:
                elem = page.query_selector(selector)
                if elem:
                    text = elem.inner_text()
                    if text and len(text) > 100:
                        description_text = text
                        print(f"  [{job_index}] description 추출 성공 ({name}): {len(text)} 글자")
                        break
            except Exception as e:
                print(f"  [{job_index}] {name} 추출 실패: {e}")

        # description 설정
        if description_text:
            job_info["description"] = description_text
        else:
            job_info["description"] = full_text
            print(f"  [{job_index}] 모든 selector 실패, body 전체 사용")

        # 전체 페이지 스크린샷 저장
        if screenshot_dir:
            try:
                print(f"  [{job_index}] 스크린샷 저장 중...")
                job_id_match = re.search(r'/jobs/(\d+)', job_url)
                if not job_id_match:
                    job_id_match = re.search(r'jobId=([^&]+)', job_url)
                job_id = job_id_match.group(1) if job_id_match else f"job_{job_index}"

                screenshot_filename = f"coupang_job_{job_id}.png"
                screenshot_path = screenshot_dir / screenshot_filename

                page.screenshot(path=str(screenshot_path), full_page=True)
                job_info["screenshots"]["combined"] = str(screenshot_path)
                print(f"  [{job_index}] 스크린샷 저장 완료: {screenshot_filename}")
            except Exception as e:
                print(f"  [{job_index}] 스크린샷 저장 실패: {e}")

        # LLM으로 나머지 필드 파싱 시도
        try:
            print(f"  [{job_index}] LLM 파싱 중...")
            parsed = summarize_with_llm(full_text)
            if parsed and len(parsed) > 0:
                parsed_data = parsed[0]
                for key in ["title", "company", "location", "employment_type", "experience",
                           "posted_date", "expired_date", "meta_data"]:
                    if key in parsed_data and parsed_data[key]:
                        job_info[key] = parsed_data[key]
                print(f"  [{job_index}] LLM 파싱 완료")
        except Exception as e:
            print(f"  [{job_index}] LLM 파싱 실패 (description은 저장됨): {e}")

        print(f"  [{job_index}] 완료: {job_info.get('title', 'N/A')}")
        return job_info

    except Exception as e:
        print(f"  [{job_index}] 상세 정보 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # 리소스 완전 정리 (역순)
        try:
            if page:
                page.close()
        except Exception:
            pass
        
        try:
            if context:
                context.close()
        except Exception:
            pass
        
        try:
            if browser:
                browser.close()
        except Exception:
            pass
        
        try:
            if playwright:
                playwright.stop()
        except Exception:
            pass

def run_scrape(
    location: str = "South Korea",
    keyword: str = "",
    out_dir: Path = None,
    screenshot_dir: Path = None,
    fast: bool = False
) -> tuple[Dict[str, Path], List[Dict[str, Any]]]:
    """메인 크롤링 함수"""
    out_dir = resolve_dir(out_dir, get_output_dir())
    screenshot_dir = resolve_dir(screenshot_dir, get_img_dir())
    ensure_dir(out_dir)
    ensure_dir(screenshot_dir)

    outputs = {
        "raw_html": out_dir / "coupang_raw.html",
        "clean_txt": out_dir / "coupang_clean.txt",
        "json": out_dir / "coupang_jobs.json",
        "screenshots": screenshot_dir,
    }

    all_job_urls = []
    jobs_list = []
    
    playwright = None
    browser = None
    context = None
    page = None

    try:
        print("[1/10] 브라우저 실행 중...")
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(
            headless=HEADLESS_MODE,  # WSL 환경 자동 감지
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-site-isolation-trials',
            ]
        )

        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='ko-KR',
            timezone_id='Asia/Seoul',
            permissions=['geolocation'],
            geolocation={'latitude': 37.5665, 'longitude': 126.9780},
            color_scheme='light',
            extra_http_headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br, zstd',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            }
        )

        if fast:
            try:
                context.set_default_timeout(5000)
            except Exception:
                pass

        page = context.new_page()

        # 자동화 감지 방지
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            Object.defineProperty(navigator, 'languages', {
                get: () => ['ko-KR', 'ko', 'en-US', 'en']
            });
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };
        """)

        # 여러 페이지 크롤링
        page_num = 1
        all_html_content = []
        max_pages = 5

        while page_num <= max_pages:
            base_url = "https://www.coupang.jobs/kr/jobs/"
            params = f"?page={page_num}&location={location}&pagesize=20"
            if keyword:
                params += f"&query={keyword}"
            url = base_url + params + "#results"

            print(f"[2/10] 페이지 {page_num} 접속: {url}")
            time.sleep(random.uniform(1.5, 3.5))

            page.goto(url, timeout=60000, wait_until="domcontentloaded")

            print(f"[3/10] Cloudflare Challenge 대기 중...")
            try:
                page.wait_for_load_state("networkidle", timeout=30000)
            except Exception:
                pass

            page.wait_for_timeout(5000)

            # 사람처럼 마우스 움직임
            try:
                page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                time.sleep(random.uniform(0.5, 1.5))
            except Exception:
                pass

            # 스크롤
            print(f"[4/10] 페이지 {page_num} 스크롤하여 전체 콘텐츠 로드 중...")
            for i in range(3):
                try:
                    scroll_height = page.evaluate("document.body.scrollHeight")
                    current_position = 0
                    step = scroll_height // 5

                    while current_position < scroll_height:
                        current_position += step
                        page.evaluate(f"window.scrollTo(0, {current_position})")
                        time.sleep(random.uniform(0.3, 0.7))

                    page.wait_for_timeout(1500)
                except Exception:
                    pass

            # 채용 공고 카드 찾기
            print(f"[5/10] 페이지 {page_num} 채용 공고 카드 찾는 중...")

            try:
                card_selectors = [
                    "div.grid.job-listing a[href*='/jobs/']",
                    "div.job-listing a[href*='/jobs/']",
                    "a.job-tile",
                    "a[href*='/kr/jobs/']",
                    "div.job-tile > a",
                    ".job-results a[href*='/jobs/']",
                ]

                cards = None
                selector = None
                for sel in card_selectors:
                    cards = page.locator(sel)
                    count = cards.count()
                    if count > 0:
                        selector = sel
                        print(f"[5/10] 페이지 {page_num}: {count}개의 카드를 찾았습니다 (셀렉터: {selector})")
                        break

                if cards and cards.count() > 0:
                    total_cards = cards.count()
                    print(f"[6/10] 페이지 {page_num}: {total_cards}개의 공고 URL 수집 중...")

                    page_job_urls = []
                    for i in range(total_cards):
                        try:
                            cards = page.locator(selector)
                            card = cards.nth(i)
                            href = card.get_attribute("href")
                            if href:
                                if href.startswith("/"):
                                    full_url = f"https://www.coupang.jobs{href}"
                                elif not href.startswith("http"):
                                    full_url = f"https://www.coupang.jobs/kr/{href}"
                                else:
                                    full_url = href
                                if full_url not in all_job_urls:
                                    page_job_urls.append(full_url)
                                    all_job_urls.append(full_url)
                        except Exception as e:
                            print(f"  URL 수집 실패 {i+1}: {e}")
                            continue

                    print(f"[6/10] 페이지 {page_num}에서 수집된 신규 URL: {len(page_job_urls)}개")

                    if len(page_job_urls) == 0:
                        print(f"[6/10] 페이지 {page_num}에서 신규 공고가 없습니다. 크롤링 종료.")
                        break
                else:
                    print(f"[5/10] 페이지 {page_num}에서 공고 카드를 찾지 못했습니다. 크롤링 종료.")
                    break

            except Exception as e:
                print(f"[5/10] 페이지 {page_num} 카드 찾기 실패: {e}")
                break

            html = page.content()
            all_html_content.append(html)

            page_num += 1

        print(f"[6/10] 전체 수집된 URL: {len(all_job_urls)}개")
        print(f"[7/10] 병렬로 {len(all_job_urls)}개 공고 상세 정보 크롤링 시작...")

        # 병렬 크롤링 (워커 수 감소로 Cloudflare 우회)
        if all_job_urls:
            executor = None
            try:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
                futures = {}
                for idx, job_url in enumerate(all_job_urls, 1):
                    future = executor.submit(extract_job_detail_from_url, job_url, idx, screenshot_dir)
                    futures[future] = (idx, job_url)

                # 각 작업 완료 대기 (타임아웃 포함)
                completed = 0
                for future in concurrent.futures.as_completed(futures, timeout=600):  # 10분 타임아웃
                    idx, url = futures[future]
                    try:
                        job_info = future.result(timeout=120)  # 각 작업 2분 타임아웃
                        if job_info:
                            jobs_list.append(job_info)
                            completed += 1
                            print(f"[진행률] {completed}/{len(all_job_urls)} 완료")
                    except concurrent.futures.TimeoutError:
                        print(f"  [{idx}] 타임아웃: {url}")
                    except Exception as e:
                        print(f"  [{idx}] 작업 실패: {e}")
                        import traceback
                        traceback.print_exc()
            except concurrent.futures.TimeoutError:
                print("[WARNING] 전체 병렬 처리 타임아웃 발생")
            finally:
                if executor:
                    print("[7/10] ThreadPoolExecutor 종료 중...")
                    executor.shutdown(wait=True, cancel_futures=True)
                    print("[7/10] ThreadPoolExecutor 종료 완료")

        # HTML 저장
        if all_html_content:
            outputs["raw_html"].write_text(all_html_content[-1], encoding="utf-8")
            print(f"[8/10] 원본 HTML 저장: {outputs['raw_html']}")

    except Exception as e:
        print(f"[ERROR] 크롤링 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 리소스 완전 정리 (역순)
        print("[8/10] 브라우저 리소스 정리 중...")
        try:
            if page:
                page.close()
        except Exception:
            pass
        
        try:
            if context:
                context.close()
        except Exception:
            pass
        
        try:
            if browser:
                browser.close()
        except Exception:
            pass
        
        try:
            if playwright:
                playwright.stop()
        except Exception:
            pass
        
        print("[8/10] 브라우저 종료 완료")

    print("[9/10] 처리 완료")
    return outputs, jobs_list

def main() -> None:
    """메인 함수"""
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser(description="Coupang Careers 스크래핑")
    parser.add_argument("--location", default="South Korea", help="근무 위치 (기본: South Korea)")
    parser.add_argument("--keyword", default="", help="검색 키워드")
    parser.add_argument("--out-dir", default="../../output", help="출력 폴더 (기본: ../../output)")
    parser.add_argument("--screenshot-dir", default="../../img", help="스크린샷 폴더 (기본: ../../img)")
    parser.add_argument("--fast", action="store_true", help="빠른 모드: 대기시간 단축")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    screenshot_dir = Path(args.screenshot_dir)
    print("[0/10] Coupang 크롤링 작업 시작")

    paths, items = run_scrape(
        location=args.location,
        keyword=args.keyword,
        out_dir=out_dir,
        screenshot_dir=screenshot_dir,
        fast=args.fast
    )

    if items:
        print(f"[9/10] 총 {len(items)}개의 공고 정보 수집 완료")
    else:
        print("[9/10] 수집된 공고가 없습니다")

    # 저장
    paths["json"].write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[9/10] JSON 저장 완료: {paths['json']}")
    print(str(paths["json"]))
    print("[10/10] Coupang 크롤링 작업 완료")

if __name__ == "__main__":
    main()