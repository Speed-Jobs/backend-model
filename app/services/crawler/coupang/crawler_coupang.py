import argparse
import json
import os
import re
import time
import random
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


def load_env() -> None:
    """Load environment variables from .env with fallbacks.

    Order: nearest discoverable .env from CWD → fproject/.env → backend-model/.env
    """
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
    path.mkdir(parents=True, exist_ok=True)

def get_openai_client() -> Optional[Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if OpenAI is None:
        return None
    return OpenAI(api_key=api_key)


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(10))
def summarize_with_llm(raw_text: str, model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
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
        "반드시 JSON 리스트 형식으로 반환하세요. 다른 설명 없이 JSON만 출력하세요.\n\n"
        "---\n"
        "# Example 1\n"
        "{\n"
        '    "title": "백엔드 개발자 경력 채용",\n'
        '    "company": "Coupang",\n'
        '    "location": "서울 송파구",\n'
        '    "employment_type": "정규직",\n'
        '    "experience": "경력 3년 이상",\n'
        '    "crawl_date": "2025-11-05",\n'
        '    "posted_date": "2025-10-20",\n'
        '    "expired_date": null,\n'
        '    "description": "담당업무\\n- Spring Boot 기반 API 개발\\n- 마이크로서비스 아키텍처 설계\\n\\n자격요건\\n- Java/Kotlin 개발 경험 3년 이상",\n'
        '    "meta_data": {\n'
        '        "job_category": "개발"\n'
        '    }\n'
        "}\n"
        "---\n"
        "# Example 2\n"
        "{\n"
        '    "title": "프론트엔드 개발자 신입 채용",\n'
        '    "company": "Coupang",\n'
        '    "location": "서울 판교",\n'
        '    "employment_type": "정규직",\n'
        '    "experience": "신입",\n'
        '    "crawl_date": "2025-11-05",\n'
        '    "posted_date": "2025-11-05",\n'
        '    "expired_date": null,\n'
        '    "description": "담당업무\\n- React 기반 웹 서비스 개발\\n- UI/UX 개선\\n\\n우대사항\\n- TypeScript 사용 경험\\n- 반응형 웹 개발 경험",\n'
        '    "meta_data": {\n'
        '        "job_category": "서비스/사업 개발/운영/영업"\n'
        '    }\n'
        "}\n"
        "---\n"
    )
    user_prompt = (
        f"오늘 날짜는 {datetime.now().strftime('%Y-%m-%d')}이고, 이 날짜를 crawl_date로 사용해. "
        f"공고들을 위 스키마에 맞춰 리스트로 정리해줘.\n\n" + raw_text
    )

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
    try:
        data = json.loads(json_text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def extract_job_detail_from_url(job_url: str, job_index: int, screenshot_dir: Path = None) -> Dict[str, Any]:
    """URL로 직접 접속하여 상세 정보 추출 (병렬 처리용 - 독립 브라우저)"""
    try:
        # 요청 간격 랜덤화 (robots.txt crawl-delay 준수: 1-2초)
        time.sleep(random.uniform(1.0, 2.5))

        # 각 스레드에서 독립적인 playwright와 브라우저 인스턴스 생성
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
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
            except Exception:
                pass

            page.wait_for_timeout(3000)

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
                "screenshots": {},  # 스크린샷 경로 저장용
            }

            # description 추출 - 우선순위에 따라 시도
            full_text = page.inner_text("body")
            description_text = None

            # 우선순위 셀렉터 목록 (가장 정확한 것부터)
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
                        if text and len(text) > 100:  # 최소 100글자 이상
                            description_text = text
                            print(f"  [{job_index}] description 추출 성공 ({name}): {len(text)} 글자")
                            break
                except Exception as e:
                    print(f"  [{job_index}] {name} 추출 실패: {e}")

            # description 설정
            if description_text:
                job_info["description"] = description_text
            else:
                # 모든 시도 실패 시 body 전체 사용
                job_info["description"] = full_text
                print(f"  [{job_index}] 모든 selector 실패, body 전체 사용")

            # 전체 페이지 스크린샷 저장
            if screenshot_dir:
                try:
                    # URL에서 job ID 추출하여 파일명에 사용
                    job_id_match = re.search(r'/jobs/(\d+)', job_url)
                    if not job_id_match:
                        job_id_match = re.search(r'jobId=([^&]+)', job_url)
                    job_id = job_id_match.group(1) if job_id_match else f"job_{job_index}"

                    screenshot_filename = f"coupang_job_{job_id}.png"
                    screenshot_path = screenshot_dir / screenshot_filename

                    # 전체 페이지 스크린샷
                    page.screenshot(path=str(screenshot_path), full_page=True)

                    job_info["screenshots"]["combined"] = str(screenshot_path)
                    print(f"  [{job_index}] 전체 페이지 스크린샷 저장: {screenshot_filename}")
                except Exception as e:
                    print(f"  [{job_index}] 스크린샷 저장 실패: {e}")

            # LLM으로 나머지 필드 파싱 시도
            try:
                parsed = summarize_with_llm(full_text)
                if parsed and len(parsed) > 0:
                    # description을 제외한 다른 필드만 업데이트
                    parsed_data = parsed[0]
                    for key in ["title", "company", "location", "employment_type", "experience",
                               "posted_date", "expired_date", "meta_data"]:
                        if key in parsed_data and parsed_data[key]:
                            job_info[key] = parsed_data[key]
            except Exception as e:
                print(f"  [{job_index}] LLM 파싱 실패 (description은 저장됨): {e}")

            # 브라우저는 with문에서 자동으로 닫힘
            print(f"  [{job_index}] 완료: {job_info.get('title', 'N/A')}")
            return job_info

    except Exception as e:
        print(f"  [{job_index}] 상세 정보 추출 실패: {e}")
        return None

def run_scrape(
    location: str = "South Korea",
    keyword: str = "",
    out_dir: Path = None,
    screenshot_dir: Path = None,
    fast: bool = False
) -> Dict[str, Path]:
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

    all_job_urls = []  # 모든 페이지에서 수집한 URL 저장
    jobs_list = []

    with sync_playwright() as p:
        print("[1/10] 브라우저 실행 중...")
        browser: Browser = p.chromium.launch(
            headless=False,  # Cloudflare 우회: headless 모드 비활성화
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-site-isolation-trials',
            ]
        )

        # Cloudflare 우회를 위한 컨텍스트 설정 (강화)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080},
            locale='ko-KR',
            timezone_id='Asia/Seoul',
            permissions=['geolocation'],
            geolocation={'latitude': 37.5665, 'longitude': 126.9780},  # Seoul
            color_scheme='light',
            extra_http_headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br, zstd',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Sec-Ch-Ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"',
                'Cache-Control': 'max-age=0',
            }
        )

        if fast:
            try:
                context.set_default_timeout(5000)
            except Exception:
                pass

        page: Page = context.new_page()

        # 자동화 감지 방지 (강화)
        page.add_init_script("""
            // webdriver 숨기기
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });

            // plugins 설정
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });

            // languages 설정
            Object.defineProperty(navigator, 'languages', {
                get: () => ['ko-KR', 'ko', 'en-US', 'en']
            });

            // chrome 객체 설정
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };

            // permissions 설정
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );

            // 추가 fingerprint 우회
            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: () => 8
            });

            Object.defineProperty(navigator, 'deviceMemory', {
                get: () => 8
            });

            Object.defineProperty(navigator, 'platform', {
                get: () => 'Win32'
            });

            // WebGL vendor 정보
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                if (parameter === 37445) {
                    return 'Intel Inc.';
                }
                if (parameter === 37446) {
                    return 'Intel Iris OpenGL Engine';
                }
                return getParameter(parameter);
            };
        """)

        # 여러 페이지 크롤링
        page_num = 1
        all_html_content = []
        max_pages = 5  # 최대 5페이지까지 크롤링

        while page_num <= max_pages:
            # URL 생성
            base_url = "https://www.coupang.jobs/kr/jobs/"
            params = f"?page={page_num}&location={location}&pagesize=20"
            if keyword:
                params += f"&query={keyword}"
            url = base_url + params + "#results"

            print(f"[2/10] 페이지 {page_num} 접속: {url}")

            # 랜덤 대기 (사람처럼 보이기)
            time.sleep(random.uniform(1.5, 3.5))

            page.goto(url, timeout=60000, wait_until="domcontentloaded")

            # Cloudflare Challenge 대기 (강화 - 최대 30초)
            print(f"[3/10] Cloudflare Challenge 대기 중...")
            try:
                # Cloudflare Challenge가 있으면 통과될 때까지 대기
                page.wait_for_load_state("networkidle", timeout=30000)
            except Exception:
                pass

            # 추가 대기 (JavaScript 렌더링) - 더 길게
            page.wait_for_timeout(5000)

            # 사람처럼 마우스 움직임 시뮬레이션
            try:
                page.mouse.move(random.randint(100, 500), random.randint(100, 500))
                time.sleep(random.uniform(0.5, 1.5))
            except Exception:
                pass

            # 스크롤을 여러 번 내려서 모든 콘텐츠 로드 (사람처럼 천천히)
            print(f"[4/10] 페이지 {page_num} 스크롤하여 전체 콘텐츠 로드 중...")
            for i in range(3):
                try:
                    # 점진적으로 스크롤 (사람처럼)
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

            # 디버그: 페이지에 어떤 요소들이 있는지 확인
            try:
                # 페이지 HTML 일부 저장하여 구조 파악
                html = page.content()
                debug_html_path = out_dir / f"coupang_debug_page{page_num}.html"
                debug_html_path.write_text(html, encoding="utf-8")
                print(f"[DEBUG] 페이지 HTML 저장: {debug_html_path}")

                # 몇 가지 일반적인 요소들 체크
                div_grid_count = page.locator("div.grid").count()
                div_job_listing_count = page.locator("div.job-listing").count()
                all_links_count = page.locator("a").count()
                print(f"[DEBUG] div.grid 요소: {div_grid_count}개")
                print(f"[DEBUG] div.job-listing 요소: {div_job_listing_count}개")
                print(f"[DEBUG] 전체 링크 수: {all_links_count}개")
            except Exception as e:
                print(f"[DEBUG] 디버그 정보 수집 실패: {e}")

            try:
                # 쿠팡 채용 페이지의 카드 셀렉터
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
                    print(f"[DEBUG] 셀렉터 '{sel}': {count}개 매치")
                    if count > 0:
                        selector = sel
                        print(f"[5/10] 페이지 {page_num}: {cards.count()}개의 카드를 찾았습니다 (셀렉터: {selector})")
                        break

                if cards and cards.count() > 0:
                    total_cards = cards.count()
                    print(f"[6/10] 페이지 {page_num}: {total_cards}개의 공고 URL 수집 중...")

                    # URL 수집
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
                                # 중복 체크
                                if full_url not in all_job_urls:
                                    page_job_urls.append(full_url)
                                    all_job_urls.append(full_url)
                        except Exception as e:
                            print(f"  URL 수집 실패 {i+1}: {e}")
                            continue

                    print(f"[6/10] 페이지 {page_num}에서 수집된 신규 URL: {len(page_job_urls)}개")

                    # 이 페이지에서 새로운 공고가 없으면 종료
                    if len(page_job_urls) == 0:
                        print(f"[6/10] 페이지 {page_num}에서 신규 공고가 없습니다. 크롤링 종료.")
                        break
                else:
                    # 카드를 찾지 못하면 크롤링 종료
                    print(f"[5/10] 페이지 {page_num}에서 공고 카드를 찾지 못했습니다. 크롤링 종료.")
                    break

            except Exception as e:
                print(f"[5/10] 페이지 {page_num} 카드 찾기 실패: {e}")
                break

            # HTML 저장 (마지막 페이지)
            html = page.content()
            all_html_content.append(html)

            page_num += 1

        print(f"[6/10] 전체 수집된 URL: {len(all_job_urls)}개")
        print(f"[7/10] 병렬로 {len(all_job_urls)}개 공고 상세 정보 크롤링 시작...")

        # 병렬로 각 URL의 상세 정보 크롤링 (ThreadPoolExecutor 사용)
        # Cloudflare 우회: 병렬 워커 수 감소 (20 → 3)
        if all_job_urls:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for idx, job_url in enumerate(all_job_urls, 1):
                    future = executor.submit(extract_job_detail_from_url, job_url, idx, screenshot_dir)
                    futures.append(future)

                # 결과 수집
                for future in concurrent.futures.as_completed(futures):
                    try:
                        job_info = future.result()
                        if job_info:
                            jobs_list.append(job_info)
                    except Exception as e:
                        print(f"  작업 실패: {e}")

        # 마지막 HTML 저장
        if all_html_content:
            outputs["raw_html"].write_text(all_html_content[-1], encoding="utf-8")
            print(f"[8/10] 원본 HTML 저장: {outputs['raw_html']}")

        context.close()
        browser.close()
        print("[8/10] 브라우저 종료")

    # HTML 저장만
    if all_html_content:
        print("[9/10] HTML 저장 완료")

    return outputs, jobs_list

def main() -> None:
    # .env 파일 경로: backend-model 디렉토리
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
    print("[0/10] 작업 시작")

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
    print("[10/10] 작업 완료")

if __name__ == "__main__":
    main()
