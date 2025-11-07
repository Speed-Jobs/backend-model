import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from playwright.sync_api import sync_playwright, Browser, Page, BrowserContext
import concurrent.futures
from threading import Lock

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

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
        # API 키가 없으면 빈 배열 반환 (수동 파싱 전용)
        return []

    system_prompt = (
        """
        당신은 채용 공고 웹페이지에서 구조화된 정보를 추출하는 전문가입니다.
        주어진 HTML 콘텐츠에서 다음 필드들을 정확하게 추출하여 JSON 형식으로 반환하세요.

        # 추출할 필드
        - title: 공고 제목
        - company: 회사 이름
        - location: 근무 위치
        - employment_type: 고용 형태 (정규직, 계약직, 파트타임 등)
        - experience: 경력 요구사항 (신입, 경력, 경력무관, 인턴 등)
        - crawl_date: 크롤링 날짜 (YYYY-MM-DD 형식)
        - posted_date: 공고 게시일 (YYYY-MM-DD 형식, 상시채용인 경우 크롤링 날짜와 동일)
        - expired_date: 공고 마감일 (YYYY-MM-DD 형식, 없으면 null)
        - description: 채용공고 전문 텍스트 (HTML 태그 제거)
        - meta_data: 위 필드 외 추가 정보를 담은 JSON 객체 (예: 직군, 연봉정보, 복리후생, 우대사항, 기술스택 등)

        ※ url은 별도로 입력받으므로 추출하지 않습니다.

        # 중요 지침
        1. 날짜는 반드시 YYYY-MM-DD 형식으로 통일
        2. 정보가 없는 경우 null 반환 (빈 문자열 X)
        3. description은 HTML 태그를 제거한 순수 텍스트
        4. meta_data는 의미있는 키 이름으로 구조화 (영문 snake_case 사용)
        5. 모든 텍스트는 공백 정리 및 정규화

        ---
        # Example 1
        {
            "title": "백엔드 개발자 (Python/Django)",
            "company": "(주)테크스타트업",
            "location": "서울 강남구",
            "employment_type": "정규직",
            "experience": "경력 3~5년",
            "crawl_date": "2025-11-05",
            "posted_date": "2025-10-28",
            "expired_date": "2025-11-30",
            "description": "주요업무\n- Django 기반 API 개발\n- 데이터베이스 설계 및 최적화\n\n자격요건\n- Python 3년 이상\n- Django, DRF 경험자",
            "meta_data": {
                "job_category": "IT/개발"
            }
        }

        ---
        # Example 2
        {
            "title": "프론트엔드 개발자 신입 채용",
            "company": "스타트업코리아",
            "location": "서울 판교",
            "employment_type": "정규직",
            "experience": "신입",
            "crawl_date": "2025-11-05",
            "posted_date": "2025-11-05",
            "expired_date": null,
            "description": "담당업무\n- React 기반 웹 서비스 개발\n- UI/UX 개선\n\n우대사항\n- TypeScript 사용 경험\n- 반응형 웹 개발 경험",
            "meta_data": {
                "job_category": "서비스/사업 개발/운영/영업"
            }
        }
        ---
        """
    )
    # 실제 사용
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
        # 각 스레드에서 독립적인 playwright와 브라우저 인스턴스 생성
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            print(f"  [{job_index}] 상세 페이지 로딩...")
            page.goto(job_url, timeout=30000)
            page.wait_for_timeout(1500)

            today = datetime.now().strftime('%Y-%m-%d')

            job_info = {
                "title": None,
                "company": "Kakao",
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

            # description 추출 - div.area_cont에서 직접 가져오기 (수동)
            full_text = page.inner_text("body")
            try:
                area_cont = page.query_selector("div.area_cont")
                if area_cont:
                    # area_cont의 텍스트만 추출 (네비게이션/푸터 제외된 순수 공고 내용)
                    description_text = area_cont.inner_text()
                    if description_text:
                        job_info["description"] = description_text
                else:
                    # 백업: body 전체 사용
                    job_info["description"] = full_text

                # 전체 페이지 스크린샷 저장
                if screenshot_dir:
                    try:
                        # URL에서 job ID 추출하여 파일명에 사용
                        job_id_match = re.search(r'/jobs/(\d+)', job_url)
                        job_id = job_id_match.group(1) if job_id_match else f"job_{job_index}"

                        screenshot_filename = f"kakao_job_{job_id}.png"
                        screenshot_path = screenshot_dir / screenshot_filename

                        # 전체 페이지 스크린샷
                        page.screenshot(path=str(screenshot_path), full_page=True)

                        job_info["screenshots"]["combined"] = str(screenshot_path)
                        print(f"  [{job_index}] 전체 페이지 스크린샷 저장: {screenshot_filename}")
                    except Exception as e:
                        print(f"  [{job_index}] 스크린샷 저장 실패: {e}")

            except Exception as e:
                print(f"  [{job_index}] area_cont 추출 실패, body 전체 사용: {e}")
                job_info["description"] = full_text

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
    part: str = "TECHNOLOGY",
    company: str = "KAKAO",
    keyword: str = "",
    out_dir: Path = Path("../../output"),
    screenshot_dir: Path = Path("../../img"),
    fast: bool = False
) -> Dict[str, Path]:
    ensure_dir(out_dir)
    ensure_dir(screenshot_dir)

    outputs = {
        "raw_html": out_dir / "kakao_raw.html",
        "clean_txt": out_dir / "kakao_clean.txt",
        "json": out_dir / "kakao_jobs.json",
        "screenshots": screenshot_dir,
    }

    all_job_urls = []  # 모든 페이지에서 수집한 URL 저장
    jobs_list = []

    with sync_playwright() as p:
        print("[1/10] 브라우저 실행 중...")
        browser: Browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        if fast:
            try:
                context.set_default_timeout(5000)
            except Exception:
                pass
        page: Page = context.new_page()

        # URL 생성
        base_url = "https://careers.kakao.com/jobs"

        # 여러 페이지 크롤링
        page_num = 1
        all_html_content = []

        while True:
            url = f"{base_url}?part={part}&company={company}&keyword={keyword}&page={page_num}"

            print(f"[2/10] 페이지 {page_num} 접속: {url}")
            page.goto(url, timeout=60000)
            page.wait_for_load_state("domcontentloaded")

            # 추가 대기 (JavaScript 렌더링)
            page.wait_for_timeout(2000)

            # 스크롤을 여러 번 내려서 모든 콘텐츠 로드
            print(f"[4/10] 페이지 {page_num} 스크롤하여 전체 콘텐츠 로드 중...")
            for _ in range(3):
                try:
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(800)
                except Exception:
                    pass

            # 채용 공고 카드 찾기
            print(f"[5/10] 페이지 {page_num} 채용 공고 카드 찾는 중...")
            try:
                # Kakao 채용 페이지의 카드 셀렉터 (여러 가능성 시도)
                card_selectors = [
                    "a[href*='/jobs/']",
                    "[class*='card']",
                    ".list_jobs > *",
                ]

                cards = None
                selector = None
                for sel in card_selectors:
                    cards = page.locator(sel)
                    if cards.count() > 0:
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
                                    full_url = f"https://careers.kakao.com{href}"
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

            # 다음 페이지로
            page_num += 1
            page.wait_for_timeout(1000)  # 페이지 간 간격

        print(f"[6/10] 전체 수집된 URL: {len(all_job_urls)}개")
        print(f"[7/10] 병렬로 {len(all_job_urls)}개 공고 상세 정보 크롤링 시작...")

        # 병렬로 각 URL의 상세 정보 크롤링 (ThreadPoolExecutor 사용)
        if all_job_urls:
            with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
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

    parser = argparse.ArgumentParser(description="Kakao Careers 스크래핑")
    parser.add_argument("--part", default="TECHNOLOGY", help="직군 (기본: TECHNOLOGY)")
    parser.add_argument("--company", default="KAKAO", help="회사 (기본: KAKAO)")
    parser.add_argument("--keyword", default="", help="검색 키워드")
    parser.add_argument("--out-dir", default="../../output", help="출력 폴더 (기본: ../../output)")
    parser.add_argument("--screenshot-dir", default="../../img", help="스크린샷 폴더 (기본: ../../img)")
    parser.add_argument("--fast", action="store_true", help="빠른 모드: 대기시간 단축")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    screenshot_dir = Path(args.screenshot_dir)
    print("[0/10] 작업 시작")

    paths, items = run_scrape(
        part=args.part,
        company=args.company,
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