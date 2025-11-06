import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from playwright.sync_api import sync_playwright

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

try:
    from .new_prompt import SYSTEM_PROMPT
except Exception:
    try:
        from new_prompt import SYSTEM_PROMPT
    except Exception:
        SYSTEM_PROMPT = ""

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def get_openai_client() -> Optional[Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(10))
def summarize_with_llm(raw_text: str, model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    client = get_openai_client()
    if client is None:
        return []
    system_prompt = SYSTEM_PROMPT
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
    json_text_match = re.search(r"(\[.*\])", content, re.DOTALL)
    json_text = json_text_match.group(1) if json_text_match else content
    try:
        data = json.loads(json_text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def extract_job_detail_from_url(job_url: str, job_index: int) -> Dict[str, Any]:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            print(f"  [{job_index}] 상세 페이지 로딩: {job_url}")
            page.goto(job_url, timeout=30000)
            page.wait_for_timeout(2000)

            full_html = page.content()
            today = datetime.now().strftime('%Y-%m-%d')

            job_info = {
                "title": None,
                "company": "Woowahan",
                "location": None,
                "employment_type": None,
                "experience": None,
                "crawl_date": today,
                "posted_date": None,
                "expired_date": None,
                "description": None,
                "html": full_html,
                "url": job_url,
                "meta_data": "{}",
            }

            full_text = page.inner_text("body")
            
            # 상세 정보 추출을 위한 다양한 셀렉터 시도
            description_selectors = [
                "div.recruit-list",
                "div.recruit-detail",
                "div.content",
                "main",
                "article"
            ]
            
            description_found = False
            for sel in description_selectors:
                try:
                    element = page.query_selector(sel)
                    if element:
                        description_text = element.inner_text()
                        if description_text and len(description_text) > 100:
                            job_info["description"] = description_text
                            description_found = True
                            print(f"  [{job_index}] 설명 추출 성공 (셀렉터: {sel})")
                            break
                except Exception:
                    continue
            
            if not description_found:
                print(f"  [{job_index}] 특정 셀렉터에서 설명을 찾지 못해 body 전체 사용")
                job_info["description"] = full_text

            # LLM을 통한 정보 파싱
            try:
                parsed = summarize_with_llm(full_text)
                if parsed and len(parsed) > 0:
                    parsed_data = parsed[0]
                    for key in ["title", "company", "location", "employment_type", "experience",
                               "posted_date", "expired_date", "meta_data"]:
                        if key in parsed_data and parsed_data[key]:
                            job_info[key] = parsed_data[key]
                    job_info["html"] = full_html
                    job_info["url"] = job_url
                    print(f"  [{job_index}] LLM 파싱 완료")
            except Exception as e:
                print(f"  [{job_index}] LLM 파싱 실패 (description은 저장됨): {e}")

            print(f"  [{job_index}] ✓ 완료: {job_info.get('title', 'N/A')}")
            
            browser.close()
            return job_info

    except Exception as e:
        print(f"  [{job_index}] ✗ 상세 정보 추출 실패: {e}")
        return None

def run_scrape(
    keyword: str = "",
    out_dir: Path = Path("data"),
    fast: bool = False,
    debug: bool = False
):
    ensure_dir(out_dir)
    outputs = {
        "raw_html": out_dir / "woowahan_raw.html",
        "clean_txt": out_dir / "woowahan_clean.txt",
        "json": out_dir / "woowahan_jobs.json",
    }
    all_job_urls = []
    jobs_list = []

    with sync_playwright() as p:
        print("[1/10] 브라우저 실행 중...")
        browser = p.chromium.launch(headless=not debug)
        context = browser.new_context()
        if fast:
            try:
                context.set_default_timeout(5000)
            except Exception:
                pass
        page = context.new_page()

        base_url = "https://career.woowahan.com/?jobCodes=&employmentTypeCodes=&serviceSectionCodes=&careerPeriod=&keyword=&category=jobGroupCodes%3ABA005001#recruit-list"
        page_num = 1
        all_html_content = []

        while True:
            url = f"{base_url}?keyword={keyword}" if keyword else base_url
            print(f"\n[2/10] 페이지 {page_num} 접속: {url}")
            page.goto(url, timeout=60000)
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(30000)

            print(f"[3/10] 페이지 {page_num} 스크롤하여 전체 콘텐츠 로드 중...")
            for i in range(5):
                try:
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(10000)
                except Exception:
                    pass

            print(f"[4/10] 페이지 {page_num} 채용 공고 링크 찾는 중...")
            
            # 디버그 모드에서 HTML 구조 확인
            if debug:
                debug_html = out_dir / f"debug_page_{page_num}.html"
                debug_html.write_text(page.content(), encoding="utf-8")
                print(f"[DEBUG] HTML 저장: {debug_html}")
            
            try:
                # 스크린샷에서 확인된 구조에 맞춰 셀렉터 개선
                card_selectors = ["a[href*='/recruitment/']"]
    
                
                cards = None
                selector = None
                for sel in card_selectors:
                    cards = page.locator(sel)
                    count = cards.count()
                    if count > 0:
                        selector = sel
                        print(f"[5/10] ✓ 페이지 {page_num}: {count}개의 링크 발견 (셀렉터: {selector})")
                        break
                    else:
                        print(f"[5/10]   시도: {sel} → 0개")

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
                                # 상대 경로를 절대 경로로 변환
                                if href.startswith("/"):
                                    full_url = f"https://career.woowahan.com{href}"
                                else:
                                    full_url = href
                                
                                # 중복 체크 및 유효성 검사
                                if (full_url not in all_job_urls) and ("/recruitment/" in full_url):
                                    page_job_urls.append(full_url)
                                    all_job_urls.append(full_url)
                                    if debug:
                                        print(f"  [{i+1}] URL: {full_url}")
                        except Exception as e:
                            print(f"  ✗ URL 수집 실패 {i+1}: {e}")
                            continue

                    print(f"[6/10] ✓ 페이지 {page_num}에서 수집된 신규 URL: {len(page_job_urls)}개")
                    
                    if len(page_job_urls) == 0:
                        print(f"[6/10] 페이지 {page_num}에서 신규 공고가 없습니다. 크롤링 종료.")
                        break
                else:
                    print(f"[5/10] ✗ 페이지 {page_num}에서 공고 카드를 찾지 못했습니다.")
                    
                    # 디버그: 페이지에 있는 모든 a 태그 출력
                    if debug:
                        all_links = page.locator("a").all()
                        print(f"[DEBUG] 페이지의 총 링크 수: {len(all_links)}")
                        recruitment_links = [link.get_attribute("href") for link in all_links if link.get_attribute("href") and "recruitment" in link.get_attribute("href")]
                        print(f"[DEBUG] recruitment 포함 링크: {len(recruitment_links)}개")
                        if recruitment_links:
                            print(f"[DEBUG] 예시 링크들:")
                            for link in recruitment_links[:5]:
                                print(f"  - {link}")
                    break

            except Exception as e:
                print(f"[5/10] ✗ 페이지 {page_num} 카드 찾기 실패: {e}")
                import traceback
                traceback.print_exc()
                break

            html = page.content()
            all_html_content.append(html)
            
            # 단일 페이지만 크롤링 (필요시 페이지네이션 로직 추가)
            break

        print(f"\n[7/10] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[7/10] 전체 수집된 URL: {len(all_job_urls)}개")
        print(f"[7/10] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

        if all_job_urls:
            print(f"[8/10] 병렬로 {len(all_job_urls)}개 공고 상세 정보 크롤링 시작...\n")
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for idx, job_url in enumerate(all_job_urls, 1):
                    future = executor.submit(extract_job_detail_from_url, job_url, idx)
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        job_info = future.result()
                        if job_info:
                            jobs_list.append(job_info)
                    except Exception as e:
                        print(f"  ✗ 작업 실패: {e}")
        else:
            print("[8/10] ✗ 수집된 URL이 없어 상세 정보 크롤링을 건너뜁니다.")

        if all_html_content:
            outputs["raw_html"].write_text(all_html_content[-1], encoding="utf-8")
            print(f"\n[9/10] ✓ 원본 HTML 저장: {outputs['raw_html']}")

        context.close()
        browser.close()
        print("[9/10] ✓ 브라우저 종료")

    return outputs, jobs_list

def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Woowahan Careers 스크래핑")
    parser.add_argument("--keyword", default="", help="검색 키워드")
    parser.add_argument("--out-dir", default="data", help="출력 폴더 (기본: data)")
    parser.add_argument("--fast", action="store_true", help="빠른 모드: 대기시간 단축")
    parser.add_argument("--debug", action="store_true", help="디버그 모드: 브라우저 표시 및 상세 로그")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    print("\n" + "="*50)
    print("  우아한형제들 채용공고 스크래퍼")
    print("="*50 + "\n")
    print("[0/10] 작업 시작\n")

    paths, items = run_scrape(
        keyword=args.keyword,
        out_dir=out_dir,
        fast=args.fast,
        debug=args.debug
    )

    print("\n" + "="*50)
    if items:
        print(f"[10/10] ✓ 총 {len(items)}개의 공고 정보 수집 완료")
        paths["json"].write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[10/10] ✓ JSON 저장 완료: {paths['json']}")
        print(f"\n결과 파일: {paths['json']}")
    else:
        print("[10/10] ✗ 수집된 공고가 없습니다")
    
    print("="*50 + "\n")
    print("[완료] 모든 작업이 완료되었습니다.\n")

if __name__ == "__main__":
    main()