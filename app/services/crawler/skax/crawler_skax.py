"""
SK주식회사(AX) 채용공고 크롤러

특징:
1. 검색해도 URL이 변경되지 않는 SPA 구조
2. div.company 텍스트로 "SK주식회사(AX)" 필터링
3. ThreadPoolExecutor로 병렬 크롤링

구조:
- div.announcement-list-container
  └─ div.list-wrapper
      └─ .RecruitList
          └─ div.list-item (각 채용공고)
              ├─ div.company (회사명 필터링)
              └─ a.list-link.url (상세 페이지 링크)
"""

import argparse
import json
import os
import re
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

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

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


def ensure_dir(path: Path) -> None:
    """디렉토리 생성"""
    path.mkdir(parents=True, exist_ok=True)


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def summarize_with_llm(raw_text: str, model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    """LLM을 사용한 공고 정보 추출"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return []

    system_prompt = """
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

# 중요 지침
1. 날짜는 반드시 YYYY-MM-DD 형식으로 통일
2. 정보가 없는 경우 null 반환 (빈 문자열 X)
3. description은 HTML 태그를 제거한 순수 텍스트
4. **meta_data는 반드시 한국어 키로만 구성** (예: "직무분야", "우대사항", "복리후생")
5. **meta_data에는 위의 기본 필드(title, company, location, employment_type, experience, crawl_date, posted_date, expired_date, description)와 중복되는 정보를 절대 포함하지 말 것**
6. meta_data에는 오직 추가적인 보조 정보만 포함 (예: 자격요건, 우대사항, 복리후생, 기술스택, 담당업무, 학력요건, 전형절차 등)
7. meta_data에는 기술스택은 넣지 않는다. 즉 softskill은 넣지않는다 예시) python, django, aws, docker등을 넣지 않는다.

# 출력 형식
반드시 JSON 리스트 형식으로 반환하세요.
---
# Example
{
    "title": "백엔드 개발자 (Python/Django)",
    "company": "(주)테크스타트업",
    "location": "서울 강남구",
    "employment_type": "정규직",
    "experience": "경력 3~5년",
    "crawl_date": "2025-11-05",
    "posted_date": "2025-10-28",
    "expired_date": "2025-11-30",
    "description": "주요업무...",
    "meta_data": {
        "직무분야": "IT/개발",
        "우대사항": ["AWS 경험", "Docker/K8s 사용 경험"],
        "복리후생": ["건강검진", "자기계발비 지원"],
        "학력요건": "학사 이상",
        "전형절차": "서류전형 > 1차 면접 > 2차 면접 > 최종합격",
    }
}
---
"""
    
    user_prompt = (
        f"오늘 날짜는 {datetime.now().strftime('%Y-%m-%d')}이고, 이 날짜를 crawl_date로 사용해. "
        f"공고들을 위 스키마에 맞춰 리스트로 정리해줘.\n\n" + raw_text
    )

    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=8000,
        )
        content = response.choices[0].message.content if response and response.choices else "[]"

        json_text_match = re.search(r"(\[.*\])", content, re.DOTALL)
        json_text = json_text_match.group(1) if json_text_match else content
        
        data = json.loads(json_text)
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"  [LLM] 파싱 실패: {e}")
    finally:
        client.close()
    
    return []


def extract_job_detail_from_url(job_url: str, job_index: int, screenshot_dir: Path = None) -> Optional[Dict[str, Any]]:
    """URL로 직접 접속하여 상세 정보 추출 (병렬 처리용 - 독립 브라우저)"""
    browser = None
    context = None
    page = None
    playwright = None
    
    try:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        
        # 봇 감지 우회를 위한 context 설정
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="ko-KR",
            timezone_id="Asia/Seoul",
        )
        page = context.new_page()

        print(f"  [{job_index}] 상세 페이지 로딩...")
        page.goto(job_url, timeout=30000)
        page.wait_for_timeout(2000)

        today = datetime.now().strftime('%Y-%m-%d')

        job_info = {
            "title": None,
            "company": "SK주식회사(AX)",
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

        # description 추출 - 여러 셀렉터 시도
        full_text = page.inner_text("body")
        description_found = False
        
        try:
            # 여러 가능한 셀렉터 시도
            selectors = [
                "div.section-large-inner",
                "div.detail-section",
                "div.job-description",
                "div.content-area",
                "main#content",
                "div.announcement-detail-content",
                "section.detail-content",
                "div[class*='detail']",
                "div[class*='content']",
            ]
            
            for selector in selectors:
                try:
                    section = page.query_selector(selector)
                    if section:
                        description_text = section.inner_text()
                        # 헤더 텍스트만 있는지 확인 (최소 100자 이상이어야 함)
                        if description_text and len(description_text) > 100:
                            job_info["description"] = description_text
                            description_found = True
                            print(f"  [{job_index}] description 추출 성공 (셀렉터: {selector}, {len(description_text)}자)")
                            break
                except Exception:
                    continue
            
            # 모든 셀렉터 실패 시 body에서 메뉴 제거
            if not description_found:
                # 헤더/메뉴 부분 제거
                try:
                    # 메뉴, 헤더 등 제거
                    page.evaluate("""
                        () => {
                            const elementsToRemove = [
                                'header', 'nav', '.header', '.nav', 
                                '.gnb', '.menu', '.sidebar'
                            ];
                            elementsToRemove.forEach(sel => {
                                document.querySelectorAll(sel).forEach(el => el.remove());
                            });
                        }
                    """)
                    page.wait_for_timeout(500)
                    full_text = page.inner_text("body")
                    if len(full_text) > 100:
                        job_info["description"] = full_text
                        print(f"  [{job_index}] description 추출 (body 정리 후, {len(full_text)}자)")
                    else:
                        job_info["description"] = full_text
                        print(f"  [{job_index}] WARNING: description이 너무 짧음 ({len(full_text)}자)")
                except Exception:
                    job_info["description"] = full_text

            # 스크린샷 S3 업로드
            if screenshot_dir:
                try:
                    job_id_match = re.search(r'/Detail/(\w+)', job_url)
                    job_id = job_id_match.group(1) if job_id_match else f"job_{job_index}"

                    screenshot_filename = f"skax_job_{job_id}.png"

                    # S3에 직접 업로드
                    screenshot_bytes = page.screenshot(full_page=True)
                    s3_url = upload_screenshot_to_s3(screenshot_bytes, screenshot_filename)

                    if s3_url:
                        job_info["screenshots"]["combined"] = s3_url
                        print(f"  [{job_index}] S3 업로드 성공: {s3_url}")
                    else:
                        print(f"  [{job_index}] S3 업로드 실패")
                except Exception as e:
                    print(f"  [{job_index}] 스크린샷 처리 실패: {e}")

        except Exception as e:
            print(f"  [{job_index}] description 추출 중 오류: {e}")
            job_info["description"] = full_text if full_text else "내용을 가져올 수 없습니다."

        # LLM 파싱
        try:
            parsed = summarize_with_llm(full_text)
            if parsed and len(parsed) > 0:
                parsed_data = parsed[0]
                for key in ["title", "company", "location", "employment_type", "experience",
                           "posted_date", "expired_date", "meta_data"]:
                    if key in parsed_data and parsed_data[key]:
                        job_info[key] = parsed_data[key]
        except Exception as e:
            print(f"  [{job_index}] LLM 파싱 실패: {e}")

        print(f"  [{job_index}] 완료: {job_info.get('title', 'N/A')}")
        return job_info

    except Exception as e:
        print(f"  [{job_index}] 상세 정보 추출 실패: {e}")
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
        "raw_html": out_dir / "skax_raw.html",
        "json": out_dir / "skax_jobs.json",
        "screenshots": screenshot_dir,
    }

    skax_job_urls = []
    jobs_list = []
    
    playwright = None
    browser = None
    context = None
    page = None

    try:
        print("[1/10] 브라우저 실행 중...")
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        
        # 봇 감지 우회를 위한 context 설정
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="ko-KR",
            timezone_id="Asia/Seoul",
        )
        
        if fast:
            try:
                context.set_default_timeout(5000)
            except Exception:
                pass
        
        page = context.new_page()

        # SK Careers 채용 페이지
        base_url = "https://www.skcareers.com/Recruit"

        # API 요청 가로채기 설정
        api_responses = []
        def handle_response(response):
            if "/Recruit/GetRecruitList" in response.url:
                try:
                    data = response.json()
                    api_responses.append(data)
                    print(f"[DEBUG] API 응답 캡처: {len(data.get('list', []))}개 공고")
                except Exception:
                    pass
        
        page.on("response", handle_response)
        
        print(f"[2/10] 페이지 접속: {base_url}")
        page.goto(base_url, timeout=60000)
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(5000)  # JavaScript 실행 대기

        # 페이지 로딩 대기 - 네트워크 안정화까지 기다림
        print("[3/10] 채용공고 리스트 로딩 대기...")
        try:
            page.wait_for_load_state("networkidle", timeout=20000)
            page.wait_for_timeout(3000)  # AJAX 요청 대기
            
            # 실제 채용공고(.list-item)가 나타날 때까지 대기
            print("[3/10] 채용공고 아이템 로딩 대기...")
            page.wait_for_selector("div.list-item", timeout=15000)
            print("[3/10] 채용공고 감지됨!")
        except Exception as e:
            print(f"[3/10] 채용공고 로딩 대기 실패: {e}")
            print(f"[3/10] 현재 페이지 URL: {page.url}")

        # 스크롤로 모든 콘텐츠 로드
        print("[4/10] 페이지 스크롤하여 전체 콘텐츠 로드 중...")
        for _ in range(3):
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1000)
            except Exception:
                pass
        
        # "더보기" 버튼 클릭하여 모든 공고 로드
        print("[4/10] '더보기' 버튼 클릭하여 모든 공고 로드 중...")
        max_clicks = 20  # 최대 20번 클릭
        click_count = 0
        
        while click_count < max_clicks:
            try:
                # 더보기 버튼 찾기 (여러 가능한 셀렉터)
                more_button_selectors = [
                    "button:has-text('더보기')",
                    "button:has-text('More')",
                    "a:has-text('더보기')",
                    ".btn-more",
                    "#btnMore",
                    "button.more",
                ]
                
                button_found = False
                for selector in more_button_selectors:
                    try:
                        more_button = page.locator(selector)
                        if more_button.count() > 0 and more_button.is_visible():
                            print(f"  [더보기 {click_count + 1}] 버튼 클릭...")
                            more_button.click()
                            page.wait_for_timeout(2000)  # AJAX 로딩 대기
                            click_count += 1
                            button_found = True
                            break
                    except Exception:
                        continue
                
                if not button_found:
                    print(f"[4/10] 더보기 버튼을 찾을 수 없음. 총 {click_count}번 클릭함")
                    break
                    
            except Exception as e:
                print(f"[4/10] 더보기 클릭 중 오류: {e}")
                break
        
        # 최종 대기
        page.wait_for_timeout(2000)

        # HTML 저장
        html_content = page.content()
        
        # API 응답 저장 (디버깅용)
        if api_responses:
            api_debug_path = out_dir / "skax_api_responses.json"
            api_debug_path.write_text(json.dumps(api_responses, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[DEBUG] API 응답 저장: {api_debug_path}")

        # 채용 공고 필터링 및 URL 수집
        print("[5/10] SK주식회사(AX) 채용 공고 필터링 중...")

        try:
            # 더 정확한 셀렉터 사용
            all_items = page.locator("div#RecruitList div.list-item")
            total_items = all_items.count()
            print(f"[5/10] 전체 채용공고: {total_items}개")
            
            if total_items == 0:
                print("[5/10] 채용공고를 찾을 수 없습니다. 다른 셀렉터 시도...")
                all_items = page.locator(".list-item")
                total_items = all_items.count()
                print(f"[5/10] 대체 셀렉터로 찾은 채용공고: {total_items}개")

            for i in range(total_items):
                try:
                    item = all_items.nth(i)
                    
                    # div.company 텍스트 확인
                    try:
                        company_div = item.locator("div.company")
                        company_name = company_div.inner_text().strip()
                    except Exception:
                        # company div를 찾을 수 없으면 다른 셀렉터 시도
                        try:
                            company_name = item.locator(".company").inner_text().strip()
                        except Exception:
                            if i < 3:  # 처음 3개만 출력
                                print(f"  [경고] item {i+1}: company 정보를 찾을 수 없음")
                            continue
                    
                    # 디버깅: 처음 5개 회사명 출력
                    if i < 5:
                        print(f"  [DEBUG] item {i+1}: {company_name}")
                    
                    # "SK주식회사(AX)"인 경우에만 처리
                    if company_name == "SK주식회사(AX)":
                        # a.list-link.url에서 href 추출
                        try:
                            link = item.locator("a.list-link.url")
                            href = link.get_attribute("href")
                        except Exception:
                            # 다른 링크 셀렉터 시도
                            try:
                                link = item.locator("a[href*='/Recruit/Detail/']")
                                href = link.get_attribute("href")
                            except Exception as e:
                                print(f"  [경고] item {i+1}: 링크를 찾을 수 없음 - {e}")
                                continue
                        
                        if href and href not in skax_job_urls:
                            skax_job_urls.append(href)
                            print(f"  [{len(skax_job_urls)}] ✓ SKAX 공고: {href}")
                except Exception as e:
                    if i < 3:  # 처음 3개만 출력
                        print(f"  [경고] item {i+1} 처리 중 오류: {e}")
                    continue

            print(f"[6/10] SK주식회사(AX) 채용 공고: {len(skax_job_urls)}개")

        except Exception as e:
            print(f"[5/10] 링크 추출 실패: {e}")

        if not skax_job_urls:
            print("[!] SK주식회사(AX) 채용 공고를 찾을 수 없습니다.")
            outputs["raw_html"].write_text(html_content, encoding="utf-8")
            print(f"[DEBUG] 원본 HTML 저장: {outputs['raw_html']}")
            return outputs, []
        
        # HTML 저장
        outputs["raw_html"].write_text(html_content, encoding="utf-8")
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

    # 병렬 크롤링
    if skax_job_urls:
        print(f"[7/10] 병렬로 {len(skax_job_urls)}개 공고 상세 크롤링 시작...")
        executor = None
        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
            futures = {
                executor.submit(extract_job_detail_from_url, url, idx, screenshot_dir): idx
                for idx, url in enumerate(skax_job_urls, start=1)
            }

            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        jobs_list.append(result)
                except Exception as e:
                    print(f"  [{idx}] 처리 중 예외 발생: {e}")
        finally:
            if executor:
                executor.shutdown(wait=True)

    print("[9/10] 처리 완료")
    return outputs, jobs_list


def main() -> None:
    """메인 함수"""
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser(description="SK주식회사(AX) 채용공고 스크래핑")
    parser.add_argument("--out-dir", default="../../output", help="출력 폴더")
    parser.add_argument("--screenshot-dir", default="../../img", help="스크린샷 폴더")
    parser.add_argument("--fast", action="store_true", help="빠른 모드")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    screenshot_dir = Path(args.screenshot_dir)
    print("[0/10] SK주식회사(AX) 크롤링 작업 시작")

    paths, items = run_scrape(
        out_dir=out_dir,
        screenshot_dir=screenshot_dir,
        fast=args.fast
    )

    if items:
        print(f"[9/10] 총 {len(items)}개의 공고 정보 수집 완료")
    else:
        print("[9/10] 수집된 공고가 없습니다")

    paths["json"].write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[9/10] JSON 저장 완료: {paths['json']}")
    print(str(paths["json"]))
    print("[10/10] SK주식회사(AX) 크롤링 작업 완료")


if __name__ == "__main__":
    main()

