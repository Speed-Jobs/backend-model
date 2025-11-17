"""
사람인 채용공고 크롤러 (IT/Internet 카테고리)
1. 페이지별로 URL 수집 (cat_mcls=2: IT/Internet)
2. ThreadPoolExecutor로 병렬 크롤링 (각 스레드가 독립 브라우저)
3. 이미지 기반 공고는 OpenAI Vision API로 OCR 처리

주요 특징:
- OpenAI 클라이언트 명시적 관리
- Playwright 리소스 확실한 cleanup
- 텍스트/이미지 공고 자동 판별 및 처리
- 에러 핸들링 개선
"""

import argparse
import json
import os
import re
import base64
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
from app.services import resolve_dir, get_output_dir, get_img_dir
import concurrent.futures

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


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
    path.mkdir(parents=True, exist_ok=True)


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(10))
def summarize_with_llm(raw_text: str, model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
    """OpenAI 클라이언트를 명시적으로 관리"""
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
- meta_data: 위 필드 외 추가 정보를 담은 JSON 객체

# 중요 지침
1. 날짜는 반드시 YYYY-MM-DD 형식으로 통일
2. 정보가 없는 경우 null 반환 (빈 문자열 X)
3. description은 HTML 태그를 제거한 순수 텍스트
4. meta_data는 의미있는 키 이름으로 구조화 (영문 snake_case 사용)

---
# Example
{
    "title": "백엔드 개발자",
    "company": "(주)테크",
    "location": "서울",
    "employment_type": "정규직",
    "experience": "경력 3~5년",
    "crawl_date": "2025-11-05",
    "posted_date": "2025-10-28",
    "expired_date": "2025-11-30",
    "description": "주요업무...",
    "meta_data": {"job_category": "IT/개발"}
}
---
"""

    user_prompt = (
        f"오늘 날짜는 {datetime.now().strftime('%Y-%m-%d')}이고, 이 날짜를 crawl_date로 사용해. "
        f"공고들을 위 스키마에 맞춰 리스트로 정리해줘.\n\n" + raw_text
    )

    # OpenAI 클라이언트 생성 및 사용
    client = OpenAI(api_key=api_key)

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

        json_text_match = re.search(r"(\[.*\])", content, re.DOTALL)
        json_text = json_text_match.group(1) if json_text_match else content

        try:
            data = json.loads(json_text)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []
    finally:
        # OpenAI 클라이언트 명시적 정리
        client.close()


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def ocr_with_vision(image_path: str, model: str = "gpt-4o-mini") -> str:
    """OpenAI Vision API로 이미지 기반 공고 OCR 처리"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        print(f"  OCR 스킵: OpenAI API 키 없음")
        return ""

    # 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"  OCR 실패: 파일 없음 - {image_path}")
        return ""

    # 파일 크기 확인 (20MB 제한)
    file_size = os.path.getsize(image_path)
    if file_size > 20 * 1024 * 1024:
        print(f"  OCR 스킵: 파일 크기 초과 ({file_size / 1024 / 1024:.2f}MB)")
        return ""

    try:
        # 이미지를 base64로 인코딩
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        system_prompt = """
당신은 채용 공고 이미지에서 텍스트를 추출하는 전문가입니다.
이미지에 있는 모든 텍스트를 정확하게 읽어서 반환하세요.

# 지침
1. 이미지의 모든 텍스트를 순서대로 추출 (위에서 아래로, 왼쪽에서 오른쪽으로)
2. 제목, 부제목, 본문 등의 구조를 유지하면서 텍스트화
3. 불필요한 해석이나 요약 없이 있는 그대로 추출
4. 텍스트만 추출하고 다른 설명은 추가하지 마세요
"""

        client = OpenAI(api_key=api_key)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": system_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                    "detail": "high"  # 고해상도 분석
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.0,  # 더 결정적인 출력
            )
            content = response.choices[0].message.content if response and response.choices else ""
            return content.strip()
        finally:
            client.close()

    except Exception as e:
        print(f"  OCR 에러: {type(e).__name__} - {str(e)[:100]}")
        return ""


def extract_job_detail_from_url(job_url: str, job_index: int, screenshot_dir: Path = None) -> Dict[str, Any]:
    """URL로 직접 접속하여 상세 정보 추출 (병렬 처리용 - 독립 브라우저)"""
    playwright_instance = None
    browser = None
    context = None
    page = None

    try:
        playwright_instance = sync_playwright().start()
        browser = playwright_instance.chromium.launch(
            headless=False,  # 헤드리스 모드 해제 (봇 감지 우회)
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox'
            ]
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={'width': 1920, 'height': 1080},
            locale='ko-KR',
            timezone_id='Asia/Seoul'
        )

        # 웹드라이버 감지 방지
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        page = context.new_page()

        print(f"  [{job_index}] 상세 페이지 로딩...")
        page.goto(job_url, timeout=30000, wait_until="load")
        page.wait_for_timeout(3000)

        # 리다이렉트 후 실제 URL 확인
        actual_url = page.url
        if actual_url != job_url:
            print(f"  [{job_index}] 리다이렉트: {actual_url}")

        # 페이지가 완전히 로드될 때까지 대기 (h1.tit_job 요소 기다림)
        try:
            page.wait_for_selector("h1.tit_job", timeout=10000)
        except Exception as e:
            print(f"  [{job_index}] h1.tit_job 대기 실패: {e}")

        today = datetime.now().strftime('%Y-%m-%d')

        job_info = {
            "title": None,
            "company": None,
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

        # 제목 추출 (h1.tit_job)
        try:
            title_elem = page.query_selector("h1.tit_job")
            if title_elem:
                job_info["title"] = title_elem.inner_text().strip()
        except Exception as e:
            print(f"  [{job_index}] 제목 추출 실패: {e}")

        # div.col 2개 추출 (회사 정보, 공고 정보 등)
        col_texts = []
        try:
            col_elems = page.query_selector_all("div.col")
            for col in col_elems[:2]:  # 최대 2개
                col_texts.append(col.inner_text().strip())
        except Exception as e:
            print(f"  [{job_index}] div.col 추출 실패: {e}")

        # div.wrap_jv_cont에서 description 추출 (텍스트 or 이미지)
        full_text = page.inner_text("body")
        description_text = ""
        is_image_based = False

        try:
            wrap_jv_cont = page.query_selector("div.wrap_jv_cont")
            if wrap_jv_cont:
                # 이미지 기반 공고인지 확인 (더 정확한 판별)
                img_tags = wrap_jv_cont.query_selector_all("img")

                # 텍스트 내용 추출
                text_content = wrap_jv_cont.inner_text().strip()

                # 이미지가 많고 텍스트가 적으면 이미지 기반으로 판단
                if img_tags and len(img_tags) > 0 and len(text_content) < 200:
                    is_image_based = True
                    print(f"  [{job_index}] 이미지 기반 공고 감지 (이미지: {len(img_tags)}개, 텍스트: {len(text_content)}자)")

                    # 개별 이미지 다운로드 및 OCR 처리
                    ocr_results = []
                    if screenshot_dir:
                        job_id_match = re.search(r'rec_idx=(\d+)', job_url)
                        job_id = job_id_match.group(1) if job_id_match else f"job_{job_index}"

                        for img_idx, img_tag in enumerate(img_tags, 1):
                            try:
                                img_src = img_tag.get_attribute("src")
                                if not img_src:
                                    continue

                                # 작은 아이콘/로고는 제외 (width, height 체크)
                                img_width = img_tag.evaluate("el => el.naturalWidth")
                                img_height = img_tag.evaluate("el => el.naturalHeight")

                                if img_width < 100 or img_height < 100:
                                    print(f"  [{job_index}] 이미지 {img_idx} 스킵 (크기: {img_width}x{img_height})")
                                    continue

                                print(f"  [{job_index}] 이미지 {img_idx} 처리 중... (크기: {img_width}x{img_height})")

                                # 이미지 요소만 스크린샷
                                img_filename = f"saramin_job_{job_id}_img{img_idx}.png"
                                img_path = screenshot_dir / img_filename

                                img_tag.screenshot(path=str(img_path))

                                # OCR 처리
                                ocr_text = ocr_with_vision(str(img_path))
                                if ocr_text:
                                    ocr_results.append(f"[이미지 {img_idx}]\n{ocr_text}")
                                    print(f"  [{job_index}] 이미지 {img_idx} OCR 완료 ({len(ocr_text)} 글자)")

                            except Exception as e:
                                print(f"  [{job_index}] 이미지 {img_idx} 처리 실패: {e}")
                                continue

                        # 전체 페이지 스크린샷도 저장
                        try:
                            screenshot_filename = f"saramin_job_{job_id}_full.png"
                            screenshot_path = screenshot_dir / screenshot_filename
                            page.screenshot(path=str(screenshot_path), full_page=True)
                            job_info["screenshots"]["combined"] = str(screenshot_path)
                            print(f"  [{job_index}] 전체 스크린샷 저장: {screenshot_filename}")
                        except Exception as e:
                            print(f"  [{job_index}] 전체 스크린샷 저장 실패: {e}")

                    # OCR 결과 결합
                    if ocr_results:
                        description_text = "\n\n".join(ocr_results)
                        print(f"  [{job_index}] 총 OCR 완료: {len(ocr_results)}개 이미지, {len(description_text)} 글자")
                    else:
                        # OCR 실패 시 폴백: 전체 페이지 스크린샷으로 OCR
                        print(f"  [{job_index}] 개별 이미지 OCR 실패, 전체 페이지로 재시도...")
                        if screenshot_dir and "combined" in job_info["screenshots"]:
                            try:
                                ocr_text = ocr_with_vision(job_info["screenshots"]["combined"])
                                if ocr_text:
                                    description_text = ocr_text
                                    print(f"  [{job_index}] 전체 페이지 OCR 완료 ({len(ocr_text)} 글자)")
                            except Exception as e:
                                print(f"  [{job_index}] 전체 페이지 OCR 실패: {e}")
                                description_text = text_content  # 최후의 폴백
                else:
                    # 텍스트 기반 공고
                    description_text = text_content
                    print(f"  [{job_index}] 텍스트 기반 공고 (텍스트: {len(text_content)}자)")
            else:
                description_text = full_text

            job_info["description"] = description_text if description_text else full_text

            # 이미지 기반이 아닌 경우에만 스크린샷 저장
            if screenshot_dir and not is_image_based:
                try:
                    job_id_match = re.search(r'rec_idx=(\d+)', job_url)
                    job_id = job_id_match.group(1) if job_id_match else f"job_{job_index}"

                    screenshot_filename = f"saramin_job_{job_id}.png"
                    screenshot_path = screenshot_dir / screenshot_filename

                    page.screenshot(path=str(screenshot_path), full_page=True)
                    job_info["screenshots"]["combined"] = str(screenshot_path)
                    print(f"  [{job_index}] 스크린샷 저장: {screenshot_filename}")
                except Exception as e:
                    print(f"  [{job_index}] 스크린샷 저장 실패: {e}")

        except Exception as e:
            print(f"  [{job_index}] wrap_jv_cont 추출 실패, body 전체 사용: {e}")
            job_info["description"] = full_text

        # LLM으로 나머지 필드 파싱 시도
        try:
            # col_texts를 포함하여 전체 컨텍스트 구성
            llm_input = full_text
            if col_texts:
                llm_input = "\n\n".join(col_texts) + "\n\n" + full_text

            parsed = summarize_with_llm(llm_input)
            if parsed and len(parsed) > 0:
                parsed_data = parsed[0]
                for key in ["title", "company", "location", "employment_type", "experience",
                           "posted_date", "expired_date", "meta_data"]:
                    if key in parsed_data and parsed_data[key]:
                        job_info[key] = parsed_data[key]
        except Exception as e:
            print(f"  [{job_index}] LLM 파싱 실패 (description은 저장됨): {e}")

        print(f"  [{job_index}] 완료: {job_info.get('title', 'N/A')}")
        return job_info

    except Exception as e:
        print(f"  [{job_index}] 상세 정보 추출 실패: {e}")
        return None

    finally:
        # 리소스 확실하게 정리
        if page:
            try:
                page.close()
            except Exception:
                pass
        if context:
            try:
                context.close()
            except Exception:
                pass
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        if playwright_instance:
            try:
                playwright_instance.stop()
            except Exception:
                pass


def run_scrape(
    out_dir: Path = None,
    screenshot_dir: Path = None,
    fast: bool = False
) -> Dict[str, Path]:
    """메인 크롤링 함수 (사람인 IT/Internet 카테고리)"""
    out_dir = resolve_dir(out_dir, get_output_dir())
    screenshot_dir = resolve_dir(screenshot_dir, get_img_dir())
    ensure_dir(out_dir)
    ensure_dir(screenshot_dir)

    outputs = {
        "raw_html": out_dir / "saramin_raw.html",
        "clean_txt": out_dir / "saramin_clean.txt",
        "json": out_dir / "saramin_jobs.json",
        "screenshots": screenshot_dir,
    }

    all_job_urls = []
    jobs_list = []

    playwright_instance = None
    browser = None
    context = None
    page = None

    try:
        print("[1/10] 브라우저 실행 중...")
        playwright_instance = sync_playwright().start()
        browser = playwright_instance.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        if fast:
            try:
                context.set_default_timeout(5000)
            except Exception:
                pass
        page = context.new_page()

        base_url = "https://www.saramin.co.kr/zf_user/jobs/list/job-category"
        page_num = 1
        all_html_content = []

        # 테스트: 1페이지만 크롤링
        url = f"{base_url}?page=1&cat_mcls=2&page_count=50"

        print(f"[2/10] 페이지 1 접속: {url}")
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)

        # 채용 공고 링크 찾기 (각 공고당 하나의 링크만)
        print(f"[5/10] 페이지 1 채용 공고 링크 찾는 중...")
        try:
            # 제목 링크만 선택 (공고당 1개)
            selectors_to_try = [
                "div.job_tit a.str_tit",  # 제목 링크만
                "h2.job_tit a",
                "a.job_tit",
            ]

            links = []
            for sel in selectors_to_try:
                links = page.query_selector_all(sel)
                if links and len(links) > 0:
                    print(f"[5/10] 셀렉터 '{sel}'로 {len(links)}개 링크 발견")
                    break

            if links and len(links) > 0:
                total_links = len(links)
                print(f"[5/10] 페이지 1: {total_links}개 링크 발견")

                for link in links:
                    try:
                        href = link.get_attribute("href")
                        if href:
                            if href.startswith("/"):
                                full_url = f"https://www.saramin.co.kr{href}"
                            else:
                                full_url = href
                            if full_url not in all_job_urls:
                                all_job_urls.append(full_url)
                    except Exception as e:
                        print(f"  URL 수집 실패: {e}")
                        continue

                print(f"[6/10] 페이지 1에서 신규 URL: {len(all_job_urls)}개")
            else:
                print(f"[5/10] 페이지 1에서 링크 없음")

        except Exception as e:
            print(f"[5/10] 페이지 1 링크 찾기 실패: {e}")

        html = page.content()
        all_html_content.append(html)

        print(f"[6/10] 전체 수집된 URL: {len(all_job_urls)}개")

        # HTML 저장
        if all_html_content:
            outputs["raw_html"].write_text(all_html_content[-1], encoding="utf-8")
            print(f"[8/10] 원본 HTML 저장: {outputs['raw_html']}")

    finally:
        # 리소스 확실하게 정리
        if page:
            try:
                page.close()
            except Exception:
                pass
        if context:
            try:
                context.close()
            except Exception:
                pass
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        if playwright_instance:
            try:
                playwright_instance.stop()
            except Exception:
                pass

    # 병렬로 각 URL의 상세 정보 크롤링
    print(f"[7/10] 병렬로 {len(all_job_urls)}개 공고 상세 정보 크롤링 시작...")
    if all_job_urls:
        # 워커 수를 줄여서 안정성 향상 (테스트는 5개로 제한)
        max_workers = min(5, len(all_job_urls))
        print(f"[7/10] ThreadPoolExecutor 워커 수: {max_workers}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, job_url in enumerate(all_job_urls, 1):
                future = executor.submit(extract_job_detail_from_url, job_url, idx, screenshot_dir)
                futures[future] = (idx, job_url)

            completed_count = 0
            failed_count = 0

            for future in concurrent.futures.as_completed(futures):
                idx, job_url = futures[future]
                try:
                    job_info = future.result(timeout=120)  # 최대 2분 대기
                    if job_info:
                        jobs_list.append(job_info)
                        completed_count += 1
                    else:
                        failed_count += 1
                        print(f"  [{idx}] 결과 없음: {job_url}")
                except concurrent.futures.TimeoutError:
                    failed_count += 1
                    print(f"  [{idx}] 타임아웃: {job_url}")
                except Exception as e:
                    failed_count += 1
                    print(f"  [{idx}] 작업 실패: {type(e).__name__} - {str(e)[:100]}")

                # 진행 상황 출력
                total_processed = completed_count + failed_count
                if total_processed % 5 == 0:
                    print(f"[7/10] 진행: {total_processed}/{len(all_job_urls)} (성공: {completed_count}, 실패: {failed_count})")

            print(f"[7/10] 크롤링 완료: 성공 {completed_count}개, 실패 {failed_count}개")

    print("[8/10] 브라우저 종료")
    if all_html_content:
        print("[9/10] HTML 저장 완료")

    return outputs, jobs_list


def main() -> None:
    """메인 함수"""
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser(description="사람인 채용공고 크롤러 테스트 (1페이지만)")
    parser.add_argument("--out-dir", default="../../output", help="출력 폴더")
    parser.add_argument("--screenshot-dir", default="../../img", help="스크린샷 폴더")
    parser.add_argument("--fast", action="store_true", help="빠른 모드")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    screenshot_dir = Path(args.screenshot_dir)

    print("="*80)
    print("사람인 채용공고 크롤러 테스트 시작 (1페이지만)")
    print("="*80)
    print("[0/10] 작업 시작")

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
    print("[10/10] 작업 완료")
    print("="*80)


if __name__ == "__main__":
    main()