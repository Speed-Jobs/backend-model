"""
한화시스템 채용공고 크롤러 (리팩토링)
1. HTML 파싱으로 공고 URL 리스트 수집
2. ThreadPoolExecutor로 병렬 크롤링 (각 스레드가 독립 브라우저)

주요 개선사항:
- OpenAI 클라이언트 명시적 관리
- Playwright 리소스 확실한 cleanup
- 에러 핸들링 개선
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
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
        legacy_env = Path(__file__).resolve().parents[4] / ".env"
        if legacy_env.exists():
            load_dotenv(dotenv_path=legacy_env, override=False)
    except Exception:
        pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


load_env()


def clean_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t\x0b\x0c\r]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
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
- meta_data: 위 필드 외 추가 정보를 담은 JSON 객체 (예: 직군, 연봉정보, 복리후생, 우대사항, 기술스택 등)

# 중요 지침
1. 날짜는 반드시 YYYY-MM-DD 형식으로 통일
2. 정보가 없는 경우 null 반환 (빈 문자열 X)
3. description은 HTML 태그를 제거한 순수 텍스트
4. meta_data는 의미있는 키 이름으로 구조화 (영문 snake_case 사용)

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
            max_tokens=8000,
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
    finally:
        # OpenAI 클라이언트 명시적 정리
        client.close()


def wait_for_results(page: Page, timeout_ms: int = 10000) -> None:
    """한화시스템 채용 페이지 로딩 대기"""
    print(f"[3/10] 검색 결과 렌더링 대기 중... (최대 {timeout_ms}ms)")
    selectors = [
        "li[onclick*='goView']",
        ".recruit_list",
        "ul li",
    ]
    elapsed = 0
    step_ms = 700
    while elapsed < timeout_ms:
        for sel in selectors:
            try:
                page.wait_for_selector(sel, timeout=900)
                print(f"[3/10] 대표 셀렉터 감지: {sel}")
                return
            except Exception:
                continue
        try:
            page.mouse.wheel(0, 800)
        except Exception:
            pass
        page.wait_for_timeout(step_ms)
        elapsed += step_ms
    print("[3/10] 타임아웃 도달: 셀렉터 미감지 - 다음 단계로 진행")


def extract_job_detail_from_url(job_url: str, job_index: int, screenshot_dir: Path = None) -> Dict[str, Any]:
    """URL로 직접 접속하여 상세 정보 추출 (병렬 처리용 - 독립 브라우저)"""
    playwright_instance = None
    browser = None
    context = None
    page = None
    
    try:
        # 각 스레드에서 독립적인 playwright와 브라우저 인스턴스 생성
        playwright_instance = sync_playwright().start()
        browser = playwright_instance.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        print(f"  [{job_index}] 상세 페이지 로딩...")
        page.goto(job_url, timeout=30000)
        page.wait_for_timeout(2000)

        today = datetime.now().strftime('%Y-%m-%d')

        job_info = {
            "title": None,
            "company": "한화시스템",
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

        # description 추출 - div.recruit1과 div.recruit2에서 가져오기
        full_text = page.inner_text("body")
        try:
            recruit1 = page.query_selector("div.recruit1")
            recruit2 = page.query_selector("div.recruit2")

            description_parts = []

            if recruit1:
                recruit1_text = recruit1.inner_text()
                if recruit1_text:
                    description_parts.append(recruit1_text)

            if recruit2:
                recruit2_text = recruit2.inner_text()
                if recruit2_text:
                    description_parts.append(recruit2_text)

            if description_parts:
                job_info["description"] = "\n\n".join(description_parts)
            else:
                job_info["description"] = full_text

            # 스크린샷 저장
            if screenshot_dir:
                try:
                    seq_match = re.search(r'rtSeq=(\d+)', job_url)
                    seq_id = seq_match.group(1) if seq_match else f"job_{job_index}"

                    screenshot_filename = f"hanwha_job_{seq_id}.png"
                    screenshot_path = screenshot_dir / screenshot_filename

                    page.screenshot(path=str(screenshot_path), full_page=True)
                    job_info["screenshots"]["combined"] = str(screenshot_path)
                    print(f"  [{job_index}] 스크린샷 저장: {screenshot_filename}")
                except Exception as e:
                    print(f"  [{job_index}] 스크린샷 저장 실패: {e}")

        except Exception as e:
            print(f"  [{job_index}] recruit1/recruit2 추출 실패, body 전체 사용: {e}")
            job_info["description"] = full_text

        # LLM으로 파싱 시도 (다른 필드만 추출)
        try:
            parsed = summarize_with_llm(full_text)
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
    """메인 크롤링 함수"""
    out_dir = resolve_dir(out_dir, get_output_dir())
    screenshot_dir = resolve_dir(screenshot_dir, get_img_dir())
    ensure_dir(out_dir)
    ensure_dir(screenshot_dir)

    outputs = {
        "raw_html": out_dir / "hanwha_raw.html",
        "clean_txt": out_dir / "hanwha_clean.txt",
        "json": out_dir / "hanwha_jobs.json",
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
        context = browser.new_context()
        if fast:
            try:
                context.set_default_timeout(5000)
            except Exception:
                pass
        page = context.new_page()

        base_url = "https://www.hanwhain.com/web/apply/notification/list.do?schTp=&schNm=&schSdSeqsParam=215&schRjSeqsParam=29&schRtRegularYnParam="

        print(f"[2/10] 페이지 접속: {base_url}")
        page.goto(base_url, timeout=60000)
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(3000)

        wait_for_results(page, timeout_ms=(7000 if fast else 12000))

        # 스크롤
        print(f"[4/10] 페이지 스크롤하여 전체 콘텐츠 로드 중...")
        for _ in range(3):
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(800)
            except Exception:
                pass

        # 채용 공고 링크 추출
        print(f"[5/10] 채용 공고 링크 추출 중...")
        html = page.content()
        pattern = r"goView\((\d+)\)"
        matches = re.findall(pattern, html)

        for seq in matches:
            job_url = f"https://www.hanwhain.com/web/apply/notification/view.do?rtSeq={seq}"
            if job_url not in all_job_urls:
                all_job_urls.append(job_url)

        print(f"[6/10] 전체 수집된 URL: {len(all_job_urls)}개")

        # HTML 저장
        outputs["raw_html"].write_text(html, encoding="utf-8")
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
            futures = []
            for idx, job_url in enumerate(all_job_urls, 1):
                future = executor.submit(extract_job_detail_from_url, job_url, idx, screenshot_dir)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    job_info = future.result()
                    if job_info:
                        jobs_list.append(job_info)
                except Exception as e:
                    print(f"  작업 실패: {e}")

    # 정제
    if html:
        clean_txt = clean_text_from_html(html)
        print("[9/10] HTML 기반 정제 완료")
        outputs["clean_txt"].write_text(clean_txt, encoding="utf-8")
        print(f"[9/10] 정제 텍스트 저장: {outputs['clean_txt']}")

    return outputs, jobs_list


def main() -> None:
    """메인 함수"""
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    parser = argparse.ArgumentParser(description="한화시스템 채용 스크래핑 (Refactored)")
    parser.add_argument("--out-dir", default="../../output", help="출력 폴더")
    parser.add_argument("--screenshot-dir", default="../../img", help="스크린샷 폴더")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="OpenAI 모델")
    parser.add_argument("--fast", action="store_true", help="빠른 모드")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    screenshot_dir = Path(args.screenshot_dir)
    
    print("="*80)
    print("🚀 한화시스템 채용공고 크롤러 시작 (Refactored)")
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

    # 저장
    paths["json"].write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[9/10] JSON 저장 완료: {paths['json']}")

    print(str(paths["json"]))
    print("[10/10] 작업 완료")
    print("="*80)


if __name__ == "__main__":
    main()