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
        "- company: 회사 이름\n"
        "- location: 근무 위치\n"
        "- employment_type: 고용 형태\n"
        "- experience: 경력 요구사항\n"
        "- crawl_date: 크롤링 날짜 (YYYY-MM-DD)\n"
        "- posted_date: 공고 게시일 (YYYY-MM-DD)\n"
        "- expired_date: 공고 마감일 (YYYY-MM-DD, 없으면 null)\n"
        "- description: 채용공고 전문 텍스트\n"
        "- meta_data: 추가 정보 (JSON 객체)\n\n"
        "# 출력 형식\n"
        "반드시 JSON 리스트 형식으로 반환하세요.\n"
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
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        print(f"  [{job_index}] 상세 페이지 로딩...")
        page.goto(job_url, timeout=30000)
        page.wait_for_timeout(1500)

        today = datetime.now().strftime('%Y-%m-%d')

        job_info = {
            "title": None,
            "company": "Naver",
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

        # description 추출
        full_text = page.inner_text("body")
        try:
            title_box = page.query_selector("div.card_title_box")
            detail_wrap = page.query_selector("div.detail_wrap")

            description_parts = []
            if title_box:
                title_text = title_box.inner_text()
                if title_text:
                    description_parts.append(title_text)

            if detail_wrap:
                detail_text = detail_wrap.inner_text()
                if detail_text:
                    description_parts.append(detail_text)

            if description_parts:
                job_info["description"] = "\n\n".join(description_parts)
            else:
                job_info["description"] = full_text

            # 스크린샷 저장
            if screenshot_dir:
                try:
                    anno_id_match = re.search(r'annoId=(\d+)', job_url)
                    anno_id = anno_id_match.group(1) if anno_id_match else f"job_{job_index}"

                    screenshot_filename = f"naver_job_{anno_id}.png"
                    screenshot_path = screenshot_dir / screenshot_filename

                    page.screenshot(path=str(screenshot_path), full_page=True)
                    job_info["screenshots"]["combined"] = str(screenshot_path)
                    print(f"  [{job_index}] 스크린샷 저장: {screenshot_filename}")
                except Exception as e:
                    print(f"  [{job_index}] 스크린샷 저장 실패: {e}")

        except Exception as e:
            print(f"  [{job_index}] description 추출 실패, body 전체 사용: {e}")
            job_info["description"] = full_text

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
        "raw_html": out_dir / "naver_raw.html",
        "json": out_dir / "naver_jobs.json",
        "screenshots": screenshot_dir,
    }

    all_job_urls = []
    jobs_list = []
    all_html_content = []
    
    playwright = None
    browser = None
    context = None
    page = None

    try:
        print("[1/10] 브라우저 실행 중...")
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        
        if fast:
            try:
                context.set_default_timeout(5000)
            except Exception:
                pass
        
        page = context.new_page()

        # Naver 채용 페이지 URL (IT 직군 필터)
        base_url = "https://recruit.navercorp.com/rcrt/list.do?srchClassCd=1000000"

        print(f"[2/10] 페이지 접속: {base_url}")
        page.goto(base_url, timeout=60000)
        page.wait_for_load_state("domcontentloaded")
        page.wait_for_timeout(3000)

        # 무한 스크롤로 모든 콘텐츠 로드
        print(f"[4/10] 페이지 스크롤하여 전체 콘텐츠 로드 중...")
        prev_height = 0
        scroll_attempts = 0
        max_scrolls = 10

        while scroll_attempts < max_scrolls:
            current_height = page.evaluate("document.body.scrollHeight")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)
            new_height = page.evaluate("document.body.scrollHeight")

            if new_height == prev_height:
                break

            prev_height = new_height
            scroll_attempts += 1

        # HTML 저장
        html_content = page.content()
        all_html_content.append(html_content)

        # 채용 공고 링크 수집
        print("[5/10] 채용 공고 링크 추출 중...")

        try:
            links = page.query_selector_all("a.card_link[onclick]")

            for link in links:
                onclick = link.get_attribute("onclick")
                if onclick:
                    match = re.search(r"show\(['\"]?(\d+)['\"]?\)", onclick)
                    if match:
                        anno_id = match.group(1)
                        job_url = f"https://recruit.navercorp.com/rcrt/view.do?annoId={anno_id}&lang=ko"
                        if job_url not in all_job_urls:
                            all_job_urls.append(job_url)

            print(f"[5/10] {len(all_job_urls)}개 링크 추출 완료")

        except Exception as e:
            print(f"[5/10] 링크 추출 실패: {e}")

        print(f"[6/10] 전체 채용 공고 URL: {len(all_job_urls)}개")

        if not all_job_urls:
            print("[!] 채용 공고 링크를 찾을 수 없습니다.")
            if all_html_content:
                outputs["raw_html"].write_text(all_html_content[-1], encoding="utf-8")
                print(f"[DEBUG] 원본 HTML 저장: {outputs['raw_html']}")
            return outputs, []
        
        # 병렬 크롤링
        print(f"[7/10] 병렬로 {len(all_job_urls)}개 공고 상세 크롤링 시작...")
        executor = None
        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
            futures = {
                executor.submit(extract_job_detail_from_url, url, idx, screenshot_dir): idx
                for idx, url in enumerate(all_job_urls, start=1)
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

    parser = argparse.ArgumentParser(description="Naver Careers 스크래핑")
    parser.add_argument("--out-dir", default="../../output", help="출력 폴더")
    parser.add_argument("--screenshot-dir", default="../../img", help="스크린샷 폴더")
    parser.add_argument("--fast", action="store_true", help="빠른 모드")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    screenshot_dir = Path(args.screenshot_dir)
    print("[0/10] Naver 크롤링 작업 시작")

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
    print("[10/10] Naver 크롤링 작업 완료")

if __name__ == "__main__":
    main()