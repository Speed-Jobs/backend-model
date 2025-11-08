"""쿠팡 채용 페이지 구조 분석 스크립트"""
import time
from playwright.sync_api import sync_playwright

url = "https://www.coupang.jobs/kr/jobs/7317086"

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

    # 자동화 감지 방지
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

    print(f"페이지 로딩: {url}")
    page.goto(url, timeout=60000, wait_until="domcontentloaded")

    # Cloudflare 대기
    try:
        page.wait_for_load_state("networkidle", timeout=20000)
    except Exception:
        pass

    page.wait_for_timeout(5000)

    print("\n=== 페이지 분석 시작 ===\n")

    # 모든 div 클래스 찾기
    divs = page.query_selector_all("div[class]")
    classes = set()
    for div in divs[:100]:  # 처음 100개만
        class_attr = div.get_attribute("class")
        if class_attr:
            classes.add(class_attr)

    print("발견된 주요 div 클래스들:")
    for cls in sorted(classes):
        print(f"  - {cls}")

    print("\n=== 콘텐츠 추출 테스트 ===\n")

    # 여러 셀렉터 시도
    selectors = [
        "div.col-lg-8.main-col",
        "div.job_table",
        "div.main-col",
        "div.content",
        "div.job-description",
        "div.description",
        "div#job-description",
        "div[class*='description']",
        "div[class*='content']",
        "div[class*='detail']",
        "main",
        "article",
    ]

    for selector in selectors:
        try:
            elem = page.query_selector(selector)
            if elem:
                text = elem.inner_text()
                print(f"\n셀렉터: {selector}")
                print(f"텍스트 길이: {len(text)} 글자")
                print(f"미리보기: {text[:200]}...")
                print("-" * 80)
        except Exception as e:
            print(f"셀렉터 {selector} 실패: {e}")

    # body 전체 텍스트
    body_text = page.inner_text("body")
    print(f"\n전체 body 텍스트 길이: {len(body_text)} 글자")
    print(f"body 미리보기:\n{body_text[:500]}")

    # HTML 저장
    html = page.content()
    with open("c:/workspace/fproject/backend-model/output/test_coupang_page.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("\n\nHTML 저장: output/test_coupang_page.html")

    browser.close()
    print("\n완료!")
