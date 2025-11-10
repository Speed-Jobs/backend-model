"""단일 채용 공고 크롤링 테스트"""
import sys
from pathlib import Path

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.services.crawler.coupang.scraper_coupang import extract_job_detail_from_url

# 테스트 URL
test_url = "https://www.coupang.jobs/kr/jobs/7317086"
screenshot_dir = Path(__file__).parent / "img"
screenshot_dir.mkdir(exist_ok=True)

print(f"테스트 URL: {test_url}\n")
print("크롤링 시작...\n")

result = extract_job_detail_from_url(test_url, 1, screenshot_dir)

if result:
    print("\n" + "="*80)
    print("크롤링 결과:")
    print("="*80)
    print(f"제목: {result['title']}")
    print(f"회사: {result['company']}")
    print(f"위치: {result['location']}")
    print(f"고용형태: {result['employment_type']}")
    print(f"경력: {result['experience']}")
    print(f"크롤링 날짜: {result['crawl_date']}")
    print(f"게시일: {result['posted_date']}")
    print(f"마감일: {result['expired_date']}")
    print(f"\nDescription 길이: {len(result['description'])} 글자")
    print(f"\nDescription 미리보기 (처음 500자):")
    print("-"*80)
    print(result['description'][:500])
    print("-"*80)
    print(f"\n스크린샷: {result['screenshots']}")
else:
    print("크롤링 실패!")
