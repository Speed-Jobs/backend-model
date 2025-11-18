"""
9개 리팩토링된 크롤러 순차 실행

기존 4개 크롤러:
- 현대오토에버 (비동기)
- LG CNS (비동기)
- 한화시스템 (동기)
- 카카오 (동기)

추가된 5개 크롤러 (모두 동기):
- Coupang
- Line
- Naver
- Toss
- Woowahan (배달의민족)
"""

# 기존 리팩토링된 크롤러 import
from app.services.crawler.hyundai_autoever.crawler_hyundai_autoever import main as hyundai_crawler
from app.services.crawler.lg_cns.crawler_lg_cns import main as lg_crawler
from app.services.crawler.hanwha.crawler_hanwha import main as hanwha_crawler
from app.services.crawler.kakao.crawler_kakao import main as kakao_crawler

# 새로 추가된 크롤러 import
from app.services.crawler.coupang.crawler_coupang import main as coupang_crawler
from app.services.crawler.line.crawler_line import main as line_crawler
from app.services.crawler.naver.crawler_naver import main as naver_crawler
from app.services.crawler.toss.crawler_toss import main as toss_crawler
from app.services.crawler.woowahan.crawler_woowahan import main as woowahan_crawler

import asyncio
import inspect
import logging
from datetime import datetime
import time
import warnings


# AsyncOpenAI cleanup 에러 로깅 억제
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def run_crawler_safely(crawler_func, name):
    """개별 크롤러를 안전하게 실행 (리소스 완전 격리)"""
    try:
        print(f"\n{'='*80}")
        print(f"[{datetime.now()}] 🚀 {name} 시작")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        if inspect.iscoroutinefunction(crawler_func):
            # 비동기 함수: 새 이벤트 루프에서 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(crawler_func())
            finally:
                # 보류 중인 태스크 정리
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                
                # 취소된 태스크 완료 대기 (에러 무시)
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
                
                # 비동기 제너레이터 종료 (에러 무시)
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                
                # 루프 종료
                loop.close()
        else:
            # 동기 함수: 그냥 실행
            crawler_func()
        
        duration = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"[{datetime.now()}] ✅ {name} 완료 ({duration/60:.1f}분)")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"[{datetime.now()}] ❌ {name} 실패: {e}")
        print(f"{'='*80}\n")
        
        import traceback
        traceback.print_exc()
        
        return False


def run_all_crawlers_sequentially():
    """9개 리팩토링된 크롤러를 순차적으로 실행"""
    
    crawlers = [
        (hyundai_crawler, "현대오토에버 (비동기)"),
        (lg_crawler, "LG CNS (비동기)"),
        (hanwha_crawler, "한화시스템 (동기)"),
        (kakao_crawler, "카카오 (동기)"),
        (coupang_crawler, "Coupang (동기)"),
        (line_crawler, "Line (동기)"),
        (naver_crawler, "Naver (동기)"),
        (toss_crawler, "Toss (동기)"),
        (woowahan_crawler, "Woowahan/배달의민족 (동기)"),
    ]
    
    print(f"\n{'='*80}")
    print(f"[{datetime.now()}] 🔬 9개 리팩토링된 크롤러 배치 시작")
    print(f"{'='*80}\n")
    
    results = {}
    total_start = time.time()
    
    for idx, (crawler_func, name) in enumerate(crawlers, 1):
        print(f"\n[{idx}/{len(crawlers)}] {name} 실행 중...")
        
        success = run_crawler_safely(crawler_func, name)
        results[name] = success
    
        # 크롤러 간 대기 시간 (리소스 정리)
        if idx < len(crawlers):
            wait_time = 15 if idx in [4, 5] else 10  # Coupang, Line은 15초 대기 (Cloudflare 고려)
            print(f"\n⏳ 다음 크롤러 시작 전 {wait_time}초 대기 (리소스 정리)...")
            time.sleep(wait_time)
    
    # 최종 결과 출력
    total_duration = time.time() - total_start
    success_count = sum(1 for success in results.values() if success)
    fail_count = len(crawlers) - success_count
    
    print(f"\n{'='*80}")
    print(f"[{datetime.now()}] 📊 전체 크롤링 배치 완료")
    print(f"{'='*80}")
    print(f"총 실행 시간: {total_duration/60:.1f}분")
    print(f"성공: {success_count}개 | 실패: {fail_count}개\n")
    """
    print("상세 결과:")
    print("\n[기존 4개 크롤러]")
    for name, success in list(results.items())[:4]:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  - {name}: {status}")
    """
    print("\n[새로 추가된 5개 크롤러]")
    for name, success in list(results.items())[:]:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  - {name}: {status}")
    
    print(f"{'='*80}\n")
    
    if success_count == len(crawlers):
        print("🎉 모든 크롤러가 성공적으로 실행되었습니다!")
        print("✅ 리소스 충돌 없음 - 리팩토링 성공!\n")
    else:
        print(f"⚠️ {fail_count}개 크롤러 실패 - 로그를 확인하세요.\n")


