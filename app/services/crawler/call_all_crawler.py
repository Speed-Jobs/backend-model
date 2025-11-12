"""
9개 리팩토링된 크롤러 순차 실행 스케줄러

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

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
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