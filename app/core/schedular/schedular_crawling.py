# 크롤링 스케줄러 3일에 한번씩 백그라운드 수행
from app.services.crawler.hanwha.scraper_hanwha import main as hanwha_crawler
from app.services.crawler.lg_cns.crawler_lg_cns import main as lg_crawler
from app.services.crawler.hyundai_autoever.crawler_hyundai_autoever import main as hyundai_crawler
from app.services.crawler.kakao.scraper_kakao import main as kakao_crawler
from app.services.crawler.line.scraper_line import main as line_crawler
from app.services.crawler.naver.scraper_naver import main as naver_crawler
from app.services.crawler.woowahan.scraper_woowahan import main as woowahan_crawler
from app.services.crawler.coupang.scraper_coupang import main as coupang_crawler
from app.services.crawler.toss.toss_crawler import main as toss_crawler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import asyncio
from datetime import datetime

def run_lg_cns():
    try:
        print(f"[{datetime.now()}] LG CNS 시작")
        asyncio.run(lg_crawler())
        print(f"[{datetime.now()}] LG CNS 완료")
    except Exception as e:
        print(f"[{datetime.now()}] LG CNS 실패: {e}")

def run_hanwha():
    try:
        print(f"[{datetime.now()}] 한화 시작")
        asyncio.run(hanwha_crawler())
        print(f"[{datetime.now()}] 한화 완료")
    except Exception as e:
        print(f"[{datetime.now()}] 한화 실패: {e}")

def run_hyundai_autoever():
    try:
        print(f"[{datetime.now()}] 현대오토에버 시작")
        asyncio.run(hyundai_crawler())
        print(f"[{datetime.now()}] 현대오토에버 완료")
    except Exception as e:
        print(f"[{datetime.now()}] 현대오토에버 실패: {e}")

def run_kakao():
    try:
        print(f"[{datetime.now()}] 카카오 시작")
        asyncio.run(kakao_crawler())
        print(f"[{datetime.now()}] 카카오 완료")
    except Exception as e:
        print(f"[{datetime.now()}] 카카오 실패: {e}")

def run_line():
    try:
        print(f"[{datetime.now()}] 라인 시작")
        asyncio.run(line_crawler())
        print(f"[{datetime.now()}] 라인 완료")
    except Exception as e:
        print(f"[{datetime.now()}] 라인 실패: {e}")

def run_naver():
    try:
        print(f"[{datetime.now()}] 네이버 시작")
        asyncio.run(naver_crawler())
        print(f"[{datetime.now()}] 네이버 완료")
    except Exception as e:
        print(f"[{datetime.now()}] 네이버 실패: {e}")

def run_woowahan():
    try:
        print(f"[{datetime.now()}] 우아한형제들 시작")
        asyncio.run(woowahan_crawler())
        print(f"[{datetime.now()}] 우아한형제들 완료")
    except Exception as e:
        print(f"[{datetime.now()}] 우아한형제들 실패: {e}")

def run_coupang():
    try:
        print(f"[{datetime.now()}] 쿠팡 시작")
        asyncio.run(coupang_crawler())
        print(f"[{datetime.now()}] 쿠팡 완료")
    except Exception as e:
        print(f"[{datetime.now()}] 쿠팡 실패: {e}")

def run_toss():
    try:
        print(f"[{datetime.now()}] 토스 시작")
        asyncio.run(toss_crawler())
        print(f"[{datetime.now()}] 토스 완료")
    except Exception as e:
        print(f"[{datetime.now()}] 토스 실패: {e}")

def run_schedulers():
    scheduler = BlockingScheduler()

    # 3일에 한번씩 크롤링 작업 스케줄링 (시작하자마자 즉시 실행)
    scheduler.add_job(run_lg_cns, IntervalTrigger(days=3), name="LG CNS 크롤러", replace_existing=True, next_run_time=datetime.now())
    scheduler.add_job(run_hanwha, IntervalTrigger(days=3), name="한화 크롤러", replace_existing=True, next_run_time=datetime.now())
    scheduler.add_job(run_hyundai_autoever, IntervalTrigger(days=3), name="현대오토에버 크롤러", replace_existing=True, next_run_time=datetime.now())
    scheduler.add_job(run_kakao, IntervalTrigger(days=3), name="카카오 크롤러", replace_existing=True, next_run_time=datetime.now())
    scheduler.add_job(run_line, IntervalTrigger(days=3), name="라인 크롤러", replace_existing=True, next_run_time=datetime.now())
    scheduler.add_job(run_naver, IntervalTrigger(days=3), name="네이버 크롤러", replace_existing=True, next_run_time=datetime.now())
    scheduler.add_job(run_woowahan, IntervalTrigger(days=3), name="우아한형제들 크롤러", replace_existing=True, next_run_time=datetime.now())
    scheduler.add_job(run_coupang, IntervalTrigger(days=3), name="쿠팡 크롤러", replace_existing=True, next_run_time=datetime.now())
    scheduler.add_job(run_toss, IntervalTrigger(days=3), name="토스 크롤러", replace_existing=True, next_run_time=datetime.now())

    print("크롤링 스케줄러 시작")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass

if __name__ == "__main__":
    run_schedulers()