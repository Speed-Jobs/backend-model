"""
데이터 파이프라인 스케줄러

3일마다 app/scripts/pipeline/data_pipeline.py를 실행합니다.
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import logging
from datetime import datetime
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
backend_root = Path(__file__).resolve().parents[2]
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# 데이터 파이프라인 import
from app.scripts.pipeline.data_pipeline import main as data_pipeline_main

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_pipeline():
    """데이터 파이프라인 실행"""
    try:
        logger.info("="*80)
        logger.info(f"[{datetime.now()}] 데이터 파이프라인 시작")
        logger.info("="*80)
        
        # data_pipeline의 main() 함수 실행
        # main() 함수는 argparse를 사용하므로 인자 없이 호출
        # sys.argv를 임시로 설정하여 인자 없이 실행되도록 함
        original_argv = sys.argv
        sys.argv = [sys.argv[0]]  # 스크립트 이름만 남김
        
        try:
            data_pipeline_main()
        finally:
            sys.argv = original_argv  # 원래 argv 복원
        
        logger.info("="*80)
        logger.info(f"[{datetime.now()}] 데이터 파이프라인 완료")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("="*80)
        logger.error(f"[{datetime.now()}] 데이터 파이프라인 실패: {e}")
        logger.error("="*80)
        
        import traceback
        traceback.print_exc()


def run_scheduler():
    """스케줄러 시작"""
    scheduler = BlockingScheduler()
    
    scheduler.add_job(
        run_data_pipeline,
        IntervalTrigger(days=3),  # 3일마다 실행
        name="데이터 파이프라인 실행",
        replace_existing=True,
        next_run_time=datetime.now()  # 즉시 실행
    )
    

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("\n스케줄러 종료")


if __name__ == "__main__":
    run_scheduler()

