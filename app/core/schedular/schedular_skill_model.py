"""
분기마다 스킬 연관성(Node2Vec) 모델을 재학습하는 스케줄러
"""

from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core.training_model.skill_similarity_training_model import (
    train_skill_association_model_from_db,
)


def run_skill_model_training_job():
    """
    DB 기반으로 스킬 연관성 모델을 재학습하는 단일 배치 작업.
    """
    print("\n" + "=" * 80)
    print(f"[{datetime.now()}] 🚀 스킬 연관성 Node2Vec 모델 재학습 시작")
    print("=" * 80 + "\n")

    try:
        # 최근 1년 데이터를 기준으로 학습 (필요 시 days 인자 조정 가능)
        train_skill_association_model_from_db(days=365)
        print(f"[{datetime.now()}] ✅ 스킬 연관성 모델 재학습 완료")
    except Exception as e:
        print(f"[{datetime.now()}] ❌ 스킬 연관성 모델 재학습 실패: {e}")


def run_skill_model_scheduler():
    """
    분기(약 13주)마다 스킬 연관성 모델을 재학습하도록 스케줄러 실행.

    - IntervalTrigger(weeks=13)를 사용해 분기 주기 근사
    - next_run_time=datetime.now() 로 즉시 1회 실행 후 주기 반복
    """
    scheduler = BlockingScheduler()

    scheduler.add_job(
        run_skill_model_training_job,
        IntervalTrigger(weeks=13),
        name="Skill Association Node2Vec 모델 분기별 재학습",
        replace_existing=True,
        next_run_time=datetime.now(),  # 시작 시점에 즉시 1회 실행
    )

    print("=" * 80)
    print("🧠 스킬 연관성(Node2Vec) 모델 재학습 스케줄러 시작")
    print("=" * 80)
    print("실행 주기: 약 1분기(13주)마다")
    print("다음 실행: 지금 즉시")
    print("=" * 80)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n스킬 모델 스케줄러 종료")


if __name__ == "__main__":
    run_skill_model_scheduler()


