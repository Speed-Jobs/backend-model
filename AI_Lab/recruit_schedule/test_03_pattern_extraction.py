"""
채용 일정 패턴 추출 및 예측 (통합)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import Column, Integer, String, JSON, ForeignKey, or_
from sqlalchemy.orm import relationship, joinedload
from app.db.config.base import Base, get_db
from app.models.company import Company
from app.models.post import Post
# SQLAlchemy relationship 초기화를 위한 모델 import
from app.models.industry import Industry
from app.models.post_skill import PostSkill
from app.models.position import Position
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill
from app.models.skill import Skill
from datetime import datetime, timedelta
from collections import defaultdict
import calendar
import json


# ==================== 모델 정의 ====================
class RecruitmentSchedule(Base):
    __tablename__ = "recruit_schedule"
    
    schedule_id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey("post.id"))
    company_id = Column(Integer, ForeignKey("company.id"), nullable=False)
    semester = Column(String(20))
    application_date = Column(JSON)
    document_screening_date = Column(JSON)
    first_interview = Column(JSON)
    second_interview = Column(JSON)
    join_date = Column(JSON)
    
    company = relationship("Company", foreign_keys=[company_id])
    post = relationship("Post", foreign_keys=[post_id])


# ==================== 상수 정의 ====================
STAGES = ["application_date", "document_screening_date", "first_interview", "second_interview", "join_date"]
STAGE_NAMES = {
    "application_date": "서류접수",
    "document_screening_date": "서류전형",
    "first_interview": "1차면접",
    "second_interview": "2차면접",
    "join_date": "입사일"
}
DURATIONS = {
    "application_date": 7,
    "document_screening_date": 3,
    "first_interview": 2,
    "second_interview": 2,
    "join_date": 0
}


# ==================== 유틸리티 함수 ====================
def normalize_date(date_str):
    """날짜 정규화"""
    if not date_str:
        return None
    return date_str.replace(".", "-").replace("/", "-")


def get_dates_from_json(date_json):
    """JSON 배열에서 시작/종료 날짜 추출"""
    if not date_json:
        return None, None
    
    dates = []
    for date_range in date_json:
        if date_range and date_range[0]:
            dates.append(date_range[0])
        if date_range and len(date_range) > 1 and date_range[1]:
            dates.append(date_range[1])
    
    if not dates:
        return None, None
    
    return min(dates), max(dates)


def to_pattern(date_str):
    """날짜 → 패턴 (월, 주차, 요일)"""
    if not date_str:
        return None
    
    try:
        normalized = normalize_date(date_str)
        date_obj = datetime.strptime(normalized, "%Y-%m-%d")
        
        cal = calendar.monthcalendar(date_obj.year, date_obj.month)
        week = next((i for i, w in enumerate(cal, 1) if date_obj.day in w), 1)
        
        return {
            "month": date_obj.month,
            "week": week,
            "weekday": date_obj.weekday()
        }
    except:
        return None


def to_date(pattern, year):
    """패턴 → 날짜"""
    if not pattern:
        return None
    
    try:
        cal = calendar.monthcalendar(year, pattern["month"])
        week_idx = min(pattern["week"] - 1, len(cal) - 1)
        day = cal[week_idx][pattern["weekday"]]
        
        if day == 0:
            for week in cal:
                if week[pattern["weekday"]] != 0:
                    day = week[pattern["weekday"]]
                    break
        
        if day == 0:
            return None
        
        return f"{year}-{pattern['month']:02d}-{day:02d}"
    except:
        return None


def avg_pattern(patterns):
    """패턴 평균 계산"""
    if not patterns:
        return None
    
    return {
        "month": round(sum(p["month"] for p in patterns) / len(patterns)),
        "week": round(sum(p["week"] for p in patterns) / len(patterns)),
        "weekday": round(sum(p["weekday"] for p in patterns) / len(patterns))
    }


# ==================== 패턴 추출 ====================
def extract_company_patterns(company_id, db):
    """특정 회사의 과거 채용 패턴 추출"""
    posts = db.query(Post)\
        .options(joinedload(Post.company))\
        .filter(Post.company_id == company_id, Post.experience == "신입")\
        .all()
    
    if not posts:
        return None
    
    company_name = posts[0].company.name if posts[0].company else "Unknown"
    
    # Schedule 조회
    post_ids = [p.id for p in posts]
    schedules = {
        s.post_id: s 
        for s in db.query(RecruitmentSchedule).filter(RecruitmentSchedule.post_id.in_(post_ids)).all()
    }
    
    # 연도별 그룹화
    yearly = defaultdict(lambda: {"patterns": defaultdict(list)})
    
    for post in posts:
        schedule = schedules.get(post.id)
        year = None
        
        # 연도 추출
        if schedule and schedule.application_date:
            start, _ = get_dates_from_json(schedule.application_date)
            if start:
                try:
                    year = datetime.strptime(normalize_date(start), "%Y-%m-%d").year
                except:
                    pass
        
        if not year and post.posted_at:
            year = post.posted_at.year
        
        if not year:
            continue
        
        # 각 단계별 패턴 추출
        for stage in STAGES:
            if schedule:
                date_json = getattr(schedule, stage, None)
                if date_json:
                    start, _ = get_dates_from_json(date_json)
                    if start:
                        pattern = to_pattern(start)
                        if pattern:
                            yearly[year]["patterns"][stage].append(pattern)
            
            # Schedule 없으면 Post에서
            if stage == "application_date" and post.posted_at and not yearly[year]["patterns"][stage]:
                pattern = to_pattern(post.posted_at.strftime("%Y-%m-%d"))
                if pattern:
                    yearly[year]["patterns"][stage].append(pattern)
    
    if not yearly:
        return None
    
    # 연도별 평균 패턴 계산
    result = {}
    for year, data in yearly.items():
        avg_patterns = {}
        for stage, patterns in data["patterns"].items():
            if patterns:
                avg_patterns[stage] = avg_pattern(patterns)
        
        if avg_patterns:
            result[year] = avg_patterns
    
    if not result:
        return None
    
    return {
        "company_id": company_id,
        "company_name": company_name,
        "patterns": result
    }


# ==================== 일정 예측 ====================
def predict_schedule(patterns, target_year, target_semester):
    """패턴 기반 채용 일정 예측"""
    if not patterns or not patterns.get("patterns"):
        return None
    
    # 최신 연도 패턴 사용
    latest_year = max(patterns["patterns"].keys())
    latest_patterns = patterns["patterns"][latest_year]
    
    predicted = {}
    prev_date = None
    
    for stage in STAGES:
        if stage not in latest_patterns:
            continue
        
        pattern = latest_patterns[stage]
        date_str = to_date(pattern, target_year)
        
        if not date_str:
            continue
        
        # 날짜 순서 검증
        try:
            curr_date = datetime.strptime(date_str, "%Y-%m-%d")
            if prev_date and curr_date < prev_date:
                continue
            prev_date = curr_date
        except:
            continue
        
        # 종료일 계산
        try:
            end_date = (curr_date + timedelta(days=DURATIONS.get(stage, 1))).strftime("%Y-%m-%d")
        except:
            end_date = date_str
        
        predicted[stage] = {
            "start_date": date_str,
            "end_date": end_date
        }
    
    if not predicted:
        return None
    
    return {
        "company_id": patterns["company_id"],
        "company_name": patterns["company_name"],
        "stages": predicted
    }


# ==================== Swagger 형식 변환 ====================
def format_swagger_response(predictions):
    """Swagger UI 응답 형식 변환"""
    schedules = []
    
    for pred in predictions:
        stages = []
        stage_id_counter = 1
        
        for stage, dates in pred["stages"].items():
            stages.append({
                "id": f"{pred['company_id']}-{stage_id_counter}",
                "stage": STAGE_NAMES[stage],
                "start_date": dates["start_date"],
                "end_date": dates["end_date"]
            })
            stage_id_counter += 1
        
        schedules.append({
            "id": str(pred["company_id"]),
            "company_id": pred["company_id"],
            "company_name": pred["company_name"],
            "type": "신입",
            "data_type": "predicted",
            "stages": stages
        })
    
    return {
        "status": 200,
        "code": "SUCCESS",
        "message": "회사별 채용 일정 조회 성공",
        "data": {
            "schedules": schedules
        }
    }


# ==================== 경쟁사 조회 ====================
def get_competitors(db):
    """경쟁사 목록 조회"""
    from app.db.crud.db_competitors_skills import COMPETITOR_GROUPS
    
    recruiting_companies = ["네이버", "카카오", "현대오토에버", "한화시스템", "LG CNS"]
    like_conditions = []
    
    for company_name in recruiting_companies:
        if company_name in COMPETITOR_GROUPS:
            for keyword in COMPETITOR_GROUPS[company_name]:
                like_conditions.append(Company.name.like(keyword))
    
    return db.query(Company.id, Company.name)\
        .filter(or_(*like_conditions))\
        .order_by(Company.id)\
        .all()


# ==================== 메인 ====================
def main():
    print("\n" + "=" * 60)
    print("채용 일정 패턴 추출 및 예측")
    print("=" * 60)
    
    db = next(get_db())
    
    try:
        # 인자 파싱
        target_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
        target_semester = sys.argv[2] if len(sys.argv) > 2 else "상반기"
        
        print(f"\n[설정]")
        print(f"  - 예측 연도: {target_year}")
        print(f"  - 예측 학기: {target_semester}")
        
        # 경쟁사 조회
        companies = get_competitors(db)
        print(f"\n[경쟁사 조회] 총 {len(companies)}개 회사")
        
        # 패턴 추출 및 예측
        predictions = []
        for company_id, company_name in companies:
            patterns = extract_company_patterns(company_id, db)
            if patterns:
                prediction = predict_schedule(patterns, target_year, target_semester)
                if prediction:
                    predictions.append(prediction)
                    print(f"  - {company_name}: {len(prediction['stages'])}개 단계 예측")
        
        # Swagger 형식으로 결과 반환
        result = format_swagger_response(predictions)
        
        print("\n" + "-" * 60)
        print("결과")
        print("-" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        print(f"\n총 {len(predictions)}개 회사 예측 완료!")
    
    except Exception as e:
        print(f"\n오류: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()


if __name__ == "__main__":
    main()