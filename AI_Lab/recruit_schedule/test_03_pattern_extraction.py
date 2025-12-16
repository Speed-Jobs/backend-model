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
    """
    날짜 → 패턴 (월, N번째 해당 요일, 요일)

    예: 2024.07.01 (월) → {month: 7, nth: 1, weekday: 0} (7월의 1번째 월요일)
    """
    if not date_str:
        return None

    try:
        normalized = normalize_date(date_str)
        date_obj = datetime.strptime(normalized, "%Y-%m-%d")

        # 해당 월의 해당 요일이 몇 번째인지 계산
        # 예: 7월 1일(월) → (1-1)//7 + 1 = 1번째 월요일
        #     7월 8일(월) → (8-1)//7 + 1 = 2번째 월요일
        nth = (date_obj.day - 1) // 7 + 1

        return {
            "month": date_obj.month,
            "nth": nth,  # N번째 해당 요일
            "weekday": date_obj.weekday()
        }
    except:
        return None


def to_date(pattern, year):
    """
    패턴 → 날짜 (N번째 해당 요일 방식)

    예: {month: 7, nth: 1, weekday: 0} + year=2026
        → 2026년 7월의 1번째 월요일 = 2026-07-06
    """
    if not pattern:
        return None

    try:
        month = pattern["month"]
        nth = pattern["nth"]
        weekday = pattern["weekday"]

        # 해당 월의 첫 날
        first_day = datetime(year, month, 1)

        # 첫 번째 해당 요일까지의 일수 계산
        days_until_weekday = (weekday - first_day.weekday()) % 7

        # 첫 번째 해당 요일
        first_occurrence = first_day + timedelta(days=days_until_weekday)

        # N번째 해당 요일
        target_date = first_occurrence + timedelta(weeks=nth - 1)

        # 월을 넘어가면 None (5번째 월요일이 없는 경우 등)
        if target_date.month != month:
            return None

        return target_date.strftime("%Y-%m-%d")
    except:
        return None


def avg_pattern(patterns):
    """패턴 평균 계산 (N번째 해당 요일 방식)"""
    if not patterns:
        return None

    return {
        "month": round(sum(p["month"] for p in patterns) / len(patterns)),
        "nth": round(sum(p["nth"] for p in patterns) / len(patterns)),  # N번째 해당 요일
        "weekday": round(sum(p["weekday"] for p in patterns) / len(patterns))
    }


def avg_semester_patterns(pattern_list):
    """
    동일 반기 내 여러 공고의 패턴 평균 계산

    Args:
        pattern_list: [{"start_pattern": {...}, "end_pattern": {...}}, ...]

    Returns:
        {"start_pattern": {...}, "end_pattern": {...}}
    """
    if not pattern_list:
        return None

    # start_pattern 평균
    start_patterns = [p["start_pattern"] for p in pattern_list]
    avg_start = avg_pattern(start_patterns)

    # end_pattern 평균
    end_patterns = [p["end_pattern"] for p in pattern_list]
    avg_end = avg_pattern(end_patterns)

    return {
        "start_pattern": avg_start,
        "end_pattern": avg_end
    }


def determine_semester(date_str):
    """
    날짜로부터 반기 결정

    Args:
        date_str: "YYYY-MM-DD" 형식 날짜

    Returns:
        "YYYY_상반기" or "YYYY_하반기"
    """
    if not date_str:
        return None

    try:
        date_obj = datetime.strptime(normalize_date(date_str), "%Y-%m-%d")
        year = date_obj.year
        semester = "상반기" if date_obj.month <= 6 else "하반기"
        return f"{year}_{semester}"
    except:
        return None


def has_all_required_stages(schedule):
    """
    필수 단계가 모두 있는지 확인

    필수 단계: application, document_screening, first_interview, second_interview
    (join_date는 선택)

    Args:
        schedule: RecruitmentSchedule 객체

    Returns:
        bool
    """
    required_stages = [
        "application_date",
        "document_screening_date",
        "first_interview",
        "second_interview"
    ]

    for stage in required_stages:
        date_json = getattr(schedule, stage, None)
        if not date_json or len(date_json) == 0:
            return False

    return True


def aggregate_all_matching_semesters(semester_data, target_semester):
    """
    target_semester와 동일한 반기의 모든 완전한 데이터를 평균화

    수정 사항:
    - 모든 과거 동일 반기 데이터를 사용 (최신 하나가 아님)
    - 각 stage별로 모든 년도의 start_pattern과 end_pattern을 모아서 평균화

    Args:
        semester_data: 전체 반기 데이터 {"2024_하반기": {...}, "2023_하반기": {...}, ...}
        target_semester: "상반기" or "하반기"

    Returns:
        평균화된 패턴 dict or None
    """
    # 해당 반기만 필터링 (필수 단계 검증)
    required_stages = [
        "application_date",
        "document_screening_date",
        "first_interview",
        "second_interview"
    ]

    matching_patterns = []
    for key, patterns in semester_data.items():
        if key.endswith(target_semester):
            # 모든 필수 단계가 있는지 확인
            if all(stage in patterns and patterns[stage] is not None for stage in required_stages):
                matching_patterns.append(patterns)

    if not matching_patterns:
        return None

    # 모든 년도의 패턴을 stage별로 평균화
    result = {}
    for stage in STAGES:
        all_start_patterns = []
        all_end_patterns = []

        for pattern_dict in matching_patterns:
            if stage in pattern_dict and pattern_dict[stage]:
                all_start_patterns.append(pattern_dict[stage]["start_pattern"])
                all_end_patterns.append(pattern_dict[stage]["end_pattern"])

        if all_start_patterns and all_end_patterns:
            result[stage] = {
                "start_pattern": avg_pattern(all_start_patterns),
                "end_pattern": avg_pattern(all_end_patterns)
            }

    return result if result else None


# ==================== 패턴 추출 ====================
def extract_company_patterns(company_id, db):
    """
    특정 회사의 과거 채용 패턴 추출 (반기별 그룹화)

    수정 사항:
    - 반기별 그룹화 (상반기/하반기 구분)
    - 필수 단계 null 체크 (application, document, first, second 모두 필수)
    - start_pattern과 end_pattern 각각 패턴화

    Returns:
        {
            "company_id": int,
            "company_name": str,
            "patterns": {
                "2024_하반기": {
                    "application_date": {"start_pattern": {...}, "end_pattern": {...}},
                    ...
                },
                ...
            }
        }
    """
    # RecruitmentSchedule에서 직접 조회 (application_date 필수)
    schedules = db.query(RecruitmentSchedule)\
        .options(joinedload(RecruitmentSchedule.company))\
        .filter(
            RecruitmentSchedule.company_id == company_id,
            RecruitmentSchedule.application_date.isnot(None)
        )\
        .all()

    if not schedules:
        return None

    # 회사명 조회
    company_name = schedules[0].company.name if schedules[0].company else "Unknown"

    # 반기별 그룹화
    semester_data = defaultdict(lambda: {"patterns": defaultdict(list)})

    for schedule in schedules:
        # 필수 단계 null 체크
        if not has_all_required_stages(schedule):
            continue

        # 반기 결정
        start, _ = get_dates_from_json(schedule.application_date)
        semester_key = determine_semester(start)

        if not semester_key:
            continue

        # 각 단계별 start/end 패턴 추출
        for stage in STAGES:
            date_json = getattr(schedule, stage, None)
            if not date_json:
                continue

            start, end = get_dates_from_json(date_json)
            if not start or not end:
                continue

            start_pattern = to_pattern(start)
            end_pattern = to_pattern(end)

            if not start_pattern or not end_pattern:
                continue

            semester_data[semester_key]["patterns"][stage].append({
                "start_pattern": start_pattern,
                "end_pattern": end_pattern
            })

    if not semester_data:
        return None

    # 반기별 평균 패턴 계산
    result = {}
    for semester_key, data in semester_data.items():
        avg_patterns = {}
        for stage, pattern_list in data["patterns"].items():
            if pattern_list:
                avg_patterns[stage] = avg_semester_patterns(pattern_list)

        if avg_patterns:
            result[semester_key] = avg_patterns

    if not result:
        return None

    return {
        "company_id": company_id,
        "company_name": company_name,
        "patterns": result
    }


# ==================== 일정 예측 ====================
def predict_schedule(patterns, target_year, target_semester):
    """
    패턴 기반 채용 일정 예측 (반기별)

    Args:
        patterns: extract_company_patterns의 결과
        target_year: 예측 대상 연도 (2026)
        target_semester: "상반기" or "하반기"

    Returns:
        {
            "company_id": int,
            "company_name": str,
            "stages": {
                "application_date": {"start_date": "2026-03-03", "end_date": "2026-03-10"},
                ...
            }
        }
    """
    if not patterns or not patterns.get("patterns"):
        return None

    # 최신 완전한 반기 찾기
    semester_key, semester_patterns = find_latest_complete_semester(
        patterns["patterns"],
        target_semester
    )

    if not semester_key or not semester_patterns:
        return None

    predicted = {}

    for stage in STAGES:
        if stage not in semester_patterns:
            continue

        pattern_data = semester_patterns[stage]
        if not pattern_data:
            continue

        # start_pattern, end_pattern 각각 예측
        start_date = to_date(pattern_data["start_pattern"], target_year)
        end_date = to_date(pattern_data["end_pattern"], target_year)

        if not start_date or not end_date:
            continue

        # 날짜 순서 검증 및 조정 (단순)
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            # 종료일이 시작일보다 빠르면 다음 주로 조정
            if end_dt < start_dt:
                end_dt = end_dt + timedelta(weeks=1)
                end_date = end_dt.strftime("%Y-%m-%d")
        except:
            pass

        predicted[stage] = {
            "start_date": start_date,
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
    from app.config.company_groups import COMPANY_GROUPS

    # COMPANY_GROUPS의 모든 그룹 패턴 사용
    like_conditions = []
    for patterns in COMPANY_GROUPS.values():
        for pattern in patterns:
            like_conditions.append(Company.name.like(pattern))

    if not like_conditions:
        return []

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
                prediction = predict_schedule(patterns, target_year, target_semester, include_sources=True)
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