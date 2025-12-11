"""
Recruitment Schedule Service
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_
from datetime import datetime, timedelta
from collections import defaultdict
import calendar
from app.db.crud.db_recruitment_schedule import (
    get_recruitment_schedules,
    get_competition_intensity_analysis
)
from app.config.company_groups import get_company_patterns
from app.models.recruitment_schedule import RecruitmentSchedule
from app.models.company import Company
from app.models.post import Post
# SQLAlchemy relationship 초기화를 위한 모델 import
from app.models.industry import Industry
from app.models.post_skill import PostSkill
from app.models.position import Position
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill
from app.models.skill import Skill

# ==================== 예측 관련 상수 ====================
PREDICTION_STAGES = ["application_date", "document_screening_date", "first_interview", "second_interview", "join_date"]
PREDICTION_STAGE_NAMES = {
    "application_date": "서류접수",
    "document_screening_date": "서류전형",
    "first_interview": "1차면접",
    "second_interview": "2차면접",
    "join_date": "입사일"
}
PREDICTION_DURATIONS = {
    "application_date": 7,
    "document_screening_date": 3,
    "first_interview": 2,
    "second_interview": 2,
    "join_date": 0
}


def parse_date_range(date_array: list) -> tuple:
    """
    JSON 날짜 배열을 파싱하여 (start_date, end_date)를 반환합니다.
    여러 개의 날짜 범위가 있으면 전체 범위(최소 ~ 최대)를 계산합니다.
    """
    if not date_array or len(date_array) == 0:
        return (None, None)
    
    all_dates = []
    for date_range in date_array:
        if len(date_range) < 2:
            continue
        
        if date_range[0]:
            all_dates.append(date_range[0])
        
        if date_range[1]:
            all_dates.append(date_range[1])
        elif date_range[0]:
            all_dates.append(date_range[0])
    
    if not all_dates:
        return (None, None)
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    return (start_date, end_date)


def is_date_in_range(date_str: str, start_filter: str, end_filter: str) -> bool:
    """날짜가 필터 범위 내에 있는지 확인합니다."""
    if not date_str:
        return False
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        start_obj = datetime.strptime(start_filter, "%Y-%m-%d")
        end_obj = datetime.strptime(end_filter, "%Y-%m-%d")
        return start_obj <= date_obj <= end_obj
    except:
        return False


def convert_to_stages(
    schedule: RecruitmentSchedule,
    start_filter: str,
    end_filter: str
) -> List[Dict[str, Any]]:
    """RecruitmentSchedule의 JSON 날짜 필드들을 stages 배열로 변환합니다."""
    stages = []
    stage_id_counter = 1
    
    stage_mapping = [
        (schedule.application_date, "서류접수"),
        (schedule.document_screening_date, "서류전형"),
        (schedule.first_interview, "1차 면접"),
        (schedule.second_interview, "2차 면접"),
        (schedule.join_date, "입사일"),
    ]
    
    for date_field, stage_name in stage_mapping:
        start_date, end_date = parse_date_range(date_field)
        
        if start_date and is_date_in_range(start_date, start_filter, end_filter):
            stages.append({
                "id": f"{schedule.company_id}-{stage_id_counter}",
                "stage": stage_name,
                "start_date": start_date,
                "end_date": end_date
            })
            stage_id_counter += 1
    
    return stages


def filter_and_convert_schedules(
    schedules: List[RecruitmentSchedule],
    type_filter: str,
    start_date: str,
    end_date: str,
    position_ids: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    채용 일정을 필터링하고 Swagger 형식으로 변환합니다.
    각 schedule은 별도로 반환됩니다 (다른 공고는 분리).

    Args:
        schedules: RecruitmentSchedule 객체 리스트
        type_filter: "신입" 또는 "경력"
        start_date: 조회 시작일 (YYYY-MM-DD)
        end_date: 조회 종료일 (YYYY-MM-DD)
        position_ids: 직군 ID 리스트 (None이면 전체)

    Returns:
        변환된 일정 리스트 (각 schedule별로 분리됨)
    """
    result_schedules = []

    for schedule in schedules:
        # post가 없거나 experience가 null이면 제외
        if not schedule.post or not schedule.post.experience:
            continue

        # 신입 필터: 정확히 "신입"만 통과
        if type_filter == "신입":
            if schedule.post.experience != "신입":
                continue
        # 경력 필터: "신입"이 아닌 모든 것 통과
        elif type_filter == "경력":
            if schedule.post.experience == "신입":
                continue

        # position_ids 필터 (메모리 필터 - 쿼리에서 못 거른 경우 대비)
        position_id = None
        if schedule.post and schedule.post.industry:
            position_id = schedule.post.industry.position_id
            if position_ids and position_id not in position_ids:
                continue

        # application_date가 없으면 제외
        if not schedule.application_date or len(schedule.application_date) == 0:
            continue

        # stages 변환
        stages = convert_to_stages(schedule, start_date, end_date)

        # stages가 비어있으면 제외
        if not stages:
            continue

        # 회사 정보
        company_id = schedule.company_id
        company_name = schedule.company.name if schedule.company else "Unknown"

        # 각 schedule을 별도로 추가 (병합 없음)
        schedule_data = {
            "id": str(company_id),
            "company_id": company_id,
            "company_name": company_name,
            "type": type_filter,
            "position_id": position_id,
            "data_type": "actual",
            "stages": stages
        }

        result_schedules.append(schedule_data)

    return result_schedules


# ==================== 예측 관련 유틸리티 함수 ====================
def normalize_date(date_str: str) -> Optional[str]:
    """날짜 정규화"""
    if not date_str:
        return None
    return date_str.replace(".", "-").replace("/", "-")


def get_dates_from_json_for_prediction(date_json: list) -> tuple:
    """JSON 배열에서 시작/종료 날짜 추출 (예측용)"""
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


def to_pattern(date_str: str) -> Optional[Dict[str, int]]:
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


def to_date(pattern: Dict[str, int], year: int) -> Optional[str]:
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


def avg_pattern(patterns: List[Dict[str, int]]) -> Optional[Dict[str, int]]:
    """패턴 평균 계산"""
    if not patterns:
        return None
    
    return {
        "month": round(sum(p["month"] for p in patterns) / len(patterns)),
        "week": round(sum(p["week"] for p in patterns) / len(patterns)),
        "weekday": round(sum(p["weekday"] for p in patterns) / len(patterns))
    }


# ==================== 패턴 추출 ====================
def extract_company_patterns(company_id: int, db: Session, type_filter: str) -> Optional[Dict[str, Any]]:
    """특정 회사의 과거 채용 패턴 추출"""
    posts = db.query(Post)\
        .options(joinedload(Post.company))\
        .filter(Post.company_id == company_id, Post.experience == type_filter)\
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
            start, _ = get_dates_from_json_for_prediction(schedule.application_date)
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
        for stage in PREDICTION_STAGES:
            if schedule:
                date_json = getattr(schedule, stage, None)
                if date_json:
                    start, _ = get_dates_from_json_for_prediction(date_json)
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
def predict_schedule(patterns: Dict[str, Any], target_year: int) -> Optional[Dict[str, Any]]:
    """패턴 기반 채용 일정 예측"""
    if not patterns or not patterns.get("patterns"):
        return None
    
    # 최신 연도 패턴 사용
    latest_year = max(patterns["patterns"].keys())
    latest_patterns = patterns["patterns"][latest_year]
    
    predicted = {}
    prev_date = None
    
    for stage in PREDICTION_STAGES:
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
            end_date = (curr_date + timedelta(days=PREDICTION_DURATIONS.get(stage, 1))).strftime("%Y-%m-%d")
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


def format_predicted_response(predictions: List[Dict[str, Any]], type_filter: str) -> List[Dict[str, Any]]:
    """예측 결과를 Swagger 형식으로 변환"""
    schedules = []
    
    for pred in predictions:
        stages = []
        stage_id_counter = 1
        
        for stage, dates in pred["stages"].items():
            stages.append({
                "id": f"{pred['company_id']}-{stage_id_counter}",
                "stage": PREDICTION_STAGE_NAMES[stage],
                "start_date": dates["start_date"],
                "end_date": dates["end_date"]
            })
            stage_id_counter += 1
        
        schedules.append({
            "id": str(pred["company_id"]),
            "company_id": pred["company_id"],
            "company_name": pred["company_name"],
            "type": type_filter,
            "data_type": "predicted",
            "stages": stages
        })
    
    return schedules


def get_predicted_schedules(
    db: Session,
    type_filter: str,
    start_date: str,
    end_date: str,
    company_keywords: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    예측된 채용 일정을 조회합니다.

    Args:
        db: Database session
        type_filter: "신입" 또는 "경력"
        start_date: 조회 시작일 (YYYY-MM-DD) - 예측 연도 결정에 사용
        end_date: 조회 종료일 (YYYY-MM-DD) - 예측 연도 결정에 사용
        company_keywords: 회사 키워드 리스트 (None이면 경쟁사 전체, 예: ["toss", "kakao"])

    Returns:
        예측된 일정 리스트
    """
    try:
        # 예측 연도 결정 (start_date의 연도 사용)
        target_year = datetime.strptime(start_date, "%Y-%m-%d").year

        # 조회할 회사 결정
        if company_keywords:
            # 키워드를 패턴으로 변환
            all_patterns = []
            for keyword in company_keywords:
                patterns = get_company_patterns(keyword)
                all_patterns.extend(patterns)

            # 패턴으로 회사 조회
            companies = db.query(Company.id, Company.name)\
                .filter(or_(*[Company.name.like(pattern) for pattern in all_patterns]))\
                .order_by(Company.id)\
                .all()
        else:
            # 경쟁사 전체 조회
            from app.db.crud.db_recruitment_schedule import get_competitor_companies
            companies = get_competitor_companies(db)
        
        # 패턴 추출 및 예측
        predictions = []
        for company_id, company_name in companies:
            patterns = extract_company_patterns(company_id, db, type_filter)
            if patterns:
                prediction = predict_schedule(patterns, target_year)
                if prediction:
                    # 날짜 필터링 (start_date ~ end_date 범위 내의 예측만 포함)
                    filtered_stages = {}
                    for stage, dates in prediction["stages"].items():
                        pred_start = dates["start_date"]
                        if is_date_in_range(pred_start, start_date, end_date):
                            filtered_stages[stage] = dates
                    
                    if filtered_stages:
                        prediction["stages"] = filtered_stages
                        predictions.append(prediction)
        
        # Swagger 형식으로 변환
        return format_predicted_response(predictions, type_filter)
    
    except Exception as e:
        return []


def merge_actual_and_predicted(
    actual_schedules: List[Dict[str, Any]],
    predicted_schedules: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    actual과 predicted 일정을 병합합니다.
    data_type="all"일 때는 같은 회사라도 actual과 predicted를 구분해서 반환합니다.
    
    Args:
        actual_schedules: 실제 일정 리스트
        predicted_schedules: 예측 일정 리스트
        
    Returns:
        병합된 일정 리스트 (actual과 predicted가 구분됨)
    """
    # actual과 predicted를 모두 포함 (같은 회사라도 data_type이 다르면 별도 항목)
    result = []
    
    # actual 추가
    result.extend(actual_schedules)
    
    # predicted 추가 (별도 항목으로)
    result.extend(predicted_schedules)
    
    return result

def get_all_recruitment_schedules(
    db: Session,
    type_filter: str,
    start_date: str,
    end_date: str,
    company_keywords: Optional[List[str]] = None,
    data_type: str = "actual"
) -> Dict[str, Any]:
    """
    전체 또는 특정 회사들의 채용 일정을 조회합니다.

    Args:
        db: Database session
        type_filter: "신입" 또는 "경력"
        start_date: 조회 시작일 (YYYY-MM-DD)
        end_date: 조회 종료일 (YYYY-MM-DD)
        company_keywords: 회사 키워드 리스트 (None이면 전체, 예: ["toss", "kakao"])
        data_type: "actual", "predicted", "all"

    Returns:
        Swagger 형식의 응답 딕셔너리
    """
    try:
        result_schedules = []

        # 키워드를 패턴으로 변환
        company_patterns = None
        if company_keywords:
            company_patterns = []
            for keyword in company_keywords:
                patterns = get_company_patterns(keyword)
                company_patterns.extend(patterns)

        # actual 데이터 조회
        if data_type in ["actual", "all"]:
            schedules = get_recruitment_schedules(
                db=db,
                company_patterns=company_patterns
            )

            actual_schedules = filter_and_convert_schedules(
                schedules=schedules,
                type_filter=type_filter,
                start_date=start_date,
                end_date=end_date
            )

            if data_type == "actual":
                result_schedules = actual_schedules
            else:
                result_schedules = actual_schedules

        # predicted 데이터 조회 (경력은 예측 불가)
        if data_type in ["predicted", "all"] and type_filter == "신입":
            predicted_schedules = get_predicted_schedules(
                db=db,
                type_filter=type_filter,
                start_date=start_date,
                end_date=end_date,
                company_keywords=company_keywords
            )

            if data_type == "predicted":
                result_schedules = predicted_schedules
            else:
                # all인 경우 병합
                result_schedules = merge_actual_and_predicted(
                    result_schedules,
                    predicted_schedules
                )

        return {
            "status": 200,
            "code": "SUCCESS",
            "message": "회사별 채용 일정 조회 성공",
            "data": {
                "schedules": result_schedules
            }
        }

    except Exception as e:
        return {
            "status": 500,
            "code": "INTERNAL_ERROR",
            "message": f"오류 발생: {str(e)}",
            "data": None
        }


def get_competition_intensity(
    db: Session,
    start_date: str,
    end_date: str,
    type_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    날짜별 경쟁 강도 분석

    Args:
        db: Database session
        start_date: 분석 시작일 (YYYY-MM-DD)
        end_date: 분석 종료일 (YYYY-MM-DD)
        type_filter: 채용 유형 필터 ("신입", "경력", None)

    Returns:
        Swagger 형식의 응답 딕셔너리
    """
    try:
        return get_competition_intensity_analysis(
            db=db,
            start_date=start_date,
            end_date=end_date,
            type_filter=type_filter
        )
    except Exception as e:
        return {
            "status": 500,
            "code": "INTERNAL_ERROR",
            "message": f"오류 발생: {str(e)}",
            "data": None
        }

