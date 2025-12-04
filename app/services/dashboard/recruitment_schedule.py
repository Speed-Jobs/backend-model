"""
Recruitment Schedule Service
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
from app.db.crud.db_recruitment_schedule import (
    get_recruitment_schedules_by_company,
    get_recruitment_schedules
)
from app.models.recruitment_schedule import RecruitmentSchedule


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
                "id": f"{schedule.schedule_id}-{stage_id_counter}",
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
    end_date: str
) -> List[Dict[str, Any]]:
    """
    채용 일정을 필터링하고 Swagger 형식으로 변환합니다.
    
    Args:
        schedules: RecruitmentSchedule 객체 리스트
        type_filter: "신입" 또는 "경력"
        start_date: 조회 시작일 (YYYY-MM-DD)
        end_date: 조회 종료일 (YYYY-MM-DD)
        
    Returns:
        변환된 일정 리스트
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
        
        # application_date가 없으면 제외
        if not schedule.application_date or len(schedule.application_date) == 0:
            continue
        
        # stages 변환
        stages = convert_to_stages(schedule, start_date, end_date)
        
        # stages가 비어있으면 제외
        if not stages:
            continue
        
        # 회사 정보
        company_name = schedule.company.name if schedule.company else "Unknown"
        
        # Swagger 형식으로 변환
        schedule_data = {
            "id": str(schedule.schedule_id),
            "company_id": schedule.company_id,
            "company_name": company_name,
            "type": type_filter,
            "data_type": "actual",
            "stages": stages
        }
        
        result_schedules.append(schedule_data)
    
    return result_schedules


def get_company_recruitment_schedule(
    db: Session,
    company_id: int,
    type_filter: str,
    start_date: str,
    end_date: str
) -> Dict[str, Any]:
    """
    특정 회사의 채용 일정을 조회합니다.
    
    Args:
        db: Database session
        company_id: 회사 ID
        type_filter: "신입" 또는 "경력"
        start_date: 조회 시작일 (YYYY-MM-DD)
        end_date: 조회 종료일 (YYYY-MM-DD)
        
    Returns:
        Swagger 형식의 응답 딕셔너리
    """
    try:
        # CRUD로 데이터 조회
        schedules = get_recruitment_schedules_by_company(
            db=db,
            company_id=company_id
        )
        
        # 필터링 및 변환
        result_schedules = filter_and_convert_schedules(
            schedules=schedules,
            type_filter=type_filter,
            start_date=start_date,
            end_date=end_date
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


def get_all_recruitment_schedules(
    db: Session,
    type_filter: str,
    start_date: str,
    end_date: str,
    company_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    전체 또는 특정 회사들의 채용 일정을 조회합니다.
    
    Args:
        db: Database session
        type_filter: "신입" 또는 "경력"
        start_date: 조회 시작일 (YYYY-MM-DD)
        end_date: 조회 종료일 (YYYY-MM-DD)
        company_ids: 회사 ID 리스트 (None이면 전체)
        
    Returns:
        Swagger 형식의 응답 딕셔너리
    """
    try:
        # CRUD로 데이터 조회
        schedules = get_recruitment_schedules(
            db=db,
            company_ids=company_ids
        )
        
        # 필터링 및 변환
        result_schedules = filter_and_convert_schedules(
            schedules=schedules,
            type_filter=type_filter,
            start_date=start_date,
            end_date=end_date
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

