import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import relationship, joinedload
from app.db.config.base import Base, get_db
from app.models.company import Company
from app.models.industry import Industry
from app.models.post import Post
from app.models.post_skill import PostSkill
from app.models.position import Position
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill
from app.models.skill import Skill
from datetime import datetime
import json


# RecruitmentSchedule 모델 정의 (실제 DB 테이블 구조)
class RecruitmentSchedule(Base):
    __tablename__ = "recruit_schedule"

    schedule_id = Column(Integer, primary_key=True, comment="일정 ID")
    post_id = Column(Integer, ForeignKey("post.id"), nullable=True, comment="공고 ID")
    company_id = Column(Integer, ForeignKey("company.id"), nullable=False, comment="회사 ID")
    industry_id = Column(Integer, ForeignKey("industry.id"), nullable=True, comment="산업 ID")
    semester = Column(String(20), nullable=True, comment="학기 (상반기/하반기)")
    application_date = Column(JSON, nullable=True, comment="지원서 접수 기간 (JSON 배열)")
    document_screening_date = Column(JSON, nullable=True, comment="서류 전형 기간 (JSON 배열)")
    first_interview = Column(JSON, nullable=True, comment="1차 면접 기간 (JSON 배열)")
    second_interview = Column(JSON, nullable=True, comment="2차 면접 기간 (JSON 배열)")
    join_date = Column(JSON, nullable=True, comment="입사일 (JSON 배열)")

    # Relationships
    company = relationship("Company", foreign_keys=[company_id])
    industry = relationship("Industry", foreign_keys=[industry_id])
    post = relationship("Post", foreign_keys=[post_id])




def parse_date_range(date_array: list) -> tuple:
    """
    JSON 날짜 배열을 파싱하여 (start_date, end_date)를 반환합니다.
    여러 개의 날짜 범위가 있으면 전체 범위(최소 ~ 최대)를 계산합니다.
    
    입력 예시:
    - [["2025-11-18", "2025-11-24"]] → ("2025-11-18", "2025-11-24")
    - [["2025-10-11", "2025-10-11"], ["2025-10-26", "2025-10-26"]] 
      → ("2025-10-11", "2025-10-26")  # 최소 ~ 최대
    - [["2025-11-18", null]] → ("2025-11-18", "2025-11-18")
    - [] → (None, None)
    
    Args:
        date_array: JSON 날짜 배열 (예: [["2025-11-18", "2025-11-24"]])
        
    Returns:
        (start_date, end_date) 튜플 또는 (None, None)
    """
    if not date_array or len(date_array) == 0:
        return (None, None)
    
    # 모든 날짜를 수집
    all_dates = []
    for date_range in date_array:
        if len(date_range) < 2:
            continue
        
        # 시작일 추가
        if date_range[0]:
            all_dates.append(date_range[0])
        
        # 종료일 추가 (null이 아닌 경우만)
        if date_range[1]:
            all_dates.append(date_range[1])
        elif date_range[0]:
            # 종료일이 null이면 시작일 사용
            all_dates.append(date_range[0])
    
    if not all_dates:
        return (None, None)
    
    # 최소 날짜 ~ 최대 날짜 반환
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    return (start_date, end_date)


def is_date_in_range(date_str: str, start_filter: str, end_filter: str) -> bool:
    """
    날짜가 필터 범위 내에 있는지 확인합니다.
    
    Args:
        date_str: 확인할 날짜 (YYYY-MM-DD)
        start_filter: 필터 시작일 (YYYY-MM-DD)
        end_filter: 필터 종료일 (YYYY-MM-DD)
        
    Returns:
        범위 내에 있으면 True, 아니면 False
    """
    if not date_str:
        return False
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        start_obj = datetime.strptime(start_filter, "%Y-%m-%d")
        end_obj = datetime.strptime(end_filter, "%Y-%m-%d")
        return start_obj <= date_obj <= end_obj
    except:
        return False


def convert_to_stages(schedule: RecruitmentSchedule, start_filter: str, end_filter: str) -> list:
    """
    RecruitmentSchedule의 JSON 날짜 필드들을 stages 배열로 변환합니다.
    필터 범위 내의 날짜만 포함합니다.
    
    반환 형식:
    [
        {"id": "1-1", "stage": "서류접수", "start_date": "2025-11-18", "end_date": "2025-11-24"},
        {"id": "1-2", "stage": "1차 면접", "start_date": "2025-11-25", "end_date": "2025-11-25"}
    ]
    
    Args:
        schedule: RecruitmentSchedule 객체
        start_filter: 필터 시작일 (YYYY-MM-DD)
        end_filter: 필터 종료일 (YYYY-MM-DD)
        
    Returns:
        stages 배열 (빈 배열일 수 있음)
    """
    stages = []
    stage_id_counter = 1
    
    # 각 날짜 필드를 stage로 변환
    stage_mapping = [
        (schedule.application_date, "서류접수"),
        (schedule.document_screening_date, "서류전형"),
        (schedule.first_interview, "1차 면접"),
        (schedule.second_interview, "2차 면접"),
        (schedule.join_date, "입사일"),
    ]
    
    for date_field, stage_name in stage_mapping:
        start_date, end_date = parse_date_range(date_field)
        
        # 날짜가 있고, 필터 범위 내에 있는 경우만 추가
        if start_date and is_date_in_range(start_date, start_filter, end_filter):
            stages.append({
                "id": f"{schedule.schedule_id}-{stage_id_counter}",
                "stage": stage_name,
                "start_date": start_date,
                "end_date": end_date
            })
            stage_id_counter += 1
    
    return stages


def test_swagger_format(
    type_filter: str = "신입",
    data_type_filter: str = "actual",
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    company_ids: list = None,
    position_ids: list = None
):
    """
    DB 데이터를 Swagger 형식으로 변환하여 출력합니다.

    Args:
        type_filter: "신입" 또는 "경력"
        data_type_filter: "actual", "predicted", "all" (현재는 actual만 지원)
        start_date: 조회 시작일 (YYYY-MM-DD)
        end_date: 조회 종료일 (YYYY-MM-DD)
        company_ids: 회사 ID 리스트 (None이면 전체)
        position_ids: 직군 ID 리스트 (None이면 전체)

    Returns:
        Swagger 형식의 응답 딕셔너리
    """
    print("=" * 60)
    print("테스트 2: Swagger 형식 변환")
    print("=" * 60)
    print(f"\n[필터 조건]")
    print(f"   - type: {type_filter}")
    print(f"   - data_type: {data_type_filter}")
    print(f"   - start_date: {start_date}")
    print(f"   - end_date: {end_date}")
    print(f"   - company_ids: {company_ids if company_ids else '전체'}")
    print(f"   - position_ids: {position_ids if position_ids else '전체'}")

    db = next(get_db())

    try:
        # 모든 모델 매퍼 초기화 (관계 오류 방지)
        _ = [Post, Company, Industry, PostSkill, Position, PositionSkill, IndustrySkill, Skill]

        # 1. recruit_schedule 조회 (company, post JOIN)
        print("\n[1] DB 조회 중...")
        query = db.query(RecruitmentSchedule)\
            .options(
                joinedload(RecruitmentSchedule.company),
                joinedload(RecruitmentSchedule.post)
            )

        # company_ids 필터 적용
        if company_ids:
            query = query.filter(RecruitmentSchedule.company_id.in_(company_ids))

        # position_ids 필터 적용
        if position_ids:
            query = query.join(Post, RecruitmentSchedule.post_id == Post.id)\
                .join(Industry, Post.industry_id == Industry.id)\
                .filter(Industry.position_id.in_(position_ids))

        schedules = query.all()
        print(f"   - 조회된 일정 수: {len(schedules)}")

        # 2. 데이터 변환
        print("\n[2] 데이터 변환 중...")
        result_schedules = []
        skipped_no_application = 0  # application_date 없어서 제외된 개수
        skipped_type = 0  # type 필터로 제외된 개수
        skipped_position = 0  # position 필터로 제외된 개수
        skipped_no_stages = 0  # stages 없어서 제외된 개수

        for schedule in schedules:
            # type 필터 (post.experience)
            # post가 없거나 experience가 null이면 제외
            if not schedule.post or not schedule.post.experience:
                skipped_type += 1
                continue

            # 신입 필터: 정확히 "신입"만 통과
            if type_filter == "신입":
                if schedule.post.experience != "신입":
                    skipped_type += 1
                    continue
            # 경력 필터: "신입"이 아닌 모든 것 통과
            elif type_filter == "경력":
                if schedule.post.experience == "신입":
                    skipped_type += 1
                    continue

            # position 필터 (메모리 필터 - 쿼리에서 못 거른 경우 대비)
            if position_ids and schedule.post.industry:
                if schedule.post.industry.position_id not in position_ids:
                    skipped_position += 1
                    continue
            
            # application_date가 없으면 제외 (미구현으로 간주)
            if not schedule.application_date or len(schedule.application_date) == 0:
                skipped_no_application += 1
                continue
            
            # stages 변환 (날짜 필터 포함)
            stages = convert_to_stages(schedule, start_date, end_date)
            
            # stages가 비어있으면 제외
            if not stages:
                skipped_no_stages += 1
                continue
            
            # 회사 정보 가져오기
            company_name = schedule.company.name if schedule.company else "Unknown"

            # 직군 정보 가져오기
            position_id = None
            if schedule.post and schedule.post.industry:
                position_id = schedule.post.industry.position_id

            # Swagger 형식으로 변환
            schedule_data = {
                "id": str(schedule.schedule_id),
                "company_id": schedule.company_id,
                "company_name": company_name,
                "type": type_filter,
                "position_id": position_id,
                "data_type": data_type_filter,
                "stages": stages
            }
            
            result_schedules.append(schedule_data)
        
        # 3. 최종 응답 형식
        response = {
            "status": 200,
            "code": "SUCCESS",
            "message": "회사별 채용 일정 조회 성공",
            "data": {
                "schedules": result_schedules
            }
        }
        
        # 4. 결과 출력
        print(f"\n[3] 변환 완료")
        print(f"   - 변환된 일정 수: {len(result_schedules)}")
        print(f"   - 총 stages 수: {sum(len(s['stages']) for s in result_schedules)}")
        print(f"\n[제외된 일정 통계]")
        print(f"   - type 필터: {skipped_type}개")
        print(f"   - position 필터: {skipped_position}개")
        print(f"   - application_date 없음: {skipped_no_application}개")
        print(f"   - stages 없음 (날짜 범위 외): {skipped_no_stages}개")
        
        print("\n" + "=" * 60)
        print("최종 결과 (JSON)")
        print("=" * 60)
        print(json.dumps(response, ensure_ascii=False, indent=2))
        
        print("\n" + "=" * 60)
        print(f"총 {len(result_schedules)}개의 일정이 추출되었습니다.")
        print("=" * 60)
        
        return response

    except Exception as e:
        print(f"\n 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()


def test_company_schedule(
    company_id: int,
    type_filter: str = "신입",
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    position_ids: list = None
):
    """
    특정 회사의 채용 일정을 조회합니다.

    Args:
        company_id: 회사 ID
        type_filter: "신입" 또는 "경력"
        start_date: 조회 시작일 (YYYY-MM-DD)
        end_date: 조회 종료일 (YYYY-MM-DD)
        position_ids: 직군 ID 리스트 (None이면 전체)
    """
    print("=" * 60)
    print(f"특정 회사 채용 일정 조회 (Company ID: {company_id})")
    print("=" * 60)
    print(f"\n[필터 조건]")
    print(f"   - company_id: {company_id}")
    print(f"   - type: {type_filter}")
    print(f"   - start_date: {start_date}")
    print(f"   - end_date: {end_date}")
    print(f"   - position_ids: {position_ids if position_ids else '전체'}")

    # test_swagger_format 함수 재사용
    return test_swagger_format(
        type_filter=type_filter,
        data_type_filter="actual",
        start_date=start_date,
        end_date=end_date,
        company_ids=[company_id],
        position_ids=position_ids
    )


def main():
    """전체 테스트 실행"""
    import sys
    
    print("\n" + "=" * 60)
    print("채용 일정 분석 시스템 - Swagger 형식 변환 테스트")
    print("=" * 60 + "\n")

    # 커맨드 라인 인자 파싱
    if len(sys.argv) > 1:
        # 첫 번째 인자가 숫자면 company_id로 간주
        try:
            company_id = int(sys.argv[1])
            type_filter = sys.argv[2] if len(sys.argv) > 2 else "신입"
            
            if type_filter not in ["신입", "경력"]:
                print("ERROR: type은 '신입' 또는 '경력'이어야 합니다.")
                print("사용법: python test_02_swagger_format.py [company_id] [신입|경력]")
                print("       python test_02_swagger_format.py [신입|경력]")
                return
            
            print(f"[특정 회사 테스트] Company ID: {company_id}, Type: {type_filter}\n")
            test_company_schedule(
                company_id=company_id,
                type_filter=type_filter
            )
            return
        except ValueError:
            # 숫자가 아니면 type_filter로 간주
            type_filter = sys.argv[1]
            if type_filter not in ["신입", "경력"]:
                print("ERROR: 올바른 type을 입력하세요: 신입 또는 경력")
                print("사용법: python test_02_swagger_format.py [신입|경력]")
                print("       python test_02_swagger_format.py [company_id] [신입|경력]")
                return
            
            print(f"[전체 회사 테스트] {type_filter}\n")
            test_swagger_format(
                type_filter=type_filter,
                data_type_filter="actual",
                start_date="2025-01-01",
                end_date="2025-12-31",
                company_ids=None
            )
            return
    
    # 인자 없으면 전체 테스트 실행
    print("[전체 테스트 모드]\n")
    
    # 테스트 케이스 1: 신입, 전체 회사
    print("\n" + "-" * 60)
    print("테스트 케이스 1: 신입, 전체 회사")
    print("-" * 60)
    test_swagger_format(
        type_filter="신입",
        data_type_filter="actual",
        start_date="2025-01-01",
        end_date="2025-12-31",
        company_ids=None
    )

    # 테스트 케이스 2: 신입, 특정 회사 (네이버: company_id=33)
    print("\n\n" + "-" * 60)
    print("테스트 케이스 2: 신입, 네이버만 (company_id=33)")
    print("-" * 60)
    test_swagger_format(
        type_filter="신입",
        data_type_filter="actual",
        start_date="2025-01-01",
        end_date="2025-12-31",
        company_ids=[33]
    )

    # 테스트 케이스 3: 경력, 전체 회사
    print("\n\n" + "-" * 60)
    print("테스트 케이스 3: 경력, 전체 회사")
    print("-" * 60)
    test_swagger_format(
        type_filter="경력",
        data_type_filter="actual",
        start_date="2025-01-01",
        end_date="2025-12-31",
        company_ids=None
    )


if __name__ == "__main__":
    main()

