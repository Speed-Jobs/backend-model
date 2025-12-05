"""
경쟁 강도 분석 - Company 단위 중복 제거

이 스크립트는 주요 경쟁사들의 채용 일정이 특정 기간 내에 얼마나 겹치는지 분석합니다.

주요 기능:
1. 9개 주요 경쟁사 그룹의 채용 일정 조회
2. 각 날짜별로 동시에 채용 중인 경쟁사 수 계산 (회사 단위 중복 제거)
3. 신입/경력 필터링 지원
4. Raw SQL을 사용하여 DB 쿼리와 완전히 동일한 결과 보장

사용법:
    python test_04_recruitment_intensity.py [시작일] [종료일] [채용유형]

    예시:
    python test_04_recruitment_intensity.py 2025-11-01 2025-11-30
    python test_04_recruitment_intensity.py 2025-11-01 2025-11-30 신입
    python test_04_recruitment_intensity.py 2025-11-01 2025-11-30 경력
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import Column, Integer, String, JSON, ForeignKey, or_, text
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
import json


# ==================== 모델 정의 ====================
class RecruitmentSchedule(Base):
    """
    채용 일정 모델

    각 채용 공고의 단계별 일정을 저장합니다.
    - application_date: 지원 접수 기간
    - document_screening_date: 서류 전형 기간
    - first_interview: 1차 면접 기간
    - second_interview: 2차 면접 기간
    - join_date: 입사 예정일

    각 필드는 JSON 형식으로 [["시작일", "종료일"]] 배열을 저장합니다.
    """
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


# ==================== 유틸리티 함수 ====================
def get_competitor_company_ids(db):
    """
    9개 경쟁사 그룹의 회사 ID 조회

    COMPANY_GROUPS 딕셔너리에 정의된 패턴을 사용하여
    데이터베이스에서 경쟁사 회사들을 찾습니다.

    경쟁사 그룹:
    - toss (토스, 토스뱅크, 토스증권, 비바리퍼블리카 등)
    - kakao (카카오)
    - hanwha (한화시스템)
    - hyundai autoever (현대오토에버)
    - woowahan (우아한형제들, 배달의민족, 배민)
    - coupang (쿠팡)
    - line (LINE, 라인)
    - naver (NAVER, 네이버)
    - lg cns (LG CNS)

    Args:
        db: SQLAlchemy 데이터베이스 세션

    Returns:
        tuple: (경쟁사 ID 리스트, 경쟁사 수)

    Note:
        - 이 함수는 디버깅용으로 사용되며, 실제 분석에서는 사용되지 않습니다.
        - analyze_competition_intensity 함수에서 직접 SQL을 생성하여 사용합니다.
    """
    from app.config.company_groups import COMPANY_GROUPS

    # COMPANY_GROUPS의 모든 그룹 패턴 사용
    like_conditions = []
    for patterns in COMPANY_GROUPS.values():
        for pattern in patterns:
            # %가 없으면 자동으로 %pattern% 형태로 변환
            if '%' not in pattern and '_' not in pattern:
                pattern = f'%{pattern}%'
            like_conditions.append(Company.name.like(pattern))

    if not like_conditions:
        return [], 0

    companies = db.query(Company.id, Company.name)\
        .filter(or_(*like_conditions))\
        .order_by(Company.id)\
        .all()

    competitor_ids = [company_id for company_id, _ in companies]

    # 디버깅: 찾은 회사들 출력
    print(f"\n[찾은 경쟁사 회사들] (총 {len(companies)}개)")
    for company_id, company_name in companies:
        print(f"  {company_id}: {company_name}")

    return competitor_ids, len(companies)


def get_schedule_overall_period(schedule):
    """
    채용 일정의 전체 기간 계산

    5개 단계의 모든 날짜 중에서 가장 이른 날짜(시작일)와 가장 늦은 날짜(종료일)를 찾습니다.

    처리 단계:
    1. application_date (지원 접수)
    2. document_screening_date (서류 전형)
    3. first_interview (1차 면접)
    4. second_interview (2차 면접)
    5. join_date (입사일)

    Args:
        schedule: RecruitmentSchedule 객체

    Returns:
        tuple: (시작일, 종료일) - 날짜 문자열 (YYYY-MM-DD)
               날짜가 없으면 (None, None) 반환

    Example:
        >>> schedule = RecruitmentSchedule(
        ...     application_date=[["2025-11-01", "2025-11-15"]],
        ...     first_interview=[["2025-11-20", "2025-11-25"]]
        ... )
        >>> get_schedule_overall_period(schedule)
        ("2025-11-01", "2025-11-25")

    Note:
        - 이 함수는 디버깅용으로 사용되며, 실제 분석에서는 SQL의 LEAST/GREATEST를 사용합니다.
    """
    all_dates = []
    stages = [
        schedule.application_date,
        schedule.document_screening_date,
        schedule.first_interview,
        schedule.second_interview,
        schedule.join_date
    ]

    for stage_data in stages:
        if not stage_data:
            continue

        for date_range in stage_data:
            if date_range and len(date_range) >= 1 and date_range[0]:
                all_dates.append(date_range[0])
            if date_range and len(date_range) >= 2 and date_range[1]:
                all_dates.append(date_range[1])

    if not all_dates:
        return None, None

    return min(all_dates), max(all_dates)


def analyze_competition_intensity(db, start_date, end_date, type_filter=None):
    """
    경쟁 강도 분석 - Company 단위 중복 제거 (Raw SQL 사용)

    특정 기간 내에 각 날짜별로 동시에 채용 중인 경쟁사 수를 계산합니다.
    같은 회사가 여러 개의 채용 공고를 올려도 1개 회사로만 카운트됩니다.

    알고리즘:
    1. COMPANY_GROUPS에서 모든 경쟁사 그룹의 LIKE 패턴 생성
    2. Raw SQL로 다음을 수행:
       a. 날짜 시퀀스 생성 (start_date ~ end_date)
       b. recruit_schedule에서 각 schedule의 전체 기간 계산 (5개 단계의 LEAST/GREATEST)
       c. 각 날짜에 대해 기간이 겹치는 schedule 찾기
       d. COUNT(DISTINCT company_id)로 회사 수 계산 (같은 회사의 여러 공고는 1번만 카운트)
    3. 결과를 날짜별로 그룹화하여 반환

    Args:
        db: SQLAlchemy 데이터베이스 세션
        start_date: 분석 시작일 (YYYY-MM-DD 형식)
        end_date: 분석 종료일 (YYYY-MM-DD 형식)
        type_filter: 채용 유형 필터 ("신입", "경력", None)
                    - "신입": 신입 채용만
                    - "경력": 경력 채용만
                    - None: 전체 (신입 + 경력)

    Returns:
        tuple: (결과 딕셔너리, 디버깅 정보 딕셔너리)

        결과 딕셔너리:
        {
            "status": 200,
            "code": "SUCCESS",
            "message": "경쟁 강도 분석 성공",
            "data": {
                "period": {"start_date": "2025-11-01", "end_date": "2025-11-30"},
                "type": "전체",
                "max_overlaps": 14,
                "daily_intensity": [
                    {"date": "2025-11-01", "overlap_count": 11},
                    {"date": "2025-11-02", "overlap_count": 11},
                    ...
                ]
            }
        }

        디버깅 정보:
        {
            "competitor_count": 47,  # 찾은 경쟁사 회사 수
            "max_overlaps": 14,      # 최대 겹침 수
            "daily_intensity_count": 30  # 분석된 날짜 수
        }

    Example:
        >>> db = next(get_db())
        >>> result, debug = analyze_competition_intensity(
        ...     db, "2025-11-01", "2025-11-30", "신입"
        ... )
        >>> print(f"최대 겹침: {result['data']['max_overlaps']}개")
        최대 겹침: 14개

    Note:
        - Raw SQL을 사용하여 test_04_simple.sql과 완전히 동일한 결과를 보장합니다.
        - p.experience IS NOT NULL 조건으로 experience가 있는 공고만 포함합니다.
        - 같은 회사의 여러 공고는 COUNT(DISTINCT company_id)로 중복 제거됩니다.
    """
    from app.config.company_groups import COMPANY_GROUPS

    # 1. 경쟁사 패턴 생성
    # COMPANY_GROUPS에서 모든 그룹의 LIKE 패턴을 가져와서 SQL WHERE 절 생성
    company_conditions = []
    for patterns in COMPANY_GROUPS.values():
        for pattern in patterns:
            # SQL 인젝션 방지를 위해 작은따옴표 이스케이프
            escaped_pattern = pattern.replace("'", "''")
            company_conditions.append(f"c.name LIKE '{escaped_pattern}'")

    if not company_conditions:
        return {
            "status": 200,
            "code": "SUCCESS",
            "message": "경쟁 강도 분석 성공",
            "data": {
                "period": {"start_date": start_date, "end_date": end_date},
                "type": type_filter if type_filter else "전체",
                "max_overlaps": 0,
                "daily_intensity": []
            }
        }, {"competitor_count": 0, "max_overlaps": 0}

    company_where = " OR ".join(company_conditions)

    # 2. Raw SQL로 날짜별 경쟁 강도 조회
    # 신입/경력 필터링 조건 생성
    experience_filter = ""
    if type_filter == "신입":
        experience_filter = "AND p.experience = '신입'"
    elif type_filter == "경력":
        experience_filter = "AND p.experience = '경력'"

    # 3. Raw SQL 쿼리 실행
    # 이 쿼리는 test_04_simple.sql과 완전히 동일한 로직을 사용합니다.
    #
    # 쿼리 구조:
    # 1) 날짜 시퀀스 생성 (dr 서브쿼리)
    #    - 0~99 숫자를 조합하여 최대 100일까지 지원
    #    - DATE_ADD로 start_date부터 순차적으로 날짜 생성
    #
    # 2) 채용 일정 서브쿼리 (rs 서브쿼리)
    #    - 각 schedule의 5개 단계에서 LEAST로 시작일, GREATEST로 종료일 계산
    #    - company_where 조건으로 경쟁사만 필터링
    #    - p.experience IS NOT NULL로 experience가 있는 공고만 포함
    #    - experience_filter로 신입/경력 필터링 (옵션)
    #
    # 3) LEFT JOIN
    #    - 날짜 시퀀스와 채용 일정을 날짜 범위로 조인
    #    - dr.date BETWEEN rs.start_date AND rs.end_date
    #
    # 4) GROUP BY & COUNT(DISTINCT)
    #    - 날짜별로 그룹화
    #    - COUNT(DISTINCT rs.company_id)로 회사 단위 중복 제거
    #    - HAVING overlap_count > 0으로 채용 중인 날짜만 반환
    sql_query = text(f"""
        SELECT
            dr.date,
            COUNT(DISTINCT rs.company_id) AS overlap_count
        FROM (
            SELECT DATE_ADD(:start_date, INTERVAL n DAY) AS date
            FROM (
                SELECT a.N + b.N * 10 AS n
                FROM
                    (SELECT 0 AS N UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
                     UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) a,
                    (SELECT 0 AS N UNION SELECT 1 UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
                     UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8 UNION SELECT 9) b
            ) numbers
            WHERE DATE_ADD(:start_date, INTERVAL n DAY) <= :end_date
        ) dr
        LEFT JOIN (
            SELECT
                rs.schedule_id,
                rs.company_id,
                c.name AS company_name,
                LEAST(
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.application_date, '$[0][0]')), '9999-12-31'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.document_screening_date, '$[0][0]')), '9999-12-31'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.first_interview, '$[0][0]')), '9999-12-31'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.second_interview, '$[0][0]')), '9999-12-31'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.join_date, '$[0][0]')), '9999-12-31')
                ) AS start_date,
                GREATEST(
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.application_date, '$[0][1]')), '0000-01-01'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.document_screening_date, '$[0][1]')), '0000-01-01'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.first_interview, '$[0][1]')), '0000-01-01'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.second_interview, '$[0][1]')), '0000-01-01'),
                    COALESCE(JSON_UNQUOTE(JSON_EXTRACT(rs.join_date, '$[0][1]')), '0000-01-01')
                ) AS end_date
            FROM recruit_schedule rs
            INNER JOIN company c ON rs.company_id = c.id
            INNER JOIN post p ON rs.post_id = p.id
            WHERE
                ({company_where})
                AND p.experience IS NOT NULL
                {experience_filter}
        ) rs
            ON dr.date BETWEEN rs.start_date AND rs.end_date
            AND rs.start_date != '9999-12-31'
            AND rs.end_date != '0000-01-01'
            AND rs.end_date >= :start_date
            AND rs.start_date <= :end_date
        GROUP BY dr.date
        HAVING overlap_count > 0
        ORDER BY dr.date
    """)

    # 4. SQL 실행
    result = db.execute(sql_query, {"start_date": start_date, "end_date": end_date})
    rows = result.fetchall()

    # 5. 결과 변환
    # SQL 결과를 Python 딕셔너리 리스트로 변환하고 최대값 계산
    daily_intensity = []
    max_overlaps = 0

    for row in rows:
        overlap_count = row.overlap_count
        max_overlaps = max(max_overlaps, overlap_count)
        daily_intensity.append({
            "date": str(row.date),
            "overlap_count": overlap_count
        })

    # 6. 디버깅 정보 수집
    # 실제로 찾은 경쟁사 회사 수를 계산 (디버깅용)
    competitor_count_query = text(f"""
        SELECT COUNT(DISTINCT c.id) AS count
        FROM company c
        WHERE {company_where}
    """)
    competitor_count = db.execute(competitor_count_query).scalar()

    # 7. 결과 반환
    return {
        "status": 200,
        "code": "SUCCESS",
        "message": "경쟁 강도 분석 성공",
        "data": {
            "period": {"start_date": start_date, "end_date": end_date},
            "type": type_filter if type_filter else "전체",
            "max_overlaps": max_overlaps,
            "daily_intensity": daily_intensity
        }
    }, {
        "competitor_count": competitor_count,
        "max_overlaps": max_overlaps,
        "daily_intensity_count": len(daily_intensity)
    }


# ==================== 메인 ====================
def main():
    """
    메인 실행 함수

    커맨드라인 인자를 파싱하여 경쟁 강도 분석을 실행하고 결과를 출력합니다.

    실행 방법:
        python test_04_recruitment_intensity.py [시작일] [종료일] [채용유형]

    인자:
        시작일: 분석 시작 날짜 (YYYY-MM-DD), 기본값: 2025-11-01
        종료일: 분석 종료 날짜 (YYYY-MM-DD), 기본값: 2025-11-30
        채용유형: "신입" 또는 "경력", 기본값: 전체

    출력:
        1. 설정 정보 (시작일, 종료일, 채용 유형)
        2. 날짜별 경쟁 강도 (날짜 \t 겹침 수)
        3. 통계 (경쟁사 수, 최대 겹침, 분석 날짜 수)

    Example:
        $ python test_04_recruitment_intensity.py 2025-11-01 2025-11-30 신입

        ============================================================
        경쟁 강도 분석 (Company 단위)
        ============================================================

        [설정]
          시작일: 2025-11-01
          종료일: 2025-11-30
          채용 유형: 신입

        [날짜별 경쟁 강도]
        2025-11-01      11
        2025-11-02      11
        ...

        [통계]
          경쟁사 수: 47개
          최대 겹침: 14개
          분석 날짜: 30일
    """
    print("\n" + "=" * 60)
    print("경쟁 강도 분석 (Company 단위)")
    print("=" * 60)

    db = next(get_db())

    try:
        # 커맨드라인 인자 파싱
        start_date = sys.argv[1] if len(sys.argv) > 1 else "2025-11-01"
        end_date = sys.argv[2] if len(sys.argv) > 2 else "2025-11-30"
        type_filter = sys.argv[3] if len(sys.argv) > 3 else None

        # 설정 출력
        print(f"\n[설정]")
        print(f"  시작일: {start_date}")
        print(f"  종료일: {end_date}")
        print(f"  채용 유형: {type_filter if type_filter else '전체'}")

        # 경쟁 강도 분석 실행
        result, debug_info = analyze_competition_intensity(
            db=db,
            start_date=start_date,
            end_date=end_date,
            type_filter=type_filter
        )

        # 날짜별 결과 출력
        print(f"\n[날짜별 경쟁 강도]")
        for item in result["data"]["daily_intensity"]:
            print(f"{item['date']}\t{item['overlap_count']}")

        # 통계 출력
        print(f"\n[통계]")
        print(f"  경쟁사 수: {debug_info['competitor_count']}개")
        print(f"  최대 겹침: {debug_info['max_overlaps']}개")
        print(f"  분석 날짜: {debug_info['daily_intensity_count']}일")

    except Exception as e:
        print(f"\n오류: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    main()
