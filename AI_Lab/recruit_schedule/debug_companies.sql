"""
경쟁 강도 분석 - Company 단위 중복 제거
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
    """9개 경쟁사 그룹의 회사 ID 조회"""
    from app.db.crud.db_competitors_skills import COMPETITOR_GROUPS

    # 9개 경쟁사 그룹
    recruiting_companies = ["네이버", "토스", "라인", "우아한형제들", "LG CNS", "현대오토에버", "한화시스템", "카카오", "쿠팡"]
    like_conditions = []

    for company_name in recruiting_companies:
        if company_name in COMPETITOR_GROUPS:
            for keyword in COMPETITOR_GROUPS[company_name]:
                # %가 없으면 자동으로 %keyword% 형태로 변환
                pattern = keyword
                if '%' not in pattern and '_' not in pattern:
                    pattern = f'%{keyword}%'

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
    Schedule의 전체 기간 계산
    5개 단계(application_date, document_screening_date, first_interview, second_interview, join_date)의
    모든 날짜에서 최소값(시작일), 최대값(종료일) 반환
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

    Args:
        db: Database session
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        type_filter: "신입" 또는 "경력" (None이면 전체)
    """
    from app.db.crud.db_competitors_skills import COMPETITOR_GROUPS

    # 1. 경쟁사 패턴 생성
    recruiting_companies = ["네이버", "토스", "라인", "우아한형제들", "LG CNS", "현대오토에버", "한화시스템", "카카오", "쿠팡"]
    company_conditions = []
    for company_name in recruiting_companies:
        if company_name in COMPETITOR_GROUPS:
            for keyword in COMPETITOR_GROUPS[company_name]:
                company_conditions.append(f"c.name LIKE '{keyword}'")

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
    experience_filter = ""
    if type_filter == "신입":
        experience_filter = "AND p.experience = '신입'"
    elif type_filter == "경력":
        experience_filter = "AND p.experience = '경력'"

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

    result = db.execute(sql_query, {"start_date": start_date, "end_date": end_date})
    rows = result.fetchall()

    # 3. 결과 변환
    daily_intensity = []
    max_overlaps = 0

    for row in rows:
        overlap_count = row.overlap_count
        max_overlaps = max(max_overlaps, overlap_count)
        daily_intensity.append({
            "date": str(row.date),
            "overlap_count": overlap_count
        })

    # 디버깅 정보 수집
    competitor_count_query = text(f"""
        SELECT COUNT(DISTINCT c.id) AS count
        FROM company c
        WHERE {company_where}
    """)
    competitor_count = db.execute(competitor_count_query).scalar()

    # 4. 결과 반환
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
    print("\n" + "=" * 60)
    print("경쟁 강도 분석 (Company 단위)")
    print("=" * 60)

    db = next(get_db())

    try:
        start_date = sys.argv[1] if len(sys.argv) > 1 else "2025-11-01"
        end_date = sys.argv[2] if len(sys.argv) > 2 else "2025-11-30"
        type_filter = sys.argv[3] if len(sys.argv) > 3 else None

        print(f"\n[설정]")
        print(f"  시작일: {start_date}")
        print(f"  종료일: {end_date}")
        print(f"  채용 유형: {type_filter if type_filter else '전체'}")

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
