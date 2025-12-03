"""
테스트 1: recruit_schedule, post, industry, company 테이블 조회
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import Column, Integer, String, Date, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import Session, relationship, joinedload
from app.db.config.base import Base, engine, get_db
from app.models.post import Post
from app.models.company import Company
from app.models.industry import Industry
from app.models.post_skill import PostSkill
from app.models.position import Position
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill
from app.models.skill import Skill
import enum
from datetime import date


# RecruitmentSchedule 모델 정의
class DataTypeEnum(str, enum.Enum):
    ACTUAL = "actual"
    PREDICTED = "predicted"


class RecruitmentSchedule(Base):
    __tablename__ = "recruit_schedule"

    schedule_id = Column(Integer, primary_key=True, index=True)
    post_id = Column(Integer, ForeignKey("post.id"), nullable=True, index=True)
    company_id = Column(Integer, ForeignKey("company.id"), nullable=False, index=True)
    industry_id = Column(Integer, ForeignKey("industry.id"), nullable=True, index=True)
    semester = Column(String(20), nullable=True)
    application_date = Column(JSON, nullable=True)
    document_screening_date = Column(JSON, nullable=True)
    first_interview = Column(JSON, nullable=True)
    second_interview = Column(JSON, nullable=True)
    join_date = Column(JSON, nullable=True)
    
    # Relationships
    post = relationship("Post", foreign_keys=[post_id])
    company = relationship("Company", foreign_keys=[company_id])


def test_db_query():
    """recruit_schedule, post, industry, company 테이블 조회"""
    print("=" * 60)
    print("테스트 1: DB 테이블 조회")
    print("=" * 60)
    
    db = next(get_db())
    
    try:
        # 모든 모델 매퍼 초기화
        _ = [Post, Company, PostSkill, Position, PositionSkill, IndustrySkill, Skill]
        
        print("\n[1] recruit_schedule 테이블 조회 (company JOIN)")
        schedules = db.query(RecruitmentSchedule)\
            .options(joinedload(RecruitmentSchedule.company))\
            .all()
        
        print(f"   - 조회된 일정 수: {len(schedules)}")
        
        # industry 정보 한번에 조회 (skillset 컬럼 문제 회피)
        print("\n[2] industry 테이블 조회")
        industry_ids = list(set(s.industry_id for s in schedules if s.industry_id))
        industries = {}
        if industry_ids:
            industry_list = db.query(Industry.id, Industry.name)\
                .filter(Industry.id.in_(industry_ids))\
                .all()
            industries = {ind.id: ind.name for ind in industry_list}
            print(f"   - 조회된 산업 수: {len(industries)}")
        
        # post 정보 조회
        print("\n[3] post 테이블 조회")
        post_ids = list(set(s.post_id for s in schedules if s.post_id))
        posts = {}
        if post_ids:
            post_list = db.query(Post.id, Post.title)\
                .filter(Post.id.in_(post_ids))\
                .all()
            posts = {p.id: p.title for p in post_list}
            print(f"   - 조회된 공고 수: {len(posts)}")
        
        # company 정보 (이미 JOIN됨)
        print("\n[4] company 테이블 (JOIN으로 조회됨)")
        company_count = len(set(s.company_id for s in schedules))
        print(f"   - 회사 수: {company_count}")
        
        # 통계 정보
        print("\n[통계 정보]")
        상반기_count = sum(1 for s in schedules if s.semester == "상반기")
        하반기_count = sum(1 for s in schedules if s.semester == "하반기")
        print(f"   - 상반기: {상반기_count}개")
        print(f"   - 하반기: {하반기_count}개")
        print(f"   - 전체: {len(schedules)}개")
        
        # 샘플 데이터 출력 (처음 3개)
        print("\n[샘플 데이터 - 처음 3개]")
        for i, schedule in enumerate(schedules[:3], 1):
            print(f"\n   [{i}] 일정 ID: {schedule.schedule_id}")
            print(f"       회사: {schedule.company.name if schedule.company else 'N/A'}")
            print(f"       산업: {industries.get(schedule.industry_id, 'N/A')}")
            if schedule.post_id and schedule.post_id in posts:
                print(f"       공고: {posts[schedule.post_id][:50]}...")
            print(f"       학기: {schedule.semester}")
            print(f"       지원 기간: {schedule.application_date}")
            print(f"       1차 면접: {schedule.first_interview}")
            print(f"       2차 면접: {schedule.second_interview}")
            print(f"       입사일: {schedule.join_date}")
        
        print("\n" + "=" * 60)
        print("DB 조회 성공!")
        print("=" * 60)
        
        return True, schedules, industries, posts
    
    except Exception as e:
        print(f"\nDB 조회 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, [], {}, {}
    finally:
        db.close()


if __name__ == "__main__":
    success, schedules, industries, posts = test_db_query()
    
    if success:
        print(f"\n총 {len(schedules)}개 일정 데이터 조회 완료")
        print(f"   - Company: {len(set(s.company_id for s in schedules))}개")
        print(f"   - Industry: {len(industries)}개")
        print(f"   - Post: {len(posts)}개")
    else:
        print("\n테스트 실패")
