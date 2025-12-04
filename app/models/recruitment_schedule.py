"""
RecruitmentSchedule Model
"""
from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.db.config.base import Base

class RecruitmentSchedule(Base):
    """채용 일정 모델"""
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

