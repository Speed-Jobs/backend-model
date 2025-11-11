from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.base import Base


class Post(Base):
    __tablename__ = "post"

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    title = Column(String(500), nullable=False, index=True, comment="제목")
    employment_type = Column(String(50), nullable=True, comment="채용형태")
    experience = Column(String(100), nullable=True, comment="경력")
    work_type = Column(String(50), nullable=True, comment="근무형태")
    
    # 추가 필드
    description = Column(Text, nullable=True, comment="공고 상세설명")
    meta_data = Column(JSON, nullable=True, comment="메타데이터 (job_category, preferred_qualifications 등)")
    
    posted_at = Column(DateTime, nullable=True, index=True, comment="게시시작")
    close_at = Column(DateTime, nullable=True, comment="종료시작")
    crawled_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True, comment="크롤링시작")
    source_url = Column(String(1000), nullable=False, unique=True, index=True, comment="원문url")
    screenshot_url = Column(String(1000), nullable=True, comment="스크린샷_url")
    
    company_id = Column(Integer, ForeignKey("company.id"), nullable=False, index=True, comment="회사id")
    industry_id = Column(Integer, ForeignKey("industry.id"), nullable=True, index=True, comment="산업id")

    # Relationships
    company = relationship("Company", back_populates="posts")
    industry = relationship("Industry", back_populates="posts")
    post_skills = relationship("PostSkill", back_populates="post", cascade="all, delete-orphan")
