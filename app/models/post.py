from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.config.base import Base


class Post(Base):
    __tablename__ = "post"

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    title = Column(String(500), nullable=False, index=True, comment="제목")
    employment_type = Column(String(50), nullable=True, comment="채용형태")
    experience = Column(String(100), nullable=True, comment="경력")
    work_type = Column(String(50), nullable=True, comment="근무형태")
    
    # 추가 필드
    # ERD 상 description 은 varchar(255)이지만, 실제 본문이 길 수 있어 Text 로 매핑
    description = Column(Text, nullable=True, comment="공고 상세설명")
    meta_data = Column(JSON, nullable=True, comment="메타데이터 (job_category, preferred_qualifications 등)")
    
    posted_at = Column(DateTime, nullable=True, index=True, comment="게시시작")
    close_at = Column(DateTime, nullable=True, comment="종료시작")
    crawled_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True, comment="크롤링시작")
    source_url = Column(String(1000), nullable=False, unique=True, index=True, comment="원문url")
    url_hash = Column(String(255), nullable=True, comment="URL 해시")
    screenshot_url = Column(String(1000), nullable=True, comment="스크린샷_url")
    
    company_id = Column(Integer, ForeignKey("company.id"), nullable=False, index=True, comment="회사id")
    industry_id = Column(Integer, ForeignKey("industry.id"), nullable=True, index=True, comment="산업id")

    # 소프트 삭제 및 생성/수정 시간 (ERD 반영)
    is_deleted = Column(Boolean, nullable=False, default=False, comment="삭제 여부")
    is_vectored = Column(Boolean, nullable=False, default=False, comment="벡터DB 적재여부")
    created_at = Column(DateTime, nullable=True, comment="생성일시")
    modified_at = Column(DateTime, nullable=True, comment="수정일시")

    # Relationships여부부
    company = relationship("Company", back_populates="posts")
    industry = relationship("Industry", back_populates="posts")
    post_skills = relationship("PostSkill", back_populates="post", cascade="all, delete-orphan")
