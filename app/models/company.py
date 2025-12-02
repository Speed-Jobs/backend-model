from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.orm import relationship
from app.db.config.base import Base


class Company(Base):
    __tablename__ = "company"

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    name = Column(String(255), nullable=False, index=True, comment="이름")
    description = Column(Text, nullable=True, comment="소개")
    domain = Column(String(255), nullable=True, comment="도메인(ERD: enum, 여기서는 String 매핑)")
    location = Column(String(255), nullable=True, comment="위치")
    founded_year = Column(Integer, nullable=True, comment="설립연도")
    size = Column(String(50), nullable=True, comment="규모(ERD: enum, 여기서는 String 매핑)")
    logo = Column(String(500), nullable=True, comment="로고_이미지")

    # ERD 기반 필드
    is_competitor = Column(Boolean, nullable=False, default=False, comment="경쟁사 여부")
    is_deleted = Column(Boolean, nullable=False, default=False, comment="삭제 여부")
    created_at = Column(DateTime, nullable=True, comment="생성일시")
    modified_at = Column(DateTime, nullable=True, comment="수정일시")

    # Relationships
    posts = relationship("Post", back_populates="company")