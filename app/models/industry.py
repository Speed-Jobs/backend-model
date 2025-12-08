from sqlalchemy import Column, Integer, String, Text, ForeignKey, Boolean, DateTime
from sqlalchemy.orm import relationship
from app.db.config.base import Base


class Industry(Base):
    __tablename__ = "industry"

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    name = Column(String(255), nullable=False, index=True, comment="이름")
    description = Column(Text, nullable=True, comment="설명")
    # skillset = Column(Text, nullable=True, comment="스킬셋")  # DB에 컬럼 없음
    position_id = Column(Integer, ForeignKey("position.id"), nullable=True, index=True, comment="직무id")

    # ERD 기반 필드
    is_deleted = Column(Boolean, nullable=False, default=False, comment="삭제 여부")
    created_at = Column(DateTime, nullable=True, comment="생성일시")
    modified_at = Column(DateTime, nullable=True, comment="수정일시")

    # Relationships
    position = relationship("Position", back_populates="industries")
    posts = relationship("Post", back_populates="industry")
    industry_skills = relationship("IndustrySkill", back_populates="industry")