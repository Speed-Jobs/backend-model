from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from app.db.config.base import Base


class Skill(Base):
    __tablename__ = "skill"

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    name = Column(String(255), nullable=False, unique=True, index=True, comment="이름")

    # ERD 기반 필드
    is_deleted = Column(Boolean, nullable=False, default=False, comment="삭제 여부")
    created_at = Column(DateTime, nullable=True, comment="생성일시")
    modified_at = Column(DateTime, nullable=True, comment="수정일시")

    # Relationships
    post_skills = relationship("PostSkill", back_populates="skill")
    industry_skills = relationship("IndustrySkill", back_populates="skill")
    position_skills = relationship("PositionSkill", back_populates="skill")