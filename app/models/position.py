from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Enum
from sqlalchemy.orm import relationship
from app.db.config.base import Base
import enum


class PositionCategory(enum.Enum):
    TECH = "TECH"
    BIZ = "BIZ"
    BIZ_SUPPORTING = "BIZ_SUPPORTING"


class Position(Base):
    __tablename__ = "position"

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    name = Column(String(255), nullable=False, index=True, comment="이름")
    description = Column(Text, nullable=True, comment="설명")
    # skillset 컬럼 제거 (ERD에 없음, DB에도 없음)
    category = Column(Enum(PositionCategory), nullable=True, index=True, comment="직무 카테고리")

    # ERD 기반 필드
    is_deleted = Column(Boolean, nullable=False, default=False, comment="삭제 여부")
    created_at = Column(DateTime, nullable=True, comment="생성일시")
    modified_at = Column(DateTime, nullable=True, comment="수정일시")

    # Relationships
    industries = relationship("Industry", back_populates="position")
    position_skills = relationship("PositionSkill", back_populates="position")