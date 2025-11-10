from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import relationship
from app.db.base import Base


class Position(Base):
    __tablename__ = "position"

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    name = Column(String(255), nullable=False, index=True, comment="이름")
    description = Column(Text, nullable=True, comment="설명")

    # Relationships
    industries = relationship("Industry", back_populates="position")
    position_skills = relationship("PositionSkill", back_populates="position")