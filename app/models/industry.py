from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base


class Industry(Base):
    __tablename__ = "industry"

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    name = Column(String(255), nullable=False, index=True, comment="이름")
    description = Column(Text, nullable=True, comment="설명")
    position_id = Column(Integer, ForeignKey("position.id"), nullable=True, index=True, comment="직무id")

    # Relationships
    position = relationship("Position", back_populates="industries")
    posts = relationship("Post", back_populates="industry")
    industry_skills = relationship("IndustrySkill", back_populates="industry")