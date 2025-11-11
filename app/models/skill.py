from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from app.db.base import Base


class Skill(Base):
    __tablename__ = "skill"

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    name = Column(String(255), nullable=False, unique=True, index=True, comment="이름")

    # Relationships
    post_skills = relationship("PostSkill", back_populates="skill")
    industry_skills = relationship("IndustrySkill", back_populates="skill")
    position_skills = relationship("PositionSkill", back_populates="skill")