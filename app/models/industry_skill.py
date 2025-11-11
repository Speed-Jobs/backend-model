from sqlalchemy import Column, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.db.base import Base


class IndustrySkill(Base):
    __tablename__ = "industry_skill"
    __table_args__ = (
        UniqueConstraint('industry_id', 'skill_id', name='unique_industry_skill'),
    )

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    skill_id = Column(Integer, ForeignKey("skill.id", ondelete="CASCADE"), nullable=False, index=True, comment="기술id")
    industry_id = Column(Integer, ForeignKey("industry.id", ondelete="CASCADE"), nullable=False, index=True, comment="산업id")

    # Relationships
    skill = relationship("Skill", back_populates="industry_skills")
    industry = relationship("Industry", back_populates="industry_skills")