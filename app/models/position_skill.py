from sqlalchemy import Column, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.db.base import Base


class PositionSkill(Base):
    __tablename__ = "position_skill"
    __table_args__ = (
        UniqueConstraint('position_id', 'skill_id', name='unique_position_skill'),
    )

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    position_id = Column(Integer, ForeignKey("position.id", ondelete="CASCADE"), nullable=False, index=True, comment="직무id")
    skill_id = Column(Integer, ForeignKey("skill.id", ondelete="CASCADE"), nullable=False, index=True, comment="기술id")

    # Relationships
    position = relationship("Position", back_populates="position_skills")
    skill = relationship("Skill", back_populates="position_skills")