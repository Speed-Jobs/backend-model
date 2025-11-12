from sqlalchemy import Column, Integer, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.db.config.base import Base


class PostSkill(Base):
    __tablename__ = "post_skill"
    __table_args__ = (
        UniqueConstraint('post_id', 'skill_id', name='unique_post_skill'),
    )

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    skill_id = Column(Integer, ForeignKey("skill.id", ondelete="CASCADE"), nullable=False, index=True, comment="기술id")
    post_id = Column(Integer, ForeignKey("post.id", ondelete="CASCADE"), nullable=False, index=True, comment="공고id")

    # Relationships
    skill = relationship("Skill", back_populates="post_skills")
    post = relationship("Post", back_populates="post_skills")