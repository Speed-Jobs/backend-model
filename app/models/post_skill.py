from sqlalchemy import Column, Integer, ForeignKey, UniqueConstraint, Boolean, DateTime
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

    # ERD 기반 필드
    is_deleted = Column(Boolean, nullable=False, default=False, comment="삭제 여부")
    created_at = Column(DateTime, nullable=True, comment="생성일시")
    modified_at = Column(DateTime, nullable=True, comment="수정일시")

    # Relationships
    skill = relationship("Skill", back_populates="post_skills")
    post = relationship("Post", back_populates="post_skills")