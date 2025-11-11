from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import relationship
from app.db.base import Base


class Company(Base):
    __tablename__ = "company"

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    name = Column(String(255), nullable=False, index=True, comment="이름")
    description = Column(Text, nullable=True, comment="소개")
    domain = Column(String(255), nullable=True, comment="도메인")
    location = Column(String(255), nullable=True, comment="위치")
    founded_year = Column(Integer, nullable=True, comment="설립연도")
    size = Column(String(50), nullable=True, comment="규모")
    logo = Column(String(500), nullable=True, comment="로고_이미지")

    # Relationships
    posts = relationship("Post", back_populates="company")