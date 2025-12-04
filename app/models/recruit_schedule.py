from sqlalchemy import Column, BigInteger, String, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.db.config.base import Base


class RecruitSchedule(Base):
    __tablename__ = "recruit_schedule"

    schedule_id = Column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
        comment="채용 일정 ID",
    )
    post_id = Column(
        BigInteger,
        ForeignKey("post.id"),
        index=True,
        nullable=False,
        comment="공고 ID",
    )
    company_id = Column(
        BigInteger,
        ForeignKey("company.id"),
        index=True,
        nullable=True,
        comment="회사 ID",
    )
    industry_id = Column(
        BigInteger,
        ForeignKey("industry.id"),
        index=True,
        nullable=True,
        comment="산업 ID",
    )

    semester = Column(String(20), nullable=True, comment="상/하반기 구분")
    # 모든 날짜 필드는 [시작일, 마감일] 형태의 1차원 배열 리스트를 JSON 컬럼에 저장 ([[start, end], ...])
    # 단일 날짜인 경우 [date, date] 형태로 동일한 날짜를 두 번 반복
    application_date = Column(JSON, nullable=True, comment="지원서 접수 기간(JSON, [[start, end], ...])")
    document_screening_date = Column(JSON, nullable=True, comment="서류 전형 기간(JSON, [[start, end], ...])")
    first_interview = Column(JSON, nullable=True, comment="1차 면접일(JSON, [[date, date], ...])")
    second_interview = Column(JSON, nullable=True, comment="2차 면접일(JSON, [[date, date], ...])")
    join_date = Column(JSON, nullable=True, comment="입사일 기간(JSON, [[start, end], ...])")

    # 관계
    post = relationship("Post", backref="recruit_schedules")


