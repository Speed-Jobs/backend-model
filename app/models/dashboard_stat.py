from sqlalchemy import Column, Integer
from app.db.config.base import Base


class DashboardStat(Base):
    __tablename__ = "dashboard_stat"

    id = Column(Integer, primary_key=True, index=True, comment="아이디")
    
    # TODO: 필요한 통계 컬럼 추가 예정