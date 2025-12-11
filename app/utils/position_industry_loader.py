"""
Position 및 Industry 이름 목록 로더 (공통 유틸리티)

애플리케이션 시작 시 한 번만 로드하여 전역 변수로 저장합니다.
"""
from typing import List
from app.db.config.base import SessionLocal


def _get_position_names() -> List[str]:
    """DB에서 position 이름 목록 가져오기 (애플리케이션 시작 시 한 번만 실행)"""
    try:
        from app.models.position import Position
        db = SessionLocal()
        try:
            positions = db.query(Position).filter(
                Position.is_deleted == False
            ).order_by(Position.name).all()
            return [p.name for p in positions] if positions else []
        finally:
            db.close()
    except Exception:
        # DB 연결 실패 시 빈 리스트 반환 (애플리케이션 시작 시 DB가 준비되지 않았을 수 있음)
        return []


def _get_industry_names() -> List[str]:
    """DB에서 industry 이름 목록 가져오기 (애플리케이션 시작 시 한 번만 실행)"""
    try:
        from app.models.industry import Industry
        db = SessionLocal()
        try:
            industries = db.query(Industry).filter(
                Industry.is_deleted == False
            ).order_by(Industry.name).all()
            return [i.name for i in industries] if industries else []
        finally:
            db.close()
    except Exception:
        # DB 연결 실패 시 빈 리스트 반환
        return []


# 전역 변수로 position/industry 목록 저장 (애플리케이션 시작 시 한 번만 로드)
POSITION_NAMES = _get_position_names()
INDUSTRY_NAMES = _get_industry_names()

