"""
채용 공고 수 추이 API Schema
"""
from pydantic import BaseModel, Field
from typing import Any, List, Dict


class DashBoardResponse(BaseModel):
    """공통 대시보드 응답 형식"""
    status: int = Field(..., description="HTTP 상태 코드")
    code: str = Field(..., description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    data: Any = Field(..., description="응답 데이터")


class TrendItem(BaseModel):
    """추이 데이터 항목"""
    period: str = Field(..., description="기간 표시 (예: '11/1', '9월 1주', '2025-01')")
    count: int = Field(..., description="해당 기간의 공고 수")


class PeriodInfo(BaseModel):
    """기간 정보"""
    start_date: str = Field(..., description="시작일 (YYYY-MM-DD)")
    end_date: str = Field(..., description="종료일 (YYYY-MM-DD)")


class JobPostingsTrendData(BaseModel):
    """채용 공고 수 추이 데이터"""
    timeframe: str = Field(..., description="시간 단위 (daily/weekly/monthly)")
    period: PeriodInfo = Field(..., description="조회 기간")
    trends: List[TrendItem] = Field(..., description="추이 데이터 목록")

    class Config:
        json_schema_extra = {
            "example": {
                "timeframe": "daily",
                "period": {
                    "start_date": "2025-11-01",
                    "end_date": "2025-11-30"
                },
                "trends": [
                    {"period": "11/1", "count": 180},
                    {"period": "11/2", "count": 195}
                ]
            }
        }

