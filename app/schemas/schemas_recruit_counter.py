"""
채용 공고 수 추이 API Schema
"""
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional


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


class TopCompanyInfo(BaseModel):
    """상위 회사 정보"""
    company_id: int = Field(..., description="회사 ID")
    company_name: str = Field(..., description="회사명")
    total_count: int = Field(..., description="해당 기간의 총 공고 수")


class SelectedCompanyInfo(BaseModel):
    """선택된 회사 정보"""
    company_id: int = Field(..., description="회사 ID")
    company_name: str = Field(..., description="회사명")
    total_count: int = Field(..., description="해당 기간의 총 공고 수")


class JobPostingsTrendData(BaseModel):
    """채용 공고 수 추이 데이터"""
    timeframe: str = Field(..., description="시간 단위 (daily/weekly/monthly)")
    period: PeriodInfo = Field(..., description="조회 기간")
    trends: List[TrendItem] = Field(..., description="추이 데이터 목록")
    top_companies: List[TopCompanyInfo] = Field(default=[], description="상위 5개 경쟁사 정보 (전체 모드일 때만)")
    selected_company: Optional[SelectedCompanyInfo] = Field(None, description="선택된 회사 정보 (특정 회사 모드일 때만)")

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
                ],
                "top_companies": [
                    {"company_id": 1, "company_name": "토스", "total_count": 450},
                    {"company_id": 2, "company_name": "네이버", "total_count": 380}
                ],
                "selected_company": None
            }
        }

