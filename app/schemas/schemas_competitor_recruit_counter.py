"""
주요 회사별 채용 활동 API Schema
"""
from pydantic import BaseModel, Field
from typing import List, Dict


class CompanyInfo(BaseModel):
    """회사 정보"""
    id: int = Field(..., description="회사 ID")
    name: str = Field(..., description="회사명")
    key: str = Field(..., description="프론트엔드용 키 (예: toss, line)")


class ActivityItem(BaseModel):
    """채용 활동 데이터 항목"""
    period: str = Field(..., description="기간 표시")
    counts: Dict[str, int] = Field(..., description="회사별 공고 수 (key-value)")

    class Config:
        json_schema_extra = {
            "example": {
                "period": "11/1",
                "counts": {
                    "toss": 18,
                    "line": 14,
                    "hanwha": 15
                }
            }
        }


class PeriodInfo(BaseModel):
    """기간 정보"""
    start_date: str = Field(..., description="시작일 (YYYY-MM-DD)")
    end_date: str = Field(..., description="종료일 (YYYY-MM-DD)")


class RecruitmentActivityData(BaseModel):
    """회사별 채용 활동 데이터"""
    timeframe: str = Field(..., description="시간 단위 (daily/weekly/monthly)")
    companies: List[CompanyInfo] = Field(..., description="회사 목록")
    activities: List[ActivityItem] = Field(..., description="채용 활동 데이터")

    class Config:
        json_schema_extra = {
            "example": {
                "timeframe": "daily",
                "companies": [
                    {"id": 3, "name": "토스", "key": "toss"},
                    {"id": 4, "name": "라인", "key": "line"}
                ],
                "activities": [
                    {
                        "period": "11/1",
                        "counts": {
                            "toss": 18,
                            "line": 14
                        }
                    }
                ]
            }
        }

