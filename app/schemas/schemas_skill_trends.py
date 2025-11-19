from pydantic import BaseModel, Field
from typing import Dict, List
from datetime import datetime

class SkillTrend(BaseModel):
    """분기별 스킬 트렌드"""
    quarter: str = Field(..., description="분기 (YYYY QN 형식, 예: '2025 Q3', '2025 Q4')")
    skills: Dict[str, int] = Field(..., description="각 스킬별 공고 수 (해당 분기의 월별 데이터 집계값)")


class SkillTrendData(BaseModel):
    """스킬 트렌드 데이터"""
    company: str = Field(..., description="회사명")
    year: int = Field(..., description="조회 연도")
    trends: List[SkillTrend] = Field(..., description="분기별 트렌드 목록")


class SkillTrendResponse(BaseModel):
    """스킬 트렌드 조회 응답"""
    status: int = Field(200, description="HTTP 상태 코드")
    code: str = Field("SUCCESS", description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    data: SkillTrendData

