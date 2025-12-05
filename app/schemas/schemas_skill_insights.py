from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class QuarterlyTrend(BaseModel):
    """분기별 스킬 트렌드 데이터"""
    Q1: Dict[str, int] = Field(..., description="Q1 분기 스킬별 공고 수")
    Q2: Dict[str, int] = Field(..., description="Q2 분기 스킬별 공고 수")
    Q3: Dict[str, int] = Field(..., description="Q3 분기 스킬별 공고 수")
    Q4: Dict[str, int] = Field(..., description="Q4 분기 스킬별 공고 수")

class SkillTrendYearData(BaseModel):
    """연도별 스킬 트렌드 데이터 (분기별)"""
    year: str = Field(..., description="연도 (YYYY 형식)")
    comparison_year: Optional[str] = Field(None, description="비교 연도 (YYYY 형식)")
    skills: List[str] = Field(..., description="상위 스킬 목록")
    quarterly_trends: Dict[str, QuarterlyTrend] = Field(..., description="연도별 분기별 트렌드 데이터")

class SkillTrendMultiYearData(BaseModel):
    """다년도 스킬 트렌드 데이터"""
    years: List[str] = Field(..., description="조회 연도 목록")
    skills: List[str] = Field(..., description="상위 스킬 목록")
    skill_frequencies: Dict[str, Dict[str, int]] = Field(..., description="연도별 스킬 빈도수 (키: 연도 문자열, 값: 스킬별 빈도수)")

class SkillTrendData(BaseModel):
    """스킬 트렌드 데이터 (단일 연도 또는 다년도)"""
    year: Optional[str] = Field(None, description="조회 연도 (단일 연도 조회 시)")
    comparison_year: Optional[str] = Field(None, description="비교 연도 (단일 연도 조회 시)")
    years: Optional[List[str]] = Field(None, description="조회 연도 목록 (다년도 조회 시)")
    skills: Optional[List[str]] = Field(None, description="상위 스킬 목록")
    quarterly_trends: Optional[Dict[str, QuarterlyTrend]] = Field(None, description="연도별 분기별 트렌드 데이터 (단일 연도 조회 시)")
    skill_frequencies: Optional[Dict[str, Dict[str, int]]] = Field(None, description="연도별 스킬 빈도수 (다년도 조회 시)")

class SkillTrendResponse(BaseModel):
    """스킬 트렌드 조회 응답"""
    status: int = Field(200, description="HTTP 상태 코드")
    code: str = Field("SUCCESS", description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    data: SkillTrendData

class ErrorResponse(BaseModel):
    """에러 응답"""
    status: int = Field(..., description="HTTP 상태 코드")
    code: str = Field(..., description="에러 코드")
    message: str = Field(..., description="에러 메시지")


