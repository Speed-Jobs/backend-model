"""
직군별 인사이트 API Schema
"""
from pydantic import BaseModel, Field
from typing import List, Optional

from app.schemas.schemas_competitor_industry_trend import PeriodSummary


class JobRoleTrend(BaseModel):
    """직군 트렌드 정보"""
    job_role_name: str = Field(..., description="직군 이름 (예: Software Development)")
    current_percentage: float = Field(..., description="현재 기간 전체 대비 비율 (%)")
    previous_percentage: float = Field(..., description="이전 기간 전체 대비 비율 (%)")
    change_rate: float = Field(..., description="비율 변화 (퍼센트 포인트)")
    current_count: int = Field(..., description="현재 기간 공고 수")
    previous_count: int = Field(..., description="이전 기간 공고 수")
    rank: int = Field(..., description="현재 기간 순위 (공고 수 기준)")


class NewsItem(BaseModel):
    """뉴스 아이템"""
    title: str = Field(..., description="뉴스 제목")
    link: str = Field(..., description="뉴스 링크")
    pub_date: str = Field(..., description="발행일")
    description: str = Field(..., description="뉴스 요약")


class EvidenceReference(BaseModel):
    """근거 참조"""
    type: str = Field(..., description="근거 유형 ('news' 또는 'data')")
    source: str = Field(..., description="근거 출처 (뉴스 제목 또는 데이터 소스 설명)")
    link: Optional[str] = Field(None, description="뉴스 링크 (뉴스인 경우)")
    data_description: Optional[str] = Field(None, description="데이터 설명 (데이터인 경우)")
    date: Optional[str] = Field(None, description="데이터 날짜/기간 (data인 경우)")
    value: Optional[str] = Field(None, description="데이터 값 (data인 경우, 예: '24.3%', '5.2%p 증가')")


class JobRoleInsightItem(BaseModel):
    """개별 직군 인사이트"""
    job_role_name: str = Field(..., description="직군 이름")
    insight: str = Field(..., description="인사이트 설명 (예: 'AI 직군이 전체의 24.3%를 차지하며 가장 많은 공고를 보유하고 있습니다')")
    change_description: Optional[str] = Field(None, description="변화 설명 (예: 'AI 직군은 직전 대비 5.2% 향상되었습니다')")
    external_factors: Optional[str] = Field(None, description="외부 요인 설명 (뉴스 기반, 예: '이는, [뉴스 내용]과 관련될 가능성이 있습니다')")
    evidence: List[EvidenceReference] = Field(default=[], description="인사이트의 근거")


class JobRoleInsightData(BaseModel):
    """직군별 인사이트 데이터"""
    # 프로그램적으로 설정되는 필드 (LLM이 생성하지 않음)
    timeframe: Optional[str] = Field(None, description="시간 단위 (monthly_same_period / quarterly_same_period) - 서버에서 자동 설정")
    category: Optional[str] = Field(None, description="직군 카테고리 (Tech / Biz / BizSupporting) - 서버에서 자동 설정")
    current_period: Optional[PeriodSummary] = Field(None, description="현재 기간 정보 - 서버에서 자동 설정")
    previous_period: Optional[PeriodSummary] = Field(None, description="이전 기간 정보 - 서버에서 자동 설정")
    company_filter: Optional[str] = Field(None, description="회사 필터 (지정된 경우) - 서버에서 자동 설정")
    
    # LLM 생성 인사이트
    summary: str = Field(..., description="종합 요약")
    summary_evidence: List[EvidenceReference] = Field(default=[], description="요약의 근거")
    key_findings: List[str] = Field(default=[], description="주요 발견 사항")
    key_findings_evidence: List[List[EvidenceReference]] = Field(default=[], description="각 key_finding의 근거 목록")
    job_role_insights: List[JobRoleInsightItem] = Field(default=[], description="개별 직군별 인사이트")

    class Config:
        json_schema_extra = {
            "example": {
                "timeframe": "monthly_same_period",
                "category": "Tech",
                "current_period": {
                    "start_date": "2025-11-26",
                    "end_date": "2025-12-03",
                    "total_count": 500
                },
                "previous_period": {
                    "start_date": "2025-11-19",
                    "end_date": "2025-11-26",
                    "total_count": 450
                },
                "summary": "Tech 직군 카테고리에서 AI 관련 직군이 가장 큰 성장을 보이고 있습니다...",
                "key_findings": [
                    "AI 직군이 전체의 24.3%를 차지하며 가장 많은 공고를 보유",
                    "AI 직군은 직전 대비 5.2% 향상"
                ],
                "job_role_insights": [
                    {
                        "job_role_name": "AI",
                        "insight": "AI 직군이 전체의 24.3%를 차지하며 가장 많은 공고를 보유하고 있습니다",
                        "change_description": "AI 직군은 직전 대비 5.2% 향상되었습니다",
                        "external_factors": "이는, 최근 AI 기술 발전과 관련된 뉴스와 관련될 가능성이 있습니다"
                    }
                ]
            }
        }

