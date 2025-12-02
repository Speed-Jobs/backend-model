"""
회사 인사이트 API Schema
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from app.schemas.schemas_recruit_counter import PeriodInfo


class CompetitorComparison(BaseModel):
    """경쟁사 비교 정보"""
    company_name: str = Field(..., description="경쟁사명")
    total_count: int = Field(..., description="해당 기간 총 공고 수")
    rank: int = Field(..., description="공고 수 기준 순위")
    market_share: float = Field(..., description="전체 대비 시장 점유율 (%)")


class NewsItem(BaseModel):
    """뉴스 아이템"""
    title: str = Field(..., description="뉴스 제목")
    link: str = Field(..., description="뉴스 링크")
    pub_date: str = Field(..., description="발행일")
    description: str = Field(..., description="뉴스 요약")


class TrendAnalysis(BaseModel):
    """트렌드 분석"""
    period: str = Field(..., description="기간 (예: '11/1', '2025-01')")
    company_count: int = Field(..., description="해당 회사 공고 수")
    total_count: int = Field(..., description="전체 공고 수")
    market_share: float = Field(..., description="시장 점유율 (%)")
    change_rate: Optional[float] = Field(None, description="전 기간 대비 변화율 (%)")


class CauseAnalysis(BaseModel):
    """원인 분석"""
    possible_causes: List[str] = Field(..., description="가능한 원인 목록")
    news_evidence: List[NewsItem] = Field(default=[], description="관련 뉴스 증거")
    confidence: str = Field(..., description="신뢰도 (high/medium/low)")


class EvidenceReference(BaseModel):
    """근거 참조"""
    type: str = Field(..., description="근거 유형 ('news' 또는 'data')")
    source: str = Field(..., description="근거 출처 (뉴스 제목 또는 데이터 소스 설명)")
    link: Optional[str] = Field(None, description="뉴스 링크 (뉴스인 경우)")
    data_description: Optional[str] = Field(None, description="데이터 설명 (데이터인 경우)")
    date: Optional[str] = Field(None, description="데이터 날짜/기간 (data인 경우, 예: '2025-11-04', '11월 4주')")
    value: Optional[str] = Field(None, description="데이터 값 (data인 경우, 문자열로 표현, 예: '100', '289건')")


class StrategicInsight(BaseModel):
    """전략적 인사이트"""
    insight_type: str = Field(..., description="인사이트 유형 (예: 'aggressive_recruitment', 'market_expansion')")
    description: str = Field(..., description="인사이트 설명")
    implications: List[str] = Field(..., description="시사점 목록")
    evidence: List[EvidenceReference] = Field(default=[], description="인사이트의 근거 (뉴스 또는 데이터 참조)")


class CompanyInsightData(BaseModel):
    """회사 인사이트 데이터"""
    company_name: str = Field(..., description="회사명")
    company_id: int = Field(..., description="회사 ID")
    timeframe: str = Field(..., description="시간 단위 (daily/weekly/monthly)")
    period: PeriodInfo = Field(..., description="조회 기간")
    
    # 기본 통계
    total_postings: int = Field(..., description="해당 기간 총 공고 수")
    average_daily_postings: Optional[float] = Field(None, description="일평균 공고 수")
    
    # 경쟁사 비교
    competitor_comparison: List[CompetitorComparison] = Field(default=[], description="경쟁사 비교 정보")
    market_rank: int = Field(..., description="시장 내 순위")
    
    # 트렌드 분석
    trend_analysis: List[TrendAnalysis] = Field(default=[], description="기간별 트렌드 분석")
    
    # 원인 분석
    cause_analysis: Optional[CauseAnalysis] = Field(None, description="공고 수 변화 원인 분석")
    
    # 전략적 인사이트
    strategic_insights: List[StrategicInsight] = Field(default=[], description="전략적 인사이트")
    
    # LLM 생성 인사이트
    summary: str = Field(..., description="종합 요약")
    summary_evidence: List[EvidenceReference] = Field(default=[], description="요약의 근거 (뉴스 또는 데이터 참조)")
    key_findings: List[str] = Field(default=[], description="주요 발견 사항")
    key_findings_evidence: List[List[EvidenceReference]] = Field(default=[], description="각 key_finding의 근거 목록 (key_findings와 동일한 순서)")

    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "토스",
                "company_id": 1,
                "timeframe": "daily",
                "period": {
                    "start_date": "2025-11-01",
                    "end_date": "2025-11-30"
                },
                "total_postings": 450,
                "average_daily_postings": 15.0,
                "market_rank": 1,
                "competitor_comparison": [
                    {
                        "company_name": "네이버",
                        "total_count": 380,
                        "rank": 2,
                        "market_share": 8.5
                    }
                ],
                "summary": "토스는 최근 30일간 가장 활발한 채용 활동을 보였습니다...",
                "key_findings": [
                    "공고 수가 전 기간 대비 25% 증가",
                    "경쟁사 대비 시장 점유율 1위"
                ]
            }
        }

