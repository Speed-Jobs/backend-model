"""
Job Description Evaluation Tools Schemas
LLM structured output 및 Tool Input을 위한 Pydantic 모델 정의
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ============ Tool Input Schemas ============

class ToolInput(BaseModel):
    """Tool 입력 스키마"""
    job_description: str = Field(..., description="채용 공고 전체 텍스트")
    company_name: str = Field(default="", description="회사명 (선택)")


# ============ Common Base Schemas ============

class BaseEvaluationResult(BaseModel):
    """평가 결과 공통 베이스 스키마"""
    original_text: str = Field(
        description="기준에 해당되는 원문 텍스트"
    )
    keywords: List[str] = Field(
        description="기준에 해당되는 키워드 리스트"
    )
    keyword_count: int = Field(
        description="키워드 개수"
    )
    reasoning: str = Field(
        description="LLM이 이렇게 판단한 근거 및 설명"
    )


# ============ Readability Output Schemas ============

class JargonResult(BaseEvaluationResult):
    """사내 전문 용어 추출 결과"""
    keywords: List[str] = Field(
        description="기준에 해당되는 키워드 리스트 (회사 고유 용어, 사내 시스템명, 특별한 프로그램명)"
    )


class ConsistencyResult(BaseEvaluationResult):
    """문단 일관성 분석 결과"""
    keywords: List[str] = Field(
        description="기준에 해당되는 키워드 리스트 (불일치 문제가 있는 섹션명 또는 문제 키워드)"
    )


class GrammarResult(BaseEvaluationResult):
    """문법 정확성 분석 결과"""
    keywords: List[str] = Field(
        description="기준에 해당되는 키워드 리스트 (문법 오류가 있는 표현 또는 오류 유형)"
    )


# ============ Specificity Schemas ============

class ResponsibilityResult(BaseEvaluationResult):
    """담당 업무 구체성 분석 결과"""
    keywords: List[str] = Field(
        description="기준에 해당되는 키워드 리스트 (기술 용어 또는 담당 업무 관련 키워드)"
    )


class QualificationResult(BaseEvaluationResult):
    """필요 역량 및 경험 구체성 분석 결과"""
    keywords: List[str] = Field(
        description="기준에 해당되는 키워드 리스트 (자격요건 또는 우대사항 관련 키워드)"
    )


class KeywordRelevanceResult(BaseEvaluationResult):
    """직군 관련 키워드 적합성 분석 결과"""
    keywords: List[str] = Field(
        description="기준에 해당되는 키워드 리스트 (담당 업무의 주요 키워드)"
    )


class RequiredFieldsResult(BaseEvaluationResult):
    """필수 항목 포함 여부 결과"""
    keywords: List[str] = Field(
        description="기준에 해당되는 키워드 리스트 (포함된 필수 항목명 또는 누락된 항목명)"
    )


# ============ Attractiveness Schemas ============

class SpecialContentInclusionResult(BaseEvaluationResult):
    """특별 콘텐츠 포함 여부 결과"""
    keywords: List[str] = Field(
        description="기준에 해당되는 키워드 리스트 (포함된 특별 콘텐츠 유형 또는 누락된 콘텐츠 유형)"
    )


class SpecialContentQualityResult(BaseEvaluationResult):
    """특별 콘텐츠 텍스트 추출 결과"""
    keywords: List[str] = Field(
        description="기준에 해당되는 키워드 리스트 (특별 콘텐츠 관련 키워드 또는 콘텐츠 유형)"
    )


# ============ Module Level Response Schemas ============

class ReadabilityModuleResult(BaseModel):
    """가독성 평가 모듈 결과 (3개 evaluator 통합)"""
    jargon: JargonResult = Field(
        description="사내 전문 용어 빈도수 평가 결과"
    )
    consistency: ConsistencyResult = Field(
        description="문단 일관성 평가 결과"
    )
    grammar: GrammarResult = Field(
        description="문법 정확성 평가 결과"
    )


class SpecificityModuleResult(BaseModel):
    """구체성 평가 모듈 결과 (4개 evaluator 통합)"""
    responsibility: ResponsibilityResult = Field(
        description="담당 업무 구체성 평가 결과"
    )
    qualification: QualificationResult = Field(
        description="자격요건 구체성 평가 결과"
    )
    keyword_relevance: KeywordRelevanceResult = Field(
        description="키워드 적합성 평가 결과"
    )
    required_fields: RequiredFieldsResult = Field(
        description="필수 항목 포함 여부 평가 결과"
    )


class AttractivenessModuleResult(BaseModel):
    """매력도 평가 모듈 결과 (2개 evaluator 통합)"""
    content_count: SpecialContentInclusionResult = Field(
        description="특별 콘텐츠 포함 여부 평가 결과"
    )
    content_quality: SpecialContentQualityResult = Field(
        description="특별 콘텐츠 충실도 평가 결과"
    )


# ============ API Response Schema ============
class EvaluationResponse(BaseModel):
    """평가 결과 API 응답"""
    readability: ReadabilityModuleResult = Field(
        description="가독성 평가 결과"
    )
    specificity: SpecificityModuleResult = Field(
        description="구체성 평가 결과"
    )
    attractiveness: AttractivenessModuleResult = Field(
        description="매력도 평가 결과"
    )