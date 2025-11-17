"""
LLM 기반 평가 함수들 (Phase 1)
Agent가 직접 호출하지 않고, evaluators에서 데이터 수집 용도로 사용
"""

from .readability import (
    measure_company_jargon_frequency,
    measure_paragraph_consistency,
    measure_grammar_accuracy,
)

from .specificity import (
    measure_responsibility_specificity,
    measure_qualification_specificity,
    measure_keyword_relevance,
    measure_required_fields_count,
)

from .attractiveness import (
    measure_special_content_count,
    measure_special_content_quality,
)

__all__ = [
    # Readability
    "measure_company_jargon_frequency",
    "measure_paragraph_consistency",
    "measure_grammar_accuracy",
    # Specificity
    "measure_responsibility_specificity",
    "measure_qualification_specificity",
    "measure_keyword_relevance",
    "measure_required_fields_count",
    # Attractiveness
    "measure_special_content_count",
    "measure_special_content_quality",
]

