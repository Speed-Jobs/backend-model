"""
Agent Schemas
에이전트 관련 스키마 정의
"""

from .schemas_tool import (
    # Tool Input Schema
    ToolInput,
    # Output Schemas
    JargonResult,
    ConsistencyResult,
    GrammarResult,
    ResponsibilityResult,
    QualificationResult,
    KeywordRelevanceResult,
    RequiredFieldsResult,
    SpecialContentInclusionResult,
    SpecialContentQualityResult,
)

__all__ = [
    # Input Schema
    "ToolInput",
    # Output Schemas
    "JargonResult",
    "ConsistencyResult",
    "GrammarResult",
    "ResponsibilityResult",
    "QualificationResult",
    "KeywordRelevanceResult",
    "RequiredFieldsResult",
    "SpecialContentInclusionResult",
    "SpecialContentQualityResult",
]
