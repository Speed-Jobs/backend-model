"""State Definitions

Defines the state structures used across the agent workflows.
"""

from typing import TypedDict, Optional, List, Literal, Any


class AgenticRAGState(TypedDict):
    """State for Agentic RAG workflow"""
    question: str
    top_k: int
    extracted_entities: Optional[dict]
    filters: Optional[dict]
    route_decision: Optional[Literal["vectordb", "websearch", "both", "statistics_with_stats"]]
    needs_stats: bool
    vectordb_results: Optional[List[dict]]
    web_results: Optional[List[dict]]
    sql_analysis: Optional[dict]
    answer: Optional[str]
    sources: Optional[List[dict]]
    error: Optional[str]
    db: Optional[Any]  # Database session for entity resolution
