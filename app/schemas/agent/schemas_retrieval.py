from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from app.schemas.ingestion import PostData


class SearchQuery(BaseModel):
    """검색 쿼리"""
    text: str = Field(..., description="검색할 텍스트")
    top_k: int = Field(5, ge=1, le=50, description="반환할 결과 수")
    company_name: Optional[str] = Field(None, description="회사명으로 필터링 (예: toss, kakao). 미입력시 전체 검색")
    filters: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터 필터")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "백엔드 개발자 채용",
                "top_k": 5,
                "company_name": "카카오"
            }
        }

class SearchResult(BaseModel):
    """검색 결과"""
    id: str = Field(..., description="문서 ID")
    post_id: int = Field(..., description="원본 Post ID")
    text: str = Field(..., description="문서 텍스트")
    metadata: Dict[str, Any] = Field(..., description="메타데이터 (Post 테이블의 전체 데이터 포함)")
    distance: float = Field(..., description="거리 (낮을수록 유사)")
    score: float = Field(..., description="유사도 점수 (높을수록 유사)")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "post_123_chunk_0",
                "post_id": 123,
                "text": "제목: 백엔드 개발자 채용\n고용형태: 정규직...",
                "metadata": {
                    "post_id": 123,
                    "company_id": 456,
                    "employment_type": "정규직",
                    "chunk_index": 0,
                    "title": "백엔드 개발자 채용",
                    "description": "상세 설명...",
                    "experience": "3년 이상",
                    "work_type": "정규직",
                    "posted_at": "2024-01-01T00:00:00",
                    "source_url": "https://..."
                },
                "distance": 0.234,
                "score": 0.766
            }
        }


class SearchResponse(BaseModel):
    """검색 응답"""
    query: str
    results: list[SearchResult]
    total_results: int


class AgenticRAGQuery(BaseModel):
    """
    Agentic RAG 검색 쿼리

    LLM이 자동으로 처리:
    - 엔티티 추출 (회사명, 날짜, 직무 등)
    - 검색 소스 결정 (vectordb/websearch/both/stats)
    - 결과 개수 자동 결정
    """
    text: str = Field(..., description="자연어 질문 (LLM이 자동으로 분석)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "2025년 하반기 토스 채용공고 총 몇개야?"
            }
        }


class SourceDocument(BaseModel):
    """검색 소스 문서"""
    id: str = Field(..., description="문서 ID")
    text: str = Field(..., description="문서 텍스트")
    metadata: Dict[str, Any] = Field(..., description="메타데이터")
    score: Optional[float] = Field(None, description="검색 점수")
    source_type: str = Field(..., description="소스 타입 (vectordb 또는 web)")


class AgenticRAGResponse(BaseModel):
    """Agentic RAG 응답"""
    query: str = Field(..., description="검색 쿼리")
    answer: str = Field(..., description="LLM이 생성한 답변")
    sources: list[SourceDocument] = Field(..., description="검색된 소스 문서들")
    route_decision: str = Field(..., description="라우팅 결정 (vectordb/websearch/both)")
    total_sources: int = Field(..., description="전체 소스 개수")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "카카오 백엔드 채용 요구사항",
                "answer": "카카오의 백엔드 개발자 채용 요구사항은 다음과 같습니다...",
                "sources": [
                    {
                        "id": "post_8349_chunk_0",
                        "text": "...",
                        "metadata": {},
                        "score": 0.85,
                        "source_type": "vectordb"
                    }
                ],
                "route_decision": "both",
                "total_sources": 8
            }
        }
