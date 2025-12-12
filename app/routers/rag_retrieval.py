"""RAG (Retrieval-Augmented Generation) Router

Executes Agentic RAG using local orchestrator
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel
import os
from app.core.agents.chatbot.orchestrator import AgenticRAGOrchestrator
from app.core.agents.chatbot.memory.states import AgenticRAGState


# Request/Response Schemas
class AgenticRAGQuery(BaseModel):
    """Agentic RAG query request"""
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "2025년 하반기 토스 채용공고 총 몇개야?"
            }
        }


class AgenticRAGResponse(BaseModel):
    """Agentic RAG response"""
    query: str
    answer: str
    sources: list
    route_decision: Optional[str] = None
    total_sources: int

    class Config:
        json_schema_extra = {
            "example": {
                "query": "2025년 하반기 토스 채용공고 총 몇개야?",
                "answer": "2025년 하반기 토스 채용공고는 총 25개입니다...",
                "sources": [],
                "route_decision": "statistics_with_stats",
                "total_sources": 0
            }
        }


router = APIRouter(prefix="/rag", tags=["RAG"])

# Initialize Agentic RAG Orchestrator (singleton)
_orchestrator: Optional[AgenticRAGOrchestrator] = None

def get_orchestrator() -> AgenticRAGOrchestrator:
    """Get or create orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgenticRAGOrchestrator()
    return _orchestrator

@router.post("/search", response_model=AgenticRAGResponse)
async def agentic_rag_search(query: AgenticRAGQuery):
    """
    Agentic RAG 검색 - 로컬 AI Agent 실행

    이 엔드포인트는 backend-model의 Agentic RAG Orchestrator를 사용합니다.
    VectorDB(Qdrant)에 직접 연결하여 검색을 수행합니다.

    ## 지능형 자동 처리
    - **엔티티 추출**: 회사명, 날짜, 직무, 고용형태 등 자동 인식
    - **라우팅 결정**: vectordb / websearch / both / statistics_with_stats 자동 선택
    - **통계 분석**: SQL 통계 분석 자동 실행

    ## 사용 예시
    ```json
    {
        "text": "2025년 하반기 토스 채용공고 총 몇개야?"
    }
    ```

    ## Response
    - **query**: 입력 질문
    - **answer**: LLM이 생성한 답변
    - **sources**: 검색된 소스 문서들
    - **route_decision**: 라우팅 결정
    - **total_sources**: 전체 소스 개수
    """
    try:
        # Get orchestrator instance
        orchestrator = get_orchestrator()
        
        # Create initial state
        initial_state: AgenticRAGState = {
            "question": query.text,
            "top_k": 5,
            "extracted_entities": None,
            "filters": None,
            "route_decision": None,
            "needs_stats": False,
            "vectordb_results": None,
            "web_results": None,
            "sql_analysis": None,
            "answer": None,
            "sources": None,
            "error": None
        }
        
        # Execute workflow
        result_state = await orchestrator.execute(initial_state)
        
        # Format response
        return AgenticRAGResponse(
            query=result_state["question"],
            answer=result_state.get("answer", "답변을 생성할 수 없습니다."),
            sources=result_state.get("sources", []),
            route_decision=result_state.get("route_decision"),
            total_sources=len(result_state.get("sources", []))
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agentic RAG failed: {str(e)}"
        )


@router.get("/health")
async def rag_health_check():
    """
    RAG system health check

    Checks local orchestrator and Qdrant connection
    """
    try:
        from app.core.agents.chatbot.vectordb import QdrantClientWrapper
        from app.core.config import settings
        
        # Check Qdrant connection
        vectordb = QdrantClientWrapper()
        vector_count = vectordb.count()
        
        # Check orchestrator
        orchestrator = get_orchestrator()
        
        return {
            "status": "healthy",
            "orchestrator": "initialized",
            "qdrant": {
                "status": "connected",
                "url": settings.QDRANT_URL,
                "collection": settings.QDRANT_COLLECTION_NAME,
                "vectors_count": vector_count
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
