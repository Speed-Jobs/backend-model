"""Agentic RAG Service Layer"""

from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from app.agents.orchestrator import AgenticRAGOrchestrator
from app.schemas.retrieval import AgenticRAGResponse, SourceDocument
from app.schemas.agent.schemas_retrieval import PostData
from app.models.company import Company
from app.models.post import Post


class AgenticRAGService:
    """Agentic RAG 비즈니스 로직"""

    def __init__(self):
        self.orchestrator = AgenticRAGOrchestrator()

    async def search(
        self,
        query: str,
        db: Optional[Session] = None
    ) -> AgenticRAGResponse:
        """
        Agentic RAG 검색 수행

        LLM이 자동으로:
        - 사용자 질문에서 엔티티(회사명, 날짜 등) 추출
        - 최적의 검색 소스 결정 (vectordb/websearch/both/stats)
        - 필요한 결과 수(top_k) 자동 결정

        Args:
            query: 사용자의 자연어 질문
            db: 데이터베이스 세션 (회사명→company_id 변환 및 Post 데이터 조회용)

        Returns:
            AgenticRAGResponse: 검색 결과 + 생성된 답변
        """
        # State 초기화 (최소한의 정보만)
        initial_state = {
            "question": query,
            "top_k": 5,  # 벡터/웹 검색용 (통계 쿼리는 top_k 무시하고 전체 집계)
            "extracted_entities": None,
            "filters": None,
            "route_decision": None,
            "needs_stats": False,
            "vectordb_results": None,
            "web_results": None,
            "sql_analysis": None,
            "answer": None,
            "sources": None,
            "error": None,
            "db": db  # DB session for entity resolution (company_name -> company_id)
        }

        # Orchestrator 실행 (LLM이 자동으로 엔티티 추출, 라우팅, top_k 결정)
        print(f"\n[AgenticRAG] Starting search for: {query}")
        result = await self.orchestrator.execute(initial_state)

        # 디버깅: 추출된 엔티티 출력
        if result.get("extracted_entities"):
            print(f"[AgenticRAG] Extracted entities: {result['extracted_entities']}")

        # VectorDB 결과에 Post 전체 데이터 추가 (기존 search API와 동일한 방식)
        if result.get("vectordb_results") and db:
            enriched_vectordb_results = []
            for vdb_result in result["vectordb_results"]:
                # post_id 추출 (metadata 또는 최상위 레벨에서)
                post_id = vdb_result.get("post_id") or vdb_result.get("metadata", {}).get("post_id")

                if post_id:
                    # DB에서 Post 전체 데이터 조회
                    post = db.query(Post).filter(Post.id == post_id).first()
                    if post:
                        # Post 데이터를 dict로 변환 (id 필드는 제외 - post_id와 중복)
                        post_data = PostData.model_validate(post).model_dump(exclude_none=True, exclude={'id'})
                        # 기존 metadata에 post 데이터 병합
                        vdb_result["metadata"].update(post_data)

                enriched_vectordb_results.append(vdb_result)

            # enriched 결과로 교체
            result["vectordb_results"] = enriched_vectordb_results

            # sources도 업데이트 (vectordb_results + web_results)
            all_sources = []
            if result.get("vectordb_results"):
                all_sources.extend(result["vectordb_results"])
            if result.get("web_results"):
                all_sources.extend(result["web_results"])
            result["sources"] = all_sources

        # 소스 문서 변환
        source_docs = []
        if result.get("sources"):
            for source in result["sources"]:
                source_docs.append(SourceDocument(
                    id=source["id"],
                    text=source["text"],
                    metadata=source["metadata"],
                    score=source.get("score"),
                    source_type=source["source_type"]
                ))

        # 응답 생성
        response = AgenticRAGResponse(
            query=query,
            answer=result.get("answer", "답변을 생성할 수 없습니다."),
            sources=source_docs,
            route_decision=result.get("route_decision", "unknown"),
            total_sources=len(source_docs)
        )

        print(f"[AgenticRAG] Completed with {response.total_sources} sources\n")
        return response
