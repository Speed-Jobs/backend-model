"""Vector Search Agent

Searches the VectorDB for relevant documents.
"""

from typing import Dict, Any
from app.core.agents.chatbot.base_agent import BaseAgent
from app.core.agents.chatbot.tools.vector_search import VectorSearchTool


class VectorSearchAgent(BaseAgent):
    """Agent responsible for vector database search"""

    def __init__(self):
        super().__init__()
        self.search_tool = VectorSearchTool()

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search VectorDB for relevant documents with query enhancement

        Args:
            state: Current state with 'question', 'top_k', 'filters', and 'extracted_entities'

        Returns:
            Updated state with 'vectordb_results'
        """
        question = state["question"]
        top_k = state.get("top_k", 5)
        extracted_entities = state.get("extracted_entities") or {}

        # Query Enhancement: 회사명을 쿼리에 강조해서 벡터 검색 정확도 향상
        enhanced_query = self._enhance_query(question, extracted_entities)
        
        if enhanced_query != question:
            self.log(f"🔍 Enhanced query: {enhanced_query}")

        self.log(f"Searching VectorDB (top_k={top_k})")

        try:
            # 회사명이 있으면 더 많이 검색 후 필터링
            search_top_k = top_k * 3 if extracted_entities.get("company_name") else top_k
            
            results = await self.search_tool.search(
                query=enhanced_query,
                top_k=search_top_k,
                filters=None  # 메타데이터 필터는 사용하지 않음
            )

            # 회사명으로 사후 필터링 (텍스트 기반)
            if extracted_entities.get("company_name"):
                results = self._filter_by_company_name(
                    results, 
                    extracted_entities["company_name"]
                )
                # top_k개만 유지
                results = results[:top_k]
                self.log(f"Filtered by company name: {len(results)} results")

            state["vectordb_results"] = results
            self.log(f"Final: {len(results)} results")

        except Exception as e:
            self.log(f"Error: {e}", "ERROR")
            state["vectordb_results"] = []
            if not state.get("error"):
                state["error"] = f"Vector search failed: {str(e)}"

        return state
    
    def _enhance_query(self, question: str, entities: Dict[str, Any]) -> str:
        """
        Enhance query by emphasizing company name for better vector search
        
        Args:
            question: Original question
            entities: Extracted entities
            
        Returns:
            Enhanced query with company name emphasized
        """
        company_name = entities.get("company_name")
        
        if not company_name:
            return question
        
        # 회사명을 쿼리에 여러 번 추가해서 임베딩에 강하게 반영
        # 예: "토스 채용공고" -> "토스 토스 토스 채용공고"
        enhanced = f"{company_name} {company_name} {company_name} {question}"
        
        return enhanced
    
    def _filter_by_company_name(
        self, 
        results: list, 
        company_name: str
    ) -> list:
        """
        Filter results by company name in text
        
        Args:
            results: Search results
            company_name: Company name to filter
            
        Returns:
            Filtered results
        """
        company_lower = company_name.lower()
        filtered = []
        
        for result in results:
            text = result.get("text", "").lower()
            
            # 텍스트에 회사명이 포함된 경우만 유지
            if company_lower in text or f"[{company_lower}]" in text or f"({company_lower})" in text:
                filtered.append(result)
        
        self.log(f"📊 Company filter: {len(results)} → {len(filtered)} results")
        
        return filtered
