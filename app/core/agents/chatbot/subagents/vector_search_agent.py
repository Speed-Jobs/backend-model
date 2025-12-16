"""Vector Search Agent

Searches the VectorDB for relevant documents.
"""

from typing import Dict, Any
from app.core.agents.chatbot.base_agent import BaseAgent
from app.core.agents.chatbot.tools.vector_search import VectorSearchTool
from app.core.agents.chatbot.tools.reranker import Reranker


class VectorSearchAgent(BaseAgent):
    """Agent responsible for vector database search"""

    def __init__(self):
        super().__init__()
        self.search_tool = VectorSearchTool()
        self.reranker = Reranker()

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search VectorDB for relevant documents with reranking

        Args:
            state: Current state with 'question', 'top_k', 'filters', and 'extracted_entities'

        Returns:
            Updated state with 'vectordb_results'
        """
        question = state["question"]
        top_k = state.get("top_k", 5)
        filters = state.get("filters") or {}
        extracted_entities = state.get("extracted_entities") or {}

        self.log(f"Searching VectorDB: {question} (top_k={top_k})")
        if filters:
            self.log(f"Filters: {filters}")

        try:
            # Search with higher top_k for reranking
            search_top_k = top_k * 3 if extracted_entities.get("company_name") else top_k
            
            results = await self.search_tool.search(
                query=question,
                top_k=search_top_k,
                filters=filters if filters else None
            )

            self.log(f"Found {len(results)} results before reranking")
            
            # Rerank results if entities are available
            if results and extracted_entities:
                results = self.reranker.rerank(
                    results=results,
                    extracted_entities=extracted_entities,
                    query=question
                )
                # Keep only top_k after reranking
                results = results[:top_k]
                self.log(f"After reranking: {len(results)} results")

            state["vectordb_results"] = results
            self.log(f"Final: {len(results)} results")

        except Exception as e:
            self.log(f"Error: {e}", "ERROR")
            state["vectordb_results"] = []
            if not state.get("error"):
                state["error"] = f"Vector search failed: {str(e)}"

        return state
