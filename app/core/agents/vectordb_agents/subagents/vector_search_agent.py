"""Vector Search Agent

Searches the VectorDB for relevant documents.
"""

from typing import Dict, Any
from app.core.agents.vectordb_agents.base_agent import BaseAgent
from app.core.agents.vectordb_agents.tools.vector_search import VectorSearchTool


class VectorSearchAgent(BaseAgent):
    """Agent responsible for vector database search"""

    def __init__(self):
        super().__init__()
        self.search_tool = VectorSearchTool()

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search VectorDB for relevant documents

        Args:
            state: Current state with 'question', 'top_k', and 'filters'

        Returns:
            Updated state with 'vectordb_results'
        """
        question = state["question"]
        top_k = state.get("top_k", 5)
        filters = state.get("filters") or {}

        self.log(f"Searching VectorDB: {question} (top_k={top_k})")

        try:
            results = await self.search_tool.search(
                query=question,
                top_k=top_k,
                filters=filters if filters else None
            )

            state["vectordb_results"] = results
            self.log(f"Found {len(results)} results")

        except Exception as e:
            self.log(f"Error: {e}", "ERROR")
            state["vectordb_results"] = []
            if not state.get("error"):
                state["error"] = f"Vector search failed: {str(e)}"

        return state
