"""Vector Search Tool

Tool for searching the VectorDB (Qdrant).
"""

from typing import List, Dict, Any, Optional
from app.core.agents.chatbot.retriever import Retriever


class VectorSearchTool:
    """Tool for VectorDB search operations"""

    def __init__(self):
        self.retriever = Retriever()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search VectorDB for similar documents

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional filters (e.g., {"company_id": 123})

        Returns:
            List of search results with metadata and scores
        """
        print(f"[VectorSearchTool] Searching: {query} (top_k={top_k})")

        try:
            results = await self.retriever.search(
                query=query,
                top_k=top_k,
                filters=filters if filters else None
            )

            # Format results with source_type
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result["id"],
                    "post_id": result.get("post_id"),
                    "text": result["text"],
                    "metadata": {
                        **result["metadata"],
                        "source_type": "vectordb"
                    },
                    "score": result["score"],
                    "source_type": "vectordb"
                })

            print(f"[VectorSearchTool] Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            print(f"[VectorSearchTool] Error: {e}")
            raise
