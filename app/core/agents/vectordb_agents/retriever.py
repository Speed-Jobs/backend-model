"""Retriever for Vector Search

Handles embedding generation and vector similarity search.
"""

from typing import List, Dict, Any, Optional
from app.core.agents.vectordb_agents.embedder import Embedder
from app.core.agents.vectordb_agents.vectordb import QdrantClientWrapper


class Retriever:
    """Vector search retriever using Qdrant"""

    def __init__(self):
        """Initialize retriever with embedder and vector DB"""
        self.embedder = Embedder()
        self.vectordb = QdrantClientWrapper()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in VectorDB

        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Optional filters (e.g., {"post_id": 123})

        Returns:
            List of search results with metadata and scores
        """
        # Generate embedding for query
        query_embedding = await self.embedder.embed(query)

        if not query_embedding:
            return []

        # Search in VectorDB
        results = self.vectordb.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )

        # Format results with metadata
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result["id"],
                "post_id": result.get("post_id"),
                "text": result["text"],
                "metadata": {
                    "post_id": result.get("post_id"),
                },
                "score": result["score"]
            })

        return formatted_results
