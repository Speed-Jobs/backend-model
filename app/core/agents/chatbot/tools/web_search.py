"""Web Search Tool

Tool for searching the web using Tavily API.
"""

from typing import List, Dict, Any
from tavily import TavilyClient
from app.core.config import settings


class WebSearchTool:
    """Tool for web search operations"""

    def __init__(self):
        self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)

    async def search(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the web for relevant information

        Args:
            query: Search query text
            max_results: Maximum number of results to return

        Returns:
            List of web search results with metadata
        """
        print(f"[WebSearchTool] Searching: {query} (max_results={max_results})")

        try:
            search_results = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=False,
                include_raw_content=False
            )

            # Format results
            formatted_results = []
            for idx, result in enumerate(search_results.get("results", [])):
                formatted_results.append({
                    "id": f"web_{idx}",
                    "text": result.get("content", ""),
                    "metadata": {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "source_type": "web",
                        "published_date": result.get("published_date")
                    },
                    "score": result.get("score", 0.0),
                    "source_type": "web"
                })

            print(f"[WebSearchTool] Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            print(f"[WebSearchTool] Error: {e}")
            raise
