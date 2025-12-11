"""Web Search Agent

Searches the web for relevant information using Tavily.
"""

from typing import Dict, Any
from app.agents.base_agent import BaseAgent
from app.tools.web_search import WebSearchTool


class WebSearchAgent(BaseAgent):
    """Agent responsible for web search"""

    def __init__(self):
        super().__init__()
        self.search_tool = WebSearchTool()

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search the web for relevant information

        Args:
            state: Current state with 'question' and 'top_k'

        Returns:
            Updated state with 'web_results'
        """
        question = state["question"]
        top_k = state.get("top_k", 5)

        self.log(f"Searching web: {question} (max_results={top_k})")

        try:
            results = await self.search_tool.search(
                query=question,
                max_results=top_k
            )

            state["web_results"] = results
            self.log(f"Found {len(results)} results")

        except Exception as e:
            self.log(f"Error: {e}", "ERROR")
            state["web_results"] = []
            if not state.get("error"):
                state["error"] = f"Web search failed: {str(e)}"

        return state
