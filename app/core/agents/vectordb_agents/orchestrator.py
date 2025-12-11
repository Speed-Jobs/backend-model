"""Agentic RAG Orchestrator

Coordinates multiple specialized agents using LangGraph workflow.
"""

from langgraph.graph import StateGraph, END

from app.core.agents.vectordb_agents.memory.states import AgenticRAGState
from app.core.agents.vectordb_agents.subagents.router_agent import RouterAgent
from app.core.agents.vectordb_agents.subagents.vector_search_agent import VectorSearchAgent
from app.core.agents.vectordb_agents.subagents.web_search_agent import WebSearchAgent
from app.core.agents.vectordb_agents.subagents.sql_analysis_agent import SQLAnalysisAgent
from app.core.agents.vectordb_agents.subagents.generator_agent import GeneratorAgent


class AgenticRAGOrchestrator:
    """Orchestrates the Agentic RAG workflow using LangGraph"""

    def __init__(self):
        """Initialize orchestrator with specialized agents"""
        self.router = RouterAgent()
        self.vector_searcher = VectorSearchAgent()
        self.web_searcher = WebSearchAgent()
        self.sql_analyzer = SQLAnalysisAgent()
        self.generator = GeneratorAgent()

        # Build workflow graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(AgenticRAGState)

        # Add nodes
        workflow.add_node("router", self.router.execute)
        workflow.add_node("vector_retriever", self.vector_searcher.execute)
        workflow.add_node("web_searcher", self.web_searcher.execute)
        workflow.add_node("sql_analyzer", self.sql_analyzer.execute)
        workflow.add_node("generator", self.generator.execute)

        # Set entry point
        workflow.set_entry_point("router")

        # Conditional edges from router
        def route_after_router(state):
            decision = state["route_decision"]
            needs_stats = state.get("needs_stats", False)

            if decision == "websearch":
                return "web_searcher"
            elif decision == "statistics_with_stats" or needs_stats:
                return "sql_analyzer"
            else:
                # vectordb or both → vector_retriever first
                return "vector_retriever"

        workflow.add_conditional_edges(
            "router",
            route_after_router,
            {
                "vector_retriever": "vector_retriever",
                "web_searcher": "web_searcher",
                "sql_analyzer": "sql_analyzer"
            }
        )

        # Vector retriever routing
        def route_after_vector(state):
            decision = state["route_decision"]
            needs_stats = state.get("needs_stats", False)

            if needs_stats or decision == "statistics_with_stats":
                return "sql_analyzer"
            elif decision == "both":
                return "web_searcher"
            else:
                return "generator"

        workflow.add_conditional_edges(
            "vector_retriever",
            route_after_vector,
            {
                "sql_analyzer": "sql_analyzer",
                "web_searcher": "web_searcher",
                "generator": "generator"
            }
        )

        # SQL analyzer → generator
        workflow.add_edge("sql_analyzer", "generator")

        # Web searcher → generator
        workflow.add_edge("web_searcher", "generator")

        # Generator → END
        workflow.add_edge("generator", END)

        # Compile and return
        return workflow.compile()

    async def execute(self, state: AgenticRAGState) -> AgenticRAGState:
        """
        Execute the Agentic RAG workflow

        Args:
            state: Initial state with question and parameters

        Returns:
            Final state with answer and sources
        """
        print(f"\n[AgenticRAGOrchestrator] Starting workflow for: {state['question']}")
        result = await self.graph.ainvoke(state)
        print(f"[AgenticRAGOrchestrator] Workflow completed\n")
        return result
