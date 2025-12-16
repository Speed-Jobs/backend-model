"""Router Agent

Analyzes user queries to extract entities and determine optimal search strategy.
"""

from typing import Dict, Any, Literal
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.agents.chatbot.base_agent import BaseAgent
from app.core.agents.chatbot.prompts.system_prompts import ROUTER_SYSTEM_PROMPT
from app.core.agents.chatbot.tools.helpers import extract_json_from_response
from app.utils.query_logger import get_query_logger


class RouterAgent(BaseAgent):
    """Agent responsible for routing queries and extracting entities"""

    def __init__(self):
        super().__init__(model="gpt-4o", temperature=0.0)
        self.query_logger = get_query_logger()

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze question and determine:
        1. Entity extraction (company name, dates, job category, etc.)
        2. Optimal search source (vectordb/websearch/both/statistics_with_stats)
        3. Appropriate top_k value

        Args:
            state: Current state with 'question' key

        Returns:
            Updated state with routing decision and extracted entities
        """
        question = state["question"]
        self.log(f"Routing query: {question}")

        # Check for statistical intent using heuristics
        stats_intent = self._has_stats_intent(question)

        user_prompt = f"""질문: {question}

위 질문을 분석하여:
1. 엔티티를 추출하고
2. 최적의 검색 소스를 결정하고
3. 적절한 top_k 값을 제안하세요.

반드시 순수 JSON만 반환하세요."""

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            result = extract_json_from_response(response.content)

            # Extract entities
            entities = result.get("entities", {})
            state["extracted_entities"] = entities

            # Routing decision
            route = result.get("route", "vectordb")
            needs_stats = result.get("needs_stats", False)

            # Stats-only route forces SQL analysis
            if route == "statistics_with_stats":
                needs_stats = True

            top_k = result.get("top_k", 5)

            # Heuristic correction: force stats route if intent is clear
            if stats_intent and route != "statistics_with_stats":
                route = "statistics_with_stats"
                needs_stats = True

            # Validation
            valid_routes = ["vectordb", "websearch", "both", "statistics_with_stats"]
            if route not in valid_routes:
                route = "vectordb"

            if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
                top_k = 5

            state["route_decision"] = route
            state["needs_stats"] = needs_stats
            state["top_k"] = top_k

            self.log(f"Decision: {route} (stats: {needs_stats}, top_k: {top_k})")
            self.log(f"Entities: {entities}")
            self.log(f"Reason: {result.get('reason', 'N/A')}")

            # 라우팅 결정 로그 저장
            self.query_logger.log_routing_decision(
                question=question,
                route_decision=route,
                extracted_entities=entities,
                needs_stats=needs_stats,
                top_k=top_k,
                reason=result.get('reason', 'N/A'),
                llm_response=response.content
            )

        except Exception as e:
            self.log(f"Error: {e}, defaulting based on stats intent", "ERROR")
            # Fallback to stats route if intent is clear
            if stats_intent:
                state["route_decision"] = "statistics_with_stats"
                state["needs_stats"] = True
            else:
                state["route_decision"] = "vectordb"
                state["needs_stats"] = False
            state["top_k"] = 5
            state["extracted_entities"] = {}

        return state

    def _has_stats_intent(self, question: str) -> bool:
        """Detect if question has statistical/aggregation intent"""
        if not question:
            return False

        stats_keywords = [
            "통계", "집계", "평균", "비율", "분포",
            "몇 개", "몇개", "몇 건", "몇건", "총 몇",
            "개수", "수량", "top", "TOP", "순위", "랭킹",
            "가장 많이", "많이 요구", "분기", "월별", "주별",
            "기간별", "추이", "증감", "트렌드",
            "count", "average", "ratio", "distribution", "stat", "statistics"
        ]

        q_lower = question.lower()
        return any(kw in question or kw.lower() in q_lower for kw in stats_keywords)
