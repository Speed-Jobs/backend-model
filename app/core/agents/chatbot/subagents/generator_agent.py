"""Generator Agent

Generates final answers based on retrieved information.
"""

import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.base_agent import BaseAgent
from app.prompts.system_prompts import GENERATOR_STATS_ONLY_PROMPT, GENERATOR_HYBRID_PROMPT


class GeneratorAgent(BaseAgent):
    """Agent responsible for generating final answers"""

    def __init__(self):
        super().__init__(model="gpt-4o", temperature=0.3)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate final answer based on search results and SQL analysis

        Args:
            state: Current state with search results and SQL analysis

        Returns:
            Updated state with 'answer' and 'sources'
        """
        question = state["question"]
        self.log("Generating answer")

        # Prepare context
        context = self._prepare_context(state)

        # Determine prompt based on result types
        is_stats_only = (
            state.get("route_decision") == "statistics_with_stats"
            and not state.get("vectordb_results")
        )

        system_prompt = GENERATOR_STATS_ONLY_PROMPT if is_stats_only else GENERATOR_HYBRID_PROMPT

        user_prompt = f"""질문: {question}

제공된 정보:
{context}

위 정보를 종합 분석하여 구조화된 답변과 인사이트를 제공해주세요."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            response = await self.llm.ainvoke(messages)
            state["answer"] = response.content

            # Collect all sources
            all_sources = []
            if state.get("vectordb_results"):
                all_sources.extend(state["vectordb_results"])
            if state.get("web_results"):
                all_sources.extend(state["web_results"])

            state["sources"] = all_sources
            self.log(f"Answer generated with {len(all_sources)} sources")

        except Exception as e:
            self.log(f"Error: {e}", "ERROR")
            state["answer"] = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            state["error"] = str(e)

        return state

    def _prepare_context(self, state: Dict[str, Any]) -> str:
        """Prepare context from search results and SQL analysis"""
        context_parts = []

        # SQL analysis results
        if state.get("sql_analysis"):
            sql_data = state["sql_analysis"]
            context_parts.append("=== 통계 분석 결과 ===")
            context_parts.append(f"분석 유형: {sql_data.get('query_type', 'N/A')}")
            context_parts.append(f"\n요약: {sql_data.get('summary', 'N/A')}")

            # Stats data table
            if sql_data.get('stats_data'):
                context_parts.append("\n통계 데이터:")
                for item in sql_data['stats_data'][:10]:
                    context_parts.append(f"  - {json.dumps(item, ensure_ascii=False)}")

            # Key findings
            if sql_data.get('key_findings'):
                context_parts.append("\n주요 발견사항:")
                for finding in sql_data['key_findings']:
                    context_parts.append(f"  • {finding}")

            # Insights
            if sql_data.get('insights'):
                context_parts.append(f"\n인사이트: {sql_data['insights']}")
            context_parts.append("")

        # VectorDB results
        if state.get("vectordb_results"):
            context_parts.append("=== 관련 채용 공고 ===")
            context_parts.append(f"총 {len(state['vectordb_results'])}개 공고 검색됨")
            for idx, doc in enumerate(state["vectordb_results"][:5], 1):
                context_parts.append(f"\n[공고 {idx}]")
                meta = doc.get('metadata', {})
                if meta.get('title'):
                    context_parts.append(f"제목: {meta['title']}")
                if meta.get('company_id'):
                    context_parts.append(f"회사 ID: {meta['company_id']}")
                if meta.get('employment_type'):
                    context_parts.append(f"고용형태: {meta['employment_type']}")
                if meta.get('experience'):
                    context_parts.append(f"경력: {meta['experience']}")
                if meta.get('work_type'):
                    context_parts.append(f"근무형태: {meta['work_type']}")
                context_parts.append(f"내용: {doc['text'][:300]}...")
                context_parts.append(f"유사도: {doc['score']:.3f}")

        # Web results
        if state.get("web_results"):
            context_parts.append("\n\n=== 웹 검색 결과 ===")
            for idx, doc in enumerate(state["web_results"][:5], 1):
                context_parts.append(f"\n[웹 문서 {idx}]")
                context_parts.append(f"제목: {doc['metadata'].get('title', 'N/A')}")
                context_parts.append(f"URL: {doc['metadata'].get('url', 'N/A')}")
                context_parts.append(f"내용: {doc['text'][:500]}...")

        return "\n".join(context_parts)
