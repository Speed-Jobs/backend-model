"""SQL Analysis Agent

Performs statistical analysis using SQL queries.
"""

import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.base_agent import BaseAgent
from app.prompts.system_prompts import SQL_GENERATION_PROMPT_TEMPLATE, INSIGHT_GENERATION_PROMPT_TEMPLATE
from app.tools.database_query import DatabaseQueryTool
from app.tools.helpers import extract_json_from_response


class SQLAnalysisAgent(BaseAgent):
    """Agent responsible for SQL statistical analysis"""

    def __init__(self):
        super().__init__(model="gpt-4o", temperature=0.0)
        self.db_tool = DatabaseQueryTool()

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform SQL statistical analysis:
        1. Generate SQL query using LLM
        2. Execute query on database
        3. Generate insights from results

        Args:
            state: Current state with 'question' and 'extracted_entities'

        Returns:
            Updated state with 'sql_analysis'
        """
        question = state["question"]
        self.log(f"Analyzing: {question}")

        try:
            # Step 1: Generate SQL query
            self.log("Step 1: Generating SQL query...")
            sql_result = await self._generate_sql_query(state)

            sql_query = sql_result.get("sql", "")
            query_type = sql_result.get("query_type", "통계 분석")

            self.log(f"Generated SQL ({query_type}): {sql_query[:100]}...")

            # Step 2: Execute query
            self.log("Step 2: Executing query on database...")
            stats_data = self.db_tool.execute_query(sql_query)
            columns = list(stats_data[0].keys()) if stats_data else []

            self.log(f"Query executed: {len(stats_data)} rows returned")

            # Step 3: Generate insights
            self.log("Step 3: Generating insights from statistics...")
            analysis_data = await self._generate_insights(
                question=question,
                query_type=query_type,
                stats_data=stats_data,
                columns=columns
            )

            # Final result
            state["sql_analysis"] = {
                "query_type": query_type,
                "sql_query": sql_query,
                "stats_data": stats_data,
                "data_columns": columns,
                "data_count": len(stats_data),
                "summary": analysis_data.get("summary", ""),
                "key_findings": analysis_data.get("key_findings", []),
                "insights": analysis_data.get("insights", "")
            }

            self.log(f"Analysis completed successfully")

        except Exception as e:
            self.log(f"Error: {e}", "ERROR")
            import traceback
            traceback.print_exc()

            state["sql_analysis"] = {
                "query_type": "분석",
                "sql_query": "",
                "stats_data": [],
                "data_columns": [],
                "data_count": 0,
                "summary": f"통계 분석 중 오류가 발생했습니다: {str(e)}",
                "key_findings": [],
                "insights": "MySQL 집계 쿼리 실행 중 문제가 발생했습니다."
            }
            if not state.get("error"):
                state["error"] = f"SQL analysis failed: {str(e)}"

        return state

    async def _generate_sql_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SQL query using LLM"""
        entities = state.get("extracted_entities", {})
        entity_info = []

        if entities:
            if entities.get("company_name"):
                entity_info.append(f"회사명: {entities['company_name']}")
            if entities.get("year"):
                entity_info.append(f"연도: {entities['year']}")
            if entities.get("period"):
                entity_info.append(f"기간: {entities['period']}")
            if entities.get("job_category"):
                entity_info.append(f"직무: {entities['job_category']}")
            if entities.get("employment_type"):
                entity_info.append(f"고용형태: {entities['employment_type']}")

        entity_context = "\n".join(entity_info) if entity_info else "필터 없음"

        sql_prompt = SQL_GENERATION_PROMPT_TEMPLATE.format(
            entity_context=entity_context
        )

        messages = [
            SystemMessage(content=sql_prompt),
            HumanMessage(
                content=f"질문: {state['question']}\n\n위 질문에 맞는 통계 분석 SQL 쿼리를 생성하세요. 반드시 순수 JSON만 반환하세요."
            )
        ]

        response = await self.llm.ainvoke(messages)
        return extract_json_from_response(response.content)

    async def _generate_insights(
        self,
        question: str,
        query_type: str,
        stats_data: list,
        columns: list
    ) -> Dict[str, Any]:
        """Generate insights from statistical data"""
        insight_prompt = INSIGHT_GENERATION_PROMPT_TEMPLATE.format(
            question=question,
            query_type=query_type,
            data_count=len(stats_data),
            columns=columns,
            stats_data=json.dumps(stats_data, ensure_ascii=False, indent=2)
        )

        messages = [
            SystemMessage(
                content="당신은 데이터 분석 전문가로서 통계를 해석하고 실용적인 인사이트를 도출합니다. 순수 JSON만 반환하세요."
            ),
            HumanMessage(content=insight_prompt)
        ]

        response = await self.llm.ainvoke(messages)
        return extract_json_from_response(response.content)
