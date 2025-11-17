"""
Readability Evaluator (Phase 1: Data Collection)
가독성 평가 데이터 수집 모듈
"""

from typing import Dict, Any
import asyncio
import os
from dotenv import load_dotenv
from app.utils.evaluation.llm_functions.readability import (
    measure_company_jargon_frequency,
    measure_paragraph_consistency,
    measure_grammar_accuracy
)

load_dotenv(override=True)


async def collect_readability_data(
    job_description: str,
    company_name: str,
    title: str
) -> Dict[str, Any]:
    """
    가독성 평가 데이터 수집 (병렬 처리)

    Args:
        job_description: 채용 공고 내용
        company_name: 회사명
        title: 채용 공고 제목

    Returns:
        Dict: 각 tool의 원형 결과
            {
                "jargon": "=== 사내 전문 용어 빈도수 평가 ===...",
                "consistency": "=== 문단 일관성 평가 ===...",
                "grammar": "=== 문법 정확성 평가 ===..."
            }
    """
    print(f"[Readability Evaluator] Starting data collection for: {title}")

    try:
        # 모든 tool을 병렬로 호출
        jargon_result, consistency_result, grammar_result = await asyncio.gather(
            asyncio.to_thread(
                measure_company_jargon_frequency.invoke,
                {"job_description": job_description, "company_name": company_name}
            ),
            asyncio.to_thread(
                measure_paragraph_consistency.invoke,
                {"job_description": job_description}
            ),
            asyncio.to_thread(
                measure_grammar_accuracy.invoke,
                {"job_description": job_description}
            ),
            return_exceptions=True
        )

        # 예외 처리
        results = {}
        
        if isinstance(jargon_result, Exception):
            print(f"[Readability Evaluator] Jargon tool failed: {jargon_result}")
            results["jargon"] = f"평가 실패: {str(jargon_result)}"
        else:
            results["jargon"] = jargon_result
            print(f"[Readability Evaluator] Jargon completed")

        if isinstance(consistency_result, Exception):
            print(f"[Readability Evaluator] Consistency tool failed: {consistency_result}")
            results["consistency"] = f"평가 실패: {str(consistency_result)}"
        else:
            results["consistency"] = consistency_result
            print(f"[Readability Evaluator] Consistency completed")

        if isinstance(grammar_result, Exception):
            print(f"[Readability Evaluator] Grammar tool failed: {grammar_result}")
            results["grammar"] = f"평가 실패: {str(grammar_result)}"
        else:
            results["grammar"] = grammar_result
            print(f"[Readability Evaluator] Grammar completed")

        print(f"[Readability Evaluator] Data collection completed for: {title}")
        return results

    except Exception as e:
        print(f"[Readability Evaluator] Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

