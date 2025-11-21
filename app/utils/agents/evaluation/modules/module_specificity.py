"""
Specificity Module (Phase 1: Data Collection)
구체성 평가 데이터 수집 모듈
"""

from typing import Dict, Any
import asyncio
import os
from dotenv import load_dotenv
from app.utils.agents.evaluation.evaluators.evaluator_specificity import (
    measure_responsibility_specificity,
    measure_qualification_specificity,
    measure_keyword_relevance,
    measure_required_fields_count
)

load_dotenv(override=True)


async def collect_specificity_data(
    job_description: str,
    company_name: str,
    title: str
) -> Dict[str, Any]:
    """
    구체성 평가 데이터 수집 (병렬 처리)

    Args:
        job_description: 채용 공고 내용
        company_name: 회사명
        title: 채용 공고 제목

    Returns:
        Dict: 각 tool의 원형 결과
            {
                "responsibility": "=== 담당 업무 구체성 평가 ===...",
                "qualification": "=== 자격요건 구체성 평가 ===...",
                "keyword_relevance": "=== 키워드 적합성 평가 ===...",
                "required_fields": "=== 필수 항목 포함 여부 평가 ===..."
            }
    """
    print(f"[Specificity Module] Starting data collection for: {title}")

    try:
        # 모든 tool을 병렬로 호출
        responsibility_result, qualification_result, keyword_result, required_fields_result = await asyncio.gather(
            asyncio.to_thread(
                measure_responsibility_specificity.invoke,
                {"job_description": job_description}
            ),
            asyncio.to_thread(
                measure_qualification_specificity.invoke,
                {"job_description": job_description}
            ),
            asyncio.to_thread(
                measure_keyword_relevance.invoke,
                {"job_description": job_description}
            ),
            asyncio.to_thread(
                measure_required_fields_count.invoke,
                {"job_description": job_description}
            ),
            return_exceptions=True
        )

        # 예외 처리
        results = {}
        
        if isinstance(responsibility_result, Exception):
            print(f"[Specificity Module] Responsibility tool failed: {responsibility_result}")
            results["responsibility"] = f"평가 실패: {str(responsibility_result)}"
        else:
            results["responsibility"] = responsibility_result
            print(f"[Specificity Module] Responsibility completed")

        if isinstance(qualification_result, Exception):
            print(f"[Specificity Module] Qualification tool failed: {qualification_result}")
            results["qualification"] = f"평가 실패: {str(qualification_result)}"
        else:
            results["qualification"] = qualification_result
            print(f"[Specificity Module] Qualification completed")

        if isinstance(keyword_result, Exception):
            print(f"[Specificity Module] Keyword relevance tool failed: {keyword_result}")
            results["keyword_relevance"] = f"평가 실패: {str(keyword_result)}"
        else:
            results["keyword_relevance"] = keyword_result
            print(f"[Specificity Module] Keyword relevance completed")

        if isinstance(required_fields_result, Exception):
            print(f"[Specificity Module] Required fields tool failed: {required_fields_result}")
            results["required_fields"] = f"평가 실패: {str(required_fields_result)}"
        else:
            results["required_fields"] = required_fields_result
            print(f"[Specificity Module] Required fields completed")

        print(f"[Specificity Module] Data collection completed for: {title}")
        return results

    except Exception as e:
        print(f"[Specificity Module] Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

