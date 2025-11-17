"""
Attractiveness Evaluator (Phase 1: Data Collection)
매력도 평가 데이터 수집 모듈
"""

from typing import Dict, Any
import asyncio
import os
from dotenv import load_dotenv
from app.utils.evaluation.llm_functions.attractiveness import (
    measure_special_content_count,
    measure_special_content_quality
)

load_dotenv(override=True)


async def collect_attractiveness_data(
    job_description: str,
    company_name: str,
    title: str
) -> Dict[str, Any]:
    """
    매력도 평가 데이터 수집 (병렬 처리)

    Args:
        job_description: 채용 공고 내용
        company_name: 회사명
        title: 채용 공고 제목

    Returns:
        Dict: 각 tool의 원형 결과
            {
                "content_count": "=== 특별 콘텐츠 포함 여부 평가 ===...",
                "content_quality": "=== 특별 콘텐츠 충실도 평가 ===..."
            }
    """
    print(f"[Attractiveness Evaluator] Starting data collection for: {title}")

    try:
        # 모든 tool을 병렬로 호출
        content_count_result, content_quality_result = await asyncio.gather(
            asyncio.to_thread(
                measure_special_content_count.invoke,
                {"job_description": job_description}
            ),
            asyncio.to_thread(
                measure_special_content_quality.invoke,
                {"job_description": job_description}
            ),
            return_exceptions=True
        )

        # 예외 처리
        results = {}
        
        if isinstance(content_count_result, Exception):
            print(f"[Attractiveness Evaluator] Content count tool failed: {content_count_result}")
            results["content_count"] = f"평가 실패: {str(content_count_result)}"
        else:
            results["content_count"] = content_count_result
            print(f"[Attractiveness Evaluator] Content count completed")

        if isinstance(content_quality_result, Exception):
            print(f"[Attractiveness Evaluator] Content quality tool failed: {content_quality_result}")
            results["content_quality"] = f"평가 실패: {str(content_quality_result)}"
        else:
            results["content_quality"] = content_quality_result
            print(f"[Attractiveness Evaluator] Content quality completed")

        print(f"[Attractiveness Evaluator] Data collection completed for: {title}")
        return results

    except Exception as e:
        print(f"[Attractiveness Evaluator] Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

