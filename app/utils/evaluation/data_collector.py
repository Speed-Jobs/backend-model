"""
Data Collector (Phase 1)
채용 공고 평가 데이터 수집 오케스트레이터 - 원형 데이터만 수집
"""

from typing import List, Dict, Any
import asyncio
import os
from dotenv import load_dotenv

from app.utils.evaluation.evaluators import (
    collect_readability_data,
    collect_specificity_data,
    collect_attractiveness_data
)
from app.db.crud.post import get_post_by_id
from app.db.config.base import get_db
from app.utils.evaluation.report_saver import save_raw_evaluation_data

load_dotenv(override=True)


async def collect_evaluation_data_async(post_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    단일 채용 공고의 평가 데이터 수집 (비동기 병렬 실행)
    Phase 1: 원형 데이터만 수집하고 JSON으로 저장

    Args:
        post_data: 채용 공고 데이터 (post_id, title, company, content)

    Returns:
        Dict: 원형 평가 데이터
            {
                'post_id': int,
                'title': str,
                'company': str,
                'raw_results': {
                    'readability': {...},
                    'specificity': {...},
                    'attractiveness': {...}
                },
                'saved_file': str  # JSON 파일 경로
            }
    """
    content = post_data['content']
    company = post_data['company']
    title = post_data['title']

    print(f"\n[Data Collector] Starting parallel data collection: {title}")
    print(f"[Data Collector] Content length: {len(content)} characters")

    # ============ Phase 1: 데이터 수집 (병렬) ============
    try:
        print("[Data Collector] Launching 3 parallel evaluators...")
        readability_results, specificity_results, attractiveness_results = await asyncio.gather(
            collect_readability_data(content, company, title),
            collect_specificity_data(content, company, title),
            collect_attractiveness_data(content, company, title),
            return_exceptions=True
        )
        print("[Data Collector] All evaluators completed!")

        # 예외 처리
        if isinstance(readability_results, Exception):
            print(f"[Error] Readability evaluation failed: {readability_results}")
            readability_results = {"error": str(readability_results)}
        else:
            print(f"[Complete] Readability - {len(readability_results)} tools")

        if isinstance(specificity_results, Exception):
            print(f"[Error] Specificity evaluation failed: {specificity_results}")
            specificity_results = {"error": str(specificity_results)}
        else:
            print(f"[Complete] Specificity - {len(specificity_results)} tools")

        if isinstance(attractiveness_results, Exception):
            print(f"[Error] Attractiveness evaluation failed: {attractiveness_results}")
            attractiveness_results = {"error": str(attractiveness_results)}
        else:
            print(f"[Complete] Attractiveness - {len(attractiveness_results)} tools")

    except Exception as e:
        print(f"[Error] Unexpected error during parallel evaluation: {e}")
        readability_results = {"error": str(e)}
        specificity_results = {"error": str(e)}
        attractiveness_results = {"error": str(e)}

    print(f"[Data Collector] Phase 1 data collection completed")

    # ============ Phase 1: 원형 데이터 저장 ============
    raw_results_dict = {
        'readability': readability_results,
        'specificity': specificity_results,
        'attractiveness': attractiveness_results
    }
    
    raw_data_filepath = None
    try:
        raw_data_filepath = save_raw_evaluation_data(
            post_id=post_data['post_id'],
            title=title,
            company=company,
            company_id=post_data.get('company_id'),
            raw_results=raw_results_dict
        )
        print(f"[Data Collector] Raw data saved: {raw_data_filepath}")
    except Exception as e:
        print(f"[Data Collector] Failed to save raw data: {e}")

    return {
        'post_id': post_data['post_id'],
        'title': title,
        'company': company,
        # 원형 tool 결과
        'raw_results': raw_results_dict,
        # 저장된 파일 경로
        'saved_file': raw_data_filepath
    }


def collect_evaluation_data(post_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    단일 채용 공고의 평가 데이터 수집 (동기 래퍼)

    Args:
        post_data: 채용 공고 데이터 (post_id, title, company, content)

    Returns:
        Dict: 평가 결과 (원형 데이터)
    """
    return asyncio.run(collect_evaluation_data_async(post_data))


async def collect_multiple_posts_async(
    post_ids: List[int]
) -> Dict[str, Any]:
    """
    여러 채용 공고의 데이터를 병렬로 수집 (비동기)
    Phase 1: 원형 데이터만 수집

    Args:
        post_ids: 평가할 채용 공고 ID 리스트 (최대 2개)

    Returns:
        Dict: 각 공고의 원형 평가 데이터
    """
    if len(post_ids) != 2:
        raise ValueError("정확히 2개의 post_id가 필요합니다")

    # DB Session 가져오기
    db = next(get_db())

    try:
        # 채용 공고 데이터 가져오기
        posts_data = []
        for post_id in post_ids:
            post = get_post_by_id(db, post_id)
            if not post:
                raise ValueError(f"Post ID {post_id}를 찾을 수 없습니다")

            posts_data.append({
                "post_id": post.id,
                "title": post.title,
                "company": post.company.name if post.company else "알 수 없음",
                "company_id": post.company_id if post.company_id else None,
                "content": post.description or ""
            })

        print(f"\n{'='*60}")
        print(f"두 개의 채용 공고 데이터 수집 시작")
        print(f"Post 1: {posts_data[0]['title']}")
        print(f"Post 2: {posts_data[1]['title']}")
        print(f"{'='*60}\n")

        # 두 개의 채용 공고를 병렬로 데이터 수집
        result_1, result_2 = await asyncio.gather(
            collect_evaluation_data_async(posts_data[0]),
            collect_evaluation_data_async(posts_data[1]),
            return_exceptions=True
        )

        # 예외 처리
        results = {}

        if isinstance(result_1, Exception):
            print(f"[Error] Post 1 evaluation failed: {result_1}")
            results["post_1"] = {"error": str(result_1)}
        else:
            results["post_1"] = result_1

        if isinstance(result_2, Exception):
            print(f"[Error] Post 2 evaluation failed: {result_2}")
            results["post_2"] = {"error": str(result_2)}
        else:
            results["post_2"] = result_2

        print(f"\n{'='*60}")
        print("모든 채용 공고 데이터 수집 완료")
        print(f"{'='*60}\n")

        return results

    finally:
        db.close()


def collect_multiple_posts(
    post_ids: List[int]
) -> Dict[str, Any]:
    """
    여러 채용 공고의 데이터를 병렬로 수집 (동기 래퍼)

    Args:
        post_ids: 평가할 채용 공고 ID 리스트 (최대 2개)

    Returns:
        Dict: 각 공고의 원형 평가 데이터
    """
    return asyncio.run(collect_multiple_posts_async(post_ids))

