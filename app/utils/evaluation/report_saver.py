"""
Report Saver Utility
평가 원형 데이터를 JSON 파일로 저장하는 유틸리티
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def save_raw_evaluation_data(
    post_id: int,
    title: str,
    company: str,
    raw_results: Dict[str, Any],
    company_id: int = None,
    output_dir: str = "data/report"
) -> str:
    """
    평가 원형 데이터를 JSON 파일로 저장

    Args:
        post_id: 채용 공고 ID
        title: 채용 공고 제목
        company: 회사명
        raw_results: 원형 평가 데이터
            {
                'readability': {...},
                'specificity': {...},
                'attractiveness': {...}
            }
        company_id: 회사 ID
        output_dir: 저장 디렉토리 경로

    Returns:
        str: 저장된 파일 경로
    """
    # 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 파일명 생성
    filename = f"post_{post_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # 저장할 데이터 구조
    data = {
        "metadata": {
            "post_id": post_id,
            "company_id": company_id,
            "title": title,
            "company": company,
            "evaluated_at": datetime.now().isoformat(),
            "timestamp": timestamp
        },
        "raw_evaluation_results": raw_results
    }

    # JSON 파일로 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[Report Saver] Raw evaluation data saved to: {filepath}")
    return filepath


def save_final_report(
    post_id: int,
    title: str,
    company: str,
    raw_results: Dict[str, Any],
    summary: str,
    company_id: int = None,
    output_dir: str = "data/report"
) -> str:
    """
    최종 보고서 (원형 데이터 + 종합 평가) 저장

    Args:
        post_id: 채용 공고 ID
        title: 채용 공고 제목
        company: 회사명
        raw_results: 원형 평가 데이터
        summary: LLM이 생성한 종합 평가 보고서
        company_id: 회사 ID
        output_dir: 저장 디렉토리 경로

    Returns:
        str: 저장된 파일 경로
    """
    # 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 파일명 생성
    filename = f"report_post_{post_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # 저장할 데이터 구조
    data = {
        "metadata": {
            "post_id": post_id,
            "company_id": company_id,
            "title": title,
            "company": company,
            "evaluated_at": datetime.now().isoformat(),
            "timestamp": timestamp
        },
        "raw_evaluation_results": raw_results,
        "summary_report": summary
    }

    # JSON 파일로 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[Report Saver] Final report saved to: {filepath}")
    return filepath


def load_evaluation_data(filepath: str) -> Dict[str, Any]:
    """
    저장된 평가 데이터 불러오기

    Args:
        filepath: JSON 파일 경로

    Returns:
        Dict: 평가 데이터
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

