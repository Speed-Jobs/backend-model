"""
Job Matching System Configuration

데이터 파일 경로 및 모델 설정
"""
from pathlib import Path
from typing import List

# 프로젝트 루트 디렉토리 (backend-model)
# app/core/job_matching/config.py에서 backend-model까지의 경로
BASE_DIR = Path(__file__).parent.parent.parent.parent

# 데이터 디렉토리 (backend-model/data)
DATA_DIR = BASE_DIR / "data"

# ============================================================================
# Job Description 파일 경로
# ============================================================================
JOB_DESCRIPTION_FILE = DATA_DIR / "new_job_description.json"

# ============================================================================
# SBERT 모델 설정
# ============================================================================
SBERT_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"

# ============================================================================
# 학습 데이터 파일 경로
# ============================================================================
# TODO: 현재는 JSON 파일에서 로드하지만, 추후 DB에서 직접 가져오도록 수정 필요
# 데이터 파이프라인 구축 완료 후 DB 쿼리로 대체 예정
TRAINING_DATA_FILES: List[str] = [
    str(DATA_DIR / "hanwha_jobs.json"),
    str(DATA_DIR / "kakao_jobs.json"),
    str(DATA_DIR / "line_jobs.json"),
    str(DATA_DIR / "naver_jobs.json"),
]

# ============================================================================
# 매칭 파라미터 설정
# ============================================================================
PPR_TOP_N = 20  # PPR로 상위 N개 직무 추출
FINAL_TOP_K = 2  # 최종 반환할 매칭 결과 개수
