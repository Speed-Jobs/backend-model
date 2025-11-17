"""
Job Matching System Model Loader

JobMatchingSystem 싱글톤 초기화 및 FastAPI dependency 제공
"""
from typing import Optional
import logging

from app.core.job_matching.job_matching_system import JobMatchingSystem

logger = logging.getLogger(__name__)

# 싱글톤 인스턴스 저장
_job_matching_system: Optional[JobMatchingSystem] = None
_initialized: bool = False


def get_job_matching_system() -> JobMatchingSystem:
    """
    JobMatchingSystem 싱글톤 인스턴스를 반환
    
    처음 호출 시에만 초기화하며, 이후 호출은 같은 인스턴스를 반환합니다.
    
    Returns:
        JobMatchingSystem: 초기화된 JobMatchingSystem 인스턴스
        
    Raises:
        RuntimeError: 초기화 실패 시
    """
    global _job_matching_system, _initialized
    
    # 이미 초기화된 경우 같은 인스턴스 반환
    if _job_matching_system is not None and _initialized:
        return _job_matching_system
    
    # 초기화 중 에러 발생 시 재시도 방지
    if _initialized and _job_matching_system is None:
        raise RuntimeError("JobMatchingSystem 초기화에 실패했습니다. 서버를 재시작해주세요.")
    
    try:
        logger.info("[JobMatching] 시스템 초기화 시작...")
        
        # JobMatchingSystem 인스턴스 생성
        _job_matching_system = JobMatchingSystem(log_file=None)
        
        # 1. 직무 정의 로드
        logger.info("[JobMatching] 직무 정의 로드 중...")
        _job_matching_system.load_job_descriptions()  # config에서 자동으로 경로 가져옴
        
        # 2. 학습 데이터 로드
        logger.info("[JobMatching] 학습 데이터 로드 중...")
        _job_matching_system.load_training_data()  # config에서 자동으로 경로 가져옴
        
        # 3. 그래프 구축
        logger.info("[JobMatching] 그래프 구축 중...")
        _job_matching_system.build_graph()
        
        # 4. Matchers 초기화 (SBERT, Cluster)
        logger.info("[JobMatching] Matchers 초기화 중...")
        _job_matching_system.build_matchers()
        
        _initialized = True
        logger.info("[JobMatching] 시스템 초기화 완료!")
        
        return _job_matching_system
        
    except Exception as e:
        logger.error(f"[JobMatching] 초기화 실패: {e}", exc_info=True)
        _job_matching_system = None
        _initialized = True  # 재시도 방지
        raise RuntimeError(f"JobMatchingSystem 초기화 실패: {str(e)}")


def reset_job_matching_system():
    """
    싱글톤 인스턴스를 초기화 (테스트 용도)
    
    주의: 프로덕션 환경에서는 사용하지 마세요.
    """
    global _job_matching_system, _initialized
    _job_matching_system = None
    _initialized = False
    logger.warning("[JobMatching] 시스템 리셋됨")

