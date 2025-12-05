"""
모델 서비스 HTTP 클라이언트

메인 API에서 모델 서비스와 통신하기 위한 클라이언트입니다.
기존 SentenceTransformer를 대체하여 HTTP 통신으로 임베딩을 생성합니다.

사용 예시:
    # 기존 코드와 동일한 인터페이스
    client = ModelServiceClient("http://model-service:8001")
    embeddings = client.encode(["Python 개발자", "Backend Engineer"])
    
    # numpy array로 변환 (기존 코드 호환)
    embeddings = client.encode(texts, convert_to_numpy=True)
"""

import requests
from typing import List, Dict, Optional, Union
import os
import logging
import numpy as np
import socket
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def _get_default_model_service_url() -> str:
    """
    환경에 따라 기본 모델 서비스 URL을 자동으로 결정합니다.
    
    Returns:
        str: 모델 서비스 URL
        - Kubernetes/Docker 환경: http://model-service:8001
        - 로컬 개발 환경: http://localhost:8000
    
    Note:
        'model-service' 호스트명이 DNS로 해석되면 Kubernetes 환경으로 판단합니다.
    """
    try:
        # model-service 호스트명이 해석되면 Kubernetes/Docker 환경
        socket.gethostbyname('model-service')
        return "http://model-service:8001"
    except socket.gaierror:
        # 해석 안 되면 로컬 개발 환경
        return "http://localhost:8000"


class ModelServiceClient:
    """
    모델 서비스 HTTP 클라이언트
    
    Sentence-BERT 모델 서비스와 통신하여 임베딩 생성 및 유사도 계산을 수행합니다.
    SentenceTransformer와 동일한 인터페이스를 제공하여 기존 코드와 호환됩니다.
    
    Attributes:
        base_url: 모델 서비스의 기본 URL
        timeout: HTTP 요청 타임아웃 (초)
    
    Example:
        >>> client = ModelServiceClient()
        >>> embeddings = client.encode(["Python 개발자"])
        >>> print(embeddings.shape)  # numpy array
        (1, 512)
    """
    
    def __init__(
        self, 
        base_url: Optional[str] = None,
        timeout: int = 30
    ):
        """
        클라이언트 초기화
        
        Args:
            base_url: 모델 서비스 URL (환경 변수 MODEL_SERVICE_URL로 설정 가능)
                     기본값: 환경에 따라 자동 결정
                     - Kubernetes: http://model-service:8001
                     - 로컬: http://localhost:8000
            timeout: HTTP 요청 타임아웃 (초)
        
        Note:
            환경 변수 MODEL_SERVICE_URL을 설정하면 자동 감지를 무시하고 해당 URL을 사용합니다.
        """
        # 우선순위: 1) 직접 전달된 base_url, 2) 환경 변수, 3) 자동 감지
        self.base_url = base_url or os.getenv(
            "MODEL_SERVICE_URL",
            _get_default_model_service_url()  # 환경에 따라 자동 결정
        )
        self.timeout = timeout
        
        logger.info(f"ModelServiceClient 초기화: {self.base_url}")
        
        # 헬스체크로 연결 확인
        self._check_health()
    
    def _check_health(self):
        """
        모델 서비스 헬스체크
        
        서비스가 정상 작동하는지 확인합니다.
        실패 시 경고 로그를 남기지만 예외는 발생시키지 않습니다.
        (서비스가 나중에 시작될 수 있으므로)
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            health_data = response.json()
            logger.info(f"모델 서비스 연결 성공: {health_data}")
        except Exception as e:
            logger.warning(f"모델 서비스 헬스체크 실패 (서비스가 시작 중일 수 있습니다): {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs  # 호환성을 위해 추가 인자 무시
    ) -> Union[np.ndarray, List[List[float]]]:
        """
        텍스트를 임베딩 벡터로 변환
        
        SentenceTransformer.encode()와 호환되는 인터페이스를 제공합니다.
        
        Args:
            texts: 임베딩할 텍스트 (단일 문자열 또는 리스트)
            convert_to_numpy: True면 numpy array 반환, False면 list 반환
            normalize_embeddings: 임베딩 정규화 여부
            **kwargs: 기타 인자 (무시됨, 호환성 유지용)
        
        Returns:
            임베딩 (numpy array 또는 list)
            - convert_to_numpy=True: np.ndarray shape (n, dim)
            - convert_to_numpy=False: List[List[float]]
        
        Raises:
            requests.exceptions.RequestException: HTTP 요청 실패 시
        
        Example:
            >>> client = ModelServiceClient()
            >>> # 단일 텍스트
            >>> emb = client.encode("Python 개발자")
            >>> print(emb.shape)
            (1, 512)
            >>> 
            >>> # 여러 텍스트
            >>> embs = client.encode(["Python", "Java"])
            >>> print(embs.shape)
            (2, 512)
        
        Note:
            - 실패 시 최대 3회 재시도 (지수 백오프)
            - 타임아웃: 30초 (기본값)
            - 배치 크기: 100개 이하 권장
        """
        # 단일 문자열을 리스트로 변환
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            logger.debug(f"임베딩 요청: {len(texts)}개 텍스트")
            
            # 모델 서비스에 POST 요청
            response = requests.post(
                f"{self.base_url}/embed",
                json={
                    "texts": texts,
                    "normalize": normalize_embeddings
                },
                timeout=self.timeout
            )
            
            # HTTP 에러 체크 (4xx, 5xx)
            response.raise_for_status()
            
            # 응답 파싱
            data = response.json()
            embeddings = data["embeddings"]
            
            logger.debug(f"임베딩 수신 완료: {len(embeddings)}개")
            
            # numpy 변환 옵션 (기존 코드 호환성)
            if convert_to_numpy:
                return np.array(embeddings)
            else:
                return embeddings
            
        except requests.exceptions.Timeout:
            logger.error(f"모델 서비스 타임아웃 ({self.timeout}초)")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"모델 서비스 요청 실패: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def calculate_similarity(
        self,
        query_text: str,
        corpus_embeddings: Union[np.ndarray, List[List[float]]]
    ) -> List[float]:
        """
        쿼리 텍스트와 코퍼스 임베딩 간의 유사도 계산
        
        Args:
            query_text: 쿼리 텍스트
            corpus_embeddings: 비교할 코퍼스의 임베딩 (numpy array 또는 list)
        
        Returns:
            각 코퍼스 항목과의 유사도 점수 리스트 (0~1 범위)
        
        Raises:
            requests.exceptions.RequestException: HTTP 요청 실패 시
        
        Example:
            >>> client = ModelServiceClient()
            >>> corpus_embs = client.encode(["Python", "Java"])
            >>> similarities = client.calculate_similarity("Python 개발자", corpus_embs)
            >>> print(similarities)
            [0.85, 0.42]
        """
        # numpy array를 list로 변환
        if isinstance(corpus_embeddings, np.ndarray):
            corpus_embeddings = corpus_embeddings.tolist()
        
        try:
            logger.debug(f"유사도 계산 요청: 쿼리 1개 vs 코퍼스 {len(corpus_embeddings)}개")
            
            # 모델 서비스에 POST 요청
            response = requests.post(
                f"{self.base_url}/similarity",
                json={
                    "query_text": query_text,
                    "corpus_embeddings": corpus_embeddings
                },
                timeout=self.timeout
            )
            
            # HTTP 에러 체크
            response.raise_for_status()
            
            # 응답 파싱
            data = response.json()
            similarities = data["similarities"]
            
            logger.debug(f"유사도 수신 완료: {len(similarities)}개")
            
            return similarities
            
        except requests.exceptions.Timeout:
            logger.error(f"모델 서비스 타임아웃 ({self.timeout}초)")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"모델 서비스 요청 실패: {e}")
            raise
    
    def get_sentence_embedding_dimension(self) -> int:
        """
        임베딩 차원 수 반환
        
        SentenceTransformer와 호환되는 메서드입니다.
        
        Returns:
            임베딩 차원 수 (예: 512)
        
        Example:
            >>> client = ModelServiceClient()
            >>> dim = client.get_sentence_embedding_dimension()
            >>> print(dim)
            512
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embedding_dimension", 512)  # 기본값 512
        except Exception as e:
            logger.warning(f"임베딩 차원 조회 실패: {e}, 기본값 512 반환")
            return 512

