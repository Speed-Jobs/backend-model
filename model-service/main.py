"""
Sentence-BERT 모델 서빙 서비스

이 서비스는 텍스트 임베딩 생성과 유사도 계산을 담당합니다.
메인 API 서버와 분리되어 독립적으로 운영됩니다.

주요 기능:
1. /embed: 텍스트를 벡터로 변환
2. /similarity: 쿼리와 코퍼스 간 유사도 계산
3. /health: 헬스체크

사용 예시:
    # 로컬 실행
    python main.py
    
    # Docker 실행
    docker build -t model-service .
    docker run -p 8000:8000 model-service
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import os

# ============================================================================
# 로깅 설정
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI 앱 초기화
# ============================================================================
app = FastAPI(
    title="Sentence-BERT Model Service",
    description="텍스트 임베딩 및 유사도 계산 서비스",
    version="1.0.0"
)

# ============================================================================
# 전역 변수: 모델 인스턴스 (앱 시작 시 한 번만 로드)
# ============================================================================
model: Optional[SentenceTransformer] = None
MODEL_NAME = os.getenv(
    "MODEL_NAME", 
    "sentence-transformers/distiluse-base-multilingual-cased-v2"
)

# ============================================================================
# 요청/응답 스키마 정의
# ============================================================================

class EmbedRequest(BaseModel):
    """
    임베딩 생성 요청
    
    Attributes:
        texts: 임베딩을 생성할 텍스트 리스트
        normalize: 임베딩을 정규화할지 여부 (코사인 유사도 계산 시 True 권장)
    
    Example:
        {
            "texts": ["Python 개발자", "Backend Engineer"],
            "normalize": true
        }
    """
    texts: List[str] = Field(..., description="임베딩할 텍스트 리스트", min_items=1)
    normalize: bool = Field(True, description="임베딩 정규화 여부")

class EmbedResponse(BaseModel):
    """
    임베딩 생성 응답
    
    Attributes:
        embeddings: 생성된 임베딩 리스트 (각 텍스트마다 하나의 벡터)
        dimension: 임베딩 차원 수
        count: 처리된 텍스트 개수
    """
    embeddings: List[List[float]] = Field(..., description="생성된 임베딩")
    dimension: int = Field(..., description="임베딩 차원")
    count: int = Field(..., description="처리된 텍스트 수")

class SimilarityRequest(BaseModel):
    """
    유사도 계산 요청
    
    Attributes:
        query_text: 쿼리 텍스트 (비교 대상)
        corpus_embeddings: 비교할 코퍼스의 임베딩 리스트
    
    Example:
        {
            "query_text": "Python 백엔드 개발자",
            "corpus_embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        }
    """
    query_text: str = Field(..., description="쿼리 텍스트")
    corpus_embeddings: List[List[float]] = Field(..., description="코퍼스 임베딩")

class SimilarityResponse(BaseModel):
    """
    유사도 계산 응답
    
    Attributes:
        similarities: 각 코퍼스 항목과의 유사도 점수 (0~1 범위)
        count: 계산된 유사도 개수
    """
    similarities: List[float] = Field(..., description="유사도 점수 리스트")
    count: int = Field(..., description="계산된 유사도 수")

# ============================================================================
# 앱 시작 시 모델 로드
# ============================================================================

@app.on_event("startup")
async def load_model():
    """
    앱 시작 시 Sentence-BERT 모델을 메모리에 로드합니다.
    
    이 작업은 시간이 걸릴 수 있으므로 (30초~1분) 
    Kubernetes livenessProbe의 initialDelaySeconds를 충분히 설정해야 합니다.
    
    Note:
        - 모델은 전역 변수에 저장되어 모든 요청에서 재사용됩니다.
        - 로딩 실패 시 앱이 시작되지 않습니다.
    """
    global model
    try:
        logger.info(f"모델 로딩 시작: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        logger.info(f"모델 로딩 완료: {MODEL_NAME}")
        logger.info(f"임베딩 차원: {model.get_sentence_embedding_dimension()}")
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        raise

# ============================================================================
# API 엔드포인트
# ============================================================================

@app.get("/health")
async def health_check():
    """
    헬스체크 엔드포인트
    
    Kubernetes liveness/readiness probe에서 사용됩니다.
    모델이 정상적으로 로드되었는지 확인합니다.
    
    Returns:
        status: "healthy" (모델 로드 완료) 또는 "loading" (로딩 중)
        model_name: 로드된 모델 이름
        embedding_dimension: 임베딩 차원 (모델 로드 완료 시)
    
    Example:
        GET /health
        
        Response:
        {
            "status": "healthy",
            "model_name": "sentence-transformers/distiluse-base-multilingual-cased-v2",
            "embedding_dimension": 512
        }
    """
    if model is None:
        return {
            "status": "loading",
            "model_name": MODEL_NAME
        }
    
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "embedding_dimension": model.get_sentence_embedding_dimension()
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """
    텍스트를 임베딩 벡터로 변환합니다.
    
    이 엔드포인트는 텍스트 리스트를 받아 각 텍스트를 고차원 벡터로 변환합니다.
    생성된 임베딩은 의미적 유사도 계산에 사용됩니다.
    
    Args:
        request: 임베딩 요청 (텍스트 리스트 포함)
    
    Returns:
        EmbedResponse: 생성된 임베딩과 메타데이터
    
    Raises:
        HTTPException 503: 모델이 로드되지 않았을 때
        HTTPException 500: 임베딩 생성 중 오류 발생 시
    
    Example:
        POST /embed
        {
            "texts": ["Python 개발자", "Backend Engineer"],
            "normalize": true
        }
        
        Response:
        {
            "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
            "dimension": 512,
            "count": 2
        }
    
    Note:
        - normalize=True 권장 (코사인 유사도 계산 시 필수)
        - 한 번에 너무 많은 텍스트를 보내면 메모리 부족 가능
        - 권장 배치 크기: 100개 이하
    """
    # 모델 로드 확인
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 아직 로드되지 않았습니다. 잠시 후 다시 시도해주세요."
        )
    
    try:
        logger.debug(f"임베딩 생성 요청: {len(request.texts)}개 텍스트")
        
        # 임베딩 생성
        embeddings = model.encode(
            request.texts,
            convert_to_numpy=True,
            normalize_embeddings=request.normalize,
            show_progress_bar=False  # API 서버에서는 진행바 불필요
        )
        
        # numpy array를 list로 변환 (JSON 직렬화 가능하도록)
        embeddings_list = embeddings.tolist()
        
        logger.debug(f"임베딩 생성 완료: {len(embeddings_list)}개")
        
        return EmbedResponse(
            embeddings=embeddings_list,
            dimension=len(embeddings_list[0]) if embeddings_list else 0,
            count=len(embeddings_list)
        )
        
    except Exception as e:
        logger.error(f"임베딩 생성 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"임베딩 생성 중 오류 발생: {str(e)}"
        )

@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """
    쿼리 텍스트와 코퍼스 임베딩 간의 유사도를 계산합니다.
    
    이 엔드포인트는 쿼리 텍스트를 임베딩으로 변환한 후,
    미리 계산된 코퍼스 임베딩과의 코사인 유사도를 계산합니다.
    
    Args:
        request: 유사도 계산 요청 (쿼리 텍스트 + 코퍼스 임베딩)
    
    Returns:
        SimilarityResponse: 각 코퍼스 항목과의 유사도 점수
    
    Raises:
        HTTPException 503: 모델이 로드되지 않았을 때
        HTTPException 500: 유사도 계산 중 오류 발생 시
    
    Example:
        POST /similarity
        {
            "query_text": "Python 백엔드 개발자",
            "corpus_embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        }
        
        Response:
        {
            "similarities": [0.85, 0.42],
            "count": 2
        }
    
    Note:
        - 유사도는 코사인 유사도를 사용합니다 (0~1 범위)
        - 임베딩이 정규화되어 있으면 내적(dot product)으로 계산 가능
        - 코퍼스 임베딩은 /embed 엔드포인트로 미리 생성해야 합니다
    """
    # 모델 로드 확인
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="모델이 아직 로드되지 않았습니다. 잠시 후 다시 시도해주세요."
        )
    
    try:
        logger.debug(f"유사도 계산 요청: 쿼리 1개 vs 코퍼스 {len(request.corpus_embeddings)}개")
        
        # 1. 쿼리 텍스트를 임베딩으로 변환
        query_embedding = model.encode(
            [request.query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,  # 코사인 유사도를 위해 정규화
            show_progress_bar=False
        )
        
        # 2. 코퍼스 임베딩을 numpy array로 변환
        corpus_embeddings = np.array(request.corpus_embeddings)
        
        # 3. 코사인 유사도 계산 (정규화된 벡터의 내적)
        # query_embedding: (1, dim)
        # corpus_embeddings: (n, dim)
        # 결과: (n,) - 각 코퍼스 항목과의 유사도
        similarities = np.dot(corpus_embeddings, query_embedding.T).flatten()
        
        # 4. numpy array를 list로 변환
        similarities_list = similarities.tolist()
        
        logger.debug(f"유사도 계산 완료: {len(similarities_list)}개")
        
        return SimilarityResponse(
            similarities=similarities_list,
            count=len(similarities_list)
        )
        
    except Exception as e:
        logger.error(f"유사도 계산 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"유사도 계산 중 오류 발생: {str(e)}"
        )

@app.get("/")
async def root():
    """
    루트 엔드포인트 - 서비스 정보 제공
    
    Returns:
        서비스 메타데이터 및 사용 가능한 엔드포인트 목록
    """
    return {
        "service": "Sentence-BERT Model Service",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "health": "GET /health - 헬스체크",
            "embed": "POST /embed - 텍스트 임베딩 생성",
            "similarity": "POST /similarity - 유사도 계산",
            "docs": "GET /docs - API 문서 (Swagger UI)"
        },
        "status": "running" if model is not None else "loading"
    }

# ============================================================================
# 실행 (개발 환경)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # 개발 환경에서 직접 실행 시
    # python main.py
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

