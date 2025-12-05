"""
모델 서비스 테스트

실행 방법:
    pytest tests/model_service/test_model_service.py -v
    
환경 변수:
    MODEL_SERVICE_URL: 모델 서비스 URL (기본: http://localhost:8001)
"""

import pytest
import requests
import os
from typing import List

# 테스트 대상 URL (환경에 따라 변경)
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001")

class TestModelService:
    """모델 서비스 API 테스트"""
    
    def test_health_check(self):
        """헬스체크 엔드포인트 테스트"""
        response = requests.get(f"{MODEL_SERVICE_URL}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_name" in data
        assert "embedding_dimension" in data
        print(f"✓ 헬스체크 성공: {data}")
    
    def test_embed_single_text(self):
        """단일 텍스트 임베딩 생성 테스트"""
        response = requests.post(
            f"{MODEL_SERVICE_URL}/embed",
            json={
                "texts": ["Python 개발자"],
                "normalize": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) == data["dimension"]
        print(f"✓ 단일 텍스트 임베딩 성공: 차원={data['dimension']}")
    
    def test_embed_multiple_texts(self):
        """다중 텍스트 임베딩 생성 테스트"""
        texts = ["Python 개발자", "Backend Engineer", "프론트엔드 개발자"]
        
        response = requests.post(
            f"{MODEL_SERVICE_URL}/embed",
            json={
                "texts": texts,
                "normalize": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == len(texts)
        assert len(data["embeddings"]) == len(texts)
        print(f"✓ 다중 텍스트 임베딩 성공: {len(texts)}개")
    
    def test_similarity_calculation(self):
        """유사도 계산 테스트"""
        # 1. 먼저 코퍼스 임베딩 생성
        corpus_texts = ["Python 백엔드 개발자", "프론트엔드 개발자", "데이터 사이언티스트"]
        embed_response = requests.post(
            f"{MODEL_SERVICE_URL}/embed",
            json={"texts": corpus_texts, "normalize": True}
        )
        corpus_embeddings = embed_response.json()["embeddings"]
        
        # 2. 유사도 계산
        query_text = "Python 개발자"
        sim_response = requests.post(
            f"{MODEL_SERVICE_URL}/similarity",
            json={
                "query_text": query_text,
                "corpus_embeddings": corpus_embeddings
            }
        )
        
        assert sim_response.status_code == 200
        data = sim_response.json()
        assert data["count"] == len(corpus_texts)
        assert len(data["similarities"]) == len(corpus_texts)
        
        # 유사도 검증 (0~1 범위)
        for sim in data["similarities"]:
            assert 0 <= sim <= 1
        
        print(f"✓ 유사도 계산 성공:")
        for i, (text, sim) in enumerate(zip(corpus_texts, data["similarities"])):
            print(f"  {i+1}. {text}: {sim:.4f}")
    
    def test_error_handling_empty_texts(self):
        """빈 텍스트 리스트 에러 처리 테스트"""
        response = requests.post(
            f"{MODEL_SERVICE_URL}/embed",
            json={"texts": [], "normalize": True}
        )
        
        # 422 Unprocessable Entity (Pydantic 유효성 검사 실패)
        assert response.status_code == 422
        print("✓ 빈 텍스트 에러 처리 성공")

class TestModelServiceClient:
    """ModelServiceClient 클라이언트 테스트"""
    
    def test_client_encode(self):
        """ModelServiceClient encode 메서드 테스트"""
        from app.utils.model import ModelServiceClient
        
        client = ModelServiceClient(base_url=MODEL_SERVICE_URL)
        
        # 임베딩 생성
        embeddings = client.encode(["Python 개발자", "Backend Engineer"])
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0
        print(f"✓ ModelServiceClient 임베딩 성공: shape={embeddings.shape}")
    
    def test_client_similarity(self):
        """ModelServiceClient 유사도 계산 테스트"""
        from app.utils.model import ModelServiceClient
        
        client = ModelServiceClient(base_url=MODEL_SERVICE_URL)
        
        # 임베딩 생성
        corpus_embeddings = client.encode(["Python 개발자", "Java 개발자"])
        
        # 유사도 계산
        similarities = client.calculate_similarity(
            "Python 백엔드 개발자",
            corpus_embeddings
        )
        
        assert len(similarities) == 2
        assert all(0 <= sim <= 1 for sim in similarities)
        print(f"✓ ModelServiceClient 유사도 계산 성공: {similarities}")
    
    def test_client_get_dimension(self):
        """ModelServiceClient 차원 조회 테스트"""
        from app.utils.model import ModelServiceClient
        
        client = ModelServiceClient(base_url=MODEL_SERVICE_URL)
        dim = client.get_sentence_embedding_dimension()
        
        assert dim > 0
        print(f"✓ 임베딩 차원 조회 성공: {dim}")

if __name__ == "__main__":
    # 개별 테스트 실행
    pytest.main([__file__, "-v", "-s"])

