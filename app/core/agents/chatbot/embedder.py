import httpx
from typing import List


class Embedder:
    """텍스트 임베딩 생성 (API 호출 방식)"""
    
    def __init__(
        self, 
        model_name: str = None,
        api_url: str = "http://model-service.skala-practice.svc.cluster.local:8001",
        endpoint: str = "/embed_bge_m3"
    ):
        self.model_name = model_name or "BAAI/bge-m3"
        self.api_url = api_url.rstrip('/')
        self.endpoint = endpoint
        self.full_url = f"{self.api_url}{self.endpoint}"
        self.client = httpx.AsyncClient(timeout=300.0)
        
        print(f"Embedder initialized: {self.model_name}")
        print(f"API URL: {self.full_url}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
    
    async def embed(self, text: str) -> List[float]:
        if not text:
            return []
        
        try:
            response = await self.client.post(
                self.full_url,
                json={"texts": [text], "normalize": True}
            )
            response.raise_for_status()
            result = response.json()
            return result["embeddings"][0]
        except httpx.HTTPError as e:
            print(f"❌ API 호출 실패: {e}")
            raise
    
    async def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        normalize: bool = True
    ) -> List[List[float]]:
        if not texts:
            return []
        
        valid_texts = [t for t in texts if t]
        if not valid_texts:
            return []
        
        print(f"🔢 Embedding {len(valid_texts)} texts via API...")
        all_embeddings = []
        
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            
            try:
                response = await self.client.post(
                    self.full_url,
                    json={"texts": batch_texts, "normalize": normalize}
                )
                response.raise_for_status()
                result = response.json()
                all_embeddings.extend(result["embeddings"])
                
                if (i + batch_size) % (batch_size * 5) == 0:
                    print(f"  Progress: {min(i + batch_size, len(valid_texts))}/{len(valid_texts)}")
            except httpx.HTTPError as e:
                print(f"❌ API 호출 실패 (batch {i}-{i+batch_size}): {e}")
                raise
        
        print(f"✅ Embeddings generated via API")
        return all_embeddings
    
    def get_embedding_dimension(self) -> int:
        try:
            response = httpx.get(f"{self.api_url}/health")
            response.raise_for_status()
            health_data = response.json()
            
            if "bge_m3" in health_data.get("models", {}):
                return health_data["models"]["bge_m3"]["embedding_dimension"]
            if "sentence_transformers" in health_data.get("models", {}):
                return health_data["models"]["sentence_transformers"]["embedding_dimension"]
            return 1024
        except Exception as e:
            print(f"⚠️ 차원 정보 조회 실패, 기본값 사용: {e}")
            return 1024
    
    async def close(self):
        if hasattr(self, 'client') and not self.client.is_closed:
            await self.client.aclose()