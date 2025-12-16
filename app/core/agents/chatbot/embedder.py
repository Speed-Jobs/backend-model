import httpx
from typing import List
from app.core.config import settings


class Embedder:
    """텍스트 임베딩 생성 (API 호출 방식)"""
    
    def __init__(
        self,
        model_name: str = None,
        api_url: str = None,
        endpoint: str = "/embed_bge_m3"
    ):
        self.model_name = model_name or "BAAI/bge-m3"
        
        # API URL 결정 (클러스터 URL이면 localhost도 시도)
        cluster_url = api_url or settings.EMBEDDING_API_URL
        localhost_url = "http://localhost:8001"
        
        self.api_url = cluster_url.rstrip('/')
        self.endpoint = endpoint
        self.full_url = f"{self.api_url}{self.endpoint}"
        self.fallback_url = f"{localhost_url}{self.endpoint}"
        self.use_fallback = "cluster.local" in cluster_url
        
        self.client = httpx.AsyncClient(timeout=300.0)
        
        print(f"Embedder initialized: {self.model_name}")
        print(f"API URL: {self.full_url}")
        if self.use_fallback:
            print(f"Fallback URL: {self.fallback_url} (kubectl port-forward 사용 시)")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
    
    async def embed(self, text: str) -> List[float]:
        if not text:
            return []
        
        # 먼저 primary URL 시도
        urls_to_try = [self.full_url]
        if self.use_fallback:
            urls_to_try.append(self.fallback_url)
        
        last_error = None
        for url in urls_to_try:
            try:
                response = await self.client.post(
                    url,
                    json={"texts": [text], "normalize": True}
                )
                response.raise_for_status()
                result = response.json()
                if url != self.full_url:
                    print(f"✅ Fallback URL로 연결 성공: {url}")
                    # 성공한 fallback URL을 primary로 업데이트
                    self.full_url = url
                    self.use_fallback = False
                return result["embeddings"][0]
            except httpx.HTTPError as e:
                last_error = e
                if url == self.full_url and self.use_fallback:
                    print(f"⚠️ Primary URL 실패, fallback 시도 중...")
                continue
        
        print(f"❌ API 호출 실패: {last_error}")
        print(f"시도한 URL: {urls_to_try}")
        print(f"로컬 개발 시: kubectl port-forward -n skala-practice svc/speedjobs-model-service 8001:8001")
        raise last_error
    
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
        
        # URL fallback 리스트
        urls_to_try = [self.full_url]
        if self.use_fallback:
            urls_to_try.append(self.fallback_url)
        
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            
            last_error = None
            for url in urls_to_try:
                try:
                    response = await self.client.post(
                        url,
                        json={"texts": batch_texts, "normalize": normalize}
                    )
                    response.raise_for_status()
                    result = response.json()
                    all_embeddings.extend(result["embeddings"])
                    
                    # 성공한 경우 fallback을 primary로 업데이트
                    if url != self.full_url:
                        print(f"✅ Fallback URL로 연결 성공: {url}")
                        self.full_url = url
                        self.use_fallback = False
                    
                    if (i + batch_size) % (batch_size * 5) == 0:
                        print(f"  Progress: {min(i + batch_size, len(valid_texts))}/{len(valid_texts)}")
                    break
                except httpx.HTTPError as e:
                    last_error = e
                    if url == urls_to_try[0] and len(urls_to_try) > 1:
                        print(f"⚠️ Primary URL 실패, fallback 시도 중...")
                    continue
            
            if last_error:
                print(f"❌ API 호출 실패 (batch {i}-{i+batch_size}): {last_error}")
                print(f"시도한 URL: {urls_to_try}")
                print(f"로컬 개발 시: kubectl port-forward -n skala-practice svc/speedjobs-model-service 8001:8001")
                raise last_error
        
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