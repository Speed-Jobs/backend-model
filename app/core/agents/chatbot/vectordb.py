from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, FilterSelector
from typing import Optional, List, Dict, Any
from app.core.config import settings
import uuid
from urllib.parse import urlparse


class QdrantClientWrapper:
    """Qdrant 싱글톤 클라이언트"""
    
    _instance: Optional['QdrantClientWrapper'] = None
    _client: Optional[QdrantClient] = None
    _collection_name: str = settings.QDRANT_COLLECTION_NAME
    _dimension: int = settings.EMBEDDING_DIM  # 모델 기본 차원
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._initialize_client()
    
    def _initialize_client(self):
        """Qdrant 클라이언트 초기화"""
        def build_client(url: str) -> QdrantClient:
            return QdrantClient(
                url=url,
                api_key=settings.QDRANT_API_KEY if hasattr(settings, "QDRANT_API_KEY") and settings.QDRANT_API_KEY else None,
            )

        def normalize_host(url: str) -> str:
            parsed = urlparse(url)
            return parsed.hostname or ""

        primary_url = settings.QDRANT_URL
        localhost_url = "http://localhost:6333"
        
        # 클러스터 URL인 경우 localhost를 fallback으로 시도 (kubectl port-forward 사용 시)
        try_urls = [primary_url]
        if "cluster.local" in primary_url or normalize_host(primary_url) == "qdrant":
            try_urls.append(localhost_url)
            print(f"클러스터 URL 감지. kubectl port-forward 사용 시를 위해 localhost도 시도합니다.")

        last_error = None
        for url in try_urls:
            try:
                print(f"Connecting to Qdrant: {url}")
                self._client = build_client(url)

                collections = self._client.get_collections().collections
                collection_names = [col.name for col in collections]

                if self._collection_name not in collection_names:
                    self._create_collection()
                    print(f"Qdrant collection created: {self._collection_name}")
                else:
                    collection_info = self._client.get_collection(self._collection_name)
                    existing_dim = collection_info.config.params.vectors.size
                    if existing_dim != settings.EMBEDDING_DIM:
                        print(
                            f"Collection dimension {existing_dim} != expected {settings.EMBEDDING_DIM}, recreating..."
                        )
                        self.set_dimension(settings.EMBEDDING_DIM)
                    else:
                        self._dimension = existing_dim
                        print(
                            f"Qdrant collection loaded: {self._collection_name} (dimension: {self._dimension})"
                        )
                # 성공 시 반환
                return
            except Exception as e:
                last_error = e
                print(f"Failed to connect to Qdrant at {url}: {e}")
                self._client = None
                continue

        error_msg = (
            f"Qdrant 연결 실패. 시도한 URL: {try_urls}. "
            "Qdrant 서버가 실행 중인지 확인하세요.\n"
            "로컬 개발 시: kubectl port-forward -n skala-practice svc/speedjobs-vectordb 6333:6333"
        )
        raise RuntimeError(f"{error_msg} | last_error={last_error}")
    
    def _create_collection(self):
        """새 collection 생성"""
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._dimension,
                distance=Distance.COSINE
            )
        )
    
    def set_dimension(self, dimension: int):
        """임베딩 차원 설정 (처음 한 번만)"""
        if self._dimension != dimension:
            self._dimension = dimension
            # 기존 collection이 있으면 삭제 후 재생성
            try:
                collections = self._client.get_collections().collections
                collection_names = [col.name for col in collections]
                if self._collection_name in collection_names:
                    self._client.delete_collection(self._collection_name)
                self._create_collection()
                print(f"Collection dimension set to: {dimension}")
            except Exception as e:
                print(f"Failed to recreate collection: {e}")
    
    def add(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        post_ids: List[int]
    ) -> List[str]:
        """
        벡터 추가
        
        Args:
            embeddings: 임베딩 벡터 리스트
            texts: 원본 텍스트 리스트
            post_ids: Post ID 리스트
        
        Returns:
            추가된 포인트 ID 리스트 (UUID 문자열)
        """
        if not embeddings or not texts or not post_ids:
            return []
        
        embedding_dim = len(embeddings[0])
        target_dim = settings.EMBEDDING_DIM

        # 모델이 예상 차원을 반환하는지 강제 검증 (1024 고정 요구)
        if embedding_dim != target_dim:
            raise ValueError(
                f"Embedding dimension mismatch. model={embedding_dim}, expected={target_dim}. "
                "임베딩 모델과 EMBEDDING_DIM 설정을 확인하세요."
            )

        # 컬렉션 차원을 강제로 1024(설정값)로 유지
        self._ensure_dimension(target_dim)
        
        # 포인트 생성
        points = []
        point_ids = []
        
        for i, (embedding, text, post_id) in enumerate(zip(embeddings, texts, post_ids)):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'text': text,
                    'post_id': post_id,
                    'index_id': i
                }
            )
            points.append(point)
        
        # Qdrant에 추가
        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points
            )
        except UnexpectedResponse as exc:
            # Qdrant가 기존 차원을 유지하고 있을 때 재시도
            msg = str(exc)
            if "Vector dimension error" in msg:
                print("Detected vector dimension mismatch. Recreating collection...")
                self.set_dimension(target_dim)
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points
                )
            else:
                raise
        
        count = self.count()
        print(f"Added {len(embeddings)} vectors (total: {count})")
        return point_ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        벡터 검색
        
        Args:
            query_embedding: 쿼리 벡터
            top_k: 반환할 결과 수
            filters: 메타데이터 필터 (예: {'post_id': 123})
        
        Returns:
            검색 결과 리스트
        """
        # 클라이언트 확인
        if self._client is None:
            raise RuntimeError("Qdrant client is not initialized")
        
        if self.count() == 0:
            return []
        
        # 필터 생성
        qdrant_filter = None
        if filters:
            conditions = []
            if 'post_id' in filters:
                conditions.append(
                    FieldCondition(
                        key="post_id",
                        match=MatchValue(value=filters['post_id'])
                    )
                )
            if 'company_id' in filters:
                conditions.append(
                    FieldCondition(
                        key="company_id",
                        match=MatchValue(value=filters['company_id'])
                    )
                )
            if conditions:
                qdrant_filter = Filter(must=conditions)
                print(f"[Qdrant] Applying filters: {filters}")
        
        # 검색 - qdrant-client API 사용
        try:
            search_results = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter
            )
        except AttributeError as e:
            # qdrant-client API 버전 차이 대응
            print(f"Search API error: {e}, trying alternative method...")
            from qdrant_client.models import SearchRequest
            search_results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter
            ).points
        
        # 결과 포맷팅
        results = []
        for result in search_results:
            payload = result.payload
            # Qdrant COSINE distance: 0~2 범위 (0이 가장 유사)
            distance = float(result.score)
            # similarity score로 변환 (0~1 범위, 1이 가장 유사)
            similarity_score = 1.0 - (distance / 2.0)
            similarity_score = max(0.0, min(1.0, similarity_score))  # 0~1 범위로 제한
            
            results.append({
                'id': str(result.id),
                'post_id': payload.get('post_id'),
                'text': payload.get('text'),
                'distance': distance,
                'score': similarity_score
            })
        
        return results
    
    def save(self):
        """Qdrant는 자동으로 저장되므로 별도 작업 없음"""
        pass
    
    def count(self) -> int:
        """저장된 벡터 개수"""
        try:
            collection_info = self._client.get_collection(self._collection_name)
            return collection_info.points_count
        except Exception as e:
            print(f"Failed to get count: {e}")
            return 0
    
    def delete_by_post_id(self, post_id: int):
        """특정 post_id의 모든 문서 삭제"""
        try:
            # post_id로 필터링하여 삭제
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="post_id",
                                match=MatchValue(value=post_id)
                            )
                        ]
                    )
                )
            )
            print(f"Deleted vectors with post_id: {post_id}")
        except Exception as e:
            print(f"Failed to delete: {e}")
    
    def reset(self):
        """Collection 초기화"""
        try:
            collections = self._client.get_collections().collections
            collection_names = [col.name for col in collections]
            if self._collection_name in collection_names:
                self._client.delete_collection(self._collection_name)
            self._create_collection()
            print("Collection reset")
        except Exception as e:
            print(f"Failed to reset: {e}")

    def _ensure_dimension(self, expected_dim: int):
        """
        Qdrant 컬렉션 차원을 기대값과 맞춰줌.
        서버 측 컬렉션이 이전 모델 크기를 유지하고 있을 수 있으므로
        매 호출 시 검증한다.
        """
        try:
            collection_info = self._client.get_collection(self._collection_name)
            current_dim = collection_info.config.params.vectors.size
            if current_dim != expected_dim:
                print(
                    f"Collection dimension {current_dim} != expected {expected_dim}, recreating..."
                )
                self.set_dimension(expected_dim)
            else:
                self._dimension = current_dim
        except Exception as e:
            print(f"Failed to verify dimension: {e}")


def get_qdrant_client() -> QdrantClientWrapper:
    """Qdrant 클라이언트 가져오기"""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClientWrapper()
    return _qdrant_client


# 싱글톤 인스턴스 (지연 생성)
_qdrant_client: Optional[QdrantClientWrapper] = None
