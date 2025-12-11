from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from app.utils.retrieval import Retriever
from app.schemas.retrieval import SearchResult
from app.schemas.agent.schemas_retrieval import PostData
from app.models.post import Post


class RetrievalService:
    """Retrieval 비즈니스 로직"""

    def __init__(self):
        self.retriever = Retriever()

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        db: Optional[Session] = None
    ) -> list[SearchResult]:
        """
        텍스트 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            filters: 메타데이터 필터
            db: 데이터베이스 세션 (Post 전체 데이터 조회용)

        Returns:
            검색 결과 리스트
        """
        # Vector search 수행
        results = await self.retriever.search(
            query=query,
            top_k=top_k,
            filters=filters
        )

        # Post 데이터 조회 및 metadata에 추가
        search_results = []
        for result in results:
            # DB session이 있으면 post 전체 데이터 조회하여 metadata에 병합
            if db:
                post = db.query(Post).filter(Post.id == result['post_id']).first()
                if post:
                    # Post 데이터를 dict로 변환 (id 필드는 제외 - post_id와 중복)
                    post_data = PostData.model_validate(post).model_dump(exclude_none=True, exclude={'id'})
                    # 기존 metadata에 post 데이터 병합
                    result['metadata'].update(post_data)

            search_result = SearchResult(**result)
            search_results.append(search_result)

        return search_results
