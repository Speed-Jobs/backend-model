from sqlalchemy.orm import Session
from typing import List
from app.utils.ingestion import (
    DocumentLoader,
    TextPreprocessor,
    TextChunker,
    Embedder,
    VectorIndexer
)
from app.schemas.ingestion import PostData
from app.core.config import settings


class IngestionService:
    """
    Ingestion 비즈니스 로직
    Post 원본 그대로 500 토큰 청킹
    """
    
    def __init__(self):
        self.loader = DocumentLoader()
        self.preprocessor = TextPreprocessor()
        self.chunker = TextChunker()
        self.embedder = Embedder()
        self.indexer = VectorIndexer()
    
    async def ingest_all_posts(self, db: Session, batch_size: int = 100):
        """모든 Post를 VectorDB에 적재 (500 토큰 청킹)"""
        print("Starting full ingestion...")
        
        total_count = await self.loader.count_posts(db)
        print(f"Total posts to ingest: {total_count}")
        
        all_doc_ids = []
        total_chunks = 0
        
        for offset in range(0, total_count, batch_size):
            print(f"\nProcessing batch: {offset} - {offset + batch_size}")
            
            posts = await self.loader.load_posts_batch(db, offset, batch_size)
            doc_ids = await self._process_posts_batch(posts)
            all_doc_ids.extend(doc_ids)
            total_chunks += len(doc_ids)
            
            print(f"Batch complete: {len(doc_ids)} chunks indexed")
        
        result = {
            "success": True,
            "message": f"Successfully ingested {total_count} posts",
            "total_posts": total_count,
            "total_chunks": total_chunks,
            "document_ids": all_doc_ids[:100]
        }
        
        print(f"\n🎉 Ingestion complete!")
        print(f"   - Total posts: {total_count}")
        print(f"   - Total chunks: {total_chunks}")
        
        return result
    
    async def ingest_posts_by_ids(self, db: Session, post_ids: List[int]):
        """특정 Post들만 적재"""
        print(f"Ingesting posts: {post_ids}")
        
        posts = await self.loader.load_posts_by_ids(db, post_ids)
        print(f"Loaded {len(posts)} posts")
        
        doc_ids = await self._process_posts_batch(posts)
        
        result = {
            "success": True,
            "message": f"Successfully ingested {len(posts)} posts",
            "total_posts": len(posts),
            "total_chunks": len(doc_ids),
            "document_ids": doc_ids
        }
        
        print(f"Ingestion complete: {len(doc_ids)} chunks")
        return result
    
    async def _process_posts_batch(self, posts: List[PostData]) -> List[str]:
        """
        Post 배치 처리 (원본 그대로 500 토큰 청킹)
        
        Pipeline:
        1. Post 원본 텍스트 (title + description 등)
        2. 전처리
        3. 500 토큰 청킹
        4. Embedding
        5. Indexing
        """
        all_texts = []
        all_metadatas = []
        
        for post in posts:
            # 원본 텍스트 조합 (포맷팅 없이)
            text_parts = []
            
            if post.title:
                text_parts.append(post.title)
            
            if post.description:
                text_parts.append(post.description)
            
            # 단순 결합 (포맷 없음)
            raw_text = "\n\n".join(text_parts)
            
            # 전처리
            cleaned_text = await self.preprocessor.clean(raw_text)
            
            # 메타데이터
            base_metadata = {
                'post_id': post.id,
                'company_id': post.company_id,
                'industry_id': post.industry_id,
                'employment_type': post.employment_type,
                'work_type': post.work_type,
                'experience': post.experience,
                'source': 'mysql'
            }
            
            # 500 토큰 청킹
            chunks = await self.chunker.split_by_sentences(
                cleaned_text, 
                metadata=base_metadata
            )
            
            # 수집
            for chunk in chunks:
                all_texts.append(chunk['text'])
                all_metadatas.append(chunk['metadata'])
        
        if not all_texts:
            print("No texts to embed")
            return []
        
        print(f"Generated {len(all_texts)} chunks from {len(posts)} posts")
        
        # Embedding
        embeddings = await self.embedder.embed_batch(all_texts)
        
        # Indexing
        doc_ids = await self.indexer.store(
            texts=all_texts,
            embeddings=embeddings,
            metadatas=all_metadatas
        )
        
        return doc_ids
    
    async def delete_post(self, post_id: int):
        """특정 Post 삭제"""
        await self.indexer.delete_by_post_id(post_id)
        return {"success": True, "message": f"Deleted post {post_id}"}
    
    async def get_stats(self):
        """VectorDB 통계"""
        count = await self.indexer.count()
        return {
            "total_documents": count,
            "qdrant_url": settings.QDRANT_URL,
            "collection_name": settings.QDRANT_COLLECTION_NAME,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.get_embedding_dimension()
        }