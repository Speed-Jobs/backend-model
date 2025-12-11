from fastapi import APIRouter, Depends, HTTPException
import logging
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.services.ingestion_service import IngestionService
from app.schemas.ingestion import IngestRequest, IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingestion"])
logger = logging.getLogger(__name__)


@router.post("/all", response_model=IngestResponse)
async def ingest_all_posts(
    request: IngestRequest = IngestRequest(),
    db: Session = Depends(get_db)
):
    """
    모든 Post 데이터를 VectorDB에 적재
    
    - MySQL의 모든 post 테이블 데이터를 가져와서 embedding
    - ChromaDB에 저장
    """
    try:
        service = IngestionService()
        result = await service.ingest_all_posts(
            db=db,
            batch_size=request.batch_size
        )
        return IngestResponse(**result)
    except Exception as e:
        logger.exception("Error in ingest_all_posts")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/by-ids", response_model=IngestResponse)
async def ingest_posts_by_ids(
    request: IngestRequest,
    db: Session = Depends(get_db)
):
    """
    특정 Post ID들만 VectorDB에 적재
    
    - 지정된 post_ids만 가져와서 embedding
    """
    if not request.post_ids:
        raise HTTPException(status_code=400, detail="post_ids is required")
    
    try:
        service = IngestionService()
        result = await service.ingest_posts_by_ids(
            db=db,
            post_ids=request.post_ids
        )
        return IngestResponse(**result)
    except Exception as e:
        logger.exception("Error in ingest_posts_by_ids")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/post/{post_id}")
async def delete_post(post_id: int):
    """
    특정 Post를 VectorDB에서 삭제
    """
    try:
        service = IngestionService()
        result = await service.delete_post(post_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """
    VectorDB 통계 조회
    
    - 총 문서 수
    - 컬렉션 이름
    - 임베딩 모델 정보
    """
    try:
        service = IngestionService()
        stats = await service.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
