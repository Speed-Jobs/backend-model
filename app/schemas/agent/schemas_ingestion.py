from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

class IngestRequest(BaseModel):
    """데이터 적재 요청"""
    post_ids: Optional[List[int]] = Field(None, description="특정 post ID들만 적재 (없으면 전체)")
    batch_size: int = Field(100, description="배치 크기")


class IngestResponse(BaseModel):
    """데이터 적재 응답"""
    success: bool
    message: str
    total_posts: int
    total_chunks: int
    document_ids: List[str]



class PostData(BaseModel):
    id: int
    title: Optional[str] = None
    employment_type: Optional[str] = None
    experience: Optional[str] = None
    work_type: Optional[str] = None
    description: Optional[str] = None
    meta_data: Optional[Dict[str, Any]] = None
    source_url: Optional[str] = None
    url_hash: Optional[str] = None
    screenshot_url: Optional[str] = None
    company_id: Optional[int] = None
    industry_id: Optional[int] = None
    posted_at: Optional[datetime] = None
    close_at: Optional[datetime] = None
    crawled_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    is_deleted: Optional[int] = None
    
    @field_validator('meta_data', mode='before')
    @classmethod
    def parse_meta_data(cls, v):
        """meta_data를 문자열에서 dict로 변환"""
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            if not v or v.strip() in ['', '{}']:
                return None
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return None

    @field_validator('posted_at', 'close_at', 'crawled_at', 'created_at', 'updated_at', 'modified_at', mode='before')
    @classmethod
    def parse_datetime(cls, v):
        """잘못된 날짜 값 처리 (MySQL의 '0000-00-00 00:00:00' 등)"""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            # MySQL의 잘못된 날짜 값 처리
            if v.startswith('0000-00-00') or not v.strip():
                return None
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return None
        return None
    
    class Config:
        from_attributes = True