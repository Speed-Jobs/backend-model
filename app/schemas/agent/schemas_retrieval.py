from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from datetime import datetime
import json


class PostData(BaseModel):
    """Post 데이터 스키마 (검색 결과 포맷팅용)"""
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


class SearchQuery(BaseModel):
    """검색 쿼리"""
    text: str = Field(..., description="검색할 텍스트")
    top_k: int = Field(5, ge=1, le=50, description="반환할 결과 수")
    company_name: Optional[str] = Field(None, description="회사명으로 필터링 (예: toss, kakao). 미입력시 전체 검색")
    filters: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터 필터")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "백엔드 개발자 채용",
                "top_k": 5,
                "company_name": "카카오"
            }
        }

class SearchResult(BaseModel):
    """검색 결과"""
    id: str = Field(..., description="문서 ID")
    post_id: int = Field(..., description="원본 Post ID")
    text: str = Field(..., description="문서 텍스트")
    metadata: Dict[str, Any] = Field(..., description="메타데이터 (Post 테이블의 전체 데이터 포함)")
    distance: float = Field(..., description="거리 (낮을수록 유사)")
    score: float = Field(..., description="유사도 점수 (높을수록 유사)")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "post_123_chunk_0",
                "post_id": 123,
                "text": "제목: 백엔드 개발자 채용\n고용형태: 정규직...",
                "metadata": {
                    "post_id": 123,
                    "company_id": 456,
                    "employment_type": "정규직",
                    "chunk_index": 0,
                    "title": "백엔드 개발자 채용",
                    "description": "상세 설명...",
                    "experience": "3년 이상",
                    "work_type": "정규직",
                    "posted_at": "2024-01-01T00:00:00",
                    "source_url": "https://..."
                },
                "distance": 0.234,
                "score": 0.766
            }
        }


class SearchResponse(BaseModel):
    """검색 응답"""
    query: str
    results: list[SearchResult]
    total_results: int


class AgenticRAGQuery(BaseModel):
    """
    Agentic RAG 검색 쿼리

    LLM이 자동으로 처리:
    - 엔티티 추출 (회사명, 날짜, 직무 등)
    - 검색 소스 결정 (vectordb/websearch/both/stats)
    - 결과 개수 자동 결정
    """
    text: str = Field(..., description="자연어 질문 (LLM이 자동으로 분석)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "2025년 하반기 토스 채용공고 총 몇개야?"
            }
        }


class SourceDocument(BaseModel):
    """검색 소스 문서"""
    id: str = Field(..., description="문서 ID")
    text: str = Field(..., description="문서 텍스트")
    metadata: Dict[str, Any] = Field(..., description="메타데이터")
    score: Optional[float] = Field(None, description="검색 점수")
    source_type: str = Field(..., description="소스 타입 (vectordb 또는 web)")


class AgenticRAGResponse(BaseModel):
    """Agentic RAG 응답"""
    query: str = Field(..., description="검색 쿼리")
    answer: str = Field(..., description="LLM이 생성한 답변")
    sources: list[SourceDocument] = Field(..., description="검색된 소스 문서들")
    route_decision: str = Field(..., description="라우팅 결정 (vectordb/websearch/both)")
    total_sources: int = Field(..., description="전체 소스 개수")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "카카오 백엔드 채용 요구사항",
                "answer": "카카오의 백엔드 개발자 채용 요구사항은 다음과 같습니다...",
                "sources": [
                    {
                        "id": "post_8349_chunk_0",
                        "text": "...",
                        "metadata": {},
                        "score": 0.85,
                        "source_type": "vectordb"
                    }
                ],
                "route_decision": "both",
                "total_sources": 8
            }
        }
