"""
Evaluation API Request Schemas
평가 API 요청 스키마 정의
"""

from pydantic import BaseModel, Field


class TwoPostsRequest(BaseModel):
    """2개 채용공고 비교 평가 요청"""
    sk_ax_post: int = Field(
        ..., 
        description="SK AX 채용공고 ID",
        ge=1,
        example=123
    )
    competitor_post: int = Field(
        ..., 
        description="경쟁사 채용공고 ID",
        ge=1,
        example=456
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sk_ax_post": 123,
                "competitor_post": 456
            }
        }

