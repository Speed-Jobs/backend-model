"""
Agent API Response Schemas
Agent 관련 응답 스키마 정의
"""

from pydantic import BaseModel, Field


class ReportGenerationResponse(BaseModel):
    """보고서 생성 응답 스키마"""
    
    status: str = Field(
        ..., 
        description="생성 상태 (success/error)",
        example="success"
    )
    improved_posting: str = Field(
        ..., 
        description="개선된 채용공고 전문",
        example="[채용공고]\n\n회사명: 현대오토에버\n직무: 백엔드 개발자\n..."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "improved_posting": "[채용공고]\n\n회사명: 현대오토에버\n직무: CCS신뢰성개발팀 백엔드 개발자\n\n담당업무:\n- 커넥티드카 서비스 안정화를 위한 코드 개선\n- 새로운 기능 개발 및 배포\n..."
            }
        }

