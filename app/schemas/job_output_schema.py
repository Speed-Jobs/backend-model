"""
Job Matching Output Schema
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class JobMatchResult(BaseModel):
    """Single job matching result (v13)"""
    job_name: str = Field(..., description="Job name")
    industry: str = Field(..., description="Industry/Sub-field")
    final_score: float = Field(..., description="Final matching score (0~2, Jaccard + SBERT)")

    jaccard_score: float = Field(default=0.0, description="Jaccard similarity score (0~1)")
    embedding_score: float = Field(default=0.0, description="SBERT embedding similarity score (0~1)")
    pagerank_score: float = Field(default=0.0, description="PPR score (for logging only, not in final score)")

    matching_skills: List[str] = Field(default_factory=list, description="Matched skill list")
    job_definition: str = Field(default="", description="Job definition")
    reason: str = Field(default="", description="Matching reason")

    class Config:
        json_schema_extra = {
            "example": {
                "job_name": "Software Development",
                "industry": "Back-end Development",
                "final_score": 1.4723,
                "jaccard_score": 0.65,
                "embedding_score": 0.82,
                "pagerank_score": 0.78,
                "matching_skills": ["python", "django", "postgresql", "redis"],
                "job_definition": "Software development using various programming languages...",
                "reason": "의미 + 스킬 매칭 강함 (SBERT: 0.820, Jacc: 0.650), 스킬 4개"
            }
        }


class SingleJobMatchingResponse(BaseModel):
    """Single job posting matching response"""
    posting_id: str = Field(..., description="Job posting ID")
    company: str = Field(..., description="Company name")
    title: str = Field(..., description="Job posting title")
    url: Optional[str] = Field(default="", description="Job posting URL")
    matches: List[JobMatchResult] = Field(..., description="Matching result list (Top-K)")

    class Config:
        json_schema_extra = {
            "example": {
                "posting_id": "kakao_001",
                "company": "Kakao",
                "title": "Backend Developer",
                "url": "https://careers.kakao.com/jobs/123",
                "matches": [
                    {
                        "job_name": "Software Development",
                        "industry": "Back-end Development",
                        "final_score": 1.4723,
                        "jaccard_score": 0.65,
                        "embedding_score": 0.82,
                        "pagerank_score": 0.78,
                        "matching_skills": ["python", "django"],
                        "job_definition": "...",
                        "reason": "매칭 스킬 2개: python, django (Jacc: 0.650)"
                    }
                ]
            }
        }


class BatchJobMatchingResponse(BaseModel):
    """Batch job posting matching response"""
    total_count: int = Field(..., description="Total processed job postings")
    success_count: int = Field(..., description="Successfully matched count")
    results: List[SingleJobMatchingResponse] = Field(..., description="Matching result list")

    class Config:
        json_schema_extra = {
            "example": {
                "total_count": 10,
                "success_count": 10,
                "results": []
            }
        }


class JobMatchingSummary(BaseModel):
    """Matching result summary statistics"""
    total_postings: int
    job_distribution: dict = Field(default_factory=dict, description="Job distribution")
    industry_distribution: dict = Field(default_factory=dict, description="Industry distribution")
