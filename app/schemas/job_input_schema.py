"""
Job Posting Input Schema
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class SkillSetInfo(BaseModel):
    """Skill set information"""
    matched: bool = False
    match_score: int = 0
    skill_set: List[str] = Field(default_factory=list)


class JobPostingInput(BaseModel):
    """Job posting input schema (kakao_jobs.json format)"""
    title: str
    company: str
    location: Optional[str] = None
    employment_type: Optional[str] = None
    experience: Optional[str] = None
    crawl_date: Optional[str] = None
    posted_date: Optional[str] = None
    expired_date: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    skill_set_info: SkillSetInfo = Field(default_factory=SkillSetInfo)

    class Config:
        """
        Pydantic BaseModel의 동작을 설정하는 Config 클래스
        
        - json_schema_extra: Swagger UI (/docs)에서 "Example Value"로 표시되는 기본 JSON 예시
          FastAPI가 자동으로 OpenAPI 스키마에 포함시켜 Swagger UI에 보여줌
          사용자가 "Try it out" 클릭 시 자동으로 채워지는 예시 데이터
        """
        json_schema_extra = {
            "example": {
                "title": "Backend Developer",
                "company": "Kakao",
                "location": "Pangyo",
                "employment_type": "Full-time",
                "experience": "3년 이상",
                "crawl_date": "2025-11-13",
                "posted_date": "2025-11-13",
                "expired_date": None,
                "description": "Backend developer recruitment",
                "url": "https://careers.kakao.com/jobs/123",
                "skill_set_info": {
                    "matched": True,
                    "match_score": 5,
                    "skill_set": ["Python", "Django", "PostgreSQL", "Redis", "Docker"]
                }
            }
        }


class BatchJobPostingInput(BaseModel):
    """Batch job posting input"""
    job_postings: List[JobPostingInput]
