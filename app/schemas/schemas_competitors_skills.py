from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class SkillDiversity(BaseModel):
    """회사별 스킬 다양성"""
    company: str = Field(..., description="회사명")
    skills: int = Field(..., description="스킬 개수", ge=0)

class SkillDiversityData(BaseModel):
    """스킬 다양성 데이터"""
    view_mode: str = Field(..., description="조회 모드 (all, yearly)")
    year: Optional[int] = Field(None, description="조회 연도")
    diversity: List[SkillDiversity] = Field(..., description="회사별 스킬 다양성")

class SkillDiversityResponse(BaseModel):
    """스킬 다양성 조회 응답"""
    status: int = Field(200, description="HTTP 상태 코드")
    code: str = Field("SUCCESS", description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    data: SkillDiversityData


class PostWithSkills(BaseModel):
    """공고 및 스킬 정보"""
    post_id: int
    post_title: str
    posted_at: Optional[datetime]
    close_at: Optional[datetime]
    crawled_at: Optional[datetime]
    company_id: int
    company_name: str
    skills: str  # 콤마로 구분된 스킬 목록
    
    class Config:
        from_attributes = True


class PostsWithSkillsData(BaseModel):
    """공고 목록 데이터"""
    company_name: Optional[str] = None
    year: Optional[int] = None
    total_count: int
    posts: List[PostWithSkills]


class PostsWithSkillsResponse(BaseModel):
    """공고 목록 조회 응답"""
    status: int = 200
    code: str = "SUCCESS"
    message: str
    data: PostsWithSkillsData


class SkillTrend(BaseModel):
    """분기별 스킬 트렌드"""
    quarter: str = Field(..., description="분기 (YYYY QN 형식, 예: '2025 Q3', '2025 Q4')")
    skills: Dict[str, int] = Field(..., description="각 스킬별 공고 수 (해당 분기의 월별 데이터 집계값)")


class SkillTrendData(BaseModel):
    """스킬 트렌드 데이터"""
    company: str = Field(..., description="회사명")
    year: Optional[int] = Field(None, description="조회 연도 (None일 경우 다년도 조회)")
    years: Optional[List[int]] = Field(None, description="조회 연도 목록 (다년도 조회 시 사용)")
    trends: List[SkillTrend] = Field(default_factory=list, description="분기별 트렌드 목록 (연도 지정 시만 사용)")
    skill_frequencies: Optional[Dict[str, Dict[str, int]]] = Field(None, description="연도별 스킬 빈도수 (연도 미지정 시 근 5개년 각 연도별 상위 스킬, 키: 연도 문자열, 값: 스킬별 빈도수)")


class SkillTrendResponse(BaseModel):
    """스킬 트렌드 조회 응답"""
    status: int = Field(200, description="HTTP 상태 코드")
    code: str = Field("SUCCESS", description="응답 코드")
    message: str = Field(..., description="응답 메시지")
    data: SkillTrendData