"""
Job Posting Detail Report Schemas
채용공고 상세 정보 응답 스키마 정의
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class JobPostingDetailReport(BaseModel):
    """채용공고 상세 정보 리포트 스키마"""
    
    # 1. 회사명
    company_name: str = Field(
        ...,
        description="회사명",
        example="현대오토에버"
    )
    
    # 2. 채용 직무/포지션
    position: str = Field(
        ...,
        description="채용 직무/포지션",
        example="백엔드 개발자"
    )
    
    # 3. 고용 형태
    employment_type: Optional[str] = Field(
        None,
        description="고용 형태 (정규직, 계약직, 인턴 등)",
        example="정규직"
    )
    
    # 4. 근무지
    work_location: Optional[str] = Field(
        None,
        description="근무지",
        example="서울시 강남구"
    )
    
    # 5. 마감일
    deadline: Optional[str] = Field(
        None,
        description="마감일",
        example="2024-12-31"
    )
    
    # 6. 회사 소개
    company_introduction: Optional[str] = Field(
        None,
        description="회사 소개",
        example="현대오토에버는 현대자동차그룹의 IT 전문 기업으로..."
    )
    
    # 7. 주요 업무 (담당업무)
    main_responsibilities: Optional[str] = Field(
        None,
        description="주요 업무 (담당업무)",
        example="- 백엔드 시스템 개발 및 유지보수\n- API 설계 및 개발\n- 데이터베이스 설계 및 최적화"
    )
    
    # 8. 자격 요건 (필수 요건)
    required_qualifications: Optional[str] = Field(
        None,
        description="자격 요건 (필수 요건)",
        example="- 컴퓨터공학 또는 관련 학과 학사 이상\n- Python, Java 등 백엔드 개발 경험 3년 이상"
    )
    
    # 9. 우대 사항
    preferred_qualifications: Optional[str] = Field(
        None,
        description="우대 사항",
        example="- 클라우드 환경(AWS, GCP) 경험\n- 마이크로서비스 아키텍처 경험"
    )
    
    # 10. 기술 스택
    tech_stack: Optional[List[str]] = Field(
        None,
        description="기술 스택",
        example=["Python", "Django", "PostgreSQL", "Docker", "Kubernetes"]
    )
    
    # 11. 팀 소개
    team_introduction: Optional[str] = Field(
        None,
        description="팀 소개",
        example="CCS신뢰성개발팀은 커넥티드카 서비스의 안정성을 담당하는 팀입니다..."
    )
    
    # 12. 개발 문화
    development_culture: Optional[str] = Field(
        None,
        description="개발 문화",
        example="- 코드 리뷰 문화\n- 주간 기술 세미나\n- 오픈소스 기여 장려"
    )
    
    # 13. 사용 도구
    tools: Optional[List[str]] = Field(
        None,
        description="사용 도구",
        example=["Jira", "Confluence", "GitLab", "Slack"]
    )
    
    # 14. 프로젝트 소개
    project_introduction: Optional[str] = Field(
        None,
        description="프로젝트 소개",
        example="커넥티드카 플랫폼 개발 프로젝트로 차량과 클라우드를 연결하는..."
    )
    
    # 15. 성장 기회
    growth_opportunities: Optional[str] = Field(
        None,
        description="성장 기회",
        example="- 다양한 기술 스택 학습 기회\n- 컨퍼런스 참가 지원\n- 내부 교육 프로그램"
    )
    
    # 16. 전형 절차
    recruitment_process: Optional[str] = Field(
        None,
        description="전형 절차",
        example="1차 서류전형 → 2차 기술면접 → 3차 임원면접"
    )
    
    # 17. 근무 조건
    work_conditions: Optional[str] = Field(
        None,
        description="근무 조건",
        example="- 주 5일 근무\n- 유연근무제\n- 재택근무 가능"
    )
    
    # 18. 복리후생
    benefits: Optional[str] = Field(
        None,
        description="복리후생",
        example="- 건강검진 지원\n- 휴가비 지원\n- 사내 카페테리아\n- 체육시설 이용"
    )
    
    # 19. 지원 방법
    application_method: Optional[str] = Field(
        None,
        description="지원 방법",
        example="채용사이트를 통해 지원하시거나 이메일로 이력서를 보내주세요."
    )
    
    # 20. 기타 사항
    additional_info: Optional[str] = Field(
        None,
        description="기타 사항",
        example="문의사항은 채용담당자에게 연락주시기 바랍니다."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "company_name": "현대오토에버",
                "position": "백엔드 개발자",
                "employment_type": "정규직",
                "work_location": "서울시 강남구",
                "deadline": "2024-12-31",
                "company_introduction": "현대오토에버는 현대자동차그룹의 IT 전문 기업으로...",
                "main_responsibilities": "- 백엔드 시스템 개발 및 유지보수\n- API 설계 및 개발",
                "required_qualifications": "- 컴퓨터공학 또는 관련 학과 학사 이상\n- Python, Java 등 백엔드 개발 경험 3년 이상",
                "preferred_qualifications": "- 클라우드 환경(AWS, GCP) 경험",
                "tech_stack": ["Python", "Django", "PostgreSQL", "Docker"],
                "team_introduction": "CCS신뢰성개발팀은...",
                "development_culture": "- 코드 리뷰 문화\n- 주간 기술 세미나",
                "tools": ["Jira", "Confluence", "GitLab"],
                "project_introduction": "커넥티드카 플랫폼 개발 프로젝트...",
                "growth_opportunities": "- 다양한 기술 스택 학습 기회",
                "recruitment_process": "1차 서류전형 → 2차 기술면접",
                "work_conditions": "- 주 5일 근무\n- 유연근무제",
                "benefits": "- 건강검진 지원\n- 휴가비 지원",
                "application_method": "채용사이트를 통해 지원",
                "additional_info": "문의사항은 채용담당자에게 연락주세요."
            }
        }


class JobPostingDetailReportResponse(BaseModel):
    """채용공고 상세 정보 리포트 응답 스키마"""
    
    status: str = Field(
        ...,
        description="응답 상태 (success/error)",
        example="success"
    )
    
    data: Optional[JobPostingDetailReport] = Field(
        None,
        description="채용공고 상세 정보"
    )
    
    message: Optional[str] = Field(
        None,
        description="응답 메시지",
        example="채용공고 상세 정보를 성공적으로 조회했습니다."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "data": {
                    "company_name": "현대오토에버",
                    "position": "백엔드 개발자",
                    "employment_type": "정규직",
                    "work_location": "서울시 강남구",
                    "deadline": "2024-12-31",
                    "company_introduction": "현대오토에버는 현대자동차그룹의 IT 전문 기업으로...",
                    "main_responsibilities": "- 백엔드 시스템 개발 및 유지보수",
                    "required_qualifications": "- 컴퓨터공학 또는 관련 학과 학사 이상",
                    "preferred_qualifications": "- 클라우드 환경(AWS, GCP) 경험",
                    "tech_stack": ["Python", "Django", "PostgreSQL"],
                    "team_introduction": "CCS신뢰성개발팀은...",
                    "development_culture": "- 코드 리뷰 문화",
                    "tools": ["Jira", "Confluence"],
                    "project_introduction": "커넥티드카 플랫폼 개발 프로젝트...",
                    "growth_opportunities": "- 다양한 기술 스택 학습 기회",
                    "recruitment_process": "1차 서류전형 → 2차 기술면접",
                    "work_conditions": "- 주 5일 근무",
                    "benefits": "- 건강검진 지원",
                    "application_method": "채용사이트를 통해 지원",
                    "additional_info": "문의사항은 채용담당자에게 연락주세요."
                },
                "message": "채용공고 상세 정보를 성공적으로 조회했습니다."
            }
        }

