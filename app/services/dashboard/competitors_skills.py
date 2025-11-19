from sqlalchemy.orm import Session
from typing import Optional
from app.db.crud import db_competitors_skills
from app.schemas import schemas_competitors_skills

class CompetitorsSkillsService:
    """경쟁사 스킬 분석 서비스"""
    
    def get_skill_diversity(
        self, 
        db: Session, 
        year: Optional[int] = None
    ) -> schemas_competitors_skills.SkillDiversityData:
        """회사별 스킬 다양성 조회"""
        
        if year is None:
            diversity_data = db_competitors_skills.get_competitors_skill_diversity_all(db)
            view_mode = "all"
        else:
            diversity_data = db_competitors_skills.get_competitors_skill_diversity_by_year(db, year)
            view_mode = "yearly"
        
        return schemas_competitors_skills.SkillDiversityData(
            view_mode=view_mode,
            year=year,
            diversity=[
                schemas_competitors_skills.SkillDiversity(**item) 
                for item in diversity_data
            ]
        )
    
    def get_posts_with_skills(
        self,
        db: Session,
        company_name: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 100
    ) -> schemas_competitors_skills.PostsWithSkillsData:
        """경쟁사 공고 및 스킬 상세 조회"""
        
        posts_data = db_competitors_skills.get_competitors_posts_with_skills(
            db=db,
            company_name=company_name,
            year=year,
            limit=limit
        )
        
        return schemas_competitors_skills.PostsWithSkillsData(
            company_name=company_name,
            year=year,
            total_count=len(posts_data),
            posts=[
                schemas_competitors_skills.PostWithSkills(**post) 
                for post in posts_data
            ]
        )
    
    def get_skill_trends(
        self,
        db: Session,
        company_id: str,
        year: Optional[int] = None,
        top_n: int = 10
    ) -> schemas_competitors_skills.SkillTrendData:
        """회사별 상위 스킬 분기별 트렌드 조회
        
        year가 None일 경우 현재 연도 기준 근 5개년치 각 연도별 상위 스킬의 빈도수를 반환합니다.
        year가 지정된 경우 해당 연도의 분기별 트렌드를 반환합니다.
        """
        
        trend_data = db_competitors_skills.get_company_skill_trends(
            db=db,
            company_id=company_id,
            year=year,
            top_n=top_n
        )
        
        return schemas_competitors_skills.SkillTrendData(
            company=trend_data["company"],
            year=trend_data.get("year"),
            years=trend_data.get("years"),
            trends=[
                schemas_competitors_skills.SkillTrend(**trend)
                for trend in trend_data.get("trends", [])
            ],
            skill_frequencies=trend_data.get("skill_frequencies")
        )

# 싱글톤 인스턴스
competitors_skills_service = CompetitorsSkillsService()