"""
스킬 트렌드 서비스
"""
from sqlalchemy.orm import Session
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd

from app.db.crud import db_skill_insights
from app.schemas import schemas_skill_insights
from app.config.company_groups import get_company_patterns


class SkillInsightsService:
    """스킬 트렌드 분석 서비스"""
    
    def get_skill_trends(
        self,
        db: Session,
        year: Optional[str] = None,
        top_n: int = 10
    ) -> schemas_skill_insights.SkillTrendData:
        """스킬 트렌드 조회
        
        Args:
            db: 데이터베이스 세션
            year: 조회 연도 (None일 경우 최근 5개년)
            top_n: 상위 N개 스킬
            
        Returns:
            SkillTrendData: 스킬 트렌드 데이터
        """
        if year is None:
            # 최근 5개년 조회
            return self._get_multi_year_trends(db, top_n)
        else:
            # 단일 연도 조회 (분기별)
            year_int = int(year)
            return self._get_single_year_trends(db, year_int, top_n)
    
    def _get_multi_year_trends(
        self,
        db: Session,
        top_n: int
    ) -> schemas_skill_insights.SkillTrendData:
        """최근 5개년 스킬 트렌드 조회"""
        # 최근 5년 연도 목록 조회
        years = db_skill_insights.get_recent_years(db, years=5)
        
        if not years:
            return schemas_skill_insights.SkillTrendData(
                years=[],
                skills=[],
                skill_frequencies={}
            )
        
        # 연도별 스킬 빈도수 조회
        df = db_skill_insights.get_skill_frequencies_by_years(db, years, top_n)
        
        if df.empty:
            return schemas_skill_insights.SkillTrendData(
                years=[str(y) for y in years],
                skills=[],
                skill_frequencies={str(y): {} for y in years}
            )
        
        # 스킬 목록 추출 (전체 기간 합계 기준 정렬)
        skill_totals = df.groupby('skill_name')['frequency'].sum().sort_values(ascending=False)
        skills = skill_totals.head(top_n).index.tolist()
        
        # 연도별 스킬 빈도수 딕셔너리 생성
        skill_frequencies = {}
        for y in years:
            year_df = df[df['year'] == y]
            year_skill_dict = {}
            for skill in skills:
                skill_data = year_df[year_df['skill_name'] == skill]
                if not skill_data.empty:
                    year_skill_dict[skill] = int(skill_data.iloc[0]['frequency'])
                else:
                    year_skill_dict[skill] = 0
            skill_frequencies[str(y)] = year_skill_dict
        
        return schemas_skill_insights.SkillTrendData(
            years=[str(y) for y in years],
            skills=skills,
            skill_frequencies=skill_frequencies
        )
    
    def _get_single_year_trends(
        self,
        db: Session,
        year: int,
        top_n: int
    ) -> schemas_skill_insights.SkillTrendData:
        """단일 연도 분기별 스킬 트렌드 조회"""
        comparison_year = year - 1
        
        # 분기별 데이터 조회
        df = db_skill_insights.get_quarterly_skill_trends(
            db, year, comparison_year, top_n
        )
        
        if df.empty:
            return schemas_skill_insights.SkillTrendData(
                year=str(year),
                comparison_year=str(comparison_year),
                skills=[],
                quarterly_trends={
                    str(year): schemas_skill_insights.QuarterlyTrend(
                        Q1={}, Q2={}, Q3={}, Q4={}
                    ),
                    str(comparison_year): schemas_skill_insights.QuarterlyTrend(
                        Q1={}, Q2={}, Q3={}, Q4={}
                    )
                }
            )
        
        # 스킬 목록 추출 (해당 연도 전체 합계 기준 정렬)
        year_df = df[df['year'] == year]
        skill_totals = year_df.groupby('skill_name')['count'].sum().sort_values(ascending=False)
        skills = skill_totals.head(top_n).index.tolist()
        
        # 연도별 분기별 데이터 구성
        quarterly_trends = {}
        
        for y in [year, comparison_year]:
            year_str = str(y)
            year_data = df[df['year'] == y]
            
            # 각 분기별 데이터 딕셔너리 생성
            quarters = {}
            for quarter in [1, 2, 3, 4]:
                quarter_df = year_data[year_data['quarter'] == quarter]
                quarter_dict = {}
                for skill in skills:
                    skill_data = quarter_df[quarter_df['skill_name'] == skill]
                    if not skill_data.empty:
                        quarter_dict[skill] = int(skill_data.iloc[0]['count'])
                    else:
                        quarter_dict[skill] = 0
                quarters[f'Q{quarter}'] = quarter_dict
            
            quarterly_trends[year_str] = schemas_skill_insights.QuarterlyTrend(
                Q1=quarters['Q1'],
                Q2=quarters['Q2'],
                Q3=quarters['Q3'],
                Q4=quarters['Q4']
            )
        
        return schemas_skill_insights.SkillTrendData(
            year=str(year),
            comparison_year=str(comparison_year),
            skills=skills,
            quarterly_trends=quarterly_trends
        )
    
    def get_company_skill_trends(
        self,
        db: Session,
        company_keyword: str,
        year: Optional[str] = None,
        top_n: int = 10
    ) -> schemas_skill_insights.SkillTrendData:
        """회사별 스킬 트렌드 조회
        
        Args:
            db: 데이터베이스 세션
            company_keyword: 회사 키워드 (예: "toss", "kakao") - COMPANY_GROUPS 딕셔너리 참조
            year: 조회 연도 (None일 경우 최근 5개년)
            top_n: 상위 N개 스킬
            
        Returns:
            SkillTrendData: 스킬 트렌드 데이터
        """
        # 키워드를 패턴 리스트로 변환
        company_patterns = get_company_patterns(company_keyword)
        
        if year is None:
            # 최근 5개년 조회
            return self._get_company_multi_year_trends(db, company_keyword, company_patterns, top_n)
        else:
            # 단일 연도 조회 (분기별)
            year_int = int(year)
            return self._get_company_single_year_trends(db, company_keyword, company_patterns, year_int, top_n)
    
    def _get_company_multi_year_trends(
        self,
        db: Session,
        company_keyword: str,
        company_patterns: List[str],
        top_n: int
    ) -> schemas_skill_insights.SkillTrendData:
        """회사별 최근 5개년 스킬 트렌드 조회"""
        # 최근 5년 연도 목록 조회
        years = db_skill_insights.get_recent_years_by_company(db, company_patterns, years=5)
        
        if not years:
            return schemas_skill_insights.SkillTrendData(
                company_name=company_keyword,
                years=[],
                skills=[],
                skill_frequencies={}
            )
        
        # 연도별 스킬 빈도수 조회
        df = db_skill_insights.get_skill_frequencies_by_years_and_company(
            db, company_patterns, years, top_n
        )
        
        if df.empty:
            return schemas_skill_insights.SkillTrendData(
                company_name=company_keyword,
                years=[str(y) for y in years],
                skills=[],
                skill_frequencies={str(y): {} for y in years}
            )
        
        # 스킬 목록 추출 (전체 기간 합계 기준 정렬)
        skill_totals = df.groupby('skill_name')['frequency'].sum().sort_values(ascending=False)
        skills = skill_totals.head(top_n).index.tolist()
        
        # 연도별 스킬 빈도수 딕셔너리 생성
        skill_frequencies = {}
        for y in years:
            year_df = df[df['year'] == y]
            year_skill_dict = {}
            for skill in skills:
                skill_data = year_df[year_df['skill_name'] == skill]
                if not skill_data.empty:
                    year_skill_dict[skill] = int(skill_data.iloc[0]['frequency'])
                else:
                    year_skill_dict[skill] = 0
            skill_frequencies[str(y)] = year_skill_dict
        
        return schemas_skill_insights.SkillTrendData(
            company_name=company_keyword,
            years=[str(y) for y in years],
            skills=skills,
            skill_frequencies=skill_frequencies
        )
    
    def _get_company_single_year_trends(
        self,
        db: Session,
        company_keyword: str,
        company_patterns: List[str],
        year: int,
        top_n: int
    ) -> schemas_skill_insights.SkillTrendData:
        """회사별 단일 연도 분기별 스킬 트렌드 조회"""
        comparison_year = year - 1
        
        # 분기별 데이터 조회
        df = db_skill_insights.get_quarterly_skill_trends_by_company(
            db, company_patterns, year, comparison_year, top_n
        )
        
        if df.empty:
            return schemas_skill_insights.SkillTrendData(
                company_name=company_keyword,
                year=str(year),
                comparison_year=str(comparison_year),
                skills=[],
                quarterly_trends={
                    str(year): schemas_skill_insights.QuarterlyTrend(
                        Q1={}, Q2={}, Q3={}, Q4={}
                    ),
                    str(comparison_year): schemas_skill_insights.QuarterlyTrend(
                        Q1={}, Q2={}, Q3={}, Q4={}
                    )
                }
            )
        
        # 스킬 목록 추출 (해당 연도 전체 합계 기준 정렬)
        year_df = df[df['year'] == year]
        skill_totals = year_df.groupby('skill_name')['count'].sum().sort_values(ascending=False)
        skills = skill_totals.head(top_n).index.tolist()
        
        # 연도별 분기별 데이터 구성
        quarterly_trends = {}
        
        for y in [year, comparison_year]:
            year_str = str(y)
            year_data = df[df['year'] == y]
            
            # 각 분기별 데이터 딕셔너리 생성
            quarters = {}
            for quarter in [1, 2, 3, 4]:
                quarter_df = year_data[year_data['quarter'] == quarter]
                quarter_dict = {}
                for skill in skills:
                    skill_data = quarter_df[quarter_df['skill_name'] == skill]
                    if not skill_data.empty:
                        quarter_dict[skill] = int(skill_data.iloc[0]['count'])
                    else:
                        quarter_dict[skill] = 0
                quarters[f'Q{quarter}'] = quarter_dict
            
            quarterly_trends[year_str] = schemas_skill_insights.QuarterlyTrend(
                Q1=quarters['Q1'],
                Q2=quarters['Q2'],
                Q3=quarters['Q3'],
                Q4=quarters['Q4']
            )
        
        return schemas_skill_insights.SkillTrendData(
            company_name=company_keyword,
            year=str(year),
            comparison_year=str(comparison_year),
            skills=skills,
            quarterly_trends=quarterly_trends
        )


# 싱글톤 인스턴스
skill_insights_service = SkillInsightsService()

