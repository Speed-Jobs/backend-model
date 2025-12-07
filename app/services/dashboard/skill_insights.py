"""
스킬 트렌드 서비스
"""
from sqlalchemy.orm import Session
from typing import Optional, Dict, List
from datetime import datetime, date, timedelta
import pandas as pd

from app.db.crud import db_skill_insights
from app.schemas import schemas_skill_insights
from app.config.company_groups import get_company_patterns
from app.services.skill_match.skill_match import load_skill_association_model, get_similar_skills, MODEL_PATH


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


    def get_skill_statistics(
        self,
        db: Session,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        company: Optional[str] = None,
        limit: int = 5
    ) -> schemas_skill_insights.SkillStatisticsData:
        """스킬 통계 조회 (스킬 클라우드용)
        
        Args:
            db: 데이터베이스 세션
            start_date: 시작 날짜 (YYYY-MM-DD, None일 경우 현재 연도 1월 1일)
            end_date: 종료 날짜 (YYYY-MM-DD, None일 경우 현재 날짜)
            company: 회사 키워드 (선택사항, 예: "toss", "kakao") - COMPANY_GROUPS에서 패턴 매핑
            limit: 상위 N개 스킬
            
        Returns:
            SkillStatisticsData: 스킬 통계 데이터
        """
        # 날짜 파싱
        today = date.today()
        if start_date is None:
            start_dt = date(today.year, 1, 1)
        else:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
        
        if end_date is None:
            end_dt = today
        else:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # 회사 키워드를 패턴 리스트로 변환
        company_patterns = None
        if company:
            company_patterns = get_company_patterns(company)
        
        # 스킬 통계 조회 (관련 스킬을 포함하기 위해 limit를 더 크게 설정)
        # 유사 스킬이 상위 N개에 포함되지 않을 수 있으므로 limit * 3으로 조회
        df = db_skill_insights.get_skill_statistics(
            db, start_dt, end_dt, company_patterns, limit * 3
        )
        
        if df.empty:
            return schemas_skill_insights.SkillStatisticsData(
                period=schemas_skill_insights.PeriodInfo(
                    start_date=start_dt.strftime('%Y-%m-%d'),
                    end_date=end_dt.strftime('%Y-%m-%d')
                ),
                company=company,
                total_job_postings=0,
                skills=[]
            )
        
        # 전체 공고 수 조회
        total_postings = db_skill_insights.get_total_job_postings(
            db, start_dt, end_dt, company_patterns
        )
        
        # 스킬 연관성 모델 로드
        try:
            model = load_skill_association_model(MODEL_PATH)
        except Exception as e:
            print(f"스킬 연관성 모델 로드 실패: {e}")
            model = None
        
        # 스킬 통계 정보 생성 (원래 limit만큼만 반환)
        skills = []
        for idx, (_, row) in enumerate(df.iterrows()):
            # 원래 limit만큼만 처리
            if idx >= limit:
                break
            skill_name = row['skill_name']
            skill_count = int(row['count'])
            skill_percentage = round((skill_count * 100.0 / total_postings), 1) if total_postings > 0 else 0.0
            skill_change = float(row['change'])
            
            # 관련 스킬 조회 (유사도 기반)
            related_skills = []
            if model is not None:
                try:
                    # 스킬 이름 정규화 (대소문자 통일, 공백 제거)
                    normalized_skill_name = skill_name.strip()
                    
                    # 유사 스킬을 더 많이 찾기 (상위 20개)
                    similar_skills = get_similar_skills(model, normalized_skill_name, top_n=5)
                    
                    if not similar_skills:
                        print(f"유사 스킬을 찾을 수 없습니다: '{normalized_skill_name}' (모델에 존재하지 않을 수 있음)")
                    else:
                        print(f"'{normalized_skill_name}' 유사 스킬 {len(similar_skills)}개 찾음")
                    
                    # 관련 스킬의 통계 정보 조회 (유사 스킬을 찾았으면 무조건 포함)
                    for related_skill_name, similarity in similar_skills:
                        # df에서 해당 스킬의 통계 찾기
                        related_df = df[df['skill_name'] == related_skill_name]
                        
                        if not related_df.empty:
                            # df에 있는 경우: 실제 통계 사용
                            related_row = related_df.iloc[0]
                            related_count = int(related_row['count'])
                            related_percentage = round((related_count * 100.0 / total_postings), 1) if total_postings > 0 else 0.0
                            related_change = float(related_row['change'])
                        else:
                            # df에 없는 경우: count=0, percentage=0.0, change=0.0으로 설정
                            related_count = 0
                            related_percentage = 0.0
                            related_change = 0.0
                        
                        # 유사 스킬을 찾았으면 무조건 추가
                        related_skills.append(
                            schemas_skill_insights.RelatedSkillInfo(
                                name=related_skill_name,
                                count=related_count,
                                percentage=related_percentage,
                                change=related_change,
                                similarity=round(float(similarity), 3)
                            )
                        )
                    
                    # 유사도 순으로 정렬하고 상위 5개만 선택
                    related_skills.sort(key=lambda x: x.similarity, reverse=True)
                    related_skills = related_skills[:5]
                    
                except Exception as e:
                    print(f"❌ 유사 스킬 조회 실패 ({skill_name}): {e}")
                    import traceback
                    traceback.print_exc()
            
            skills.append(
                schemas_skill_insights.SkillStatisticsInfo(
                    name=skill_name,
                    count=skill_count,
                    percentage=skill_percentage,
                    change=skill_change,
                    relatedSkills=related_skills
                )
            )
        
        return schemas_skill_insights.SkillStatisticsData(
            period=schemas_skill_insights.PeriodInfo(
                start_date=start_dt.strftime('%Y-%m-%d'),
                end_date=end_dt.strftime('%Y-%m-%d')
            ),
            company=company,
            total_job_postings=total_postings,
            skills=skills
        )


# 싱글톤 인스턴스
skill_insights_service = SkillInsightsService()

