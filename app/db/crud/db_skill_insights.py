"""
스킬 트렌드 관련 DB CRUD
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd

from app.models.post import Post
from app.models.post_skill import PostSkill
from app.models.skill import Skill

EFFECTIVE_POSTED_AT = func.coalesce(Post.posted_at, Post.crawled_at)

def get_recent_years(db: Session, years: int = 5) -> List[int]:
    """최근 N년의 연도 목록 조회"""
    query = text("""
        SELECT DISTINCT YEAR(COALESCE(p.posted_at, p.crawled_at)) as year
        FROM post p
        WHERE COALESCE(p.posted_at, p.crawled_at) >= DATE_SUB(CURDATE(), INTERVAL :years YEAR)
            AND p.is_deleted = 0
        ORDER BY year DESC
        LIMIT :years
    """)
    
    result = db.execute(query, {"years": years})
    return [row.year for row in result]

def get_top_skills_by_period(
    db: Session,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    top_n: int = 10
) -> List[str]:
    """특정 기간 동안의 상위 N개 스킬 조회"""
    effective_date = EFFECTIVE_POSTED_AT
    
    query = db.query(
        Skill.name,
        func.count(PostSkill.id).label('count')
    ).join(
        PostSkill, Skill.id == PostSkill.skill_id
    ).join(
        Post, PostSkill.post_id == Post.id
    ).filter(
        Post.is_deleted.is_(False)
    )
    
    if start_date:
        query = query.filter(effective_date >= start_date)
    if end_date:
        query = query.filter(effective_date <= end_date)
    
    query = query.group_by(Skill.name).order_by(
        func.count(PostSkill.id).desc()
    ).limit(top_n)
    
    result = query.all()
    return [row.name for row in result]

def get_skill_frequencies_by_years(
    db: Session,
    years: List[int],
    top_n: int = 10
) -> pd.DataFrame:
    """연도별 스킬 빈도수 조회 (판다스 DataFrame 반환)"""
    # 먼저 최근 5년간 상위 N개 스킬 식별
    five_years_ago = date.today().replace(year=date.today().year - 5)
    
    top_skills_query = db.query(
        Skill.name,
        func.count(PostSkill.id).label('total_count')
    ).join(
        PostSkill, Skill.id == PostSkill.skill_id
    ).join(
        Post, PostSkill.post_id == Post.id
    ).filter(
        EFFECTIVE_POSTED_AT >= five_years_ago,
        Post.is_deleted.is_(False)
    ).group_by(
        Skill.name
    ).order_by(
        func.count(PostSkill.id).desc()
    ).limit(top_n)
    
    top_skills = [row.name for row in top_skills_query.all()]
    
    if not top_skills:
        return pd.DataFrame(columns=['year', 'skill_name', 'frequency'])
    
    # 연도별 스킬 빈도수 조회
    rows = []
    for year in years:
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        
        query = db.query(
            func.extract('year', EFFECTIVE_POSTED_AT).label('year'),
            Skill.name.label('skill_name'),
            func.count(PostSkill.id).label('frequency')
        ).join(
            PostSkill, Skill.id == PostSkill.skill_id
        ).join(
            Post, PostSkill.post_id == Post.id
        ).filter(
            EFFECTIVE_POSTED_AT >= year_start,
            EFFECTIVE_POSTED_AT <= year_end,
            Post.is_deleted.is_(False),
            Skill.name.in_(top_skills) if top_skills else True
        ).group_by(
            func.extract('year', EFFECTIVE_POSTED_AT),
            Skill.name
        )
        
        result = query.all()
        for row in result:
            rows.append({
                'year': int(row.year),
                'skill_name': row.skill_name,
                'frequency': row.frequency
            })
    
    df = pd.DataFrame(rows)
    return df

def get_quarterly_skill_trends(
    db: Session,
    year: int,
    comparison_year: Optional[int] = None,
    top_n: int = 10
) -> pd.DataFrame:
    """연도별 분기별 스킬 트렌드 조회 (판다스 DataFrame 반환)"""
    if comparison_year is None:
        comparison_year = year - 1
    
    # 현재 시점 기준으로 비교할 분기들 계산
    current_date = datetime.now().date()
    current_year = current_date.year
    current_quarter = (current_date.month - 1) // 3 + 1
    
    # 현재 연도의 상위 N개 스킬 식별 (전체 연도 데이터 기준)
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)
    
    top_skills_query = db.query(
        Skill.name,
        func.count(PostSkill.id).label('total_count')
    ).join(
        PostSkill, Skill.id == PostSkill.skill_id
    ).join(
        Post, PostSkill.post_id == Post.id
    ).filter(
        EFFECTIVE_POSTED_AT >= year_start,
        EFFECTIVE_POSTED_AT <= year_end,
        Post.is_deleted.is_(False)
    ).group_by(
        Skill.name
    ).order_by(
        func.count(PostSkill.id).desc()
    ).limit(top_n)
    
    top_skills = [row.name for row in top_skills_query.all()]
    
    if not top_skills:
        return pd.DataFrame(columns=['year', 'quarter', 'skill_name', 'count'])
    
    # 분기별 데이터 조회
    rows = []
    for y in [year, comparison_year]:
        for quarter in [1, 2, 3, 4]:
            # 현재 분기보다 미래 분기는 제외
            if y == current_year and quarter > current_quarter:
                continue
            if y == comparison_year and current_year == year and quarter > current_quarter:
                continue
            
            quarter_start = date(y, (quarter - 1) * 3 + 1, 1)
            if quarter == 4:
                quarter_end = date(y, 12, 31)
            else:
                # 다음 분기 시작일의 하루 전
                next_quarter_month = quarter * 3 + 1
                if next_quarter_month > 12:
                    quarter_end = date(y, 12, 31)
                else:
                    quarter_end = date(y, next_quarter_month, 1) - timedelta(days=1)
            
            # 현재 분기의 경우 오늘까지의 데이터만
            if y == current_year and quarter == current_quarter:
                quarter_end = current_date
            elif y == comparison_year and current_year == year and quarter == current_quarter:
                quarter_end = date(comparison_year, current_date.month, current_date.day)
            
            query = db.query(
                func.extract('year', EFFECTIVE_POSTED_AT).label('year'),
                func.extract('quarter', EFFECTIVE_POSTED_AT).label('quarter'),
                Skill.name.label('skill_name'),
                func.count(PostSkill.id).label('count')
            ).join(
                PostSkill, Skill.id == PostSkill.skill_id
            ).join(
                Post, PostSkill.post_id == Post.id
            ).filter(
                EFFECTIVE_POSTED_AT >= quarter_start,
                EFFECTIVE_POSTED_AT <= quarter_end,
                Post.is_deleted.is_(False),
                Skill.name.in_(top_skills) if top_skills else True
            ).group_by(
                func.extract('year', EFFECTIVE_POSTED_AT),
                func.extract('quarter', EFFECTIVE_POSTED_AT),
                Skill.name
            )
            
            result = query.all()
            for row in result:
                rows.append({
                    'year': int(row.year),
                    'quarter': int(row.quarter),
                    'skill_name': row.skill_name,
                    'count': row.count
                })
    
    df = pd.DataFrame(rows)
    return df

