"""
HHI (Herfindahl-Hirschman Index) 수요 집중도 분석

이 스크립트는 채용 시장의 직군/직무별 수요 집중도를 HHI 지수로 분석합니다.

주요 기능:
1. 전체 HHI 집중도 계산
2. 직군(Position)별 HHI 집중도 계산
3. 직무(Industry)별 HHI 집중도 계산
4. 수요 편중 정도를 0~1 점수로 산출
5. 인사이트 텍스트 생성 (대시보드 인사이트 창용)

계산 공식:
    HHI = Σ(si²)
    - si: 각 직무/직군이 차지하는 채용 비율 (0~1)
    - 값이 클수록 특정 영역에 수요가 집중됨

해석:
    0.00 ~ 0.15: 다양한 직무가 고르게 채용 (경쟁 분산, 난이도 낮음)
    0.15 ~ 0.25: 특정 직무 채용이 증가 (경쟁 부분 집중, 난이도 중간)
    0.25 이상:   특정 직무로 쏠림 발생 (인재 쟁탈전, 난이도 높음)

사용법:
    python test_07_hhi_concentration.py [시작일] [종료일]

    예시:
    python test_07_hhi_concentration.py 2025-07-01 2025-12-31
    python test_07_hhi_concentration.py 2024-01-01 2024-12-31
"""

import sys
import os

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from app.db.config.base import get_db
from app.models.post import Post
from app.models.company import Company
from app.models.position import Position
from app.models.industry import Industry
from app.config.company_groups import COMPANY_GROUPS
# SQLAlchemy relationship 초기화를 위한 모델 import
from app.models.skill import Skill
from app.models.post_skill import PostSkill
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill
from datetime import datetime
import json


# ==================== HHI 계산 ====================

def calculate_hhi(shares: list) -> float:
    """
    HHI (Herfindahl-Hirschman Index) 계산

    Args:
        shares: 각 카테고리의 점유율 리스트 (0~1)

    Returns:
        HHI 값 (0~1)

    Formula:
        HHI = Σ(si²)
        where si = 각 카테고리의 점유율

    Example:
        >>> calculate_hhi([0.4, 0.3, 0.2, 0.1])
        0.30  # 0.4² + 0.3² + 0.2² + 0.1² = 0.16 + 0.09 + 0.04 + 0.01
    """
    if not shares:
        return 0.0

    # HHI = 점유율 제곱의 합
    hhi = sum(s ** 2 for s in shares)

    return round(hhi, 4)


def calculate_concentration_ratio(shares: list, k: int = 2) -> float:
    """
    Concentration Ratio 계산 (상위 k개 직무 점유율 합)

    채용 시장에서 상위 k개 직무가 차지하는 비중으로,
    특정 직무에 수요가 얼마나 쏠려 있는지 측정합니다.

    Args:
        shares: 점유율 리스트 (0~1)
        k: 상위 k개 항목 (기본값: 2)

    Returns:
        CR 값 (0~1)

    Formula:
        CRₖ = Σ(i=1 to k) sᵢ
        where sᵢ = 점유율이 높은 순서대로 정렬된 i번째 항목의 점유율

    출처:
        https://en.wikipedia.org/wiki/Concentration_ratio

    Example:
        >>> calculate_concentration_ratio([0.45, 0.25, 0.15, 0.10, 0.05], k=2)
        0.70  # 상위 2개 직무가 70% 차지
    """
    if not shares or len(shares) == 0:
        return 0.0

    # 내림차순 정렬
    sorted_shares = sorted(shares, reverse=True)

    # 상위 k개의 합
    cr = sum(sorted_shares[:min(k, len(sorted_shares))])

    return round(cr, 4)


def calculate_entropy(shares: list) -> dict:
    """
    Shannon Entropy 계산 (직무 다양성 지수)

    직군 내 직무 포트폴리오가 얼마나 다양하고 복잡한지 측정합니다.
    값이 높을수록 직무가 고르게 분포되어 있음을 의미합니다.

    Args:
        shares: 점유율 리스트 (0~1)

    Returns:
        {
            "entropy": float,              # 원본 엔트로피
            "normalized_entropy": float,   # 정규화된 엔트로피 (0~1)
            "max_entropy": float           # 최대 엔트로피 (log(N))
        }

    Formula:
        H = -Σ(pᵢ × log(pᵢ))
        H_norm = H / log(N)
        where pᵢ = 각 항목의 점유율, N = 항목 개수

    Example:
        >>> calculate_entropy([0.25, 0.25, 0.25, 0.25])
        {'entropy': 1.386, 'normalized_entropy': 1.0, 'max_entropy': 1.386}
        # 완전히 균등한 분포 → 최대 다양성
    """
    import math

    if not shares or len(shares) == 0:
        return {
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "max_entropy": 0.0
        }

    # 0이 아닌 점유율만 필터링
    non_zero_shares = [s for s in shares if s > 0]
    n = len(non_zero_shares)

    if n == 0:
        return {
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "max_entropy": 0.0
        }

    # Entropy 계산: H = -Σ(pᵢ × log(pᵢ))
    entropy = -sum(p * math.log(p) for p in non_zero_shares)

    # 최대 엔트로피 (모든 항목이 동일한 비율일 때)
    max_entropy = math.log(n)

    # 정규화된 엔트로피 (0~1 범위)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "entropy": round(entropy, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "max_entropy": round(max_entropy, 4)
    }


def interpret_hhi(hhi: float) -> dict:
    """
    HHI 값 해석

    Args:
        hhi: HHI 값 (0~1)

    Returns:
        {
            "level": str,          # "분산", "부분집중", "쏠림"
            "difficulty": str,     # "낮음", "중간", "높음"
            "description": str     # 상세 설명
        }
    """
    if hhi < 0.15:
        return {
            "level": "분산",
            "difficulty": "낮음",
            "description": "다양한 직무가 고르게 채용되어 경쟁이 분산되어 있습니다."
        }
    elif hhi < 0.25:
        return {
            "level": "부분집중",
            "difficulty": "중간",
            "description": "특정 직무의 채용이 증가하여 경쟁이 부분적으로 집중되고 있습니다."
        }
    else:
        return {
            "level": "쏠림",
            "difficulty": "높음",
            "description": "특정 직무로 수요가 쏠려 인재 쟁탈전이 발생하고 있습니다."
        }


def interpret_concentration_ratio(cr2: float) -> str:
    """
    CR₂ (상위 2개 직무 점유율 합) 해석

    Args:
        cr2: CR₂ 값 (0~1)

    Returns:
        현업 담당자용 해석 텍스트
    """
    cr2_percent = cr2 * 100

    if cr2 >= 0.70:
        return f"상위 2개 직무가 시장의 {cr2_percent:.1f}%를 주도하고 있어, 핵심 인재 확보 전략이 필요합니다."
    elif cr2 >= 0.50:
        return f"상위 2개 직무가 {cr2_percent:.1f}%를 차지하며 쏠림이 있어, 채용 타이밍과 브랜딩이 중요합니다."
    else:
        return f"상위 2개 직무가 {cr2_percent:.1f}%로 수요가 다양한 직무에 분산되어 있어, 다양한 직무 채용이 가능합니다."


def interpret_entropy(normalized_entropy: float) -> str:
    """
    정규화된 Entropy (직무 다양성 지수) 해석

    Args:
        normalized_entropy: 정규화된 엔트로피 (0~1)

    Returns:
        현업 담당자용 해석 텍스트
    """
    if normalized_entropy >= 0.80:
        return "직무 구성이 매우 다양하여, 직무별 세분화된 채용 전략이 필요합니다."
    elif normalized_entropy >= 0.60:
        return "일부 주력 직무와 기타 직무가 균형을 이루고 있습니다."
    else:
        return "소수 직무 중심으로 집중되어 있어, 단일 집중형 채용 전략이 효과적입니다."


# ==================== 경쟁사 필터링 ====================

def get_competitor_conditions():
    """경쟁사 필터링 조건 생성"""
    like_conditions = []
    for patterns in COMPANY_GROUPS.values():
        for pattern in patterns:
            like_conditions.append(Company.name.like(pattern))
    return like_conditions


# ==================== 데이터 조회 ====================

def get_position_distribution(db: Session, start_date: str, end_date: str):
    """
    직군별 공고 수 분포 조회

    Returns:
        List of (position_id, position_name, posts_count)
    """
    competitor_conditions = get_competitor_conditions()

    results = db.query(
        Position.id,
        Position.name,
        func.count(Post.id).label('posts_count')
    )\
        .join(Industry, Industry.position_id == Position.id)\
        .join(Post, Post.industry_id == Industry.id)\
        .join(Company, Post.company_id == Company.id)\
        .filter(
            or_(*competitor_conditions),
            Post.posted_at >= start_date,
            Post.posted_at <= end_date
        )\
        .group_by(Position.id, Position.name)\
        .order_by(func.count(Post.id).desc())\
        .all()

    return results


def get_industry_distribution(db: Session, start_date: str, end_date: str):
    """
    직무별 공고 수 분포 조회

    Returns:
        List of (industry_id, industry_name, position_id, position_name, posts_count)
    """
    competitor_conditions = get_competitor_conditions()

    results = db.query(
        Industry.id,
        Industry.name,
        Position.id.label('position_id'),
        Position.name.label('position_name'),
        func.count(Post.id).label('posts_count')
    )\
        .join(Post, Post.industry_id == Industry.id)\
        .join(Position, Industry.position_id == Position.id)\
        .join(Company, Post.company_id == Company.id)\
        .filter(
            or_(*competitor_conditions),
            Post.posted_at >= start_date,
            Post.posted_at <= end_date
        )\
        .group_by(Industry.id, Industry.name, Position.id, Position.name)\
        .order_by(func.count(Post.id).desc())\
        .all()

    return results


# ==================== HHI 분석 ====================

def analyze_overall_hhi(db: Session, start_date: str, end_date: str):
    """
    전체 HHI 집중도 분석

    Note:
        전체 레벨에서는 직군(Position) 기준으로 HHI, CR₂, Entropy 계산

    Returns:
        {
            "period": {"start": str, "end": str},
            "total_posts": int,
            "hhi": float,
            "interpretation": {...},
            "cr2": float,
            "cr2_interpretation": str,
            "entropy": {...},
            "entropy_interpretation": str,
            "top_categories": [...],
            "insights": [str]
        }
    """
    # 직군별 분포 조회
    position_data = get_position_distribution(db, start_date, end_date)

    if not position_data:
        return {
            "period": {"start": start_date, "end": end_date},
            "total_posts": 0,
            "hhi": 0.0,
            "interpretation": interpret_hhi(0.0),
            "top_categories": [],
            "insights": ["데이터가 없습니다."]
        }

    # 총 공고 수
    total_posts = sum(count for _, _, count in position_data)

    # 각 직군의 점유율 계산
    shares = [count / total_posts for _, _, count in position_data]

    # HHI 계산
    hhi = calculate_hhi(shares)

    # CR₂ 계산 (상위 2개 직무 점유율 합)
    cr2 = calculate_concentration_ratio(shares, k=2)

    # Entropy 계산 (직무 다양성 지수)
    entropy_result = calculate_entropy(shares)

    # 해석
    interpretation = interpret_hhi(hhi)
    cr2_interpretation = interpret_concentration_ratio(cr2)
    entropy_interpretation = interpret_entropy(entropy_result['normalized_entropy'])

    # TOP 카테고리 (상위 5개)
    top_categories = [
        {
            "position_id": pos_id,
            "position_name": pos_name,
            "posts_count": count,
            "share": round(count / total_posts * 100, 2)
        }
        for pos_id, pos_name, count in position_data[:5]
    ]

    # 인사이트 생성 (전체 position_data 전달)
    insights = generate_overall_insights(hhi, interpretation, top_categories, total_posts, db, position_data)

    return {
        "period": {"start": start_date, "end": end_date},
        "total_posts": total_posts,
        "hhi": hhi,
        "interpretation": interpretation,
        "cr2": cr2,
        "cr2_interpretation": cr2_interpretation,
        "entropy": entropy_result,
        "entropy_interpretation": entropy_interpretation,
        "top_categories": top_categories,
        "insights": insights
    }


def analyze_position_hhi(db: Session, start_date: str, end_date: str):
    """
    직군별 HHI 집중도 분석

    Note:
        각 직군 내에서 산업(Industry) 기준으로 HHI, CR₂, Entropy 계산

    Returns:
        List of {
            "position_id": int,
            "position_name": str,
            "total_posts": int,
            "hhi": float,
            "interpretation": {...},
            "cr2": float,
            "cr2_interpretation": str,
            "entropy": {...},
            "entropy_interpretation": str,
            "top_industries": [...],
            "insights": [str]
        }
    """
    # 직무별 분포 조회
    industry_data = get_industry_distribution(db, start_date, end_date)

    if not industry_data:
        return []

    # 직군별로 그룹화
    from collections import defaultdict
    position_groups = defaultdict(list)

    for ind_id, ind_name, pos_id, pos_name, count in industry_data:
        position_groups[pos_id].append({
            "position_id": pos_id,
            "position_name": pos_name,
            "industry_id": ind_id,
            "industry_name": ind_name,
            "posts_count": count
        })

    # 각 직군별 HHI 계산
    results = []

    for pos_id, industries in position_groups.items():
        # 해당 직군의 총 공고 수
        total_posts = sum(ind["posts_count"] for ind in industries)

        # 각 직무의 점유율
        shares = [ind["posts_count"] / total_posts for ind in industries]

        # HHI 계산
        hhi = calculate_hhi(shares)

        # CR₂ 계산 (상위 2개 직무 점유율 합)
        cr2 = calculate_concentration_ratio(shares, k=2)

        # Entropy 계산 (직무 다양성 지수)
        entropy_result = calculate_entropy(shares)

        # 해석
        interpretation = interpret_hhi(hhi)
        cr2_interpretation = interpret_concentration_ratio(cr2)
        entropy_interpretation = interpret_entropy(entropy_result['normalized_entropy'])

        # TOP 직무 (상위 3개)
        top_industries = [
            {
                "industry_id": ind["industry_id"],
                "industry_name": ind["industry_name"],
                "posts_count": ind["posts_count"],
                "share": round(ind["posts_count"] / total_posts * 100, 2)
            }
            for ind in sorted(industries, key=lambda x: x["posts_count"], reverse=True)[:3]
        ]

        # 인사이트 생성 (전체 industries 전달)
        insights = generate_position_insights(
            industries[0]["position_name"],
            hhi,
            interpretation,
            top_industries,
            total_posts,
            db,
            industries
        )

        results.append({
            "position_id": pos_id,
            "position_name": industries[0]["position_name"],
            "total_posts": total_posts,
            "hhi": hhi,
            "interpretation": interpretation,
            "cr2": cr2,
            "cr2_interpretation": cr2_interpretation,
            "entropy": entropy_result,
            "entropy_interpretation": entropy_interpretation,
            "top_industries": top_industries,
            "insights": insights
        })

    # HHI 순으로 정렬 (쏠림이 심한 순서)
    results.sort(key=lambda x: x["hhi"], reverse=True)

    return results


def analyze_industry_hhi(db: Session, start_date: str, end_date: str):
    """
    직무별 집중도 분석 (단순 점유율)

    Note:
        직무는 최하위 레벨이므로 HHI 대신 점유율만 계산

    Returns:
        List of {
            "industry_id": int,
            "industry_name": str,
            "position_id": int,
            "position_name": str,
            "posts_count": int,
            "share": float
        }
    """
    # 직무별 분포 조회
    industry_data = get_industry_distribution(db, start_date, end_date)

    if not industry_data:
        return []

    # 총 공고 수
    total_posts = sum(count for _, _, _, _, count in industry_data)

    # 결과 생성
    results = [
        {
            "industry_id": ind_id,
            "industry_name": ind_name,
            "position_id": pos_id,
            "position_name": pos_name,
            "posts_count": count,
            "share": round(count / total_posts * 100, 2)
        }
        for ind_id, ind_name, pos_id, pos_name, count in industry_data
    ]

    return results


# ==================== 스킬 기반 유사도 분석 ====================

def get_similar_positions_by_skill(db: Session, target_position_id: int, limit: int = 3):
    """
    스킬 기반 유사 직군 조회

    Args:
        db: DB 세션
        target_position_id: 타겟 직군 ID
        limit: 반환할 최대 개수

    Returns:
        List of (position_id, position_name, common_skill_count)
    """
    from sqlalchemy import select

    # 1. 타겟 position의 스킬 목록 서브쿼리
    target_skills_subquery = (
        select(PositionSkill.skill_id)
        .filter(PositionSkill.position_id == target_position_id)
        .scalar_subquery()
    )

    # 2. 다른 position들의 공통 스킬 개수 계산
    query = (
        db.query(
            Position.id.label("position_id"),
            Position.name.label("position_name"),
            func.count(PositionSkill.skill_id).label("common_skills"),
        )
        .join(PositionSkill, Position.id == PositionSkill.position_id)
        .filter(
            PositionSkill.skill_id.in_(target_skills_subquery),
            Position.id != target_position_id,
        )
        .group_by(Position.id, Position.name)
        .order_by(func.count(PositionSkill.skill_id).desc())
        .limit(limit)
    )

    return query.all()


def get_similar_industries_by_skill(db: Session, target_industry_id: int, limit: int = 3):
    """
    스킬 기반 유사 직무 조회

    Args:
        db: DB 세션
        target_industry_id: 타겟 직무 ID
        limit: 반환할 최대 개수

    Returns:
        List of (industry_id, industry_name, common_skill_count)
    """
    from sqlalchemy import select

    # 1. 타겟 industry의 스킬 목록 서브쿼리
    target_skills_subquery = (
        select(IndustrySkill.skill_id)
        .filter(IndustrySkill.industry_id == target_industry_id)
        .scalar_subquery()
    )

    # 2. 다른 industry들의 공통 스킬 개수 계산
    query = (
        db.query(
            Industry.id.label("industry_id"),
            Industry.name.label("industry_name"),
            func.count(IndustrySkill.skill_id).label("common_skills"),
        )
        .join(IndustrySkill, Industry.id == IndustrySkill.industry_id)
        .filter(
            IndustrySkill.skill_id.in_(target_skills_subquery),
            Industry.id != target_industry_id,
        )
        .group_by(Industry.id, Industry.name)
        .order_by(func.count(IndustrySkill.skill_id).desc())
        .limit(limit)
    )

    return query.all()


# ==================== 인사이트 생성 ====================

def generate_overall_insights(hhi: float, interpretation: dict, top_categories: list, total_posts: int, db: Session = None, position_data: list = None) -> list:
    """전체 레벨 인사이트 생성 (채용담당자 관점)"""
    insights = []

    # 1. 시장 경쟁 현황 한눈에 파악
    if top_categories:
        top1 = top_categories[0]
        top_share = top1['share']

        # 채용 난이도에 따른 인사이트 문구
        if top_share > 40:
            insights.append(
                f"현재 시장에서 '{top1['position_name']}' 직군이 {top_share:.1f}%로 가장 높은 수요를 보입니다. "
                f"채용 난이도가 매우 높아 극심한 경쟁이 예상됩니다."
            )
        elif top_share > 15:
            insights.append(
                f"현재 시장에서 '{top1['position_name']}' 직군이 {top_share:.1f}%로 가장 높은 수요를 보입니다. "
                f"채용 난이도는 보통 수준으로, 적절한 채용 전략이 필요합니다."
            )
        else:
            insights.append(
                f"현재 시장에서 '{top1['position_name']}' 직군이 {top_share:.1f}%로 가장 높은 수요를 보입니다. "
                f"채용 난이도가 낮아 인재 확보가 용이한 상황입니다."
            )

        # 2. 대안 직군 제시 (스킬 기반, 경쟁이 심한 경우만)
        if top_share > 40 and db:
            similar_positions = get_similar_positions_by_skill(db, top1['position_id'], limit=3)

            if similar_positions:
                alternatives = []
                # 전체 position_data에서 점유율 계산
                position_dict = {pos_id: (pos_name, count) for pos_id, pos_name, count in (position_data or [])}

                for alt_id, alt_name, common_skills in similar_positions:
                    # 전체 데이터에서 해당 position의 점유율 찾기
                    if alt_id in position_dict:
                        _, alt_count = position_dict[alt_id]
                        alt_share = (alt_count / total_posts * 100) if total_posts > 0 else 0
                        alternatives.append(f"{alt_name} - 점유율 {alt_share:.1f}%, 유사 스킬 {common_skills}개")

                if alternatives:
                    insights.append(
                        f"'{top1['position_name']}' 직군은 경쟁이 매우 심합니다. "
                        f"비슷한 스킬을 요구하지만 경쟁이 덜한 대안: {', '.join(alternatives[:3])}"
                    )
                else:
                    insights.append(
                        f"'{top1['position_name']}' 직군은 경쟁이 매우 심합니다. "
                        f"유사한 스킬을 가진 대안 직군이 없습니다."
                    )
            else:
                insights.append(
                    f"'{top1['position_name']}' 직군은 경쟁이 매우 심합니다. "
                    f"유사한 스킬을 가진 대안 직군이 없습니다."
                )

    # 3. 시장 트렌드 요약
    if interpretation['difficulty'] == '높음':
        insights.append(
            f"전반적으로 쏠림 현상 심화 중 (HHI {hhi:.2f}). "
            f"인기 직군은 더 어려워지고, 틈새 직군은 기회입니다."
        )
    elif interpretation['difficulty'] == '중간':
        insights.append(
            f"일부 직군에 수요가 집중되는 경향 (HHI {hhi:.2f}). "
            f"해당 분야의 경쟁이 점차 심화될 수 있습니다."
        )
    else:
        insights.append(
            f"다양한 직군에서 고르게 채용 진행 중 (HHI {hhi:.2f}). "
            f"전반적인 인재 수급 경쟁은 완화된 상태입니다."
        )

    return insights


def generate_position_insights(position_name: str, hhi: float, interpretation: dict,
                                 top_industries: list, total_posts: int, db: Session = None, industry_data: list = None) -> list:
    """직군별 인사이트 생성 (채용담당자 관점)"""
    insights = []

    # 1. 시장 경쟁 현황 한눈에 파악
    if top_industries:
        top1 = top_industries[0]
        top_share = top1['share']

        # 채용 난이도에 따른 인사이트 문구
        if top_share > 40:
            insights.append(
                f"'{position_name}' 직군 내에서 '{top1['industry_name']}' 직무가 {top_share:.1f}%로 가장 높은 수요를 보입니다. "
                f"채용 난이도가 매우 높아 극심한 경쟁이 예상됩니다."
            )
        elif top_share > 15:
            insights.append(
                f"'{position_name}' 직군 내에서 '{top1['industry_name']}' 직무가 {top_share:.1f}%로 가장 높은 수요를 보입니다. "
                f"채용 난이도는 보통 수준으로, 계획적인 채용 접근이 권장됩니다."
            )
        else:
            insights.append(
                f"'{position_name}' 직군 내에서 '{top1['industry_name']}' 직무가 {top_share:.1f}%로 가장 높은 수요를 보입니다. "
                f"채용 난이도가 낮아 해당 직무의 인재 확보가 용이합니다."
            )

        # 2. 대안 직무 제시 (스킬 기반, 경쟁이 심한 경우만)
        if top_share > 40 and db:
            similar_industries = get_similar_industries_by_skill(db, top1['industry_id'], limit=3)

            if similar_industries:
                alternatives = []
                # 전체 industry_data에서 점유율 계산
                industry_dict = {ind["industry_id"]: (ind["industry_name"], ind["posts_count"]) for ind in (industry_data or [])}

                for alt_id, alt_name, common_skills in similar_industries:
                    # 전체 데이터에서 해당 industry의 점유율 찾기
                    if alt_id in industry_dict:
                        _, alt_count = industry_dict[alt_id]
                        alt_share = (alt_count / total_posts * 100) if total_posts > 0 else 0
                        alternatives.append(f"{alt_name} - 점유율 {alt_share:.1f}%, 유사 스킬 {common_skills}개")

                if alternatives:
                    insights.append(
                        f"'{top1['industry_name']}' 직무는 경쟁이 매우 심합니다. "
                        f"비슷한 스킬을 요구하지만 경쟁이 덜한 대안: {', '.join(alternatives[:3])}"
                    )
                else:
                    insights.append(
                        f"'{top1['industry_name']}' 직무는 경쟁이 매우 심합니다. "
                        f"유사한 스킬을 가진 대안 직무가 없습니다."
                    )
            else:
                insights.append(
                    f"'{top1['industry_name']}' 직무는 경쟁이 매우 심합니다. "
                    f"유사한 스킬을 가진 대안 직무가 없습니다."
                )

    # 3. 시장 트렌드 요약
    if interpretation['difficulty'] == '높음':
        insights.append(
            f"'{position_name}' 내에서 특정 직무로 수요가 쏠려 있어 해당 직무의 인재 확보가 어려울 수 있습니다."
        )
    elif interpretation['difficulty'] == '중간':
        insights.append(
            f"'{position_name}' 내 일부 직무에 수요가 집중되는 경향을 보입니다."
        )
    else:
        insights.append(
            f"'{position_name}' 내 다양한 직무에서 고르게 채용이 이루어지고 있습니다."
        )

    return insights


# ==================== 출력 ====================

def format_json_output(overall_result: dict, position_results: list, industry_results: list) -> dict:
    """
    대시보드 인사이트 창용 JSON 형식으로 변환

    Returns:
        {
            "status": 200,
            "code": "SUCCESS",
            "message": "HHI 수요 집중도 분석 완료",
            "data": {
                "overall": {...},
                "by_position": [...],
                "by_industry": [...]
            }
        }
    """
    return {
        "status": 200,
        "code": "SUCCESS",
        "message": "HHI 수요 집중도 분석 완료",
        "data": {
            "overall": overall_result,
            "by_position": position_results,
            "by_industry": industry_results
        }
    }


def print_text_insights(overall_result: dict, position_results: list):
    """텍스트 형식으로 인사이트 출력"""
    print(f"\n{'='*80}")
    print(" HHI 수요 집중도 분석 인사이트")
    print(f"{'='*80}")

    # 1. 전체 인사이트
    print(f"\n{'─'*80}")
    print(" 전체 시장 분석")
    print(f"{'─'*80}")
    print(f"기간: {overall_result['period']['start']} ~ {overall_result['period']['end']}")
    print(f"총 공고 수: {overall_result['total_posts']}개")
    print(f"\n[인사이트]")
    for insight in overall_result['insights']:
        print(f"  - {insight}")

    # 2. 직군별 인사이트 (상위 5개만)
    print(f"\n{'─'*80}")
    print(" 직군별 집중도 분석 (TOP 5)")
    print(f"{'─'*80}")

    for i, result in enumerate(position_results[:5], 1):
        print(f"\n{i}. {result['position_name']}")
        print(f"   총 공고: {result['total_posts']}개")
        print(f"   [인사이트]")
        for insight in result['insights']:
            print(f"     - {insight}")

    print(f"\n{'='*80}")


# ==================== 메인 ====================

def main():
    """메인 실행 함수"""
    import argparse
    from datetime import datetime, timedelta

    print("\n" + "=" * 80)
    print("HHI (Herfindahl-Hirschman Index) 수요 집중도 분석")
    print("=" * 80)

    # 인자 파싱
    parser = argparse.ArgumentParser(description='HHI 집중도 분석')
    parser.add_argument('start_date', nargs='?', default='2025-07-01',
                        help='시작일 (YYYY-MM-DD), 기본값: 2025-07-01')
    parser.add_argument('--position', type=int, default=None,
                        help='특정 직군 ID (미지정 시 전체 직군 분석)')
    parser.add_argument('--industry', type=int, default=None,
                        help='특정 산업 ID (position과 함께 사용)')

    args = parser.parse_args()
    start_date = args.start_date
    position_id = args.position
    industry_id = args.industry

    # end_date는 start_date + 3개월로 자동 계산
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=90)  # 약 3개월
    end_date = end_dt.strftime("%Y-%m-%d")

    # 분석 모드 결정
    if position_id is None and industry_id is None:
        analysis_mode = "전체 시장"
    elif position_id is not None and industry_id is None:
        analysis_mode = f"직군 ID {position_id}"
    elif position_id is not None and industry_id is not None:
        analysis_mode = f"직군 ID {position_id} > 산업 ID {industry_id}"
    else:
        print("오류: industry_id는 position_id와 함께 사용해야 합니다.")
        return

    print(f"\n[설정]")
    print(f"  분석 기간: {start_date} ~ {end_date} (3개월)")
    print(f"  분석 모드: {analysis_mode}")
    print(f"  분석 대상: 경쟁사 (9개 그룹)")
    print(f"  계산 방식: HHI = Σ(si²)")
    print(f"\n[HHI 해석 기준]")
    print(f"  0.00 ~ 0.15: 분산 (경쟁 낮음)")
    print(f"  0.15 ~ 0.25: 부분집중 (경쟁 중간)")
    print(f"  0.25 이상:   쏠림 (경쟁 높음)")

    db = next(get_db())

    try:
        if position_id is None and industry_id is None:
            # ========== 시나리오 1: 전체 시장 분석 ==========
            print(f"\n{'-'*80}")
            print("[시나리오 1] 전체 시장 분석 중...")
            print(f"{'-'*80}")
            overall_result = analyze_overall_hhi(db, start_date, end_date)
            print(f"완료 - 전체 HHI: {overall_result['hhi']:.4f}")
            print(f"완료 - 총 공고 수: {overall_result['total_posts']}개")

            # 텍스트 인사이트 출력
            print(f"\n{'='*80}")
            print(" 전체 시장 분석 결과")
            print(f"{'='*80}")
            print(f"기간: {overall_result['period']['start']} ~ {overall_result['period']['end']}")
            print(f"총 공고 수: {overall_result['total_posts']}개")
            print(f"\n[인사이트]")
            for insight in overall_result['insights']:
                print(f"  - {insight}")

            # JSON 출력
            print(f"\n{'='*80}")
            print(" JSON 출력")
            print(f"{'='*80}")
            print(json.dumps(overall_result, ensure_ascii=False, indent=2))

        elif position_id is not None and industry_id is None:
            # ========== 시나리오 2: 특정 직군 내 산업 분석 ==========
            print(f"\n{'-'*80}")
            print(f"[시나리오 2] 직군 ID {position_id} 내 산업 분석 중...")
            print(f"{'-'*80}")

            # 전체 직군 분석 실행 후 해당 직군만 필터링
            position_results = analyze_position_hhi(db, start_date, end_date)

            # 해당 직군 찾기
            target_result = None
            for result in position_results:
                if result['position_id'] == position_id:
                    target_result = result
                    break

            if target_result is None:
                print(f"오류: 직군 ID {position_id}를 찾을 수 없습니다.")
                return

            print(f"완료 - 직군: {target_result['position_name']}")
            print(f"완료 - HHI: {target_result['hhi']:.4f}")
            print(f"완료 - 총 공고 수: {target_result['total_posts']}개")

            # 텍스트 인사이트 출력
            print(f"\n{'='*80}")
            print(f" 직군 분석 결과: {target_result['position_name']}")
            print(f"{'='*80}")
            print(f"총 공고: {target_result['total_posts']}개")
            print(f"\n[인사이트]")
            for insight in target_result['insights']:
                print(f"  - {insight}")

            # JSON 출력
            print(f"\n{'='*80}")
            print(" JSON 출력")
            print(f"{'='*80}")
            print(json.dumps(target_result, ensure_ascii=False, indent=2))

        elif position_id is not None and industry_id is not None:
            # ========== 시나리오 3: 특정 산업 분석 ==========
            print(f"\n{'-'*80}")
            print(f"[시나리오 3] 직군 ID {position_id} > 산업 ID {industry_id} 분석 중...")
            print(f"{'-'*80}")

            # 산업별 분포 조회
            industry_results = get_industry_distribution(db, start_date, end_date)

            # 해당 산업 찾기
            target_industry = None
            for ind_id, ind_name, pos_id, pos_name, count in industry_results:
                if ind_id == industry_id and pos_id == position_id:
                    target_industry = {
                        "industry_id": ind_id,
                        "industry_name": ind_name,
                        "position_id": pos_id,
                        "position_name": pos_name,
                        "posts_count": count
                    }
                    break

            if target_industry is None:
                print(f"오류: 직군 ID {position_id} 내에 산업 ID {industry_id}를 찾을 수 없습니다.")
                return

            # 해당 직군 내 전체 공고 수 및 산업 목록
            same_position_industries = [
                {"industry_id": ind_id, "industry_name": ind_name, "posts_count": count}
                for ind_id, ind_name, pos_id, pos_name, count in industry_results
                if pos_id == position_id
            ]
            same_position_industries.sort(key=lambda x: x["posts_count"], reverse=True)

            total_posts_in_position = sum(ind["posts_count"] for ind in same_position_industries)
            target_share = (target_industry["posts_count"] / total_posts_in_position * 100) if total_posts_in_position > 0 else 0
            target_rank = next((i + 1 for i, ind in enumerate(same_position_industries) if ind["industry_id"] == industry_id), None)

            print(f"완료 - 산업: {target_industry['industry_name']}")
            print(f"완료 - 소속 직군: {target_industry['position_name']}")
            print(f"완료 - 공고 수: {target_industry['posts_count']}개")
            print(f"완료 - 직군 내 순위: {target_rank}위 / {len(same_position_industries)}개")
            print(f"완료 - 직군 내 점유율: {target_share:.1f}%")

            # 인사이트 생성
            insights = []
            insights.append(
                f"'{target_industry['position_name']}' 직군 내에서 '{target_industry['industry_name']}' 산업은 "
                f"{target_rank}위를 차지하고 있으며, {target_share:.1f}%의 점유율을 보입니다."
            )

            if target_share > 40:
                insights.append(
                    f"'{target_industry['industry_name']}' 산업은 직군 내 최상위 수요로, 채용 난이도가 매우 높습니다. "
                    f"극심한 경쟁이 예상되므로 신중한 채용 전략이 필요합니다."
                )
            elif target_share > 15:
                insights.append(
                    f"'{target_industry['industry_name']}' 산업은 중간 수준의 수요를 보이고 있습니다. "
                    f"계획적인 채용 접근이 권장됩니다."
                )
            else:
                insights.append(
                    f"'{target_industry['industry_name']}' 산업은 상대적으로 낮은 수요를 보이고 있어, "
                    f"인재 확보가 비교적 용이할 것으로 예상됩니다."
                )

            # 유사 스킬 기반 대안 산업 찾기
            similar_industries = get_similar_industries_by_skill(db, industry_id, limit=3)
            if similar_industries:
                alternatives = []
                industry_dict = {ind["industry_id"]: (ind["industry_name"], ind["posts_count"]) for ind in same_position_industries}

                for alt_id, alt_name, common_skills in similar_industries:
                    if alt_id in industry_dict:
                        _, alt_count = industry_dict[alt_id]
                        alt_share = (alt_count / total_posts_in_position * 100) if total_posts_in_position > 0 else 0
                        alternatives.append(f"{alt_name} - 점유율 {alt_share:.1f}%, 유사 스킬 {common_skills}개")

                if alternatives:
                    insights.append(
                        f"유사한 스킬을 가진 대안 산업: {', '.join(alternatives[:3])}"
                    )

            # 텍스트 인사이트 출력
            print(f"\n{'='*80}")
            print(f" 산업 분석 결과: {target_industry['industry_name']}")
            print(f"{'='*80}")
            print(f"소속 직군: {target_industry['position_name']}")
            print(f"공고 수: {target_industry['posts_count']}개")
            print(f"직군 내 순위: {target_rank}위 / {len(same_position_industries)}개")
            print(f"직군 내 점유율: {target_share:.1f}%")
            print(f"\n[인사이트]")
            for insight in insights:
                print(f"  - {insight}")

            # JSON 출력
            result_data = {
                "industry_id": target_industry["industry_id"],
                "industry_name": target_industry["industry_name"],
                "position_id": target_industry["position_id"],
                "position_name": target_industry["position_name"],
                "posts_count": target_industry["posts_count"],
                "rank_in_position": target_rank,
                "total_industries_in_position": len(same_position_industries),
                "share_in_position": round(target_share, 2),
                "insights": insights
            }
            print(f"\n{'='*80}")
            print(" JSON 출력")
            print(f"{'='*80}")
            print(json.dumps(result_data, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"\n오류: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    main()
