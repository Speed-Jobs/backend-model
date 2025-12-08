"""
채용 난이도 지수 분석

이 스크립트는 직무별 인재 수급 난이도를 0-100 점수로 계산합니다.

주요 기능:
1. 전체 난이도 지수 계산
2. 직군(Position)별 난이도 지수 계산
3. 직무(Industry)별 난이도 지수 계산
4. 2024년(baseline) 대비 2025년(current) 비교
5. Min-Max 스케일링을 통한 0-100 점수 산출

계산 공식:
    score = (current - baseline_min) / (baseline_max - baseline_min) * 100
    - 100 초과시 100으로 cap
    - 0 미만시 0으로 cap

사용법:
    python test_05_difficulty.py
"""

import sys
import os

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sqlalchemy.orm import Session
from sqlalchemy import func, or_, extract
from app.db.config.base import get_db
from app.models.post import Post
from app.models.company import Company
from app.models.position import Position
from app.models.industry import Industry
from app.models.skill import Skill
# SQLAlchemy relationship 초기화를 위한 모델 import
from app.models.post_skill import PostSkill
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill
from app.config.company_groups import COMPANY_GROUPS


# ==================== 난이도 계산 로직 ====================

def calculate_difficulty_score(current: int, baseline_min: int, baseline_max: int) -> int:
    """
    Min-Max 스케일링을 사용한 난이도 점수 계산

    Args:
        current: 현재(2025년) 공고 수
        baseline_min: 작년(2024년) 최소 공고 수
        baseline_max: 작년(2024년) 최대 공고 수

    Returns:
        0-100 범위의 난이도 점수
    """
    # 기준이 같으면 중간값
    if baseline_max == baseline_min:
        return 50

    # Min-Max 스케일링
    score = (current - baseline_min) / (baseline_max - baseline_min) * 100

    # 0-100 범위로 cap
    if score > 100:
        return 100
    elif score < 0:
        return 0
    else:
        return int(round(score))


# ==================== 데이터 조회 ====================

def get_competitor_conditions():
    """경쟁사 필터링 조건 생성"""
    like_conditions = []
    for patterns in COMPANY_GROUPS.values():
        for pattern in patterns:
            like_conditions.append(Company.name.like(pattern))
    return like_conditions


def get_posts_count_by_year(
    db: Session,
    year: int,
    position_id: int = None,
    industry_id: int = None
) -> int:
    """
    특정 연도의 경쟁사 공고 수 조회

    Args:
        db: Database session
        year: 연도 (2024, 2025)
        position_id: 직군 ID (optional)
        industry_id: 직무 ID (optional)

    Returns:
        공고 수
    """
    # 경쟁사 조건
    competitor_conditions = get_competitor_conditions()

    # 기본 쿼리
    query = db.query(func.count(Post.id))\
        .join(Company, Post.company_id == Company.id)\
        .filter(
            or_(*competitor_conditions),
            extract('year', Post.created_at) == year
        )

    # 직군 필터
    if position_id is not None:
        query = query.filter(Post.position_id == position_id)

    # 직무 필터
    if industry_id is not None:
        query = query.filter(Post.industry_id == industry_id)

    return query.scalar() or 0


def get_all_positions_with_counts(db: Session, year: int):
    """
    모든 직군과 해당 연도의 공고 수 조회

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
            extract('year', Post.created_at) == year
        )\
        .group_by(Position.id, Position.name)\
        .order_by(Position.id)\
        .all()

    return results


def get_all_industries_with_counts(db: Session, year: int):
    """
    모든 직무와 해당 연도의 공고 수 조회

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
            extract('year', Post.created_at) == year
        )\
        .group_by(Industry.id, Industry.name, Position.id, Position.name)\
        .order_by(Industry.id)\
        .all()

    return results


# ==================== 난이도 분석 ====================

def analyze_overall_difficulty(db: Session):
    """
    전체 난이도 분석

    Returns:
        {
            "name": "전체",
            "baseline_year": 2024,
            "baseline_posts": int,
            "current_posts": int,
            "score": int
        }
    """
    # 2024년 전체 공고 수
    baseline_posts = get_posts_count_by_year(db, 2024)

    # 2025년 전체 공고 수
    current_posts = get_posts_count_by_year(db, 2025)

    # 난이도 계산 (단일 값이므로 baseline_min = baseline_max)
    # 증가율 기준으로 계산
    if baseline_posts == 0:
        score = 0
    else:
        growth_rate = (current_posts - baseline_posts) / baseline_posts * 100
        # 증가율을 0-100 점수로 변환 (100% 증가 = 100점)
        score = min(max(int(growth_rate), 0), 100)

    return {
        "name": "전체",
        "baseline_year": 2024,
        "baseline_posts": baseline_posts,
        "current_posts": current_posts,
        "score": score
    }


def analyze_position_difficulty(db: Session):
    """
    직군별 난이도 분석

    Returns:
        List of {
            "position_id": int,
            "name": str,
            "baseline_year": 2024,
            "baseline_posts": int,
            "current_posts": int,
            "score": int
        }
    """
    # 2024년 직군별 공고 수
    baseline_data = get_all_positions_with_counts(db, 2024)
    baseline_dict = {pos_id: count for pos_id, name, count in baseline_data}

    # 2025년 직군별 공고 수
    current_data = get_all_positions_with_counts(db, 2025)

    # baseline의 min/max 계산
    baseline_counts = list(baseline_dict.values())
    baseline_min = min(baseline_counts) if baseline_counts else 0
    baseline_max = max(baseline_counts) if baseline_counts else 0

    results = []
    for pos_id, pos_name, current_posts in current_data:
        baseline_posts = baseline_dict.get(pos_id, 0)

        # 난이도 점수 계산
        score = calculate_difficulty_score(current_posts, baseline_min, baseline_max)

        results.append({
            "position_id": pos_id,
            "name": pos_name,
            "baseline_year": 2024,
            "baseline_posts": baseline_posts,
            "current_posts": current_posts,
            "score": score
        })

    # 점수 순으로 정렬
    results.sort(key=lambda x: x["score"], reverse=True)

    return results


def analyze_industry_difficulty(db: Session):
    """
    직무별 난이도 분석

    Returns:
        List of {
            "industry_id": int,
            "name": str,
            "position_id": int,
            "position_name": str,
            "baseline_year": 2024,
            "baseline_posts": int,
            "current_posts": int,
            "score": int
        }
    """
    # 2024년 직무별 공고 수
    baseline_data = get_all_industries_with_counts(db, 2024)
    baseline_dict = {ind_id: count for ind_id, name, pos_id, pos_name, count in baseline_data}

    # 2025년 직무별 공고 수
    current_data = get_all_industries_with_counts(db, 2025)

    # baseline의 min/max 계산
    baseline_counts = list(baseline_dict.values())
    baseline_min = min(baseline_counts) if baseline_counts else 0
    baseline_max = max(baseline_counts) if baseline_counts else 0

    results = []
    for ind_id, ind_name, pos_id, pos_name, current_posts in current_data:
        baseline_posts = baseline_dict.get(ind_id, 0)

        # 난이도 점수 계산
        score = calculate_difficulty_score(current_posts, baseline_min, baseline_max)

        results.append({
            "industry_id": ind_id,
            "name": ind_name,
            "position_id": pos_id,
            "position_name": pos_name,
            "baseline_year": 2024,
            "baseline_posts": baseline_posts,
            "current_posts": current_posts,
            "score": score
        })

    # 점수 순으로 정렬
    results.sort(key=lambda x: x["score"], reverse=True)

    return results


# ==================== 출력 ====================

def print_overall_result(result):
    """전체 난이도 결과 출력"""
    print(f"\n{'='*60}")
    print(f"[전체 난이도 지수]")
    print(f"{'='*60}")
    print(f"  이름: {result['name']}")
    print(f"  기준년도: {result['baseline_year']}년")
    print(f"  2024년 공고 수: {result['baseline_posts']}개")
    print(f"  2025년 공고 수: {result['current_posts']}개")
    print(f"  난이도 점수: {result['score']}점")


def print_position_results(results):
    """직군별 난이도 결과 출력"""
    print(f"\n{'='*60}")
    print(f"[직군별 난이도 지수]")
    print(f"{'='*60}")
    print(f"\n{'직군명':<30} {'2024':<8} {'2025':<8} {'점수':<6}")
    print("-" * 60)

    for item in results:
        print(f"{item['name']:<30} {item['baseline_posts']:<8} {item['current_posts']:<8} {item['score']:<6}")

    print(f"\n총 {len(results)}개 직군")

    # TOP 5 난이도 높은 직군
    print(f"\n[TOP 5 난이도 높은 직군]")
    for i, item in enumerate(results[:5], 1):
        print(f"  {i}. {item['name']} - {item['score']}점 (2024: {item['baseline_posts']}개 → 2025: {item['current_posts']}개)")

    # TOP 5 난이도 낮은 직군
    print(f"\n[TOP 5 난이도 낮은 직군]")
    for i, item in enumerate(sorted(results, key=lambda x: x["score"])[:5], 1):
        print(f"  {i}. {item['name']} - {item['score']}점 (2024: {item['baseline_posts']}개 → 2025: {item['current_posts']}개)")


def print_industry_results(results):
    """직무별 난이도 결과 출력"""
    print(f"\n{'='*60}")
    print(f"[직무별 난이도 지수]")
    print(f"{'='*60}")
    print(f"\n{'직무명':<30} {'직군':<20} {'2024':<8} {'2025':<8} {'점수':<6}")
    print("-" * 80)

    for item in results:
        print(f"{item['name']:<30} {item['position_name']:<20} {item['baseline_posts']:<8} {item['current_posts']:<8} {item['score']:<6}")

    print(f"\n총 {len(results)}개 직무")

    # TOP 10 난이도 높은 직무
    print(f"\n[TOP 10 난이도 높은 직무]")
    for i, item in enumerate(results[:10], 1):
        print(f"  {i}. {item['name']} ({item['position_name']}) - {item['score']}점")
        print(f"      2024: {item['baseline_posts']}개 → 2025: {item['current_posts']}개")

    # TOP 10 난이도 낮은 직무
    print(f"\n[TOP 10 난이도 낮은 직무]")
    for i, item in enumerate(sorted(results, key=lambda x: x["score"])[:10], 1):
        print(f"  {i}. {item['name']} ({item['position_name']}) - {item['score']}점")
        print(f"      2024: {item['baseline_posts']}개 → 2025: {item['current_posts']}개")


# ==================== 메인 ====================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print("채용 난이도 지수 분석")
    print("=" * 60)
    print("\n[설정]")
    print("  기준년도: 2024년")
    print("  현재년도: 2025년")
    print("  분석 대상: 경쟁사 (9개 그룹)")
    print("  계산 방식: Min-Max 스케일링 (0-100)")

    db = next(get_db())

    try:
        # 1. 전체 난이도 분석
        overall_result = analyze_overall_difficulty(db)
        print_overall_result(overall_result)

        # 2. 직군별 난이도 분석
        position_results = analyze_position_difficulty(db)
        print_position_results(position_results)

        # 3. 직무별 난이도 분석
        industry_results = analyze_industry_difficulty(db)
        print_industry_results(industry_results)

        # 요약
        print(f"\n{'='*60}")
        print("[분석 요약]")
        print(f"{'='*60}")
        print(f"전체 난이도: {overall_result['score']}점")
        print(f"직군 수: {len(position_results)}개")
        print(f"직무 수: {len(industry_results)}개")
        if position_results:
            print(f"가장 경쟁이 심한 직군: {position_results[0]['name']} ({position_results[0]['score']}점)")
        if industry_results:
            print(f"가장 경쟁이 심한 직무: {industry_results[0]['name']} ({industry_results[0]['score']}점)")

    except Exception as e:
        print(f"\n오류: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    main()
