"""
YoY (Year-over-Year) Overheat Index 분석

이 스크립트는 전년도 대비 금년도 채용 과열도를 3개월 슬라이딩 윈도우로 분석합니다.

주요 기능:
1. 전체 YoY 과열도 지수 계산
2. 직군(Position)별 YoY 과열도 지수 계산
3. 직무(Industry)별 YoY 과열도 지수 계산
4. 현재 월 기준 3개월 윈도우 비교 (예: 2025-12 → 2025.10~12 vs 2024.10~12)
5. YoY 공식을 통한 0-100 점수 산출

계산 공식:
    YoY(t) = min(100, max(0, Ct/Bt-1 × 50))
    - Ct: 현재 년도 3개월 공고 수
    - Bt-1: 전년도 동일 3개월 공고 수
    - 0-100 범위로 cap

사용법:
    python test_06_yoy_index.py [년] [월]

    예시:
    python test_06_yoy_index.py 2025 12
    python test_06_yoy_index.py 2025 6
"""

import sys
import os

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sqlalchemy.orm import Session
from sqlalchemy import func, or_, extract, and_
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
from dateutil.relativedelta import relativedelta


# ==================== 유틸리티 함수 ====================

def calculate_3month_window(year: int, month: int):
    """
    현재 월 기준 3개월 윈도우 계산

    Args:
        year: 연도 (예: 2025)
        month: 월 (1-12)

    Returns:
        tuple: (시작월, 종료월) - datetime 객체

    Example:
        >>> calculate_3month_window(2025, 12)
        (datetime(2025, 10, 1), datetime(2025, 12, 31))
    """
    # 종료일: 해당 월의 마지막 날
    end_date = datetime(year, month, 1) + relativedelta(months=1) - relativedelta(days=1)

    # 시작일: 3개월 전 (종료월 - 2개월)
    start_date = datetime(year, month, 1) - relativedelta(months=2)

    return start_date, end_date


def calculate_yoy_score(current_count: int, previous_count: int) -> int:
    """
    YoY 과열도 점수 계산

    Args:
        current_count: 현재 년도 3개월 공고 수 (Ct)
        previous_count: 전년도 동일 3개월 공고 수 (Bt-1)

    Returns:
        0-100 범위의 YoY 점수

    Formula:
        YoY(t) = min(100, max(0, Ct/Bt-1 × 50))
    """
    # 전년도 공고가 없으면 비교 불가
    if previous_count == 0:
        # 현재 공고가 있으면 최대값, 없으면 0
        return 100 if current_count > 0 else 0

    # YoY 계산
    ratio = current_count / previous_count
    score = ratio * 50

    # 0-100 범위로 cap
    if score > 100:
        return 100
    elif score < 0:
        return 0
    else:
        return int(round(score))


# ==================== 경쟁사 필터링 ====================

def get_competitor_conditions():
    """경쟁사 필터링 조건 생성"""
    like_conditions = []
    for patterns in COMPANY_GROUPS.values():
        for pattern in patterns:
            like_conditions.append(Company.name.like(pattern))
    return like_conditions


# ==================== 데이터 조회 ====================

def get_posts_count_in_window(
    db: Session,
    start_date: datetime,
    end_date: datetime,
    position_id: int = None,
    industry_id: int = None
) -> int:
    """
    특정 기간 내 경쟁사 공고 수 조회

    Args:
        db: Database session
        start_date: 시작일
        end_date: 종료일
        position_id: 직군 ID (optional)
        industry_id: 직무 ID (optional)

    Returns:
        공고 수
    """
    # 경쟁사 조건
    competitor_conditions = get_competitor_conditions()

    # 기본 쿼리 (Industry를 통한 JOIN)
    query = db.query(func.count(Post.id))\
        .join(Company, Post.company_id == Company.id)\
        .join(Industry, Post.industry_id == Industry.id)\
        .filter(
            or_(*competitor_conditions),
            Post.posted_at >= start_date,
            Post.posted_at <= end_date
        )

    # 직군 필터 (Industry를 통해 필터링)
    if position_id is not None:
        query = query.filter(Industry.position_id == position_id)

    # 직무 필터
    if industry_id is not None:
        query = query.filter(Post.industry_id == industry_id)

    return query.scalar() or 0


def get_all_positions_with_yoy(db: Session, current_start: datetime, current_end: datetime,
                                 previous_start: datetime, previous_end: datetime):
    """
    모든 직군과 해당 기간의 YoY 데이터 조회

    Returns:
        List of (position_id, position_name, current_count, previous_count)
    """
    competitor_conditions = get_competitor_conditions()

    # Current 기간 집계
    current_results = db.query(
        Position.id,
        Position.name,
        func.count(Post.id).label('posts_count')
    )\
        .join(Industry, Industry.position_id == Position.id)\
        .join(Post, Post.industry_id == Industry.id)\
        .join(Company, Post.company_id == Company.id)\
        .filter(
            or_(*competitor_conditions),
            Post.posted_at >= current_start,
            Post.posted_at <= current_end
        )\
        .group_by(Position.id, Position.name)\
        .all()

    current_dict = {pos_id: (pos_name, count) for pos_id, pos_name, count in current_results}

    # Previous 기간 집계
    previous_results = db.query(
        Position.id,
        func.count(Post.id).label('posts_count')
    )\
        .join(Industry, Industry.position_id == Position.id)\
        .join(Post, Post.industry_id == Industry.id)\
        .join(Company, Post.company_id == Company.id)\
        .filter(
            or_(*competitor_conditions),
            Post.posted_at >= previous_start,
            Post.posted_at <= previous_end
        )\
        .group_by(Position.id)\
        .all()

    previous_dict = {pos_id: count for pos_id, count in previous_results}

    # 통합
    all_position_ids = set(current_dict.keys()) | set(previous_dict.keys())

    results = []
    for pos_id in all_position_ids:
        if pos_id in current_dict:
            pos_name, current_count = current_dict[pos_id]
        else:
            # Previous에만 있는 경우 (Position 조회 필요)
            position = db.query(Position).filter(Position.id == pos_id).first()
            pos_name = position.name if position else f"Unknown_{pos_id}"
            current_count = 0

        previous_count = previous_dict.get(pos_id, 0)

        results.append((pos_id, pos_name, current_count, previous_count))

    return results


def get_all_industries_with_yoy(db: Session, current_start: datetime, current_end: datetime,
                                  previous_start: datetime, previous_end: datetime):
    """
    모든 직무와 해당 기간의 YoY 데이터 조회

    Returns:
        List of (industry_id, industry_name, position_id, position_name, current_count, previous_count)
    """
    competitor_conditions = get_competitor_conditions()

    # Current 기간 집계
    current_results = db.query(
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
            Post.posted_at >= current_start,
            Post.posted_at <= current_end
        )\
        .group_by(Industry.id, Industry.name, Position.id, Position.name)\
        .all()

    current_dict = {ind_id: (ind_name, pos_id, pos_name, count)
                    for ind_id, ind_name, pos_id, pos_name, count in current_results}

    # Previous 기간 집계
    previous_results = db.query(
        Industry.id,
        func.count(Post.id).label('posts_count')
    )\
        .join(Post, Post.industry_id == Industry.id)\
        .join(Company, Post.company_id == Company.id)\
        .filter(
            or_(*competitor_conditions),
            Post.posted_at >= previous_start,
            Post.posted_at <= previous_end
        )\
        .group_by(Industry.id)\
        .all()

    previous_dict = {ind_id: count for ind_id, count in previous_results}

    # 통합
    all_industry_ids = set(current_dict.keys()) | set(previous_dict.keys())

    results = []
    for ind_id in all_industry_ids:
        if ind_id in current_dict:
            ind_name, pos_id, pos_name, current_count = current_dict[ind_id]
        else:
            # Previous에만 있는 경우 (Industry 조회 필요)
            industry = db.query(Industry, Position)\
                .join(Position, Industry.position_id == Position.id)\
                .filter(Industry.id == ind_id)\
                .first()

            if industry:
                ind_name = industry.Industry.name
                pos_id = industry.Position.id
                pos_name = industry.Position.name
            else:
                ind_name = f"Unknown_{ind_id}"
                pos_id = None
                pos_name = "Unknown"

            current_count = 0

        previous_count = previous_dict.get(ind_id, 0)

        results.append((ind_id, ind_name, pos_id, pos_name, current_count, previous_count))

    return results


# ==================== YoY 분석 ====================

def analyze_overall_yoy(db: Session, current_start: datetime, current_end: datetime,
                         previous_start: datetime, previous_end: datetime):
    """
    전체 YoY 과열도 분석

    Returns:
        {
            "name": "전체",
            "current_period": {"start": str, "end": str},
            "previous_period": {"start": str, "end": str},
            "current_posts": int,
            "previous_posts": int,
            "yoy_score": int
        }
    """
    # 현재 기간 공고 수
    current_posts = get_posts_count_in_window(db, current_start, current_end)

    # 전년도 동일 기간 공고 수
    previous_posts = get_posts_count_in_window(db, previous_start, previous_end)

    # YoY 점수 계산
    yoy_score = calculate_yoy_score(current_posts, previous_posts)

    return {
        "name": "전체",
        "current_period": {
            "start": current_start.strftime("%Y-%m-%d"),
            "end": current_end.strftime("%Y-%m-%d")
        },
        "previous_period": {
            "start": previous_start.strftime("%Y-%m-%d"),
            "end": previous_end.strftime("%Y-%m-%d")
        },
        "current_posts": current_posts,
        "previous_posts": previous_posts,
        "yoy_score": yoy_score
    }


def analyze_position_yoy(db: Session, current_start: datetime, current_end: datetime,
                          previous_start: datetime, previous_end: datetime):
    """
    직군별 YoY 과열도 분석

    Returns:
        List of {
            "position_id": int,
            "name": str,
            "current_posts": int,
            "previous_posts": int,
            "yoy_score": int
        }
    """
    # 직군별 데이터 조회
    position_data = get_all_positions_with_yoy(db, current_start, current_end,
                                                previous_start, previous_end)

    results = []
    for pos_id, pos_name, current_posts, previous_posts in position_data:
        # YoY 점수 계산
        yoy_score = calculate_yoy_score(current_posts, previous_posts)

        results.append({
            "position_id": pos_id,
            "name": pos_name,
            "current_posts": current_posts,
            "previous_posts": previous_posts,
            "yoy_score": yoy_score
        })

    # 점수 순으로 정렬
    results.sort(key=lambda x: x["yoy_score"], reverse=True)

    return results


def analyze_industry_yoy(db: Session, current_start: datetime, current_end: datetime,
                          previous_start: datetime, previous_end: datetime):
    """
    직무별 YoY 과열도 분석

    Returns:
        List of {
            "industry_id": int,
            "name": str,
            "position_id": int,
            "position_name": str,
            "current_posts": int,
            "previous_posts": int,
            "yoy_score": int
        }
    """
    # 직무별 데이터 조회
    industry_data = get_all_industries_with_yoy(db, current_start, current_end,
                                                 previous_start, previous_end)

    results = []
    for ind_id, ind_name, pos_id, pos_name, current_posts, previous_posts in industry_data:
        # YoY 점수 계산
        yoy_score = calculate_yoy_score(current_posts, previous_posts)

        results.append({
            "industry_id": ind_id,
            "name": ind_name,
            "position_id": pos_id,
            "position_name": pos_name,
            "current_posts": current_posts,
            "previous_posts": previous_posts,
            "yoy_score": yoy_score
        })

    # 점수 순으로 정렬
    results.sort(key=lambda x: x["yoy_score"], reverse=True)

    return results


# ==================== 출력 ====================

def print_overall_result(result):
    """전체 YoY 결과 출력"""
    print(f"\n{'='*60}")
    print(f"[전체 YoY 과열도 지수]")
    print(f"{'='*60}")
    print(f"  이름: {result['name']}")
    print(f"  현재 기간: {result['current_period']['start']} ~ {result['current_period']['end']}")
    print(f"  전년 기간: {result['previous_period']['start']} ~ {result['previous_period']['end']}")
    print(f"  현재 공고 수: {result['current_posts']}개")
    print(f"  전년 공고 수: {result['previous_posts']}개")
    print(f"  YoY 점수: {result['yoy_score']}점")

    # 해석
    if result['yoy_score'] >= 75:
        interpretation = "매우 높음 (과열)"
    elif result['yoy_score'] >= 60:
        interpretation = "높음"
    elif result['yoy_score'] >= 40:
        interpretation = "보통"
    elif result['yoy_score'] >= 25:
        interpretation = "낮음"
    else:
        interpretation = "매우 낮음 (침체)"

    print(f"  해석: {interpretation}")


def print_position_results(results):
    """직군별 YoY 결과 출력"""
    print(f"\n{'='*60}")
    print(f"[직군별 YoY 과열도 지수]")
    print(f"{'='*60}")
    print(f"\n{'직군명':<30} {'현재':<8} {'전년':<8} {'점수':<6}")
    print("-" * 60)

    for item in results:
        print(f"{item['name']:<30} {item['current_posts']:<8} {item['previous_posts']:<8} {item['yoy_score']:<6}")

    print(f"\n총 {len(results)}개 직군")

    # TOP 5 과열도 높은 직군
    print(f"\n[TOP 5 과열도 높은 직군]")
    for i, item in enumerate(results[:5], 1):
        change = item['current_posts'] - item['previous_posts']
        change_str = f"+{change}" if change >= 0 else str(change)
        print(f"  {i}. {item['name']} - {item['yoy_score']}점")
        print(f"      전년: {item['previous_posts']}개 → 현재: {item['current_posts']}개 ({change_str})")

    # TOP 5 과열도 낮은 직군
    print(f"\n[TOP 5 과열도 낮은 직군]")
    sorted_low = sorted(results, key=lambda x: x["yoy_score"])
    for i, item in enumerate(sorted_low[:5], 1):
        change = item['current_posts'] - item['previous_posts']
        change_str = f"+{change}" if change >= 0 else str(change)
        print(f"  {i}. {item['name']} - {item['yoy_score']}점")
        print(f"      전년: {item['previous_posts']}개 → 현재: {item['current_posts']}개 ({change_str})")


def print_industry_results(results):
    """직무별 YoY 결과 출력"""
    print(f"\n{'='*60}")
    print(f"[직무별 YoY 과열도 지수]")
    print(f"{'='*60}")
    print(f"\n{'직무명':<30} {'직군':<20} {'현재':<8} {'전년':<8} {'점수':<6}")
    print("-" * 80)

    for item in results:
        print(f"{item['name']:<30} {item['position_name']:<20} {item['current_posts']:<8} {item['previous_posts']:<8} {item['yoy_score']:<6}")

    print(f"\n총 {len(results)}개 직무")

    # TOP 10 과열도 높은 직무
    print(f"\n[TOP 10 과열도 높은 직무]")
    for i, item in enumerate(results[:10], 1):
        change = item['current_posts'] - item['previous_posts']
        change_str = f"+{change}" if change >= 0 else str(change)
        print(f"  {i}. {item['name']} ({item['position_name']}) - {item['yoy_score']}점")
        print(f"      전년: {item['previous_posts']}개 → 현재: {item['current_posts']}개 ({change_str})")

    # TOP 10 과열도 낮은 직무
    print(f"\n[TOP 10 과열도 낮은 직무]")
    sorted_low = sorted(results, key=lambda x: x["yoy_score"])
    for i, item in enumerate(sorted_low[:10], 1):
        change = item['current_posts'] - item['previous_posts']
        change_str = f"+{change}" if change >= 0 else str(change)
        print(f"  {i}. {item['name']} ({item['position_name']}) - {item['yoy_score']}점")
        print(f"      전년: {item['previous_posts']}개 → 현재: {item['current_posts']}개 ({change_str})")


# ==================== 메인 ====================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print("YoY (Year-over-Year) Overheat Index 분석")
    print("=" * 60)

    # 인자 파싱
    if len(sys.argv) > 2:
        target_year = int(sys.argv[1])
        target_month = int(sys.argv[2])
    else:
        # 기본값: 현재 날짜
        now = datetime.now()
        target_year = now.year
        target_month = now.month

    print(f"\n[설정]")
    print(f"  기준 년월: {target_year}년 {target_month}월")
    print(f"  분석 대상: 경쟁사 (9개 그룹)")
    print(f"  윈도우: 3개월 슬라이딩")
    print(f"  계산 방식: YoY(t) = min(100, max(0, Ct/Bt-1 × 50))")

    # 3개월 윈도우 계산
    current_start, current_end = calculate_3month_window(target_year, target_month)
    previous_start, previous_end = calculate_3month_window(target_year - 1, target_month)

    print(f"\n  현재 기간: {current_start.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')}")
    print(f"  전년 기간: {previous_start.strftime('%Y-%m-%d')} ~ {previous_end.strftime('%Y-%m-%d')}")

    db = next(get_db())

    try:
        # 1. 전체 YoY 분석
        overall_result = analyze_overall_yoy(db, current_start, current_end,
                                              previous_start, previous_end)
        print_overall_result(overall_result)

        # 2. 직군별 YoY 분석
        position_results = analyze_position_yoy(db, current_start, current_end,
                                                 previous_start, previous_end)
        print_position_results(position_results)

        # 3. 직무별 YoY 분석
        industry_results = analyze_industry_yoy(db, current_start, current_end,
                                                 previous_start, previous_end)
        print_industry_results(industry_results)

        # 요약
        print(f"\n{'='*60}")
        print("[분석 요약]")
        print(f"{'='*60}")
        print(f"전체 YoY 점수: {overall_result['yoy_score']}점")
        print(f"직군 수: {len(position_results)}개")
        print(f"직무 수: {len(industry_results)}개")
        if position_results:
            print(f"가장 과열된 직군: {position_results[0]['name']} ({position_results[0]['yoy_score']}점)")
        if industry_results:
            print(f"가장 과열된 직무: {industry_results[0]['name']} ({industry_results[0]['yoy_score']}점)")

    except Exception as e:
        print(f"\n오류: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    main()
