"""
HHI (Herfindahl-Hirschman Index) 수요 집중도 분석 - 1개월 윈도우

이 스크립트는 채용 시장의 직군/직무별 수요 집중도를 1개월 단위로 HHI 지수로 분석합니다.

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
    python test_07_1month.py [년] [월]

    예시:
    python test_07_1month.py 2025 12
    python test_07_1month.py 2025 11
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
from dateutil.relativedelta import relativedelta
import json


# ==================== 유틸리티 함수 ====================

def calculate_1month_window(year: int, month: int):
    """
    1개월 윈도우 계산

    Args:
        year: 연도 (예: 2025)
        month: 월 (1-12)

    Returns:
        tuple: (시작일, 종료일) - 문자열 (YYYY-MM-DD)

    Example:
        >>> calculate_1month_window(2025, 12)
        ('2025-12-01', '2025-12-31')
    """
    # 시작일: 해당 월의 첫날
    start_date = datetime(year, month-1, 1)

    # 종료일: 해당 월의 마지막 날
    end_date = start_date + relativedelta(months=1) - relativedelta(days=1)

    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


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
        전체 레벨에서는 직군(Position) 기준으로 HHI 계산

    Returns:
        {
            "period": {"start": str, "end": str},
            "total_posts": int,
            "hhi": float,
            "interpretation": {...},
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

    # HHI 해석
    interpretation = interpret_hhi(hhi)

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

    # 인사이트 생성
    insights = generate_overall_insights(hhi, interpretation, top_categories, total_posts)

    return {
        "period": {"start": start_date, "end": end_date},
        "total_posts": total_posts,
        "hhi": hhi,
        "interpretation": interpretation,
        "top_categories": top_categories,
        "insights": insights
    }


def analyze_position_hhi(db: Session, start_date: str, end_date: str):
    """
    직군별 HHI 집중도 분석

    Note:
        각 직군 내에서 산업(Industry) 기준으로 HHI 계산

    Returns:
        List of {
            "position_id": int,
            "position_name": str,
            "total_posts": int,
            "hhi": float,
            "interpretation": {...},
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

        # HHI 해석
        interpretation = interpret_hhi(hhi)

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

        # 인사이트 생성
        insights = generate_position_insights(
            industries[0]["position_name"],
            hhi,
            interpretation,
            top_industries,
            total_posts
        )

        results.append({
            "position_id": pos_id,
            "position_name": industries[0]["position_name"],
            "total_posts": total_posts,
            "hhi": hhi,
            "interpretation": interpretation,
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


# ==================== 인사이트 생성 ====================

def generate_overall_insights(hhi: float, interpretation: dict, top_categories: list, total_posts: int) -> list:
    """전체 레벨 인사이트 생성"""
    insights = []

    # 1. HHI 해석
    insights.append(f" 전체 채용 시장의 HHI는 {hhi:.4f}로, {interpretation['description']}")

    # 2. TOP 직군 분석
    if top_categories:
        top1 = top_categories[0]
        insights.append(
            f" '{top1['position_name']}'이(가) 전체의 {top1['share']}%({top1['posts_count']}개)를 차지하며 가장 높은 수요를 보입니다."
        )

        # 상위 3개 직군이 차지하는 비중
        if len(top_categories) >= 3:
            top3_share = sum(cat['share'] for cat in top_categories[:3])
            top3_names = ", ".join([cat['position_name'] for cat in top_categories[:3]])
            insights.append(
                f"상위 3개 직군({top3_names})이 전체의 {top3_share:.1f}%를 차지합니다."
            )

    # 3. 난이도 인사이트
    if interpretation['difficulty'] == '높음':
        insights.append(
            " 특정 직군에 수요가 집중되어 해당 분야의 인재 확보 경쟁이 치열합니다."
        )
    elif interpretation['difficulty'] == '중간':
        insights.append(
            " 일부 직군에서 수요 증가가 나타나고 있어 해당 분야의 경쟁이 점차 심화될 수 있습니다."
        )
    else:
        insights.append(
            " 다양한 직군에서 고르게 채용이 이루어져 전반적인 인재 수급 경쟁은 완화된 상태입니다."
        )

    return insights


def generate_position_insights(position_name: str, hhi: float, interpretation: dict,
                                 top_industries: list, total_posts: int) -> list:
    """직군별 인사이트 생성"""
    insights = []

    # 1. HHI 해석
    insights.append(
        f" '{position_name}' 직군의 HHI는 {hhi:.4f}로, 직무 간 {interpretation['level']} 상태입니다."
    )

    # 2. TOP 직무 분석
    if top_industries:
        top1 = top_industries[0]
        insights.append(
            f" '{top1['industry_name']}'이(가) {top1['share']}%({top1['posts_count']}개)로 가장 높은 수요를 보입니다."
        )

        # 최상위 직무가 과반을 차지하는 경우
        if top1['share'] >= 50:
            insights.append(
                f" 단일 직무가 절반 이상을 차지하여 해당 분야의 경쟁이 매우 치열합니다."
            )

    # 3. 난이도 인사이트
    if interpretation['difficulty'] == '높음':
        insights.append(
            f" '{position_name}' 내에서 특정 직무로 수요가 쏠려 있어 해당 직무의 인재 확보가 어려울 수 있습니다."
        )
    elif interpretation['difficulty'] == '중간':
        insights.append(
            f" '{position_name}' 내 일부 직무에 수요가 집중되는 경향을 보입니다."
        )
    else:
        insights.append(
            f" '{position_name}' 내 다양한 직무에서 고르게 채용이 이루어지고 있습니다."
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
    print(f"\n{'-'*80}")
    print(" 전체 시장 분석")
    print(f"{'-'*80}")
    print(f"기간: {overall_result['period']['start']} ~ {overall_result['period']['end']}")
    print(f"총 공고 수: {overall_result['total_posts']}개")
    print(f"HHI: {overall_result['hhi']:.4f} ({overall_result['interpretation']['level']} / 난이도: {overall_result['interpretation']['difficulty']})")
    print()
    for insight in overall_result['insights']:
        print(f"  {insight}")

    # 2. 직군별 인사이트 (상위 5개만)
    print(f"\n{'-'*80}")
    print(" 직군별 집중도 분석 (TOP 5)")
    print(f"{'-'*80}")

    for i, result in enumerate(position_results[:5], 1):
        print(f"\n{i}. {result['position_name']}")
        print(f"   HHI: {result['hhi']:.4f} ({result['interpretation']['level']} / 난이도: {result['interpretation']['difficulty']})")
        print(f"   총 공고: {result['total_posts']}개")
        print()
        for insight in result['insights']:
            print(f"     {insight}")

    print(f"\n{'='*80}")


# ==================== 메인 ====================

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("HHI (Herfindahl-Hirschman Index) 수요 집중도 분석 - 1개월 윈도우")
    print("=" * 80)

    # 인자 파싱
    if len(sys.argv) > 2:
        target_year = int(sys.argv[1])
        target_month = int(sys.argv[2])
    else:
        # 기본값: 현재 날짜
        now = datetime.now()
        target_year = now.year
        target_month = now.month

    # 1개월 윈도우 계산
    start_date, end_date = calculate_1month_window(target_year, target_month)

    print(f"\n[설정]")
    print(f"  분석 기간: {start_date} ~ {end_date}")
    print(f"  분석 대상: 경쟁사 (9개 그룹)")
    print(f"  계산 방식: HHI = Σ(si²)")
    print(f"\n[HHI 해석 기준]")
    print(f"  0.00 ~ 0.15: 분산 (경쟁 낮음)")
    print(f"  0.15 ~ 0.25: 부분집중 (경쟁 중간)")
    print(f"  0.25 이상:   쏠림 (경쟁 높음)")

    db = next(get_db())

    try:
        # 1. 전체 HHI 분석
        print(f"\n{'-'*80}")
        print("[1] 전체 HHI 분석 중...")
        print(f"{'-'*80}")
        overall_result = analyze_overall_hhi(db, start_date, end_date)
        print(f"완료 - 전체 HHI: {overall_result['hhi']:.4f}")
        print(f"완료 - 총 공고 수: {overall_result['total_posts']}개")

        # 2. 직군별 HHI 분석
        print(f"\n{'-'*80}")
        print("[2] 직군별 HHI 분석 중...")
        print(f"{'-'*80}")
        position_results = analyze_position_hhi(db, start_date, end_date)
        print(f"완료 - 분석 완료: {len(position_results)}개 직군")

        # 3. 직무별 점유율 분석
        print(f"\n{'-'*80}")
        print("[3] 직무별 점유율 분석 중...")
        print(f"{'-'*80}")
        industry_results = analyze_industry_hhi(db, start_date, end_date)
        print(f"완료 - 분석 완료: {len(industry_results)}개 직무")

        # 4. 텍스트 인사이트 출력
        print_text_insights(overall_result, position_results)

        # 5. JSON 형식 출력
        print(f"\n{'='*80}")
        print(" JSON 출력 (대시보드 인사이트 창용)")
        print(f"{'='*80}")
        json_output = format_json_output(overall_result, position_results, industry_results)
        print(json.dumps(json_output, ensure_ascii=False, indent=2))

        # 요약
        print(f"\n{'='*80}")
        print("[분석 완료]")
        print(f"{'='*80}")
        print(f"전체 HHI: {overall_result['hhi']:.4f} ({overall_result['interpretation']['level']})")
        print(f"직군 수: {len(position_results)}개")
        print(f"직무 수: {len(industry_results)}개")
        if position_results:
            print(f"가장 쏠림이 심한 직군: {position_results[0]['position_name']} (HHI: {position_results[0]['hhi']:.4f})")

    except Exception as e:
        print(f"\n오류: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db.close()


if __name__ == "__main__":
    main()
