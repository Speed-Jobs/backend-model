"""
HHI Concentration Index Service
"""
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple
import math

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.crud import db_hhi_concentration
from app.schemas.schemas_hhi_concentration import (
    PositionConcentration,
    IndustryConcentration,
    OverallAnalysisData,
    PositionAnalysisData,
    IndustryAnalysisData,
    CombinedIndustryAnalysisData,
    PeriodInfo,
    HHIInterpretation,
    AlternativeIndustry,
)
from app.utils.date_calculator import calculate_3month_period, calculate_previous_year_period
# YoY Overheat 계산 함수 import
from app.services.dashboard.yoy_overheat import _calculate_yoy_score, _get_trend
from app.db.crud import db_yoy_overheat
# Models are imported lazily to avoid circular dependencies


def _calculate_hhi(counts: List[int]) -> float:
    """
    HHI 지수 계산

    공식: HHI = Σ(si²) where si = 항목i의 점유율

    Args:
        counts: 항목별 건수 리스트

    Returns:
        HHI 지수 (0~1)
    """
    total = sum(counts)
    if total == 0:
        return 0.0

    hhi = sum((count / total) ** 2 for count in counts)
    return hhi


# ===== 새로운 서비스 함수 =====

def _calculate_cr2(counts: List[int]) -> float:
    """
    CR₂ (Concentration Ratio) 계산 - 상위 2개 항목 점유율

    Args:
        counts: 항목별 건수 리스트

    Returns:
        CR₂ 지수 (0~1)
    """
    if not counts:
        return 0.0

    total = sum(counts)
    if total == 0:
        return 0.0

    sorted_counts = sorted(counts, reverse=True)
    top2_sum = sum(sorted_counts[:2])
    return top2_sum / total


def _calculate_entropy(counts: List[int]) -> float:
    """
    Shannon Entropy 계산 - 다양성 지수

    Args:
        counts: 항목별 건수 리스트

    Returns:
        Shannon Entropy 값 (높을수록 다양)
    """
    if not counts:
        return 0.0

    total = sum(counts)
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def _interpret_hhi_new(hhi_value: float) -> Tuple[str, str]:
    """
    HHI 값을 수준과 난이도로 해석 (신규)

    Args:
        hhi_value: HHI 지수 (0~1)

    Returns:
        (level, difficulty)
        - level: "분산" / "부분집중" / "쏠림"
        - difficulty: "낮음" / "보통" / "높음"
    """
    if hhi_value < 0.15:
        return "분산", "낮음"
    elif hhi_value < 0.25:
        return "부분집중", "보통"
    else:
        return "쏠림", "높음"


def _get_position_data(
    db: Session,
    start_date: date,
    end_date: date,
) -> List[Tuple[int, str, int]]:
    """
    직군별 채용 공고 수 조회

    Post 테이블을 사용하여 posted_at (없으면 crawled_at) 기준으로 집계합니다.

    Returns:
        List of (position_id, position_name, count)
    """
    from app.models.post import Post
    from app.models.industry import Industry
    from app.models.position import Position

    # posted_at이 없으면 crawled_at 사용
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)

    results = (
        db.query(
            Position.id,
            Position.name,
            func.count(Post.id).label("count"),  # Post.id를 카운트
        )
        .join(Industry, Post.industry_id == Industry.id)
        .join(Position, Industry.position_id == Position.id)
        .filter(
            Post.is_deleted == False,  # 삭제되지 않은 공고만
            effective_date >= start_date,
            effective_date <= end_date,
        )
        .group_by(Position.id, Position.name)
        .all()
    )

    return results


def _get_industry_data_for_position(
    db: Session,
    start_date: date,
    end_date: date,
    position_id: int,
) -> List[Tuple[int, str, int]]:
    """
    특정 직군 내 산업별 채용 공고 수 조회

    Post 테이블을 사용하여 posted_at (없으면 crawled_at) 기준으로 집계합니다.

    Returns:
        List of (industry_id, industry_name, count)
    """
    from app.models.post import Post
    from app.models.industry import Industry

    # posted_at이 없으면 crawled_at 사용
    effective_date = func.coalesce(Post.posted_at, Post.crawled_at)

    results = (
        db.query(
            Industry.id,
            Industry.name,
            func.count(Post.id).label("count"),  # Post.id를 카운트
        )
        .join(Industry, Post.industry_id == Industry.id)
        .filter(
            Post.is_deleted == False,  # 삭제되지 않은 공고만
            effective_date >= start_date,
            effective_date <= end_date,
            Industry.position_id == position_id,
        )
        .group_by(Industry.id, Industry.name)
        .all()
    )

    return results


def _get_skill_similarity_industries(
    db: Session,
    target_industry_id: int,
    position_id: int,
    start_date: date,
    end_date: date,
    limit: int = 3,
) -> List[AlternativeIndustry]:
    """
    스킬 유사도 기반 대안 산업 추천

    Args:
        db: DB 세션
        target_industry_id: 대상 산업 ID
        position_id: 직군 ID
        start_date: 시작일
        end_date: 종료일
        limit: 최대 추천 개수

    Returns:
        List of AlternativeIndustry
    """
    from app.models.industry import Industry
    from app.models.industry_skill import IndustrySkill

    # 1. 대상 산업의 스킬 목록
    target_skills = (
        db.query(IndustrySkill.skill_id)
        .filter(IndustrySkill.industry_id == target_industry_id)
        .all()
    )
    target_skill_ids = {s[0] for s in target_skills}

    if not target_skill_ids:
        return []

    # 2. 같은 직군 내 다른 산업들
    other_industries = (
        db.query(Industry.id, Industry.name)
        .filter(
            Industry.position_id == position_id,
            Industry.id != target_industry_id,
        )
        .all()
    )

    # 3. 각 산업별 스킬 유사도 계산
    similarities = []
    for ind_id, ind_name in other_industries:
        # 해당 산업의 스킬 목록
        ind_skills = (
            db.query(IndustrySkill.skill_id)
            .filter(IndustrySkill.industry_id == ind_id)
            .all()
        )
        ind_skill_ids = {s[0] for s in ind_skills}

        if not ind_skill_ids:
            continue

        # Jaccard 유사도
        intersection = len(target_skill_ids & ind_skill_ids)
        union = len(target_skill_ids | ind_skill_ids)
        similarity = intersection / union if union > 0 else 0.0

        similarities.append((ind_id, ind_name, similarity))

    # 4. 유사도 순으로 정렬
    similarities.sort(key=lambda x: x[2], reverse=True)

    # 5. 상위 N개 선택하고 점유율 추가
    industry_data = _get_industry_data_for_position(db, start_date, end_date, position_id)
    industry_dict = {ind_id: count for ind_id, ind_name, count in industry_data}
    total_count = sum(industry_dict.values())

    alternatives = []
    for ind_id, ind_name, similarity in similarities[:limit]:
        count = industry_dict.get(ind_id, 0)
        share = (count / total_count * 100) if total_count > 0 else 0.0

        alternatives.append(
            AlternativeIndustry(
                industry_id=ind_id,
                industry_name=ind_name,
                skill_similarity=round(similarity, 2),
                share_percentage=round(share, 2),
            )
        )

    return alternatives


# ===== 새로운 API 서비스 함수 (Scenario-based) =====

def analyze_overall_market(
    db: Session,
    start_date_str: str,
    include_insights: bool = False,
) -> OverallAnalysisData:
    """
    전체 시장 HHI + YoY 분석

    Args:
        db: DB 세션
        start_date_str: 종료일 (YYYY-MM-DD)
        include_insights: 인사이트 생성 여부 (default: False)

    Returns:
        OverallAnalysisData
    """
    # 기간 계산 (3개월 고정, 과거 3개월)
    start_date, end_date = calculate_3month_period(start_date_str)
    previous_start_date, previous_end_date = calculate_previous_year_period(start_date, end_date)

    # === HHI 계산 ===
    position_data = _get_position_data(db, start_date, end_date)

    if not position_data:
        raise ValueError("해당 기간에 데이터가 없습니다")

    counts = [count for _, _, count in position_data]
    total_posts = sum(counts)

    hhi_value = _calculate_hhi(counts)
    cr2_value = _calculate_cr2(counts)
    entropy_value = _calculate_entropy(counts)
    level, difficulty = _interpret_hhi_new(hhi_value)

    # 상위 직군 (최대 5개)
    position_data_sorted = sorted(position_data, key=lambda x: x[2], reverse=True)[:5]
    top_positions = [
        PositionConcentration(
            position_id=pos_id,
            position_name=pos_name,
            count=count,
            share_percentage=round((count / total_posts * 100), 2),
            rank=idx + 1,
        )
        for idx, (pos_id, pos_name, count) in enumerate(position_data_sorted)
    ]

    # === YoY 계산 ===
    current_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=start_date,
        end_date=end_date,
        company_patterns=None,
    )
    previous_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=previous_start_date,
        end_date=previous_end_date,
        company_patterns=None,
    )

    overall_current = sum(count for _, _, _, _, count in current_rows)
    overall_previous = sum(count for _, _, _, _, count in previous_rows)
    yoy_score = _calculate_yoy_score(overall_current, overall_previous)
    yoy_trend = _get_trend(yoy_score)

    # === 인사이트 생성 (include_insights=true일 때만) ===
    insights = []
    if include_insights:
        # HHI 인사이트 (숫자 앞에 배치)
        if hhi_value < 0.15:
            insights.append(
                f"집중도 지수가 {hhi_value:.2f}로 나타나 다양한 직군에서 고르게 채용이 진행되고 있어 "
                f"시장 경쟁이 분산되어 있습니다"
            )
        elif hhi_value < 0.25:
            insights.append(
                f"집중도 지수가 {hhi_value:.2f}로 나타나 일부 직군에 수요가 몰리는 경향은 있으나, "
                f"전체적으로는 채용이 다양한 직군으로 분산되는 흐름입니다"
            )
        else:
            insights.append(
                f"집중도 지수가 {hhi_value:.2f}로 나타나 특정 직군에 채용이 과도하게 집중되어 있어 "
                f"포트폴리오 다양화가 필요합니다"
            )

        # CR₂ 인사이트
        cr2_percentage = cr2_value * 100
        if cr2_value > 0.5:
            insights.append(
                f"상위 2개 직군이 전체의 {cr2_percentage:.1f}%를 차지하고 있어 높은 집중도를 보이고 있습니다"
            )
        else:
            insights.append(
                f"상위 2개 직군이 전체의 {cr2_percentage:.1f}%를 차지하고 있어 일부 집중 경향이 있습니다"
            )

        # Entropy 인사이트 (다양성 지수로 표현)
        max_entropy = math.log2(len(position_data)) if len(position_data) > 0 else 1
        normalized_entropy = entropy_value / max_entropy if max_entropy > 0 else 0

        if normalized_entropy > 0.8:
            insights.append(
                f"다양성 지수가 {normalized_entropy:.2f}로 나타나 직군 구성이 매우 다양하여 "
                f"지원자 입장에서 다양한 선택지를 확보할 수 있습니다"
            )
        elif normalized_entropy > 0.6:
            insights.append(
                f"다양성 지수가 {normalized_entropy:.2f}로 나타나 직군 구성이 비교적 다양하나, "
                f"일부 직군 간 격차가 존재합니다"
            )
        else:
            insights.append(
                f"다양성 지수가 {normalized_entropy:.2f}로 나타나 직군 구성이 특정 분야에 치우쳐 있어 "
                f"다양성 확보가 필요합니다"
            )

        # YoY 인사이트
        if yoy_score > 50:
            change_percentage = ((overall_current / overall_previous) - 1) * 100 if overall_previous > 0 else 0
            insights.append(
                f"전년 대비 과열도 지수가 {yoy_score:.2f}로, 작년보다 {change_percentage:.1f}% 증가하여 "
                f"채용 시장이 확대되는 추세입니다"
            )
        elif yoy_score == 50:
            insights.append(
                f"전년 대비 과열도 지수가 {yoy_score:.2f}로, 작년과 동일한 수준을 유지하고 있습니다"
            )
        else:
            change_percentage = (1 - (overall_current / overall_previous)) * 100 if overall_previous > 0 else 0
            insights.append(
                f"전년 대비 과열도 지수가 {yoy_score:.2f}로, 작년보다 {change_percentage:.1f}% 감소하여 "
                f"채용 시장이 냉각되는 추세입니다"
            )

    return OverallAnalysisData(
        analysis_type="overall",
        period=PeriodInfo(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")),
        total_posts=total_posts,
        hhi=round(hhi_value, 4),
        interpretation=HHIInterpretation(level=level, difficulty=difficulty),
        top_positions=top_positions,
        yoy_overheat_score=round(yoy_score, 2),
        yoy_trend=yoy_trend,
        yoy_current_count=overall_current,
        yoy_previous_count=overall_previous,
        insights=insights,
    )


def analyze_position_industries(
    db: Session,
    start_date_str: str,
    position_id: int,
    include_insights: bool = False,
) -> PositionAnalysisData:
    """
    특정 직군 내 산업별 HHI + YoY 분석

    Args:
        db: DB 세션
        start_date_str: 종료일 (YYYY-MM-DD)
        position_id: 직군 ID
        include_insights: 인사이트 생성 여부 (default: False)

    Returns:
        PositionAnalysisData
    """
    from app.models.position import Position

    # 기간 계산 (3개월 고정, 과거 3개월)
    start_date, end_date = calculate_3month_period(start_date_str)
    previous_start_date, previous_end_date = calculate_previous_year_period(start_date, end_date)

    # 직군 정보 조회
    position = db.query(Position).filter(Position.id == position_id).first()
    if not position:
        raise ValueError(f"직군 ID {position_id}를 찾을 수 없습니다")

    # === HHI 계산 ===
    industry_data = _get_industry_data_for_position(db, start_date, end_date, position_id)

    if not industry_data:
        raise ValueError(f"해당 직군({position.name})에 데이터가 없습니다")

    counts = [count for _, _, count in industry_data]
    total_posts = sum(counts)

    hhi_value = _calculate_hhi(counts)
    cr2_value = _calculate_cr2(counts)
    entropy_value = _calculate_entropy(counts)
    level, difficulty = _interpret_hhi_new(hhi_value)

    # 상위 산업 (최대 5개)
    industry_data_sorted = sorted(industry_data, key=lambda x: x[2], reverse=True)[:5]
    top_industries = [
        IndustryConcentration(
            industry_id=ind_id,
            industry_name=ind_name,
            count=count,
            share_percentage=round((count / total_posts * 100), 2),
            rank=idx + 1,
        )
        for idx, (ind_id, ind_name, count) in enumerate(industry_data_sorted)
    ]

    # === YoY 계산 ===
    current_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=start_date,
        end_date=end_date,
        company_patterns=None,
    )
    previous_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=previous_start_date,
        end_date=previous_end_date,
        company_patterns=None,
    )

    # 해당 직군 데이터만 필터링
    position_current = sum(
        count for pos_id, _, _, _, count in current_rows
        if pos_id == position_id
    )
    position_previous = sum(
        count for pos_id, _, _, _, count in previous_rows
        if pos_id == position_id
    )

    yoy_score = _calculate_yoy_score(position_current, position_previous)
    yoy_trend = _get_trend(yoy_score)

    # === 인사이트 생성 (include_insights=true일 때만) ===
    insights = []
    if include_insights:
        # HHI 인사이트 (숫자 앞에 배치)
        if hhi_value < 0.15:
            insights.append(
                f"집중도 지수가 {hhi_value:.2f}로 나타나 '{position.name}' 직군 내에서 산업별 채용이 다양하게 분포되어 있습니다"
            )
        elif hhi_value < 0.25:
            insights.append(
                f"집중도 지수가 {hhi_value:.2f}로 나타나 '{position.name}' 직군 내에서 산업별 채용이 일부 집중되어 있습니다"
            )
        else:
            insights.append(
                f"집중도 지수가 {hhi_value:.2f}로 나타나 '{position.name}' 직군 내에서 특정 산업에 채용이 과도하게 집중되어 있습니다"
            )

        # CR₂ 인사이트
        cr2_percentage = cr2_value * 100
        if industry_data_sorted and len(industry_data_sorted) >= 2:
            top1_name = industry_data_sorted[0][1]
            top2_name = industry_data_sorted[1][1]
            insights.append(
                f"상위 2개 산업({top1_name}, {top2_name})이 전체의 {cr2_percentage:.1f}%를 차지하며 주류를 이루고 있습니다"
            )

        # Entropy 인사이트 (다양성 지수로 표현)
        max_entropy = math.log2(len(industry_data)) if len(industry_data) > 0 else 1
        normalized_entropy = entropy_value / max_entropy if max_entropy > 0 else 0

        if normalized_entropy > 0.7:
            if len(industry_data_sorted) > 2:
                other_industries = ", ".join([ind[1] for ind in industry_data_sorted[2:4]])
                insights.append(
                    f"다양성 지수가 {normalized_entropy:.2f}로 나타나 {other_industries} 등의 분야로 다양화가 진행 중입니다"
                )
        else:
            insights.append(
                f"다양성 지수가 {normalized_entropy:.2f}로 나타나 산업 다양성이 낮아 일부 산업에 집중되어 있습니다"
            )

        # YoY 인사이트
        if yoy_score > 50:
            change_percentage = ((position_current / position_previous) - 1) * 100 if position_previous > 0 else 0
            insights.append(
                f"전년 대비 과열도 지수가 {yoy_score:.2f}로, '{position.name}' 직군의 채용이 작년보다 {change_percentage:.1f}% 증가하여 확대되는 추세입니다"
            )
        elif yoy_score == 50:
            insights.append(
                f"전년 대비 과열도 지수가 {yoy_score:.2f}로, '{position.name}' 직군의 채용이 작년과 동일한 수준을 유지하고 있습니다"
            )
        else:
            change_percentage = (1 - (position_current / position_previous)) * 100 if position_previous > 0 else 0
            insights.append(
                f"전년 대비 과열도 지수가 {yoy_score:.2f}로, '{position.name}' 직군의 채용이 작년보다 {change_percentage:.1f}% 감소하여 냉각되는 추세입니다"
            )

    return PositionAnalysisData(
        analysis_type="position",
        period=PeriodInfo(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")),
        position_id=position_id,
        position_name=position.name,
        total_posts=total_posts,
        hhi=round(hhi_value, 4),
        interpretation=HHIInterpretation(level=level, difficulty=difficulty),
        top_industries=top_industries,
        yoy_overheat_score=round(yoy_score, 2),
        yoy_trend=yoy_trend,
        yoy_current_count=position_current,
        yoy_previous_count=position_previous,
        insights=insights,
    )


def analyze_specific_industry(
    db: Session,
    start_date_str: str,
    position_id: int,
    industry_id: int,
    include_insights: bool = False,
) -> IndustryAnalysisData:
    """
    시나리오 3: 특정 산업 분석 (순위, 점유율, 대안 추천, HHI + YoY 통합)

    Args:
        db: DB 세션
        start_date_str: 시작일 (YYYY-MM-DD)
        position_id: 직군 ID
        industry_id: 산업 ID
        include_insights: 인사이트 생성 여부 (default: False)

    Returns:
        IndustryAnalysisData
    """
    from app.models.position import Position
    from app.models.industry import Industry

    # 기간 계산 (3개월 고정, 과거 3개월)
    # start_date_str은 실제로는 end_date (분석 종료일)
    start_date, end_date = calculate_3month_period(start_date_str)
    previous_start_date, previous_end_date = calculate_previous_year_period(start_date, end_date)

    # 직군/산업 정보 조회
    position = db.query(Position).filter(Position.id == position_id).first()
    if not position:
        raise ValueError(f"직군 ID {position_id}를 찾을 수 없습니다")

    industry = db.query(Industry).filter(Industry.id == industry_id).first()
    if not industry:
        raise ValueError(f"산업 ID {industry_id}를 찾을 수 없습니다")

    if industry.position_id != position_id:
        raise ValueError(f"산업 '{industry.name}'은 직군 '{position.name}'에 속하지 않습니다")

    # 해당 직군 내 산업별 데이터 조회
    industry_data = _get_industry_data_for_position(db, start_date, end_date, position_id)

    if not industry_data:
        raise ValueError(f"해당 직군({position.name})에 데이터가 없습니다")

    # 대상 산업 찾기
    industry_dict = {ind_id: (ind_name, count) for ind_id, ind_name, count in industry_data}

    if industry_id not in industry_dict:
        raise ValueError(f"해당 기간에 '{industry.name}' 산업의 채용 공고가 없습니다")

    target_name, target_count = industry_dict[industry_id]

    # 총 공고 수 및 순위 계산
    total_count = sum(count for _, count in industry_dict.values())
    sorted_industries = sorted(industry_data, key=lambda x: x[2], reverse=True)
    target_rank = next(
        idx + 1 for idx, (ind_id, _, _) in enumerate(sorted_industries) if ind_id == industry_id
    )
    target_share = (target_count / total_count * 100) if total_count > 0 else 0.0

    # 대안 산업 추천 (스킬 유사도 기반)
    alternatives = _get_skill_similarity_industries(
        db=db,
        target_industry_id=industry_id,
        position_id=position_id,
        start_date=start_date,
        end_date=end_date,
        limit=3,
    )

    # === YoY 계산 ===
    current_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=start_date,
        end_date=end_date,
        company_patterns=None,
    )
    previous_rows = db_yoy_overheat.get_recruit_counts_for_period(
        db=db,
        start_date=previous_start_date,
        end_date=previous_end_date,
        company_patterns=None,
    )

    # 해당 산업 데이터만 필터링
    industry_current = sum(
        count for _, _, ind_id, _, count in current_rows
        if ind_id == industry_id
    )
    industry_previous = sum(
        count for _, _, ind_id, _, count in previous_rows
        if ind_id == industry_id
    )

    yoy_score = _calculate_yoy_score(industry_current, industry_previous)
    yoy_trend = _get_trend(yoy_score)

    # === 인사이트 생성 (include_insights=true일 때만) ===
    insights = []
    if include_insights:
        # 순위 및 점유율 인사이트
        insights.append(
            f"'{position.name}' 직군 내에서 '{industry.name}' 산업은 {target_rank}위로 {target_share:.1f}%의 점유율을 차지하고 있습니다"
        )

        # 경쟁 난이도 인사이트
        if target_share > 30:
            insights.append(
                "높은 점유율로 인해 경쟁이 매우 치열할 수 있습니다"
            )
        elif target_share > 20:
            insights.append(
                "중간 수준의 점유율로 적절한 경쟁 환경입니다"
            )
        else:
            insights.append(
                "상대적으로 낮은 점유율로 경쟁 강도가 낮을 수 있습니다"
            )

        # 대안 추천 인사이트
        if alternatives:
            top_alt = alternatives[0]
            insights.append(
                f"대안으로 '{top_alt.industry_name}' (유사도 {top_alt.skill_similarity * 100:.0f}%, 점유율 {top_alt.share_percentage:.1f}%) 산업을 고려해보세요"
            )

        # YoY 인사이트
        if yoy_score > 50:
            change_percentage = ((industry_current / industry_previous) - 1) * 100 if industry_previous > 0 else 0
            insights.append(
                f"전년 대비 과열도 지수가 {yoy_score:.2f}로, '{industry.name}' 산업의 채용이 작년보다 {change_percentage:.1f}% 증가하여 확대되는 추세입니다"
            )
        elif yoy_score == 50:
            insights.append(
                f"전년 대비 과열도 지수가 {yoy_score:.2f}로, '{industry.name}' 산업의 채용이 작년과 동일한 수준을 유지하고 있습니다"
            )
        else:
            change_percentage = (1 - (industry_current / industry_previous)) * 100 if industry_previous > 0 else 0
            insights.append(
                f"전년 대비 과열도 지수가 {yoy_score:.2f}로, '{industry.name}' 산업의 채용이 작년보다 {change_percentage:.1f}% 감소하여 냉각되는 추세입니다"
            )

    return IndustryAnalysisData(
        analysis_type="industry",
        period=PeriodInfo(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")),
        position_id=position_id,
        position_name=position.name,
        industry_id=industry_id,
        industry_name=industry.name,
        posts_count=target_count,
        rank=target_rank,
        share_percentage=round(target_share, 2),
        alternative_industries=alternatives,
        yoy_overheat_score=round(yoy_score, 2),
        yoy_trend=yoy_trend,
        yoy_current_count=industry_current,
        yoy_previous_count=industry_previous,
        insights=insights,
    )


def analyze_combined_insights(
    db: Session,
    start_date_str: str,
    position_id: Optional[int] = None,
    industry_id: Optional[int] = None,
    include_insights: bool = False,
) -> CombinedIndustryAnalysisData:
    """
    통합 인사이트 분석 (Total + Position + Industry)
    
    항상 Total 인사이트를 포함하고, position_id와 industry_id가 있으면 해당 인사이트도 포함합니다.

    Args:
        db: DB 세션
        start_date_str: 시작일 (YYYY-MM-DD)
        position_id: 직군 ID (선택)
        industry_id: 산업 ID (선택, position_id가 필수)
        include_insights: 인사이트 생성 여부 (default: False)

    Returns:
        CombinedIndustryAnalysisData
    """
    # 1. Total 시장 분석 (항상 포함)
    total_insight = analyze_overall_market(
        db=db,
        start_date_str=start_date_str,
        include_insights=include_insights,
    )

    # 2. Position 분석 (position_id가 있으면 포함)
    position_insight = None
    if position_id:
        position_insight = analyze_position_industries(
            db=db,
            start_date_str=start_date_str,
            position_id=position_id,
            include_insights=include_insights,
        )

    # 3. Industry 분석 (industry_id가 있으면 포함)
    industry_insight = None
    if industry_id and position_id:
        industry_insight = analyze_specific_industry(
            db=db,
            start_date_str=start_date_str,
            position_id=position_id,
            industry_id=industry_id,
            include_insights=include_insights,
        )

    return CombinedIndustryAnalysisData(
        analysis_type="combined",
        period=total_insight.period,  # 모든 분석이 동일한 기간 사용
        total_insight=total_insight,
        position_insight=position_insight,
        industry_insight=industry_insight,
    )
