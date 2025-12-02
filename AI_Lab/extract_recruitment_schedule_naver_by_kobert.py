"""
채용 일정 추출 스크립트 - 네이버 전용 (KoBERT 버전)

네이버 회사의 Post에서 채용 일정 정보를 KoBERT로 추출하여 JSON으로 반환
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from calendar import monthrange
from dotenv import load_dotenv

# 프로젝트 루트를 Python 경로에 추가
current_file = Path(__file__).resolve()
backend_model_dir = current_file.parents[1]
if str(backend_model_dir) not in sys.path:
    sys.path.append(str(backend_model_dir))

from sqlalchemy.orm import Session, joinedload
from app.db.config.base import SessionLocal, Base
from app.models.post import Post
from app.models.company import Company
from app.models.industry import Industry
from app.models.post_skill import PostSkill
from app.models.position import Position
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill
from app.models.skill import Skill

# 모든 mapper가 초기화되도록 Base.metadata 참조
_ = Base.metadata

# KoBERT 관련 import
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    KOBERT_AVAILABLE = True
except ImportError:
    KOBERT_AVAILABLE = False
    print("경고: transformers, torch, sklearn이 설치되지 않았습니다. pip install transformers torch scikit-learn")

load_dotenv()

# 전역 변수에 실패 플래그 추가
_kobert_tokenizer = None
_kobert_model = None
_kobert_load_failed = False  # 추가

def load_kobert():
    """KoBERT 모델 로드 (한 번만 실행)"""
    global _kobert_tokenizer, _kobert_model, _kobert_load_failed
    
    if not KOBERT_AVAILABLE:
        return None, None
    
    # 이미 실패한 경우 재시도하지 않음
    if _kobert_load_failed:
        return None, None
    
    if _kobert_tokenizer is None or _kobert_model is None:
        try:
            model_name = "monologg/kobert"
            _kobert_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)  # 수정
            _kobert_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            _kobert_model.eval()
            print(f"KoBERT 모델 로드 완료: {model_name}")
        except Exception as e:
            print(f"KoBERT 모델 로드 실패: {e}")
            _kobert_load_failed = True  # 실패 플래그 설정
            return None, None
    
    return _kobert_tokenizer, _kobert_model

# ============================================================================
# 전처리 함수
# ============================================================================

def extract_schedule_section(description: str) -> str:
    """description에서 채용 일정 관련 섹션만 추출"""
    if not description:
        return ""
    
    schedule_keywords = [
        r'\d{4}[년.-]\s*\d{1,2}[월.-]\s*\d{1,2}일?',
        r'\d{1,2}[월/.-]\s*\d{1,2}일?',
        r'\d{4}\.\d{1,2}\.\d{1,2}',
        '모집.*?기간', '접수.*?기간', '지원.*?기간', '마감일?', '접수.*?마감',
        '서류.*?전형', '서류.*?심사', '서류.*?평가',
        '인적성', '코딩.*?테스트', '코테',
        '1차.*?면접', '2차.*?면접', '최종.*?면접', '면접.*?일정',
        '입사일?', '입사.*?예정', '근무.*?시작',
        '상반기', '하반기',
        '채용.*?프로세스', '전형.*?절차', '채용.*?절차', '선발.*?절차',
        '첫째.*?주', '둘째.*?주', '셋째.*?주', '넷째.*?주',
        '초', '중순', '하순', '말',
    ]
    
    section_headers = [
        '모집요강', '채용절차', '전형절차', '채용프로세스', '선발프로세스',
        '모집.*?사항', '채용.*?일정', '전형.*?일정', '지원.*?방법',
        '전형.*?단계', '채용.*?단계'
    ]
    
    relevant_sections = []
    for header_pattern in section_headers:
        pattern = f'({header_pattern}).*?(?=\n\n[가-힣A-Z]{{2,}}|$)'
        matches = re.finditer(pattern, description, re.IGNORECASE | re.DOTALL)
        for match in matches:
            section_text = match.group(0)
            if len(section_text) > 10:
                relevant_sections.append(section_text)
    
    keyword_contexts = []
    for keyword_pattern in schedule_keywords:
        matches = re.finditer(keyword_pattern, description, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - 200)
            end = min(len(description), match.end() + 200)
            context = description[start:end]
            if context not in keyword_contexts:
                keyword_contexts.append(context)
    
    combined_text = '\n\n'.join(relevant_sections + keyword_contexts)
    
    if len(combined_text) < 100:
        combined_text = description[:2000]
    
    if len(combined_text) > 3000:
        combined_text = combined_text[:3000]
    
    return combined_text

# ============================================================================
# 추론 함수
# ============================================================================

def infer_semester_from_date(date: Optional[datetime]) -> Optional[str]:
    """날짜 기준으로 상/하반기 판단"""
    if not date:
        return None
    
    month = date.month
    if 2 <= month <= 7:
        return "상반기"
    elif month >= 8 or month == 1:
        return "하반기"
    return None

def infer_application_period(posted_at: Optional[datetime], close_at: Optional[datetime]) -> List[str]:
    """posted_at과 close_at 기반으로 지원 기간 추론"""
    dates = []
    
    if posted_at:
        dates.append(posted_at.strftime("%Y-%m-%d"))
    
    if close_at:
        dates.append(close_at.strftime("%Y-%m-%d"))
    
    return sorted(list(set(dates)))

# ============================================================================
# 날짜 파싱 함수 (정규식 기반)
# ============================================================================

def parse_date_string(date_str: str, base_date: datetime) -> Optional[str]:
    """날짜 문자열을 YYYY-MM-DD 형식으로 변환"""
    date_str = date_str.strip()
    current_year = base_date.year
    
    # "2025.12.01" 형식
    match = re.match(r'(\d{4})\.(\d{1,2})\.(\d{1,2})', date_str)
    if match:
        year, month, day = map(int, match.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except:
            return None
    
    # "12.01" 또는 "12.1" 형식
    match = re.match(r'(\d{1,2})\.(\d{1,2})', date_str)
    if match:
        month, day = map(int, match.groups())
        try:
            return datetime(current_year, month, day).strftime("%Y-%m-%d")
        except:
            return None
    
    # "2025년 12월 1일" 형식
    match = re.match(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', date_str)
    if match:
        year, month, day = map(int, match.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except:
            return None
    
    # "12월 1일" 형식
    match = re.match(r'(\d{1,2})월\s*(\d{1,2})일', date_str)
    if match:
        month, day = map(int, match.groups())
        try:
            return datetime(current_year, month, day).strftime("%Y-%m-%d")
        except:
            return None
    
    return None

def parse_date_range(date_str: str, base_date: datetime) -> List[str]:
    """날짜 범위를 파싱 (예: "3.5 ~ 3.18", "2026.01.05 ~ 2026.02.27")"""
    # "~" 또는 "-"로 구분된 범위
    parts = re.split(r'[~\-]\s*', date_str)
    if len(parts) == 2:
        start_date = parse_date_string(parts[0].strip(), base_date)
        end_date = parse_date_string(parts[1].strip(), base_date)
        if start_date and end_date:
            return [start_date, end_date]
        elif start_date:
            return [start_date]
    
    # 단일 날짜
    single_date = parse_date_string(date_str, base_date)
    if single_date:
        return [single_date]
    
    return []

def parse_week_expression(week_str: str, base_date: datetime) -> List[str]:
    """주차 표현을 날짜 범위로 변환 (예: "12월 1주차" → ["2025-12-01", "2025-12-07"])"""
    current_year = base_date.year
    
    # "12월 1주차" 또는 "12월 첫째주"
    match = re.search(r'(\d{1,2})월\s*(첫째|1|둘째|2|셋째|3|넷째|4|다섯째|5)주', week_str)
    if match:
        month = int(match.group(1))
        week_num = match.group(2)
        
        # 주차 번호 변환
        week_map = {"첫째": 1, "둘째": 2, "셋째": 3, "넷째": 4, "다섯째": 5}
        week = week_map.get(week_num, int(week_num) if week_num.isdigit() else 1)
        
        # 해당 월의 첫 번째 날
        first_day = datetime(current_year, month, 1)
        # 첫 번째 월요일 찾기
        days_until_monday = (7 - first_day.weekday()) % 7
        if first_day.weekday() == 0:  # 월요일이면
            days_until_monday = 0
        
        # 해당 주차의 월요일
        week_start = first_day + timedelta(days=days_until_monday + (week - 1) * 7)
        week_end = week_start + timedelta(days=6)  # 일요일
        
        return [week_start.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d")]
    
    return []

def parse_period_expression(period_str: str, base_date: datetime) -> List[str]:
    """순 표현을 날짜 범위로 변환 (예: "3월 초" → ["2025-03-01", "2025-03-10"])"""
    current_year = base_date.year
    
    match = re.search(r'(\d{1,2})월\s*(초|중순|하순|말)', period_str)
    if match:
        month = int(match.group(1))
        period = match.group(2)
        
        days_in_month = monthrange(current_year, month)[1]
        
        if period == "초":
            return [datetime(current_year, month, 1).strftime("%Y-%m-%d"),
                   datetime(current_year, month, 10).strftime("%Y-%m-%d")]
        elif period == "중순":
            return [datetime(current_year, month, 11).strftime("%Y-%m-%d"),
                   datetime(current_year, month, 20).strftime("%Y-%m-%d")]
        elif period == "하순":
            return [datetime(current_year, month, 21).strftime("%Y-%m-%d"),
                   datetime(current_year, month, days_in_month).strftime("%Y-%m-%d")]
        elif period == "말":
            return [datetime(current_year, month, 25).strftime("%Y-%m-%d"),
                   datetime(current_year, month, days_in_month).strftime("%Y-%m-%d")]
    
    return []

def extract_all_dates(text: str, base_date: datetime) -> List[Tuple[str, int, int]]:
    """텍스트에서 모든 날짜 패턴 추출 (날짜 문자열, 시작 위치, 끝 위치)"""
    dates = []
    
    # 명확한 날짜 패턴들
    patterns = [
        (r'\d{4}\.\d{1,2}\.\d{1,2}', 'explicit'),
        (r'\d{1,2}\.\d{1,2}', 'month_day'),
        (r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일', 'korean_full'),
        (r'\d{1,2}월\s*\d{1,2}일', 'korean_partial'),
        (r'\d{1,2}월\s*(첫째|둘째|셋째|넷째|다섯째|\d{1,2})주', 'week'),
        (r'\d{1,2}월\s*(초|중순|하순|말)', 'period'),
    ]
    
    for pattern, pattern_type in patterns:
        for match in re.finditer(pattern, text):
            date_str = match.group(0)
            dates.append((date_str, match.start(), match.end(), pattern_type))
    
    return dates

# ============================================================================
# KoBERT 기반 컨텍스트 매칭
# ============================================================================

def get_text_embedding(text: str, tokenizer, model) -> Optional[np.ndarray]:
    """텍스트의 KoBERT 임베딩 벡터 반환"""
    if tokenizer is None or model is None:
        return None
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            # [CLS] 토큰의 임베딩 사용
            embedding = outputs.last_hidden_state[0][0].numpy()
        return embedding
    except Exception as e:
        print(f"  [경고] 임베딩 생성 실패: {e}")
        return None

def match_date_to_field(date_context: str, field_keywords: List[str], tokenizer, model) -> float:
    """날짜 컨텍스트와 필드 키워드의 유사도 계산"""
    if tokenizer is None or model is None:
        # KoBERT 없으면 키워드 매칭으로 대체
        for keyword in field_keywords:
            if keyword in date_context:
                return 0.8
        return 0.0
    
    date_embedding = get_text_embedding(date_context, tokenizer, model)
    if date_embedding is None:
        return 0.0
    
    # 각 키워드와의 유사도 계산
    similarities = []
    for keyword in field_keywords:
        keyword_embedding = get_text_embedding(keyword, tokenizer, model)
        if keyword_embedding is not None:
            similarity = cosine_similarity([date_embedding], [keyword_embedding])[0][0]
            similarities.append(similarity)
    
    return max(similarities) if similarities else 0.0

# 필드별 키워드 정의
FIELD_KEYWORDS = {
    "application_deadline": [
        "지원서 접수 마감", "지원 마감", "접수 마감", "모집 마감",
        "지원서 마감", "접수 기간", "지원 기간", "모집 기간"
    ],
    "document_screening": [
        "서류 전형", "서류 심사", "서류 평가", "인적성 검사",
        "코딩테스트", "코테", "서류전형", "서류심사"
    ],
    "first_interview": [
        "1차 면접", "직무 면접", "직무 인터뷰", "실무 면접",
        "기술 면접", "기술 역량 인터뷰", "1차 인터뷰"
    ],
    "second_interview": [
        "2차 면접", "임원 면접", "최종 면접", "종합 역량 인터뷰",
        "2차 인터뷰", "최종 인터뷰"
    ],
    "join_date": [
        "입사일", "입사 예정일", "근무 시작일", "인턴십 수행",
        "인턴십 기간", "인턴십 시작", "입사"
    ]
}

# ============================================================================
# KoBERT 기반 추출 함수
# ============================================================================

def extract_schedule_with_kobert(
    description: str,
    posted_at: Optional[datetime],
    close_at: Optional[datetime],
    crawled_at: datetime,
    post_id: int,
    company_id: int,
    company_name: str,
    industry_id: Optional[int]
) -> Optional[Dict[str, Any]]:
    """KoBERT로 채용 일정 추출"""
    
    # 1. 추론 로직 먼저 실행
    inferred_semester = infer_semester_from_date(crawled_at)
    inferred_application = infer_application_period(posted_at, close_at)
    
    print(f"  [추론] semester: {inferred_semester}")
    print(f"  [추론] application_date: {inferred_application}")
    
    # 2. KoBERT 모델 로드
    tokenizer, model = load_kobert()
    if tokenizer is None or model is None:
        print(f"  [경고] KoBERT 모델을 사용할 수 없습니다. 추론 결과만 사용합니다.")
        return {
            "post_id": post_id,
            "company_id": company_id,
            "semester": inferred_semester,
            "industry_id": industry_id,
            "application_date": inferred_application,
            "document_screening_date": [],
            "first_interview": [],
            "second_interview": [],
            "join_date": [],
        }
    
    # 3. 전처리
    schedule_text = extract_schedule_section(description)
    base_date = posted_at if posted_at else crawled_at
    
    # 4. semester 추출
    semester = None
    if re.search(r'상반기', schedule_text):
        semester = "상반기"
    elif re.search(r'하반기', schedule_text):
        semester = "하반기"
    semester = semester or inferred_semester
    
    # 5. 모든 날짜 패턴 추출
    all_dates = extract_all_dates(schedule_text, base_date)
    
    # 6. 각 날짜를 필드에 매칭
    field_dates = {
        "application_deadline": [],
        "document_screening": [],
        "first_interview": [],
        "second_interview": [],
        "join_date": []
    }
    
    for date_str, start_pos, end_pos, pattern_type in all_dates:
        # 날짜 주변 컨텍스트 추출 (앞뒤 100자)
        context_start = max(0, start_pos - 100)
        context_end = min(len(schedule_text), end_pos + 100)
        context = schedule_text[context_start:context_end]
        
        # 날짜 변환
        parsed_dates = []
        if pattern_type == 'week':
            parsed_dates = parse_week_expression(date_str, base_date)
        elif pattern_type == 'period':
            parsed_dates = parse_period_expression(date_str, base_date)
        else:
            # 범위 처리
            if '~' in date_str or '-' in date_str:
                parsed_dates = parse_date_range(date_str, base_date)
            else:
                single_date = parse_date_string(date_str, base_date)
                if single_date:
                    parsed_dates = [single_date]
        
        if not parsed_dates:
            continue
        
        # 각 필드와의 유사도 계산
        field_scores = {}
        for field_name, keywords in FIELD_KEYWORDS.items():
            score = match_date_to_field(context, keywords, tokenizer, model)
            field_scores[field_name] = score
        
        # 가장 높은 점수의 필드에 할당 (임계값 0.3 이상)
        best_field = max(field_scores.items(), key=lambda x: x[1])
        if best_field[1] >= 0.3:
            field_dates[best_field[0]].extend(parsed_dates)
    
    # 7. 중복 제거 및 정렬
    for field_name in field_dates:
        field_dates[field_name] = sorted(list(set(field_dates[field_name])))
    
    # 8. application_date는 추론 결과 우선
    application_date = field_dates["application_deadline"] if field_dates["application_deadline"] else inferred_application
    
    # 9. second_interview는 명시적으로 언급된 경우만
    if not re.search(r'2차.*?면접|임원.*?면접|최종.*?면접|종합.*?인터뷰', schedule_text, re.IGNORECASE):
        field_dates["second_interview"] = []
    
    result = {
        "post_id": post_id,
        "company_id": company_id,
        "semester": semester,
        "industry_id": industry_id,
        "application_date": application_date,
        "document_screening_date": field_dates["document_screening"],
        "first_interview": field_dates["first_interview"],
        "second_interview": field_dates["second_interview"],
        "join_date": field_dates["join_date"],
    }
    
    print(f"  [KoBERT] 추출 완료")
    print(f"    - 접수: {len(result['application_date'])}개, "
          f"서류: {len(result['document_screening_date'])}개, "
          f"1차: {len(result['first_interview'])}개, "
          f"2차: {len(result['second_interview'])}개, "
          f"입사: {len(result['join_date'])}개")
    
    return result

# ============================================================================
# 데이터 수집 함수 (기존과 동일)
# ============================================================================

def get_naver_companies(db: Session) -> List[Company]:
    """네이버 관련 모든 회사 조회"""
    try:
        from app.db.crud.db_competitors_skills import COMPETITOR_GROUPS
        from sqlalchemy import or_
        
        if "네이버" in COMPETITOR_GROUPS:
            keywords = COMPETITOR_GROUPS["네이버"]
            or_conditions = []
            for keyword in keywords:
                pattern = keyword.replace("%", "")
                or_conditions.append(Company.name.like(f"{pattern}%"))
            
            if or_conditions:
                companies = db.query(Company).filter(or_(*or_conditions)).distinct().all()
                if companies:
                    return companies
        
        companies = db.query(Company).filter(
            or_(
                Company.name.like("네이버%"),
                Company.name.like("NAVER%")
            )
        ).distinct().all()
        
        return companies
        
    except ImportError:
        from sqlalchemy import or_
        companies = db.query(Company).filter(
            or_(
                Company.name.like("네이버%"),
                Company.name.like("NAVER%")
            )
        ).distinct().all()
        
        return companies

def get_posts_by_company(db: Session, company_id: int, limit: int = 100) -> List[Post]:
    """회사별 Post 조회"""
    posts = (
        db.query(Post)
        .options(joinedload(Post.company))
        .filter(
            Post.company_id == company_id,
            Post.description.isnot(None),
            Post.description != ""
        )
        .order_by(Post.crawled_at.desc())
        .limit(limit)
        .all()
    )
    return posts

def extract_recruitment_schedules(db: Session, limit_per_company: int = 100) -> List[Dict[str, Any]]:
    """네이버 Post에서 채용 일정 추출 (KoBERT 버전)"""
    print("="*80)
    print("네이버 채용 일정 추출 시작 (KoBERT 버전)")
    print("="*80)
    
    print("\n[1/3] 네이버 회사 조회 중...")
    companies = get_naver_companies(db)
    
    if not companies:
        print("  네이버 회사를 찾을 수 없습니다.")
        return []
    
    print(f"  조회된 네이버 회사: {len(companies)}개")
    for company in companies:
        print(f"    - {company.name} (ID: {company.id})")
    
    print(f"\n[2/3] 각 네이버 회사별 Post 조회 중... (회사당 최대 {limit_per_company}개)")
    all_posts = []
    for company in companies:
        posts = get_posts_by_company(db, company.id, limit_per_company)
        all_posts.extend(posts)
        print(f"  {company.name}: {len(posts)}개")
    
    print(f"\n  총 {len(all_posts)}개의 Post 조회 완료")
    
    if not all_posts:
        print("  조회된 Post가 없습니다.")
        return []
    
    # Post 데이터를 딕셔너리로 변환 (DB 세션 닫기 전에)
    print("\n  [Post 데이터 변환 중...]")
    posts_data = []
    for post in all_posts:
        posts_data.append({
            "id": post.id,
            "title": post.title,
            "description": post.description,
            "posted_at": post.posted_at,
            "close_at": post.close_at,
            "crawled_at": post.crawled_at,
            "company_id": post.company_id,
            "company_name": post.company.name if post.company else "Unknown",
            "industry_id": post.industry_id
        })
    
    # DB 세션 닫기 (다른 사용자 접근 가능하도록)
    db.close()
    print("  [DB 세션 닫기 완료 - 다른 사용자 접근 가능]")
    
    # 3. KoBERT로 일정 추출
    print(f"\n[3/3] KoBERT로 채용 일정 추출 중... (총 {len(posts_data)}개)")
    results = []
    
    for idx, post_data in enumerate(posts_data, 1):
        print(f"\n[{idx}/{len(posts_data)}] Post ID: {post_data['id']}")
        print(f"  제목: {post_data['title'][:50]}...")
        print(f"  회사: {post_data['company_name']}")
        
        extracted = extract_schedule_with_kobert(
            description=post_data['description'],
            posted_at=post_data['posted_at'],
            close_at=post_data['close_at'],
            crawled_at=post_data['crawled_at'],
            post_id=post_data['id'],
            company_id=post_data['company_id'],
            company_name=post_data['company_name'],
            industry_id=post_data['industry_id']
        )
        
        if extracted:
            results.append(extracted)
    
    print(f"\n{'='*80}")
    print(f"추출 완료: {len(results)}/{len(posts_data)}개 성공")
    print(f"{'='*80}")
    
    return results

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    db = SessionLocal()
    
    try:
        results = extract_recruitment_schedules(db, limit_per_company=100)
        
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"recruitment_schedules_naver_kobert_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과 저장: {output_file}")
        print(f"총 {len(results)}개 추출 완료")
        
        if results:
            print("\n[통계]")
            
            has_semester = sum(1 for r in results if r.get('semester'))
            has_application = sum(1 for r in results if r.get('application_date'))
            has_dates = sum(1 for r in results if any([
                r.get('document_screening_date'),
                r.get('first_interview'),
                r.get('second_interview'),
                r.get('join_date')
            ]))
            
            print(f"  전체 {len(results)}개")
            print(f"  - semester 있음: {has_semester}개 ({has_semester/len(results)*100:.1f}%)")
            print(f"  - application_date 있음: {has_application}개 ({has_application/len(results)*100:.1f}%)")
            print(f"  - 기타 날짜 있음: {has_dates}개 ({has_dates/len(results)*100:.1f}%)")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if db.is_active:
                db.close()
        except:
            pass

if __name__ == "__main__":
    main()