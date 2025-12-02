"""
채용 일정 추출 스크립트 - 날짜가 있는 문장만 추출

경쟁사 9개 회사의 Post에서 description에 날짜가 있는 문장만 LLM으로 추출하여 JSON으로 반환
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

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

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

debug_count = 0

# ============================================================================
# LLM 추출 함수 - 날짜가 있는 문장만 추출
# ============================================================================

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def extract_date_sentences_with_llm(
    description: str,
    post_id: int,
    company_name: str
) -> Optional[List[str]]:
    """LLM으로 description에서 날짜가 있는 문장만 추출"""
    
    if not description:
        return []
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"  [경고] OpenAI API 키가 없습니다.")
        return []
    
    if OpenAI is None:
        print(f"  [경고] OpenAI 모듈을 불러올 수 없습니다.")
        return []
    
    system_prompt = """당신은 한국어 텍스트에서 날짜가 포함된 문장만 추출하는 전문가입니다.

# 작업
주어진 텍스트에서 날짜 표현이 포함된 문장만 추출하세요.

# 날짜 표현 예시
- 명확한 날짜: "2025년 3월 15일", "3월 15일", "2025.03.15", "3.5", "3/15"
- 주차 표현: "12월 첫째주", "4월 3째주", "12월 1주차"
- 순 표현: "3월 초", "7월 중순", "12월 하순", "4월 말"
- 기간 표현: "3/1~3/15", "3월 1일 - 3월 15일", "2026.01.05 ~ 2026.02.27"
- 상대적 표현: "내일", "다음 주", "이번 달" (게시일 기준으로 해석)

# 추출 규칙
1. 날짜 표현이 포함된 문장만 추출
2. 문장 단위로 추출 (마침표, 줄바꿈 기준)
3. 중복 제거
4. 날짜가 없는 문장은 제외
5. 원문 그대로 보존 (날짜 형식 변환하지 않음)

# 출력 형식
JSON 배열로 문장 리스트 반환:
{
    "sentences": [
        "지원서 접수 마감: 2025.12.01(월) 10:00",
        "서류 전형 및 코딩테스트: 12월 1주차 ~ 12월 2주차 중",
        "직무 인터뷰: 12월 2주차 ~ 3주차 중"
    ]
}"""
    
    user_prompt = f"""# 채용 공고 정보

**회사명**: {company_name}
**Post ID**: {post_id}

**채용 공고 내용**:
{description[:5000]}  # 최대 5000자로 제한

---

위 내용에서 날짜 표현이 포함된 문장만 추출하여 JSON 배열로 반환하세요."""
    
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000,
        )
        
        content = response.choices[0].message.content if response and response.choices else "{}"
        
        try:
            extracted_data = json.loads(content)
            sentences = extracted_data.get("sentences", [])
            
            global debug_count
            if debug_count < 3:
                print(f"\n  [DEBUG] 추출된 문장 수: {len(sentences)}")
                for i, sentence in enumerate(sentences[:3], 1):
                    print(f"    {i}. {sentence[:100]}...")
                debug_count += 1
            
            return sentences if isinstance(sentences, list) else []
                
        except json.JSONDecodeError as e:
            print(f"  [에러] JSON 파싱 실패: {e}")
            print(f"  [에러] 응답 내용: {content[:500]}")
            return []
        
    except Exception as e:
        print(f"  [에러] LLM 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        if client:
            client.close()

# ============================================================================
# 데이터 수집 함수
# ============================================================================

def get_competitor_companies(db: Session, company_filter: Optional[str] = None) -> List[Company]:
    """경쟁사 9개 회사 조회 (회사 이름 필터링 옵션)"""
    try:
        from app.db.crud.db_competitors_skills import COMPETITOR_GROUPS
        from sqlalchemy import or_
        
        # 회사 이름 필터가 있으면 해당 회사만 조회
        if company_filter and company_filter.strip():
            company_filter = company_filter.strip()
            print(f"  [필터] 회사 이름: '{company_filter}'")
            
            # 부분 일치로 검색
            companies = db.query(Company).filter(
                Company.name.like(f"%{company_filter}%")
            ).distinct().all()
            
            if companies:
                print(f"  조회된 회사: {len(companies)}개")
                for company in companies:
                    print(f"    - {company.name} (ID: {company.id})")
                return companies
            else:
                print(f"  [경고] '{company_filter}'와 일치하는 회사를 찾을 수 없습니다.")
                print(f"  전체 경쟁사 목록을 조회합니다...")
        
        # 필터가 없거나 매칭 실패 시 전체 경쟁사 조회
        all_companies = []
        or_conditions = []
        
        # 9개 경쟁사 그룹 모두 처리
        for group_name, keywords in COMPETITOR_GROUPS.items():
            for keyword in keywords:
                pattern = keyword.replace("%", "")
                or_conditions.append(Company.name.like(f"{pattern}%"))
        
        if or_conditions:
            companies = db.query(Company).filter(or_(*or_conditions)).distinct().all()
            return companies
        
        # COMPETITOR_GROUPS가 없으면 직접 검색 (fallback)
        companies = db.query(Company).filter(
            or_(
                Company.name.like("토스%"),
                Company.name.like("카카오%"),
                Company.name.like("한화시스템%"),
                Company.name.like("현대오토에버%"),
                Company.name.like("우아한%"),
                Company.name.like("쿠팡%"),
                Company.name.like("라인%"),
                Company.name.like("네이버%"),
                Company.name.like("NAVER%"),
                Company.name.like("LG_CNS%"),
                Company.name.like("LG CNS%"),
            )
        ).distinct().all()
        
        return companies
        
    except ImportError:
        # COMPETITOR_GROUPS를 불러올 수 없으면 직접 검색
        from sqlalchemy import or_
        
        # 회사 이름 필터가 있으면 해당 회사만 조회
        if company_filter and company_filter.strip():
            company_filter = company_filter.strip()
            companies = db.query(Company).filter(
                Company.name.like(f"%{company_filter}%")
            ).distinct().all()
            if companies:
                return companies
        
        companies = db.query(Company).filter(
            or_(
                Company.name.like("토스%"),
                Company.name.like("카카오%"),
                Company.name.like("한화시스템%"),
                Company.name.like("현대오토에버%"),
                Company.name.like("우아한%"),
                Company.name.like("쿠팡%"),
                Company.name.like("라인%"),
                Company.name.like("네이버%"),
                Company.name.like("NAVER%"),
                Company.name.like("LG_CNS%"),
                Company.name.like("LG CNS%"),
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

def extract_recruitment_schedules(db: Session, limit_per_company: int = 100, company_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """경쟁사 9개 회사 Post에서 날짜가 있는 문장만 추출"""
    print("="*80)
    print("경쟁사 9개 회사 - 날짜가 있는 문장 추출 시작")
    if company_filter:
        print(f"회사 필터: {company_filter}")
    print("="*80)
    
    print("\n[1/3] 경쟁사 회사 조회 중...")
    companies = get_competitor_companies(db, company_filter=company_filter)  # 필터 파라미터 추가
    
    if not companies:
        print("  경쟁사 회사를 찾을 수 없습니다.")
        return []
    
    print(f"  조회된 경쟁사 회사: {len(companies)}개")
    
    # 회사 그룹별로 분류하여 출력
    from collections import defaultdict
    company_groups = defaultdict(list)
    for company in companies:
        name = company.name
        if "토스" in name or "비바리퍼블리카" in name:
            company_groups["토스"].append(company)
        elif "카카오" in name:
            company_groups["카카오"].append(company)
        elif "한화시스템" in name:
            company_groups["한화시스템"].append(company)
        elif "현대오토에버" in name:
            company_groups["현대오토에버"].append(company)
        elif "우아한" in name or "배달의민족" in name or "배민" in name:
            company_groups["우아한형제들"].append(company)
        elif "쿠팡" in name or "Coupang" in name.upper():
            company_groups["쿠팡"].append(company)
        elif "라인" in name or "LINE" in name.upper():
            company_groups["라인"].append(company)
        elif "네이버" in name or "NAVER" in name.upper():
            company_groups["네이버"].append(company)
        elif "LG" in name.upper() and ("CNS" in name.upper() or "_" in name):
            company_groups["LG CNS"].append(company)
    
    for group_name, group_companies in sorted(company_groups.items()):
        print(f"    [{group_name}] {len(group_companies)}개 회사")
        for company in group_companies[:3]:
            print(f"      - {company.name} (ID: {company.id})")
        if len(group_companies) > 3:
            print(f"      ... 외 {len(group_companies) - 3}개")
    
    print(f"\n[2/3] 각 경쟁사 회사별 Post 조회 중... (회사당 최대 {limit_per_company}개)")
    all_posts = []
    for company in companies:
        posts = get_posts_by_company(db, company.id, limit_per_company)
        all_posts.extend(posts)
        if posts:
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
            "industry_id": post.industry_id,
            # 추가 날짜 정보 (created_at 등이 있다면)
            # "created_at": post.created_at if hasattr(post, 'created_at') else None,
        })
    
    # DB 세션 닫기 (다른 사용자 접근 가능하도록)
    db.close()
    print("  [DB 세션 닫기 완료 - 다른 사용자 접근 가능]")
    
    # 3. LLM으로 날짜가 있는 문장 추출 (DB 세션 없이)
    print(f"\n[3/3] LLM으로 날짜가 있는 문장 추출 중... (총 {len(posts_data)}개)")
    results = []
    
    for idx, post_data in enumerate(posts_data, 1):
        print(f"\n[{idx}/{len(posts_data)}] Post ID: {post_data['id']}")
        print(f"  제목: {post_data['title'][:50]}...")
        print(f"  회사: {post_data['company_name']}")
        
        date_sentences = extract_date_sentences_with_llm(
            description=post_data['description'],
            post_id=post_data['id'],
            company_name=post_data['company_name']
        )
        
        result = {
            "post_id": post_data['id'],
            "company_id": post_data['company_id'],
            "company_name": post_data['company_name'],
            "title": post_data['title'],
            "description": post_data['description'],  # 원본 description 그대로
            "posted_at": post_data['posted_at'].isoformat() if post_data['posted_at'] else None,
            "close_at": post_data['close_at'].isoformat() if post_data['close_at'] else None,
            "crawled_at": post_data['crawled_at'].isoformat() if post_data['crawled_at'] else None,
            "industry_id": post_data['industry_id'],
            "date_sentences": date_sentences if date_sentences else [],  # 날짜가 있는 문장만
        }
        
        results.append(result)
        
        print(f"  ✓ 추출 완료 (날짜 문장 {len(date_sentences)}개)")
        if date_sentences:
            print(f"    예시: {date_sentences[0][:80]}...")
    
    print(f"\n{'='*80}")
    print(f"추출 완료: {len(results)}/{len(posts_data)}개 성공")
    print(f"{'='*80}")
    
    return results

# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    # 회사 이름 입력 받기
    print("="*80)
    print("채용 일정 추출 - 날짜가 있는 문장만 추출")
    print("="*80)
    print("\n회사 이름을 입력하세요 (전체 경쟁사 조회하려면 Enter):")
    print("  예시: 네이버, NAVER, 쿠팡, 카카오 등")
    print("  (빈 입력 시 전체 경쟁사 9개 회사 처리)")
    
    company_filter = input("\n회사 이름: ").strip()
    
    if company_filter:
        print(f"\n'{company_filter}' 회사만 처리합니다.")
    else:
        print("\n전체 경쟁사 9개 회사를 처리합니다.")
    
    db = SessionLocal()
    
    try:
        results = extract_recruitment_schedules(db, limit_per_company=100, company_filter=company_filter if company_filter else None)
        
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filter_suffix = f"_{company_filter.replace(' ', '_')}" if company_filter else "_all"
        output_file = output_dir / f"recruitment_schedules_date_sentences{filter_suffix}_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과 저장: {output_file}")
        print(f"총 {len(results)}개 추출 완료")
        
        if results:
            print("\n[통계]")
            
            has_date_sentences = sum(1 for r in results if r.get('date_sentences'))
            total_sentences = sum(len(r.get('date_sentences', [])) for r in results)
            
            print(f"  전체 {len(results)}개")
            print(f"  - 날짜 문장 추출 성공: {has_date_sentences}개 ({has_date_sentences/len(results)*100:.1f}%)")
            print(f"  - 총 추출된 문장 수: {total_sentences}개")
            print(f"  - 평균 문장 수: {total_sentences/has_date_sentences if has_date_sentences > 0 else 0:.1f}개/Post")
        
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
