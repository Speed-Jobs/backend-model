"""
채용 일정 추출 스크립트 v4

상대적 날짜 표현도 변환하는 버전
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
# LLM 추출 함수
# ============================================================================

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def extract_schedule_with_llm(
    description: str,
    posted_at: Optional[datetime],
    close_at: Optional[datetime],
    crawled_at: datetime,
    post_id: int,
    company_id: int,
    company_name: str,
    industry_id: Optional[int]
) -> Optional[Dict[str, Any]]:
    """LLM으로 채용 일정 추출"""
    
    # 1. 추론 로직 먼저 실행
    inferred_semester = infer_semester_from_date(crawled_at)
    inferred_application = infer_application_period(posted_at, close_at)
    
    print(f"  [추론] semester: {inferred_semester}")
    print(f"  [추론] application_deadline: {inferred_application}")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"  [경고] OpenAI API 키가 없습니다. 추론 결과만 사용합니다.")
        return {
            "post_id": post_id,
            "company_id": company_id,
            "semester": inferred_semester,
            "industry_id": industry_id,
            "application_deadline": inferred_application,
            "document_screening": [],
            "first_interview": [],
            "second_interview": [],
            "join_date": [],
        }
    
    if OpenAI is None:
        print(f"  [경고] OpenAI 모듈을 불러올 수 없습니다. 추론 결과만 사용합니다.")
        return {
            "post_id": post_id,
            "company_id": company_id,
            "semester": inferred_semester,
            "industry_id": industry_id,
            "application_deadline": inferred_application,
            "document_screening": [],
            "first_interview": [],
            "second_interview": [],
            "join_date": [],
        }
    
    # 전처리
    schedule_text = extract_schedule_section(description)
    
    posted_at_str = posted_at.strftime("%Y-%m-%d") if posted_at else "알 수 없음"
    close_at_str = close_at.strftime("%Y-%m-%d") if close_at else "알 수 없음"
    current_year = datetime.now().year
    
    system_prompt = """당신은 한국 채용 공고에서 날짜를 추출하고 변환하는 전문가입니다.

# 핵심 원칙
- **모든 날짜 표현을 YYYY-MM-DD 형식으로 변환**
- **상대적 날짜 표현도 적극 변환** (게시일 기준으로 추정)
- **기간 표현은 시작일과 종료일 모두 추출**

# 날짜 변환 규칙

1. **명확한 날짜**
   - "2025년 3월 15일" → "2025-03-15"
   - "3월 15일" → 게시일 기준 연도 추정 (예: "2025-03-15")
   - "2025.03.15" → "2025-03-15"
   - "3.5" 또는 "3.5(화)" → "2025-03-05"
   - "3/15" → "2025-03-15"

2. **주차 표현** (해당 주의 월요일과 일요일로 변환)
   - "12월 첫째주" → ["2025-12-01", "2025-12-07"]
   - "12월 1주차" → ["2025-12-01", "2025-12-07"]
   - "4월 3째주" → ["2025-04-14", "2025-04-20"]
   - "5월 3주 - 5월 4주" → ["2025-05-12", "2025-05-26"]

3. **순 표현** (대략적인 기간으로 변환)
   - "3월 초" → ["2025-03-01", "2025-03-10"]
   - "7월 중순" → ["2025-07-11", "2025-07-20"]
   - "12월 하순" → ["2025-12-21", "2025-12-31"]
   - "4월 말" → ["2025-04-25", "2025-04-30"]

4. **기간 표현**
   - "3/1~3/15" → ["2025-03-01", "2025-03-15"]
   - "3월 1일 - 3월 15일" → ["2025-03-01", "2025-03-15"]

5. **상시 채용**
   - "상시 채용", "수시 채용" → 빈 리스트 []

# 각 필드별 추출

1. **semester (상/하반기)**
   - "상반기", "하반기" 명시가 있으면 추출
   - 없으면 null

2. **application_deadline (지원 마감)**
   - "지원 마감", "접수 마감", "모집 마감" 관련 날짜 추출
   - 기간이면 마감일 위주로 추출

3. **document_screening (서류 전형)**
   - "서류 전형", "서류 심사", "인적성 검사", "코딩테스트" 날짜 추출
   - 기간이면 시작일과 종료일 모두 추출

4. **first_interview (1차 면접)**
   - "1차 면접", "실무 면접", "기술 면접", "기술 역량 인터뷰" 날짜 추출
   - 기간이면 시작일과 종료일 모두 추출

5. **second_interview (2차 면접)**
   - "2차 면접", "임원 면접", "최종 면접", "종합 역량 인터뷰" 날짜 추출
   - 기간이면 시작일과 종료일 모두 추출

6. **join_date (입사일)**
   - "입사일", "입사 예정일", "근무 시작일" 날짜 추출
   - "7월 초 입사" → ["2025-07-01", "2025-07-10"]

# 중요 예시

입력: "지원 마감: 12월 첫째주"
출력: application_deadline: ["2025-12-01", "2025-12-07"]

입력: "서류 전형: 4월 중순"
출력: document_screening: ["2025-04-11", "2025-04-20"]

입력: "1차 면접: 5월 3주 - 5월 4주"
출력: first_interview: ["2025-05-12", "2025-05-26"]

입력: "입사: 7월 초"
출력: join_date: ["2025-07-01", "2025-07-10"]

JSON 형식으로만 응답:
{
    "semester": null or "상반기" or "하반기",
    "application_deadline": [],
    "document_screening": [],
    "first_interview": [],
    "second_interview": [],
    "join_date": []
}"""
    
    user_prompt = f"""# 채용 공고 정보

**회사명**: {company_name}
**게시일**: {posted_at_str}
**마감일**: {close_at_str}
**현재 연도**: {current_year}

**채용 일정 관련 내용**:
{schedule_text}

---

위 내용에서 모든 날짜 표현을 YYYY-MM-DD 형식으로 변환하여 추출하세요.
상대적 날짜 표현("12월 첫째주", "4월 중순" 등)도 적극적으로 날짜로 변환하세요."""
    
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
            max_tokens=1000,
        )
        
        content = response.choices[0].message.content if response and response.choices else "{}"
        
        try:
            extracted_data = json.loads(content)
            
            global debug_count
            if debug_count < 3:
                print(f"\n  [DEBUG] LLM 응답:")
                print(f"  {json.dumps(extracted_data, ensure_ascii=False, indent=2)}")
                debug_count += 1
                
        except json.JSONDecodeError as e:
            print(f"  [에러] JSON 파싱 실패: {e}")
            print(f"  [에러] 응답 내용: {content[:500]}")
            extracted_data = {}
        
        # LLM 결과와 추론 결과 병합
        llm_semester = extracted_data.get("semester")
        llm_application = extracted_data.get("application_deadline", [])
        
        print(f"  [LLM] semester: {llm_semester}")
        print(f"  [LLM] application_deadline: {llm_application}")
        
        # or 연산자로 병합 (LLM 결과 우선, 없으면 추론)
        final_semester = llm_semester or inferred_semester
        final_application = llm_application if llm_application else inferred_application
        
        print(f"  [최종] semester: {final_semester}")
        print(f"  [최종] application_date: {final_application}")
        
        result = {
            "post_id": post_id,
            "company_id": company_id,
            "semester": final_semester,
            "industry_id": industry_id,
            "application_date": final_application,  # application_deadline → application_date
            "document_screening_date": extracted_data.get("document_screening", []),  # document_screening → document_screening_date
            "first_interview": extracted_data.get("first_interview", []),
            "second_interview": extracted_data.get("second_interview", []),
            "join_date": extracted_data.get("join_date", []),
        }
        
        # 리스트 형식 검증
        for key in ["application_date", "document_screening_date", "first_interview", "second_interview", "join_date"]:
            if not isinstance(result[key], list):
                result[key] = [] if result[key] is None else [result[key]]
        
        # 날짜 형식 검증
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        for key in ["application_date", "document_screening_date", "first_interview", "second_interview", "join_date"]:
            result[key] = [d for d in result[key] if isinstance(d, str) and date_pattern.match(d)]
        
        return result
        
    except Exception as e:
        print(f"  [에러] LLM 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # 에러 발생해도 추론 결과는 반환
        return {
            "post_id": post_id,
            "company_id": company_id,
            "semester": inferred_semester,
            "industry_id": industry_id,
            "application_deadline": inferred_application,
            "document_screening": [],
            "first_interview": [],
            "second_interview": [],
            "join_date": [],
        }
    finally:
        if client:
            client.close()

# ============================================================================
# 데이터 수집 함수
# ============================================================================

def get_competitor_companies(db: Session) -> List[Company]:
    """경쟁사 회사 조회"""
    try:
        from app.db.crud.db_competitors_skills import COMPETITOR_GROUPS
        from sqlalchemy import or_
        
        or_conditions = []
        for group_name, keywords in COMPETITOR_GROUPS.items():
            for keyword in keywords:
                pattern = keyword.replace("%", "")
                or_conditions.append(Company.name.like(f"{pattern}%"))
        
        if not or_conditions:
            print("  [경고] COMPETITOR_GROUPS가 비어있습니다. 기본 회사 리스트 사용")
            competitor_names = [
                "토스", "카카오", "한화시스템", "현대오토에버",
                "우아한형제들", "LG CNS", "네이버", "쿠팡", "라인"
            ]
            return db.query(Company).filter(Company.name.in_(competitor_names)).all()
        
        companies = db.query(Company).filter(or_(*or_conditions)).distinct().all()
        return companies
        
    except ImportError:
        print("  [경고] COMPETITOR_GROUPS를 불러올 수 없습니다. 기본 회사 리스트 사용")
        competitor_names = [
            "토스", "카카오", "한화시스템", "현대오토에버",
            "우아한형제들", "LG CNS", "네이버", "쿠팡", "라인"
        ]
        return db.query(Company).filter(Company.name.in_(competitor_names)).all()

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
    """경쟁사 Post에서 채용 일정 추출"""
    print("="*80)
    print("채용 일정 추출 시작 (v4 - 상대적 날짜 표현 변환)")
    print("="*80)
    
    print("\n[1/3] 경쟁사 회사 조회 중...")
    companies = get_competitor_companies(db)
    print(f"  조회된 회사: {len(companies)}개")
    for company in companies:
        print(f"    - {company.name} (ID: {company.id})")
    
    if not companies:
        print("  경쟁사 회사를 찾을 수 없습니다.")
        return []
    
    print(f"\n[2/3] 각 회사별 Post 조회 중... (회사당 최대 {limit_per_company}개)")
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
    
    # 3. LLM으로 일정 추출 (DB 세션 없이)
    print(f"\n[3/3] LLM으로 채용 일정 추출 중... (총 {len(posts_data)}개)")
    results = []
    
    for idx, post_data in enumerate(posts_data, 1):
        print(f"\n[{idx}/{len(posts_data)}] Post ID: {post_data['id']}")
        print(f"  제목: {post_data['title'][:50]}...")
        print(f"  회사: {post_data['company_name']}")
        print(f"  posted_at: {post_data['posted_at']}")
        print(f"  close_at: {post_data['close_at']}")
        print(f"  crawled_at: {post_data['crawled_at']}")
        
        extracted = extract_schedule_with_llm(
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
            
            total_dates = sum([
                len(extracted['application_date']),
                len(extracted['document_screening_date']),
                len(extracted['first_interview']),
                len(extracted['second_interview']),
                len(extracted['join_date'])
            ])
            
            print(f"  ✓ 추출 성공 (날짜 {total_dates}개)")
            print(f"    - 접수: {len(extracted['application_date'])}개, "
                  f"서류: {len(extracted['document_screening_date'])}개, "
                  f"1차: {len(extracted['first_interview'])}개, "
                  f"2차: {len(extracted['second_interview'])}개, "
                  f"입사: {len(extracted['join_date'])}개")
        else:
            print(f"  ✗ 추출 실패 (None 반환)")
    
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
        # extract_recruitment_schedules 내부에서 이미 db.close() 호출됨
        
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"recruitment_schedules_{timestamp}.json"
        
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
        # extract_recruitment_schedules 내부에서 이미 닫혔을 수 있으므로 안전하게 처리
        try:
            if db.is_active:  # 세션이 아직 활성화되어 있으면 닫기
                db.close()
        except:
            pass

if __name__ == "__main__":
    main()