import os
import sys
import json
import re
import time
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 프로젝트 루트를 sys.path에 추가 (직접 실행 시 모듈을 찾을 수 있도록)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from openai import OpenAI
from openai import RateLimitError
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.config.base import SessionLocal

# 관계로 참조되는 모든 모델을 import 해서 SQLAlchemy 매퍼 초기화 시
# 'Industry', 'PostSkill' 등의 이름을 제대로 resolve 하도록 한다.
from app.models.post import Post
from app.models.company import Company
from app.models.industry import Industry
from app.models.skill import Skill
from app.models.post_skill import PostSkill
from app.models.position import Position
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill
from app.models.recruit_schedule import RecruitSchedule
from app.schemas.schemas_recruit_schedule import RecruitScheduleData


class ScheduleParserExtractor:
    """
    채용 공고 설명에서 LLM을 사용하여 채용 일정 정보를 추출하는 클래스
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def extract_schedule_info(self, description: str, title: str = "") -> RecruitScheduleData:
        """LLM을 사용하여 채용 일정 정보 추출"""

        system_prompt = """
당신은 한국어 채용 공고(Job Description) 텍스트를 분석하여 **채용 전형 일정(Hiring Timeline)**을 정밀하게 추출하는 전문 AI입니다.
공고 내용을 바탕으로 아래 6개 핵심 필드의 날짜 정보를 추출하여, 오직 **JSON 포맷**으로만 응답하세요.

### 1. 목표 출력 포맷 (JSON Schema)
{
  "application_date": [["YYYY-MM-DD", "YYYY-MM-DD"], ...],   // 서류 접수/모집 기간
  "semester": "상반기" | "하반기" | null,                     // 채용 시즌 (접수 시작월 기준 자동 판단)
  "document_screening_date": [["YYYY-MM-DD", "YYYY-MM-DD"], ...], // 서류발표, 필기시험, 코딩테스트, AI역량검사, 과제전형
  "first_interview": [["YYYY-MM-DD", "YYYY-MM-DD"], ...],    // 1차 면접, 실무/직무 면접, 화상 면접
  "second_interview": [["YYYY-MM-DD", "YYYY-MM-DD"], ...],   // 2차 면접, 임원/최종 면접 (단, 처우협의는 제외)
  "join_date": [["YYYY-MM-DD", "YYYY-MM-DD"], ...]           // 입사 예정일, 입문 교육 시작일
}

### 2. 핵심 추출 원칙 (Strict Rules)
1. **JSON Only:** 설명, 주석, 마크다운 없이 순수한 JSON 객체 하나만 반환합니다.
2. **Date Format:** 모든 날짜는 `YYYY-MM-DD` 문자열 형식입니다.
3. **2D Array Structure:** 모든 날짜 필드는 `[[시작일, 종료일], ...]` 형태의 리스트입니다.
   - **단일 날짜(Point):** `["2025-11-20", "2025-11-20"]` (시작과 끝을 동일하게 처리)
   - **기간(Range):** `["2025-11-20", "2025-11-30"]`
   - **복수 일정:** `[["2025-11-20", "2025-11-20"], ["2025-12-01", "2025-12-05"]]`
4. **Data Missing:** 해당 단계의 날짜 정보가 없거나 "추후 안내", "개별 통보"인 경우 빈 리스트 `[]`를 반환합니다.

### 3. 날짜 및 연도 추론 로직 (Advanced Logic)
**[규칙 A] 모호한 날짜 표현의 정규화 (Normalization)**
구체적인 날짜 없이 시점만 묘사된 경우, 아래 기준표에 따라 **기간(Range)**으로 변환합니다.
- **월초 / 상순:** 해당 월 01일 ~ 10일
- **월중 / 중순:** 해당 월 11일 ~ 20일
- **월말 / 하순:** 해당 월 21일 ~ 말일(28/30/31일)
- **N월 중 / N월 내:** 해당 월 01일 ~ 해당 월 말일 (예: "11월 중 면접" → 11월 전체)
- **주차(Week) 표현:**
  - 1주차/첫째주: 01일 ~ 07일
  - 2주차/둘째주: 08일 ~ 14일
  - 3주차/셋째주: 15일 ~ 21일
  - 4주차/넷째주: 22일 ~ 28일
  - 5주차/마지막주: 29일 ~ 말일

**[규칙 B] 연도(Year) 자동 추정 알고리즘**
공고에 연도가 명시되지 않은 경우 다음 로직을 따릅니다.
1. **기준 설정:** `application_date`의 시작 연도를 기준 연도(Y)로 잡습니다. (텍스트에 없으면 현재 연도 2025년으로 가정)
2. **해 넘김(Year Turnover):** 전형 순서상(서류→면접→입사) 나중에 오는 단계의 월(Month) 숫자가 이전 단계보다 현저히 작다면, 다음 해(Y+1)로 계산합니다.
   - 예: 접수(11월) → 면접(1월) : 면접은 내년 1월로 판단.
3. **과거 방지:** 모든 일정은 접수 시작일 이후라고 가정합니다.

**[규칙 C] 단계별 포함/제외 기준**
- **semester:** 1~6월 시작은 "상반기", 7~12월 시작은 "하반기".
- **document_screening_date:** 서류 합격자 발표뿐만 아니라 **코딩테스트, 직무/필기 테스트, AI역량검사, 과제 제출** 일정을 모두 포함합니다.
- **first_interview:** 1차, 실무, 기술, 직무 면접.
- **second_interview:** 2차, 임원, 경영진, 최종, CEO 면접. ("레퍼런스 체크"는 이 단계에 포함하되, "처우협의/신체검사"는 제외)
- **join_date:** 입사일, OJT 시작일. ("수습기간 3개월" 같은 기간 정보는 제외)

### 4. 입력 데이터 처리 예시 (Few-shot)
**Input Text:**
"2025년 하반기 공채. 11월 1일~14일 접수. 11월 말 서류 발표. 12월 첫째 주 코딩테스트 및 1차 면접. 내년 1월 중 최종 면접 후 입사."

**Internal Reasoning:**
- 접수: 2025-11-01 ~ 2025-11-14
- 서류발표(11월 말): 2025-11-21 ~ 2025-11-30
- 코테/1차면접(12월 1주): 2025-12-01 ~ 2025-12-07 (둘 다 해당 기간에 넣음)
- 최종면접(1월 중): 접수가 11월이므로 1월은 2026년. → 2026-01-01 ~ 2026-01-31
- 입사: 최종 면접 후이므로 1월 또는 그 이후이나, 문맥상 1월 중으로 추정 → 2026-01-01 ~ 2026-01-31

**Output JSON:**
{
  "application_date": [["2025-11-01", "2025-11-14"]],
  "semester": "하반기",
  "document_screening_date": [["2025-11-21", "2025-11-30"], ["2025-12-01", "2025-12-07"]],
  "first_interview": [["2025-12-01", "2025-12-07"]],
  "second_interview": [["2026-01-01", "2026-01-31"]],
  "join_date": [["2026-01-01", "2026-01-31"]]
}

---
**이제 아래의 채용 공고 텍스트를 분석하여 JSON 결과만 출력하세요.**
"""



        user_prompt = f"""채용 공고 제목: {title}

채용 공고 내용:
{description}

위 채용 공고에서 '채용 전형 일정'을 추출해 JSON 스키마에 맞춰 정확히 한 개의 JSON 객체로만 답변해주세요."""

        # Rate limit 오류 발생 시 재시도 로직
        max_retries = 5
        retry_delay = 2  # 초기 대기 시간 (초)
        response = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                break  # 성공 시 루프 종료
                
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    # 에러 메시지에서 대기 시간 추출 시도
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    
                    # 에러 메시지에서 "try again in X.XXXs" 추출
                    error_str = str(e)
                    if "try again in" in error_str:
                        try:
                            match = re.search(r'try again in ([\d.]+)s', error_str)
                            if match:
                                wait_time = float(match.group(1)) + 1  # 안전을 위해 1초 추가
                        except:
                            pass
                    
                    print(f"  ⚠ Rate limit 도달. {wait_time:.1f}초 대기 후 재시도 ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    # 최대 재시도 횟수 초과
                    print(f"  ✗ Rate limit 오류: 최대 재시도 횟수({max_retries}) 초과")
                    raise
        
        if response is None:
            raise Exception("API 호출 실패: 응답을 받지 못했습니다.")

        try:
            result = json.loads(response.choices[0].message.content)

            # 데이터 추출
            app_dates_raw = result.get("application_date", [])
            doc_dates_raw = result.get("document_screening_date", [])
            join_dates_raw = result.get("join_date", [])
            first_int_raw = result.get("first_interview", [])
            second_int_raw = result.get("second_interview", [])

            # 모든 날짜 필드를 [[start, end]] 형태로 정규화
            app_dates = self._normalize_period_list(app_dates_raw)
            doc_dates = self._normalize_period_list(doc_dates_raw)
            join_dates = self._normalize_period_list(join_dates_raw)
            first_int = self._normalize_period_list(first_int_raw)
            second_int = self._normalize_period_list(second_int_raw)

            semester = result.get("semester")

            # semester가 LLM에서 제대로 추출되지 않은 경우 수동 판단
            if not semester and app_dates:
                semester = self._determine_semester(app_dates)

            # Pydantic 모델로 검증 및 변환
            schedule_data = RecruitScheduleData(
                application_date=app_dates,
                semester=semester,
                document_screening_date=doc_dates,
                first_interview=first_int,
                second_interview=second_int,
                join_date=join_dates,
            )

            return schedule_data

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 실패: {str(e)}")
            print(f"LLM 응답 내용: {response.choices[0].message.content[:500]}")
            return RecruitScheduleData(
                application_date=[],
                semester=None,
                document_screening_date=[],
                first_interview=[],
                second_interview=[],
                join_date=[]
            )
        except Exception as e:
            print(f"일정 추출 오류: {str(e)}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return RecruitScheduleData(
                application_date=[],
                semester=None,
                document_screening_date=[],
                first_interview=[],
                second_interview=[],
                join_date=[]
            )

    def _normalize_period_list(self, periods: List) -> List[List[str]]:
        """
        다양한 형태의 날짜 데이터를 [["start", "end"], ...] 형태로 정규화한다.
        
        입력 형태:
        - [["start", "end"], ...] (이미 정규화된 형태)
        - [[["start", "end"]], ...] (3중 중첩)
        - ["date", ...] (단일 날짜 리스트 - 면접일 등)
        - "date" (단일 문자열)
        
        출력 형태:
        - [["start", "end"], ...] (항상 동일한 형태)
        - 단일 날짜는 [date, date] 형태로 변환
        """
        if not isinstance(periods, list):
            return []

        normalized: List[List[str]] = []
        for p in periods:
            # case 1: ["start", "end"] 형태 (이미 정규화됨)
            if isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) for x in p):
                normalized.append(p)
            # case 2: [["start", "end"]] 형태 (3중 중첩)
            elif isinstance(p, list) and len(p) >= 1:
                if isinstance(p[0], list):
                    inner = p[0]
                    if isinstance(inner, list) and len(inner) == 2 and all(isinstance(x, str) for x in inner):
                        normalized.append(inner)
                # case 3: ["date"] 형태 (단일 날짜 문자열)
                elif isinstance(p[0], str) and len(p) == 1:
                    # 단일 날짜를 [date, date] 형태로 변환
                    normalized.append([p[0], p[0]])
            # case 4: 단일 문자열 "date"
            elif isinstance(p, str):
                normalized.append([p, p])
            # 그 외 형태는 무시

        return normalized

    def _determine_semester(self, application_dates: List) -> Optional[str]:
        """
        application_date를 기반으로 상반기/하반기 자동 판단

        상반기: 1월~6월 (입사는 주로 상반기 또는 7월)
        하반기: 7월~12월 (입사는 주로 하반기 또는 다음해 1월)

        Args:
            application_dates: [["시작일", "마감일"], ...] 형태의 날짜 리스트

        Returns:
            "상반기", "하반기", 또는 None
        """
        if not application_dates or len(application_dates) == 0:
            return None

        try:
            # [["yyyy-mm-dd", "yyyy-mm-dd"], ...] 구조 처리
            first_period = application_dates[0]
            
            if isinstance(first_period, list) and len(first_period) >= 2:
                deadline = first_period[1]  # 마감일
                month = int(deadline.split('-')[1])
                
                if 1 <= month <= 6:
                    return "상반기"
                elif 7 <= month <= 12:
                    return "하반기"
            
            return None

        except Exception:
            return None


class ScheduleParserProcessor:
    """채용 일정 추출 및 저장을 처리하는 클래스"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.extractor = ScheduleParserExtractor(api_key=api_key, model=model)
        self.SessionLocal = SessionLocal

    def process_posts(
        self,
        batch_size: int = 10,
        limit: Optional[int] = None,
        company_keyword: Optional[str] = None,
        competitor_only: bool = False,
        max_workers: int = 5,
    ):
        """
        Post 데이터를 배치로 처리하여 일정 추출 및 저장

        Args:
            batch_size: 커밋 단위 배치 크기
            limit: 최대 처리할 포스트 수 (None 이면 전체)
            company_keyword: 회사명에 포함되어야 할 키워드 (예: "naver"). None 이면 전체 회사 대상
            competitor_only: True이면 주요 경쟁사 그룹만 필터링 (LINE, NAVER, 토스, 우아한형제들, 현대오토에버, Coupang, 카카오, 한화시스템, 한화손해보험)
            max_workers: 병렬 처리 시 최대 동시 실행 스레드 수 (기본값: 5, Rate limit 고려)
        """
        session = self.SessionLocal()

        try:
            # 기본: Post 테이블에서 시작
            query = session.query(Post)

            # 주요 경쟁사 그룹 필터링
            if competitor_only:
                from sqlalchemy import or_
                query = query.join(Company, Post.company_id == Company.id).filter(
                    or_(
                        func.lower(Company.name).like('%line%'),
                        func.lower(Company.name) == 'ipx',
                        func.lower(Company.name).like('%naver%'),
                        func.lower(Company.name).like('%토스%'),
                        func.lower(Company.name).like('%toss%'),
                        func.lower(Company.name).in_(['비바리퍼블리카', 'aicc']),
                        func.lower(Company.name).like('%우아한형제들%'),
                        func.lower(Company.name).like('%배달의민족%'),
                        func.lower(Company.name).like('%현대오토에버%'),
                        func.lower(Company.name).like('%coupang%'),
                        func.lower(Company.name).like('%쿠팡%'),
                        func.lower(Company.name).like('%카카오%'),
                        func.lower(Company.name).like('%한화시스템%'),
                        func.lower(Company.name).like('%한화손해보험%'),
                    )
                )
            # 회사 키워드가 없는 경우에만
            # "아직 일정이 추출되지 않은 포스트"로 한정
            elif not company_keyword:
                query = query.outerjoin(
                    RecruitSchedule,
                    Post.id == RecruitSchedule.post_id
                ).filter(
                    RecruitSchedule.schedule_id.is_(None)
                )

            if company_keyword and not competitor_only:
                # 회사명 부분 일치 필터 (대소문자 무시, LIKE '%키워드%')
                # 예: "NAVER" / "naver" 입력 시 "네이버페이", "Naver Cloud" 등 포함한 모든 회사명 매칭
                pattern = company_keyword.strip()
                if pattern:
                    like_pattern = f"%{pattern.lower()}%"
                    query = query.join(Company, Post.company_id == Company.id).filter(
                        func.lower(Company.name).like(like_pattern)
                    )

            if limit:
                query = query.limit(limit)

            posts = query.all()

            print(f"처리할 포스트 수: {len(posts)}")

            processed_count = 0
            for i, post in enumerate(posts, 1):
                try:
                    print(f"\n[{i}/{len(posts)}] Processing Post ID: {post.id}")
                    print(f"  제목: {post.title[:100]}")
                    desc = (post.description or "").strip()
                    print(f"  description 길이: {len(desc)}")
                    if desc:
                        preview = desc[:800].replace("\n", " ")
                        print(f"  description 미리보기(앞 800자): {preview}")
                    else:
                        print("  description 없음 또는 빈 문자열")
                    
                    # Rate limit 방지를 위한 요청 간 딜레이 (0.5초)
                    if i > 1:
                        time.sleep(0.5)
                    
                    # 일정 정보 추출
                    schedule_data = self.extractor.extract_schedule_info(
                        description=post.description or "",
                        title=post.title or ""
                    )
                    # DB에 저장
                    self.save_schedule(session, post, schedule_data)

                    processed_count += 1

                    if processed_count % batch_size == 0:
                        session.commit()
                        print(f"  ✓ {processed_count}개 포스트 처리 완료 (커밋)")
                except RateLimitError as e:
                    print(f"  ✗ Post ID {post.id} 처리 실패: Rate limit 오류 (재시도 불가)")
                    print(f"    오류 메시지: {str(e)}")
                    session.rollback()
                    # Rate limit 오류는 재시도 로직에서 처리되므로 여기서는 건너뛰기
                    continue
                except Exception as e:
                    print(f"  ✗ Post ID {post.id} 처리 실패: {str(e)}")
                    session.rollback()
                    continue

            if processed_count % batch_size != 0:
                session.commit()
                print(f"\n최종 커밋: 총 {processed_count}개 포스트 처리 완료")

        except Exception as e:
            print(f"배치 처리 중 오류 발생: {str(e)}")
            session.rollback()
            raise
        finally:
            session.close()

    def save_schedule(self, session: Session, post: Post, schedule_data: RecruitScheduleData):
        """추출된 일정 정보를 DB에 저장"""

        # 기존 일정 레코드 조회
        existing = session.query(RecruitSchedule).filter(
            RecruitSchedule.post_id == post.id
        ).first()

        if existing:
            # 이미 일정이 있으면 해당 레코드를 업데이트
            existing.semester = schedule_data.semester
            existing.application_date = schedule_data.application_date
            existing.document_screening_date = schedule_data.document_screening_date
            existing.first_interview = schedule_data.first_interview
            existing.second_interview = schedule_data.second_interview
            existing.join_date = schedule_data.join_date

            print(f"  ✓ Post ID {post.id}의 일정 정보 업데이트:")
            print(f"    - semester: {schedule_data.semester}")
            print(f"    - application_date: {schedule_data.application_date}")
            print(f"    - document_screening_date: {schedule_data.document_screening_date}")
            print(f"    - first_interview: {schedule_data.first_interview}")
            print(f"    - second_interview: {schedule_data.second_interview}")
            print(f"    - join_date: {schedule_data.join_date}")
            return

        # 없으면 새로 생성
        recruit_schedule = RecruitSchedule(
            post_id=post.id,
            company_id=post.company_id,
            industry_id=post.industry_id,
            semester=schedule_data.semester,
            application_date=schedule_data.application_date,
            document_screening_date=schedule_data.document_screening_date,
            first_interview=schedule_data.first_interview,
            second_interview=schedule_data.second_interview,
            join_date=schedule_data.join_date,
        )

        session.add(recruit_schedule)

        print(f"  ✓ 일정 정보 저장:")
        print(f"    - semester: {schedule_data.semester}")
        print(f"    - application_date: {schedule_data.application_date}")
        print(f"    - document_screening_date: {schedule_data.document_screening_date}")
        print(f"    - first_interview: {schedule_data.first_interview}")
        print(f"    - second_interview: {schedule_data.second_interview}")
        print(f"    - join_date: {schedule_data.join_date}")

    def get_schedule_by_post_id(self, post_id: int) -> Optional[Dict[str, Any]]:
        """특정 포스트의 일정 정보 조회"""
        session = self.SessionLocal()

        try:
            schedule = session.query(RecruitSchedule).filter(
                RecruitSchedule.post_id == post_id
            ).first()

            if not schedule:
                return None

            return {
                "schedule_id": schedule.schedule_id,
                "post_id": schedule.post_id,
                "company_id": schedule.company_id,
                "industry_id": schedule.industry_id,
                "semester": schedule.semester,
                "application_date": schedule.application_date or [],
                "document_screening_date": schedule.document_screening_date or [],
                "first_interview": schedule.first_interview or [],
                "second_interview": schedule.second_interview or [],
                "join_date": schedule.join_date or [],
            }
        finally:
            session.close()

    def get_all_schedules(self, limit: int = 100) -> List[Dict[str, Any]]:
        """모든 일정 정보 조회"""
        session = self.SessionLocal()

        try:
            schedules = session.query(RecruitSchedule).limit(limit).all()

            return [
                {
                    "schedule_id": s.schedule_id,
                    "post_id": s.post_id,
                    "company_id": s.company_id,
                    "industry_id": s.industry_id,
                    "semester": s.semester,
                    "application_date": s.application_date or [],
                    "document_screening_date": s.document_screening_date or [],
                    "first_interview": s.first_interview or [],
                    "second_interview": s.second_interview or [],
                    "join_date": s.join_date or [],
                }
                for s in schedules
            ]
        finally:
            session.close()


def main():
    """메인 실행 함수"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY 환경 변수를 설정해주세요")

    # 프로세서 초기화
    processor = ScheduleParserProcessor(api_key=OPENAI_API_KEY)

    print("=" * 60)
    print("주요 경쟁사 채용 일정 추출 시작")
    print("=" * 60)
    print("대상 경쟁사 그룹:")
    print("  - LINE (IPX 포함)")
    print("  - NAVER")
    print("  - 토스 (비바리퍼블리카, AICC 포함)")
    print("  - 우아한형제들 (배달의민족 포함)")
    print("  - 현대오토에버")
    print("  - Coupang (쿠팡)")
    print("  - 카카오")
    print("  - 한화시스템")
    print("  - 한화손해보험")
    print("=" * 60)

    # 주요 경쟁사 그룹의 채용 공고 처리
    # limit=None이면 전체 처리, 특정 개수로 제한하려면 limit=숫자 지정
    processor.process_posts(
        batch_size=10, 
        limit=None,  # 전체 처리 (필요시 숫자로 변경)
        competitor_only=True  # 주요 경쟁사만 필터링
    )

    print("\n" + "=" * 60)
    print("처리 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
