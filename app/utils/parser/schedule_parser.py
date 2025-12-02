import os
import sys
import json
import re
from typing import Optional, Dict, Any, List

# 프로젝트 루트를 sys.path에 추가 (직접 실행 시 모듈을 찾을 수 있도록)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from openai import OpenAI
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

        system_prompt = """당신은 한국어 채용 공고에서 '채용 전형 일정'을 정밀하게 추출하는 전문가입니다.

다음 6개의 필드를 추출해서 JSON 객체로 반환하세요.

**필드 정의:**
1. application_date: 지원서 접수/모집 기간 리스트 [[["시작일", "마감일"]], ...] (여러 접수 기간이 있을 수 있음)
2. semester: "상반기" 또는 "하반기" (application_date 기반으로 자동 판단)
3. document_screening_date: 서류전형/코딩테스트 기간 리스트 [[["시작일", "마감일"]], ...] (여러 개 가능)
4. first_interview: 1차 면접 날짜 리스트 ["YYYY-MM-DD", ...] (여러 개 가능)
5. second_interview: 2차/최종 면접 날짜 리스트 ["YYYY-MM-DD", ...] (여러 개 가능)
6. join_date: 입사일 기간 리스트 [[["시작일", "마감일"]], ...] (여러 입사 기회가 있을 수 있음)

**JSON 스키마:**
{
  "application_date": [[["YYYY-MM-DD", "YYYY-MM-DD"]], ...],
  "semester": "상반기" 또는 "하반기" 또는 null,
  "document_screening_date": [[["YYYY-MM-DD", "YYYY-MM-DD"]], ...],
  "first_interview": ["YYYY-MM-DD", ...],
  "second_interview": ["YYYY-MM-DD", ...],
  "join_date": [[["YYYY-MM-DD", "YYYY-MM-DD"]], ...]
}

**핵심 규칙:**
1. 반드시 위 JSON 객체 한 개만 반환하고, 추가 설명은 절대 포함하지 마세요.
2. 모든 날짜는 YYYY-MM-DD 형식의 문자열로만 반환하세요.
3. application_date, document_screening_date, join_date는 [[["시작일", "마감일"]]] 형태의 2차원 리스트입니다.
4. first_interview, second_interview는 단일 날짜의 1차원 리스트입니다.
5. 기간이 하나의 날짜만 있으면 [[["YYYY-MM-DD", "YYYY-MM-DD"]]]처럼 같은 날짜를 두 번 반복합니다.

**날짜 추출 상세 규칙:**

[1] 모집/접수 기간 (application_date):
- "2025.11.10 ~ 2025.11.28" → [[["2025-11-10", "2025-11-28"]]]
- "2025.11.10 ~ 2025.11.28 (17:00)" → [[["2025-11-10", "2025-11-28"]]] (시간 정보는 무시)
- "11월 28일까지" 또는 "11월 28일 마감" → [[["2025-11-01", "2025-11-28"]]] (월 초부터로 추정)
- 시작일이 명시되지 않은 경우 해당 월의 1일로 추정
- 단일 날짜: "3월 15일" → [[["2025-03-15", "2025-03-15"]]]
- "모집 기간", "지원 기간", "접수 기간" 등의 표현 모두 포함
- 날짜 정보가 전혀 없으면 빈 리스트 []

[2] 학기 구분 (semester):
- application_date의 시작작일 기준으로 자동 판단
- 1월~6월 시작: "상반기"
- 7월~12월 시작작: "하반기"
- application_date가 없거나 판단 불가능한 경우: null

[3] 서류전형 필기시험 (document_screening_date):
- "서류전형", "서류심사", "서류발표", "서류합격자 발표", "서류 검토" → document_screening_date에 추가
- "코딩테스트", "필기시험", "직무 테스트", "기업문화적합도 검사" → 모두 document_screening_date에 추가
- "1차 인터뷰", "면접" 이전의 모든 평가 단계
- 여러 차수가 있으면 각각 별도 항목으로:
  예: "서류전형: 10월 11일", "코딩테스트: 10월 26일" → [[["2025-10-11", "2025-10-11"]], [["2025-10-26", "2025-10-26"]]]
- 기간 형태: "서류심사: 9월 1일 ~ 9월 5일" → [[["2025-09-01", "2025-09-05"]]]
- 단일 날짜: "서류발표: 3월 10일" → [[["2025-03-10", "2025-03-10"]]]
- 날짜가 명시되지 않고 "서류전형 ▶ 1차 인터뷰"처럼 순서만 나열된 경우 빈 리스트 []

[4] 면접 (first_interview, second_interview):
- "1차 면접", "1차 인터뷰", "초기 면접", "실무 면접" → first_interview
- "2차 면접", "2차 인터뷰", "최종 면접", "임원 면접", "처우협의" → second_interview
- "면접"이라고만 되어 있고 차수가 없으면 → first_interview
- "레퍼런스체크"는 면접 단계로 간주하여 해당 차수에 포함
- 면접은 단일 날짜 리스트입니다: ["2025-03-20", "2025-03-21"]
- 기간이 주어진 경우 시작일과 종료일을 모두 포함: "3월 20일 ~ 3월 25일" → ["2025-03-20", "2025-03-25"]
- 날짜가 명시되지 않고 단순히 "1차 인터뷰 ▶ 2차 인터뷰"처럼 순서만 있으면 빈 리스트 []
- "추가 인터뷰"는 상황에 따라 추가될 수 있으므로 기본적으로 무시

[5] 입사일 (join_date):
- "입사일", "입사 예정일", "합류일", "근무 시작일", "최종합격" 이후 입사 등
- 단일 날짜: "7월 1일" → [[["2025-07-01", "2025-07-01"]]]
- 여러 입사일: "7월 1일 또는 9월 1일" → [[["2025-07-01", "2025-07-01"]], [["2025-09-01", "2025-09-01"]]]
- 입사 가능 기간: "7월 중" → [[["2025-07-01", "2025-07-31"]]]
- "수습 기간 3개월"은 입사 후의 정보이므로 무시
- 날짜가 명시되지 않으면 빈 리스트 []

[6] 날짜 표현 변환:
- "1월 초" → 해당 연도-01-02 ~ 해당 연도-01-10
- "1월 말" → 해당 연도-01-21 ~ 해당 연도-01-31
- "3월 중순" → 해당 연도-03-11 ~ 해당 연도-03-20
- "11월 중", "12월 중" → 해당 연도-11-01 ~ 해당 연도-11-30
- "상순" → 1일 ~ 10일, "중순" → 11일 ~ 20일, "하순" → 21일 ~ 말일
- 구체적 날짜 없이 "11월 중"처럼만 표현된 경우 중간값 사용: 15일

[7] 연도 추정:
- 연도가 명시되지 않은 경우, 채용 공고의 문맥과 현재 시점(2025년 12월)을 고려
- 일반적으로 접수일 이후 날짜들은 같은 연도이거나 다음 연도
- 예: 11월 접수 → 12월은 같은 연도 → 1월은 다음 연도(2026)
- 과거 날짜는 피하고, 현재(2025년 12월)보다 미래의 날짜로 추정
- 1~6월이 나오면 일반적으로 2026년, 7~12월이면 2025년 하반기 또는 2026년 판단
- 공고에 연도가 명시된 경우(예: "2025.11.10") 그대로 사용

[8] 무시해야 하는 표현:
- "추후 안내", "별도 안내", "개별 안내", "상세 일정은 추후 공지" 등
- "수시 채용", "상시 모집", "수시로" 등 구체적 날짜가 없는 표현
- "조기 마감될 수 있음", "연장될 수 있음" 등 변동 가능성 표현
- "결격 사유", "병역 의무", "지원서 허위 기재" 등 자격 요건
- 회사 주소, 문의처, 우대사항 등 채용 일정과 무관한 정보

**예시 1: NAVER Cloud 채용 (날짜 명시, 단계만 있는 경우)**
입력:
[NAVER Cloud] NCP DNS / GSLB 경량화 개발 (경력)
모집 기간: 2025.11.10 ~ 2025.11.28 (17:00)

전형절차:
서류전형(기업문화적합도 검사 및 직무 테스트 포함) ▶ 1차 인터뷰 ▶ 레퍼런스체크 및 2차 인터뷰 ▶ 처우협의 ▶ 최종합격

출력:
{
  "application_date": [[["2025-11-10", "2025-11-28"]]],
  "semester": "하반기",
  "document_screening_date": [],
  "first_interview": [],
  "second_interview": [],
  "join_date": []
}

**예시 2: 카카오 공채 (모든 날짜 명시)**
입력:
◆ 전형단계 및 일정
서류 접수: 2025.09.08(월) 14:00 ~ 2025.09.28(일) 23:59
1차 코딩테스트(온라인): 2025.10.11(토)
2차 코딩테스트(온라인): 2025.10.26(일)
서류합격자 발표: 2025.11.01(금)
1차 인터뷰(오프라인): 2025.11.15(금)
2차 인터뷰(오프라인): 2025.12.10(화)
최종 합격: 2025.12.20(금)
입사일: 2026년 1월 2일

출력:
{
  "application_date": [[["2025-09-08", "2025-09-28"]]],
  "semester": "하반기",
  "document_screening_date": [[["2025-10-11", "2025-10-11"]], [["2025-10-26", "2025-10-26"]], [["2025-11-01", "2025-11-01"]]],
  "first_interview": ["2025-11-15"],
  "second_interview": ["2025-12-10"],
  "join_date": [[["2026-01-02", "2026-01-02"]]]
}

**예시 3: 상반기 채용 (일부 날짜만 명시)**
입력:
2026년 상반기 신입 채용
모집 기간: 2월 1일 ~ 2월 28일
서류 전형 결과 발표: 3월 5일
1차 면접: 3월 중순
2차 면접: 3월 하순
입사: 7월 1일

출력:
{
  "application_date": [[["2026-02-01", "2026-02-28"]]],
  "semester": "상반기",
  "document_screening_date": [[["2026-03-05", "2026-03-05"]]],
  "first_interview": ["2026-03-15"],
  "second_interview": ["2026-03-25"],
  "join_date": [[["2026-07-01", "2026-07-01"]]]
}

**예시 4: 수시 채용 (상시 채용)**
입력:
◆ 채용 일정
- 지원 기간: 상시 채용
- 전형 절차: 서류 → 면접 → 처우 협의
- 입사일: 협의 후 결정

출력:
{
  "application_date": [],
  "semester": null,
  "document_screening_date": [],
  "first_interview": [],
  "second_interview": [],
  "join_date": []
}

**중요 주의사항:**
- 코딩테스트, 필기시험, 직무테스트는 절대 면접에 넣지 마세요. 반드시 document_screening_date에 넣습니다.
- application_date, document_screening_date, join_date는 반드시 [[["시작일", "마감일"]]] 형태의 2차원 리스트입니다.
- first_interview, second_interview는 ["날짜"] 형태의 1차원 리스트입니다.
- semester는 application_date의 마감일 월을 기준으로 판단합니다 (1~6월: 상반기, 7~12월: 하반기).
- 날짜 정보가 전혀 없거나 "추후 안내"인 경우 빈 리스트 []를 반환하세요.
- 모든 날짜는 YYYY-MM-DD 형식이어야 합니다.
- 현재 시점(2025년 12월)을 고려하여 과거 날짜가 되지 않도록 연도를 조정하세요.
- "처우협의", "최종합격"은 면접 단계가 아니므로 추출하지 않습니다.
"""

        user_prompt = f"""채용 공고 제목: {title}

채용 공고 내용:
{description}

위 채용 공고에서 '채용 전형 일정'을 추출해 JSON 스키마에 맞춰 정확히 한 개의 JSON 객체로만 답변해주세요."""

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

            result = json.loads(response.choices[0].message.content)

            # 데이터 추출
            app_dates_raw = result.get("application_date", [])
            doc_dates_raw = result.get("document_screening_date", [])
            join_dates_raw = result.get("join_date", [])

            # [[[start, end]]] 형태 등을 [[start, end]] 형태로 정규화
            app_dates = self._normalize_period_list(app_dates_raw)
            doc_dates = self._normalize_period_list(doc_dates_raw)
            join_dates = self._normalize_period_list(join_dates_raw)

            semester = result.get("semester")
            first_int = result.get("first_interview", [])
            second_int = result.get("second_interview", [])

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
        [[[start, end]]] 또는 [[start, end]] 등 다양한 중첩 구조를
        Pydantic 스키마에 맞는 [[start, end]] 형태로 정규화한다.
        """
        if not isinstance(periods, list):
            return []

        normalized: List[List[str]] = []
        for p in periods:
            # case 1: [[start, end]]
            if isinstance(p, list) and len(p) == 2 and all(isinstance(x, str) for x in p):
                normalized.append(p)
            # case 2: [[[start, end]]] 또는 [[["start", "end"], ...]] 중 첫 번째만 사용
            elif isinstance(p, list) and len(p) >= 1 and isinstance(p[0], list):
                inner = p[0]
                if isinstance(inner, list) and len(inner) == 2 and all(isinstance(x, str) for x in inner):
                    normalized.append(inner)
            # 그 외 형태는 무시

        return normalized

    def _determine_semester(self, application_dates: List) -> Optional[str]:
        """
        application_date를 기반으로 상반기/하반기 자동 판단

        상반기: 1월~6월 (입사는 주로 상반기 또는 7월)
        하반기: 7월~12월 (입사는 주로 하반기 또는 다음해 1월)

        Args:
            application_dates: [[["시작일", "마감일"]], ...] 형태의 날짜 리스트

        Returns:
            "상반기", "하반기", 또는 None
        """
        if not application_dates or len(application_dates) == 0:
            return None

        try:
            # [[["yyyy-mm-dd", "yyyy-mm-dd"]], ...] 구조 처리
            first_period = application_dates[0]
            
            if isinstance(first_period, list) and len(first_period) > 0:
                # 가장 안쪽 리스트 찾기
                inner_period = first_period[0] if isinstance(first_period[0], list) else first_period
                
                if isinstance(inner_period, list) and len(inner_period) >= 2:
                    deadline = inner_period[1]  # 마감일
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
    ):
        """
        Post 데이터를 배치로 처리하여 일정 추출 및 저장

        Args:
            batch_size: 커밋 단위 배치 크기
            limit: 최대 처리할 포스트 수 (None 이면 전체)
            company_keyword: 회사명에 포함되어야 할 키워드 (예: "naver"). None 이면 전체 회사 대상
        """
        session = self.SessionLocal()

        try:
            # 기본: Post 테이블에서 시작
            query = session.query(Post)

            # 회사 키워드가 없는 경우에만
            # "아직 일정이 추출되지 않은 포스트"로 한정
            if not company_keyword:
                query = query.outerjoin(
                    RecruitSchedule,
                    Post.id == RecruitSchedule.post_id
                ).filter(
                    RecruitSchedule.schedule_id.is_(None)
                )

            if company_keyword:
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

    # 회사명 키워드 입력 (예: "naver", "카카오" 등)
    company_keyword = input("회사명 키워드 (예: naver, kakao; 전체 대상이면 엔터): ").strip() or None

    print("=" * 60)
    print("채용 일정 추출 시작")
    print("=" * 60)

    # 상위 N개만 시험용으로 처리 (배치 크기: 10, limit=20)
    processor.process_posts(batch_size=10, limit=200, company_keyword=company_keyword)

    print("\n" + "=" * 60)
    print("처리 완료")
    print("=" * 60)

    # 특정 포스트의 결과 확인 (예시)
    print("\n[결과 확인 예시]")
    schedule = processor.get_schedule_by_post_id(1)
    if schedule:
        print(json.dumps(schedule, indent=2, ensure_ascii=False))
    else:
        print("Post ID 1의 일정 정보가 없습니다.")


if __name__ == "__main__":
    main()
