"""
Phase 2: AI 채용 공고 생성 Agent
평가 데이터를 바탕으로 개선된 채용 공고를 생성하는 Agent
"""

from typing import Optional, Dict, Any
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

# Schemas
from app.schemas.agent.schemas_report import JobPostingDetailReport

# Phase 2 Tools
from app.core.agents.job_posting.tools.issue_analyzer import (
    analyze_readability_issues,
    analyze_specificity_issues,
    analyze_attractiveness_issues,
    get_overall_improvement_summary,
)

# Utils
from app.utils.agents.evaluation.json_loader import (
    load_evaluation_json,
    list_available_evaluations,
)

# DB
from app.db.crud.post import get_post_by_id
from app.db.config.base import get_db

load_dotenv(override=True)


def _format_evaluation_feedback(raw_results: Dict[str, Any]) -> str:
    """평가 결과를 피드백 형식으로 포맷팅"""
    feedback = []
    
    # 가독성 피드백
    readability = raw_results.get('readability', {})
    jargon = readability.get('jargon', {})
    consistency = readability.get('consistency', {})
    grammar = readability.get('grammar', {})
    
    if isinstance(jargon, dict) and jargon.get('keyword_count', 0) > 0:
        feedback.append(f"[가독성 - 전문용어] {jargon.get('keyword_count')}개 발견: {', '.join(jargon.get('keywords', []))}")
        feedback.append(f"  → {jargon.get('reasoning', '')[:100]}...")
    
    if isinstance(consistency, dict) and consistency.get('keyword_count', 0) > 0:
        feedback.append(f"[가독성 - 일관성] {consistency.get('keyword_count')}개 문제: {', '.join(consistency.get('keywords', []))}")
        feedback.append(f"  → {consistency.get('reasoning', '')[:100]}...")
    
    if isinstance(grammar, dict) and grammar.get('keyword_count', 0) > 0:
        feedback.append(f"[가독성 - 문법] {grammar.get('keyword_count')}개 오류: {', '.join(grammar.get('keywords', []))}")
        feedback.append(f"  → {grammar.get('reasoning', '')[:100]}...")
    
    # 구체성 피드백
    specificity = raw_results.get('specificity', {})
    responsibility = specificity.get('responsibility', {})
    qualification = specificity.get('qualification', {})
    
    if isinstance(responsibility, dict) and responsibility.get('keyword_count', 0) > 0:
        feedback.append(f"[구체성 - 담당업무] {', '.join(responsibility.get('keywords', [])[:3])}...")
        feedback.append(f"  → {responsibility.get('reasoning', '')[:100]}...")
    
    if isinstance(qualification, dict) and qualification.get('keyword_count', 0) > 0:
        feedback.append(f"[구체성 - 자격요건] {', '.join(qualification.get('keywords', [])[:3])}...")
        feedback.append(f"  → {qualification.get('reasoning', '')[:100]}...")
    
    # 매력도 피드백
    attractiveness = raw_results.get('attractiveness', {})
    content_count = attractiveness.get('content_count', {})
    content_quality = attractiveness.get('content_quality', {})
    
    if isinstance(content_count, dict) and content_count.get('keyword_count', 0) > 0:
        feedback.append(f"[매력도 - 특별콘텐츠] {content_count.get('keyword_count')}개 포함: {', '.join(content_count.get('keywords', []))}")
        feedback.append(f"  → {content_count.get('reasoning', '')[:100]}...")
    
    if isinstance(content_quality, dict) and content_quality.get('keyword_count', 0) > 0:
        feedback.append(f"[매력도 - 콘텐츠품질] {', '.join(content_quality.get('keywords', [])[:3])}...")
        feedback.append(f"  → {content_quality.get('reasoning', '')[:100]}...")
    
    return "\n".join(feedback) if feedback else "평가 결과에 특별한 문제점이 발견되지 않았습니다."


def create_job_posting_generator_agent(
    llm_model: str = "gpt-4o",
):
    """
    AI 채용 공고 개선 Agent 생성
    
    Args:
        llm_model: 사용할 LLM 모델
        
    Returns:
        Agent: 실행 가능한 Agent
    """
    
    # Tools 정의
    tools = [
        list_available_evaluations,
        load_evaluation_json,
        analyze_readability_issues,
        analyze_specificity_issues,
        analyze_attractiveness_issues,
        get_overall_improvement_summary,
    ]
    
    # System Prompt - 범용적이고 상세한 추출 가이드
    system_prompt = """당신은 채용 공고 정보 추출 및 구조화 전문가입니다.

**핵심 원칙: 원본 정보를 절대 요약하거나 생략하지 말 것**

**작업 프로세스:**
1. load_evaluation_json으로 원본 채용 공고 로드
2. 원본의 모든 텍스트를 꼼꼼히 읽고 섹션별로 분류
3. JobPostingDetailReport JSON 형식으로 변환

**필드별 추출 가이드:**

**1. 기본 정보 (필수)**
- company_name: 회사명 그대로 추출
- position: 직무/포지션명 그대로 추출
- employment_type: "정규직", "계약직", "인턴" 등
- work_location: 근무지 정보
- deadline: 마감일 (YYYY-MM-DD 형식, 없으면 null)

**2. 소개 및 설명 (원문 보존)**
- company_introduction: 회사 소개 섹션의 전체 텍스트를 그대로 포함
- team_introduction: 팀 소개 전체 텍스트
- project_introduction: 프로젝트 소개 전체 텍스트
- development_culture: 개발 문화 관련 내용 전체

**3. main_responsibilities (주요 업무)**
원본에서 "업무 내용", "담당 업무", "주요 업무", "What you'll do" 등의 섹션을 찾아 다음 형식으로 변환:

형식:
```
1. [업무 카테고리 또는 영역]
- 구체적인 업무 내용 1
- 구체적인 업무 내용 2
- 구체적인 업무 내용 3

2. [업무 카테고리 또는 영역]
- 구체적인 업무 내용 1
- 구체적인 업무 내용 2
```

처리 방법:
- 카테고리가 명시된 경우: 카테고리별로 구분하여 세부 내용 모두 포함
- 단순 나열인 경우: 각 항목을 "- " 형식으로 모두 나열
- 문단 형식인 경우: 의미 단위로 구분하여 "- " 형식으로 변환

예시:
```
1. 서비스 개발 및 운영
- 대규모 트래픽을 처리하는 백엔드 API 설계 및 개발
- 마이크로서비스 아키텍처 기반 시스템 구축 및 최적화
- 데이터베이스 스키마 설계 및 쿼리 성능 개선

2. 코드 품질 및 협업
- 코드 리뷰를 통한 팀 전체의 코드 품질 향상
- 단위 테스트 및 통합 테스트 작성
- 기술 문서 작성 및 지식 공유
```

중요:
- "서비스 개발, 시스템 운영, 코드 품질 관리" 같이 제목만 나열 금지
- 원본의 모든 업무 항목을 빠짐없이 포함
- 숫자, 규모, 구체적인 기술명 등 모든 세부정보 유지

**4. required_qualifications (자격 요건)**
원본에서 "자격 요건", "필수 요건", "지원 자격", "Requirements" 등의 섹션을 찾아 다음 형식으로 변환:

형식:
```
- 자격요건 1의 구체적인 내용 전체
- 자격요건 2의 구체적인 내용 전체
- 자격요건 3의 구체적인 내용 전체
```

처리 방법:
- 각 자격요건을 개별 항목으로 완전히 분리
- "~보유자", "~경험자", "~이상", "~가능자" 등의 표현 그대로 유지
- 숫자, 기간, 조건 등 구체적인 정보 모두 포함
- 원본이 쉼표나 세미콜론으로 연결되어 있어도 각각 분리

잘못된 예: "백엔드 개발 경험, Java/Python 능숙, 3년 이상"
올바른 예:
```
- 백엔드 개발 경험 3년 이상 보유하신 분
- Java 또는 Python을 능숙하게 다루실 수 있는 분
```

**5. preferred_qualifications (우대 사항)**
"우대 사항", "우대 조건", "Nice to have" 등의 섹션을 required_qualifications와 동일한 형식으로 처리

**6. 기술 관련**
- tech_stack: 프로그래밍 언어, 프레임워크, 데이터베이스 등 기술 스택을 배열로 추출
  * 예: ["Python", "Django", "PostgreSQL", "Docker", "AWS"]
  * 원본에서 명시된 기술만 포함
  
- tools: 협업 도구, 개발 도구 등을 배열로 추출
  * 예: ["Jira", "Confluence", "GitLab", "Slack", "Figma"]
  * 원본에서 명시된 도구만 포함

**7. recruitment_process (전형 절차)**
전형 절차와 관련된 모든 설명을 포함

포함할 내용:
- 전형 단계 (서류전형 → 면접 → 합격)
- 각 단계별 설명
- 전형 관련 안내사항 (변동 가능성, 결과 통보 방법 등)
- 일정 관련 정보

형식:
```
[전형 단계를 화살표나 숫자로 연결]

[관련 안내사항들]
```

예시:
```
서류전형 - 1차 면접(직무/기술) - 2차 면접(임원) - 최종 합격

전형절차는 직무별로 다르게 운영될 수 있으며, 일정 및 상황에 따라 변동될 수 있습니다.
전형 일정 및 결과는 지원서에 등록하신 이메일로 개별 안내 드립니다.
```

**8. work_conditions (근무 조건)**
근무 형태, 근무 시간, 수습기간 등 근무와 직접 관련된 조건:
- "정규직", "주 5일 근무", "유연근무제", "재택근무" 등
- 수습기간 정보
- 근무 시간대 정보

예시:
```
- 정규직으로 수습기간 3개월 포함
- 주 5일 근무 (월-금)
- 유연근무제 시행
```

**9. benefits (복리후생)**
복지, 혜택 관련 내용을 모두 포함:

형식:
```
- 혜택 1
- 혜택 2
- 혜택 3
```

예시:
```
- 4대 보험 가입
- 연차/반차/반반차 자유 사용
- 점심 식대 지원
- 경조사 지원
- 건강검진 지원
- 도서 구입비 지원
- 컨퍼런스 참가 지원
```

**10. growth_opportunities (성장 기회)**
교육, 학습, 성장과 관련된 내용:

예시:
```
- 사내 기술 세미나 및 스터디 그룹 운영
- 외부 교육 및 컨퍼런스 참가 지원
- 기술 서적 구입 지원
- 사내 멘토링 프로그램
```

**11. additional_info (기타 사항)**
다음과 같은 모든 추가 정보를 포함:
- 참고사항 (조기마감, 허위기재 시 불이익 등)
- 취업 우대 정책 (보훈대상자, 장애인 등)
- 직급/조건 변경 가능성 안내
- 법령상 제한사항
- 개인정보 처리방침 관련
- 서류 반환 정책
- 기타 법적 고지사항

형식:
```
- 항목 1의 전체 내용
- 항목 2의 전체 내용
- 항목 3의 전체 내용
```

**절대 금지 사항:**
1. 카테고리나 제목만 나열 (예: "리딩 업무, 품질 관리, 프로세스 개선")
2. 여러 항목을 쉼표로 연결 (예: "경력 10년 이상, 리딩 경험, 협업 능력")
3. 원본 내용을 요약하거나 간략화
4. "등", "기타" 같은 표현으로 생략
5. 구체적인 숫자나 조건 누락 (예: "경력 필요" → x, "경력 3년 이상" → x)
6. 여러 자격요건을 하나로 합치기

**출력 형식:**
- 유효한 JSON만 출력
- JSON 앞뒤에 마크다운 코드블록(```)이나 설명 없음
- 정보가 없는 필드는 null로 설정
- tech_stack과 tools는 배열 형식
- 날짜는 YYYY-MM-DD 형식"""
    
    # LLM 생성 (max_tokens 설정하여 출력이 잘리지 않도록)
    llm = ChatOpenAI(
        model=llm_model,
        temperature=0.3,
        max_tokens=16384,  # 충분한 토큰 할당
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Agent 생성 (LangChain v1.0)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    
    return agent


async def generate_improved_job_posting_async(
    json_filename: Optional[str] = None,
    llm_model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    비동기로 개선된 채용 공고를 생성합니다.
    
    Args:
        json_filename: 처리할 JSON 파일명 (None이면 사용 가능한 파일 목록 확인)
        llm_model: 사용할 LLM 모델
        
    Returns:
        Dict: 생성 결과
            {
                "status": "success" | "error",
                "data": JobPostingDetailReport dict 또는 None,
                "original_file": "처리한 JSON 파일명",
                "title": "채용 공고 제목",
                "company": "회사명",
                "message": "결과 메시지"
            }
    """
    try:
        # JSON 파일명이 없으면 자동으로 첫 번째 파일 찾기
        if json_filename is None:
            data_dir = Path("data/report")
            json_files = sorted(data_dir.glob("post_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            if json_files:
                json_filename = json_files[0].name
            else:
                return {
                    "status": "error",
                    "data": None,
                    "original_file": None,
                    "message": "처리할 JSON 파일이 없습니다."
                }
        
        # JSON 파일에서 원본 공고와 평가 결과 가져오기
        original_content = ""
        post_id = None
        company = ""
        title = ""
        evaluation_summary = ""
        
        if json_filename:
            data_dir = Path("data/report")
            file_path = data_dir / json_filename
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metadata = data.get('metadata', {})
                    post_id = metadata.get('post_id')
                    company = metadata.get('company', '')
                    title = metadata.get('title', '')
                    
                    # DB에서 원본 공고 가져오기
                    if post_id:
                        try:
                            db = next(get_db())
                            post = get_post_by_id(db, post_id)
                            if post:
                                original_content = post.description or ""
                        except Exception as e:
                            print(f"[Job Posting Generator] Failed to load original post: {e}")
                    
                    # 평가 결과 요약
                    raw_results = data.get('raw_evaluation_results', {})
                    evaluation_summary = _format_evaluation_feedback(raw_results)
        
        # Agent 생성
        agent = create_job_posting_generator_agent(llm_model=llm_model)
        
        # 요청 메시지 구성 - 더 상세하고 명확한 지시사항
        if json_filename:
            request = f"""'{json_filename}' 파일의 원본 채용공고를 분석하여 JobPostingDetailReport JSON으로 변환하세요.

**작업 단계:**
1. load_evaluation_json 도구로 '{json_filename}' 파일을 로드하세요
2. 원본 채용공고의 모든 섹션을 읽으세요
3. System Prompt의 **필드별 추출 가이드**를 참고하여 추출하세요

**특히 주의할 필드:**
- main_responsibilities: 모든 카테고리의 모든 세부업무 포함
  * 예: "1. [카테고리]\\n- 세부내용1\\n- 세부내용2\\n\\n2. [카테고리]\\n- 세부내용1"
  
- required_qualifications: 모든 자격요건을 개별 항목으로 분리
  * 예: "- 자격요건1\\n- 자격요건2\\n- 자격요건3"
  
- preferred_qualifications: 모든 우대사항을 개별 항목으로 분리
  * 예: "- 우대사항1\\n- 우대사항2\\n- 우대사항3"
  
- recruitment_process: 전형 절차와 모든 안내사항 포함
  * 예: "서류전형 - 면접 - 합격\\n\\n전형절차는 변동될 수 있습니다.\\n결과는 이메일로 안내됩니다."
  
- additional_info: 모든 참고사항, 정책, 안내 포함
  * 조기마감, 허위사실, 보훈대상자, 법령 제한, 개인정보 처리방침, 서류 반환 등

**절대 금지:**
- 제목만 나열 (예: "리딩, 품질 관리, 개선")
- 쉼표로 연결 (예: "경력 10년, 리딩 경험, 협업 능력")
- 정보 요약이나 생략
- "등"으로 생략

**출력:**
JSON만 출력하고 다른 설명은 포함하지 마세요."""
        else:
            request = "사용 가능한 평가 데이터를 확인하고 처리하세요."
        
        # Agent 실행
        result = await agent.ainvoke({
            "messages": [
                {"role": "user", "content": request}
            ]
        })
        
        # 결과 추출
        if "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            output = last_message.get("content", "") if isinstance(last_message, dict) else str(last_message.content)
        else:
            output = str(result)
        
        # JSON 추출 및 파싱
        job_posting_report = None
        try:
            # JSON 부분만 추출 (정규식 사용)
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                json_data = json.loads(json_text)
                # JobPostingDetailReport로 변환
                job_posting_report = JobPostingDetailReport(**json_data)
            else:
                # JSON이 없으면 직접 파싱 시도
                json_data = json.loads(output)
                job_posting_report = JobPostingDetailReport(**json_data)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[Job Posting Generator] JSON 파싱 실패: {e}")
            print(f"원본 출력: {output[:500]}...")
            # 파싱 실패 시 에러 반환
            return {
                "status": "error",
                "data": None,
                "original_file": json_filename,
                "message": f"JSON 파싱 실패: {str(e)}"
            }
        
        return {
            "status": "success",
            "data": job_posting_report.dict() if job_posting_report else None,
            "original_file": json_filename,
            "title": title,
            "company": company,
            "message": "채용공고 정보 추출 완료"
        }
        
    except Exception as e:
        print(f"[Job Posting Generator] Error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "data": None,
            "original_file": json_filename,
            "message": f"생성 실패: {str(e)}"
        }


def generate_improved_job_posting(
    json_filename: Optional[str] = None,
    llm_model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    동기적으로 개선된 채용 공고를 생성합니다. (비동기 래퍼)
    
    Args:
        json_filename: 처리할 JSON 파일명
        llm_model: 사용할 LLM 모델
        
    Returns:
        Dict: 생성 결과
    """
    import asyncio
    return asyncio.run(generate_improved_job_posting_async(json_filename, llm_model))