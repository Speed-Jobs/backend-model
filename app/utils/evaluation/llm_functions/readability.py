"""
가독성(Readability) 측정 도구
직무 기술서의 가독성 평가를 위한 LangChain 도구들
"""

import re
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from app.schemas.agent import (
    JargonResult,
    ConsistencyResult,
    GrammarResult
)

load_dotenv(override=True)

# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=16384,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


@tool
def measure_company_jargon_frequency(job_description: str, company_name: str = "") -> dict:
    """채용 공고에서 회사 고유의 사내 전문 용어 빈도수를 측정합니다. 회사 고유 용어, 사내 시스템명, 특별한 프로그램명 등을 찾아 개수와 비율을 반환합니다."""
    print(f"[TOOL CALL] measure_company_jargon_frequency called for company: {company_name}")
    prompt_template = """
다음 채용 공고에서 **외부에서 모르는 사내 내부 전문 용어**만 찾아주세요.

[채용 공고]
{job_description}

[회사명]
{company_name}

[작업 지침]
1. **포함해야 할 것**:
   - 회사 내부에서만 쓰이는 고유 용어 (예: "AI Default Company", "리커버리데이", "통화 트라이브", "크루")
   - 사내 시스템명, 내부 프로그램명
   - 회사 고유의 복리후생 제도명
   - 사내 조직명, 팀명 (일반적이지 않은 것)

2. **제외해야 할 것**:
   - 외부에서도 쓰이는 서비스명 (예: "보이스톡", "페이스톡", "카카오톡")
   - 일반적인 IT 용어 (예: "WebRTC", "Admin", "API")
   - 업계 표준 용어 (예: "완전선택근무제" - 일반적인 근무 제도)

3. original_text에는 전문 용어가 발견된 원문 텍스트를 포함합니다
4. keywords에는 발견된 모든 전문 용어를 리스트로 포함합니다
5. keyword_count에는 keywords 리스트의 개수를 포함합니다
6. reasoning에는 왜 이 용어들이 사내 전문 용어로 판단되었는지, 어떤 기준으로 포함/제외했는지에 대한 상세한 근거와 설명을 **한국어로** 작성합니다

예시:
- "AI First" → SK의 사내 전략 용어
- "TDS(Toss Design System)" → 토스의 내부 디자인 시스템
- "리커버리데이" → 회사 고유 복리후생 제도

전문 용어가 없으면 original_text는 빈 문자열, keywords는 빈 리스트, keyword_count는 0을 반환하고, reasoning에는 왜 전문 용어가 없다고 판단했는지 **한국어로** 설명하세요.

**중요: 모든 응답은 반드시 한국어로 작성해야 합니다.**
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_with_structure = llm.with_structured_output(JargonResult)
    chain = prompt | llm_with_structure

    result = chain.invoke({
        "job_description": job_description,
        "company_name": company_name or "정보 없음"
    })

    # Pydantic 모델을 딕셔너리로 변환하여 반환
    return result.model_dump()


@tool
def measure_paragraph_consistency(job_description: str) -> dict:
    """채용 공고의 각 섹션 내 문단의 맥락 일관성을 판단합니다. 섹션 제목과 내용이 일치하는지 확인합니다."""
    print(f"[TOOL CALL] measure_paragraph_consistency called")
    prompt_template = """
다음 채용 공고를 분석하여 각 섹션의 문단 일관성을 평가해주세요.

[채용 공고]
{job_description}

[작업 지침]
1. 주요 섹션들을 식별합니다 (담당업무, 자격요건, 우대사항, 복리후생, 근무환경 등)
2. 각 섹션의 내용이 섹션 제목과 일관되는지 확인합니다
3. 섹션 제목과 맞지 않는 내용이 있는지 찾습니다
4. original_text에는 불일치 문제가 발견된 원문 텍스트를 포함합니다
5. keywords에는 불일치 문제가 있는 섹션명 또는 문제 키워드를 리스트로 포함합니다
6. keyword_count에는 keywords 리스트의 개수를 포함합니다
7. reasoning에는 왜 이 섹션들이 불일치로 판단되었는지, 어떤 기준으로 평가했는지에 대한 상세한 근거와 설명을 **한국어로** 작성합니다

예시 문제:
- "업무 환경 및 문화" 섹션에 "React 기반 프론트엔드 UI/UX를 구현합니다" (담당업무 내용) ⇒ 불일치
- "담당 업무" 섹션에 복리후생 내용이 포함 ⇒ 불일치

문제가 없으면 original_text는 빈 문자열, keywords는 빈 리스트, keyword_count는 0을 반환하고, reasoning에는 왜 일관성이 유지되고 있다고 판단했는지 **한국어로** 설명하세요.

**중요: 모든 응답은 반드시 한국어로 작성해야 합니다.**
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_with_structure = llm.with_structured_output(ConsistencyResult)
    chain = prompt | llm_with_structure

    result = chain.invoke({"job_description": job_description})

    # Pydantic 모델을 딕셔너리로 변환하여 반환
    return result.model_dump()


@tool
def measure_grammar_accuracy(job_description: str) -> dict:
    """채용 공고의 맞춤법, 띄어쓰기, 문장 구조 오류를 검사합니다."""
    print(f"[TOOL CALL] measure_grammar_accuracy called")
    prompt_template = """
다음 채용 공고의 맞춤법, 띄어쓰기, 문장 구조 오류를 찾아주세요.

[채용 공고]
{job_description}

[검사 항목]
1. 맞춤법 오류
2. 띄어쓰기 오류
3. 문장 구조 오류
4. 불필요한 중복 표현

5. original_text에는 문법 오류가 발견된 원문 텍스트를 포함합니다
6. keywords에는 문법 오류가 있는 표현 또는 오류 유형을 리스트로 포함합니다
7. keyword_count에는 keywords 리스트의 개수를 포함합니다
8. reasoning에는 왜 이 표현들이 문법 오류로 판단되었는지, 어떤 기준으로 검사했는지에 대한 상세한 근거와 설명을 **한국어로** 작성합니다

오류가 없으면 original_text는 빈 문자열, keywords는 빈 리스트, keyword_count는 0을 반환하고, reasoning에는 왜 문법 오류가 없다고 판단했는지 **한국어로** 설명하세요.

**중요: 모든 응답은 반드시 한국어로 작성해야 합니다.**
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_with_structure = llm.with_structured_output(GrammarResult)
    chain = prompt | llm_with_structure

    result = chain.invoke({"job_description": job_description})

    # Pydantic 모델을 딕셔너리로 변환하여 반환
    return result.model_dump()

