"""
구체성(Specificity) 측정 도구
직무 기술서의 구체성 평가를 위한 LangChain 도구들
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from app.schemas.agent import (
    ResponsibilityResult,
    QualificationResult,
    KeywordRelevanceResult,
    RequiredFieldsResult
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
def measure_responsibility_specificity(job_description: str) -> dict:
    """담당 업무 설명에서 기술 용어를 제외한 설명 부분의 글자 수를 측정하여 업무 설명이 얼마나 구체적인지 평가합니다."""
    print(f"[TOOL CALL] measure_responsibility_specificity called")
    prompt_template = """
다음 채용 공고에서 "담당 업무" 또는 "주요 업무" 섹션을 찾아 분석해주세요.

[채용 공고]
{job_description}

[작업 지침]
1. 담당 업무/주요 업무 섹션의 텍스트를 추출합니다
2. 해당 텍스트에서 기술 용어(HVAC, SCADA, React, Python 등)를 식별합니다
3. original_text에는 담당 업무 섹션의 원문 텍스트를 포함합니다
4. keywords에는 발견된 모든 기술 용어를 리스트로 포함합니다
5. keyword_count에는 keywords 리스트의 개수를 포함합니다
6. reasoning에는 왜 이 기술 용어들이 식별되었는지, 담당 업무가 얼마나 구체적인지에 대한 상세한 근거와 설명을 **한국어로** 작성합니다

예시:
"HVAC, SCADA, WWT, Chemical/GAS 공급장치 제어 등의 Domain Knowledge를 보유하신 분"
→ 기술 용어: "HVAC, SCADA, WWT, Chemical/GAS"
→ 설명: "공급장치 제어 등의 Domain Knowledge를 보유하신 분"

**중요: 모든 응답은 반드시 한국어로 작성해야 합니다.**
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_with_structure = llm.with_structured_output(ResponsibilityResult)
    chain = prompt | llm_with_structure

    result = chain.invoke({"job_description": job_description})

    # Pydantic 모델을 딕셔너리로 변환하여 반환
    return result.model_dump()


@tool
def measure_qualification_specificity(job_description: str) -> dict:
    """자격요건 및 우대사항 설명의 단어 수를 측정하여 얼마나 구체적으로 기술되어 있는지 평가합니다."""
    print(f"[TOOL CALL] measure_qualification_specificity called")
    prompt_template = """
다음 채용 공고에서 "자격요건", "필수 자격", "우대사항", "우대조건" 섹션을 찾아 분석해주세요.

[채용 공고]
{job_description}

[작업 지침]
1. 필수 자격요건 섹션의 텍스트를 추출합니다
2. 우대사항 섹션의 텍스트를 추출합니다
3. original_text에는 자격요건 및 우대사항 섹션의 원문 텍스트를 포함합니다
4. keywords에는 자격요건 및 우대사항의 주요 키워드나 항목을 리스트로 포함합니다
5. keyword_count에는 keywords 리스트의 개수를 포함합니다
6. reasoning에는 자격요건과 우대사항이 얼마나 구체적으로 기술되어 있는지에 대한 상세한 근거와 설명을 **한국어로** 작성합니다

**중요: 모든 응답은 반드시 한국어로 작성해야 합니다.**
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_with_structure = llm.with_structured_output(QualificationResult)
    chain = prompt | llm_with_structure

    result = chain.invoke({"job_description": job_description})

    # Pydantic 모델을 딕셔너리로 변환하여 반환
    return result.model_dump()


@tool
def measure_keyword_relevance(job_description: str) -> dict:
    """직군을 판단하고 담당 업무의 키워드가 해당 직군과 관련 있는지 평가합니다."""
    print(f"[TOOL CALL] measure_keyword_relevance called")
    prompt_template = """
다음 채용 공고를 분석하여 직군을 판단하고, 담당 업무의 키워드가 해당 직군과 관련 있는지 평가해주세요.

[채용 공고]
{job_description}

[작업 지침]
1. 직무명, 타이틀, 조직 소개를 보고 직군을 판단합니다 (예: AI 개발자, 프론트엔드 개발자, HR 등)
2. 담당 업무에서 주요 키워드를 추출합니다
3. original_text에는 담당 업무 섹션의 원문 텍스트를 포함합니다
4. keywords에는 담당 업무의 주요 키워드를 리스트로 포함합니다
5. keyword_count에는 keywords 리스트의 개수를 포함합니다
6. reasoning에는 각 키워드가 직군과 관련이 있는지, 왜 그렇게 판단했는지에 대한 상세한 근거와 설명을 **한국어로** 작성합니다

예시:
- "AI 기반 Application 개발/설계 분야 전문가" → 직군: AI 개발자
  → "Python, TensorFlow, 모델 개발" ⇒ 관련 있음 (good)
  → "인사 평가, 채용 프로세스" ⇒ 관련 없음 (bad)

**중요: 모든 응답은 반드시 한국어로 작성해야 합니다.**
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_with_structure = llm.with_structured_output(KeywordRelevanceResult)
    chain = prompt | llm_with_structure

    result = chain.invoke({"job_description": job_description})

    # Pydantic 모델을 딕셔너리로 변환하여 반환
    return result.model_dump()


@tool
def measure_required_fields_count(job_description: str) -> dict:
    """직무명, 담당업무, 자격요건, 우대사항, 복리후생, 근무환경, 채용절차 등 필수 항목이 포함되어 있는지 확인합니다."""
    print(f"[TOOL CALL] measure_required_fields_count called")
    required_fields = {
        "직무명": ["직무명", "포지션", "채용 직무", "모집 직무"],
        "담당업무": ["담당업무", "주요업무", "업무내용", "수행업무"],
        "자격요건": ["자격요건", "필수자격", "지원자격", "응시자격"],
        "우대사항": ["우대사항", "우대조건", "우대요건"],
        "복리후생": ["복리후생", "혜택", "복지"],
        "근무환경": ["근무환경", "업무환경", "근무조건"],
        "채용절차": ["채용절차", "전형절차", "채용과정", "지원방법"]
    }

    prompt_template = """
다음 채용 공고에서 각 필수 항목이 포함되어 있는지 확인해주세요.

[채용 공고]
{job_description}

[필수 항목]
{fields_list}

[작업 지침]
1. 각 항목이 채용 공고에 명시적으로 포함되어 있는지 확인합니다
2. 제목이나 내용에 해당 항목이 있으면 포함된 것으로 간주합니다
3. original_text에는 필수 항목이 포함된 원문 텍스트를 포함합니다
4. keywords에는 포함된 필수 항목명 또는 누락된 항목명을 리스트로 포함합니다
5. keyword_count에는 keywords 리스트의 개수를 포함합니다
6. reasoning에는 각 항목이 포함되었는지 또는 누락되었는지 판단한 근거와 설명을 **한국어로** 작성합니다

**중요: 모든 응답은 반드시 한국어로 작성해야 합니다.**
"""

    fields_list = "\n".join([f"- {field}: {', '.join(variants)}" for field, variants in required_fields.items()])

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_with_structure = llm.with_structured_output(RequiredFieldsResult)
    chain = prompt | llm_with_structure

    result = chain.invoke({
        "job_description": job_description,
        "fields_list": fields_list
    })

    # Pydantic 모델을 딕셔너리로 변환하여 반환
    return result.model_dump()

