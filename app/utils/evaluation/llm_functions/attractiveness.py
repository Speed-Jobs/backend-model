"""
매력도(Attractiveness) 측정 도구
직무 기술서의 매력도 평가를 위한 LangChain 도구들
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from app.schemas.agent import SpecialContentInclusionResult, SpecialContentQualityResult

load_dotenv(override=True)

# LLM 초기화
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=16384,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


@tool
def measure_special_content_count(job_description: str) -> dict:
    """현직자 인터뷰, 비전/가치, 이력서 작성 가이드 등 특별 콘텐츠의 포함 여부를 확인합니다."""
    print(f"[TOOL CALL] measure_special_content_count called")
    prompt_template = """
다음 채용 공고에서 특별 콘텐츠가 포함되어 있는지 확인해주세요.

[채용 공고]
{job_description}

[확인할 특별 콘텐츠]
1. 현직자 인터뷰/스토리: 현직자의 경험담, 인터뷰, 실제 업무 이야기 등
2. 비전/가치 제시: 회사의 비전, 핵심 가치, 미션 등
3. 이력서 작성 가이드: 지원서 작성 팁, 포트폴리오 가이드, 면접 준비 안내 등

[작업 지침]
1. 각 콘텐츠가 채용 공고에 명시적으로 포함되어 있는지 확인합니다
2. 단순히 키워드만 있는 것이 아니라, 실제로 의미 있는 내용이 있어야 합니다
3. original_text에는 특별 콘텐츠가 포함된 원문 텍스트를 포함합니다
4. keywords에는 포함된 특별 콘텐츠 유형 또는 누락된 콘텐츠 유형을 리스트로 포함합니다
5. keyword_count에는 keywords 리스트의 개수를 포함합니다
6. reasoning에는 각 콘텐츠가 포함되었는지 또는 누락되었는지 판단한 근거와 설명을 **한국어로** 작성합니다

**중요: 모든 응답은 반드시 한국어로 작성해야 합니다.**
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_with_structure = llm.with_structured_output(SpecialContentInclusionResult)
    chain = prompt | llm_with_structure

    result = chain.invoke({"job_description": job_description})

    # Pydantic 모델을 딕셔너리로 변환하여 반환
    return result.model_dump()


@tool
def measure_special_content_quality(job_description: str) -> dict:
    """특별 콘텐츠(현직자 인터뷰, 비전/가치, 이력서 가이드)의 글자 수를 측정하여 충실도를 평가합니다."""
    print(f"[TOOL CALL] measure_special_content_quality called")
    prompt_template = """
다음 채용 공고에서 특별 콘텐츠의 내용을 추출해주세요.

[채용 공고]
{job_description}

[추출할 콘텐츠]
1. 현직자 인터뷰/스토리: 현직자의 경험담, 인터뷰, 실제 업무 이야기 관련 텍스트
2. 비전/가치 제시: 회사의 비전, 핵심 가치, 미션 관련 텍스트
3. 이력서 작성 가이드: 지원서 작성 팁, 포트폴리오 가이드, 면접 준비 안내 관련 텍스트

[작업 지침]
1. 각 콘텐츠의 실제 텍스트를 추출합니다
2. 해당 콘텐츠가 없으면 빈 문자열을 반환합니다
3. original_text에는 특별 콘텐츠가 포함된 원문 텍스트를 포함합니다
4. keywords에는 특별 콘텐츠 관련 키워드 또는 콘텐츠 유형을 리스트로 포함합니다
5. keyword_count에는 keywords 리스트의 개수를 포함합니다
6. reasoning에는 각 콘텐츠의 충실도와 품질에 대한 상세한 근거와 설명을 **한국어로** 작성합니다

**중요: 모든 응답은 반드시 한국어로 작성해야 합니다.**
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm_with_structure = llm.with_structured_output(SpecialContentQualityResult)
    chain = prompt | llm_with_structure

    result = chain.invoke({"job_description": job_description})

    # Pydantic 모델을 딕셔너리로 변환하여 반환
    return result.model_dump()

