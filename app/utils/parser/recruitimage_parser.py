import pytesseract
from PIL import Image
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


def resolve_path(path: str | Path) -> Path:
    """주어진 경로를 프로젝트 루트를 기준으로 절대경로로 변환."""
    target = Path(path)
    if target.is_absolute():
        return target
    return (PROJECT_ROOT / target).resolve()


def load_all_skills(description_json_path: str | Path) -> List[str]:
    """description.json에서 전체(canonical) 스킬셋 목록을 불러옴."""
    path = resolve_path(description_json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    skill_set = set()
    if isinstance(data, dict):
        skill_set.update(data.get("공통_skill_set", []))
        skill_set.update(data.get("skill_set", []))
    elif isinstance(data, list):
        for job in data:
            skill_set.update(job.get("공통_skill_set", []))
            s = job.get("skill_set", [])
            if isinstance(s, list):
                skill_set.update(s)
            elif isinstance(s, str):
                skill_set.update([x.strip() for x in re.split("[,/]", s) if x.strip()])
    return sorted([s for s in skill_set if s and isinstance(s, str)])

# ------------------------------------------------------------------------------

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_default_description_json_path() -> Path:
    env_val = os.getenv("DESCRIPTION_JSON_PATH")
    if env_val:
        env_path = resolve_path(env_val)
        if env_path.is_file():
            return env_path

    candidates = [
        DATA_DIR / "description.json",
        Path.cwd() / "data" / "description.json",
        Path.cwd().parent / "data" / "description.json",
        Path.cwd().parent.parent / "data" / "description.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return DATA_DIR / "description.json"


DEFAULT_DESCRIPTION_JSON_PATH = get_default_description_json_path()

if not DEFAULT_DESCRIPTION_JSON_PATH.is_file():
    print(f"⚠️ 경고: description.json 파일을 찾을 수 없습니다.")
    print(f"   찾은 경로: {DEFAULT_DESCRIPTION_JSON_PATH}")
    print(f"   현재 워킹 디렉토리: {os.getcwd()}")
    print(f"   환경 변수 DESCRIPTION_JSON_PATH를 설정하거나,")
    print(f"   올바른 경로를 직접 지정해주세요.")
    print(f"   예상 경로: data/description.json (프로젝트 루트 기준)")

# 1. Pydantic 모델 정의
class WorkConditions(BaseModel):
    flexible_working_hours: Optional[str] = Field(None, description="근무 시간 제도")
    recovery_day: Optional[str] = Field(None, description="리커버리데이 정보")
    remote_work: Optional[str] = Field(None, description="원격근무 정보")

class MetaData(BaseModel):
    job_category: Optional[str] = Field(None, description="직무 카테고리")
    preferred_qualifications: List[str] = Field(default_factory=list, description="우대사항 목록")
    work_conditions: Optional[WorkConditions] = Field(None, description="근무 조건")

class SkillSetInfo(BaseModel):
    matched: bool = Field(False, description="스킬셋 매칭 여부")
    match_score: Optional[int] = Field(None, description="매칭 점수")
    skill_set: List[str] = Field(default_factory=list, description="기술 스택 목록")

class JobPosting(BaseModel):
    title: str = Field(..., description="채용 공고 제목")
    company: str = Field(..., description="회사명")
    location: Optional[str] = Field(None, description="근무 지역")
    employment_type: Optional[str] = Field(None, description="고용 형태 (정규직, 계약직 등)")
    experience: Optional[str] = Field(None, description="경력 요구사항 (신입, 경력 등)")
    crawl_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"), description="크롤링 날짜")
    posted_date: Optional[str] = Field(None, description="공고 게시일")
    expired_date: Optional[str] = Field(None, description="공고 마감일")
    description: str = Field(..., description="채용 공고 전체 내용")
    url: Optional[str] = Field(None, description="채용 공고 URL")
    meta_data: MetaData = Field(default_factory=MetaData, description="메타 데이터")
    screenshots: Optional[Dict[str, str]] = Field(None, description="스크린샷 경로")
    skill_set_info: SkillSetInfo = Field(default_factory=SkillSetInfo, description="스킬셋 정보")

# 2. OCR 및 텍스트 복원 함수
def extract_and_refine_text(image_path: str | Path) -> str:
    """이미지에서 텍스트 추출 및 정제"""
    image = Image.open(resolve_path(image_path))
    ocr_text = pytesseract.image_to_string(image, lang='kor+eng')
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)
    # --------- 프롬프트: @file_context_0 + OCR 정제 ---------
    system_prompt = (
        "@file_context_0\n"
        "아래 채용공고의 OCR 텍스트를 한국어로 자연스럽고 정보 손실 없이 채용 공고 원문으로 복원해서 출력하세요."
        "\n[OCR 추출 원문]\n"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ocr_text},
        ],
        temperature=0.1,
        max_tokens=4096,
    )
    return response.choices[0].message.content

# 3. LangChain 파서 설정
def create_job_parser():
    """채용공고 파싱을 위한 LangChain 파서 생성"""
    parser = PydanticOutputParser(pydantic_object=JobPosting)
    # --- 프롬프트 company 이름 잘 추출하도록 강화 ---
    prompt_template = """
아래는 제공된 '사용 가능한 스킬 목록'입니다. 이 목록(표준 표기법, canonical names)을 참고해서 채용공고 텍스트에서 등장하는 기술 스택만을 정확히 추출하세요.
채용공고 내에 있는 기술명이 provided skill 목록과 다소 다르더라도 (오탈자, 대소문자, 띄어쓰기, 동의어 등) 모두 provided 목록의 canonical 명칭으로 맞춰 mapping해서 출력해야 합니다.
예시: Node, nodejs, node.js → Node.js | PyTorch, Pytorch, pytorch → PyTorch
소프트스킬, 도메인, 업무방식 등은 반드시 제외하세요.

사용 가능한 스킬 목록: {all_skills}

당신은 채용공고를 분석하는 전문가입니다.
아래 채용공고 텍스트를 분석하여 구조화된 JSON 형식으로 변환하세요.

특히 "회사명(company)" 추출을 주의하세요.
- 회사명은 상단, 하단, 또는 본문 어딘가에 명확하게 적혀있지 않을 수도 많습니다.
- 텍스트 내에 회사 oo, ○○회사, (주)○○, ㈜○○, ○○주식회사, ○○ Inc, ○○ Corp 등 회사명을 연상케 하는 부분이 있으면 꼭 찾아서 넣어주세요.
- 명칭 오타, 괄호( ), 주식회사 문구, 영문 표기, 회사 로고 근처, 또는 이메일/홈페이지 주소 등에서 유추되는 명칭 등을 적극적으로 추정하여 'company' 필드에 넣어주세요.
- 특별히 명시된 회사명이 없더라도, 텍스트 내에서 회사명을 "추정"할 수 있으면 candidate로라도 반드시 넣어주세요.
- 만약 여러 후보가 있다면, 가장 가능성 높은 1개만 넣으세요.
- 회사명이 명확하지 않을 때는, 텍스트의 상단/하단/이메일/주소/대표전화/소개문구 등 회사 관련된 부분을 자세히 살펴 추정해주세요.
- 회사명 추출이 가능한 근거가 있다면, 그 부분의 일부 문구를 그대로 사용해도 좋습니다.

채용공고 원문:
{description}

추가 정보:
- 이미지 경로: {image_path}

다음 가이드라인을 따라주세요:
1. 직무 제목, 회사명, 위치, 고용 형태, 경력 요구사항을 정확히 추출하세요.
   - 특히 회사명(company)의 경우 문서 어디라도 연관성 있는 단어/문장/이메일/주소/상호명 등의 후보를 적극적으로 탐색해서 반드시 추정해서 추출해주세요. 주식회사·㈜·(주)·Inc·Corp·회사·컴퍼니 등이 붙는 곳, 또는 로고 주변, 또는 문장 내에서 후보가 되는 부분을 주요 candidate로 사용하세요.
2. 우대사항은 bullet point나 "우대사항" 섹션에서 추출하세요.
3. 기술 스택(skill_set)은 '사용 가능한 스킬 목록'을 바탕으로 구체적인 기술명을 canonical 명칭으로 추출하세요. 만약에 없다면 채용공고 원문에서 키워드 추정해서 넣어주세요.
4. 근무 조건(work_conditions)은 "근로제도", "근무시간", "원격근무" 관련 정보를 추출하세요.
5. 직무 카테고리(job_category)는 AI/ML, Backend, Frontend 등으로 분류하세요.
6. description 필드에는 전체 채용공고 원문을 그대로 포함하세요.
7. posted_date와 expired_date는 "YYYY-MM-DD" 형식으로 추출하세요. 정보가 없으면 null로 설정하세요.

{format_instructions}
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["description", "image_path", "all_skills"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    return prompt, parser

# 4. 메인 파싱 함수 (skillset 재추출 과정 없음)
def parse_job_posting(
    image_path: str | Path,
    use_ocr: bool = True,
    description_json_path: Optional[str] = None,
) -> JobPosting:
    """
    채용공고 이미지를 파싱하여 구조화된 데이터로 변환

    Args:
        image_path: 채용공고 이미지 경로
        use_ocr: OCR 사용 여부
        description_json_path: 전체 스킬셋 목록을 담은 description.json 경로 (생략 가능)

    Returns:
        JobPosting: 파싱된 채용공고 데이터
    """
    # OCR 및 텍스트 정제
    image_path = resolve_path(image_path)

    if use_ocr:
        description = extract_and_refine_text(image_path)
    else:
        txt_path = image_path.with_suffix(".txt")
        with txt_path.open('r', encoding='utf-8') as f:
            description = f.read()

    # 반드시 all_skills를 먼저 로드
    candidate_path = description_json_path or os.getenv("DESCRIPTION_JSON_PATH")
    skill_path = resolve_path(candidate_path) if candidate_path else DEFAULT_DESCRIPTION_JSON_PATH
    if not skill_path.is_file():
        raise FileNotFoundError(
            f"description.json 파일을 찾을 수 없습니다.\n"
            f"  찾은 경로: {skill_path}\n"
            f"  현재 워킹 디렉토리: {os.getcwd()}\n"
            f"  올바른 경로: data/description.json (프로젝트 루트 기준)\n"
            f"  환경 변수 DESCRIPTION_JSON_PATH를 설정하거나 description_json_path 파라미터로 직접 지정해주세요."
        )
    all_skills = load_all_skills(skill_path)
    all_skills_str = ", ".join(all_skills)

    # LangChain 파서로 구조화 (스킬목록 전달)
    prompt, parser = create_job_parser()
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        openai_api_key=openai_api_key
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "description": description,
        "image_path": str(image_path),
        "all_skills": all_skills_str
    })
    result.screenshots = {"combined": str(image_path)}

    return result

# 5. 사용 예시
if __name__ == "__main__":
    image_path = DATA_DIR / "skax_job_R251754.png"
    description_json_path = DEFAULT_DESCRIPTION_JSON_PATH  # 위에서 자동 세팅됨

    try:
        job_data = parse_job_posting(image_path, description_json_path=description_json_path)
        job_dict = job_data.model_dump()
        print(json.dumps(job_dict, ensure_ascii=False, indent=2))
        output_path = image_path.with_name(f"{image_path.stem}_parsed.json")
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(job_dict, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 파싱 완료! 결과 저장: {output_path}")
    except Exception as e:
        print(f"❌ 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()