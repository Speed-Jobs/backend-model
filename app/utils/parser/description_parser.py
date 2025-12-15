import fitz  # PyMuPDF
import os
import json
import csv
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


'''
직무 기술서의 내용을 parsing하는 .py 파일
'''

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

pdf_path = r"data\SKAX_Jobdescription.pdf"

# ChatOpenAI 사용
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)

prompt_template = """
[원문]
{context}

[스키마]
직무: string
직무 정의: string
industry: string 
공통_skill_set_description: string
skill_set_description: string
공통_skill_set: string[]
skill_set: string[]

[요구사항]
1) 원문에서 모든 직무의 industry별 블록을 식별하여 각각 1개의 JSON 객체로 출력
2) 각 직무는 여러 개의 industry를 가질 수 있으며, 원문에 명시된 만큼 모두 추출
3) 공통_skill_set_description은 해당 직무의 '공통' 섹션 항목을 원문 그대로 string으로 저장
   - '공통', '공통 역량', '공통 요구사항', '공통 스킬' 등의 키워드로 시작하는 섹션을 찾으세요
   - 직무 정의 직후에 나오는 공통적인 내용도 포함하세요
   - 모든 industry에 공통으로 적용되는 내용을 찾으세요
   - 없으면 빈 문자열로 저장
4) skill_set_description은 해당 industry 섹션의 기술을 원문 그대로 string으로 저장 (없으면 빈 문자열)
5) 공통_skill_set은 '공통' 섹션의 항목을 리스트로 수집 후 IT용어 중복/변형어 정규화
   - 공통_skill_set_description에서 기술, 도구, 역량 등을 추출하여 리스트로 만드세요
   - 없으면 빈 배열로 저장
6) skill_set은 해당 industry 섹션의 기술을 리스트로 수집 후 IT용어 중복/변형어 정규화 (없으면 빈 배열)
7) 반드시 원문에 있는 모든 정보를 포함하여 출력하세요. 정보가 없으면 빈 값으로라도 필드를 채워주세요.
8) 특별히 주의: Domain Expert 같은 직무는 공통 섹션이 명시적으로 없을 수 있지만, 직무 정의나 전체 구조에서 공통적으로 요구되는 역량을 찾아서 공통_skill_set에 포함시키세요.

[직무별 Industry 구성 참고]
- Software Development: Front-end Development, Back-end Development, Mobile Development
- Factory AX Engineering: Simulation, 기구설계, 전장/제어
- Solution Development: ERP_FCM, ERP_SCM, ERP_HCM, ERP_T&E, Biz. Solution
- Cloud/Infra Engineering: System/Network Engineering, Middleware/Database Engineering, Data Center Engineering
- Architect: Software Architect, Data Architect, Infra Architect, AI Architect, Automation Architect
- Project Management: Application PM, Infra PM, Solution PM, AI PM, Automation PM
- Quality Management: PMO, Quality Engineering, Offshoring Service Professional
- AI: AI/Data Development, Generative AI Development, Physical AI Development
- 정보보호: 보안 Governance/Compliance, 보안 진단/Consulting, 보안 Solution Service
- Sales: 제1금융, 제2금융, 제조 대외, 제조 대내Hi-Tech, 제조 대내Process, 통신, 유통/물류/서비스, 미디어/콘텐츠, 공공, Global
- Domain Expert: (Sales와 동일한 구조)
- Consulting: ESG, SHE, ERP, SCM, CRM, AI
- Biz. Supporting: Strategy Planning, New Biz. Development, Financial Management, Human Resource Management, Stakeholder Management, Governance & Public Management


[결과 예시]
{{
  "직무": "Software Development",
  "직무 정의": "다양한 프로그래밍 언어와 Industry관련 지식과 경험을 활용하여, 고객 Needs에 맞는 소프트웨어/시스템/기능 구현",
  "industry": "Front-end Development",
  "공통_skill_set_description": "• [프로그래밍언어] Java, node.js, Python, C / C++, Go, ASP.NET, Perl, Ruby, C#, PHP, Visual Basic 등\\n• [버전관리도구] Git, Github, SVN, Bitbucket 등\\n• [협업Tool] Jira, Confluence, Slack, Teams, Notion, Google Docs 등\\n• [AI 활용] AI Literacy / Collaboration 역량\\n• [IndustryKnowledge] 1/2금융, 대외제조, 대내Hi-Tech, 대내Process, 통신, 유통/물류/서비스, 미디어/콘텐츠, 공공, Global 등",
  "skill_set_description": "• [UI/UX_디자인도구] Sketch, Adobe XD, Figma 등\\n• [웹프레임워크/라이브러리] React, Angular, Vue.js, Node.js, Next.js, Nust.js, jQuery 등\\n• [웹퍼블리싱] HTML, CSS, CSS 프레임워크(Bootstrap, MaterialUI 등), CSS 전처리기(sass, scss, less 등)",
  "공통_skill_set": [
    "Java",
    "Node.js",
    "Python",
    "C/C++",
    "Go",
    "ASP.NET",
    "Git",
    "GitHub",
    "SVN",
    "Jira",
    "Confluence",
    "Slack",
    "Microsoft Teams",
    "Notion",
    "Google Docs",
    "AI Literacy"
  ],
  "skill_set": [
    "Sketch",
    "Adobe XD",
    "Figma",
    "React",
    "Angular",
    "Vue.js",
    "Next.js",
    "Nuxt.js",
    "jQuery",
    "HTML",
    "CSS",
    "Bootstrap",
    "Material-UI",
    "Sass",
    "SCSS",
    "Less"
  ]
}}

위 예시처럼 각 industry별로 JSON 객체를 생성하되, 반드시 JSON 배열 형태로 출력하세요.
"""

# LCEL 방식: prompt | llm | output_parser
prompt = ChatPromptTemplate.from_template(prompt_template)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


def extract_job_sections(doc):
    """
    PDF에서 직무별 섹션을 추출하는 함수
    직무 제목을 기준으로 페이지를 분할합니다.
    """
    # 직무 제목 패턴 (예: "Software Development", "Data Science" 등)
    # 일반적으로 큰 폰트로 작성되거나 특정 패턴을 가짐
    job_patterns = [
        r'^[A-Z][a-zA-Z\s&/\-]+$',  # 영문 대문자로 시작하는 직무명
        r'^\d+\.\s*[A-Z][a-zA-Z\s&/\-]+',  # 번호가 붙은 직무명
        r'^■\s*[A-Z가-힣][a-zA-Z가-힣\s&/\-]+',  # ■ 기호로 시작
    ]
    
    job_sections = []
    current_job = None
    current_pages = []
    current_text = []
    
    total_pages = len(doc)
    
    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text()
        lines = page_text.split('\n')
        
        # 페이지의 첫 몇 줄에서 직무 제목 찾기
        found_new_job = False
        for i, line in enumerate(lines[:10]):  # 상위 10줄만 체크
            line_stripped = line.strip()
            
            # 패턴 매칭으로 직무 제목 감지
            for pattern in job_patterns:
                if re.match(pattern, line_stripped) and len(line_stripped) > 5:
                    # 새로운 직무 발견
                    if current_job is not None:
                        # 이전 직무 저장
                        job_sections.append({
                            'job_title': current_job,
                            'pages': current_pages.copy(),
                            'text': '\n\n'.join(current_text)
                        })
                    
                    # 새 직무 시작
                    current_job = line_stripped
                    current_pages = [page_num]
                    current_text = [page_text]
                    found_new_job = True
                    print(f"📌 새 직무 발견: '{current_job}' (페이지 {page_num + 1})")
                    break
            
            if found_new_job:
                break
        
        # 기존 직무에 페이지 추가
        if not found_new_job and current_job is not None:
            current_pages.append(page_num)
            current_text.append(page_text)
    
    # 마지막 직무 저장
    if current_job is not None:
        job_sections.append({
            'job_title': current_job,
            'pages': current_pages,
            'text': '\n\n'.join(current_text)
        })
    
    return job_sections


def parse_llm_response(response):
    """LLM 응답을 파싱하는 함수"""
    response_cleaned = response.strip()
    
    # 마크다운 코드 블록 제거
    if response_cleaned.startswith("```json"):
        response_cleaned = response_cleaned[7:]
    if response_cleaned.startswith("```"):
        response_cleaned = response_cleaned[3:]
    if response_cleaned.endswith("```"):
        response_cleaned = response_cleaned[:-3]
    response_cleaned = response_cleaned.strip()
    
    parsed_items = []
    
    # 방법 1: JSON 배열로 파싱
    try:
        items = json.loads(response_cleaned)
        if isinstance(items, list):
            parsed_items = items
            print(f"  ✅ JSON 배열 파싱 성공: {len(parsed_items)}개 항목")
        elif isinstance(items, dict):
            parsed_items = [items]
            print(f"  ✅ JSON 객체 파싱 성공 (단일 항목)")
        return parsed_items
    except json.JSONDecodeError:
        pass
    
    # 방법 2: JSONL 형식
    print(f"  ⚠️ JSON 배열 파싱 실패, JSONL 형식으로 시도...")
    lines = [line.strip() for line in response_cleaned.splitlines() if line.strip()]
    
    for idx, line in enumerate(lines, 1):
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                parsed_items.append(item)
                print(f"    ✅ 줄 {idx}: 파싱 성공")
            elif isinstance(item, list):
                parsed_items.extend(item)
                print(f"    ✅ 줄 {idx}: 배열 파싱 성공 - {len(item)}개 항목")
        except json.JSONDecodeError:
            continue
    
    # 방법 3: 부분 문자열 추출
    if not parsed_items:
        print(f"  ⚠️ 전체 텍스트에서 JSON 추출 시도...")
        try:
            start_idx = response_cleaned.find('[')
            end_idx = response_cleaned.rfind(']')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_cleaned[start_idx:end_idx+1]
                items = json.loads(json_str)
                if isinstance(items, list):
                    parsed_items = items
                    print(f"    ✅ JSON 배열 추출 성공: {len(parsed_items)}개 항목")
        except Exception:
            pass
    
    return parsed_items


STRING_COMMON_FIELDS = ["직무 정의", "공통_skill_set_description"]
LIST_COMMON_FIELDS = ["공통_skill_set"]


def normalize_skill_list(value):
    """스킬 리스트를 정규화하여 리스트 형태로 반환"""
    items = set()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                normalized = item.strip()
                if normalized:
                    items.add(normalized)
    elif isinstance(value, str):
        candidates = re.split(r'[,\\n]+', value)
        for candidate in candidates:
            normalized = candidate.strip()
            if normalized:
                items.add(normalized)
    return sorted(items)


def select_preferred_string(values):
    """여러 문자열 중 가장 정보가 많은 값을 선택"""
    cleaned = [
        v.strip()
        for v in values
        if isinstance(v, str) and v.strip()
    ]
    if not cleaned:
        return ""
    cleaned.sort(key=len, reverse=True)
    return cleaned[0]


def ensure_job_common_fields(all_results, job_title):
    """같은 직무에 대해 공통 필드가 동일하게 유지되도록 조정"""
    job_items = [
        item for item in all_results
        if item.get('직무') == job_title
    ]
    if not job_items:
        return

    # 공통 문자열 필드 선택
    canonical_strings = {}
    for field in STRING_COMMON_FIELDS:
        field_values = [item.get(field, "") for item in job_items]
        canonical_strings[field] = select_preferred_string(field_values)

    # 공통 리스트 필드 병합
    canonical_lists = {}
    for field in LIST_COMMON_FIELDS:
        merged_items = set()
        for item in job_items:
            merged_items.update(normalize_skill_list(item.get(field, [])))
        canonical_lists[field] = sorted(merged_items)

    # 모든 항목에 canonical 값 적용
    for item in job_items:
        for field, value in canonical_strings.items():
            item[field] = value
        for field, value in canonical_lists.items():
            item[field] = value[:]


def merge_duplicate_items(all_results, new_item):
    """중복 항목을 병합하는 함수"""
    직무 = new_item.get('직무', '')
    industry = new_item.get('industry', '')
    
    if not 직무 or not industry:
        print(f"  ⚠️ 필수 필드 누락: 직무={직무}, industry={industry}")
        return False
    
    # 중복 체크
    for existing in all_results:
        if (existing.get('직무') == 직무 and 
            existing.get('industry') == industry):
            # 중복 발견 - 병합
            if new_item.get('공통_skill_set_description') and not existing.get('공통_skill_set_description'):
                existing['공통_skill_set_description'] = new_item.get('공통_skill_set_description', '')
            if new_item.get('skill_set_description') and not existing.get('skill_set_description'):
                existing['skill_set_description'] = new_item.get('skill_set_description', '')
            
            # 리스트 병합
            existing_common = set(existing.get('공통_skill_set', []))
            new_common = set(new_item.get('공통_skill_set', []))
            if new_common:
                existing['공통_skill_set'] = list(existing_common | new_common)
            
            existing_skill = set(existing.get('skill_set', []))
            new_skill = set(new_item.get('skill_set', []))
            if new_skill:
                existing['skill_set'] = list(existing_skill | new_skill)
            
            print(f"  🔄 중복 항목 업데이트: 직무={직무}, industry={industry}")
            ensure_job_common_fields(all_results, 직무)
            return True
    
    # 중복 아님 - 새로 추가
    all_results.append(new_item)
    print(f"  ✅ 항목 추가: 직무={직무}, industry={industry}")
    ensure_job_common_fields(all_results, 직무)
    return True


# 메인 실행 부분
output_dir = os.path.dirname(pdf_path)
csv_file_path = os.path.join(output_dir, "output.csv")
json_file_path = os.path.join(output_dir, "output.json")

all_results = []

print(f"[확인] 출력 파일 경로:")
print(f"  CSV: {csv_file_path}")
print(f"  JSON: {json_file_path}\n")

with fitz.open(pdf_path) as doc:
    total_pages = len(doc)
    print(f"총 페이지 수: {total_pages}\n")
    print(f"{'='*80}")
    print("1단계: 직무별 섹션 추출")
    print(f"{'='*80}\n")
    
    # 직무별 섹션 추출
    job_sections = extract_job_sections(doc)
    
    if not job_sections:
        print("⚠️ 직무 섹션을 자동으로 찾지 못했습니다. 전체 문서를 하나의 섹션으로 처리합니다.")
        full_text = ""
        for page_num in range(total_pages):
            page = doc[page_num]
            full_text += page.get_text() + "\n\n"
        
        job_sections = [{
            'job_title': 'Unknown Job',
            'pages': list(range(total_pages)),
            'text': full_text
        }]
    
    print(f"\n총 {len(job_sections)}개의 직무 섹션 발견\n")
    print(f"{'='*80}")
    print("2단계: 각 직무별 LLM 처리")
    print(f"{'='*80}\n")
    
    # 각 직무별로 LLM 처리
    for idx, job_section in enumerate(job_sections, 1):
        job_title = job_section['job_title']
        pages = job_section['pages']
        text = job_section['text']
        
        if not text.strip():
            print(f"[{idx}/{len(job_sections)}] 건너뜀: '{job_title}' - 빈 내용")
            continue
        
        print(f"\n[{idx}/{len(job_sections)}] 처리 중: '{job_title}'")
        print(f"  페이지: {pages[0]+1}-{pages[-1]+1} (총 {len(pages)}페이지)")
        print(f"  텍스트 길이: {len(text)} 문자")
        
        try:
            # LLM 호출
            response = chain.invoke({"context": text})
            
            # 응답 파싱
            parsed_items = parse_llm_response(response)
            
            # 결과 병합
            if parsed_items:
                for item in parsed_items:
                    if isinstance(item, dict):
                        merge_duplicate_items(all_results, item)
            else:
                print(f"  ⚠️ 파싱된 항목이 없습니다.")
            
        except Exception as e:
            print(f"  ❌ 에러 발생: {e}")
        
        print("-" * 80)

# 결과 저장
print(f"\n{'='*80}")
print(f"총 {len(all_results)}개 레코드 수집됨")
print(f"{'='*80}\n")

if all_results:
    # JSON 저장
    with open(json_file_path, "w", encoding="utf-8") as jf:
        json.dump(all_results, jf, ensure_ascii=False, indent=2)
    print(f"✅ JSON 저장 완료: {json_file_path}")
    print(f"   레코드 수: {len(all_results)}")
    
    # CSV 저장
    keys = set()
    for item in all_results:
        keys.update(item.keys())
    keys = list(keys)
    
    with open(csv_file_path, "w", newline='', encoding="utf-8-sig") as cf:
        writer = csv.DictWriter(cf, fieldnames=keys)
        writer.writeheader()
        for item in all_results:
            item_row = item.copy()
            for k, v in item_row.items():
                if isinstance(v, list):
                    item_row[k] = ", ".join(v)
            writer.writerow(item_row)
    
    print(f"✅ CSV 저장 완료: {csv_file_path}")
    print(f"   행 수: {len(all_results)}, 열 수: {len(keys)}")
else:
    print("❌ 저장할 레코드가 없습니다!")