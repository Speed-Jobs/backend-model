# tutorial7: tutorial6 + LLM 기반 의미론적 사전 필터링 추가
# 직무기술서 PDF의 직무/산업 설명을 활용하여 명백히 맞지 않는 매칭(예: LLM Engineer → Consulting) 방지

import pandas as pd
import json
import os
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 환경 변수 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
env_path = os.path.join(project_root, ".env")

if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY", None)
if openai_api_key is None:
    raise ValueError(f".env 파일에 'OPENAI_API_KEY'가 없습니다. (.env 경로: {env_path})")
os.environ["OPENAI_API_KEY"] = openai_api_key

# LangChain 라이브러리
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any

# ============================================================================
# 1. 데이터 로드
# ============================================================================
job_description_dir = os.path.join(script_dir, "data")
json_path = os.path.join(job_description_dir, "job_description.json")
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)
print(f"직무기술서 데이터: {len(df)}개")

# 모든 *_jobs.json 파일 자동 로드
job_postings = []
root_data_dir = os.path.join(script_dir, "data")

job_paths = []
for fname in os.listdir(root_data_dir):
    if fname.endswith("_jobs.json"):
        job_paths.append(os.path.join(root_data_dir, fname))

loaded_files = []
for path in job_paths:
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                jobs = json.load(f)
                if isinstance(jobs, list):
                    job_postings.extend(jobs)
                    job_count = len(jobs)
                elif isinstance(jobs, dict):
                    job_postings.append(jobs)
                    job_count = 1
                else:
                    print(f"[WARNING] 지원하지 않는 데이터 형식: {os.path.basename(path)}")
                    continue
                loaded_files.append(os.path.basename(path))
                print(f"[OK] 파일 로드 성공: {os.path.basename(path)} ({job_count}개)")
        else:
            print(f"[WARNING] 파일을 찾을 수 없습니다: {path}")
    except Exception as e:
        print(f"[ERROR] 파일 로드 오류: {os.path.basename(path)} - {e}")

print(f"총 채용공고 수: {len(job_postings)}개 (로드된 파일: {', '.join(loaded_files) if loaded_files else '없음'})")

if len(job_postings) == 0:
    raise ValueError("채용공고 파일을 찾을 수 없습니다.")

# ============================================================================
# 2. LLM 기반 사전 필터링 함수 (PDF 청크 기반)
# ============================================================================
filter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
filter_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 채용공고와 직무 적합성을 엄격하게 판단하는 전문가입니다.

**필수 판단 기준 (하나라도 위반 시 무조건 "부적합")**:

1. 직무 타입 불일치
   - 개발(Engineer, Developer, Development) ↔ Consulting = 부적합
   - 개발 ↔ 기획/전략(Planning, Strategy) = 부적합
   - Consulting ↔ Software Development = 부적합

2. 명확한 키워드 체크
   채용공고에 다음 키워드가 있으면:
   - "Engineer", "개발", "Developer", "Development", "Programming" → Consulting 직무는 절대 부적합
   - "Consultant", "컨설팅", "전략", "기획" → Development 직무는 절대 부적합

**응답 형식**:
첫 줄에 "적합" 또는 "부적합"만 작성하세요.

예시:
부적합
적합"""),
    ("user", """채용공고:
제목: {job_title}
회사: {company}
스킬: {skills}

직무기술서:
직무: {job_role}
산업: {industry}
설명: {description}

판단하세요.""")
])

filter_chain = filter_prompt | filter_llm

def is_job_suitable(job_posting, job_role, industry, description):
    """LLM을 활용한 채용공고-직무 적합성 판단"""
    try:
        # 채용공고 정보 추출
        title = job_posting.get('title', '')
        company = job_posting.get('company', '')

        # 스킬 추출
        if 'skill_set_info' in job_posting and isinstance(job_posting['skill_set_info'], dict):
            skill_set = job_posting['skill_set_info'].get('skill_set', [])
        else:
            skill_set = job_posting.get('skill_set', [])
        skills_text = ", ".join(skill_set[:5]) if isinstance(skill_set, list) else str(skill_set)

        # LLM 호출
        result = filter_chain.invoke({
            "job_title": title,
            "company": company,
            "skills": skills_text,
            "job_role": job_role,
            "industry": industry,
            "description": description if description else "설명 없음"
        })

        response = result.content.strip()
        # "적합" 또는 "부적합" 판단
        is_suitable = "적합" in response.split('\n')[0]

        return is_suitable

    except Exception as e:
        print(f"필터링 오류: {e}")
        return True  # 오류 시 기본적으로 허용

# ============================================================================
# 3. 스킬 추출 유틸리티 함수
# ============================================================================
def extract_skills_from_text(skill_text):
    """문자열에서 스킬 키워드 추출"""
    if not isinstance(skill_text, str):
        return []

    skills = []

    # 괄호로 묶인 스킬 추출
    in_parentheses = re.findall(r'\(([^)]+)\)', skill_text)
    for item in in_parentheses:
        skills.extend([s.strip() for s in item.split(',')])

    # 주요 기술 키워드 패턴 매칭
    tech_patterns = [
        r'\b[A-Z][a-zA-Z0-9\.\-_]+\b',
        r'\b[a-z]+\.[a-z]+\b',
    ]
    for pattern in tech_patterns:
        matches = re.findall(pattern, skill_text)
        skills.extend(matches)

    skills = list(set([s for s in skills if len(s) > 1]))
    return skills

def parse_job_description_skills(row):
    """직무기술서 행에서 전체 스킬셋 추출"""
    all_skills = []
    if isinstance(row.get('공통_skill_set'), list):
        all_skills.extend(row['공통_skill_set'])

    skill_set = row.get('skill_set', '')
    if isinstance(skill_set, str):
        extracted = extract_skills_from_text(skill_set)
        all_skills.extend(extracted)
    elif isinstance(skill_set, list):
        all_skills.extend(skill_set)
    return list(set(all_skills))

df['parsed_skills'] = df.apply(parse_job_description_skills, axis=1)

# ============================================================================
# 4. Document 변환 - 직무기술서 JSON + PDF 청크
# ============================================================================
def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """ChromaDB가 허용하는 타입만 남기고 복잡한 타입은 문자열로 변환"""
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = None
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list):
            cleaned[key] = ", ".join(str(v) for v in value) if value else ""
        elif isinstance(value, dict):
            cleaned[key] = str(value)
        else:
            cleaned[key] = str(value)
    return cleaned

# 직무기술서를 Document로 변환
documents = []
for idx, row in df.iterrows():
    skill_set_text = ", ".join(row['parsed_skills']) if isinstance(row['parsed_skills'], list) else str(row['parsed_skills'])
    common_skills_text = ", ".join(row.get('공통_skill_set', [])) if isinstance(row.get('공통_skill_set'), list) else str(row.get('공통_skill_set', ''))

    content = f"""
    직무: {row.get('직무', '')}
    산업: {row.get('industry', '')}
    핵심기술: {skill_set_text}
    공통스킬: {common_skills_text}
    상세설명: {row.get('skill_set', '')[:500] if row.get('skill_set') else ''}
    """

    metadata = {
        "job_title": row.get('직무', ''),
        "industry": row.get('industry', ''),
        "skill_set": skill_set_text,
        "common_skill_set": common_skills_text,
        "job_description_id": str(idx),
        "source": "json"
    }

    cleaned_metadata = clean_metadata(metadata)
    doc = Document(page_content=content.strip(), metadata=cleaned_metadata)
    documents.append(doc)

print(f"직무기술서 JSON Document 변환 완료: {len(documents)}개")

# PDF 문서 RecursiveCharacterTextSplitter로 청크 단위로 분할
pdf_path = os.path.join(job_description_dir, "job_description.pdf")
pdf_documents = []

if os.path.exists(pdf_path):
    print(f"\nPDF 파일 로드 중: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        pdf_pages = loader.load()
        print(f"PDF 페이지 수: {len(pdf_pages)}개")

        # RecursiveCharacterTextSplitter 설정
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # PDF 전체를 청크로 분할 (페이지별로 직무/산업 정보 추출 후 청크에 전파)
        for page_idx, page in enumerate(pdf_pages):
            page_text = page.page_content

            # 페이지 전체에서 직무/산업 정보 추출
            job_title = "Unknown"
            industry = "Unknown"

            job_match = re.search(r'(?:직무|Job)[:\s]*([^\n]+)', page_text, re.IGNORECASE)
            industry_match = re.search(r'(?:산업|Industry)[:\s]*([^\n]+)', page_text, re.IGNORECASE)

            if job_match:
                job_title = job_match.group(1).strip()
            if industry_match:
                industry = industry_match.group(1).strip()

            # 페이지를 청크로 분할
            page_chunks = text_splitter.split_documents([page])

            # 각 청크에 페이지 정보 전파
            for chunk_idx, chunk in enumerate(page_chunks):
                chunk.metadata.update({
                    "job_title": job_title,
                    "industry": industry,
                    "skill_set": "",
                    "source": "pdf_chunk",
                    "page_number": page_idx + 1,
                    "chunk_index": len(pdf_documents) + chunk_idx
                })

                # 메타데이터 정리
                chunk.metadata = clean_metadata(chunk.metadata)
                pdf_documents.append(chunk)

        print(f"PDF 청크 Document 변환 완료: {len(pdf_documents)}개")
    except Exception as e:
        print(f"[WARNING] PDF 로드 실패: {e}")
else:
    print(f"[WARNING] PDF 파일을 찾을 수 없습니다: {pdf_path}")

documents.extend(pdf_documents)
print(f"전체 Document 수 (JSON + PDF 청크): {len(documents)}개")

if len(documents) == 0:
    raise ValueError("변환된 Document가 없습니다.")

# ============================================================================
# 5. ChromaDB 및 BM25 Retriever 생성
# ============================================================================
print("ChromaDB 벡터 스토어 생성 중...")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="job_descriptions"
)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
print(f"ChromaDB 생성 완료")

print("BM25 Retriever 생성 중...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10
print("BM25 Retriever 생성 완료")

# ============================================================================
# 6. Hybrid Retriever
# ============================================================================
class CustomEnsembleRetriever(BaseRetriever):
    retrievers: List[BaseRetriever]
    weights: List[float]
    k: int = 60

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        all_results = {}
        retriever_names = ['ChromaDB', 'BM25']
        bm25_raw_scores = {}
        chroma_raw_scores = {}

        for idx, (retriever, weight) in enumerate(zip(self.retrievers, self.weights)):
            retriever_name = retriever_names[idx] if idx < len(retriever_names) else f"Retriever_{idx}"

            if retriever_name == 'ChromaDB':
                vectorstore = retriever.vectorstore
                docs_with_scores = vectorstore.similarity_search_with_score(query, k=10)
                results = [doc for doc, score in docs_with_scores]
                for doc, score in docs_with_scores:
                    doc_id = doc.page_content[:100]
                    chroma_raw_scores[doc_id] = max(0, 1.0 - score)
            elif retriever_name == 'BM25':
                results = retriever.invoke(query)
                try:
                    if hasattr(retriever, 'vectorizer'):
                        vectorizer = retriever.vectorizer
                        if hasattr(vectorizer, 'tokenizer'):
                            query_tokens = vectorizer.tokenizer(query)
                        else:
                            query_tokens = query.split()
                        scores = vectorizer.get_scores(query_tokens)
                        for doc in results:
                            doc_id = doc.page_content[:100]
                            for i, orig_doc in enumerate(retriever.docs):
                                if orig_doc.page_content[:100] == doc_id:
                                    bm25_raw_scores[doc_id] = scores[i]
                                    break
                    else:
                        raise AttributeError("No vectorizer")
                except:
                    for rank, doc in enumerate(results, start=1):
                        doc_id = doc.page_content[:100]
                        bm25_raw_scores[doc_id] = 10.0 / rank
            else:
                results = retriever.invoke(query)

            for rank, doc in enumerate(results, start=1):
                doc_id = doc.page_content[:100]
                rrf_score = weight / (rank + self.k)
                detail = {
                    'retriever': retriever_name,
                    'rank': rank,
                    'weight': weight,
                    'rrf_score': rrf_score,
                }

                if doc_id in all_results:
                    all_results[doc_id]['rrf_score'] += rrf_score
                    all_results[doc_id]['details'].append(detail)
                else:
                    all_results[doc_id] = {
                        'doc': doc,
                        'rrf_score': rrf_score,
                        'details': [detail]
                    }

        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )

        for item in sorted_results:
            doc = item['doc']
            doc_id = doc.page_content[:100]
            doc.metadata['rrf_score'] = item['rrf_score']
            doc.metadata['score_details'] = item['details']
            doc.metadata['bm25_raw_score'] = bm25_raw_scores.get(doc_id, 0)
            doc.metadata['chroma_raw_score'] = chroma_raw_scores.get(doc_id, 0)

        return [item['doc'] for item in sorted_results]

# ============================================================================
# 7. 스킬 매칭 점수 계산
# ============================================================================
def parse_skill_string_to_list(skill_data):
    """스킬 데이터를 리스트로 변환"""
    if isinstance(skill_data, list):
        return skill_data
    elif isinstance(skill_data, str):
        return [s.strip() for s in skill_data.split(',') if s.strip()]
    else:
        return []

def calculate_skill_match_score(job_desc_skills, job_posting_skills):
    """스킬 집합 간 유사도 계산"""
    if not job_desc_skills or not job_posting_skills:
        return 0.0

    job_desc_skills_list = parse_skill_string_to_list(job_desc_skills)
    job_posting_skills_list = parse_skill_string_to_list(job_posting_skills)

    job_desc_skills_set = set([s.lower().strip() for s in job_desc_skills_list if s])
    job_posting_skills_set = set([s.lower().strip() for s in job_posting_skills_list if s])

    if not job_desc_skills_set or not job_posting_skills_set:
        return 0.0

    intersection = len(job_desc_skills_set & job_posting_skills_set)
    union = len(job_desc_skills_set | job_posting_skills_set)
    jaccard = intersection / union if union > 0 else 0

    coverage = intersection / len(job_posting_skills_set) if job_posting_skills_set else 0

    return 0.7 * jaccard + 0.3 * coverage

# ============================================================================
# 8. Hybrid Scorer
# ============================================================================
CHROMA_WEIGHT = 0.25
BM25_WEIGHT = 0.35
SKILL_WEIGHT = 0.40

class EnhancedHybridScorer:
    """BM25 Softmax 정규화 + ChromaDB 코사인 + 스킬 매칭 가중 합산"""
    def __init__(self, bm25_weight=BM25_WEIGHT, chroma_weight=CHROMA_WEIGHT, skill_weight=SKILL_WEIGHT):
        self.bm25_weight = bm25_weight
        self.chroma_weight = chroma_weight
        self.skill_weight = skill_weight

    def calculate_scores(self, job_desc_results, job_posting_skills):
        if not job_desc_results:
            return []

        bm25_scores = []
        chroma_scores = []
        skill_scores = []

        for job_desc in job_desc_results:
            bm25_scores.append(job_desc.metadata.get('bm25_raw_score', 0))
            chroma_scores.append(job_desc.metadata.get('chroma_raw_score', 0))

            job_desc_skill_set = job_desc.metadata.get('skill_set', [])
            skill_match = calculate_skill_match_score(job_desc_skill_set, job_posting_skills)
            skill_scores.append(skill_match)

        bm25_array = np.array(bm25_scores)
        if np.all(bm25_array == 0):
            bm25_normalized = np.zeros_like(bm25_array)
        else:
            exp_bm25 = np.exp(bm25_array - np.max(bm25_array))
            bm25_normalized = exp_bm25 / exp_bm25.sum()

        results = []
        for i, (bm25_norm, chroma, skill) in enumerate(zip(bm25_normalized, chroma_scores, skill_scores)):
            weighted_sum = (
                self.bm25_weight * bm25_norm +
                self.chroma_weight * chroma +
                self.skill_weight * skill
            )
            results.append({
                'bm25_normalized': float(bm25_norm),
                'chroma_cosine': float(chroma),
                'skill_match': float(skill),
                'weighted_sum': float(weighted_sum)
            })

        return results

ensemble_retriever = CustomEnsembleRetriever(
    retrievers=[chroma_retriever, bm25_retriever],
    weights=[CHROMA_WEIGHT, BM25_WEIGHT]
)
print(f"Ensemble Retriever 생성 완료")

hybrid_scorer = EnhancedHybridScorer()

# ============================================================================
# 9. Query Rewrite
# ============================================================================
query_rewrite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 채용공고를 분석하여 적합한 직무와 산업을 찾는 전문가입니다.
주어진 채용공고 정보를 바탕으로 어떤 직무(job_title)와 산업(industry)에 적합한지 검색하기 위한 자연어 쿼리를 생성하세요.

요구사항:
- 핵심 키워드를 자연스럽게 포함
- 검색 의도가 명확한 문장 형태
- 50-100자 이내로 간결하게
- 직무, 산업, 기술 스택이 모두 포함되도록

예시:
입력: 제목=백엔드 개발자, 회사=네이버, 스킬=Python, FastAPI, Docker
출력: Python, FastAPI, Docker를 활용하는 IT 산업의 백엔드 개발 직무"""),
    ("user", """제목: {title}
회사: {company}
직무분야: {job_category}
핵심 스킬: {core_skills}

위 채용공고에 적합한 직무와 산업을 찾기 위한 자연어 쿼리를 생성해주세요.""")
])

query_rewrite_chain = query_rewrite_prompt | query_rewrite_llm

def extract_job_posting_skills(job):
    """채용공고에서 스킬 추출"""
    if 'skill_set_info' in job and isinstance(job['skill_set_info'], dict):
        skill_set = job['skill_set_info'].get('skill_set', [])
    else:
        skill_set = job.get('skill_set', [])

    return skill_set if isinstance(skill_set, list) else []

def rewrite_query_for_job_posting(job):
    """채용공고를 쿼리로 변환"""
    queries = []

    title = job.get('title', '')
    company = job.get('company', '')
    skill_set = extract_job_posting_skills(job)
    skill_text = ", ".join(skill_set[:5]) if skill_set else ""

    meta_data = job.get('meta_data', {})
    job_category = meta_data.get('job_category', '') if isinstance(meta_data, dict) else ''

    queries.append(f"{title} {skill_text}")

    if job_category and skill_text:
        queries.append(f"{job_category} {skill_text}")

    try:
        result = query_rewrite_chain.invoke({
            "title": title,
            "company": company,
            "job_category": job_category,
            "core_skills": skill_text
        })
        queries.append(result.content.strip())
    except Exception as e:
        print(f"Query rewrite 실패: {e}")

    return queries

def get_job_type(job_title):
    """채용공고 제목에서 직무 타입 추출 (development/consulting/planning)"""
    title_lower = job_title.lower()

    # 개발 키워드
    dev_keywords = ['engineer', 'developer', 'development', '개발', 'programming', 'software', 'backend', 'frontend', 'fullstack']
    # 컨설팅 키워드
    consulting_keywords = ['consultant', 'consulting', '컨설팅', 'advisor']
    # 기획/전략 키워드
    planning_keywords = ['planner', 'planning', '기획', 'strategy', '전략', 'pm', 'product manager']

    if any(kw in title_lower for kw in dev_keywords):
        return 'development'
    elif any(kw in title_lower for kw in consulting_keywords):
        return 'consulting'
    elif any(kw in title_lower for kw in planning_keywords):
        return 'planning'
    else:
        return 'unknown'

def search_matching_job_descriptions_with_filter(job, top_k=5):
    """채용공고에 매칭되는 직무기술서 검색 + 직무 타입 기반 사전 필터링"""
    queries = rewrite_query_for_job_posting(job)

    # 채용공고의 직무 타입 추출
    job_title = job.get('title', '')
    job_type = get_job_type(job_title)

    all_docs = {}

    for query in queries:
        results = ensemble_retriever.invoke(query)

        for doc in results:
            # JSON 문서만 처리
            if doc.metadata.get('source') != 'json':
                continue

            # 직무 타입 기반 사전 필터링
            job_role = doc.metadata.get('job_title', '').lower()

            # 개발 직무는 Consulting/Planning 제외
            if job_type == 'development':
                if 'consulting' in job_role or 'planning' in job_role or 'strategy' in job_role:
                    continue
            # 컨설팅 직무는 Development 제외
            elif job_type == 'consulting':
                if 'development' in job_role or 'software' in job_role or 'engineer' in job_role:
                    continue
            # 기획 직무는 Development 제외
            elif job_type == 'planning':
                if 'development' in job_role or 'software' in job_role or 'engineer' in job_role:
                    continue

            doc_id = doc.page_content[:100]

            if doc_id in all_docs:
                all_docs[doc_id]['count'] += 1
                all_docs[doc_id]['total_rrf'] += doc.metadata.get('rrf_score', 0)
                all_docs[doc_id]['total_bm25'] += doc.metadata.get('bm25_raw_score', 0)
                all_docs[doc_id]['total_chroma'] += doc.metadata.get('chroma_raw_score', 0)
            else:
                all_docs[doc_id] = {
                    'doc': doc,
                    'count': 1,
                    'total_rrf': doc.metadata.get('rrf_score', 0),
                    'total_bm25': doc.metadata.get('bm25_raw_score', 0),
                    'total_chroma': doc.metadata.get('chroma_raw_score', 0)
                }

    # 평균 점수 계산
    for doc_data in all_docs.values():
        count = doc_data['count']
        boost = 1 + 0.15 * (count - 1)

        doc_data['doc'].metadata['bm25_raw_score'] = (doc_data['total_bm25'] / count) * boost
        doc_data['doc'].metadata['chroma_raw_score'] = (doc_data['total_chroma'] / count) * boost
        doc_data['doc'].metadata['multi_query_count'] = count

    sorted_docs = sorted(
        all_docs.values(),
        key=lambda x: (x['total_rrf'] / x['count']) * (1 + 0.15 * (x['count'] - 1)),
        reverse=True
    )

    return [item['doc'] for item in sorted_docs[:top_k]]

# ============================================================================
# 10. 검색 실행
# ============================================================================
all_results = []

print("\n" + "="*100)
print("전체 채용공고에 대한 직무기술서 매칭 시작 (LLM 필터링 + 멀티쿼리 + 스킬매칭)")
print("="*100)

for idx, job in enumerate(job_postings):
    title = job.get('title', 'N/A')
    company = job.get('company', 'N/A')
    job_skills = extract_job_posting_skills(job)

    # 필터링 포함 검색
    matching_job_descs = search_matching_job_descriptions_with_filter(job, top_k=10)

    # 스킬 매칭 포함 스코어링
    hybrid_scores = hybrid_scorer.calculate_scores(matching_job_descs, job_skills)

    # 가중평균 합산 점수로 재정렬
    job_descs_with_scores = []
    for i, job_desc in enumerate(matching_job_descs):
        hybrid_score = hybrid_scores[i] if i < len(hybrid_scores) else {
            'weighted_sum': 0,
            'skill_match': 0,
            'bm25_normalized': 0,
            'chroma_cosine': 0
        }
        job_descs_with_scores.append((job_desc, hybrid_score))

    job_descs_with_scores.sort(key=lambda x: x[1]['weighted_sum'], reverse=True)

    top_3_job_descs = job_descs_with_scores[:3]

    result_item = {
        'job_posting_title': title,
        'company': company,
        'job_url': job.get('url', 'N/A'),
        'job_skills': job_skills,
        'matched_job_roles': []
    }

    for rank, (job_desc, hybrid_score) in enumerate(top_3_job_descs, 1):
        job_desc_skills_raw = job_desc.metadata.get('skill_set', '')
        job_desc_skills = parse_skill_string_to_list(job_desc_skills_raw) if job_desc_skills_raw else []

        result_item['matched_job_roles'].append({
            'rank': rank,
            'recommended_job_title': job_desc.metadata.get('job_title', 'N/A'),
            'recommended_industry': job_desc.metadata.get('industry', 'N/A'),
            'required_skill_set': job_desc_skills,
            'weighted_sum': hybrid_score['weighted_sum'],
            'skill_match': hybrid_score['skill_match'],
            'bm25_normalized': hybrid_score['bm25_normalized'],
            'chroma_cosine': hybrid_score['chroma_cosine'],
            'multi_query_count': job_desc.metadata.get('multi_query_count', 1)
        })

    all_results.append(result_item)

    if (idx + 1) % 50 == 0:
        print(f"진행 중... {idx + 1}/{len(job_postings)} 완료")

print(f"\n매칭 완료! 총 {len(all_results)}개 채용공고 처리")

# ============================================================================
# 11. TXT 파일 저장
# ============================================================================
def save_results_to_txt(results, filename):
    """검색 결과를 TXT 파일로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("채용공고 → 적합 직무 & 산업 매칭 결과 (LLM 필터링 적용)\n")
        f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 채용공고 수: {len(results)}개\n")
        f.write(f"총 직무기술서 수: {len(df)}개\n")
        f.write(f"스코어링 가중치: BM25={BM25_WEIGHT}, Chroma={CHROMA_WEIGHT}, Skill={SKILL_WEIGHT}\n")
        f.write("="*100 + "\n\n")

        for idx, result in enumerate(results, 1):
            f.write(f"\n{'='*100}\n")
            f.write(f"[{idx}] 채용공고: {result['job_posting_title']}\n")
            f.write(f"회사: {result['company']}\n")
            skill_set_display = ", ".join(result['job_skills']) if isinstance(result['job_skills'], list) else str(result['job_skills'])
            f.write(f"요구 스킬: {skill_set_display}\n")
            f.write(f"{'='*100}\n\n")

            for job_role in result['matched_job_roles']:
                f.write(f"{job_role['rank']}. 직무: {job_role['recommended_job_title']} | 산업: {job_role['recommended_industry']}\n")
                skill_set_str = ", ".join(job_role['required_skill_set']) if isinstance(job_role['required_skill_set'], list) and job_role['required_skill_set'] else "N/A"
                f.write(f"   해당 직무 필수 스킬: {skill_set_str}\n")
                f.write(f"   \n")
                f.write(f"   [매칭 점수 상세]\n")
                f.write(f"   - 최종 가중 합산: {job_role['weighted_sum']:.6f}\n")
                f.write(f"   - 스킬 매칭 점수: {job_role['skill_match']:.6f}\n")
                f.write(f"   - BM25 정규화: {job_role['bm25_normalized']:.6f}\n")
                f.write(f"   - ChromaDB 코사인: {job_role['chroma_cosine']:.6f}\n")
                f.write(f"   - 멀티쿼리 매칭: {job_role['multi_query_count']}회\n")
                f.write(f"   \n\n")

    print(f"TXT 파일 저장 완료: {filename}")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

txt_filename = os.path.join(output_dir, f"reverse_job_matching_results_{timestamp}.txt")

save_results_to_txt(all_results, txt_filename)

print("\n" + "="*100)
print("결과 파일 저장 완료!")
print(f"TXT 파일: {txt_filename}")
print("="*100)

# ============================================================================
# 12. 결과 출력 (콘솔)
# ============================================================================
print("\n\n[검색 결과 샘플 - 처음 3개]")
for idx, result in enumerate(all_results[:3], 1):
    print(f"\n{'='*100}")
    print(f"[{idx}] 채용공고: {result['job_posting_title']}")
    print(f"회사: {result['company']}")
    skill_set_display = ", ".join(result['job_skills']) if isinstance(result['job_skills'], list) else str(result['job_skills'])
    print(f"요구 스킬: {skill_set_display}")
    print(f"{'='*100}")
    for job_role in result['matched_job_roles']:
        print(f"\n{job_role['rank']}. 직무: {job_role['recommended_job_title']} | 산업: {job_role['recommended_industry']}")
        skill_set_str = ", ".join(job_role['required_skill_set']) if isinstance(job_role['required_skill_set'], list) and job_role['required_skill_set'] else "N/A"
        print(f"   해당 직무 필수 스킬: {skill_set_str}")
        print(f"   ")
        print(f"   [매칭 점수 상세]")
        print(f"   - 최종 가중 합산: {job_role['weighted_sum']:.6f}")
        print(f"   - 스킬 매칭 점수: {job_role['skill_match']:.6f}")
        print(f"   - BM25 정규화: {job_role['bm25_normalized']:.6f}")
        print(f"   - ChromaDB 코사인: {job_role['chroma_cosine']:.6f}")
        print(f"   - 멀티쿼리 매칭: {job_role['multi_query_count']}회")
        print(f"   ")

print("\n" + "="*100)
print("채용공고 → 직무 & 산업 매칭 완료! (LLM 필터링 적용)")
print("="*100)
