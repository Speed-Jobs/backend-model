# tutorial6: tutorial5에서 PDF 원본 문서를 추가로 Vector DB에 저장
# JSON(구조화 데이터) + PDF(풍부한 텍스트)를 함께 사용하여 BM25 및 의미 매칭 성능 향상
# 특히 Generative AI Development 같은 LLM 관련 직무의 BM25 점수 개선 목표

import pandas as pd
import json
import os
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import re
from langchain_community.document_loaders import PyPDFLoader

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

# data 디렉토리 안의 *_jobs.json 파일 모두 탐색
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
# 2. 스킬 추출 유틸리티 함수
# ============================================================================
def extract_skills_from_text(skill_text):
    """문자열에서 스킬 키워드 추출"""
    if not isinstance(skill_text, str):
        return []

    skills = []

    # 괄호로 묶인 스킬 추출 (React, Angular, Vue.js)
    in_parentheses = re.findall(r'\(([^)]+)\)', skill_text)
    for item in in_parentheses:
        skills.extend([s.strip() for s in item.split(',')])

    # 주요 기술 키워드 패턴 매칭
    tech_patterns = [
        r'\b[A-Z][a-zA-Z0-9\.\-_]+\b',  # CamelCase, 약어
        r'\b[a-z]+\.[a-z]+\b',  # node.js, vue.js
    ]
    for pattern in tech_patterns:
        matches = re.findall(pattern, skill_text)
        skills.extend(matches)

    # 중복 제거 및 정리
    skills = list(set([s for s in skills if len(s) > 1]))
    return skills

def parse_job_description_skills(row):
    """직무기술서 행에서 전체 스킬셋 추출"""
    all_skills = []
    # 공통 스킬
    if isinstance(row.get('공통_skill_set'), list):
        all_skills.extend(row['공통_skill_set'])

    # skill_set이 문자열인 경우 파싱
    skill_set = row.get('skill_set', '')
    if isinstance(skill_set, str):
        extracted = extract_skills_from_text(skill_set)
        all_skills.extend(extracted)
    elif isinstance(skill_set, list):
        all_skills.extend(skill_set)
    # 중복 제거
    return list(set(all_skills))

# 직무기술서 데이터에 파싱된 스킬 추가
df['parsed_skills'] = df.apply(parse_job_description_skills, axis=1)

# ============================================================================
# 3. Document 변환 - 직무기술서를 Document로 변환 (tutorial4와 반대)
# ============================================================================
def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChromaDB가 허용하는 타입(str, int, float, bool, None)만 남기고
    복잡한 타입(리스트, 딕셔너리)은 문자열로 변환하거나 제거
    """
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

# 직무기술서를 Document로 변환 (JSON 기반)
documents = []
for idx, row in df.iterrows():
    skill_set_text = ", ".join(row['parsed_skills']) if isinstance(row['parsed_skills'], list) else str(row['parsed_skills'])
    common_skills_text = ", ".join(row.get('공통_skill_set', [])) if isinstance(row.get('공통_skill_set'), list) else str(row.get('공통_skill_set', ''))

    # 더 풍부한 검색을 위한 콘텐츠 구성
    content = f"""
    직무: {row.get('직무', '')}
    산업: {row.get('industry', '')}
    핵심기술: {skill_set_text}
    공통스킬: {common_skills_text}
    상세설명: {row.get('skill_set', '')[:500] if row.get('skill_set') else ''}
    """

    # Metadata 구성
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

# ============================================================================
# 3-2. PDF 문서 로드 및 Document 변환 추가
# ============================================================================
pdf_path = os.path.join(job_description_dir, "job_description.pdf")
pdf_documents = []

if os.path.exists(pdf_path):
    print(f"PDF 파일 로드 중: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        pdf_pages = loader.load()
        print(f"PDF 페이지 수: {len(pdf_pages)}개")

        # PDF의 각 페이지를 Document로 변환
        for page_idx, page in enumerate(pdf_pages):
            # PDF 페이지에서 직무/산업 정보 추출 시도
            page_text = page.page_content

            # 간단한 휴리스틱으로 직무와 산업 추정
            job_title = "Unknown"
            industry = "Unknown"

            # "직무:", "산업:" 키워드로 추출 시도
            job_match = re.search(r'직무[:\s]*([^\n]+)', page_text)
            industry_match = re.search(r'산업[:\s]*([^\n]+)', page_text) or re.search(r'industry[:\s]*([^\n]+)', page_text, re.IGNORECASE)

            if job_match:
                job_title = job_match.group(1).strip()
            if industry_match:
                industry = industry_match.group(1).strip()

            # Metadata 구성
            pdf_metadata = {
                "job_title": job_title,
                "industry": industry,
                "skill_set": "",  # PDF에서는 skill_set을 별도로 추출하지 않음 (전체 텍스트에 포함)
                "source": "pdf",
                "page_number": page_idx + 1
            }

            cleaned_pdf_metadata = clean_metadata(pdf_metadata)
            pdf_doc = Document(page_content=page_text, metadata=cleaned_pdf_metadata)
            pdf_documents.append(pdf_doc)

        print(f"PDF Document 변환 완료: {len(pdf_documents)}개")
    except Exception as e:
        print(f"[WARNING] PDF 로드 실패: {e}")
else:
    print(f"[WARNING] PDF 파일을 찾을 수 없습니다: {pdf_path}")

# JSON + PDF Documents 통합
documents.extend(pdf_documents)
print(f"전체 Document 수 (JSON + PDF): {len(documents)}개")

if len(documents) == 0:
    raise ValueError("변환된 Document가 없습니다.")

# ============================================================================
# 4. ChromaDB 및 BM25 Retriever 생성 (직무기술서 기반)
# ============================================================================
print("ChromaDB 벡터 스토어 생성 중 (직무기술서 저장)...")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="job_descriptions"
)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
print(f"ChromaDB 생성 완료 (in-memory 모드)")

print("BM25 Retriever 생성 중 (직무기술서 기반)...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10
print("BM25 Retriever 생성 완료")

# ============================================================================
# 5. Hybrid Retriever
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
# 6. 스킬 매칭 점수 계산
# ============================================================================
def parse_skill_string_to_list(skill_data):
    """
    스킬 데이터를 리스트로 변환
    - 문자열인 경우: 쉼표로 분리
    - 리스트인 경우: 그대로 반환
    """
    if isinstance(skill_data, list):
        return skill_data
    elif isinstance(skill_data, str):
        return [s.strip() for s in skill_data.split(',') if s.strip()]
    else:
        return []

def calculate_skill_match_score(job_desc_skills, job_posting_skills):
    """스킬 집합 간 유사도 계산 (Jaccard + Coverage)"""
    if not job_desc_skills or not job_posting_skills:
        return 0.0

    # 문자열을 리스트로 변환 후 set 생성
    job_desc_skills_list = parse_skill_string_to_list(job_desc_skills)
    job_posting_skills_list = parse_skill_string_to_list(job_posting_skills)

    job_desc_skills_set = set([s.lower().strip() for s in job_desc_skills_list if s])
    job_posting_skills_set = set([s.lower().strip() for s in job_posting_skills_list if s])

    if not job_desc_skills_set or not job_posting_skills_set:
        return 0.0

    # Jaccard 유사도
    intersection = len(job_desc_skills_set & job_posting_skills_set)
    union = len(job_desc_skills_set | job_posting_skills_set)
    jaccard = intersection / union if union > 0 else 0

    # Coverage (채용공고 스킬 중 몇 %가 직무기술서와 매칭되는가)
    coverage = intersection / len(job_posting_skills_set) if job_posting_skills_set else 0

    # 가중 평균
    return 0.7 * jaccard + 0.3 * coverage

# ============================================================================
# 7. 개선된 Hybrid Scorer (스킬 매칭 포함)
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

            # 직무기술서의 스킬 추출
            job_desc_skill_set = job_desc.metadata.get('skill_set', [])

            # 스킬 매칭 점수 계산
            skill_match = calculate_skill_match_score(job_desc_skill_set, job_posting_skills)
            skill_scores.append(skill_match)

        # BM25 Softmax 정규화
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
print(f"Ensemble Retriever 생성 완료 (ChromaDB: {CHROMA_WEIGHT}, BM25: {BM25_WEIGHT}, Skill: {SKILL_WEIGHT})")

hybrid_scorer = EnhancedHybridScorer()

# ============================================================================
# 8. 개선된 Query Rewrite (채용공고를 쿼리로 변환)
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

    # skill_set 추출
    skill_set = extract_job_posting_skills(job)
    skill_text = ", ".join(skill_set[:5]) if skill_set else ""

    # meta_data에서 job_category 추출
    meta_data = job.get('meta_data', {})
    job_category = meta_data.get('job_category', '') if isinstance(meta_data, dict) else ''

    # 쿼리 1: 제목 + 스킬 중심
    queries.append(f"{title} {skill_text}")

    # 쿼리 2: 직무분야 + 스킬 중심
    if job_category and skill_text:
        queries.append(f"{job_category} {skill_text}")

    # 쿼리 3: LLM 생성
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

def search_matching_job_descriptions(job, top_k=5):
    """채용공고에 매칭되는 직무기술서 검색"""
    queries = rewrite_query_for_job_posting(job)

    all_docs = {}

    for query in queries:
        results = ensemble_retriever.invoke(query)

        for doc in results:
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

    # 평균 점수와 출현 횟수를 모두 고려
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
# 9. 검색 실행 및 가중평균 합산 기반 순위 산출
# ============================================================================
all_results = []

print("\n" + "="*100)
print("전체 채용공고에 대한 직무기술서 매칭 시작 (멀티쿼리 + 스킬매칭 강화)...")
print("="*100)

for idx, job in enumerate(job_postings):
    # 채용공고 정보 추출
    title = job.get('title', 'N/A')
    company = job.get('company', 'N/A')

    # 스킬 추출
    job_skills = extract_job_posting_skills(job)

    # 멀티 쿼리 검색
    matching_job_descs = search_matching_job_descriptions(job, top_k=10)

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

    # weighted_sum 기준 내림차순 정렬
    job_descs_with_scores.sort(key=lambda x: x[1]['weighted_sum'], reverse=True)

    # 상위 3개만 선택
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
# 10. TXT 파일 저장
# ============================================================================
def save_results_to_txt(results, filename):
    """검색 결과를 TXT 파일로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("채용공고 → 적합 직무 & 산업 매칭 결과 (직무기술서를 Vector DB에 저장)\n")
        f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 채용공고 수: {len(results)}개\n")
        f.write(f"총 직무기술서 수: {len(df)}개\n")
        f.write(f"스코어링 가중치: BM25={BM25_WEIGHT}, Chroma={CHROMA_WEIGHT}, Skill={SKILL_WEIGHT}\n")
        f.write("="*100 + "\n\n")

        for idx, result in enumerate(results, 1):
            f.write(f"\n{'='*100}\n")
            f.write(f"[{idx}] 채용공고: {result['job_posting_title']}\n")
            f.write(f"{'='*100}\n")
            f.write(f"회사: {result['company']}\n")
            skill_set_display = ", ".join(result['job_skills']) if isinstance(result['job_skills'], list) else str(result['job_skills'])
            f.write(f"요구 스킬: {skill_set_display}\n")
            f.write(f"URL: {result['job_url']}\n")
            f.write(f"\n{'추천 직무 & 산업 Top 3':-^90}\n\n")

            for job_role in result['matched_job_roles']:
                f.write(f"{job_role['rank']}. 직무: {job_role['recommended_job_title']} | 산업: {job_role['recommended_industry']}\n")
                skill_set_str = ", ".join(job_role['required_skill_set']) if isinstance(job_role['required_skill_set'], list) and job_role['required_skill_set'] else "N/A"
                f.write(f"   해당 직무 필수 스킬: {skill_set_str}\n")
                f.write(f"   \n")
                f.write(f"   [매칭 점수 상세]\n")
                f.write(f"   - 최종 가중 합산: {job_role['weighted_sum']:.6f}\n")
                f.write(f"   - 스킬 매칭 점수 ({SKILL_WEIGHT}): {job_role['skill_match']:.6f}\n")
                f.write(f"   - BM25 정규화 ({BM25_WEIGHT}): {job_role['bm25_normalized']:.6f}\n")
                f.write(f"   - ChromaDB 코사인 ({CHROMA_WEIGHT}): {job_role['chroma_cosine']:.6f}\n")
                f.write(f"   - 멀티쿼리 매칭 횟수: {job_role['multi_query_count']}회\n")
                f.write(f"   \n\n")

    print(f"TXT 파일 저장 완료: {filename}")

# 결과 파일 저장
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
# 11. 결과 출력 (콘솔)
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
print("채용공고 → 직무 & 산업 매칭 완료! (직무기술서 Vector DB 기반)")
print("="*100)
