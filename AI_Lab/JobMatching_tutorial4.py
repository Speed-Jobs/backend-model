import pandas as pd
import json
import os
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
import re

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
# AI_Lab/data 폴더에서 파일 찾기
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
    return skills  # <<<<<<< 여기서 상한제거

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
# 3. Document 변환 (개선)
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

documents = []
for job in job_postings:
    # Check nested structure first: skill_set_info.skill_set
    if 'skill_set_info' in job and isinstance(job['skill_set_info'], dict):
        skill_set = job['skill_set_info'].get('skill_set', [])
    else:
        # Fallback to top-level skill_set
        skill_set = job.get('skill_set', [])

    skill_set_text = ", ".join(skill_set) if isinstance(skill_set, list) else str(skill_set)
    required_skills_text = ""
    job_category = ""

    meta_data = job.get('meta_data', {})
    if isinstance(meta_data, dict) and meta_data:
        if 'required_skills' in meta_data and meta_data['required_skills']:
            required_skills_text = ", ".join(meta_data['required_skills']) if isinstance(meta_data['required_skills'], list) else str(meta_data['required_skills'])
        if 'job_category' in meta_data:
            job_category = meta_data['job_category']

    # 더 풍부한 검색을 위한 콘텐츠 구성
    content = f"""
    제목: {job.get('title', '')}
    회사: {job.get('company', '')}
    직무분야: {job_category}
    경력: {job.get('experience', '')}
    핵심기술: {skill_set_text}
    필수역량: {required_skills_text}
    상세내용: {job.get('description', '')[:500] if job.get('description') else ''}
    """
    
    # ChromaDB는 metadata에 리스트 타입을 허용하지 않으므로 문자열로 변환
    metadata = {
        "title": job.get('title', ''),
        "company": job.get('company', ''),
        "url": job.get('url', ''),
        "job_category": job_category,
        "experience": job.get('experience', ''),
        "skill_set": skill_set_text if skill_set_text else ""  # 리스트 대신 문자열 사용
    }
    # 복잡한 metadata 필터링 (리스트, 딕셔너리 등 제거 또는 변환)
    cleaned_metadata = clean_metadata(metadata)
    doc = Document(page_content=content.strip(), metadata=cleaned_metadata)
    documents.append(doc)

print(f"Document 변환 완료: {len(documents)}개")

if len(documents) == 0:
    raise ValueError("변환된 Document가 없습니다.")

# ============================================================================
# 4. ChromaDB 및 BM25 Retriever 생성
# ============================================================================
print("ChromaDB 벡터 스토어 생성 중...")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# ChromaDB persist 이슈로 인해 in-memory 모드 사용
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="job_postings"
)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
print(f"ChromaDB 생성 완료 (in-memory 모드)")

print("BM25 Retriever 생성 중...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 15
print("BM25 Retriever 생성 완료")

# ============================================================================
# 5. Hybrid Retriever (개선)
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
                docs_with_scores = vectorstore.similarity_search_with_score(query, k=15)
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
# 6. 스킬 매칭 점수 계산 (수정: 문자열 파싱 문제 해결)
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

def calculate_skill_match_score(job_skills, query_skills):
    """스킬 집합 간 유사도 계산 (Jaccard + Coverage)"""
    if not job_skills or not query_skills:
        return 0.0

    # 문자열을 리스트로 변환 후 set 생성
    job_skills_list = parse_skill_string_to_list(job_skills)
    query_skills_list = parse_skill_string_to_list(query_skills)

    job_skills_set = set([s.lower().strip() for s in job_skills_list if s])
    query_skills_set = set([s.lower().strip() for s in query_skills_list if s])

    if not job_skills_set or not query_skills_set:
        return 0.0

    # Jaccard 유사도
    intersection = len(job_skills_set & query_skills_set)
    union = len(job_skills_set | query_skills_set)
    jaccard = intersection / union if union > 0 else 0

    # Coverage (쿼리 스킬 중 몇 %가 매칭되는가)
    coverage = intersection / len(query_skills_set) if query_skills_set else 0

    # 가중 평균 (Jaccard에 더 높은 가중치)
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

    def calculate_scores(self, job_results, query_skills):
        if not job_results:
            return []

        bm25_scores = []
        chroma_scores = []
        skill_scores = []

        for job in job_results:
            bm25_scores.append(job.metadata.get('bm25_raw_score', 0))
            chroma_scores.append(job.metadata.get('chroma_raw_score', 0))
            
            # 채용공고의 스킬 추출
            job_skill_set = job.metadata.get('skill_set', [])
            
            # 스킬 매칭 점수 계산
            skill_match = calculate_skill_match_score(job_skill_set, query_skills)
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
# 8. 개선된 Query Rewrite (멀티 쿼리 생성)
# ============================================================================
query_rewrite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 채용공고 검색을 위한 쿼리 최적화 전문가입니다.
주어진 정보를 바탕으로 검색에 최적화된 자연어 쿼리를 생성하세요.

요구사항:
- 핵심 키워드를 자연스럽게 포함
- 검색 의도가 명확한 문장 형태
- 50-100자 이내로 간결하게
- 직무, 산업, 기술 스택이 모두 포함되도록

예시:
입력: 직무=백엔드 개발자, 산업=IT, 스킬=Python, FastAPI, Docker
출력: IT 분야에서 Python, FastAPI, Docker를 활용하는 백엔드 개발자 채용공고"""),
    ("user", """직무: {job_title}
산업: {industry}
핵심 스킬: {core_skills}

위 정보로 채용공고 검색에 최적화된 자연어 쿼리를 생성해주세요.""")
])

query_rewrite_chain = query_rewrite_prompt | query_rewrite_llm

def rewrite_query_multi(industry, skill_set, job_title):
    """멀티 쿼리 생성으로 더 다양한 검색"""
    queries = []
    
    # 스킬 전체를 사용하도록 수정
    if isinstance(skill_set, list) and len(skill_set) > 0:
        core_skill_text = ", ".join(skill_set)
    else:
        core_skill_text = ""
    
    # 쿼리 1: 직무 중심
    queries.append(f"{job_title} {industry} {core_skill_text}")
    
    # 쿼리 2: 스킬 중심
    if core_skill_text:
        queries.append(f"{core_skill_text} 활용 {job_title} 채용")
    
    # 쿼리 3: LLM 생성
    try:
        result = query_rewrite_chain.invoke({
            "job_title": job_title,
            "industry": industry,
            "core_skills": core_skill_text
        })
        queries.append(result.content.strip())
    except Exception as e:
        print(f"Query rewrite 실패: {e}")
    
    return queries

def search_similar_jobs_multi_query(industry, skill_set, job_title, top_k=30):
    """멀티 쿼리로 검색하고 결과 통합"""
    queries = rewrite_query_multi(industry, skill_set, job_title)
    
    all_docs = {}
    
    for query in queries:
        results = ensemble_retriever.invoke(query)
        
        for doc in results:
            doc_id = doc.page_content[:100]
            
            if doc_id in all_docs:
                # 여러 쿼리에서 나온 문서는 점수 부스팅
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
        # 여러 쿼리에서 매칭될수록 보너스
        boost = 1 + 0.15 * (count - 1)
        
        # 평균 점수에 boost 적용
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
print("전체 직무기술서에 대한 채용공고 매칭 시작 (멀티쿼리 + 스킬매칭 강화)...")
print("="*100)

for idx, row in df.iterrows():
    industry = row['industry']
    job_title = row['직무']
    
    # 파싱된 전체 스킬셋 사용
    skill_set = row['parsed_skills']
    
    # 멀티 쿼리 검색
    similar_jobs = search_similar_jobs_multi_query(industry, skill_set, job_title, top_k=30)
    
    # 스킬 매칭 포함 스코어링
    hybrid_scores = hybrid_scorer.calculate_scores(similar_jobs, skill_set)

    # 가중평균 합산 점수로 재정렬
    jobs_with_scores = []
    for i, job in enumerate(similar_jobs):
        hybrid_score = hybrid_scores[i] if i < len(hybrid_scores) else {
            'weighted_sum': 0,
            'skill_match': 0,
            'bm25_normalized': 0,
            'chroma_cosine': 0
        }
        jobs_with_scores.append((job, hybrid_score))

    # weighted_sum 기준 내림차순 정렬
    jobs_with_scores.sort(key=lambda x: x[1]['weighted_sum'], reverse=True)

    # 상위 5개만 선택
    top_5_jobs = jobs_with_scores[:5]

    result_item = {
        'job_title': job_title,
        'industry': industry,
        'skill_set': skill_set,
        'matched_jobs': []
    }

    for rank, (job, hybrid_score) in enumerate(top_5_jobs, 1):
        # metadata의 skill_set은 문자열이므로 리스트로 변환
        job_skill_set_raw = job.metadata.get('skill_set', '')
        job_skill_set = parse_skill_string_to_list(job_skill_set_raw) if job_skill_set_raw else []
        
        result_item['matched_jobs'].append({
            'rank': rank,
            'title': job.metadata.get('title', 'N/A'),
            'company': job.metadata.get('company', 'N/A'),
            'job_category': job.metadata.get('job_category', 'N/A'),
            'experience': job.metadata.get('experience', 'N/A'),
            'skill_set': job_skill_set,
            'url': job.metadata.get('url', 'N/A'),
            'weighted_sum': hybrid_score['weighted_sum'],
            'skill_match': hybrid_score['skill_match'],
            'bm25_normalized': hybrid_score['bm25_normalized'],
            'chroma_cosine': hybrid_score['chroma_cosine'],
            'multi_query_count': job.metadata.get('multi_query_count', 1)
        })

    all_results.append(result_item)

    if (idx + 1) % 10 == 0:
        print(f"진행 중... {idx + 1}/{len(df)} 완료")

print(f"\n매칭 완료! 총 {len(all_results)}개 직무기술서 처리")

# ============================================================================
# 10. TXT 파일 저장
# ============================================================================
def save_results_to_txt(results, filename):
    """검색 결과를 TXT 파일로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("직무기술서 - 채용공고 매칭 결과 (멀티쿼리 + 스킬매칭 강화)\n")
        f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 직무기술서 수: {len(results)}개\n")
        f.write(f"총 채용공고 수: {len(job_postings)}개\n")
        f.write(f"스코어링 가중치: BM25={BM25_WEIGHT}, Chroma={CHROMA_WEIGHT}, Skill={SKILL_WEIGHT}\n")
        f.write("="*100 + "\n\n")

        for idx, result in enumerate(results, 1):
            f.write(f"\n{'='*100}\n")
            f.write(f"[{idx}] 직무: {result['job_title']}\n")
            f.write(f"{'='*100}\n")
            f.write(f"Industry: {result['industry']}\n")
            skill_set_display = ", ".join(result['skill_set']) if isinstance(result['skill_set'], list) else str(result['skill_set'])
            f.write(f"핵심 Skill Set: {skill_set_display}\n")
            f.write(f"\n{'추천 채용공고 Top 5':-^90}\n\n")

            for job in result['matched_jobs']:
                f.write(f"{job['rank']}. {job['title']}\n")
                f.write(f"   회사: {job['company']}\n")
                f.write(f"   업무분야: {job['job_category']}\n")
                f.write(f"   경력: {job['experience']}\n")
                skill_set_str = ", ".join(job['skill_set']) if isinstance(job['skill_set'], list) and job['skill_set'] else "N/A"
                f.write(f"   필요 skill_set: {skill_set_str}\n")
                f.write(f"   \n")
                f.write(f"   [매칭 점수 상세]\n")
                f.write(f"   - 최종 가중 합산: {job['weighted_sum']:.6f}\n")
                f.write(f"   - 스킬 매칭 점수 ({SKILL_WEIGHT}): {job['skill_match']:.6f}\n")
                f.write(f"   - BM25 정규화 ({BM25_WEIGHT}): {job['bm25_normalized']:.6f}\n")
                f.write(f"   - ChromaDB 코사인 ({CHROMA_WEIGHT}): {job['chroma_cosine']:.6f}\n")
                f.write(f"   - 멀티쿼리 매칭 횟수: {job['multi_query_count']}회\n")
                f.write(f"   \n")
                f.write(f"   URL: {job['url']}\n\n")

    print(f"TXT 파일 저장 완료: {filename}")

# PDF 저장 함수는 동일하게 유지 (생략)
def save_results_to_pdf(results, filename):
    """검색 결과를 PDF 파일로 저장"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.enums import TA_CENTER

        font_name = 'Helvetica'
        windows_fonts_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        
        malgun_font_path = os.path.join(windows_fonts_dir, 'malgun.ttf')
        if os.path.exists(malgun_font_path):
            try:
                pdfmetrics.registerFont(TTFont('Malgun', malgun_font_path))
                font_name = 'Malgun'
            except:
                pass
        
        if font_name == 'Helvetica':
            print("한글 폰트를 찾을 수 없습니다. 영문 폰트로 대체합니다.")

        doc = SimpleDocTemplate(filename, pagesize=A4)
        story = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName=font_name,
            fontSize=16,
            alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=12
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=9
        )

        story.append(Paragraph("직무기술서 - 채용공고 매칭 결과", title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        story.append(Paragraph(f"총 직무기술서 수: {len(results)}개 | 총 채용공고 수: {len(job_postings)}개", normal_style))
        story.append(Spacer(1, 0.3*inch))

        for idx, result in enumerate(results, 1):
            story.append(Paragraph(f"[{idx}] 직무: {result['job_title']}", heading_style))
            story.append(Paragraph(f"Industry: {result['industry']}", normal_style))
            skill_display = ", ".join(result['skill_set']) if isinstance(result['skill_set'], list) else str(result['skill_set'])
            story.append(Paragraph(f"핵심 Skill: {skill_display}", normal_style))
            story.append(Spacer(1, 0.1*inch))

            story.append(Paragraph("추천 채용공고 Top 5", heading_style))
            for job in result['matched_jobs']:
                story.append(Paragraph(
                    f"{job['rank']}. {job['title']} - {job['company']}",
                    normal_style
                ))
                story.append(Paragraph(
                    f"   업무분야: {job['job_category']} | 경력: {job['experience']}",
                    normal_style
                ))
                skill_set_str = ", ".join(job['skill_set']) if isinstance(job['skill_set'], list) and job['skill_set'] else "N/A"
                story.append(Paragraph(
                    f"   필요 skill: {skill_set_str}",
                    normal_style
                ))
                story.append(Paragraph(
                    f"   점수: {job['weighted_sum']:.4f} (스킬:{job['skill_match']:.3f}|BM25:{job['bm25_normalized']:.3f}|Chroma:{job['chroma_cosine']:.3f})",
                    normal_style
                ))
                story.append(Paragraph(f"   URL: {job['url']}", normal_style))
                story.append(Spacer(1, 0.05*inch))

            story.append(Spacer(1, 0.2*inch))

            if idx % 10 == 0:
                story.append(PageBreak())

        doc.build(story)
        print(f"PDF 파일 저장 완료: {filename}")

    except ImportError:
        print("reportlab 라이브러리가 설치되어 있지 않습니다.")
    except Exception as e:
        print(f"PDF 생성 중 오류 발생: {str(e)}")

# 결과 파일 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(script_dir, "output")
os.makedirs(output_dir, exist_ok=True)

txt_filename = os.path.join(output_dir, f"job_matching_results_{timestamp}.txt")
pdf_filename = os.path.join(output_dir, f"job_matching_results_{timestamp}.pdf")

save_results_to_txt(all_results, txt_filename)
save_results_to_pdf(all_results, pdf_filename)

print("\n" + "="*100)
print("결과 파일 저장 완료!")
print(f"TXT 파일: {txt_filename}")
print(f"PDF 파일: {pdf_filename}")
print("="*100)

# ============================================================================
# 11. 결과 출력 (콘솔)
# ============================================================================
print("\n\n[검색 결과 샘플 - 처음 3개]")
for idx, result in enumerate(all_results[:3], 1):
    print(f"\n{'='*100}")
    print(f"[{idx}] 직무: {result['job_title']}")
    print(f"Industry: {result['industry']}")
    skill_set_display = ", ".join(result['skill_set']) if isinstance(result['skill_set'], list) else str(result['skill_set'])
    print(f"핵심 Skill Set: {skill_set_display}")
    print(f"{'='*100}")
    for job in result['matched_jobs']:
        print(f"\n{job['rank']}. {job['title']}")
        print(f"   회사: {job['company']}")
        print(f"   업무분야: {job['job_category']}")
        print(f"   경력: {job['experience']}")
        skill_set_str = ", ".join(job['skill_set']) if isinstance(job['skill_set'], list) and job['skill_set'] else "N/A"
        print(f"   필요 skill_set: {skill_set_str}")
        print(f"   ")
        print(f"   [매칭 점수 상세]")
        print(f"   - 최종 가중 합산: {job['weighted_sum']:.6f}")
        print(f"   - 스킬 매칭 점수: {job['skill_match']:.6f}")
        print(f"   - BM25 정규화: {job['bm25_normalized']:.6f}")
        print(f"   - ChromaDB 코사인: {job['chroma_cosine']:.6f}")
        print(f"   - 멀티쿼리 매칭: {job['multi_query_count']}회")
        print(f"   ")
        print(f"   URL: {job['url']}")

print("\n" + "="*100)
print("멀티쿼리 + 스킬매칭 강화 매칭 완료!")
print("="*100)