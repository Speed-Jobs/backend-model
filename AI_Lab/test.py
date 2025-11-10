# %% [markdown]
# ## 직무 기술서 데이터 불러오기

# %%
import pandas as pd
import json

# JSON 파일 경로
json_path = "C:/workspace/Final_project/backend-model/AI_Lab/data/job_description.json"

# JSON 파일 불러오기
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

# 데이터가 딕셔너리 또는 리스트 형식일 수 있음에 유의
# 판단스 데이터프레임 생성
df = pd.DataFrame(data)

df.head()


# %% [markdown]
# ## 채용공고 데이터 불러오기

# %%
# 사용할 회사명을 변수로 지정 (예: "kakao" 또는 "woowahan" 등)
company_name = "kakao"

# 파일명 생성
jobs_json_path = f"C:/workspace/Final_project/backend-model/AI_Lab/data/{company_name}_jobs.json"

# JSON 파일 불러오기
with open(jobs_json_path, encoding="utf-8") as f:
    jobs_data = json.load(f)

# 데이터프레임으로 변환 (만약 리스트 구조라면)
jobs_df = pd.DataFrame(jobs_data)

# 미리보기
jobs_df.drop('html', axis=1,inplace=True)


# %%
jobs_df.head()

# %%
df_solution_dev_erp_scm = df[(df['직무'] == 'Solution Development') & (df['industry'] == 'ERP_SCM (Supply Chain Management)')]
df_solution_dev_erp_scm


# %%
df_solution_dev_erp_scm['skill_set'].values

# %% [markdown]
# ## 하이브리드 Retriever - 유사 채용공고 검색

# %%
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import pandas as pd
import json
import os

# OpenAI API 키 설정 (필요시 주석 해제)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 1. 채용공고 데이터 로드 (kakao와 woowahan 데이터 합치기)
kakao_path = "C:/workspace/Final_project/backend-model/AI_Lab/data/kakao_jobs.json"
woowahan_path = "C:/workspace/Final_project/backend-model/AI_Lab/data/woowahan_jobs.json"

job_postings = []

# kakao 채용공고 로드
with open(kakao_path, 'r', encoding='utf-8') as f:
    kakao_jobs = json.load(f)
    job_postings.extend(kakao_jobs)

# woowahan 채용공고 로드
with open(woowahan_path, 'r', encoding='utf-8') as f:
    woowahan_jobs = json.load(f)
    job_postings.extend(woowahan_jobs)

print(f"총 채용공고 수: {len(job_postings)}")

# %%
# 2. Document 객체로 변환
documents = []
for job in job_postings:
    # 검색에 사용할 텍스트 구성
    # skill_set 처리 (리스트인 경우 문자열로 변환)
    skill_set_text = ""
    if 'skill_set' in job and job['skill_set']:
        if isinstance(job['skill_set'], list):
            skill_set_text = ", ".join(job['skill_set'])
        else:
            skill_set_text = str(job['skill_set'])
    
    # meta_data에서 required_skills 추출
    required_skills_text = ""
    if 'meta_data' in job and job['meta_data']:
        if 'required_skills' in job['meta_data'] and job['meta_data']['required_skills']:
            required_skills_text = ", ".join(job['meta_data']['required_skills'])
    
    content = f"""
    제목: {job.get('title', '')}
    회사: {job.get('company', '')}
    직무내용: {job.get('description', '')[:500]}
    업무분야: {job.get('meta_data', {}).get('job_category', '')}
    필요 스킬: {required_skills_text}
    스킬셋: {skill_set_text}
    경력: {job.get('experience', '')}
    """
    
    metadata = {
        "title": job.get('title', ''),
        "company": job.get('company', ''),
        "url": job.get('url', ''),
        "job_category": job.get('meta_data', {}).get('job_category', ''),
        "experience": job.get('experience', ''),
    }
    
    documents.append(Document(page_content=content.strip(), metadata=metadata))

print(f"Document 변환 완료: {len(documents)}개")

# %%
# 3. ChromaDB retriever 생성
print("ChromaDB 벡터 스토어 생성 중...")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="job_postings"
)
chroma_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)
print("ChromaDB 생성 완료")

# %%
# 4. BM25 retriever 생성
print("BM25 Retriever 생성 중...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10
print("BM25 Retriever 생성 완료")

# %%
# 5. Ensemble Retriever 생성 (의미론적 검색과 키워드 검색 결합)
ensemble_retriever = EnsembleRetriever(
    retrievers=[chroma_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # 의미론적 검색에 60%, 키워드 검색에 40% 가중치
)
print("Ensemble Retriever 생성 완료")

# %%
# 6. 직무기술서 데이터프레임 (이미 로드된 df 사용)
print(f"직무기술서 데이터: {len(df)}개")
df.head()

# %%
def search_similar_jobs(industry, skill_set, top_k=5):
    """
    직무기술서의 industry와 skill_set으로 유사한 채용공고 검색
    
    Args:
        industry: 업종/산업
        skill_set: 필요 스킬셋 (리스트 또는 문자열)
        top_k: 반환할 상위 결과 수
    
    Returns:
        유사한 채용공고 리스트
    """
    # skill_set이 리스트인 경우 문자열로 변환
    if isinstance(skill_set, list):
        skill_set_str = " ".join(skill_set)
    else:
        skill_set_str = str(skill_set)
    
    # 검색 쿼리 구성
    query = f"{industry} {skill_set_str}"
    
    # 검색 실행
    results = ensemble_retriever.get_relevant_documents(query)
    
    return results[:top_k]

# %%
# 7. 각 직무기술서에 대해 유사한 채용공고 찾기 (예시: 처음 5개만)
for idx, row in df.head(5).iterrows():
    industry = row['industry']
    skill_set = row['공통_skill_set']
    
    print(f"\n{'='*100}")
    print(f"직무: {row['직무']}")
    print(f"Industry: {industry}")
    print(f"Skill Set (샘플): {skill_set[:5] if isinstance(skill_set, list) else skill_set[:100]}")
    print(f"{'='*100}")
    
    similar_jobs = search_similar_jobs(industry, skill_set, top_k=5)
    
    for i, job in enumerate(similar_jobs, 1):
        print(f"\n{i}. {job.metadata['title']}")
        print(f"   회사: {job.metadata['company']}")
        print(f"   업무분야: {job.metadata.get('job_category', 'N/A')}")
        print(f"   경력: {job.metadata.get('experience', 'N/A')}")
        print(f"   URL: {job.metadata['url']}")

# %%
# 특정 직무기술서 예시 조회
# Solution Development - ERP_SCM 예시
df_solution_dev = df[df['직무'] == 'Solution Development']
if len(df_solution_dev) > 0:
    print(f"\n\n{'#'*100}")
    print("# Solution Development 직무 검색 예시")
    print(f"{'#'*100}")
    
    row = df_solution_dev.iloc[0]
    industry = row['industry']
    skill_set = row['공통_skill_set']
    
    print(f"\n직무: {row['직무']}")
    print(f"Industry: {industry}")
    
    similar_jobs = search_similar_jobs(industry, skill_set, top_k=3)
    
    for i, job in enumerate(similar_jobs, 1):
        print(f"\n{i}. {job.metadata['title']}")
        print(f"   회사: {job.metadata['company']}")
        print(f"   URL: {job.metadata['url']}")

# %%




