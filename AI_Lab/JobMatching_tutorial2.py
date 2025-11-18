import pandas as pd
import json
import os
from dotenv import load_dotenv
import numpy as np
from datetime import datetime

# 환경 변수 설정
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY", None)
if openai_api_key is None:
    raise ValueError(".env 파일에 'OPENAI_API_KEY'가 없습니다.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# LangChain 라이브러리
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import ChatPromptTemplate
from typing import List

# ============================================================================
# 1. 데이터 로드
# ============================================================================
# 직무기술서 로드
json_path = "C:/workspace/fproject/backend-model/AI_Lab/data/job_description.json"
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)
print(f"직무기술서 데이터: {len(df)}개")

# 채용공고 로드 (여러 회사)
job_postings = []
job_paths = [
    "C:/workspace/fproject/backend-model/AI_Lab/data/kakao_jobs.json",
    "C:/workspace/fproject/backend-model/AI_Lab/data/woowahan_jobs.json",
    "C:/workspace/fproject/backend-model/AI_Lab/data/hanwha_jobs.json",
    "C:/workspace/fproject/backend-model/AI_Lab/data/line_jobs.json",
    "C:/workspace/fproject/backend-model/AI_Lab/data/naver_jobs.json",
]

for path in job_paths:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            job_postings.extend(json.load(f))
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {path}")
    except json.JSONDecodeError:
        print(f"JSON decode 오류: {path}")

print(f"총 채용공고 수: {len(job_postings)}")

# ============================================================================
# 2. Document 변환
# ============================================================================
documents = []
for job in job_postings:
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

    content = f"""
    제목: {job.get('title', '')}
    회사: {job.get('company', '')}
    직무내용: {job.get('description', '')[:500] if job.get('description') else ''}
    업무분야: {job_category}
    필요 스킬: {required_skills_text}
    스킬셋: {skill_set_text}
    경력: {job.get('experience', '')}
    """
    metadata = {
        "title": job.get('title', ''),
        "company": job.get('company', ''),
        "url": job.get('url', ''),
        "job_category": job_category,
        "experience": job.get('experience', ''),
    }
    documents.append(Document(page_content=content.strip(), metadata=metadata))
print(f"Document 변환 완료: {len(documents)}개")

# ============================================================================
# 3. ChromaDB 및 BM25 Retriever 생성
# ============================================================================
print("ChromaDB 벡터 스토어 생성 중...")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="job_postings"
)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
print("ChromaDB 생성 완료")

print("BM25 Retriever 생성 중...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10
print("BM25 Retriever 생성 완료")

# ============================================================================
# 4. Hybrid Retriever (원본 점수 추출 포함)
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
                    'formula': f"{weight} / ({rank} + {self.k}) = {rrf_score:.6f}"
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
# 5. 가중치 설정 및 Hybrid Scorer
# ============================================================================
CHROMA_WEIGHT = 0.3
BM25_WEIGHT = 0.7

ensemble_retriever = CustomEnsembleRetriever(
    retrievers=[chroma_retriever, bm25_retriever],
    weights=[CHROMA_WEIGHT, BM25_WEIGHT]
)
print(f"Ensemble Retriever 생성 완료 (ChromaDB: {CHROMA_WEIGHT}, BM25: {BM25_WEIGHT})")

class HybridScorer:
    """BM25 Softmax 정규화 + ChromaDB 코사인 유사도 가중 합산"""
    def __init__(self, bm25_weight=BM25_WEIGHT, chroma_weight=CHROMA_WEIGHT):
        self.bm25_weight = bm25_weight
        self.chroma_weight = chroma_weight

    def calculate_scores(self, job_results):
        if not job_results:
            return []

        bm25_scores = []
        chroma_scores = []

        for job in job_results:
            bm25_scores.append(job.metadata.get('bm25_raw_score', 0))
            chroma_scores.append(job.metadata.get('chroma_raw_score', 0))

        bm25_array = np.array(bm25_scores)

        if np.all(bm25_array == 0):
            bm25_normalized = np.zeros_like(bm25_array)
        else:
            exp_bm25 = np.exp(bm25_array - np.max(bm25_array))
            bm25_normalized = exp_bm25 / exp_bm25.sum()

        results = []
        for i, (bm25_norm, chroma) in enumerate(zip(bm25_normalized, chroma_scores)):
            weighted_sum = self.bm25_weight * bm25_norm + self.chroma_weight * chroma
            results.append({
                'bm25_normalized': float(bm25_norm),
                'chroma_cosine': float(chroma),
                'weighted_sum': float(weighted_sum)
            })

        return results

hybrid_scorer = HybridScorer()

# ============================================================================
# 6. Query Rewrite
# ============================================================================
query_rewrite_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
query_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 채용공고 검색을 위한 쿼리 최적화 전문가입니다.
주어진 정보를 바탕으로 검색에 최적화된 자연어 쿼리를 생성하세요.

요구사항:
- 핵심 키워드를 자연스럽게 포함
- 검색 의도가 명확한 문장 형태
- 50-80자 이내로 간결하게
- 직무, 산업, 기술 스택이 모두 포함되도록

예시:
입력: 직무=백엔드 개발자, 산업=IT, 스킬=Python, FastAPI, Docker
출력: IT 분야에서 Python, FastAPI, Docker를 활용하는 백엔드 개발자 채용공고"""),
    ("user", """직무: {job_title}
산업: {industry}
필수 스킬: {skills}

위 정보로 채용공고 검색에 최적화된 자연어 쿼리를 생성해주세요.""")
])

query_rewrite_chain = query_rewrite_prompt | query_rewrite_llm

def rewrite_query_with_llm(industry, skill_set, job_title):
    if isinstance(skill_set, list):
        top_skills = skill_set[:7]
        skill_text = ", ".join(top_skills)
    else:
        skill_text = str(skill_set)[:100]

    try:
        result = query_rewrite_chain.invoke({
            "job_title": job_title,
            "industry": industry,
            "skills": skill_text
        })
        rewritten_query = result.content.strip()
        return rewritten_query
    except Exception as e:
        print(f"Query rewrite 실패, 기본 쿼리 사용: {e}")
        return f"{industry} {job_title} {skill_text}"

def search_similar_jobs(industry, skill_set, job_title, top_k=20):
    query = rewrite_query_with_llm(industry, skill_set, job_title)
    if not hasattr(search_similar_jobs, 'debug_count'):
        search_similar_jobs.debug_count = 0
    if search_similar_jobs.debug_count < 5:
        print(f"검색 쿼리 [{job_title}]: {query}")
        search_similar_jobs.debug_count += 1
    results = ensemble_retriever.invoke(query)
    return results[:top_k]

# ============================================================================
# 7. 검색 실행 및 가중평균 합산 기반 순위 산출
# ============================================================================
all_results = []

print("\n" + "="*100)
print("전체 직무기술서에 대한 채용공고 매칭 시작 (가중평균 합산 기준)...")
print("="*100)

for idx, row in df.iterrows():
    industry = row['industry']
    # skill_set 타입 안전하게 처리
    common_skills = row['공통_skill_set'] if isinstance(row['공통_skill_set'], list) else []
    job_skills = row['skill_set'] if isinstance(row['skill_set'], list) else []
    skill_set = common_skills + job_skills
    job_title = row['직무']

    similar_jobs = search_similar_jobs(industry, skill_set, job_title, top_k=20)
    hybrid_scores = hybrid_scorer.calculate_scores(similar_jobs)

    # 가중평균 합산 점수로 재정렬
    jobs_with_scores = []
    for i, job in enumerate(similar_jobs):
        hybrid_score = hybrid_scores[i] if i < len(hybrid_scores) else {'weighted_sum': 0}
        jobs_with_scores.append((job, hybrid_score))

    # weighted_sum 기준 내림차순 정렬
    jobs_with_scores.sort(key=lambda x: x[1]['weighted_sum'], reverse=True)

    # 상위 5개만 선택
    top_5_jobs = jobs_with_scores[:5]

    result_item = {
        'job_title': job_title,
        'industry': industry,
        'skill_set': skill_set,  # 전체 스킬셋 저장
        'matched_jobs': []
    }

    for rank, (job, hybrid_score) in enumerate(top_5_jobs, 1):
        # 채용공고의 skill_set 가져오기
        job_skill_set = []
        job_content = job.page_content
        # page_content에서 skill_set 추출 또는 metadata에서 가져오기
        for posting in job_postings:
            if posting.get('title') == job.metadata['title'] and posting.get('company') == job.metadata['company']:
                # skill_set_info 안에 있는 경우와 직접 있는 경우 모두 처리
                if 'skill_set_info' in posting and isinstance(posting['skill_set_info'], dict):
                    job_skill_set = posting['skill_set_info'].get('skill_set', [])
                else:
                    job_skill_set = posting.get('skill_set', [])
                break

        result_item['matched_jobs'].append({
            'rank': rank,
            'title': job.metadata['title'],
            'company': job.metadata['company'],
            'job_category': job.metadata.get('job_category', 'N/A'),
            'experience': job.metadata.get('experience', 'N/A'),
            'skill_set': job_skill_set,  # 채용공고의 skill_set 추가
            'url': job.metadata['url'],
            'weighted_sum': hybrid_score['weighted_sum'],
            'bm25_normalized': hybrid_score['bm25_normalized'],
            'chroma_cosine': hybrid_score['chroma_cosine']
        })

    all_results.append(result_item)

    if (idx + 1) % 10 == 0:
        print(f"진행 중... {idx + 1}/{len(df)} 완료")

print(f"\n매칭 완료! 총 {len(all_results)}개 직무기술서 처리")

# ============================================================================
# 8. TXT 파일 저장
# ============================================================================
def save_results_to_txt(results, filename):
    """검색 결과를 TXT 파일로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("직무기술서 - 채용공고 매칭 결과 (가중평균 합산 기준)\n")
        f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 직무기술서 수: {len(results)}개\n")
        f.write(f"총 채용공고 수: {len(job_postings)}개\n")
        f.write("="*100 + "\n\n")

        for idx, result in enumerate(results, 1):
            f.write(f"\n{'='*100}\n")
            f.write(f"[{idx}] 직무: {result['job_title']}\n")
            f.write(f"{'='*100}\n")
            f.write(f"Industry: {result['industry']}\n")
            # 전체 skill_set 출력
            skill_set_display = ", ".join(result['skill_set']) if isinstance(result['skill_set'], list) else str(result['skill_set'])
            f.write(f"Skill Set: {skill_set_display}\n")
            f.write(f"\n{'추천 채용공고 Top 5':-^90}\n\n")

            for job in result['matched_jobs']:
                f.write(f"{job['rank']}. {job['title']}\n")
                f.write(f"   회사: {job['company']}\n")
                f.write(f"   업무분야: {job['job_category']}\n")
                f.write(f"   경력: {job['experience']}\n")
                # 채용공고의 skill_set 출력
                skill_set_str = ", ".join(job['skill_set']) if isinstance(job['skill_set'], list) and job['skill_set'] else "N/A"
                f.write(f"   필요 skill_set: {skill_set_str}\n")
                f.write(f"   \n")
                f.write(f"   [가중평균 합산 점수]\n")
                f.write(f"   - 가중 합산 ({BM25_WEIGHT}*BM25 + {CHROMA_WEIGHT}*Chroma): {job['weighted_sum']:.6f}\n")
                f.write(f"   - BM25 정규화: {job['bm25_normalized']:.6f}\n")
                f.write(f"   - ChromaDB 코사인: {job['chroma_cosine']:.6f}\n")
                f.write(f"   \n")
                f.write(f"   URL: {job['url']}\n\n")

    print(f"TXT 파일 저장 완료: {filename}")

# PDF 파일로 저장 (reportlab 사용)
def save_results_to_pdf(results, filename):
    """검색 결과를 PDF 파일로 저장 (한글 지원)"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.enums import TA_CENTER

        # 한글 폰트 등록 시도
        try:
            pdfmetrics.registerFont(TTFont('Malgun', 'malgun.ttf'))
            font_name = 'Malgun'
        except:
            try:
                pdfmetrics.registerFont(TTFont('Gulim', 'gulim.ttf'))
                font_name = 'Gulim'
            except:
                print("한글 폰트를 찾을 수 없습니다. 영문 폰트로 대체합니다.")
                font_name = 'Helvetica'

        # PDF 생성
        doc = SimpleDocTemplate(filename, pagesize=A4)
        story = []

        # 스타일 정의
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

        # 제목
        story.append(Paragraph("직무기술서 - 채용공고 매칭 결과 (가중평균 합산 기준)", title_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        story.append(Paragraph(f"총 직무기술서 수: {len(results)}개 | 총 채용공고 수: {len(job_postings)}개", normal_style))
        story.append(Spacer(1, 0.3*inch))

        # 각 결과 추가
        for idx, result in enumerate(results, 1):
            story.append(Paragraph(f"[{idx}] 직무: {result['job_title']}", heading_style))
            story.append(Paragraph(f"Industry: {result['industry']}", normal_style))
            story.append(Paragraph(f"Skill Set: {str(result['skill_set'])[:100]}", normal_style))
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
                # 채용공고의 skill_set 추가
                skill_set_str = ", ".join(job['skill_set']) if isinstance(job['skill_set'], list) and job['skill_set'] else "N/A"
                story.append(Paragraph(
                    f"   필요 skill_set: {skill_set_str}",
                    normal_style
                ))
                story.append(Paragraph(
                    f"   가중합산: {job['weighted_sum']:.6f} | BM25: {job['bm25_normalized']:.6f} | Chroma: {job['chroma_cosine']:.6f}",
                    normal_style
                ))
                story.append(Paragraph(f"   URL: {job['url']}", normal_style))
                story.append(Spacer(1, 0.05*inch))

            story.append(Spacer(1, 0.2*inch))

            # 10개마다 페이지 나누기
            if idx % 10 == 0:
                story.append(PageBreak())

        # PDF 빌드
        doc.build(story)
        print(f"PDF 파일 저장 완료: {filename}")

    except ImportError:
        print("reportlab 라이브러리가 설치되어 있지 않습니다.")
        print("   'pip install reportlab' 명령으로 설치 후 다시 시도하세요.")
    except Exception as e:
        print(f"PDF 생성 중 오류 발생: {str(e)}")

# 결과 파일 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = "C:/workspace/fproject/backend-model/AI_Lab/output"
os.makedirs(output_dir, exist_ok=True)

txt_filename = f"{output_dir}/job_matching_results_{timestamp}.txt"
pdf_filename = f"{output_dir}/job_matching_results_{timestamp}.pdf"

# TXT 저장
save_results_to_txt(all_results, txt_filename)

# PDF 저장
save_results_to_pdf(all_results, pdf_filename)

print("\n" + "="*100)
print("결과 파일 저장 완료!")
print(f"TXT 파일: {txt_filename}")
print(f"PDF 파일: {pdf_filename}")
print("="*100)

# ============================================================================
# 9. 결과 출력 (콘솔)
# ============================================================================
print("\n\n[검색 결과 샘플 - 처음 3개]")
for idx, result in enumerate(all_results[:3], 1):
    print(f"\n{'='*100}")
    print(f"[{idx}] 직무: {result['job_title']}")
    print(f"Industry: {result['industry']}")
    # 전체 skill_set 출력
    skill_set_display = ", ".join(result['skill_set']) if isinstance(result['skill_set'], list) else str(result['skill_set'])
    print(f"Skill Set: {skill_set_display}")
    print(f"{'='*100}")
    for job in result['matched_jobs']:
        print(f"\n{job['rank']}. {job['title']}")
        print(f"   회사: {job['company']}")
        print(f"   업무분야: {job['job_category']}")
        print(f"   경력: {job['experience']}")
        # 채용공고의 skill_set 출력
        skill_set_str = ", ".join(job['skill_set']) if isinstance(job['skill_set'], list) and job['skill_set'] else "N/A"
        print(f"   필요 skill_set: {skill_set_str}")
        print(f"   ")
        print(f"   [가중평균 합산 점수]")
        print(f"   - 가중 합산 ({BM25_WEIGHT}*BM25 + {CHROMA_WEIGHT}*Chroma): {job['weighted_sum']:.6f}")
        print(f"   - BM25 정규화: {job['bm25_normalized']:.6f}")
        print(f"   - ChromaDB 코사인: {job['chroma_cosine']:.6f}")
        print(f"   ")
        print(f"   URL: {job['url']}")

print("\n" + "="*100)
print("가중평균 합산 기준 매칭 완료!")
print("="*100)
