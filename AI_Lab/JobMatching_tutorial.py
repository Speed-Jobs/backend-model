import pandas as pd
import json
import os
from dotenv import load_dotenv

# .env 파일에서 OpenAI API 키 로드 및 환경 변수 설정
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY", None)
if openai_api_key is None:
    raise ValueError(".env 파일에 'OPENAI_API_KEY'가 없습니다. 변수 설정을 확인하세요.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# 직무기술서 데이터 로드 (JSON → DataFrame)
json_path = "C:/workspace/fproject/backend-model/AI_Lab/data/job_description.json"
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)
df.head()

# 채용공고 데이터 로드 (회사별 JSON → DataFrame)
company_name = "kakao"
jobs_json_path = f"C:/workspace/fproject/backend-model/AI_Lab/data/{company_name}_jobs.json"
with open(jobs_json_path, encoding="utf-8") as f:
    jobs_data = json.load(f)
jobs_df = pd.DataFrame(jobs_data)
jobs_df.drop('html', axis=1, inplace=True)
jobs_df.head()


# 하이브리드 검색(Retriever) 관련 라이브러리
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from typing import List

# 카카오, 우아한형제들 채용공고 전체 로드 (데이터 결합)
kakao_path = "C:/workspace/fproject/backend-model/AI_Lab/data/kakao_jobs.json"
woowahan_path = "C:/workspace/fproject/backend-model/AI_Lab/data/woowahan_jobs.json"
job_postings = []
with open(kakao_path, 'r', encoding='utf-8') as f:
    kakao_jobs = json.load(f)
    job_postings.extend(kakao_jobs)
with open(woowahan_path, 'r', encoding='utf-8') as f:
    woowahan_jobs = json.load(f)
    job_postings.extend(woowahan_jobs)
print(f"총 채용공고 수: {len(job_postings)}")

# 채용공고 리스트 → LangChain Document로 변환
documents = []
for job in job_postings:
    skill_set_text = ", ".join(job['skill_set']) if isinstance(job.get('skill_set'), list) else str(job.get('skill_set', ''))
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

# 의미론적 검색(임베딩 기반)용 ChromaDB 생성 및 Retriever 준비
print("ChromaDB 벡터 스토어 생성 중...")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="job_postings"
)
chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
print("ChromaDB 생성 완료")

# 키워드 기반 검색(BM25) Retriever 준비
print("BM25 Retriever 생성 중...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10
print("BM25 Retriever 생성 완료")

# 하이브리드(앙상블) Retriever 정의
class CustomEnsembleRetriever(BaseRetriever):
    """여러 retriever를 결합하여 결과를 종합(가중치 반영)"""
    retrievers: List[BaseRetriever]
    weights: List[float]
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        all_results = []
        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.invoke(query)
            for doc in results:
                doc.metadata['ensemble_score'] = weight
                all_results.append(doc)
        # 내용 일부(PageContent) 기준 중복 제거 및 가중치 합산
        unique_docs = {}
        for doc in all_results:
            doc_id = doc.page_content[:100]
            if doc_id in unique_docs:
                unique_docs[doc_id].metadata['ensemble_score'] += doc.metadata.get('ensemble_score', 0)
            else:
                unique_docs[doc_id] = doc
        sorted_docs = sorted(
            unique_docs.values(), 
            key=lambda x: x.metadata.get('ensemble_score', 0), 
            reverse=True
        )
        return sorted_docs

# 하이브리드 검색기 생성 (임베딩: 60%, BM25: 40%)
ensemble_retriever = CustomEnsembleRetriever(
    retrievers=[chroma_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)
print("Ensemble Retriever 생성 완료")

# 직무기술서 갯수 및 샘플 출력
print(f"직무기술서 데이터: {len(df)}개")
df.head()

def search_similar_jobs(industry, skill_set, top_k=5):
    """
    직무기술서의 industry와 skill_set을 기반으로 유사 채용공고 검색 (최대 top_k개 반환)
    """
    skill_set_str = " ".join(skill_set) if isinstance(skill_set, list) else str(skill_set)
    query = f"{industry} {skill_set_str}"
    results = ensemble_retriever.invoke(query)
    return results[:top_k]

# 결과 저장용 변수
from datetime import datetime
all_results = []

# 직무기술서 전체에 대해 유사 채용공고 검색
print("\n" + "="*100)
print("전체 직무기술서에 대한 채용공고 매칭 시작...")
print("="*100)

for idx, row in df.iterrows():
    industry = row['industry']
    skill_set = row['공통_skill_set']
    job_title = row['직무']
    
    # 검색 실행
    similar_jobs = search_similar_jobs(industry, skill_set, top_k=5)
    
    # 결과 저장
    result_item = {
        'job_title': job_title,
        'industry': industry,
        'skill_set': skill_set[:5] if isinstance(skill_set, list) else str(skill_set)[:100],
        'matched_jobs': []
    }
    
    for i, job in enumerate(similar_jobs, 1):
        result_item['matched_jobs'].append({
            'rank': i,
            'title': job.metadata['title'],
            'company': job.metadata['company'],
            'job_category': job.metadata.get('job_category', 'N/A'),
            'experience': job.metadata.get('experience', 'N/A'),
            'url': job.metadata['url'],
            'score': job.metadata.get('ensemble_score', 0)
        })
    
    all_results.append(result_item)
    
    # 진행상황 출력
    if (idx + 1) % 10 == 0:
        print(f"진행 중... {idx + 1}/{len(df)} 완료")

print(f"\n매칭 완료! 총 {len(all_results)}개 직무기술서 처리")

# TXT 파일로 저장
def save_results_to_txt(results, filename):
    """검색 결과를 TXT 파일로 저장"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("직무기술서 - 채용공고 매칭 결과\n")
        f.write(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"총 직무기술서 수: {len(results)}개\n")
        f.write(f"총 채용공고 수: {len(job_postings)}개\n")
        f.write("="*100 + "\n\n")
        
        for idx, result in enumerate(results, 1):
            f.write(f"\n{'='*100}\n")
            f.write(f"[{idx}] 직무: {result['job_title']}\n")
            f.write(f"{'='*100}\n")
            f.write(f"Industry: {result['industry']}\n")
            f.write(f"Skill Set (샘플): {result['skill_set']}\n")
            f.write(f"\n{'추천 채용공고 Top 5':-^90}\n\n")
            
            for job in result['matched_jobs']:
                f.write(f"{job['rank']}. {job['title']}\n")
                f.write(f"   회사: {job['company']}\n")
                f.write(f"   업무분야: {job['job_category']}\n")
                f.write(f"   경력: {job['experience']}\n")
                f.write(f"   매칭점수: {job['score']:.2f}\n")
                f.write(f"   URL: {job['url']}\n\n")
    
    print(f"✅ TXT 파일 저장 완료: {filename}")

# PDF 파일로 저장 (reportlab 사용)
def save_results_to_pdf(results, filename):
    """검색 결과를 PDF 파일로 저장 (한글 지원)"""
    try:
        from reportlab.lib.pagesizes import A4, letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        
        # 한글 폰트 등록 시도
        try:
            # Windows 기본 한글 폰트
            pdfmetrics.registerFont(TTFont('Malgun', 'malgun.ttf'))
            font_name = 'Malgun'
        except:
            try:
                pdfmetrics.registerFont(TTFont('Gulim', 'gulim.ttf'))
                font_name = 'Gulim'
            except:
                print("⚠️ 한글 폰트를 찾을 수 없습니다. 영문 폰트로 대체합니다.")
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
        story.append(Paragraph("직무기술서 - 채용공고 매칭 결과", title_style))
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
                    f"   업무분야: {job['job_category']} | 경력: {job['experience']} | 점수: {job['score']:.2f}", 
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
        print(f"✅ PDF 파일 저장 완료: {filename}")
        
    except ImportError:
        print("⚠️ reportlab 라이브러리가 설치되어 있지 않습니다.")
        print("   'pip install reportlab' 명령으로 설치 후 다시 시도하세요.")
    except Exception as e:
        print(f"⚠️ PDF 생성 중 오류 발생: {str(e)}")

# 결과 파일 저장
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = "C:/workspace/Final_project/backend-model/AI_Lab/output"
os.makedirs(output_dir, exist_ok=True)

txt_filename = f"{output_dir}/job_matching_results_{timestamp}.txt"
pdf_filename = f"{output_dir}/job_matching_results_{timestamp}.pdf"

# TXT 저장
save_results_to_txt(all_results, txt_filename)

# PDF 저장
save_results_to_pdf(all_results, pdf_filename)

print("\n" + "="*100)
print("결과 파일 저장 완료!")
print(f"📄 TXT 파일: {txt_filename}")
print(f"📄 PDF 파일: {pdf_filename}")
print("="*100)

# 처음 3개 결과만 콘솔에 출력
print("\n\n[검색 결과 샘플 - 처음 3개]")
for idx, result in enumerate(all_results[:3], 1):
    print(f"\n{'='*100}")
    print(f"[{idx}] 직무: {result['job_title']}")
    print(f"Industry: {result['industry']}")
    print(f"{'='*100}")
    for job in result['matched_jobs']:
        print(f"\n{job['rank']}. {job['title']}")
        print(f"   회사: {job['company']}")
        print(f"   업무분야: {job['job_category']}")
        print(f"   경력: {job['experience']}")
        print(f"   매칭점수: {job['score']:.2f}")
        print(f"   URL: {job['url']}")
