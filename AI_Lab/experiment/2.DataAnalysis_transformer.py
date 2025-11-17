import json
import os
import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 관련 추가
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 함수
def load_job_data(data_dir=None):
    """모든 회사의 채용 공고 데이터 로드"""
    # 데이터 디렉토리 자동 탐색
    if data_dir is None:
        # 현재 스크립트 위치에서 프로젝트 루트 찾기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(project_root, 'data')
    
    print(f"[INFO] 데이터 디렉토리: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        return {}
    
    job_files = glob.glob(os.path.join(data_dir, '*_jobs.json'))
    
    if not job_files:
        print(f"[ERROR] *_jobs.json 파일을 찾을 수 없습니다.")
        print(f"[INFO] 검색 경로: {os.path.join(data_dir, '*_jobs.json')}")
        return {}
    
    all_data = {}
    for file_path in job_files:
        company_name = os.path.basename(file_path).replace('_jobs.json', '')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data[company_name] = data
                print(f"{company_name}: {len(data)}개 공고 로드")
        except Exception as e:
            print(f"{company_name} 로드 실패: {e}")
    
    return all_data



def plot_skill_transformer_analysis(all_data, top_n=15, n_clusters=5):
    """
    Transformer 기반 임베딩 모델을 사용한 스킬 분석:
      - 1) Sentence-BERT로 스킬 임베딩 생성
      - 2) 의미적 유사도 기반 클러스터링
      - 3) 새 스킬도 즉시 처리 가능
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.metrics.pairwise import cosine_similarity
        import umap.umap_ as umap  # 차원 축소
    except ImportError:
        print("[ERROR] 필요한 패키지가 없습니다.")
        print("설치: pip install sentence-transformers umap-learn")
        return
    
    # 1. 고유 스킬 추출
    unique_skills = set()
    skill_job_mapping = defaultdict(list)  # 스킬별 등장 공고
    
    for company, jobs in all_data.items():
        for job_idx, job in enumerate(jobs):
            if 'skill_set_info' in job and 'skill_set' in job['skill_set_info']:
                skills = [str(s).strip() for s in job['skill_set_info']['skill_set']]
                for skill in skills:
                    unique_skills.add(skill)
                    skill_job_mapping[skill].append((company, job_idx))
    
    skills_list = list(unique_skills)
    print(f"[INFO] 총 {len(skills_list)}개 고유 스킬 발견")
    
    if len(skills_list) < 3:
        print("[SKIP] 스킬 데이터 부족")
        return
    
    # 2. Sentence-BERT로 스킬 임베딩 생성
    print("[1/5] 스킬 임베딩 생성 중...")
    # 한국어+영어 지원 모델 (다양한 옵션)
    model_options = [
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # 다국어 지원
        'jhgan/ko-sbert-multitask',  # 한국어 특화
        'sentence-transformers/all-MiniLM-L6-v2',  # 빠른 영어 모델
    ]
    
    model = None
    for model_name in model_options:
        try:
            model = SentenceTransformer(model_name)
            print(f"[INFO] 사용 모델: {model_name}")
            break
        except:
            continue
    
    if model is None:
        print("[ERROR] 모델을 로드할 수 없습니다.")
        return
    
    # 각 스킬을 문장으로 간주하고 임베딩 생성
    skill_embeddings = model.encode(
        skills_list,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )
    
    print(f"[SUCCESS] {len(skill_embeddings)}개 스킬 임베딩 생성 완료")
    print(f"[INFO] 임베딩 차원: {skill_embeddings.shape[1]}")
    
    # 3. 의미적 유사도 기반 클러스터링
    print("[2/5] 의미적 유사도 기반 클러스터링...")
    
    # DBSCAN (밀도 기반) 또는 K-Means
    # DBSCAN은 클러스터 개수를 자동 결정 (새 스킬에 유리)
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
    cluster_labels = clustering.fit_predict(skill_embeddings)
    
    # 또는 K-Means (고정 클러스터 개수)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # cluster_labels = kmeans.fit_predict(skill_embeddings)
    
    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = list(cluster_labels).count(-1)
    print(f"[INFO] 발견된 클러스터: {n_clusters_found}개, 이상치: {n_outliers}개")
    
    # 4. 클러스터별 대표 스킬 추출
    print("[3/5] 클러스터 분석 중...")
    cluster_skills = defaultdict(list)
    for skill, label in zip(skills_list, cluster_labels):
        cluster_skills[label].append(skill)
    
    # 각 클러스터의 중심 스킬 찾기
    cluster_centers = {}
    for cluster_id in range(n_clusters_found):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        cluster_embeddings = skill_embeddings[cluster_indices]
        center = cluster_embeddings.mean(axis=0)
        cluster_centers[cluster_id] = center
    
    # 5. 새 스킬 테스트 (예시)
    print("[4/5] 새 스킬 대응 테스트...")
    new_skills = ["LangChain", "Gemini API", "LLaMA", "Claude API"]  # 최신 기술
    new_embeddings = model.encode(new_skills)
    
    # 기존 스킬과의 유사도 계산
    similarity_matrix = cosine_similarity(new_embeddings, skill_embeddings)
    
    # 각 새 스킬의 가장 유사한 기존 스킬 찾기
    new_skill_matches = {}
    for new_skill, similarities in zip(new_skills, similarity_matrix):
        top_indices = similarities.argsort()[-5:][::-1]  # Top 5
        top_matches = [(skills_list[i], similarities[i]) for i in top_indices]
        new_skill_matches[new_skill] = top_matches
    
    # 6. 결과 저장 및 시각화
    print("[5/5] 결과 저장 중...")
    
    # 리포트 생성
    report = []
    report.append("=" * 80)
    report.append("Transformer 기반 스킬 의미 분석 결과")
    report.append("=" * 80)
    report.append("")
    report.append(f"분석 스킬 수: {len(skills_list)}개")
    report.append(f"발견된 클러스터: {n_clusters_found}개")
    report.append(f"이상치(독립 스킬): {n_outliers}개")
    report.append("")
    
    # 클러스터별 결과
    for cluster_id in sorted(cluster_skills.keys()):
        if cluster_id == -1:  # 이상치
            report.append(f"[이상치 클러스터] ({len(cluster_skills[cluster_id])}개 스킬)")
        else:
            report.append(f"[클러스터 {cluster_id+1}] ({len(cluster_skills[cluster_id])}개 스킬)")
        
        cluster_skill_list = cluster_skills[cluster_id]
        # 클러스터 내 빈도순 정렬
        skill_freq = Counter()
        for skill in cluster_skill_list:
            skill_freq[skill] = len(skill_job_mapping[skill])
        
        for skill, freq in skill_freq.most_common(10):
            report.append(f"   - {skill:30s} (등장: {freq:4d}회)")
        report.append("")
    
    # 새 스킬 매칭 결과
    report.append("=" * 80)
    report.append("새 스킬 의미적 유사도 분석")
    report.append("=" * 80)
    for new_skill, matches in new_skill_matches.items():
        report.append(f"\n[{new_skill}]")
        report.append("가장 유사한 기존 스킬:")
        for existing_skill, similarity in matches:
            report.append(f"   - {existing_skill:30s} (유사도: {similarity:.3f})")
    
    # 저장
    report_text = "\n".join(report)
    os.makedirs('output', exist_ok=True)
    with open('output/9_transformer_skill_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("[SAVE] Transformer 기반 스킬 분석 -> output/9_transformer_skill_analysis.txt")
    
    # 7. 시각화 (UMAP으로 2D 축소)
    try:
        print("[VIZ] 2D 시각화 생성 중...")
        reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
        embeddings_2d = reducer.fit_transform(skill_embeddings)
        
        plt.figure(figsize=(16, 12))
        scatter = plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1],
            c=cluster_labels,
            cmap='tab20',
            alpha=0.6,
            s=50
        )
        
        # 주요 스킬 라벨 표시 (빈도 높은 스킬만)
        skill_freq_counter = Counter()
        for skill in skills_list:
            skill_freq_counter[skill] = len(skill_job_mapping[skill])
        
        top_skills = [skill for skill, _ in skill_freq_counter.most_common(30)]
        for skill in top_skills:
            idx = skills_list.index(skill)
            plt.annotate(
                skill,
                (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                fontsize=8,
                alpha=0.7
            )
        
        plt.colorbar(scatter, label='클러스터')
        plt.title('Transformer 기반 스킬 의미 공간 시각화 (UMAP)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('UMAP 차원 1', fontsize=12)
        plt.ylabel('UMAP 차원 2', fontsize=12)
        plt.tight_layout()
        plt.savefig('output/9_transformer_skill_2d.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[SAVE] 2D 시각화 -> output/9_transformer_skill_2d.png")
        
    except Exception as e:
        print(f"[WARN] 시각화 생성 실패: {e}")
    
    print("\n[SUCCESS] Transformer 기반 분석 완료!")


# 10. LLM 기반 스킬 카테고리 분류 (생성형 AI 활용)
def plot_llm_skill_categorization(all_skills, api_key=None):
    """
    LLM을 활용한 스킬 자동 카테고리 분류
    - GPT/Claude 등을 사용하여 스킬을 의미적으로 분류
    - 새 스킬도 즉시 처리 가능
    """
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        print("[ERROR] openai 패키지가 필요합니다.")
        return
    
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("[SKIP] OpenAI API 키가 없어 LLM 분석을 건너뜁니다.")
        return
    
    client = OpenAI(api_key=api_key)
    
    # 상위 100개 스킬만 분석 (비용 절감)
    skill_counter = Counter(all_skills)
    top_skills = [skill for skill, _ in skill_counter.most_common(100)]
    
    print(f"[INFO] {len(top_skills)}개 스킬을 LLM으로 분류 중...")
    
    # 프롬프트 설계
    prompt_template = """
다음 IT/개발 스킬을 카테고리로 분류해주세요. 각 스킬에 대해 가장 적합한 카테고리 하나만 선택하세요.

카테고리:
1. 프로그래밍 언어 (Python, Java, JavaScript 등)
2. 프론트엔드 프레임워크 (React, Vue, Angular 등)
3. 백엔드 프레임워크 (Spring Boot, Django, Express 등)
4. 데이터베이스 (MySQL, PostgreSQL, MongoDB 등)
5. 클라우드/인프라 (AWS, Docker, Kubernetes 등)
6. AI/ML (TensorFlow, PyTorch, Scikit-learn 등)
7. DevOps/도구 (Jenkins, Git, CI/CD 등)
8. 기타

스킬 리스트:
{skills}

각 스킬에 대해 "스킬명: 카테고리명" 형식으로 JSON 배열로 답변하세요.
예: [{{"skill": "Python", "category": "프로그래밍 언어"}}, ...]
"""
    
    # 배치 처리 (한 번에 20개씩)
    results = []
    batch_size = 20
    
    for i in range(0, len(top_skills), batch_size):
        batch = top_skills[i:i+batch_size]
        skills_text = "\n".join([f"- {s}" for s in batch])
        prompt = prompt_template.format(skills=skills_text)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # 비용 효율적
                messages=[
                    {"role": "system", "content": "당신은 IT 스킬 분류 전문가입니다. 정확하고 일관된 카테고리 분류를 해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 일관성 확보
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content
            # JSON 파싱 (간단한 파싱)
            import json
            import re
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                categories = json.loads(json_match.group())
                results.extend(categories)
            
        except Exception as e:
            print(f"[WARN] 배치 {i//batch_size + 1} 처리 실패: {e}")
            continue
    
    # 카테고리별 집계
    category_skills = defaultdict(list)
    for item in results:
        if isinstance(item, dict):
            category_skills[item.get('category', '기타')].append(item.get('skill', ''))
    
    # 리포트 생성
    report = []
    report.append("=" * 80)
    report.append("LLM 기반 스킬 카테고리 분류 결과")
    report.append("=" * 80)
    report.append("")
    
    for category, skills in sorted(category_skills.items()):
        report.append(f"[{category}] ({len(skills)}개)")
        for skill in sorted(skills):
            report.append(f"   - {skill}")
        report.append("")
    
    report_text = "\n".join(report)
    with open('output/10_llm_skill_categories.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("[SAVE] LLM 카테고리 분류 -> output/10_llm_skill_categories.txt")



# 11. 보상 및 복리후생 벤치마킹
def analyze_compensation_benchmarking(all_data):
    """
    경쟁사 채용 공고에서 노출되는 보상 데이터 추출 및 비교 분석
    - 연봉 정보(범위) 추출
    - 복리후생 키워드 추출
    - 회사별 보상 패키지 비교
    """
    import re
    from datetime import datetime
    
    print("\n" + "=" * 80)
    print("보상 및 복리후생 벤치마킹 분석")
    print("=" * 80)
    
    # 보상 관련 키워드 패턴
    salary_patterns = [
        r'연봉\s*:?\s*([0-9,]+)\s*만?\s*원?',
        r'연봉\s*:?\s*([0-9,]+)\s*~?\s*([0-9,]+)?\s*만?\s*원?',
        r'급여\s*:?\s*([0-9,]+)\s*만?\s*원?',
        r'([0-9,]+)\s*~?\s*([0-9,]+)?\s*만?\s*원?\s*연봉',
        r'([0-9,]+)\s*~?\s*([0-9,]+)?\s*만원?\s*대',
    ]
    
    # 복리후생 키워드
    benefit_keywords = [
        '사이닝 보너스', '스톡옵션', '스톡 옵션', '인센티브', '성과급',
        '4대보험', '건강검진', '휴가', '리프레시', '리커버리데이',
        '주택자금', '식대', '교통비', '통신비', '복리후생', '복지',
        '자율복장', '원격근무', '재택근무', '유연근무', '선택근무',
        '교육비', '도서비', '세미나', '컨퍼런스', '학비지원',
        '건강검진', '산후조리원', '육아휴직', '출산휴가',
        '피트니스', '헬스케어', '마사지', '식당', '카페'
    ]
    
    compensation_data = {}
    
    for company, jobs in all_data.items():
        compensation_data[company] = {
            'salary_info': [],
            'benefits': defaultdict(int),
            'jobs_with_salary': 0,
            'jobs_with_benefits': 0,
            'total_jobs': len(jobs)
        }
        
        for job in jobs:
            description = job.get('description') or ''
            description = str(description).lower()
            
            # 연봉 정보 추출
            for pattern in salary_patterns:
                matches = re.finditer(pattern, description, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        salary_info = match.group(0)
                        compensation_data[company]['salary_info'].append(salary_info)
                        compensation_data[company]['jobs_with_salary'] += 1
                        break
            
            # 복리후생 키워드 추출
            found_benefits = []
            for keyword in benefit_keywords:
                if keyword.lower() in description:
                    compensation_data[company]['benefits'][keyword] += 1
                    found_benefits.append(keyword)
            
            if found_benefits:
                compensation_data[company]['jobs_with_benefits'] += 1
    
    # 리포트 생성
    report = []
    report.append("=" * 80)
    report.append("보상 및 복리후생 벤치마킹 분석 결과")
    report.append("=" * 80)
    report.append("")
    
    # 연봉 정보 분석
    report.append("[1] 연봉 정보 노출 현황")
    report.append("-" * 80)
    for company, data in sorted(compensation_data.items()):
        salary_rate = (data['jobs_with_salary'] / data['total_jobs'] * 100) if data['total_jobs'] > 0 else 0
        report.append(f"\n{company}:")
        report.append(f"  - 총 공고 수: {data['total_jobs']}개")
        report.append(f"  - 연봉 정보 노출 공고: {data['jobs_with_salary']}개 ({salary_rate:.1f}%)")
        if data['salary_info']:
            unique_salaries = list(set(data['salary_info']))[:5]  # 상위 5개만
            report.append(f"  - 발견된 연봉 정보 예시:")
            for sal in unique_salaries:
                report.append(f"    * {sal}")
    
    # 복리후생 분석
    report.append("\n" + "=" * 80)
    report.append("[2] 복리후생 제공 현황")
    report.append("-" * 80)
    for company, data in sorted(compensation_data.items()):
        benefit_rate = (data['jobs_with_benefits'] / data['total_jobs'] * 100) if data['total_jobs'] > 0 else 0
        report.append(f"\n{company}:")
        report.append(f"  - 복리후생 언급 공고: {data['jobs_with_benefits']}개 ({benefit_rate:.1f}%)")
        if data['benefits']:
            top_benefits = sorted(data['benefits'].items(), key=lambda x: x[1], reverse=True)[:5]
            report.append(f"  - 주요 복리후생:")
            for benefit, count in top_benefits:
                report.append(f"    * {benefit}: {count}회 언급")
    
    # 전체 복리후생 키워드 순위
    report.append("\n" + "=" * 80)
    report.append("[3] 전체 복리후생 키워드 순위")
    report.append("-" * 80)
    all_benefits = Counter()
    for company, data in compensation_data.items():
        for benefit, count in data['benefits'].items():
            all_benefits[benefit] += count
    
    report.append("\n가장 많이 언급되는 복리후생:")
    for benefit, count in all_benefits.most_common(20):
        report.append(f"  {count:3d}회 - {benefit}")
    
    # 저장
    report_text = "\n".join(report)
    os.makedirs('output', exist_ok=True)
    with open('output/11_compensation_benchmarking.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("[SAVE] 보상 벤치마킹 분석 -> output/11_compensation_benchmarking.txt")
    
    # 시각화
    try:
        # 회사별 복리후생 언급률
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        companies = list(compensation_data.keys())
        benefit_rates = [
            (compensation_data[c]['jobs_with_benefits'] / compensation_data[c]['total_jobs'] * 100) 
            if compensation_data[c]['total_jobs'] > 0 else 0 
            for c in companies
        ]
        
        axes[0].bar(companies, benefit_rates, color='steelblue', alpha=0.7)
        axes[0].set_ylabel('복리후생 언급률 (%)', fontsize=12)
        axes[0].set_title('회사별 복리후생 언급률 비교', fontsize=14, fontweight='bold')
        axes[0].set_xticklabels(companies, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # 상위 복리후생 키워드
        top_benefits = all_benefits.most_common(15)
        benefit_names = [b[0] for b in top_benefits]
        benefit_counts = [b[1] for b in top_benefits]
        
        axes[1].barh(benefit_names, benefit_counts, color='coral', alpha=0.7)
        axes[1].set_xlabel('언급 횟수', fontsize=12)
        axes[1].set_title('주요 복리후생 키워드 순위', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/11_compensation_benchmarking.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[SAVE] 보상 벤치마킹 시각화 -> output/11_compensation_benchmarking.png")
    except Exception as e:
        print(f"[WARN] 시각화 생성 실패: {e}")
    
    print("[SUCCESS] 보상 및 복리후생 벤치마킹 분석 완료!")


# 헬퍼 함수: meta_data 안전하게 가져오기
def get_meta_data(job):
    """meta_data를 안전하게 딕셔너리로 반환"""
    meta_data = job.get('meta_data', {})
    if isinstance(meta_data, str):
        try:
            import json
            meta_data = json.loads(meta_data)
        except:
            meta_data = {}
    if not isinstance(meta_data, dict):
        meta_data = {}
    return meta_data


# 12. 공격적인 채용 전략 감지
def analyze_aggressive_hiring_strategy(all_data):
    """
    경쟁사의 공격적인 채용 전략 감지
    - 특정 회사가 한 번에 많은 동일 직무 공고 게시
    - 신규 지점/프로젝트 팀 관련 포지션 집중 채용
    - 시간대별 채용 공고 증가 패턴 분석
    """
    print("\n" + "=" * 80)
    print("공격적인 채용 전략 감지 분석")
    print("=" * 80)
    
    # 회사별/직무별 공고 집계
    company_job_counts = defaultdict(int)
    company_category_counts = defaultdict(lambda: defaultdict(int))
    company_title_keywords = defaultdict(Counter)
    
    # 신규 프로젝트/지점 관련 키워드
    expansion_keywords = [
        '신규', '새로운', '신설', '신규 프로젝트', '신규 팀', '신규 조직',
        '신규 사업', '신규 지점', '확장', '증설', '추가 채용',
        '대규모', '다수', '전량', '전면', '집중'
    ]
    
    expansion_jobs = defaultdict(list)
    
    for company, jobs in all_data.items():
        company_job_counts[company] = len(jobs)
        
        for job in jobs:
            title = job.get('title') or ''
            description = job.get('description') or ''
            meta_data = get_meta_data(job)
            category = meta_data.get('job_category', 'Unknown')
            
            company_category_counts[company][category] += 1
            
            # 제목 키워드 추출
            if title:
                title_words = str(title).split()
                for word in title_words:
                    if len(word) > 1:
                        company_title_keywords[company][word] += 1
            
            # 확장 관련 공고 탐지
            full_text = (str(title) + ' ' + str(description)).lower()
            for keyword in expansion_keywords:
                if keyword in full_text:
                    expansion_jobs[company].append({
                        'title': title,
                        'category': category,
                        'keyword': keyword
                    })
                    break
    
    # 리포트 생성
    report = []
    report.append("=" * 80)
    report.append("공격적인 채용 전략 감지 분석 결과")
    report.append("=" * 80)
    report.append("")
    
    # 회사별 전체 공고 수
    report.append("[1] 회사별 채용 공고 수")
    report.append("-" * 80)
    sorted_companies = sorted(company_job_counts.items(), key=lambda x: x[1], reverse=True)
    for company, count in sorted_companies:
        report.append(f"  {company:20s}: {count:4d}개 공고")
    
    avg_jobs = np.mean(list(company_job_counts.values())) if company_job_counts else 0
    report.append(f"\n  평균: {avg_jobs:.1f}개 공고")
    
    # 공격적 채용 감지 (평균의 1.5배 이상)
    report.append("\n" + "=" * 80)
    report.append("[2] 공격적 채용 전략 감지 (평균의 1.5배 이상)")
    report.append("-" * 80)
    threshold = avg_jobs * 1.5
    aggressive_companies = [(c, count) for c, count in sorted_companies if count >= threshold]
    
    if aggressive_companies:
        report.append(f"\n⚠️  감지된 회사 ({threshold:.0f}개 이상 공고):")
        for company, count in aggressive_companies:
            report.append(f"\n  [{company}]")
            report.append(f"    - 총 공고 수: {count}개 (평균의 {count/avg_jobs:.2f}배)")
            
            # 직무별 분포
            if company in company_category_counts:
                report.append(f"    - 직무별 분포:")
                categories = sorted(company_category_counts[company].items(), 
                                  key=lambda x: x[1], reverse=True)
                for cat, cat_count in categories[:5]:
                    report.append(f"      * {cat}: {cat_count}개")
            
            # 확장 관련 공고
            if company in expansion_jobs and expansion_jobs[company]:
                report.append(f"    - 확장/신규 관련 공고: {len(expansion_jobs[company])}개")
                for exp_job in expansion_jobs[company][:3]:
                    report.append(f"      * [{exp_job['keyword']}] {exp_job['title']}")
    else:
        report.append("\n  감지된 공격적 채용 전략 없음")
    
    # 확장 관련 공고 집계
    report.append("\n" + "=" * 80)
    report.append("[3] 확장/신규 관련 채용 공고")
    report.append("-" * 80)
    expansion_by_company = {c: len(jobs) for c, jobs in expansion_jobs.items() if jobs}
    if expansion_by_company:
        sorted_expansion = sorted(expansion_by_company.items(), key=lambda x: x[1], reverse=True)
        for company, count in sorted_expansion:
            report.append(f"\n  {company}: {count}개 공고")
            for exp_job in expansion_jobs[company][:3]:
                report.append(f"    - [{exp_job['category']}] {exp_job['title']}")
    else:
        report.append("\n  확장 관련 공고 없음")
    
    # 직무별 집중도 분석
    report.append("\n" + "=" * 80)
    report.append("[4] 회사별 직무 집중도 분석")
    report.append("-" * 80)
    for company, categories in sorted(company_category_counts.items()):
        total = sum(categories.values())
        if total > 0:
            report.append(f"\n  {company}:")
            sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
            for cat, count in sorted_cats:
                if cat is None:
                    cat = 'Unknown'
                cat_str = str(cat) if cat else 'Unknown'
                percentage = (count / total) * 100
                report.append(f"    - {cat_str:20s}: {count:3d}개 ({percentage:5.1f}%)")
                if percentage > 50:  # 절반 이상이면 집중 채용
                    report.append(f"      ⚠️  해당 직무에 집중 채용 중!")
    
    # 저장
    report_text = "\n".join(report)
    os.makedirs('output', exist_ok=True)
    with open('output/12_aggressive_hiring_strategy.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("[SAVE] 공격적 채용 전략 분석 -> output/12_aggressive_hiring_strategy.txt")
    
    # 시각화
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 회사별 공고 수
        companies = [c[0] for c in sorted_companies]
        counts = [c[1] for c in sorted_companies]
        colors = ['red' if c >= threshold else 'steelblue' for c in counts]
        
        axes[0, 0].barh(companies, counts, color=colors, alpha=0.7)
        axes[0, 0].axvline(x=threshold, color='red', linestyle='--', 
                          label=f'경고선 ({threshold:.0f}개)')
        axes[0, 0].set_xlabel('채용 공고 수', fontsize=11)
        axes[0, 0].set_title('회사별 채용 공고 수 (공격적 채용 감지)', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. 직무별 집중도 (상위 3개 회사)
        top3_companies = companies[:3]
        for idx, company in enumerate(top3_companies):
            if company in company_category_counts:
                categories = company_category_counts[company]
                cat_names = list(categories.keys())
                cat_counts = list(categories.values())
                
                axes[0, 1].bar(range(len(cat_names)), cat_counts, 
                              alpha=0.7, label=company)
        
        if top3_companies:
            all_cats = set()
            for company in top3_companies:
                if company in company_category_counts:
                    all_cats.update(company_category_counts[company].keys())
            
            if all_cats:
                cat_list = sorted(all_cats)
                axes[0, 1].set_xticks(range(len(cat_list)))
                axes[0, 1].set_xticklabels(cat_list, rotation=45, ha='right')
                axes[0, 1].set_ylabel('공고 수', fontsize=11)
                axes[0, 1].set_title('상위 3개 회사 직무별 분포', 
                                    fontsize=12, fontweight='bold')
                axes[0, 1].legend()
                axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. 확장 관련 공고
        if expansion_by_company:
            exp_companies = list(expansion_by_company.keys())
            exp_counts = list(expansion_by_company.values())
            axes[1, 0].bar(exp_companies, exp_counts, color='coral', alpha=0.7)
            axes[1, 0].set_ylabel('확장 관련 공고 수', fontsize=11)
            axes[1, 0].set_title('확장/신규 관련 채용 공고', 
                                fontsize=12, fontweight='bold')
            axes[1, 0].set_xticklabels(exp_companies, rotation=45, ha='right')
            axes[1, 0].grid(axis='y', alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '확장 관련 공고 없음', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('확장/신규 관련 채용 공고', 
                                fontsize=12, fontweight='bold')
        
        # 4. 공격적 채용 감지 요약
        normal_count = len([c for c in counts if c < threshold])
        aggressive_count = len([c for c in counts if c >= threshold])
        axes[1, 1].pie([normal_count, aggressive_count], 
                      labels=['일반', '공격적 채용 감지'],
                      autopct='%1.1f%%',
                      colors=['steelblue', 'red'],
                      startangle=90)
        axes[1, 1].set_title('공격적 채용 전략 감지 요약', 
                            fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('output/12_aggressive_hiring_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[SAVE] 공격적 채용 전략 시각화 -> output/12_aggressive_hiring_strategy.png")
    except Exception as e:
        print(f"[WARN] 시각화 생성 실패: {e}")
    
    print("[SUCCESS] 공격적인 채용 전략 감지 분석 완료!")


# 13. 채용 메시지 및 포지셔닝 연구
def analyze_hiring_message_positioning(all_data):
    """
    경쟁사들의 채용 메시지 및 포지셔닝 분석
    - "우리는 이런 사람을 찾습니다" 섹션 키워드
    - "우리 회사의 장점" 섹션 키워드
    - Employer Brand 메시지 추출 및 비교
    """
    import re
    
    print("\n" + "=" * 80)
    print("채용 메시지 및 포지셔닝 연구 분석")
    print("=" * 80)
    
    # 포지셔닝 관련 키워드 패턴
    positioning_sections = {
        '채용하고 싶은 사람': [
            r'채용하고\s*싶은\s*사람',
            r'우리는\s*이런\s*사람을\s*찾습니다',
            r'우리가\s*찾는\s*사람',
            r'모시고\s*싶은\s*인재'
        ],
        '회사 장점': [
            r'우리\s*회사의\s*장점',
            r'우리의\s*특징',
            r'Why\s*[Ww]e?',
            r'About\s*[Uu]s',
            r'우리는\s*이렇습니다'
        ],
        '문화/가치': [
            r'문화',
            r'가치',
            r'비전',
            r'미션',
            r'핵심\s*가치'
        ]
    }
    
    # Employer Brand 관련 키워드
    employer_brand_keywords = [
        '자율성', '자율', '자유', '수평적', '수평', '평등',
        '글로벌', '글로벌 기회', '해외', '다양성', '포용',
        '성장', '성장 기회', '배움', '학습', '발전',
        '혁신', '창의', '도전', '변화',
        '워라밸', '균형', '휴식', '여가',
        '협업', '소통', '커뮤니케이션', '팀워크',
        '전문성', '전문', '깊이', '깊은',
        '영향력', '임팩트', '사회', '기여',
        '주도권', '리더십', '독립', '자립'
    ]
    
    company_messages = defaultdict(lambda: {
        'positioning_keywords': Counter(),
        'employer_brand_keywords': Counter(),
        'hiring_preferences': [],
        'job_count': 0
    })
    
    for company, jobs in all_data.items():
        company_messages[company]['job_count'] = len(jobs)
        
        for job in jobs:
            description = job.get('description') or ''
            description = str(description)
            
            # positioning 섹션 찾기
            for section_name, patterns in positioning_sections.items():
                for pattern in patterns:
                    if re.search(pattern, description, re.IGNORECASE):
                        # 해당 섹션 이후 텍스트 추출 (간단한 추출)
                        match = re.search(pattern, description, re.IGNORECASE)
                        if match:
                            start_pos = match.end()
                            section_text = description[start_pos:start_pos+500]  # 다음 500자
                            # 키워드 추출
                            words = re.findall(r'\b\w+\b', section_text)
                            for word in words:
                                if len(word) > 1:
                                    company_messages[company]['positioning_keywords'][word] += 1
            
            # hiring_preferences 추출 (meta_data에 있는 경우)
            meta_data = get_meta_data(job)
            if 'hiring_preferences' in meta_data:
                prefs = meta_data['hiring_preferences']
                if isinstance(prefs, list):
                    company_messages[company]['hiring_preferences'].extend(prefs)
            
            # Employer Brand 키워드 추출
            description_lower = description.lower()
            for keyword in employer_brand_keywords:
                if keyword in description_lower:
                    company_messages[company]['employer_brand_keywords'][keyword] += 1
    
    # 리포트 생성
    report = []
    report.append("=" * 80)
    report.append("채용 메시지 및 포지셔닝 연구 분석 결과")
    report.append("=" * 80)
    report.append("")
    
    # 회사별 Employer Brand 키워드
    report.append("[1] 회사별 Employer Brand 강조 키워드")
    report.append("-" * 80)
    for company, data in sorted(company_messages.items()):
        if data['employer_brand_keywords']:
            report.append(f"\n{company}:")
            top_keywords = data['employer_brand_keywords'].most_common(10)
            report.append(f"  - 총 공고 수: {data['job_count']}개")
            report.append(f"  - 주요 강조 키워드:")
            for keyword, count in top_keywords:
                rate = (count / data['job_count']) * 100 if data['job_count'] > 0 else 0
                report.append(f"    * {keyword:20s}: {count:3d}회 ({rate:5.1f}%)")
    
    # 전체 Employer Brand 키워드 순위
    report.append("\n" + "=" * 80)
    report.append("[2] 전체 Employer Brand 키워드 순위")
    report.append("-" * 80)
    all_brand_keywords = Counter()
    for company, data in company_messages.items():
        for keyword, count in data['employer_brand_keywords'].items():
            all_brand_keywords[keyword] += count
    
    report.append("\n가장 많이 언급되는 Employer Brand 키워드:")
    for keyword, count in all_brand_keywords.most_common(20):
        total_jobs = sum(m['job_count'] for m in company_messages.values())
        rate = (count / total_jobs) * 100 if total_jobs > 0 else 0
        report.append(f"  {count:3d}회 ({rate:5.1f}%) - {keyword}")
    
    # 채용하고 싶은 사람 메시지
    report.append("\n" + "=" * 80)
    report.append("[3] '채용하고 싶은 사람' 메시지 분석")
    report.append("-" * 80)
    for company, data in sorted(company_messages.items()):
        if data['hiring_preferences']:
            report.append(f"\n{company}:")
            # 고유 메시지만 추출
            unique_prefs = list(set(data['hiring_preferences']))[:10]
            for pref in unique_prefs:
                if len(pref.strip()) > 10:  # 짧은 메시지 제외
                    report.append(f"  - {pref}")
    
    # 회사별 차별화 포인트
    report.append("\n" + "=" * 80)
    report.append("[4] 회사별 차별화 포인트 (상위 키워드)")
    report.append("-" * 80)
    for company, data in sorted(company_messages.items()):
        if data['employer_brand_keywords']:
            report.append(f"\n{company}:")
            top3 = data['employer_brand_keywords'].most_common(3)
            keywords_str = ', '.join([k for k, _ in top3])
            report.append(f"  핵심 메시지: {keywords_str}")
    
    # 저장
    report_text = "\n".join(report)
    os.makedirs('output', exist_ok=True)
    with open('output/13_hiring_message_positioning.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("[SAVE] 채용 메시지 포지셔닝 분석 -> output/13_hiring_message_positioning.txt")
    
    # 시각화
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 회사별 Employer Brand 키워드 빈도 (상위 키워드)
        top_10_brand = all_brand_keywords.most_common(10)
        if top_10_brand:
            brand_names = [b[0] for b in top_10_brand]
            brand_counts = [b[1] for b in top_10_brand]
            axes[0, 0].barh(brand_names, brand_counts, color='steelblue', alpha=0.7)
            axes[0, 0].set_xlabel('언급 횟수', fontsize=11)
            axes[0, 0].set_title('상위 Employer Brand 키워드', 
                                fontsize=12, fontweight='bold')
            axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. 회사별 주요 키워드 비교 (워드 클라우드 스타일)
        company_brand_data = {}
        for company, data in company_messages.items():
            if data['employer_brand_keywords']:
                top = data['employer_brand_keywords'].most_common(1)[0]
                company_brand_data[company] = top[1]
        
        if company_brand_data:
            companies = list(company_brand_data.keys())
            counts = list(company_brand_data.values())
            axes[0, 1].bar(companies, counts, color='coral', alpha=0.7)
            axes[0, 1].set_ylabel('주요 키워드 언급 횟수', fontsize=11)
            axes[0, 1].set_title('회사별 Employer Brand 키워드 강도', 
                                fontsize=12, fontweight='bold')
            axes[0, 1].set_xticklabels(companies, rotation=45, ha='right')
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. 카테고리별 키워드 분류
        keyword_categories = {
            '자율/자유': ['자율성', '자율', '자유', '수평적', '수평'],
            '성장/발전': ['성장', '성장 기회', '배움', '학습', '발전'],
            '혁신/창의': ['혁신', '창의', '도전', '변화'],
            '워라밸': ['워라밸', '균형', '휴식', '여가'],
            '협업/소통': ['협업', '소통', '커뮤니케이션', '팀워크']
        }
        
        category_counts = {}
        for category, keywords in keyword_categories.items():
            total = sum(all_brand_keywords.get(kw, 0) for kw in keywords)
            category_counts[category] = total
        
        if category_counts:
            cats = list(category_counts.keys())
            counts = list(category_counts.values())
            axes[1, 0].bar(cats, counts, color='mediumseagreen', alpha=0.7)
            axes[1, 0].set_ylabel('언급 횟수', fontsize=11)
            axes[1, 0].set_title('카테고리별 Employer Brand 키워드', 
                                fontsize=12, fontweight='bold')
            axes[1, 0].set_xticklabels(cats, rotation=45, ha='right')
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. 회사별 키워드 다양성
        diversity_scores = {}
        for company, data in company_messages.items():
            if data['employer_brand_keywords']:
                unique_count = len(data['employer_brand_keywords'])
                diversity_scores[company] = unique_count
        
        if diversity_scores:
            companies = list(diversity_scores.keys())
            scores = list(diversity_scores.values())
            axes[1, 1].bar(companies, scores, color='gold', alpha=0.7)
            axes[1, 1].set_ylabel('고유 키워드 수', fontsize=11)
            axes[1, 1].set_title('회사별 Employer Brand 키워드 다양성', 
                                fontsize=12, fontweight='bold')
            axes[1, 1].set_xticklabels(companies, rotation=45, ha='right')
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/13_hiring_message_positioning.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[SAVE] 채용 메시지 포지셔닝 시각화 -> output/13_hiring_message_positioning.png")
    except Exception as e:
        print(f"[WARN] 시각화 생성 실패: {e}")
    
    print("[SUCCESS] 채용 메시지 및 포지셔닝 연구 분석 완료!")


# 14. 시간적 트렌드 분석
def analyze_temporal_trends(all_data):
    """
    날짜 정보를 활용한 시간적 트렌드 분석
    - 시계열 채용 공고 수 분석 (일별/월별)
    - 회사별 채용 활동 패턴
    - 직무별/스킬별 트렌드 (시간에 따른 인기 변화)
    - 채용 기간 분석 (게시일~마감일)
    - 신규 스킬 등장 트렌드
    """
    from datetime import datetime, timedelta
    import re
    
    print("\n" + "=" * 80)
    print("시간적 트렌드 분석")
    print("=" * 80)
    
    # 날짜 파싱 함수
    def parse_date(date_str):
        """날짜 문자열을 datetime 객체로 변환"""
        if not date_str or date_str == 'null':
            return None
        try:
            # YYYY-MM-DD 형식
            return datetime.strptime(str(date_str), '%Y-%m-%d')
        except:
            try:
                # 다른 형식 시도
                return datetime.strptime(str(date_str), '%Y/%m/%d')
            except:
                return None
    
    # 데이터 수집
    jobs_by_date = defaultdict(int)  # 날짜별 공고 수
    jobs_by_month = defaultdict(int)  # 월별 공고 수
    company_timeline = defaultdict(lambda: defaultdict(int))  # 회사별 날짜별 공고 수
    category_timeline = defaultdict(lambda: defaultdict(int))  # 직무별 날짜별 공고 수
    skill_timeline = defaultdict(lambda: defaultdict(int))  # 스킬별 날짜별 공고 수
    hiring_durations = []  # 채용 기간 (일수)
    skill_first_seen = {}  # 스킬 최초 등장 시점
    skill_count_by_period = defaultdict(lambda: defaultdict(int))  # 기간별 스킬 등장 횟수
    
    # 날짜 범위 확인
    all_dates = []
    
    for company, jobs in all_data.items():
        for job in jobs:
            # posted_date를 우선 사용, NULL이면 crawl_date 사용
            posted_date_str = job.get('posted_date')
            if posted_date_str is None or posted_date_str == 'null' or str(posted_date_str).lower() == 'null':
                date_str = job.get('crawl_date')
            else:
                date_str = posted_date_str
            
            if not date_str or date_str == 'null' or str(date_str).lower() == 'null':
                continue
            
            date_obj = parse_date(date_str)
            if not date_obj:
                continue
            
            all_dates.append(date_obj)
            
            # 날짜별 집계
            date_key = date_obj.strftime('%Y-%m-%d')
            month_key = date_obj.strftime('%Y-%m')
            
            jobs_by_date[date_key] += 1
            jobs_by_month[month_key] += 1
            company_timeline[company][month_key] += 1
            
            # 직무별 집계
            meta_data = get_meta_data(job)
            category = meta_data.get('job_category', 'Unknown')
            if category is None:
                category = 'Unknown'
            category = str(category) if category else 'Unknown'
            category_timeline[category][month_key] += 1
            
            # 스킬별 집계
            if 'skill_set_info' in job and 'skill_set' in job['skill_set_info']:
                skills = job['skill_set_info']['skill_set']
                for skill in skills:
                    skill_name = str(skill).strip()
                    skill_timeline[skill_name][month_key] += 1
                    skill_count_by_period[month_key][skill_name] += 1
                    
                    # 스킬 최초 등장 시점 기록
                    if skill_name not in skill_first_seen:
                        skill_first_seen[skill_name] = date_obj
                    elif date_obj < skill_first_seen[skill_name]:
                        skill_first_seen[skill_name] = date_obj
            
            # 채용 기간 계산
            expired_str = job.get('expired_date')
            if expired_str:
                expired_obj = parse_date(expired_str)
                if expired_obj and expired_obj > date_obj:
                    duration = (expired_obj - date_obj).days
                    hiring_durations.append(duration)
    
    if not all_dates:
        print("[ERROR] 날짜 정보가 없어 분석을 수행할 수 없습니다.")
        return
    
    min_date = min(all_dates)
    max_date = max(all_dates)
    
    print(f"[INFO] 분석 기간: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
    print(f"[INFO] 총 {len(all_dates)}개 공고 분석")
    
    # 리포트 생성
    report = []
    report.append("=" * 80)
    report.append("시간적 트렌드 분석 결과")
    report.append("=" * 80)
    report.append("")
    report.append(f"분석 기간: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
    report.append(f"총 공고 수: {len(all_dates)}개")
    report.append("")
    
    # 1. 월별 채용 공고 수 추이
    report.append("[1] 월별 채용 공고 수 추이")
    report.append("-" * 80)
    sorted_months = sorted(jobs_by_month.items())
    for month, count in sorted_months:
        report.append(f"  {month}: {count:3d}개")
    
    if len(sorted_months) > 1:
        first_month_count = sorted_months[0][1]
        last_month_count = sorted_months[-1][1]
        if first_month_count > 0:
            growth_rate = ((last_month_count - first_month_count) / first_month_count) * 100
            report.append(f"\n  성장률: {growth_rate:+.1f}% (첫 월 대비 마지막 월)")
    
    # 2. 회사별 채용 활동 패턴
    report.append("\n" + "=" * 80)
    report.append("[2] 회사별 채용 활동 패턴 (월별)")
    report.append("-" * 80)
    for company in sorted(company_timeline.keys()):
        timeline = company_timeline[company]
        sorted_timeline = sorted(timeline.items())
        total = sum(timeline.values())
        report.append(f"\n  {company} (총 {total}개):")
        for month, count in sorted_timeline[-6:]:  # 최근 6개월
            report.append(f"    {month}: {count}개")
    
    # 3. 직무별 트렌드
    report.append("\n" + "=" * 80)
    report.append("[3] 직무별 트렌드 (월별)")
    report.append("-" * 80)
    # None 값을 필터링하고 정렬
    valid_categories = [cat for cat in category_timeline.keys() if cat is not None]
    for category in sorted(valid_categories, key=lambda x: str(x) if x else 'Unknown'):
        timeline = category_timeline[category]
        sorted_timeline = sorted(timeline.items())
        total = sum(timeline.values())
        report.append(f"\n  {category} (총 {total}개):")
        for month, count in sorted_timeline[-6:]:  # 최근 6개월
            report.append(f"    {month}: {count}개")
    
    # 4. 성장하는 스킬 (최근 3개월 대비 이전 3개월)
    report.append("\n" + "=" * 80)
    report.append("[4] 성장하는 스킬 트렌드")
    report.append("-" * 80)
    
    if len(sorted_months) >= 6:
        # 최근 3개월과 그 이전 3개월 비교
        recent_months = sorted_months[-3:]
        previous_months = sorted_months[-6:-3]
        
        recent_skills = Counter()
        previous_skills = Counter()
        
        for month, _ in recent_months:
            for skill, count in skill_count_by_period[month].items():
                recent_skills[skill] += count
        
        for month, _ in previous_months:
            for skill, count in skill_count_by_period[month].items():
                previous_skills[skill] += count
        
        # 성장률 계산 (최소 5회 이상 언급된 스킬만)
        growth_rates = []
        for skill in set(list(recent_skills.keys()) + list(previous_skills.keys())):
            recent_count = recent_skills.get(skill, 0)
            previous_count = previous_skills.get(skill, 0)
            if recent_count >= 5:  # 최소 기준
                if previous_count > 0:
                    growth = ((recent_count - previous_count) / previous_count) * 100
                else:
                    growth = 100 if recent_count > 0 else 0  # 신규 등장
                growth_rates.append((skill, recent_count, previous_count, growth))
        
        growth_rates.sort(key=lambda x: x[3], reverse=True)
        
        report.append("\n  최근 3개월 대비 성장률 (상위 20개):")
        for skill, recent, prev, growth in growth_rates[:20]:
            report.append(f"    {skill:30s}: {recent:3d}회 (이전: {prev:3d}회) | 성장률: {growth:+6.1f}%")
    
    # 5. 신규 스킬 등장 분석
    report.append("\n" + "=" * 80)
    report.append("[5] 신규 스킬 등장 분석")
    report.append("-" * 80)
    
    # 각 월별로 처음 등장한 스킬 찾기
    new_skills_by_month = defaultdict(list)
    seen_skills = set()
    
    for month in sorted(skill_count_by_period.keys()):
        skills_in_month = set(skill_count_by_period[month].keys())
        new_skills = skills_in_month - seen_skills
        if new_skills:
            # 등장 횟수로 정렬
            new_skills_sorted = sorted(
                [(s, skill_count_by_period[month][s]) for s in new_skills],
                key=lambda x: x[1],
                reverse=True
            )
            new_skills_by_month[month] = [s[0] for s in new_skills_sorted[:10]]  # 상위 10개
            seen_skills.update(new_skills)
    
    for month in sorted(new_skills_by_month.keys())[-6:]:  # 최근 6개월
        new_skills = new_skills_by_month[month]
        if new_skills:
            report.append(f"\n  {month}에 처음 등장한 스킬 (상위 {len(new_skills)}개):")
            for skill in new_skills:
                count = skill_count_by_period[month][skill]
                report.append(f"    - {skill:30s}: {count}회")
    
    # 6. 채용 기간 분석
    report.append("\n" + "=" * 80)
    report.append("[6] 채용 기간 분석")
    report.append("-" * 80)
    if hiring_durations:
        avg_duration = np.mean(hiring_durations)
        median_duration = np.median(hiring_durations)
        min_duration = min(hiring_durations)
        max_duration = max(hiring_durations)
        
        report.append(f"\n  평균 채용 기간: {avg_duration:.1f}일")
        report.append(f"  중앙값 채용 기간: {median_duration:.1f}일")
        report.append(f"  최소 채용 기간: {min_duration}일")
        report.append(f"  최대 채용 기간: {max_duration}일")
        
        # 기간별 분포
        duration_bins = [0, 7, 14, 30, 60, 90, 365]
        duration_labels = ['~7일', '8~14일', '15~30일', '31~60일', '61~90일', '91일~']
        duration_dist = [0] * len(duration_labels)
        
        for duration in hiring_durations:
            for i, bin_max in enumerate(duration_bins[1:], 0):
                if duration <= bin_max:
                    duration_dist[i] += 1
                    break
            else:
                duration_dist[-1] += 1
        
        report.append(f"\n  채용 기간 분포:")
        for label, count in zip(duration_labels, duration_dist):
            percentage = (count / len(hiring_durations)) * 100
            report.append(f"    {label:10s}: {count:3d}개 ({percentage:5.1f}%)")
    else:
        report.append("\n  채용 기간 정보 없음")
    
    # 저장
    report_text = "\n".join(report)
    os.makedirs('output', exist_ok=True)
    with open('output/14_temporal_trends.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("[SAVE] 시간적 트렌드 분석 -> output/14_temporal_trends.txt")
    
    # 시각화
    try:
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 월별 채용 공고 수 추이
        ax1 = fig.add_subplot(gs[0, :])
        sorted_months = sorted(jobs_by_month.items())
        months = [m[0] for m in sorted_months]
        counts = [m[1] for m in sorted_months]
        ax1.plot(months, counts, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax1.fill_between(months, counts, alpha=0.3, color='steelblue')
        ax1.set_xlabel('월', fontsize=12)
        ax1.set_ylabel('채용 공고 수', fontsize=12)
        ax1.set_title('월별 채용 공고 수 추이', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 회사별 채용 활동 (선 그래프)
        ax2 = fig.add_subplot(gs[1, 0])
        top_companies = sorted(company_timeline.items(), 
                              key=lambda x: sum(x[1].values()), reverse=True)[:5]
        for company, timeline in top_companies:
            sorted_tl = sorted(timeline.items())
            months_plot = [m[0] for m in sorted_tl]
            counts_plot = [m[1] for m in sorted_tl]
            ax2.plot(months_plot, counts_plot, marker='o', label=company, linewidth=1.5, markersize=4)
        ax2.set_xlabel('월', fontsize=11)
        ax2.set_ylabel('공고 수', fontsize=11)
        ax2.set_title('주요 회사별 채용 활동', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 3. 직무별 트렌드 (스택 그래프)
        ax3 = fig.add_subplot(gs[1, 1])
        # None 값을 필터링하고 정렬
        valid_category_items = [(cat, tl) for cat, tl in category_timeline.items() if cat is not None]
        top_categories = sorted(valid_category_items,
                               key=lambda x: sum(x[1].values()), reverse=True)[:5]
        if top_categories:
            all_months = sorted(set([m for cat, tl in top_categories for m in tl.keys()]))
            bottom = [0] * len(all_months)
            for category, timeline in top_categories:
                counts_plot = [timeline.get(m, 0) for m in all_months]
                ax3.fill_between(range(len(all_months)), bottom, 
                                [b + c for b, c in zip(bottom, counts_plot)],
                                label=category, alpha=0.7)
                bottom = [b + c for b, c in zip(bottom, counts_plot)]
            ax3.set_xticks(range(len(all_months)))
            ax3.set_xticklabels(all_months, rotation=45, ha='right', fontsize=8)
            ax3.set_ylabel('공고 수', fontsize=11)
            ax3.set_title('직무별 트렌드', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 성장하는 스킬 (상위 10개)
        ax4 = fig.add_subplot(gs[1, 2])
        if len(sorted_months) >= 6:
            # growth_rates 재계산
            recent_months_viz = sorted_months[-3:]
            previous_months_viz = sorted_months[-6:-3]
            
            recent_skills_viz = Counter()
            previous_skills_viz = Counter()
            
            for month, _ in recent_months_viz:
                for skill, count in skill_count_by_period[month].items():
                    recent_skills_viz[skill] += count
            
            for month, _ in previous_months_viz:
                for skill, count in skill_count_by_period[month].items():
                    previous_skills_viz[skill] += count
            
            growth_rates_viz = []
            for skill in set(list(recent_skills_viz.keys()) + list(previous_skills_viz.keys())):
                recent_count = recent_skills_viz.get(skill, 0)
                previous_count = previous_skills_viz.get(skill, 0)
                if recent_count >= 5:
                    if previous_count > 0:
                        growth = ((recent_count - previous_count) / previous_count) * 100
                    else:
                        growth = 100 if recent_count > 0 else 0
                    growth_rates_viz.append((skill, recent_count, previous_count, growth))
            
            growth_rates_viz.sort(key=lambda x: x[3], reverse=True)
            top_growth = growth_rates_viz[:10]
            if top_growth:
                skills_plot = [s[0][:20] for s in top_growth]  # 이름 자르기
                growth_plot = [s[3] for s in top_growth]
                colors = ['green' if g > 0 else 'red' for g in growth_plot]
                ax4.barh(skills_plot, growth_plot, color=colors, alpha=0.7)
                ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax4.set_xlabel('성장률 (%)', fontsize=11)
                ax4.set_title('성장하는 스킬 Top 10', fontsize=12, fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. 신규 스킬 등장 (월별)
        ax5 = fig.add_subplot(gs[2, 0])
        if new_skills_by_month:
            months_new = sorted(new_skills_by_month.keys())[-6:]
            counts_new = [len(new_skills_by_month[m]) for m in months_new]
            ax5.bar(months_new, counts_new, color='coral', alpha=0.7)
            ax5.set_xlabel('월', fontsize=11)
            ax5.set_ylabel('신규 스킬 수', fontsize=11)
            ax5.set_title('월별 신규 스킬 등장 수', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.tick_params(axis='x', rotation=45, labelsize=8)
        
        # 6. 채용 기간 분포
        ax6 = fig.add_subplot(gs[2, 1])
        if hiring_durations:
            ax6.hist(hiring_durations, bins=30, color='mediumseagreen', alpha=0.7, edgecolor='black')
            ax6.axvline(np.mean(hiring_durations), color='red', linestyle='--', 
                       linewidth=2, label=f'평균: {np.mean(hiring_durations):.1f}일')
            ax6.axvline(np.median(hiring_durations), color='blue', linestyle='--',
                       linewidth=2, label=f'중앙값: {np.median(hiring_durations):.1f}일')
            ax6.set_xlabel('채용 기간 (일)', fontsize=11)
            ax6.set_ylabel('공고 수', fontsize=11)
            ax6.set_title('채용 기간 분포', fontsize=12, fontweight='bold')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. 인기 스킬 시간별 변화 (히트맵)
        ax7 = fig.add_subplot(gs[2, 2])
        if len(sorted_months) >= 3:
            # 상위 10개 스킬의 월별 등장 횟수
            total_skill_counts = Counter()
            for month_skills in skill_count_by_period.values():
                for skill, count in month_skills.items():
                    total_skill_counts[skill] += count
            
            top_skills = [s[0] for s in total_skill_counts.most_common(10)]
            months_for_heatmap = sorted(skill_count_by_period.keys())[-6:]
            
            heatmap_data = []
            for skill in top_skills:
                row = []
                for month in months_for_heatmap:
                    row.append(skill_count_by_period[month].get(skill, 0))
                heatmap_data.append(row)
            
            if heatmap_data:
                im = ax7.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
                ax7.set_xticks(range(len(months_for_heatmap)))
                ax7.set_xticklabels(months_for_heatmap, rotation=45, ha='right', fontsize=8)
                ax7.set_yticks(range(len(top_skills)))
                ax7.set_yticklabels([s[:15] for s in top_skills], fontsize=8)
                ax7.set_title('인기 스킬 월별 등장 횟수', fontsize=12, fontweight='bold')
                plt.colorbar(im, ax=ax7, label='등장 횟수')
        
        plt.savefig('output/14_temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[SAVE] 시간적 트렌드 시각화 -> output/14_temporal_trends.png")
    except Exception as e:
        print(f"[WARN] 시각화 생성 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("[SUCCESS] 시간적 트렌드 분석 완료!")


# 15. 분기별/연도별 분석 및 주요 경쟁사 비교
def analyze_quarterly_yearly_with_competitors(all_data):
    """
    분기별/연도별 분석 및 주요 경쟁사 비교
    - crawl_date, posted_date, expired_date를 활용한 분기별/연도별 집계
    - 주요 경쟁사(한화, 카카오, 네이버, 라인, 토스) vs 전체 비교
    """
    from datetime import datetime
    import calendar
    
    print("\n" + "=" * 80)
    print("분기별/연도별 분석 및 주요 경쟁사 비교")
    print("=" * 80)
    
    # 주요 경쟁사 정의
    KEY_COMPETITORS = ['hanwha', 'kakao', 'naver', 'line', 'toss']
    KEY_COMPETITOR_NAMES = {
        'hanwha': '한화',
        'kakao': '카카오',
        'naver': '네이버',
        'line': '라인',
        'toss': '토스'
    }
    
    # 날짜 파싱 함수
    def parse_date(date_str):
        """날짜 문자열을 datetime 객체로 변환"""
        if not date_str or date_str == 'null' or str(date_str).lower() == 'null':
            return None
        try:
            return datetime.strptime(str(date_str), '%Y-%m-%d')
        except:
            try:
                return datetime.strptime(str(date_str), '%Y/%m/%d')
            except:
                return None
    
    def get_quarter(date_obj):
        """날짜에서 분기 반환 (1Q, 2Q, 3Q, 4Q)"""
        quarter = (date_obj.month - 1) // 3 + 1
        return f"{date_obj.year}Q{quarter}"
    
    def get_year(date_obj):
        """날짜에서 연도 반환"""
        return str(date_obj.year)
    
    # 데이터 수집 구조
    # 분기별 집계
    jobs_by_quarter = defaultdict(int)  # 전체 분기별
    competitor_jobs_by_quarter = defaultdict(int)  # 주요 경쟁사 분기별
    company_quarter = defaultdict(lambda: defaultdict(int))  # 회사별 분기별
    
    # 연도별 집계
    jobs_by_year = defaultdict(int)  # 전체 연도별
    competitor_jobs_by_year = defaultdict(int)  # 주요 경쟁사 연도별
    company_year = defaultdict(lambda: defaultdict(int))  # 회사별 연도별
    
    # 날짜별 집계 (posted_date, crawl_date, expired_date 모두)
    posted_by_quarter = defaultdict(int)  # posted_date 기준
    crawled_by_quarter = defaultdict(int)  # crawl_date 기준
    expired_by_quarter = defaultdict(int)  # expired_date 기준
    
    posted_by_year = defaultdict(int)
    crawled_by_year = defaultdict(int)
    expired_by_year = defaultdict(int)
    
    # 회사별 분기/연도별 상세
    company_posted_quarter = defaultdict(lambda: defaultdict(int))
    company_crawled_quarter = defaultdict(lambda: defaultdict(int))
    company_expired_quarter = defaultdict(lambda: defaultdict(int))
    
    # 채용 기간 분석 (분기별/연도별)
    hiring_duration_by_quarter = defaultdict(list)
    hiring_duration_by_year = defaultdict(list)
    
    all_dates_posted = []
    all_dates_crawled = []
    all_dates_expired = []
    
    # 회사명 정규화 (파일명 기반 -> 한글명)
    def normalize_company_name(file_name):
        """파일명을 한글 회사명으로 변환"""
        name_lower = file_name.lower()
        for key, korean_name in KEY_COMPETITOR_NAMES.items():
            if key in name_lower:
                return korean_name
        return file_name
    
    # 데이터 수집
    for company_file, jobs in all_data.items():
        company_normalized = normalize_company_name(company_file)
        is_key_competitor = any(key in company_file.lower() for key in KEY_COMPETITORS)
        
        for job in jobs:
            # posted_date 분석
            posted_str = job.get('posted_date')
            if posted_str:
                posted_obj = parse_date(posted_str)
                if posted_obj:
                    all_dates_posted.append(posted_obj)
                    quarter = get_quarter(posted_obj)
                    year = get_year(posted_obj)
                    
                    posted_by_quarter[quarter] += 1
                    posted_by_year[year] += 1
                    jobs_by_quarter[quarter] += 1
                    jobs_by_year[year] += 1
                    company_quarter[company_normalized][quarter] += 1
                    company_year[company_normalized][year] += 1
                    company_posted_quarter[company_normalized][quarter] += 1
                    
                    if is_key_competitor:
                        competitor_jobs_by_quarter[quarter] += 1
                        competitor_jobs_by_year[year] += 1
            
            # crawl_date 분석
            crawled_str = job.get('crawl_date')
            if crawled_str:
                crawled_obj = parse_date(crawled_str)
                if crawled_obj:
                    all_dates_crawled.append(crawled_obj)
                    quarter = get_quarter(crawled_obj)
                    year = get_year(crawled_obj)
                    
                    crawled_by_quarter[quarter] += 1
                    crawled_by_year[year] += 1
                    company_crawled_quarter[company_normalized][quarter] += 1
            
            # expired_date 분석
            expired_str = job.get('expired_date')
            if expired_str:
                expired_obj = parse_date(expired_str)
                if expired_obj:
                    all_dates_expired.append(expired_obj)
                    quarter = get_quarter(expired_obj)
                    year = get_year(expired_obj)
                    
                    expired_by_quarter[quarter] += 1
                    expired_by_year[year] += 1
                    company_expired_quarter[company_normalized][quarter] += 1
            
            # 채용 기간 계산 (posted_date ~ expired_date)
            if posted_str and expired_str:
                posted_obj = parse_date(posted_str)
                expired_obj = parse_date(expired_str)
                if posted_obj and expired_obj and expired_obj > posted_obj:
                    duration = (expired_obj - posted_obj).days
                    quarter = get_quarter(posted_obj)
                    year = get_year(posted_obj)
                    hiring_duration_by_quarter[quarter].append(duration)
                    hiring_duration_by_year[year].append(duration)
    
    if not all_dates_posted and not all_dates_crawled:
        print("[ERROR] 날짜 정보가 없어 분석을 수행할 수 없습니다.")
        return
    
    # 리포트 생성 (차트 관련 핵심 정보만)
    report = []
    report.append("=" * 80)
    report.append("분기별/연도별 분석 및 주요 경쟁사 비교 결과")
    report.append("=" * 80)
    report.append("")
    report.append("생성되는 차트:")
    report.append("1. 분기별 채용 공고 수: 전체 vs 주요 경쟁사 (스택 바)")
    report.append("2. 연도별 채용 공고 수: 전체 vs 주요 경쟁사 (스택 바)")
    report.append("3. 날짜 기준별 비교 (Posted/Crawled/Expired) - 분기별")
    report.append("4. 주요 경쟁사별 분기별 트렌드 (선 그래프)")
    report.append("5. 주요 경쟁사별 연도별 비교 (그룹 바)")
    report.append("6. 분기별 평균 채용 기간")
    report.append("7. 주요 경쟁사 비중 변화 (분기별)")
    report.append("8. 연도별 평균 채용 기간")
    report.append("9. 날짜 기준별 비교 (Posted/Crawled/Expired) - 연도별")
    report.append("10. 주요 경쟁사 비중 변화 (연도별)")
    report.append("")
    
    # 기간 정보
    if all_dates_posted:
        min_posted = min(all_dates_posted)
        max_posted = max(all_dates_posted)
        report.append(f"[분석 기간 - Posted Date]")
        report.append(f"  시작: {min_posted.strftime('%Y-%m-%d')}")
        report.append(f"  종료: {max_posted.strftime('%Y-%m-%d')}")
        report.append("")
    
    sorted_quarters = sorted(posted_by_quarter.items())
    sorted_years = sorted(posted_by_year.items())
    all_quarters = sorted(set(list(posted_by_quarter.keys()) + 
                              list(crawled_by_quarter.keys()) + 
                              list(expired_by_quarter.keys())))
    all_years = sorted(set(list(posted_by_year.keys()) + 
                          list(crawled_by_year.keys()) + 
                          list(expired_by_year.keys())))
    
    # 저장
    report_text = "\n".join(report)
    os.makedirs('output', exist_ok=True)
    with open('output/15_quarterly_yearly_competitors.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("[SAVE] 분기별/연도별 분석 -> output/15_quarterly_yearly_competitors.txt")
    
    # 시각화
    try:
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        # 1. 분기별 전체 vs 주요 경쟁사 비교 (Posted Date)
        ax1 = fig.add_subplot(gs[0, :])
        if sorted_quarters:
            quarters = [q[0] for q in sorted_quarters]
            total_counts = [q[1] for q in sorted_quarters]
            competitor_counts = [competitor_jobs_by_quarter.get(q, 0) for q in quarters]
            other_counts = [t - c for t, c in zip(total_counts, competitor_counts)]
            
            x = np.arange(len(quarters))
            width = 0.6
            
            ax1.bar(x, other_counts, width, label='기타 회사', color='lightblue', alpha=0.7)
            ax1.bar(x, competitor_counts, width, bottom=other_counts, 
                   label='주요 경쟁사 (한화/카카오/네이버/라인/토스)', color='coral', alpha=0.7)
            
            ax1.set_xlabel('분기', fontsize=12)
            ax1.set_ylabel('채용 공고 수', fontsize=12)
            ax1.set_title('분기별 채용 공고 수: 전체 vs 주요 경쟁사 (Posted Date 기준)', 
                         fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(quarters, rotation=45, ha='right')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 연도별 전체 vs 주요 경쟁사 비교 (Posted Date)
        ax2 = fig.add_subplot(gs[1, 0])
        if sorted_years:
            years = [y[0] for y in sorted_years]
            total_counts_y = [y[1] for y in sorted_years]
            competitor_counts_y = [competitor_jobs_by_year.get(y, 0) for y in years]
            other_counts_y = [t - c for t, c in zip(total_counts_y, competitor_counts_y)]
            
            x = np.arange(len(years))
            width = 0.6
            
            ax2.bar(x, other_counts_y, width, label='기타 회사', color='lightblue', alpha=0.7)
            ax2.bar(x, competitor_counts_y, width, bottom=other_counts_y,
                   label='주요 경쟁사', color='coral', alpha=0.7)
            
            ax2.set_xlabel('연도', fontsize=11)
            ax2.set_ylabel('채용 공고 수', fontsize=11)
            ax2.set_title('연도별 채용 공고 수 비교', fontsize=12, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(years)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 날짜 기준별 비교 (Posted vs Crawled vs Expired) - 분기별
        ax3 = fig.add_subplot(gs[1, 1])
        if all_quarters:
            quarters_plot = all_quarters[:12]  # 최근 12개 분기
            posted_plot = [posted_by_quarter.get(q, 0) for q in quarters_plot]
            crawled_plot = [crawled_by_quarter.get(q, 0) for q in quarters_plot]
            expired_plot = [expired_by_quarter.get(q, 0) for q in quarters_plot]
            
            x = np.arange(len(quarters_plot))
            width = 0.25
            
            ax3.bar(x - width, posted_plot, width, label='Posted Date', color='steelblue', alpha=0.7)
            ax3.bar(x, crawled_plot, width, label='Crawl Date', color='mediumseagreen', alpha=0.7)
            ax3.bar(x + width, expired_plot, width, label='Expired Date', color='coral', alpha=0.7)
            
            ax3.set_xlabel('분기', fontsize=11)
            ax3.set_ylabel('공고 수', fontsize=11)
            ax3.set_title('날짜 기준별 집계 비교 (분기별)', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(quarters_plot, rotation=45, ha='right', fontsize=8)
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 주요 경쟁사별 분기별 트렌드 (선 그래프)
        ax4 = fig.add_subplot(gs[1, 2])
        competitor_quarter_data = {}
        for comp_key in KEY_COMPETITORS:
            comp_name = KEY_COMPETITOR_NAMES.get(comp_key, comp_key)
            for company_file in all_data.keys():
                if comp_key in company_file.lower():
                    company_normalized = normalize_company_name(company_file)
                    if company_normalized in company_posted_quarter:
                        competitor_quarter_data[comp_name] = company_posted_quarter[company_normalized]
                        break
        
        if competitor_quarter_data and sorted_quarters:
            quarters_line = sorted(set([q for q, _ in sorted_quarters]))
            for comp_name, quarter_data in competitor_quarter_data.items():
                counts = [quarter_data.get(q, 0) for q in quarters_line]
                ax4.plot(quarters_line, counts, marker='o', label=comp_name, linewidth=2, markersize=6)
            
            ax4.set_xlabel('분기', fontsize=11)
            ax4.set_ylabel('공고 수', fontsize=11)
            ax4.set_title('주요 경쟁사별 분기별 트렌드', fontsize=12, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45, labelsize=8)
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
        
        # 5. 주요 경쟁사별 연도별 비교
        ax5 = fig.add_subplot(gs[2, 0])
        competitor_year_data = {}
        for comp_key in KEY_COMPETITORS:
            comp_name = KEY_COMPETITOR_NAMES.get(comp_key, comp_key)
            for company_file in all_data.keys():
                if comp_key in company_file.lower():
                    company_normalized = normalize_company_name(company_file)
                    if company_normalized in company_year:
                        competitor_year_data[comp_name] = company_year[company_normalized]
                        break
        
        if competitor_year_data and sorted_years:
            years_plot = [y[0] for y in sorted_years]
            x = np.arange(len(years_plot))
            width = 0.15
            colors = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'orchid']
            
            for idx, (comp_name, year_data) in enumerate(competitor_year_data.items()):
                counts = [year_data.get(y, 0) for y in years_plot]
                offset = (idx - len(competitor_year_data) / 2) * width + width / 2
                ax5.bar(x + offset, counts, width, label=comp_name, 
                       color=colors[idx % len(colors)], alpha=0.7)
            
            ax5.set_xlabel('연도', fontsize=11)
            ax5.set_ylabel('공고 수', fontsize=11)
            ax5.set_title('주요 경쟁사별 연도별 비교', fontsize=12, fontweight='bold')
            ax5.set_xticks(x)
            ax5.set_xticklabels(years_plot)
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 분기별 평균 채용 기간
        ax6 = fig.add_subplot(gs[2, 1])
        if hiring_duration_by_quarter:
            quarters_duration = sorted(hiring_duration_by_quarter.keys())[-8:]  # 최근 8개 분기
            avg_durations = [np.mean(hiring_duration_by_quarter[q]) for q in quarters_duration]
            median_durations = [np.median(hiring_duration_by_quarter[q]) for q in quarters_duration]
            
            x = np.arange(len(quarters_duration))
            width = 0.35
            
            ax6.bar(x - width/2, avg_durations, width, label='평균', color='steelblue', alpha=0.7)
            ax6.bar(x + width/2, median_durations, width, label='중앙값', color='coral', alpha=0.7)
            
            ax6.set_xlabel('분기', fontsize=11)
            ax6.set_ylabel('채용 기간 (일)', fontsize=11)
            ax6.set_title('분기별 평균 채용 기간', fontsize=12, fontweight='bold')
            ax6.set_xticks(x)
            ax6.set_xticklabels(quarters_duration, rotation=45, ha='right', fontsize=8)
            ax6.legend(fontsize=9)
            ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. 주요 경쟁사 비율 변화 (분기별)
        ax7 = fig.add_subplot(gs[2, 2])
        if sorted_quarters:
            quarters_ratio = [q[0] for q in sorted_quarters]
            total_ratio = [q[1] for q in sorted_quarters]
            competitor_ratio = [competitor_jobs_by_quarter.get(q, 0) for q in quarters_ratio]
            ratios = [(c / t * 100) if t > 0 else 0 for c, t in zip(competitor_ratio, total_ratio)]
            
            ax7.plot(quarters_ratio, ratios, marker='o', linewidth=2, markersize=8, color='red', alpha=0.7)
            ax7.fill_between(quarters_ratio, ratios, alpha=0.3, color='red')
            ax7.set_xlabel('분기', fontsize=11)
            ax7.set_ylabel('비율 (%)', fontsize=11)
            ax7.set_title('주요 경쟁사 비중 변화 (분기별)', fontsize=12, fontweight='bold')
            ax7.set_xticklabels(quarters_ratio, rotation=45, ha='right', fontsize=8)
            ax7.grid(True, alpha=0.3)
            ax7.set_ylim([0, max(ratios) * 1.1 if ratios else 100])
        
        # 8. 연도별 평균 채용 기간
        ax8 = fig.add_subplot(gs[3, 0])
        if hiring_duration_by_year:
            years_duration = sorted(hiring_duration_by_year.keys())
            avg_durations_y = [np.mean(hiring_duration_by_year[y]) for y in years_duration]
            median_durations_y = [np.median(hiring_duration_by_year[y]) for y in years_duration]
            
            x = np.arange(len(years_duration))
            width = 0.35
            
            ax8.bar(x - width/2, avg_durations_y, width, label='평균', color='steelblue', alpha=0.7)
            ax8.bar(x + width/2, median_durations_y, width, label='중앙값', color='coral', alpha=0.7)
            
            ax8.set_xlabel('연도', fontsize=11)
            ax8.set_ylabel('채용 기간 (일)', fontsize=11)
            ax8.set_title('연도별 평균 채용 기간', fontsize=12, fontweight='bold')
            ax8.set_xticks(x)
            ax8.set_xticklabels(years_duration)
            ax8.legend(fontsize=9)
            ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. 날짜 기준별 비교 (연도별)
        ax9 = fig.add_subplot(gs[3, 1])
        if all_years:
            posted_y = [posted_by_year.get(y, 0) for y in all_years]
            crawled_y = [crawled_by_year.get(y, 0) for y in all_years]
            expired_y = [expired_by_year.get(y, 0) for y in all_years]
            
            x = np.arange(len(all_years))
            width = 0.25
            
            ax9.bar(x - width, posted_y, width, label='Posted Date', color='steelblue', alpha=0.7)
            ax9.bar(x, crawled_y, width, label='Crawl Date', color='mediumseagreen', alpha=0.7)
            ax9.bar(x + width, expired_y, width, label='Expired Date', color='coral', alpha=0.7)
            
            ax9.set_xlabel('연도', fontsize=11)
            ax9.set_ylabel('공고 수', fontsize=11)
            ax9.set_title('날짜 기준별 집계 비교 (연도별)', fontsize=12, fontweight='bold')
            ax9.set_xticks(x)
            ax9.set_xticklabels(all_years)
            ax9.legend(fontsize=9)
            ax9.grid(True, alpha=0.3, axis='y')
        
        # 10. 주요 경쟁사 비율 변화 (연도별)
        ax10 = fig.add_subplot(gs[3, 2])
        if sorted_years:
            years_ratio = [y[0] for y in sorted_years]
            total_ratio_y = [y[1] for y in sorted_years]
            competitor_ratio_y = [competitor_jobs_by_year.get(y, 0) for y in years_ratio]
            ratios_y = [(c / t * 100) if t > 0 else 0 for c, t in zip(competitor_ratio_y, total_ratio_y)]
            
            ax10.bar(years_ratio, ratios_y, color='red', alpha=0.7)
            ax10.set_xlabel('연도', fontsize=11)
            ax10.set_ylabel('비율 (%)', fontsize=11)
            ax10.set_title('주요 경쟁사 비중 변화 (연도별)', fontsize=12, fontweight='bold')
            ax10.grid(True, alpha=0.3, axis='y')
            ax10.set_ylim([0, max(ratios_y) * 1.1 if ratios_y else 100])
        
        plt.savefig('output/15_quarterly_yearly_competitors.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[SAVE] 분기별/연도별 비교 시각화 -> output/15_quarterly_yearly_competitors.png")
    except Exception as e:
        print(f"[WARN] 시각화 생성 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("[SUCCESS] 분기별/연도별 분석 및 주요 경쟁사 비교 완료!")


# 16. 주요 경쟁사별 스킬 비교 분석
def analyze_competitor_skills_comparison(all_data):
    """
    주요 경쟁사별 스킬 비교 분석
    - 회사별 필수 스킬 vs 선택 스킬 분석
    - 주요 경쟁사 간 스킬 격차 분석
    - 분기별/연도별 스킬 트렌드 비교
    """
    from datetime import datetime
    
    print("\n" + "=" * 80)
    print("주요 경쟁사별 스킬 비교 분석")
    print("=" * 80)
    
    # 주요 경쟁사 정의
    KEY_COMPETITORS = ['hanwha', 'kakao', 'naver', 'line', 'toss']
    KEY_COMPETITOR_NAMES = {
        'hanwha': '한화',
        'kakao': '카카오',
        'naver': '네이버',
        'line': '라인',
        'toss': '토스'
    }
    
    def normalize_company_name(file_name):
        """파일명을 한글 회사명으로 변환"""
        name_lower = file_name.lower()
        for key, korean_name in KEY_COMPETITOR_NAMES.items():
            if key in name_lower:
                return korean_name
        return file_name
    
    def get_year(date_obj):
        """날짜에서 연도 반환"""
        if date_obj:
            return str(date_obj.year)
        return None
    
    def get_quarter(date_obj):
        """날짜에서 분기 반환"""
        if date_obj:
            quarter = (date_obj.month - 1) // 3 + 1
            return f"{date_obj.year}Q{quarter}"
        return None
    
    def parse_date(date_str):
        """날짜 문자열을 datetime 객체로 변환"""
        if not date_str or date_str == 'null' or str(date_str).lower() == 'null':
            return None
        try:
            return datetime.strptime(str(date_str), '%Y-%m-%d')
        except:
            try:
                return datetime.strptime(str(date_str), '%Y/%m/%d')
            except:
                return None
    
    # 회사별 스킬 집계
    company_skills = defaultdict(lambda: Counter())  # 회사별 스킬 빈도
    company_skills_by_quarter = defaultdict(lambda: defaultdict(Counter))  # 회사별 분기별 스킬
    company_skills_by_year = defaultdict(lambda: defaultdict(Counter))  # 회사별 연도별 스킬
    company_job_count = defaultdict(int)  # 회사별 공고 수
    
    for company_file, jobs in all_data.items():
        company_normalized = normalize_company_name(company_file)
        is_key_competitor = any(key in company_file.lower() for key in KEY_COMPETITORS)
        
        if not is_key_competitor:
            continue
        
        for job in jobs:
            company_job_count[company_normalized] += 1
            
            # 스킬 추출
            if 'skill_set_info' in job and 'skill_set' in job['skill_set_info']:
                skills = job['skill_set_info']['skill_set']
                for skill in skills:
                    skill_name = str(skill).strip()
                    if skill_name:
                        company_skills[company_normalized][skill_name] += 1
                        
                        # 날짜별 집계
                        posted_str = job.get('posted_date')
                        if posted_str:
                            posted_obj = parse_date(posted_str)
                            if posted_obj:
                                quarter = get_quarter(posted_obj)
                                year = get_year(posted_obj)
                                if quarter:
                                    company_skills_by_quarter[company_normalized][quarter][skill_name] += 1
                                if year:
                                    company_skills_by_year[company_normalized][year][skill_name] += 1
    
    if not company_skills:
        print("[SKIP] 주요 경쟁사 스킬 데이터가 없습니다.")
        return
    
    # 리포트 생성
    report = []
    report.append("=" * 80)
    report.append("주요 경쟁사별 스킬 비교 분석 결과")
    report.append("=" * 80)
    report.append("")
    
    # 1. 회사별 주요 스킬 (Top 10)
    report.append("[1] 회사별 주요 스킬 (Top 10)")
    report.append("-" * 80)
    for company in sorted(company_skills.keys()):
        top_skills = company_skills[company].most_common(10)
        report.append(f"\n{company} (총 {company_job_count[company]}개 공고):")
        for skill, count in top_skills:
            percentage = (count / company_job_count[company] * 100) if company_job_count[company] > 0 else 0
            report.append(f"  {count:3d}회 ({percentage:5.1f}%) - {skill}")
    
    # 2. 공통 스킬 vs 차별화 스킬 분석
    report.append("\n" + "=" * 80)
    report.append("[2] 공통 스킬 vs 차별화 스킬 분석")
    report.append("-" * 80)
    
    # 모든 회사의 스킬 통합
    all_skill_counts = Counter()
    for company, skills in company_skills.items():
        for skill, count in skills.items():
            all_skill_counts[skill] += count
    
    # 공통 스킬 (2개 이상 회사에서 언급)
    skill_company_count = defaultdict(set)
    for company, skills in company_skills.items():
        for skill in skills.keys():
            skill_company_count[skill].add(company)
    
    common_skills = {skill: len(companies) for skill, companies in skill_company_count.items() 
                     if len(companies) >= 2}
    unique_skills = {skill: list(companies)[0] for skill, companies in skill_company_count.items() 
                     if len(companies) == 1}
    
    report.append("\n공통 스킬 (2개 이상 회사에서 언급):")
    sorted_common = sorted(common_skills.items(), key=lambda x: x[1], reverse=True)[:20]
    for skill, company_count in sorted_common:
        report.append(f"  {skill:30s} - {company_count}개 회사")
    
    report.append("\n차별화 스킬 (특정 회사만 사용):")
    company_unique = defaultdict(list)
    for skill, company in unique_skills.items():
        company_unique[company].append(skill)
    for company in sorted(company_unique.keys()):
        skills_list = company_unique[company][:10]
        report.append(f"\n  {company}: {len(company_unique[company])}개 고유 스킬")
        for skill in skills_list[:10]:
            count = company_skills[company].get(skill, 0)
            report.append(f"    - {skill} ({count}회)")
    
    # 저장
    report_text = "\n".join(report)
    os.makedirs('output', exist_ok=True)
    with open('output/16_competitor_skills_comparison.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("[SAVE] 경쟁사 스킬 비교 분석 -> output/16_competitor_skills_comparison.txt")
    
    # 시각화
    try:
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. 회사별 주요 스킬 비교 (히트맵 스타일)
        ax1 = fig.add_subplot(gs[0, :])
        companies_list = sorted(company_skills.keys())
        top_20_skills = [s[0] for s in all_skill_counts.most_common(20)]
        
        heatmap_data = []
        for skill in top_20_skills:
            row = []
            for company in companies_list:
                count = company_skills[company].get(skill, 0)
                # 정규화 (0-100)
                total = company_job_count[company]
                percentage = (count / total * 100) if total > 0 else 0
                row.append(percentage)
            heatmap_data.append(row)
        
        im = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        ax1.set_xticks(range(len(companies_list)))
        ax1.set_xticklabels(companies_list, rotation=45, ha='right', fontsize=10)
        ax1.set_yticks(range(len(top_20_skills)))
        ax1.set_yticklabels([s[:25] for s in top_20_skills], fontsize=9)
        ax1.set_title('주요 경쟁사별 상위 20개 스킬 활용률 비교 (%)', 
                     fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax1, label='활용률 (%)')
        
        # 2. 회사별 스킬 다양성 비교
        ax2 = fig.add_subplot(gs[1, 0])
        skill_diversity = {company: len(skills) for company, skills in company_skills.items()}
        companies_div = list(skill_diversity.keys())
        diversities = list(skill_diversity.values())
        
        ax2.barh(companies_div, diversities, color='steelblue', alpha=0.7)
        ax2.set_xlabel('고유 스킬 수', fontsize=11)
        ax2.set_title('회사별 스킬 다양성', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. 공통 스킬 vs 차별화 스킬 비율
        ax3 = fig.add_subplot(gs[1, 1])
        common_count = len(common_skills)
        unique_count = len(unique_skills)
        ax3.pie([common_count, unique_count], 
               labels=[f'공통 스킬\n({common_count}개)', f'차별화 스킬\n({unique_count}개)'],
               autopct='%1.1f%%',
               colors=['coral', 'lightblue'],
               startangle=90)
        ax3.set_title('공통 vs 차별화 스킬 비율', fontsize=12, fontweight='bold')
        
        # 4. 회사별 상위 스킬 (Top 5)
        ax4 = fig.add_subplot(gs[1, 2])
        company_top_skills = {}
        for company in companies_list:
            top = company_skills[company].most_common(1)
            if top:
                company_top_skills[company] = top[0][1]
        
        if company_top_skills:
            companies_plot = list(company_top_skills.keys())
            counts_plot = list(company_top_skills.values())
            ax4.bar(companies_plot, counts_plot, color='mediumseagreen', alpha=0.7)
            ax4.set_ylabel('최고 빈도 스킬 횟수', fontsize=11)
            ax4.set_title('회사별 최고 빈도 스킬 강도', fontsize=12, fontweight='bold')
            ax4.set_xticklabels(companies_plot, rotation=45, ha='right', fontsize=9)
            ax4.grid(axis='y', alpha=0.3)
        
        # 5. 분기별 스킬 트렌드 (상위 5개 회사, 상위 10개 스킬)
        ax5 = fig.add_subplot(gs[2, :])
        if company_skills_by_quarter:
            # 모든 분기 수집
            all_quarters = set()
            for company_data in company_skills_by_quarter.values():
                all_quarters.update(company_data.keys())
            sorted_quarters = sorted(all_quarters)[-8:]  # 최근 8개 분기
            
            # 상위 5개 스킬 선택
            top_5_skills_trend = [s[0] for s in all_skill_counts.most_common(5)]
            
            x = np.arange(len(sorted_quarters))
            width = 0.15
            colors_trend = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'orchid']
            
            for idx, skill in enumerate(top_5_skills_trend):
                skill_counts_by_quarter = []
                for quarter in sorted_quarters:
                    total = 0
                    for company in companies_list[:5]:  # 상위 5개 회사만
                        if company in company_skills_by_quarter:
                            total += company_skills_by_quarter[company][quarter].get(skill, 0)
                    skill_counts_by_quarter.append(total)
                
                offset = (idx - len(top_5_skills_trend) / 2) * width + width / 2
                ax5.bar(x + offset, skill_counts_by_quarter, width, 
                       label=skill[:20], color=colors_trend[idx % len(colors_trend)], alpha=0.7)
            
            ax5.set_xlabel('분기', fontsize=11)
            ax5.set_ylabel('스킬 언급 횟수', fontsize=11)
            ax5.set_title('상위 5개 스킬의 분기별 트렌드 (주요 경쟁사 합계)', 
                         fontsize=12, fontweight='bold')
            ax5.set_xticks(x)
            ax5.set_xticklabels(sorted_quarters, rotation=45, ha='right', fontsize=8)
            ax5.legend(fontsize=8, ncol=2)
            ax5.grid(axis='y', alpha=0.3)
        
        plt.savefig('output/16_competitor_skills_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[SAVE] 경쟁사 스킬 비교 시각화 -> output/16_competitor_skills_comparison.png")
    except Exception as e:
        print(f"[WARN] 시각화 생성 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("[SUCCESS] 주요 경쟁사별 스킬 비교 분석 완료!")


# 17. 신규 스킬 등장 및 채택률 분석
def analyze_new_skills_adoption(all_data):
    """
    신규 스킬 등장 및 채택률 분석
    - 신규 스킬 등장 시점 및 속도 분석
    - 회사별 신규 스킬 채택 속도 비교
    - 신규 스킬의 생존률 및 성장률 분석
    """
    from datetime import datetime, timedelta
    
    print("\n" + "=" * 80)
    print("신규 스킬 등장 및 채택률 분석")
    print("=" * 80)
    
    def parse_date(date_str):
        """날짜 문자열을 datetime 객체로 변환"""
        if not date_str or date_str == 'null' or str(date_str).lower() == 'null':
            return None
        try:
            return datetime.strptime(str(date_str), '%Y-%m-%d')
        except:
            try:
                return datetime.strptime(str(date_str), '%Y/%m/%d')
            except:
                return None
    
    # 스킬별 최초 등장 정보
    skill_first_seen = {}  # 스킬: (날짜, 회사)
    skill_adoption_by_company = defaultdict(lambda: defaultdict(list))  # 스킬: 회사: 등장 날짜들
    skill_monthly_count = defaultdict(lambda: defaultdict(int))  # 스킬: 월: 등장 횟수
    
    all_dates = []
    
    for company, jobs in all_data.items():
        for job in jobs:
            posted_str = job.get('posted_date') or job.get('crawl_date')
            date_obj = parse_date(posted_str) if posted_str else None
            
            if date_obj:
                all_dates.append(date_obj)
                month_key = date_obj.strftime('%Y-%m')
            
            if 'skill_set_info' in job and 'skill_set' in job['skill_set_info']:
                skills = job['skill_set_info']['skill_set']
                for skill in skills:
                    skill_name = str(skill).strip()
                    if not skill_name:
                        continue
                    
                    if date_obj:
                        # 최초 등장 기록
                        if skill_name not in skill_first_seen:
                            skill_first_seen[skill_name] = (date_obj, company)
                        
                        # 회사별 채택 기록
                        skill_adoption_by_company[skill_name][company].append(date_obj)
                        
                        # 월별 집계
                        skill_monthly_count[skill_name][month_key] += 1
    
    if not skill_first_seen:
        print("[SKIP] 스킬 데이터가 없습니다.")
        return
    
    # 신규 스킬 정의 (전체 기간의 앞 1/3 기간에 등장한 스킬)
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        period_days = (max_date - min_date).days
        cutoff_date = min_date + timedelta(days=period_days // 3)
        
        new_skills = {skill: (first_date, first_company) 
                     for skill, (first_date, first_company) in skill_first_seen.items()
                     if first_date > cutoff_date}
    else:
        new_skills = {}
    
    # 신규 스킬의 성장률 분석
    new_skill_growth = []
    for skill, (first_date, first_company) in new_skills.items():
        if skill not in skill_monthly_count:
            continue
        
        months_data = sorted(skill_monthly_count[skill].items())
        if len(months_data) < 2:
            continue
        
        # 처음 3개월 vs 마지막 3개월 비교
        first_3_months = sum(count for _, count in months_data[:3])
        last_3_months = sum(count for _, count in months_data[-3:])
        
        if first_3_months > 0:
            growth_rate = ((last_3_months - first_3_months) / first_3_months) * 100
        else:
            growth_rate = 100 if last_3_months > 0 else 0
        
        total_companies = len(skill_adoption_by_company[skill])
        total_mentions = sum(count for _, count in months_data)
        
        new_skill_growth.append((skill, first_date, first_company, total_companies, 
                                total_mentions, growth_rate))
    
    # 성장률 순 정렬
    new_skill_growth.sort(key=lambda x: x[5], reverse=True)
    
    # 리포트 생성
    report = []
    report.append("=" * 80)
    report.append("신규 스킬 등장 및 채택률 분석 결과")
    report.append("=" * 80)
    report.append("")
    
    if all_dates:
        report.append(f"분석 기간: {min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}")
        report.append(f"신규 스킬 기준일 (전체 기간의 1/3 이후): {cutoff_date.strftime('%Y-%m-%d')}")
        report.append(f"신규 스킬 수: {len(new_skills)}개")
        report.append("")
    
    # 상위 성장 스킬
    report.append("[1] 신규 스킬 성장률 Top 20")
    report.append("-" * 80)
    for skill, first_date, first_company, companies, mentions, growth in new_skill_growth[:20]:
        report.append(f"\n{skill}")
        report.append(f"  최초 등장: {first_date.strftime('%Y-%m-%d')} ({first_company})")
        report.append(f"  채택 회사 수: {companies}개")
        report.append(f"  총 언급 횟수: {mentions}회")
        report.append(f"  성장률: {growth:+.1f}%")
    
    # 회사별 신규 스킬 채택 속도
    report.append("\n" + "=" * 80)
    report.append("[2] 회사별 신규 스킬 채택 속도")
    report.append("-" * 80)
    
    company_new_skill_adoption = defaultdict(list)
    for skill, (first_date, first_company) in new_skills.items():
        for company, dates in skill_adoption_by_company[skill].items():
            if dates:
                first_company_date = min(dates)
                days_diff = (first_company_date - first_date).days
                company_new_skill_adoption[company].append((skill, days_diff))
    
    for company in sorted(company_new_skill_adoption.keys()):
        adoptions = company_new_skill_adoption[company]
        avg_days = np.mean([d[1] for d in adoptions]) if adoptions else 0
        report.append(f"\n{company}:")
        report.append(f"  신규 스킬 채택 수: {len(adoptions)}개")
        report.append(f"  평균 채택 지연일: {avg_days:.1f}일")
        report.append(f"  가장 빠르게 채택한 스킬:")
        sorted_by_speed = sorted(adoptions, key=lambda x: x[1])[:5]
        for skill, days in sorted_by_speed:
            report.append(f"    - {skill} ({days}일 후 채택)")
    
    # 저장
    report_text = "\n".join(report)
    os.makedirs('output', exist_ok=True)
    with open('output/17_new_skills_adoption.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("[SAVE] 신규 스킬 채택률 분석 -> output/17_new_skills_adoption.txt")
    
    # 시각화
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 신규 스킬 성장률 Top 15
        if new_skill_growth[:15]:
            skills_growth = [s[0][:25] for s in new_skill_growth[:15]]
            growth_rates = [s[5] for s in new_skill_growth[:15]]
            colors = ['green' if g > 0 else 'red' for g in growth_rates]
            
            axes[0, 0].barh(skills_growth, growth_rates, color=colors, alpha=0.7)
            axes[0, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            axes[0, 0].set_xlabel('성장률 (%)', fontsize=11)
            axes[0, 0].set_title('신규 스킬 성장률 Top 15', fontsize=12, fontweight='bold')
            axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. 회사별 신규 스킬 채택 속도
        if company_new_skill_adoption:
            companies_adopt = list(company_new_skill_adoption.keys())
            avg_delays = [np.mean([d[1] for d in company_new_skill_adoption[c]]) 
                         for c in companies_adopt]
            adoption_counts = [len(company_new_skill_adoption[c]) for c in companies_adopt]
            
            axes[0, 1].bar(companies_adopt, avg_delays, color='steelblue', alpha=0.7)
            axes[0, 1].set_ylabel('평균 채택 지연일 (일)', fontsize=11)
            axes[0, 1].set_title('회사별 신규 스킬 채택 속도', fontsize=12, fontweight='bold')
            axes[0, 1].set_xticklabels(companies_adopt, rotation=45, ha='right', fontsize=9)
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. 회사별 신규 스킬 채택 수
        if company_new_skill_adoption:
            axes[1, 0].bar(companies_adopt, adoption_counts, color='coral', alpha=0.7)
            axes[1, 0].set_ylabel('신규 스킬 채택 수', fontsize=11)
            axes[1, 0].set_title('회사별 신규 스킬 채택 수', fontsize=12, fontweight='bold')
            axes[1, 0].set_xticklabels(companies_adopt, rotation=45, ha='right', fontsize=9)
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. 신규 스킬 월별 등장 추이
        if new_skills and all_dates:
            monthly_new_skills = defaultdict(int)
            for skill, (first_date, _) in new_skills.items():
                month_key = first_date.strftime('%Y-%m')
                monthly_new_skills[month_key] += 1
            
            sorted_months = sorted(monthly_new_skills.items())
            months = [m[0] for m in sorted_months]
            counts = [m[1] for m in sorted_months]
            
            axes[1, 1].plot(months, counts, marker='o', linewidth=2, markersize=8, color='green', alpha=0.7)
            axes[1, 1].fill_between(months, counts, alpha=0.3, color='green')
            axes[1, 1].set_xlabel('월', fontsize=11)
            axes[1, 1].set_ylabel('신규 스킬 등장 수', fontsize=11)
            axes[1, 1].set_title('월별 신규 스킬 등장 추이', fontsize=12, fontweight='bold')
            axes[1, 1].set_xticklabels(months, rotation=45, ha='right', fontsize=8)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/17_new_skills_adoption.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("[SAVE] 신규 스킬 채택률 시각화 -> output/17_new_skills_adoption.png")
    except Exception as e:
        print(f"[WARN] 시각화 생성 실패: {e}")
        import traceback
        traceback.print_exc()
    
    print("[SUCCESS] 신규 스킬 등장 및 채택률 분석 완료!")


def main():
    all_data = load_job_data()  # 자동 경로 탐색
    
    # 9. Transformer 기반 스킬 분석 (새 스킬 대응)
    plot_skill_transformer_analysis(all_data, top_n=15, n_clusters=5)
    
    # 11. 보상 및 복리후생 벤치마킹
    analyze_compensation_benchmarking(all_data)
    
    # 12. 공격적인 채용 전략 감지
    analyze_aggressive_hiring_strategy(all_data)
    
    # 13. 채용 메시지 및 포지셔닝 연구
    analyze_hiring_message_positioning(all_data)
    
    # 14. 시간적 트렌드 분석
    analyze_temporal_trends(all_data)
    
    # 15. 분기별/연도별 분석 및 주요 경쟁사 비교
    analyze_quarterly_yearly_with_competitors(all_data)
    
    # 16. 주요 경쟁사별 스킬 비교 분석
    analyze_competitor_skills_comparison(all_data)
    
    # 17. 신규 스킬 등장 및 채택률 분석
    analyze_new_skills_adoption(all_data)



if __name__ == "__main__":
    main()
