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


def extract_skills_from_data(all_data):
    """데이터에서 스킬 정보 추출"""
    company_skills = defaultdict(list)
    all_skills = []
    
    for company, jobs in all_data.items():
        for job in jobs:
            if 'skill_set_info' in job and 'skill_set' in job['skill_set_info']:
                skills = job['skill_set_info']['skill_set']
                company_skills[company].extend(skills)
                all_skills.extend(skills)
    
    return company_skills, all_skills


def extract_experience_from_data(all_data):
    """데이터에서 경력 정보 추출"""
    company_experience = defaultdict(list)
    
    for company, jobs in all_data.items():
        for job in jobs:
            if 'experience' in job and job['experience']:
                exp = job['experience']
                company_experience[company].append(exp)
    
    return company_experience
    


def extract_job_categories(all_data):
    """데이터에서 직무 카테고리 추출"""
    company_categories = defaultdict(list)
    
    for company, jobs in all_data.items():
        for job in jobs:
            if 'meta_data' in job and 'job_category' in job['meta_data']:
                category = job['meta_data']['job_category']
                if category:
                    company_categories[company].append(category)
    
    return company_categories


# 1. 회사별 채용 공고 수 비교
def plot_job_count_by_company(all_data):
    """회사별 채용 공고 수 시각화"""
    companies = list(all_data.keys())
    counts = [len(jobs) for jobs in all_data.values()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(companies, counts, color=sns.color_palette("husl", len(companies)))
    plt.xlabel('회사', fontsize=12, fontweight='bold')
    plt.ylabel('채용 공고 수', fontsize=12, fontweight='bold')
    plt.title('회사별 채용 공고 수 비교', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    
    # 값 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/1_company_job_count.png', dpi=300, bbox_inches='tight')
    print("[SAVE] 회사별 채용 공고 수 -> output/1_company_job_count.png")
    plt.close()


# 2. Top 20 인기 스킬
def plot_top_skills(all_skills, top_n=20):
    """가장 많이 요구되는 스킬 Top N"""
    skill_counter = Counter(all_skills)
    top_skills = skill_counter.most_common(top_n)
    
    skills, counts = zip(*top_skills)
    
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(skills)), counts, color=sns.color_palette("viridis", len(skills)))
    plt.yticks(range(len(skills)), skills)
    plt.xlabel('빈도', fontsize=12, fontweight='bold')
    plt.ylabel('스킬', fontsize=12, fontweight='bold')
    plt.title(f'가장 많이 요구되는 스킬 Top {top_n}', fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    
    # 값 표시
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(count, i, f' {count}',
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'output/2_top_{top_n}_skills.png', dpi=300, bbox_inches='tight')
    print(f"[SAVE] Top {top_n} 스킬 -> output/2_top_{top_n}_skills.png")
    plt.close()


# 3. 회사별 Top 10 스킬 비교
def plot_company_skills_comparison(company_skills, top_n=10):
    """회사별 주요 스킬 비교"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (company, skills) in enumerate(company_skills.items()):
        if idx >= 6:
            break
            
        skill_counter = Counter(skills)
        top_skills = skill_counter.most_common(top_n)
        
        if top_skills:
            skills_list, counts = zip(*top_skills)
            
            axes[idx].barh(range(len(skills_list)), counts, 
                          color=sns.color_palette("rocket", len(skills_list)))
            axes[idx].set_yticks(range(len(skills_list)))
            axes[idx].set_yticklabels(skills_list, fontsize=9)
            axes[idx].set_xlabel('빈도', fontsize=10)
            axes[idx].set_title(f'{company.upper()} - Top {top_n} 스킬', 
                              fontsize=12, fontweight='bold')
            axes[idx].invert_yaxis()
            
            # 값 표시
            for i, count in enumerate(counts):
                axes[idx].text(count, i, f' {count}',
                             va='center', fontsize=8)
    
    # 빈 subplot 숨기기
    for idx in range(len(company_skills), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('output/3_company_skills_comparison.png', dpi=300, bbox_inches='tight')
    print("[SAVE] 회사별 스킬 비교 -> output/3_company_skills_comparison.png")
    plt.close()


# 4. 스킬 카테고리별 분포 (히트맵)
def plot_skills_heatmap(company_skills, top_n=15):
    """회사별 주요 스킬 히트맵"""
    # 전체에서 가장 많이 나오는 스킬 선택
    all_skills_flat = []
    for skills in company_skills.values():
        all_skills_flat.extend(skills)
    
    top_skills = [skill for skill, _ in Counter(all_skills_flat).most_common(top_n)]
    
    # 회사별로 각 스킬의 빈도 계산
    heatmap_data = []
    companies = list(company_skills.keys())
    
    for company in companies:
        skill_counter = Counter(company_skills[company])
        row = [skill_counter.get(skill, 0) for skill in top_skills]
        heatmap_data.append(row)
    
    # DataFrame 생성
    df = pd.DataFrame(heatmap_data, index=companies, columns=top_skills)
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': '빈도'})
    plt.xlabel('스킬', fontsize=12, fontweight='bold')
    plt.ylabel('회사', fontsize=12, fontweight='bold')
    plt.title(f'회사별 주요 스킬 분포 히트맵 (Top {top_n})', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/4_skills_heatmap.png', dpi=300, bbox_inches='tight')
    print("[SAVE] 스킬 히트맵 -> output/4_skills_heatmap.png")
    plt.close()


# 5. 경력 요구사항 분포
def plot_experience_distribution(company_experience):
    """회사별 경력 요구사항 분포"""
    # 경력 카테고리 정의
    def categorize_experience(exp_str):
        if not exp_str:
            return '명시안함'
        exp_lower = exp_str.lower()
        if '신입' in exp_lower or '무관' in exp_lower or 'entry' in exp_lower:
            return '신입/무관'
        elif '1년' in exp_str or '2년' in exp_str or '3년' in exp_str:
            return '1-3년'
        elif '4년' in exp_str or '5년' in exp_str:
            return '4-5년'
        elif '6년' in exp_str or '7년' in exp_str or '8년' in exp_str or '9년' in exp_str:
            return '6-9년'
        elif '10년' in exp_str or '10년 이상' in exp_str:
            return '10년 이상'
        elif '경력' in exp_lower:
            return '경력'
        else:
            return '기타'
    
    exp_categories = defaultdict(lambda: defaultdict(int))
    
    for company, experiences in company_experience.items():
        for exp in experiences:
            category = categorize_experience(exp)
            exp_categories[company][category] += 1
    
    # DataFrame 생성
    df = pd.DataFrame(exp_categories).fillna(0).T
    
    plt.figure(figsize=(14, 8))
    df.plot(kind='bar', stacked=True, figsize=(14, 8), 
            colormap='tab10', width=0.7)
    plt.xlabel('회사', fontsize=12, fontweight='bold')
    plt.ylabel('채용 공고 수', fontsize=12, fontweight='bold')
    plt.title('회사별 경력 요구사항 분포', fontsize=14, fontweight='bold', pad=20)
    plt.legend(title='경력', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/5_experience_distribution.png', dpi=300, bbox_inches='tight')
    print("[SAVE] 경력 분포 -> output/5_experience_distribution.png")
    plt.close()


# 6. 직무 카테고리 분포
def plot_job_category_distribution(company_categories):
    """회사별 직무 카테고리 분포"""
    all_categories = []
    for categories in company_categories.values():
        all_categories.extend(categories)
    
    category_counter = Counter(all_categories)
    top_categories = category_counter.most_common(10)
    
    categories, counts = zip(*top_categories)
    
    plt.figure(figsize=(12, 6))
    plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90,
            colors=sns.color_palette("pastel", len(categories)))
    plt.title('직무 카테고리 분포 (Top 10)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('output/6_job_category_distribution.png', dpi=300, bbox_inches='tight')
    print("[SAVE] 직무 카테고리 분포 -> output/6_job_category_distribution.png")
    plt.close()


# 7. 스킬 트렌드 워드 클라우드
def plot_skill_wordcloud(all_skills):
    """스킬 워드 클라우드"""
    try:
        from wordcloud import WordCloud
        
        skill_counter = Counter(all_skills)
        
        # 워드클라우드 생성
        wordcloud = WordCloud(
            width=1600, 
            height=800,
            background_color='white',
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate_from_frequencies(skill_counter)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('스킬 트렌드 워드 클라우드', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('output/7_skill_wordcloud.png', dpi=300, bbox_inches='tight')
        print("[SAVE] 워드 클라우드 -> output/7_skill_wordcloud.png")
        plt.close()
    except ImportError:
        print("[SKIP] wordcloud 패키지가 설치되지 않아 워드 클라우드 생성을 건너뜁니다.")
        print("       설치: pip install wordcloud")


# 8. 머신러닝 기반 스킬 조합 분석 (LDA 및 KMeans 클러스터링)
def plot_skill_ml_cooccurrence(all_data, top_n=15, n_topics=5, n_clusters=5):
    """
    머신러닝 기법을 이용한 스킬 조합/트렌드 분석:
      - 1) LDA 토픽 모델링: 자주 같이 요구되는 스킬 군(토픽) 추출
      - 2) KMeans 클러스터링: 비슷한 성격의 스킬 조합 집단 추출
    시각화 결과도 함께 저장
    """
    # 1. 각 채용공고별 스킬 리스트를 공고 하나=문서 하나처럼 간주 (Bag of Skills)
    job_skill_list = []
    company_list = []
    for company, jobs in all_data.items():
        for job in jobs:
            if 'skill_set_info' in job and 'skill_set' in job['skill_set_info']:
                skills = [str(s) for s in job['skill_set_info']['skill_set']]
                job_skill_list.append(" ".join(skills))
                company_list.append(company)
    if len(job_skill_list) < 3:
        print("[SKIP] 머신러닝 기반 스킬 조합 분석을 위한 데이터 부족")
        return

    # 2. 벡터화 (스킬을 단어로 간주, 빈도 기반 BoW)
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b', min_df=2)
    X = vectorizer.fit_transform(job_skill_list)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # 3. LDA 토픽 모델링 (스킬 토픽 추출)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='batch')
    lda.fit(X)
    
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_features_idx = topic.argsort()[::-1][:top_n]
        top_features = feature_names[top_features_idx]
        weights = topic[top_features_idx]
        topics.append((topic_idx, list(zip(top_features, weights))))
    
    # 결과 요약 출력 및 저장
    report = []
    report.append("=" * 80)
    report.append("머신러닝 기반 스킬 조합(LDA 토픽 모델링) 결과")
    report.append("=" * 80)
    for topic_idx, top_terms in topics:
        report.append(f"[LDA 토픽 {topic_idx+1}]")
        for skill, score in top_terms:
            report.append(f"   - {skill:20s} : {score:6.2f}")
        report.append("")
    report_text = "\n".join(report)
    os.makedirs('output', exist_ok=True)
    with open('output/8_lda_skill_topics.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("[SAVE] LDA 기반 스킬 토픽 분석 결과 저장 -> output/8_lda_skill_topics.txt\n")

    # 시각화: 토픽 별로 주요 상위 스킬 표시
    plt.figure(figsize=(12, 2*n_topics))
    for i, (topic_idx, top_terms) in enumerate(topics):
        plt.subplot(n_topics, 1, i+1)
        term_names = [t[0] for t in top_terms]
        term_weights = [t[1] for t in top_terms]
        bars = plt.barh(term_names, term_weights, color=sns.color_palette("tab20", len(term_names)))
        plt.title(f"LDA 토픽 {topic_idx+1} 주요 스킬", fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('output/8_lda_skill_topics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVE] LDA 스킬토픽 히트맵 저장 -> output/8_lda_skill_topics.png")
    
    # 4. KMeans 클러스터링 (비슷한 조합 집단화)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    cluster_cnt = Counter(kmeans_labels)

    # 5. 각 클러스터별 대표 스킬(상위) 추출
    cluster_skill_matrix = []
    for c in range(n_clusters):
        indices = np.where(kmeans_labels == c)[0]
        sub_matrix = X[indices]
        mean_skill = np.array(sub_matrix.sum(axis=0)).flatten()
        top_idx = mean_skill.argsort()[::-1][:top_n]
        top_skills = feature_names[top_idx]
        cluster_skill_matrix.append((c, top_skills, mean_skill[top_idx]))
    
    # 클러스터 레포트와 시각화
    report2 = []
    report2.append("="*80)
    report2.append("머신러닝 기반 스킬 조합(KMeans 클러스터링) 결과")
    report2.append("="*80)
    for c, top_skills, top_counts in cluster_skill_matrix:
        report2.append(f"[클러스터 {c+1} - ({cluster_cnt[c]}건)]")
        for skill, cnt in zip(top_skills, top_counts):
            report2.append(f"   - {skill:20s} : {cnt:6.1f}")
        report2.append("")
    report2_text = "\n".join(report2)
    with open('output/8_kmeans_skill_clusters.txt', 'w', encoding='utf-8') as f:
        f.write(report2_text)
    print("[SAVE] KMeans 기반 스킬 클러스터 분석 결과 저장 -> output/8_kmeans_skill_clusters.txt")

    # 시각화: 클러스터별 상위 스킬
    plt.figure(figsize=(12, 2*n_clusters))
    for i, (cluster_idx, top_skills, top_counts) in enumerate(cluster_skill_matrix):
        plt.subplot(n_clusters, 1, i+1)
        bars = plt.barh(top_skills, top_counts, color=sns.color_palette("tab10", len(top_skills)))
        plt.title(f"클러스터 {cluster_idx+1} - 대표 스킬 조합", fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('output/8_kmeans_skill_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("[SAVE] KMeans 스킬클러스터 시각화 저장 -> output/8_kmeans_skill_clusters.png")


# 기존 카운팅 방식 공출현 분석(막대그래프)도 그대로 보존
def plot_skill_cooccurrence_simple(all_data, top_n=15):
    """스킬 공출현 네트워크(기존 카운팅만 사용한 방법)"""
    from itertools import combinations
    skill_pairs = Counter()
    for company, jobs in all_data.items():
        for job in jobs:
            if 'skill_set_info' in job and 'skill_set' in job['skill_set_info']:
                skills = job['skill_set_info']['skill_set']
                if len(skills) >= 2:
                    for pair in combinations(sorted(skills), 2):
                        skill_pairs[pair] += 1
    top_pairs = skill_pairs.most_common(top_n)
    if not top_pairs:
        print("[SKIP] 스킬 공출현 데이터가 충분하지 않습니다.")
        return
    edges = []
    weights = []
    for (skill1, skill2), count in top_pairs:
        edges.append((skill1, skill2))
        weights.append(count)
    df = pd.DataFrame({
        'Skill 1': [e[0] for e in edges],
        'Skill 2': [e[1] for e in edges],
        'Count': weights
    })
    plt.figure(figsize=(14, 8))
    labels = [f"{s1} ↔ {s2}" for s1, s2 in edges]
    plt.barh(range(len(labels)), weights, color=sns.color_palette("coolwarm", len(labels)))
    plt.yticks(range(len(labels)), labels, fontsize=9)
    plt.xlabel('공출현 빈도', fontsize=12, fontweight='bold')
    plt.title(f'가장 자주 함께 나타나는 스킬 조합 Top {top_n}', fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    for i, weight in enumerate(weights):
        plt.text(weight, i, f' {weight}',
                va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/8_skill_cooccurrence_simple.png', dpi=300, bbox_inches='tight')
    print(f"[SAVE] (카운팅 기준) 스킬 공출현 분석 -> output/8_skill_cooccurrence_simple.png")
    plt.close()


# 9. 통계 요약 리포트 생성
def generate_summary_report(all_data, all_skills, company_skills):
    """통계 요약 리포트"""
    report = []
    report.append("=" * 80)
    report.append("채용 공고 데이터 분석 요약 리포트")
    report.append("=" * 80)
    report.append("")
    
    # 전체 통계
    total_jobs = sum(len(jobs) for jobs in all_data.values())
    report.append(f"[통계] 전체 채용 공고 수: {total_jobs}개")
    report.append(f"[통계] 분석 회사 수: {len(all_data)}개")
    report.append(f"[통계] 전체 고유 스킬 수: {len(set(all_skills))}개")
    report.append(f"[통계] 전체 스킬 언급 횟수: {len(all_skills)}회")
    report.append("")
    
    # 회사별 통계
    report.append("[회사별] 상세 정보:")
    report.append("-" * 80)
    for company, jobs in all_data.items():
        skills = company_skills[company]
        unique_skills = len(set(skills))
        avg_skills = len(skills) / len(jobs) if len(jobs) > 0 else 0
        report.append(f"  * {company.upper()}")
        report.append(f"    - 채용 공고: {len(jobs)}개")
        report.append(f"    - 고유 스킬: {unique_skills}개")
        report.append(f"    - 평균 스킬/공고: {avg_skills:.1f}개")
        report.append("")
    
    # Top 10 스킬
    report.append("[TOP 10] 가장 많이 요구되는 스킬:")
    report.append("-" * 80)
    skill_counter = Counter(all_skills)
    for idx, (skill, count) in enumerate(skill_counter.most_common(10), 1):
        report.append(f"  {idx:2d}. {skill:30s} : {count:4d}회")
    report.append("")
    
    # 리포트 저장
    report_text = "\n".join(report)
    
    os.makedirs('output', exist_ok=True)
    with open('output/0_analysis_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("[SAVE] 요약 리포트 -> output/0_analysis_summary_report.txt")
    print("\n" + report_text)


# 메인 실행 함수
def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("채용 공고 데이터 트렌드 분석 시작")
    print("=" * 80)
    print()
    
    # output 폴더 생성
    os.makedirs('output', exist_ok=True)
    
    # 1. 데이터 로드
    print("[1/4] 데이터 로딩 중...")
    all_data = load_job_data()  # 자동 경로 탐색
    
    if not all_data:
        print("[ERROR] 데이터를 로드할 수 없습니다.")
        print("[INFO] 수동으로 경로를 지정하려면 load_job_data('경로')를 사용하세요.")
        return
    
    print(f"\n[SUCCESS] 총 {len(all_data)}개 회사의 데이터 로드 완료\n")
    
    # 2. 데이터 추출
    print("[2/4] 데이터 분석 중...")
    company_skills, all_skills = extract_skills_from_data(all_data)
    company_experience = extract_experience_from_data(all_data)
    company_categories = extract_job_categories(all_data)
    print("[SUCCESS] 데이터 추출 완료\n")
    
    # 3. 시각화 생성
    print("[3/4] 시각화 생성 중...\n")
    
    # 0. 요약 리포트
    generate_summary_report(all_data, all_skills, company_skills)
    print()
    
    # 1. 회사별 채용 공고 수
    plot_job_count_by_company(all_data)
    
    # 2. Top 스킬
    plot_top_skills(all_skills, top_n=20)
    
    # 3. 회사별 스킬 비교
    plot_company_skills_comparison(company_skills, top_n=10)
    
    # 4. 스킬 히트맵
    plot_skills_heatmap(company_skills, top_n=15)
    
    # 5. 경력 분포
    if company_experience:
        plot_experience_distribution(company_experience)
    
    # 6. 직무 카테고리
    if company_categories:
        plot_job_category_distribution(company_categories)
    
    # 7. 워드 클라우드
    plot_skill_wordcloud(all_skills)
    
    # 8. 머신러닝 기반 스킬 조합/트렌드 분석 (LDA, 클러스터링)
    plot_skill_ml_cooccurrence(all_data, top_n=15, n_topics=5, n_clusters=5)
    # 8-1. 기존 단순 공출현 카운트 분석도 병행
    plot_skill_cooccurrence_simple(all_data, top_n=15)
    
    print("\n" + "=" * 80)
    print("[4/4] 모든 시각화 완료! output/ 폴더를 확인하세요.")
    print("=" * 80)


if __name__ == "__main__":
    main()

