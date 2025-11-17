import json
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from collections import defaultdict
import pickle
from typing import List, Tuple, Dict, Union
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from openai import OpenAI

# .env로부터 OPENAI_API_KEY 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SkillAssociationModel:
    """
    스킬 간의 연관성을 Node2Vec을 통해 학습하는 모델
    """

    def __init__(self, dimensions=128, walk_length=30, num_walks=300, workers=4, 
                 min_skill_freq=2, min_cooccurrence=2, p=1.0, q=0.5):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.min_skill_freq = min_skill_freq
        self.min_cooccurrence = min_cooccurrence
        self.p = p
        self.q = q
        self.graph = None
        self.model = None
        self.skill_to_idx = {}
        self.idx_to_skill = {}
        self.llm_norm_cache = {}
        
        # 자주 사용되는 스킬 정규화 규칙
        self.normalization_rules = self._build_normalization_rules()

    def _build_normalization_rules(self) -> Dict[str, str]:
        """
        자주 나오는 패턴 기반 규칙 생성 (LLM 호출 줄이기)
        """
        return {
            # JavaScript 생태계
            r'(?i)^react\.?js$': 'React',
            r'(?i)^react$': 'React',
            r'(?i)^node\.?js$': 'Node.js',
            r'(?i)^node$': 'Node.js',
            r'(?i)^vue\.?js$': 'Vue.js',
            r'(?i)^vue$': 'Vue.js',
            r'(?i)^angular\.?js$': 'Angular',
            r'(?i)^angular$': 'Angular',
            r'(?i)^next\.?js$': 'Next.js',
            r'(?i)^express\.?js$': 'Express.js',
            r'(?i)^express$': 'Express.js',
            
            # 데이터베이스
            r'(?i)^postgre(?:sql)?$': 'PostgreSQL',
            r'(?i)^mysql$': 'MySQL',
            r'(?i)^mongodb$': 'MongoDB',
            r'(?i)^mongo$': 'MongoDB',
            r'(?i)^redis$': 'Redis',
            r'(?i)^mariadb$': 'MariaDB',
            r'(?i)^oracle$': 'Oracle',
            r'(?i)^mssql$': 'Microsoft SQL Server',
            r'(?i)^ms\s*sql$': 'Microsoft SQL Server',
            
            # 클라우드
            r'(?i)^aws$': 'AWS',
            r'(?i)^amazon\s*web\s*services$': 'AWS',
            r'(?i)^gcp$': 'Google Cloud Platform',
            r'(?i)^google\s*cloud$': 'Google Cloud Platform',
            r'(?i)^azure$': 'Microsoft Azure',
            r'(?i)^ms\s*azure$': 'Microsoft Azure',
            
            # DevOps
            r'(?i)^docker$': 'Docker',
            r'(?i)^kubernetes$': 'Kubernetes',
            r'(?i)^k8s$': 'Kubernetes',
            r'(?i)^jenkins$': 'Jenkins',
            r'(?i)^gitlab$': 'GitLab',
            r'(?i)^github$': 'GitHub',
            r'(?i)^circleci$': 'CircleCI',
            
            # 프로그래밍 언어
            r'(?i)^javascript$': 'JavaScript',
            r'(?i)^js$': 'JavaScript',
            r'(?i)^typescript$': 'TypeScript',
            r'(?i)^ts$': 'TypeScript',
            r'(?i)^python$': 'Python',
            r'(?i)^java$': 'Java',
            r'(?i)^c\+\+$': 'C++',
            r'(?i)^cpp$': 'C++',
            r'(?i)^c#$': 'C#',
            r'(?i)^csharp$': 'C#',
            r'(?i)^golang$': 'Go',
            r'(?i)^go$': 'Go',
            r'(?i)^rust$': 'Rust',
            r'(?i)^kotlin$': 'Kotlin',
            r'(?i)^swift$': 'Swift',
            r'(?i)^ruby$': 'Ruby',
            r'(?i)^php$': 'PHP',
            
            # 프레임워크
            r'(?i)^spring\s*boot$': 'Spring Boot',
            r'(?i)^spring$': 'Spring',
            r'(?i)^django$': 'Django',
            r'(?i)^flask$': 'Flask',
            r'(?i)^fastapi$': 'FastAPI',
            r'(?i)^fast\s*api$': 'FastAPI',
            r'(?i)^rails$': 'Ruby on Rails',
            r'(?i)^laravel$': 'Laravel',
            r'(?i)^\.net$': '.NET',
            r'(?i)^dotnet$': '.NET',
            
            # 기타 도구
            r'(?i)^git$': 'Git',
            r'(?i)^jira$': 'Jira',
            r'(?i)^confluence$': 'Confluence',
            r'(?i)^slack$': 'Slack',
            r'(?i)^figma$': 'Figma',
            r'(?i)^tensorflow$': 'TensorFlow',
            r'(?i)^pytorch$': 'PyTorch',
            r'(?i)^scikit[_\-\s]learn$': 'Scikit-learn',
            r'(?i)^pandas$': 'Pandas',
            r'(?i)^numpy$': 'NumPy',
        }

    def _apply_normalization_rules(self, skill: str) -> str:
        """
        규칙 기반으로 스킬 정규화 시도
        """
        skill_stripped = skill.strip()
        
        for pattern, normalized in self.normalization_rules.items():
            if re.match(pattern, skill_stripped):
                return normalized
        
        return None  # 규칙에 매칭되지 않음

    def batch_normalize_skills(self, skills: List[str], batch_size: int = 50) -> Dict[str, str]:
        """
        여러 스킬을 한 번에 정규화 (API 호출 최소화)
        
        Args:
            skills: 정규화할 스킬 리스트
            batch_size: 한 번에 처리할 스킬 개수
        
        Returns:
            원본 스킬 -> 정규화된 스킬 매핑
        """
        results = {}
        
        # 1단계: 캐시에서 확인
        uncached_skills = []
        for skill in skills:
            if skill in self.llm_norm_cache:
                results[skill] = self.llm_norm_cache[skill]
            else:
                uncached_skills.append(skill)
        
        if not uncached_skills:
            return results
        
        print(f"  캐시에서 {len(results)}개 스킬 로드, {len(uncached_skills)}개 스킬 정규화 필요")
        
        # 2단계: 규칙 기반으로 처리
        rule_processed = {}
        llm_needed = []
        
        for skill in uncached_skills:
            normalized = self._apply_normalization_rules(skill)
            if normalized:
                rule_processed[skill] = normalized
                self.llm_norm_cache[skill] = normalized
            else:
                llm_needed.append(skill)
        
        results.update(rule_processed)
        print(f"  규칙 기반으로 {len(rule_processed)}개 스킬 처리, {len(llm_needed)}개 스킬 LLM 처리 필요")
        
        if not llm_needed:
            return results
        
        # 3단계: LLM 배치 처리
        for i in range(0, len(llm_needed), batch_size):
            batch = llm_needed[i:i+batch_size]
            
            prompt = f"""다음 IT 스킬들을 표준 명칭으로 정규화하세요.

규칙:
1. 공식 명칭 사용 (예: Node.js, PostgreSQL, React, Spring Boot)
2. 대소문자 정확히 (예: JavaScript, TypeScript, AWS)
3. 약어는 잘 알려진 형태로 (예: JS -> JavaScript)
4. 프레임워크/라이브러리는 띄어쓰기 포함 (예: Spring Boot, React Native)

입력 스킬: {json.dumps(batch, ensure_ascii=False)}

다음 JSON 형식으로만 답변하세요 (다른 설명 없이):
{{
  "원본스킬1": "정규화된스킬1",
  "원본스킬2": "정규화된스킬2",
  ...
}}"""
            
            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                
                batch_results = json.loads(completion.choices[0].message.content)
                
                # 결과 검증 및 저장
                for original_skill in batch:
                    if original_skill in batch_results:
                        normalized = batch_results[original_skill].strip()
                        if normalized:  # 빈 문자열이 아닌 경우만
                            results[original_skill] = normalized
                            self.llm_norm_cache[original_skill] = normalized
                        else:
                            # Fallback
                            results[original_skill] = original_skill.strip().title()
                            self.llm_norm_cache[original_skill] = original_skill.strip().title()
                    else:
                        # LLM이 해당 스킬을 반환하지 않은 경우 Fallback
                        results[original_skill] = original_skill.strip().title()
                        self.llm_norm_cache[original_skill] = original_skill.strip().title()
                
                print(f"  배치 {i//batch_size + 1}: {len(batch)}개 스킬 정규화 완료")
                
            except Exception as e:
                print(f"  [경고] 배치 {i//batch_size + 1} 정규화 실패: {e}")
                # Fallback: 각 스킬을 title case로 변환
                for skill in batch:
                    fallback_normalized = skill.strip().title()
                    results[skill] = fallback_normalized
                    self.llm_norm_cache[skill] = fallback_normalized
        
        return results

    def normalize_skill(self, skill: str) -> str:
        """
        단일 스킬 정규화 (하위 호환성 유지)
        실제로는 batch_normalize_skills를 호출
        """
        result = self.batch_normalize_skills([skill])
        return result.get(skill, skill.strip().title())

    def load_jobs_data(self, json_file_paths: Union[str, List[str]]) -> List[List[str]]:
        if isinstance(json_file_paths, str):
            json_file_paths = [json_file_paths]
        
        all_skill_sets = []
        all_unique_skills = set()
        
        # 1단계: 모든 파일에서 스킬 수집
        print("\n=== 1단계: 스킬 데이터 수집 ===")
        for json_file_path in json_file_paths:
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    jobs = json.load(f)
                
                file_skill_sets = []
                for job in jobs:
                    if 'skill_set_info' in job and 'skill_set' in job['skill_set_info']:
                        skill_set = job['skill_set_info']['skill_set']
                        if skill_set:  # 빈 리스트가 아닌 경우만
                            # 공백 제거만 수행 (정규화는 나중에 배치로)
                            cleaned_skills = [skill.strip() for skill in skill_set if skill.strip()]
                            if cleaned_skills:
                                file_skill_sets.append(cleaned_skills)
                                all_unique_skills.update(cleaned_skills)
                
                all_skill_sets.extend(file_skill_sets)
                print(f"  {json_file_path}: {len(file_skill_sets)}개의 job")
                
            except FileNotFoundError:
                print(f"  경고: {json_file_path} 파일을 찾을 수 없습니다.")
            except Exception as e:
                print(f"  경고: {json_file_path} 파일 로드 중 오류 발생: {e}")
        
        print(f"\n총 {len(all_skill_sets)}개의 job, {len(all_unique_skills)}개의 고유 스킬 수집")
        
        # 2단계: 모든 고유 스킬을 한 번에 정규화
        print("\n=== 2단계: 스킬 정규화 (배치 처리) ===")
        unique_skills_list = list(all_unique_skills)
        normalization_map = self.batch_normalize_skills(unique_skills_list, batch_size=100)
        
        print(f"총 {len(normalization_map)}개 스킬 정규화 완료")
        print(f"캐시 크기: {len(self.llm_norm_cache)}개")
        
        # 3단계: 정규화된 스킬로 skill_sets 업데이트
        print("\n=== 3단계: 스킬셋 업데이트 ===")
        normalized_skill_sets = []
        for skill_set in all_skill_sets:
            normalized_set = [normalization_map.get(skill, skill) for skill in skill_set]
            # 중복 제거 (정규화로 인해 같아진 스킬들)
            normalized_set = list(dict.fromkeys(normalized_set))  # 순서 유지하며 중복 제거
            if normalized_set:
                normalized_skill_sets.append(normalized_set)
        
        print(f"최종 {len(normalized_skill_sets)}개의 정규화된 스킬셋 생성")
        
        # 정규화 예시 출력
        print("\n=== 정규화 예시 (처음 20개) ===")
        sample_items = list(normalization_map.items())[:20]
        for original, normalized in sample_items:
            if original != normalized:
                print(f"  '{original}' -> '{normalized}'")
        
        return normalized_skill_sets

    def build_graph(self, skill_sets: List[List[str]]):
        """
        그래프 구축 - PMI 기반 가중치 사용
        """
        self.graph = nx.Graph()
        co_occurrence = defaultdict(int)
        skill_freq = defaultdict(int)
        
        total_jobs = len(skill_sets)
        
        for skill_set in skill_sets:
            unique_skills = list(set(skill_set))
            for skill in unique_skills:
                skill_freq[skill] += 1
            for i, skill1 in enumerate(unique_skills):
                for skill2 in unique_skills[i+1:]:
                    pair = tuple(sorted([skill1, skill2]))
                    co_occurrence[pair] += 1
        
        # 필터링
        filtered_skills = {skill: freq for skill, freq in skill_freq.items() if freq >= self.min_skill_freq}
        print(f"필터링 전: {len(skill_freq)}개 스킬, 필터링 후: {len(filtered_skills)}개 스킬 (최소 빈도: {self.min_skill_freq})")
        
        # 노드 추가
        for skill in filtered_skills.keys():
            self.graph.add_node(skill, frequency=skill_freq[skill])
        
        # PMI 기반 엣지 가중치 계산
        edge_count = 0
        pmi_values = []
        
        for (skill1, skill2), count in co_occurrence.items():
            if skill1 in filtered_skills and skill2 in filtered_skills and count >= self.min_cooccurrence:
                # PMI 계산
                p_skill1 = skill_freq[skill1] / total_jobs
                p_skill2 = skill_freq[skill2] / total_jobs
                p_cooccur = count / total_jobs
                
                # PMI with smoothing
                pmi = np.log((p_cooccur + 1e-10) / ((p_skill1 * p_skill2) + 1e-10))
                
                # Positive PMI만 사용
                if pmi > 0:
                    # 추가: co-occurrence 빈도도 반영 (자주 같이 나오는 것에 더 높은 가중치)
                    weight = pmi * np.log1p(count)
                    self.graph.add_edge(skill1, skill2, weight=weight, pmi=pmi, count=count)
                    edge_count += 1
                    pmi_values.append(pmi)
        
        print(f"그래프 생성 완료: {self.graph.number_of_nodes()}개의 노드, {edge_count}개의 엣지 (최소 co-occurrence: {self.min_cooccurrence})")
        
        if pmi_values:
            print(f"PMI 통계: 평균={np.mean(pmi_values):.4f}, 중앙값={np.median(pmi_values):.4f}, 최대={np.max(pmi_values):.4f}")
        
        degrees = dict(self.graph.degree())
        top_skills = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:15]
        print("\n상위 15개 연결 스킬:")
        for skill, degree in top_skills:
            print(f"  {skill}: {degree}개의 연결 (빈도: {skill_freq[skill]})")

    def train(self, skill_sets: List[List[str]]):
        if self.graph is None:
            self.build_graph(skill_sets)
        
        print("\nNode2Vec 모델 학습 시작...")
        print(f"파라미터: p={self.p}, q={self.q}, walk_length={self.walk_length}, num_walks={self.num_walks}")
        
        node2vec = Node2Vec(
            self.graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            weight_key='weight',
            p=self.p,
            q=self.q
        )
        
        walks = [[str(node) for node in walk] for walk in node2vec.walks]
        print(f"생성된 walks 수: {len(walks)}")
        print(f"평균 walk 길이: {np.mean([len(walk) for walk in walks]):.2f}")
        
        from gensim.models import Word2Vec
        min_count_for_vocab = 1  # 작은 데이터셋에서는 1로 설정
        
        self.model = Word2Vec(
            vector_size=self.dimensions,
            window=10,
            min_count=min_count_for_vocab,
            batch_words=10000,
            workers=self.workers,
            sg=1,  # Skip-gram
            hs=0,  # Negative sampling 사용
            negative=10,  # 작은 데이터셋에서는 negative sampling 늘림
            ns_exponent=0.75,
            alpha=0.025,
            min_alpha=0.0001,
            epochs=20  # epoch 증가
        )
        
        self.model.build_vocab(walks, update=False)
        print(f"Vocabulary 크기: {len(self.model.wv)}")
        
        self.model.train(
            walks,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
            compute_loss=True
        )
        
        if hasattr(self.model, 'get_latest_training_loss'):
            print(f"학습 손실: {self.model.get_latest_training_loss():.2f}")
        
        print("Node2Vec 모델 학습 완료!")
        
        self.skill_to_idx = {skill: idx for idx, skill in enumerate(self.model.wv.index_to_key)}
        self.idx_to_skill = {idx: skill for skill, idx in self.skill_to_idx.items()}

    def get_similar_skills(self, skill: str, top_n: int = 10, use_graph: bool = True) -> List[Tuple[str, float]]:
        """
        유사 스킬 검색 - 그래프 정보도 활용
        
        Args:
            skill: 검색할 스킬
            top_n: 반환할 결과 수
            use_graph: 그래프 연결 정보도 함께 고려할지 여부
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train() 메서드를 먼저 호출하세요.")
        
        if skill not in self.model.wv:
            print(f"경고: '{skill}' 스킬이 모델에 존재하지 않습니다.")
            similar_names = [s for s in self.model.wv.index_to_key if skill.lower() in s.lower() or s.lower() in skill.lower()]
            if similar_names:
                print(f"유사한 스킬 이름 후보: {similar_names[:5]}")
            return []
        
        # Node2Vec 임베딩 기반 유사도
        similar_skills = self.model.wv.most_similar(skill, topn=top_n * 3)  # 더 많이 가져오기
        
        if not use_graph or self.graph is None:
            return similar_skills[:top_n]
        
        # 그래프 연결 정보로 재순위화
        reranked = []
        for similar_skill, embedding_score in similar_skills:
            score = embedding_score
            
            # 직접 연결되어 있으면 점수 부스트
            if self.graph.has_edge(skill, similar_skill):
                edge_data = self.graph[skill][similar_skill]
                pmi = edge_data.get('pmi', 0)
                count = edge_data.get('count', 0)
                
                # PMI와 co-occurrence를 결합한 보너스
                boost = 0.2 * (pmi / 10) + 0.1 * np.log1p(count) / 10
                score = min(1.0, score + boost)
            
            reranked.append((similar_skill, score))
        
        # 점수로 재정렬
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_n]

    def get_skills_by_context(self, skills: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train() 메서드를 먼저 호출하세요.")
        
        valid_skills = [s for s in skills if s in self.model.wv]
        if not valid_skills:
            print(f"경고: 입력된 스킬 중 모델에 존재하는 스킬이 없습니다.")
            return []
        
        skill_vectors = [self.model.wv[skill] for skill in valid_skills]
        avg_vector = np.mean(skill_vectors, axis=0)
        similar_skills = self.model.wv.similar_by_vector(avg_vector, topn=top_n + len(valid_skills))
        result = [(skill, score) for skill, score in similar_skills if skill not in valid_skills]
        return result[:top_n]

    def analyze_skill_community(self, skill: str, max_depth: int = 2) -> Dict:
        """
        특정 스킬 주변의 커뮤니티 분석
        """
        if self.graph is None or skill not in self.graph:
            return {}
        
        # BFS로 주변 노드 탐색
        import networkx as nx
        from collections import deque
        
        visited = {skill}
        queue = deque([(skill, 0)])
        neighbors_by_depth = defaultdict(list)
        
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    edge_data = self.graph[current][neighbor]
                    neighbors_by_depth[depth + 1].append({
                        'skill': neighbor,
                        'weight': edge_data.get('weight', 0),
                        'pmi': edge_data.get('pmi', 0),
                        'count': edge_data.get('count', 0)
                    })
                    queue.append((neighbor, depth + 1))
        
        return {
            'center': skill,
            'neighbors_by_depth': dict(neighbors_by_depth),
            'total_neighbors': len(visited) - 1
        }

    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        model_data = {
            'model': self.model,
            'graph': self.graph,
            'skill_to_idx': self.skill_to_idx,
            'idx_to_skill': self.idx_to_skill,
            'dimensions': self.dimensions,
            'walk_length': self.walk_length,
            'num_walks': self.num_walks,
            'llm_norm_cache': self.llm_norm_cache,
            'normalization_rules': self.normalization_rules,
            'p': self.p,
            'q': self.q
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"모델이 {filepath}에 저장되었습니다.")
        print(f"  - 정규화 캐시: {len(self.llm_norm_cache)}개 항목")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.graph = model_data['graph']
        self.skill_to_idx = model_data['skill_to_idx']
        self.idx_to_skill = model_data['idx_to_skill']
        self.dimensions = model_data['dimensions']
        self.walk_length = model_data['walk_length']
        self.num_walks = model_data['num_walks']
        self.p = model_data.get('p', 1.0)
        self.q = model_data.get('q', 1.0)
        
        if 'llm_norm_cache' in model_data:
            self.llm_norm_cache = model_data['llm_norm_cache']
            print(f"  - 정규화 캐시: {len(self.llm_norm_cache)}개 항목 로드")
        
        if 'normalization_rules' in model_data:
            self.normalization_rules = model_data['normalization_rules']
        
        print(f"모델이 {filepath}에서 로드되었습니다.")

    def get_all_skills(self) -> List[str]:
        if self.model is None:
            return []
        return list(self.model.wv.index_to_key)
    
    def evaluate_model(self, test_skill_pairs: List[Tuple[str, str, float]] = None) -> Dict:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        if self.graph is None:
            raise ValueError("그래프가 없습니다.")
        
        results = {
            'vocabulary_size': len(self.model.wv),
            'graph_nodes': self.graph.number_of_nodes(),
            'graph_edges': self.graph.number_of_edges(),
            'avg_degree': np.mean(list(dict(self.graph.degree()).values())),
            'density': nx.density(self.graph)
        }
        
        if test_skill_pairs is None:
            edges = list(self.graph.edges())[:100]
            test_pairs = [(str(u), str(v)) for u, v in edges]
        else:
            test_pairs = [(s1, s2) for s1, s2, _ in test_skill_pairs]
        
        similarities = []
        graph_connected = []
        
        for skill1, skill2 in test_pairs:
            if skill1 in self.model.wv and skill2 in self.model.wv:
                similarity = self.model.wv.similarity(skill1, skill2)
                similarities.append(similarity)
                if self.graph.has_edge(skill1, skill2):
                    graph_connected.append(1)
                else:
                    graph_connected.append(0)
        
        if similarities:
            results['mean_similarity'] = np.mean(similarities)
            results['std_similarity'] = np.std(similarities)
            results['min_similarity'] = np.min(similarities)
            results['max_similarity'] = np.max(similarities)
            results['graph_connection_rate'] = np.mean(graph_connected) if graph_connected else 0
            results['evaluated_pairs'] = len(similarities)
        
        return results


def main():
    """
    메인 실행 함수
    """
    # 모든 _jobs.json 파일 읽기
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / 'data'
    pattern = data_dir / '*_jobs.json'
    print("pattern: ", pattern)
    json_files = [str(path) for path in data_dir.glob('*_jobs.json')]

    print("=== 학습에 사용될 _jobs.json 파일 목록 ===")
    for f in json_files:
        print(f" - {f}")
    print(f"총 {len(json_files)}개 파일을 찾았습니다.")

    # 모델 초기화 (작은 데이터셋에 최적화된 파라미터)
    model = SkillAssociationModel(
        dimensions=128,
        walk_length=30,
        num_walks=300,  # 증가
        workers=6,
        min_skill_freq=2,  # 완화 (3 -> 2)
        min_cooccurrence=2,  # 완화 (3 -> 2)
        p=1.0,  # return parameter
        q=0.5   # in-out parameter (더 넓은 탐색, BFS 경향)
    )

    # 데이터 로드 및 학습
    print("=" * 60)
    print("스킬 연관성 학습 모델 (고도화 버전)")
    print("=" * 60)

    skill_sets = model.load_jobs_data(json_files)
    model.train(skill_sets)

    # 모델 평가
    print("\n" + "=" * 60)
    print("모델 평가")
    print("=" * 60)
    evaluation = model.evaluate_model()
    for key, value in evaluation.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 모델 저장
    model_path = project_root / 'app' / 'core' / 'data_model' / 'skill_association_model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    # 테스트
    test_skills = ['Java', 'Python', 'Docker', 'React', 'Spring Boot']
    
    for test_skill in test_skills:
        if test_skill in model.get_all_skills():
            print("\n" + "=" * 60)
            print(f"테스트: {test_skill}와 유사한 스킬")
            print("=" * 60)
            similar_skills = model.get_similar_skills(test_skill, top_n=10, use_graph=True)
            for skill, score in similar_skills:
                # 그래프 연결 정보 표시
                if model.graph.has_edge(test_skill, skill):
                    edge_data = model.graph[test_skill][skill]
                    print(f"  {skill}: {score:.4f} [직접연결, PMI={edge_data.get('pmi', 0):.2f}, count={edge_data.get('count', 0)}]")
                else:
                    print(f"  {skill}: {score:.4f}")
            
            # 커뮤니티 분석
            community = model.analyze_skill_community(test_skill, max_depth=2)
            if community.get('neighbors_by_depth'):
                print(f"\n  커뮤니티 크기: {community['total_neighbors']}개 스킬")
                for depth, neighbors in community['neighbors_by_depth'].items():
                    print(f"  - Depth {depth}: {len(neighbors)}개")

    # 컨텍스트 기반 추천 테스트
    print("\n" + "=" * 60)
    print("테스트: Java, Spring Boot 컨텍스트 기반 추천")
    print("=" * 60)
    context_skills = model.get_skills_by_context(['Java', 'Spring Boot'], top_n=10)
    for skill, score in context_skills:
        print(f"  {skill}: {score:.4f}")

    # 스킬 통계
    print("\n" + "=" * 60)
    print("스킬 통계")
    print("=" * 60)
    all_skills = model.get_all_skills()
    print(f"총 {len(all_skills)}개의 스킬")
    
    # 빈도 높은 스킬 출력
    if model.graph:
        skill_freq = {node: model.graph.nodes[node]['frequency'] for node in model.graph.nodes()}
        top_freq = sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        print("\n빈도 Top 20 스킬:")
        for skill, freq in top_freq:
            print(f"  {skill}: {freq}회")


if __name__ == "__main__":
    main()