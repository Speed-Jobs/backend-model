import json
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from collections import defaultdict
import pickle
from typing import List, Tuple, Dict, Union
import os
from pathlib import Path

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
                 min_skill_freq=2, min_cooccurrence=3, p=1.0, q=1.0):
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

    def _llm_skill_normalization(self, skill: str) -> str:
        """
        LLM을 이용해 스킬 명칭을 표준화 (prompt-engineering)
        """
        if skill in self.llm_norm_cache:
            return self.llm_norm_cache[skill]

        # 최대한 짧고 구조화된 답을 기대
        prompt = (
            f"아래의 IT/개발 관련 스킬(기술 스택, 도구, 언어, 프레임워크 등) 명칭을 표준화된 글로벌 영어 표기(예: 'Node.js', 'JavaScript', 'Spring Boot', 'AWS')로 답변하세요.\n"
            f"애매하면 가장 잘 알려진 버전으로, 대소문자도 원칙적으로 맞춰주세요.\n"
            f"스킬: \"{skill.strip()}\"\n"
            "정답만 한 줄로 답해주세요."
        )

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                n=1,
            )
            normalized_skill = completion.choices[0].message.content.strip()
            if not normalized_skill:
                normalized_skill = skill.strip()
        except Exception as e:
            print(f"  [경고] LLM 스킬 정규화 실패({skill}): {e}")
            normalized_skill = skill.strip()

        # 캐싱
        self.llm_norm_cache[skill] = normalized_skill
        return normalized_skill

    def normalize_skill(self, skill: str) -> str:
        """
        스킬 이름 정규화 (대소문자 통일, 공백 제거 등)
        - 일반적인 변형 처리는 LLM 기반으로 처리

        Args:
            skill: 원본 스킬 이름
        Returns:
            정규화된 스킬 이름
        """
        # 공백 제거 및 소문자 변환(기본)
        normalized = skill.strip().lower()
        # 만약 완전히 ascii 문자도 아니거나, 너무 흔한 축약/오타/비표준이면 LLM으로 처리
        # (모든 경우 LLM에 보내면 느릴 수 있으니, 간단하게 "대소문자만 다른 경우"는 직접 리턴)
        # -> 예시: skill이 15자 이하, 알파벳 숫자 공백/./-/+/_만 있으면 그 skill을 타이틀 케이스로 리턴
        import re
        if re.fullmatch(r"[a-zA-Z0-9 .+\-_/]{2,15}", skill.strip()):
            # 타이틀 케이스(자주 쓰는 영어)로 우선 변환
            return skill.strip().title()
        # 나머지는 LLM에게 위임
        return self._llm_skill_normalization(skill)

    def load_jobs_data(self, json_file_paths: Union[str, List[str]]) -> List[List[str]]:
        if isinstance(json_file_paths, str):
            json_file_paths = [json_file_paths]
        all_skill_sets = []
        for json_file_path in json_file_paths:
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    jobs = json.load(f)
                for job in jobs:
                    if 'skill_set_info' in job and 'skill_set' in job['skill_set_info']:
                        skill_set = job['skill_set_info']['skill_set']
                        if skill_set:  # 빈 리스트가 아닌 경우만
                            # LLM 기반 스킬 정규화 적용
                            normalized_skills = [self.normalize_skill(skill) for skill in skill_set if skill.strip()]
                            if normalized_skills:
                                all_skill_sets.append(normalized_skills)
                print(f"  {json_file_path}: {len([j for j in jobs if 'skill_set_info' in j and j['skill_set_info'].get('skill_set')])}개의 job")
            except FileNotFoundError:
                print(f"  경고: {json_file_path} 파일을 찾을 수 없습니다.")
            except Exception as e:
                print(f"  경고: {json_file_path} 파일 로드 중 오류 발생: {e}")
        print(f"총 {len(all_skill_sets)}개의 job에서 스킬 데이터를 추출했습니다.")
        return all_skill_sets

    def build_graph(self, skill_sets: List[List[str]]):
        self.graph = nx.Graph()
        co_occurrence = defaultdict(int)
        skill_freq = defaultdict(int)
        for skill_set in skill_sets:
            unique_skills = list(set(skill_set))
            for skill in unique_skills:
                skill_freq[skill] += 1
            for i, skill1 in enumerate(unique_skills):
                for skill2 in unique_skills[i+1:]:
                    pair = tuple(sorted([skill1, skill2]))
                    co_occurrence[pair] += 1
        filtered_skills = {skill: freq for skill, freq in skill_freq.items() if freq >= self.min_skill_freq}
        print(f"필터링 전: {len(skill_freq)}개 스킬, 필터링 후: {len(filtered_skills)}개 스킬 (최소 빈도: {self.min_skill_freq})")
        for skill in filtered_skills.keys():
            self.graph.add_node(skill, frequency=skill_freq[skill])
        edge_count = 0
        for (skill1, skill2), count in co_occurrence.items():
            if skill1 in filtered_skills and skill2 in filtered_skills and count >= self.min_cooccurrence:
                normalized_weight = np.log1p(count)
                self.graph.add_edge(skill1, skill2, weight=normalized_weight)
                edge_count += 1
        print(f"그래프 생성 완료: {self.graph.number_of_nodes()}개의 노드, {edge_count}개의 엣지 (최소 co-occurrence: {self.min_cooccurrence})")
        degrees = dict(self.graph.degree())
        top_skills = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n상위 10개 연결 스킬:")
        for skill, degree in top_skills:
            print(f"  {skill}: {degree}개의 연결")

    def train(self, skill_sets: List[List[str]]):
        if self.graph is None:
            self.build_graph(skill_sets)
        print("\nNode2Vec 모델 학습 시작...")
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
        min_count_for_vocab = max(1, self.min_skill_freq // 2)
        self.model = Word2Vec(
            vector_size=self.dimensions,
            window=10,
            min_count=min_count_for_vocab,
            batch_words=10000,
            workers=self.workers,
            sg=1,
            hs=0,
            negative=5,
            ns_exponent=0.75,
            alpha=0.025,
            min_alpha=0.0001,
            epochs=10
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

    def get_similar_skills(self, skill: str, top_n: int = 10) -> List[Tuple[str, float]]:
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train() 메서드를 먼저 호출하세요.")
        if skill not in self.model.wv:
            print(f"경고: '{skill}' 스킬이 모델에 존재하지 않습니다.")
            similar_names = [s for s in self.model.wv.index_to_key if skill.lower() in s.lower() or s.lower() in skill.lower()]
            if similar_names:
                print(f"유사한 스킬 이름 후보: {similar_names[:5]}")
            return []
        similar_skills = self.model.wv.most_similar(skill, topn=top_n)
        return similar_skills

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
            'num_walks': self.num_walks
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"모델이 {filepath}에 저장되었습니다.")

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
    # 모든 _jobs.json 파일 읽기 - 프로젝트 루트를 기준으로 상대 경로 사용
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / 'data'
    pattern = data_dir / '*_jobs.json'
    print("pattern: ", pattern)
    json_files = [str(path) for path in data_dir.glob('*_jobs.json')]

    print("=== 학습에 사용될 _jobs.json 파일 목록 ===")
    for f in json_files:
        print(f" - {f}")
    print(f"총 {len(json_files)}개 파일을 찾았습니다.")

    # 모델 초기화 (최적화된 파라미터)
    model = SkillAssociationModel(
        dimensions=128,  # 임베딩 차원 (더 높이면 정확도 향상 가능하나 메모리 증가)
        walk_length=30,  # 랜덤 워크 길이
        num_walks=200,  # 각 노드당 랜덤 워크 횟수
        workers=6,  # 병렬 처리 워커 수
        min_skill_freq=3,  # 최소 스킬 빈도 (노이즈 제거)
        min_cooccurrence=3,  # 최소 co-occurrence (약한 연결 제거)
        p=1.0,  # return 파라미터 (1.0 = 균형잡힌 탐색)
        q=1.0   # in-out 파라미터 (1.0 = 균형잡힌 탐색, <1.0 = DFS, >1.0 = BFS)
    )

    # 데이터 로드 및 학습
    print("=" * 60)
    print("스킬 연관성 학습 모델")
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

    # 테스트: Java와 유사한 스킬 검색
    print("\n" + "=" * 60)
    print("테스트: Java와 유사한 스킬")
    print("=" * 60)
    similar_skills = model.get_similar_skills('Java', top_n=10)
    for skill, score in similar_skills:
        print(f"  {skill}: {score:.4f}")

    # 테스트: Python과 유사한 스킬 검색
    print("\n" + "=" * 60)
    print("테스트: Python과 유사한 스킬")
    print("=" * 60)
    similar_skills = model.get_similar_skills('Python', top_n=10)
    for skill, score in similar_skills:
        print(f"  {skill}: {score:.4f}")

    # 테스트: 여러 스킬 컨텍스트 기반 추천
    print("\n" + "=" * 60)
    print("테스트: Java, Spring Boot 컨텍스트 기반 추천")
    print("=" * 60)
    context_skills = model.get_skills_by_context(['Java', 'Spring Boot'], top_n=10)
    for skill, score in context_skills:
        print(f"  {skill}: {score:.4f}")

    # 테스트: Docker와 유사한 스킬 검색
    print("\n" + "=" * 60)
    print("테스트: Docker와 유사한 스킬")
    print("=" * 60)
    similar_skills = model.get_similar_skills('Docker', top_n=10)
    for skill, score in similar_skills:
        print(f"  {skill}: {score:.4f}")

    # 사용 가능한 모든 스킬 출력 (일부)
    print("\n" + "=" * 60)
    print("모델에 포함된 스킬 목록 (일부)")
    print("=" * 60)
    all_skills = model.get_all_skills()
    print(f"총 {len(all_skills)}개의 스킬")
    print("스킬 샘플:", all_skills[:20])


if __name__ == "__main__":
    main()