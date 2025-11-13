"""
유사 채용공고 추천 시스템 v2 (Two-Stage Pipeline)

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    1단계: First Stage Filter                 │
│  (Embedding + Community + Jaccard로 후보군 추출)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
            Top-N 후보군 (예: 100개)
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              2단계: PageRank ReRanker                        │
│  (PPR 순위 기반 최종 정렬, 0~1 정규화)                         │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
                최종 Top-K 결과

수학적 근거:
1. First Stage (Hybrid Filtering):
   - Embedding: cos(θ) = (A·B) / (||A|| × ||B||)
   - Community: Modularity Q = 1/(2m) Σ[A_ij - k_i·k_j/(2m)]·δ(c_i, c_j)
   - Jaccard: J(A,B) = |A∩B| / |A∪B|
   - Combined: score = α·emb + β·comm + γ·jacc

2. Second Stage (PageRank ReRanking):
   - PPR: π = (1-α)M^T π + α·v
   - Rank Normalization: rank_score = (max_rank - rank) / (max_rank - 1)
   - 순위가 높을수록 1.0에 가까움, 낮을수록 0.0에 가까움
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field

import numpy as np
import networkx as nx
from node2vec import Node2Vec
from community import community_louvain


# ============================================================================
# Data Classes (기존 유지)
# ============================================================================

@dataclass
class JobPosting:
    """채용공고 데이터 클래스"""
    posting_id: str
    company: str
    title: str
    url: str
    skills: List[str]
    job_category: str = ""

    def __hash__(self):
        return hash(self.posting_id)


@dataclass
class SimilarityScore:
    """유사도 점수 상세 (2단계 파이프라인 버전)"""
    posting_id: str
    final_score: float  # 최종 점수 (PPR 순위 기반)

    # 1단계 점수들
    first_stage_score: float = 0.0
    embedding_score: float = 0.0
    community_score: float = 0.0
    jaccard_score: float = 0.0

    # 2단계 점수
    pagerank_rank_score: float = 0.0  # 순위 기반 점수 (0~1)
    pagerank_raw_score: float = 0.0   # 원본 PPR 확률값 (참고용)

    reason: str = ""


# ============================================================================
# Graph Infrastructure (기존 유지)
# ============================================================================

class JobPostingGraph:
    """채용공고 그래프 관리 클래스"""

    def __init__(self):
        self.G = nx.Graph()
        self.postings: Dict[str, JobPosting] = {}
        self.communities: Dict[str, int] = {}
        self.community_profiles: Dict[int, Dict] = {}

    def add_posting(self, posting: JobPosting):
        """채용공고를 그래프에 추가"""
        posting_node = f"posting:{posting.posting_id}"
        self.postings[posting_node] = posting
        self.G.add_node(posting_node, type='posting')

        # 회사 노드 연결
        if posting.company:
            company_node = f"company:{posting.company}"
            self.G.add_edge(posting_node, company_node, weight=1.0)

        # 직무 카테고리 연결
        if posting.job_category:
            category_node = f"category:{posting.job_category}"
            self.G.add_edge(posting_node, category_node, weight=1.0)

        # 스킬 노드 연결
        for skill in posting.skills:
            skill_normalized = self._normalize_skill(skill)
            if skill_normalized:
                skill_node = f"skill:{skill_normalized}"
                self.G.add_edge(posting_node, skill_node, weight=1.0)

    def build_skill_cooccurrence(self, min_cooccur: int = 2):
        """스킬 간 동시 출현 엣지 생성"""
        skill_pairs = Counter()

        for posting in self.postings.values():
            skills = [f"skill:{self._normalize_skill(s)}" for s in posting.skills]
            for i, skill1 in enumerate(skills):
                for skill2 in skills[i+1:]:
                    pair = tuple(sorted([skill1, skill2]))
                    skill_pairs[pair] += 1

        for (skill1, skill2), count in skill_pairs.items():
            if count >= min_cooccur:
                self.G.add_edge(skill1, skill2, weight=count)

    def detect_communities(self):
        """Louvain 알고리즘으로 커뮤니티 탐지"""
        posting_nodes = [n for n in self.G.nodes() if n.startswith('posting:')]
        posting_graph = nx.Graph()
        posting_graph.add_nodes_from(posting_nodes)

        # posting 간 간접 연결을 직접 연결로 변환
        for i, posting1 in enumerate(posting_nodes):
            posting1_skills = set(n for n in self.G.neighbors(posting1) if n.startswith('skill:'))

            for posting2 in posting_nodes[i+1:]:
                posting2_skills = set(n for n in self.G.neighbors(posting2) if n.startswith('skill:'))
                common_skills = len(posting1_skills & posting2_skills)

                if common_skills > 0:
                    posting_graph.add_edge(posting1, posting2, weight=common_skills)

        if posting_graph.number_of_edges() == 0:
            print("  ! 경고: posting 간 연결 없음, 각 공고를 별도 커뮤니티로 처리")
            self.communities = {node: i for i, node in enumerate(posting_nodes)}
            self._build_community_profiles()
            return 0.0

        partition = community_louvain.best_partition(posting_graph, weight='weight')
        self.communities = partition
        self._build_community_profiles()

        modularity = community_louvain.modularity(partition, posting_graph, weight='weight')
        return modularity

    def _build_community_profiles(self):
        """각 커뮤니티의 대표 특징 추출"""
        community_skills = defaultdict(Counter)
        community_companies = defaultdict(Counter)

        for posting_node, comm_id in self.communities.items():
            posting = self.postings[posting_node]

            for skill in posting.skills:
                skill_norm = self._normalize_skill(skill)
                if skill_norm:
                    community_skills[comm_id][skill_norm] += 1

            if posting.company:
                community_companies[comm_id][posting.company] += 1

        for comm_id in set(self.communities.values()):
            self.community_profiles[comm_id] = {
                'top_skills': community_skills[comm_id].most_common(10),
                'top_companies': community_companies[comm_id].most_common(5),
                'size': list(self.communities.values()).count(comm_id)
            }

        # 커뮤니티 분포 출력
        comm_sizes = Counter(self.communities.values())
        print(f"\n  📊 커뮤니티 크기 분포 (상위 10개):")
        for comm_id, size in sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
            profile = self.community_profiles.get(comm_id, {})
            top_skills = [s for s, _ in profile.get('top_skills', [])[:3]]
            top_companies = [c for c, _ in profile.get('top_companies', [])[:2]]
            print(f"    커뮤니티 {comm_id}: {size:2}개 공고 | 스킬: {', '.join(top_skills)} | 회사: {', '.join(top_companies)}")

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        """스킬 정규화"""
        return skill.lower().replace('-', '').replace('_', '').replace(' ', '').strip()


# ============================================================================
# Node2Vec Embedder (기존 유지)
# ============================================================================

class Node2VecEmbedder:
    """Node2Vec 기반 노드 임베딩 학습"""

    def __init__(self, graph: nx.Graph, dimensions: int = 64,
                 walk_length: int = 30, num_walks: int = 200,
                 p: float = 1.0, q: float = 1.0):
        self.graph = graph
        self.dimensions = dimensions
        self.model = None
        self.embeddings = {}

        print(f"[Node2Vec] Random walk 생성 중... (walks={num_walks}, length={walk_length})")
        node2vec = Node2Vec(
            graph,
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=p,
            q=q,
            workers=4,
            quiet=True
        )

        print("[Node2Vec] Skip-gram 모델 학습 중...")
        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)

        for node in graph.nodes():
            try:
                self.embeddings[node] = self.model.wv[node]
            except KeyError:
                self.embeddings[node] = np.zeros(dimensions)

    def get_embedding(self, node: str) -> np.ndarray:
        """노드 임베딩 반환"""
        return self.embeddings.get(node, np.zeros(self.dimensions))

    def cosine_similarity(self, node1: str, node2: str) -> float:
        """두 노드 간 코사인 유사도"""
        emb1 = self.get_embedding(node1)
        emb2 = self.get_embedding(node2)

        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(emb1, emb2) / (norm1 * norm2)


# ============================================================================
# Stage 1: First Stage Filter (새로 추가)
# ============================================================================

class FirstStageFilter:
    """
    1차 필터링: Embedding + Community + Jaccard로 후보군 추출

    목적:
    - 전체 공고 중 상위 N개 후보군만 추출하여 계산 효율성 향상
    - 3가지 지표의 하이브리드 점수로 다양한 관점에서 유사도 평가

    가중치:
    - alpha: Embedding 가중치 (의미론적 유사도)
    - beta: Community 가중치 (클러스터 유사도)
    - gamma: Jaccard 가중치 (직접 스킬 매칭)
    """

    def __init__(self, graph: JobPostingGraph, embedder: Node2VecEmbedder,
                 alpha: float = 0.33, beta: float = 0.33, gamma: float = 0.34):
        self.graph = graph
        self.embedder = embedder
        self.alpha = alpha  # Embedding 가중치
        self.beta = beta    # Community 가중치
        self.gamma = gamma  # Jaccard 가중치

    def filter_candidates(self, query_posting: JobPosting, top_n: int = 100) -> List[Tuple[str, float, Dict]]:
        """
        1차 후보군 추출

        Returns:
            List of (posting_id, first_stage_score, score_details)
            score_details = {
                'embedding': float,
                'community': float,
                'jaccard': float
            }
        """
        candidates = []

        for posting_node, target_posting in self.graph.postings.items():
            # 자기 자신 제외
            if query_posting.posting_id == target_posting.posting_id:
                continue

            # 3가지 유사도 계산
            embedding_similarity = self._calculate_embedding_similarity(query_posting, target_posting)
            community_similarity = self._calculate_community_similarity(query_posting, target_posting)
            jaccard_similarity = self._calculate_jaccard_similarity(query_posting.skills, target_posting.skills)

            # 하이브리드 점수
            first_stage_combined_score = (
                self.alpha * embedding_similarity +
                self.beta * community_similarity +
                self.gamma * jaccard_similarity
            )

            score_details = {
                'embedding': embedding_similarity,
                'community': community_similarity,
                'jaccard': jaccard_similarity
            }

            candidates.append((target_posting.posting_id, first_stage_combined_score, score_details))

        # Top-N 후보군만 선택
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_n_candidates = candidates[:top_n]

        print(f"\n[1차 필터링] 전체 {len(candidates)}개 중 상위 {len(top_n_candidates)}개 후보 추출")
        if top_n_candidates:
            print(f"  - 최고 점수: {top_n_candidates[0][1]:.4f}")
            print(f"  - 최저 점수: {top_n_candidates[-1][1]:.4f}")

        return top_n_candidates

    def _calculate_embedding_similarity(self, query: JobPosting, target: JobPosting) -> float:
        """Node2Vec 임베딩 기반 유사도"""
        if not self.embedder:
            return 0.0

        query_skill_embeddings = [
            self.embedder.get_embedding(f"skill:{self.graph._normalize_skill(s)}")
            for s in query.skills
        ]
        target_skill_embeddings = [
            self.embedder.get_embedding(f"skill:{self.graph._normalize_skill(s)}")
            for s in target.skills
        ]

        if not query_skill_embeddings or not target_skill_embeddings:
            return 0.0

        query_avg_embedding = np.mean(query_skill_embeddings, axis=0)
        target_avg_embedding = np.mean(target_skill_embeddings, axis=0)

        norm_query = np.linalg.norm(query_avg_embedding)
        norm_target = np.linalg.norm(target_avg_embedding)

        if norm_query == 0 or norm_target == 0:
            return 0.0

        return np.dot(query_avg_embedding, target_avg_embedding) / (norm_query * norm_target)

    def _calculate_community_similarity(self, query: JobPosting, target: JobPosting) -> float:
        """커뮤니티 유사도"""
        query_skills_normalized = set(self.graph._normalize_skill(s) for s in query.skills)
        target_skills_normalized = set(self.graph._normalize_skill(s) for s in target.skills)

        # 각 커뮤니티와 쿼리의 유사도 계산
        community_match_scores = {}
        for comm_id, profile in self.graph.community_profiles.items():
            comm_top_skills = set(skill for skill, _ in profile['top_skills'])
            if comm_top_skills:
                similarity = len(query_skills_normalized & comm_top_skills) / len(query_skills_normalized | comm_top_skills)
                community_match_scores[comm_id] = similarity

        target_node = f"posting:{target.posting_id}"
        target_community_id = self.graph.communities.get(target_node)

        if target_community_id is None:
            return 0.0

        community_level_score = community_match_scores.get(target_community_id, 0.0)

        # 타겟과 쿼리의 직접 스킬 유사도
        if query_skills_normalized and target_skills_normalized:
            direct_skill_similarity = len(query_skills_normalized & target_skills_normalized) / len(query_skills_normalized | target_skills_normalized)
        else:
            direct_skill_similarity = 0.0

        # 하이브리드: 50% 커뮤니티 레벨 + 50% 직접 유사도
        return 0.5 * community_level_score + 0.5 * direct_skill_similarity

    def _calculate_jaccard_similarity(self, skills1: List[str], skills2: List[str]) -> float:
        """Jaccard 유사도"""
        set1 = set(self.graph._normalize_skill(s) for s in skills1)
        set2 = set(self.graph._normalize_skill(s) for s in skills2)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0


# ============================================================================
# Stage 2: PageRank ReRanker (새로 추가)
# ============================================================================

class PageRankReRanker:
    """
    2차 리랭킹: Personalized PageRank 기반 최종 순위 결정

    핵심 변경사항:
    - 기존: PPR 확률값 직접 사용
    - 변경: PPR 점수 → 순위 변환 → 0~1 정규화

    정규화 공식:
    - rank_score = (max_rank - current_rank) / (max_rank - 1)
    - 1등: (N-1) / (N-1) = 1.0
    - 2등: (N-2) / (N-1) = 0.9x
    - 꼴등: (N-N) / (N-1) = 0.0

    효과:
    - PPR 확률값의 편차가 작아도 순위는 명확히 구분
    - 최종 점수가 직관적 (1등=1.0, 꼴등=0.0)
    """

    def __init__(self, graph: JobPostingGraph):
        self.graph = graph

    def rerank_with_pagerank(self, query_posting: JobPosting,
                            first_stage_candidates: List[Tuple[str, float, Dict]]) -> List[SimilarityScore]:
        """
        PPR 기반 리랭킹

        Args:
            query_posting: 쿼리 공고
            first_stage_candidates: 1차 필터링 결과 [(posting_id, score, details), ...]

        Returns:
            List[SimilarityScore]: 최종 순위 결과
        """
        if not first_stage_candidates:
            return []

        # 1. 모든 후보에 대해 PPR 점수 계산
        candidate_ids = [posting_id for posting_id, _, _ in first_stage_candidates]
        ppr_scores = self._calculate_ppr_scores(query_posting, candidate_ids)

        # 2. PPR 점수 기준 정렬하여 순위 부여
        sorted_ppr = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)

        # 3. 순위를 0~1로 정규화
        rank_scores = self._normalize_ranks(sorted_ppr)

        # 4. 최종 SimilarityScore 객체 생성
        final_results = []
        first_stage_dict = {posting_id: (score, details) for posting_id, score, details in first_stage_candidates}

        for posting_id, ppr_raw_score in sorted_ppr:
            first_stage_score, score_details = first_stage_dict[posting_id]
            rank_score = rank_scores[posting_id]

            similarity_score = SimilarityScore(
                posting_id=posting_id,
                final_score=rank_score,  # 최종 점수는 순위 기반 점수
                first_stage_score=first_stage_score,
                embedding_score=score_details['embedding'],
                community_score=score_details['community'],
                jaccard_score=score_details['jaccard'],
                pagerank_rank_score=rank_score,
                pagerank_raw_score=ppr_raw_score,
                reason=self._generate_reason(query_posting, posting_id)
            )

            final_results.append(similarity_score)

        print(f"\n[2차 리랭킹] PPR 순위 기반 정렬 완료")
        if final_results:
            print(f"  - 1등 PPR 순위 점수: {final_results[0].pagerank_rank_score:.4f} (원본: {final_results[0].pagerank_raw_score:.6f})")
            print(f"  - 꼴등 PPR 순위 점수: {final_results[-1].pagerank_rank_score:.4f} (원본: {final_results[-1].pagerank_raw_score:.6f})")

        return final_results

    def _calculate_ppr_scores(self, query: JobPosting, candidate_ids: List[str]) -> Dict[str, float]:
        """
        Personalized PageRank 점수 계산

        Returns:
            Dict[posting_id, ppr_score]
        """
        # Personalization vector: 쿼리 스킬에만 확률 부여
        personalization = {}
        query_skill_nodes = [f"skill:{self.graph._normalize_skill(s)}" for s in query.skills]

        for node in self.graph.G.nodes():
            if node in query_skill_nodes:
                personalization[node] = 1.0 / len(query_skill_nodes)
            else:
                personalization[node] = 0.0

        try:
            ppr_results = nx.pagerank(self.graph.G, personalization=personalization,
                                     alpha=0.85, max_iter=100, weight='weight')

            # 후보군의 PPR 점수만 추출
            ppr_scores = {}
            for posting_id in candidate_ids:
                posting_node = f"posting:{posting_id}"
                ppr_scores[posting_id] = ppr_results.get(posting_node, 0.0)

            return ppr_scores

        except Exception as e:
            print(f"  ! PPR 계산 실패: {e}")
            return {posting_id: 0.0 for posting_id in candidate_ids}

    def _normalize_ranks(self, sorted_ppr: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        순위를 0~1로 정규화

        Args:
            sorted_ppr: PPR 점수 기준 정렬된 [(posting_id, ppr_score), ...]

        Returns:
            Dict[posting_id, normalized_rank_score]

        정규화 공식:
            rank_score = (max_rank - current_rank) / (max_rank - 1)
            - 1등 (rank=0): (N-1-0) / (N-1) = 1.0
            - 2등 (rank=1): (N-1-1) / (N-1) = (N-2)/(N-1)
            - 꼴등 (rank=N-1): (N-1-(N-1)) / (N-1) = 0.0
        """
        num_candidates = len(sorted_ppr)

        if num_candidates <= 1:
            return {sorted_ppr[0][0]: 1.0} if sorted_ppr else {}

        rank_scores = {}
        for rank, (posting_id, _) in enumerate(sorted_ppr):
            # 순위를 0~1로 정규화 (1등=1.0, 꼴등=0.0)
            normalized_score = (num_candidates - 1 - rank) / (num_candidates - 1)
            rank_scores[posting_id] = normalized_score

        return rank_scores

    def _generate_reason(self, query: JobPosting, target_posting_id: str) -> str:
        """추천 이유 생성"""
        target_node = f"posting:{target_posting_id}"
        target_posting = self.graph.postings.get(target_node)

        if not target_posting:
            return "그래프 구조적 유사성"

        query_skills_normalized = set(self.graph._normalize_skill(s) for s in query.skills)
        target_skills_normalized = set(self.graph._normalize_skill(s) for s in target_posting.skills)

        common_skills = query_skills_normalized & target_skills_normalized

        if len(common_skills) >= 3:
            return f"공통 스킬 {len(common_skills)}개: {', '.join(list(common_skills)[:3])}"
        elif len(common_skills) > 0:
            return f"공통 스킬: {', '.join(common_skills)}"
        else:
            return "그래프 구조적 유사성"


# ============================================================================
# Main Recommender: Two-Stage Pipeline (새로 추가)
# ============================================================================

class TwoStageJobRecommender:
    """
    2단계 파이프라인 추천 시스템

    Pipeline:
    1. FirstStageFilter: Embedding + Community + Jaccard로 상위 N개 후보 추출
    2. PageRankReRanker: PPR 순위 기반 최종 정렬

    장점:
    - 계산 효율성: 전체가 아닌 상위 후보군만 PPR 계산
    - 정확도 향상: 다양한 지표로 1차 필터링 후 구조적 유사도로 최종 결정
    - 순위 명확화: PPR 확률값이 아닌 순위로 직관적 비교
    """

    def __init__(self):
        self.graph = JobPostingGraph()
        self.embedder: Optional[Node2VecEmbedder] = None
        self.first_stage_filter: Optional[FirstStageFilter] = None
        self.pagerank_reranker: Optional[PageRankReRanker] = None

    def _extract_posting_id(self, url: str, title: str, filepath: str, idx: int) -> str:
        """고유 ID 추출"""
        import hashlib
        from urllib.parse import urlparse, parse_qs

        company_prefix = Path(filepath).stem.split('_')[0]

        # URL에서 ID 파라미터 추출
        if url:
            try:
                parsed = urlparse(url)
                params = parse_qs(parsed.query)

                for param_name in ['rtSeq', 'job_id', 'id', 'recruitNo', 'jobNo', 'no']:
                    if param_name in params:
                        return f"{company_prefix}_{params[param_name][0]}"
            except:
                pass

        # URL + title 해시
        if url and title:
            hash_input = f"{url}:{title}".encode('utf-8')
            hash_digest = hashlib.md5(hash_input).hexdigest()[:8]
            return f"{company_prefix}_{hash_digest}"

        # Fallback
        return f"{company_prefix}_jobs_{idx}"

    def load_data(self, job_files: List[str]):
        """채용공고 데이터 로드"""
        for filepath in job_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for idx, job in enumerate(data):
                    skills = []
                    skill_info = job.get('skill_set_info', {})
                    if isinstance(skill_info, dict):
                        skill_set = skill_info.get('skill_set', [])
                        if isinstance(skill_set, list):
                            skills = skill_set

                    url = job.get('url', '')
                    title = job.get('title', '')
                    posting_id = self._extract_posting_id(url, title, filepath, idx)

                    posting = JobPosting(
                        posting_id=posting_id,
                        company=job.get('company', 'Unknown'),
                        title=title,
                        url=url,
                        skills=skills,
                        job_category=job.get('job_category', '')
                    )

                    self.graph.add_posting(posting)

                print(f"✓ {Path(filepath).name}: {len(data)}개 로드")
            except Exception as e:
                print(f"✗ {filepath} 로드 실패: {e}")

    def build_graph(self, use_skill_cooccurrence: bool = True):
        """그래프 구축 및 커뮤니티 탐지"""
        print("\n[그래프 구축]")
        print(f"  노드: {self.graph.G.number_of_nodes()}개")
        print(f"  엣지: {self.graph.G.number_of_edges()}개")

        if use_skill_cooccurrence:
            print("\n[스킬 동시 출현 엣지 추가]")
            self.graph.build_skill_cooccurrence(min_cooccur=2)
            print(f"  엣지: {self.graph.G.number_of_edges()}개 (업데이트)")

        print("\n[커뮤니티 탐지]")
        modularity = self.graph.detect_communities()
        num_communities = len(set(self.graph.communities.values()))
        print(f"  커뮤니티 수: {num_communities}개")
        print(f"  Modularity: {modularity:.4f}")

    def train_embeddings(self, dimensions: int = 64):
        """Node2Vec 임베딩 학습"""
        print("\n[Node2Vec 임베딩 학습]")
        self.embedder = Node2VecEmbedder(
            self.graph.G,
            dimensions=dimensions,
            walk_length=30,
            num_walks=200,
            p=1.0,
            q=1.0
        )
        print(f"✓ 임베딩 학습 완료 (dim={dimensions})")

        # 컴포넌트 초기화
        self.first_stage_filter = FirstStageFilter(self.graph, self.embedder)
        self.pagerank_reranker = PageRankReRanker(self.graph)

    def recommend_similar_jobs(self, query_posting: JobPosting,
                               first_stage_top_n: int = 100,
                               final_top_k: int = 10) -> List[SimilarityScore]:
        """
        2단계 파이프라인 추천

        Args:
            query_posting: 쿼리 채용공고
            first_stage_top_n: 1차 필터링 후보군 크기 (기본 100개)
            final_top_k: 최종 추천 결과 개수 (기본 10개)

        Returns:
            List[SimilarityScore]: 최종 추천 결과 (순위순 정렬)
        """
        if not self.first_stage_filter or not self.pagerank_reranker:
            raise ValueError("train_embeddings()를 먼저 실행해야 합니다.")

        print("\n" + "="*60)
        print("2단계 파이프라인 추천 시작")
        print("="*60)

        # Stage 1: First Stage Filtering
        print("\n[Stage 1] Embedding + Community + Jaccard 필터링")
        first_stage_candidates = self.first_stage_filter.filter_candidates(
            query_posting,
            top_n=first_stage_top_n
        )

        # Stage 2: PageRank ReRanking
        print("\n[Stage 2] PageRank 순위 기반 리랭킹")
        final_ranked_results = self.pagerank_reranker.rerank_with_pagerank(
            query_posting,
            first_stage_candidates
        )

        # 최종 Top-K만 반환
        return final_ranked_results[:final_top_k]

    def save_model(self, filepath: str):
        """모델 저장"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.graph,
                'embedder': self.embedder
            }, f)
        print(f"✓ 모델 저장: {filepath}")

    def load_model(self, filepath: str):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.embedder = data['embedder']

            # 컴포넌트 재초기화
            self.first_stage_filter = FirstStageFilter(self.graph, self.embedder)
            self.pagerank_reranker = PageRankReRanker(self.graph)
        print(f"✓ 모델 로드: {filepath}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """메인 실행"""
    print("="*60)
    print("유사 채용공고 추천 시스템 v2 (Two-Stage Pipeline)")
    print("="*60)

    # 1. 데이터 로드
    recommender = TwoStageJobRecommender()

    script_dir = Path(__file__).parent
    job_files = [
        str(script_dir / 'data/hanwha_jobs.json'),
        str(script_dir / 'data/kakao_jobs.json'),
        str(script_dir / 'data/line_jobs.json'),
        str(script_dir / 'data/naver_jobs.json')
    ]

    print("\n[1/4] 데이터 로드")
    recommender.load_data(job_files)

    # 2. 그래프 구축
    print("\n[2/4] 그래프 구축")
    recommender.build_graph(use_skill_cooccurrence=True)

    # 3. 임베딩 학습
    print("\n[3/4] 임베딩 학습")
    recommender.train_embeddings(dimensions=64)

    # 4. 테스트 쿼리
    print("\n[4/4] 유사 공고 검색 테스트")
    test_query = JobPosting(
        posting_id="test_query",
        company="테스트 회사",
        title="AI 엔지니어",
        url="",
        skills=["Python", "TensorFlow", "PyTorch", "Machine Learning"],
        job_category="AI"
    )

    results = recommender.recommend_similar_jobs(
        test_query,
        first_stage_top_n=100,  # 1차 필터링: 상위 100개 후보
        final_top_k=10          # 최종 결과: Top 10
    )

    print(f"\n쿼리: {test_query.title} ({', '.join(test_query.skills)})")
    print("\n최종 추천 결과 (PPR 순위 기반):")
    print("-" * 80)

    for i, score in enumerate(results, 1):
        posting = recommender.graph.postings[f"posting:{score.posting_id}"]
        print(f"{i}. {posting.company} - {posting.title}")
        print(f"   [최종 점수] {score.final_score:.4f} (PPR 순위 기반)")
        print(f"   [1차 점수] {score.first_stage_score:.4f} "
              f"(Emb: {score.embedding_score:.4f}, "
              f"Comm: {score.community_score:.4f}, "
              f"Jacc: {score.jaccard_score:.4f})")
        print(f"   [2차 점수] PPR 순위: {score.pagerank_rank_score:.4f} "
              f"(원본 확률: {score.pagerank_raw_score:.6f})")
        print(f"   [스킬] {', '.join(posting.skills[:5])}")
        print(f"   [이유] {score.reason}")
        print()

    # 모델 저장
    print("\n[모델 저장]")
    output_dir = script_dir / 'output'
    output_dir.mkdir(exist_ok=True)
    recommender.save_model(str(output_dir / 'two_stage_job_recommender.pkl'))
    print("\n완료!")


if __name__ == '__main__':
    main()
