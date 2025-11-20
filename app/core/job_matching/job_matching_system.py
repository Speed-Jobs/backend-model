"""
직무 매칭 시스템 v7 - SBERT DESCRIPTION MATCHING

주요 개선사항:
- Sentence-BERT 임베딩으로 description vs description 의미 유사도 매칭 추가
- (변경) 제목 + 본문 전체를 합쳐서 SBERT 쿼리 텍스트로 사용
- Description 유사도를 최종 점수에 45% 반영 (가장 높은 가중치)
- 기존 BM25 대신 SBERT cosine similarity 사용

점수 구성:
- 15% PPR (구조적 유사도)
- 25% Jaccard (스킬 직접 매칭)
- 15% Cluster (커뮤니티 유사도)
- 45% SBERT (Description 유사도)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import networkx as nx
from community import community_louvain
from sentence_transformers import SentenceTransformer

from app.core.job_matching.config import (
    JOB_DESCRIPTION_FILE,
    SBERT_MODEL_NAME,
    TRAINING_DATA_FILES,
)

# ============================================================================
# Output Logger (Terminal + File)
# ============================================================================

class OutputLogger:
    """Terminal과 파일에 동시 출력"""

    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class JobDescription:
    """직무 정의 (new_job_description.json)"""
    job_name: str
    job_definition: str
    industry: str
    common_skills: List[str]
    specific_skills: List[str]
    all_skills: List[str] = field(default_factory=list)
    skill_set_description: str = ""  # 주요 업무 설명 (SBERT에 사용)

    def __post_init__(self):
        self.all_skills = list(set(self.common_skills + self.specific_skills))

@dataclass
class NewJobPosting:
    """새로운 채용공고 (매칭 대상)"""
    posting_id: str
    company: str
    title: str
    skills: List[str]
    url: str = ""
    description: str = ""

@dataclass
class JobPosting:
    """기존 채용공고 (학습 데이터)"""
    posting_id: str
    company: str
    title: str
    url: str
    skills: List[str]

    def __hash__(self):
        return hash(self.posting_id)


@dataclass
class JobMatchResult:
    """직무 매칭 결과"""
    job_name: str
    industry: str
    final_score: float
    
    jaccard_score: float = 0.0
    cluster_score: float = 0.0
    pagerank_score: float = 0.0
    sbert_score: float = 0.0  # SBERT 유사도 점수
    
    matching_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    job_definition: str = ""
    
    reason: str = ""

# ============================================================================
# Graph Infrastructure
# ============================================================================

class JobPostingGraph:
    """채용공고 그래프"""

    def __init__(self):
        self.G = nx.Graph()
        self.postings: Dict[str, JobPosting] = {}

    def add_posting(self, posting: JobPosting):
        posting_node = f"posting:{posting.posting_id}"
        self.postings[posting_node] = posting
        self.G.add_node(posting_node, type='posting')

        if posting.company:
            company_node = f"company:{posting.company}"
            self.G.add_edge(posting_node, company_node, weight=1.0)

        for skill in posting.skills:
            skill_normalized = self._normalize_skill(skill)
            if skill_normalized:
                skill_node = f"skill:{skill_normalized}"
                self.G.add_edge(posting_node, skill_node, weight=1.0)

    def build_skill_cooccurrence(self, min_cooccur: int = 2):
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

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        return (
            skill.lower()
            .replace('-', '')
            .replace('_', '')
            .replace(' ', '')
            .replace('.', '')
            .strip()
        )


# ============================================================================
# SBERT Description Matcher
# ============================================================================

class SbertDescriptionMatcher:
    """
    Sentence-BERT로 description 의미 유사도 계산
    - 직무 정의 텍스트들을 임베딩해두고
    - 새 공고의 (제목 + 본문 전체) 임베딩과 cosine similarity 계산
    """

    def __init__(
        self,
        job_descriptions: List[JobDescription],
        model_name: str = None,
    ):
        self.job_descriptions = job_descriptions

        # config에서 모델명 가져오기 (없으면 기본값 사용)
        if model_name is None:
            model_name = SBERT_MODEL_NAME

        print(f"[SBERT] 모델 로딩 중... ({model_name})")
        self.model = SentenceTransformer(model_name)

        print(f"[SBERT] 직무 definition 임베딩 생성 중... (직무 정의 + industry + skill_set_description)")
        corpus = []
        
        for jd in job_descriptions:
            parts = []
            
            # 1. 직무 정의 (기존)
            if jd.job_definition:
                parts.append(jd.job_definition)
            
            # 2. Industry 추가 (Front-end vs Back-end 구분에 필수!)
            if jd.industry:
                parts.append(f"산업 분야: {jd.industry}")
            
            # 3. Skill Set Description 추가 (구체적인 업무 설명)
            if jd.skill_set_description:
                parts.append(f"주요 업무: {jd.skill_set_description}")
            
            # 모든 정보 결합 (정보가 없으면 job_name만 사용)
            combined_text = "\n\n".join(parts) if parts else jd.job_name
            corpus.append(combined_text)

        # normalize_embeddings=True → 코사인 유사도 = 내적
        self.job_embeddings = self.model.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        print(f"[OK] {len(job_descriptions)}개 직무 정의 임베딩 완료 (industry + skill_set_description 포함)")

    def calculate_similarity(self, query_text: str) -> Dict[str, float]:
        """
        새 공고의 쿼리 텍스트(제목+본문)와 모든 직무 정의의 의미 유사도 계산 (0~1)

        Returns:
            Dict[job_name, normalized_score]
        """
        if not query_text or not query_text.strip():
            return {jd.job_name: 0.0 for jd in self.job_descriptions}

        # 쿼리 임베딩
        query_emb = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # cosine similarity (normalized embeddings → dot product)
        sims = np.dot(self.job_embeddings, query_emb)  # [-1, 1]

        # 1차 변환: [-1, 1] → [0, 1]
        sims = (sims + 1.0) / 2.0

        # 2차 정규화: 최댓값 기준으로 0~1
        max_sim = sims.max() if sims.size > 0 else 1.0
        if max_sim > 0:
            sims = sims / max_sim

        result = {}
        for i, jd in enumerate(self.job_descriptions):
            result[jd.job_name] = float(sims[i])

        return result


# ============================================================================
# Cluster Matcher (Louvain 기반)
# ============================================================================

class ClusterMatcher:
    """Louvain 클러스터링 기반 매칭"""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.clusters = {}
        self.cluster_skills = defaultdict(list)
        
        print(f"[Louvain] 커뮤니티 탐지 중...")
        
        # Louvain 클러스터링
        self.clusters = community_louvain.best_partition(graph, weight='weight')
        
        # 클러스터별 스킬 노드 정리
        for node, cluster_id in self.clusters.items():
            if node.startswith('skill:'):
                self.cluster_skills[cluster_id].append(node)
        
        num_clusters = len(set(self.clusters.values()))
        num_skill_nodes = len([n for n in graph.nodes() if n.startswith('skill:')])
        
        print(f"[OK] {num_clusters}개 클러스터 탐지 (스킬 노드: {num_skill_nodes}개)")

    def get_cluster_distribution(self, skills: List[str], normalize_func) -> np.ndarray:
        """
        스킬 리스트의 클러스터 분포 계산
        """
        skill_nodes = [f"skill:{normalize_func(s)}" for s in skills]
        
        cluster_counts = Counter()
        for skill_node in skill_nodes:
            if skill_node in self.clusters:
                cluster_id = self.clusters[skill_node]
                cluster_counts[cluster_id] += 1
        
        num_clusters = max(self.clusters.values()) + 1
        distribution = np.zeros(num_clusters)
        
        for cluster_id, count in cluster_counts.items():
            distribution[cluster_id] = count
        
        total = distribution.sum()
        if total > 0:
            distribution = distribution / total
        
        return distribution
    
    def calculate_similarity(self, skills1: List[str], skills2: List[str], normalize_func) -> float:
        """
        두 스킬 집합 간 클러스터 유사도 계산 (코사인 유사도)
        """
        dist1 = self.get_cluster_distribution(skills1, normalize_func)
        dist2 = self.get_cluster_distribution(skills2, normalize_func)
        
        norm1 = np.linalg.norm(dist1)
        norm2 = np.linalg.norm(dist2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(dist1, dist2) / (norm1 * norm2)
        return similarity


# ============================================================================
# Job Matcher (핵심 로직)
# ============================================================================

class JobMatcher:
    """새 채용공고 → 직무 매칭"""

    def __init__(
        self,
        graph: JobPostingGraph,
        cluster_matcher: ClusterMatcher,
        sbert_matcher: SbertDescriptionMatcher,
        job_descriptions: List[JobDescription],
    ):
        self.graph = graph
        self.cluster_matcher = cluster_matcher
        self.sbert_matcher = sbert_matcher
        self.job_descriptions = job_descriptions

    def match_job(
        self,
        new_posting: NewJobPosting,
        ppr_top_n: int = 20,
        final_top_k: int = 2,
    ) -> List[JobMatchResult]:
        """
        새 채용공고를 직무와 매칭
        
        3단계 파이프라인:
        1. PPR로 상위 20개 직무 추출 (구조적 유사도)
        2. SBERT로 (제목+본문) 의미 유사도 계산
        3. 스킬 매칭 필터링 + 최종 Top 1~2 선정
        """
        print(f"\n[Stage 1] PPR 기반 1차 필터링 (상위 {ppr_top_n}개 추출)")
        
        # Stage 1: PPR로 직무별 점수 계산 및 상위 N개 추출
        ppr_candidates = self._get_ppr_top_jobs(new_posting, ppr_top_n)
        
        if not ppr_candidates:
            print("  ! PPR 후보 없음, 전체 직무 대상으로 진행")
            ppr_candidates = [(jd, 0.0) for jd in self.job_descriptions]
        
        print(f"  [OK] {len(ppr_candidates)} jobs selected")
        
        # ---------- 🔧 변경 포인트 1: SBERT 쿼리 텍스트 구성 (title + description) ----------
        query_text = f"{new_posting.title}\n\n{new_posting.description}".strip()
        
        # Stage 1.5: SBERT description 유사도 계산 (전체 직무 대상)
        print(f"\n[Stage 1.5] SBERT Description 유사도 계산")
        sbert_scores = self.sbert_matcher.calculate_similarity(query_text)
        
        if query_text:
            top_sbert_job = max(sbert_scores.items(), key=lambda x: x[1])
            print(f"  - SBERT 1등: {top_sbert_job[0]} (점수: {top_sbert_job[1]:.4f})")
        else:
            print(f"  ! Description/Title 없음 - SBERT 점수 모두 0")
        
        # Stage 2: 선정된 후보들에 대해서만 스킬 매칭
        print(f"\n[Stage 2] 스킬 매칭 (Jaccard + Cluster) + 필터링")
        
        results = []
        filtered_count = 0
        
        for job_desc, ppr_score in ppr_candidates:
            # Jaccard + Cluster 계산
            jaccard = self._calculate_jaccard(new_posting.skills, job_desc.all_skills)
            cluster = self._calculate_cluster_similarity(new_posting.skills, job_desc.all_skills)
            
            # SBERT 점수
            sbert = sbert_scores.get(job_desc.job_name, 0.0)
            
            # 매칭 스킬 분석
            new_skills_norm = set(self.graph._normalize_skill(s) for s in new_posting.skills)
            job_skills_norm = set(self.graph._normalize_skill(s) for s in job_desc.all_skills)
            
            matching_skills = list(new_skills_norm & job_skills_norm)
            missing_skills = list(job_skills_norm - new_skills_norm)
            
            # 필터링: 스킬 0개 매칭이면 제외 (완전 의미 매칭만으로 추천되는 것 방지)
            if len(matching_skills) == 0 and jaccard < 0.05 and cluster < 0.2:
                filtered_count += 1
                continue
            
            # 최종 점수
            final_score = (
                0.15 * ppr_score   # 구조적 유사도
                + 0.25 * jaccard   # 스킬 직접 매칭
                + 0.15 * cluster   # 클러스터 유사도
                + 0.45 * sbert     # (제목+본문) 의미 유사도
            )
            
            result = JobMatchResult(
                job_name=job_desc.job_name,
                industry=job_desc.industry,
                final_score=final_score,
                jaccard_score=jaccard,
                cluster_score=cluster,
                pagerank_score=ppr_score,
                sbert_score=sbert,
                matching_skills=matching_skills[:10],
                missing_skills=missing_skills[:5],
                job_definition=job_desc.job_definition,
                reason=self._generate_reason(matching_skills, jaccard, ppr_score, sbert),
            )
            
            results.append(result)
        
        # 정렬 및 Top-K 반환
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        if filtered_count > 0:
            print(f"  [FILTER] {filtered_count}개 직무 제외 (스킬 매칭 부족)")
        
        print(f"  [OK] Final top {min(final_top_k, len(results))} returned")
        if results:
            print(
                "  - 1등: "
                f"{results[0].job_name}/{results[0].industry}\n"
                "         점수: "
                f"{results[0].final_score:.4f} "
                f"(PPR:{results[0].pagerank_score:.4f}, "
                f"Jacc:{results[0].jaccard_score:.4f}, "
                f"Clust:{results[0].cluster_score:.4f}, "
                f"SBERT:{results[0].sbert_score:.4f})"
            )
        else:
            print(f"  [WARNING] 매칭 가능한 직무 없음")
        
        return results[:final_top_k]
    
    def _get_ppr_top_jobs(self, new_posting: NewJobPosting, top_n: int) -> List[Tuple[JobDescription, float]]:
        """
        PPR로 상위 N개 직무 추출
        """
        try:
            personalization = {}
            new_skill_nodes = [
                f"skill:{self.graph._normalize_skill(s)}"
                for s in new_posting.skills
            ]
            
            for node in self.graph.G.nodes():
                if node in new_skill_nodes:
                    personalization[node] = 1.0 / len(new_skill_nodes)
                else:
                    personalization[node] = 0.0
            
            ppr = nx.pagerank(
                self.graph.G,
                personalization=personalization,
                alpha=0.85,
                max_iter=100,
                weight='weight',
            )
            
            job_ppr_scores = []
            
            for job_desc in self.job_descriptions:
                skill_nodes = [
                    f"skill:{self.graph._normalize_skill(s)}"
                    for s in job_desc.all_skills
                ]
                
                ppr_scores = [ppr.get(node, 0.0) for node in skill_nodes]
                avg_ppr = np.mean(ppr_scores) if ppr_scores else 0.0
                
                job_ppr_scores.append((job_desc, avg_ppr))
            
            if job_ppr_scores:
                max_ppr = max(score for _, score in job_ppr_scores)
                if max_ppr > 0:
                    job_ppr_scores = [(jd, score / max_ppr) for jd, score in job_ppr_scores]
            
            job_ppr_scores.sort(key=lambda x: x[1], reverse=True)
            top_candidates = job_ppr_scores[:top_n]
            
            if top_candidates:
                print(
                    f"  - PPR 1등: {top_candidates[0][0].job_name}/"
                    f"{top_candidates[0][0].industry} (점수: {top_candidates[0][1]:.4f})"
                )
                print(
                    f"  - PPR {len(top_candidates)}등: {top_candidates[-1][0].job_name}/"
                    f"{top_candidates[-1][0].industry} (점수: {top_candidates[-1][1]:.4f})"
                )
            
            return top_candidates
        
        except Exception as e:
            print(f"  ! PPR 계산 실패: {e}")
            return []

    def _calculate_jaccard(self, skills1: List[str], skills2: List[str]) -> float:
        set1 = set(self.graph._normalize_skill(s) for s in skills1)
        set2 = set(self.graph._normalize_skill(s) for s in skills2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0

    def _calculate_cluster_similarity(self, skills1: List[str], skills2: List[str]) -> float:
        """클러스터 기반 유사도 계산"""
        return self.cluster_matcher.calculate_similarity(
            skills1, skills2, self.graph._normalize_skill
        )

    def _generate_reason(self, matching_skills: List[str], jaccard: float, ppr: float, sbert: float) -> str:
        num_matches = len(matching_skills)
        
        if sbert > 0.5:
            return f"Description 의미 매칭 강함 (SBERT: {sbert:.3f}), 스킬 {num_matches}개"
        elif num_matches >= 5:
            return f"매칭 스킬 {num_matches}개 (PPR: {ppr:.3f}, Jacc: {jaccard:.2%})"
        elif num_matches >= 3:
            return f"매칭 스킬 {num_matches}개: {', '.join(matching_skills[:3])} (PPR: {ppr:.3f})"
        elif num_matches > 0:
            return f"매칭 스킬: {', '.join(matching_skills)} (PPR: {ppr:.3f})"
        else:
            return f"구조적 유사도 + 의미 유사도 기반 (PPR: {ppr:.3f}, SBERT: {sbert:.3f})"


# ============================================================================
# Main System
# ============================================================================

class JobMatchingSystem:
    """통합 직무 매칭 시스템"""

    def __init__(self, log_file: Optional[str] = None):
        self.graph = JobPostingGraph()
        self.cluster_matcher: Optional[ClusterMatcher] = None
        self.sbert_matcher: Optional[SbertDescriptionMatcher] = None
        self.job_descriptions: List[JobDescription] = []
        self.matcher: Optional[JobMatcher] = None

        # 로그 파일 설정
        if log_file:
            self.logger = OutputLogger(log_file)
            sys.stdout = self.logger

    def __del__(self):
        # 종료 시 로그 파일 닫기
        if hasattr(self, 'logger'):
            sys.stdout = self.logger.terminal
            self.logger.close()

    def load_job_descriptions(self, filepath: str = None):
        """
        직무 정의 로드
        
        Args:
            filepath: 직무 정의 JSON 파일 경로 (None이면 config에서 가져옴)
        """
        if filepath is None:
            filepath = str(JOB_DESCRIPTION_FILE)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            job_desc = JobDescription(
                job_name=item.get('직무', ''),
                job_definition=item.get('직무 정의', ''),
                industry=item.get('industry', ''),
                common_skills=item.get('공통_skill_set', []),
                specific_skills=item.get('skill_set', []),
                skill_set_description=item.get('skill_set_description', ''),
            )
            self.job_descriptions.append(job_desc)
        
        print(f"[OK] Job descriptions loaded: {len(self.job_descriptions)}")

    def load_training_data(self, job_files: List[str] = None):
        """
        기존 채용공고 로드
        
        Args:
            job_files: 학습 데이터 JSON 파일 경로 리스트 (None이면 config에서 가져옴)
            
        Note:
            TODO: 추후 DB에서 직접 가져오도록 수정 필요
            데이터 파이프라인 구축 완료 후 DB 쿼리로 대체 예정
        """
        if job_files is None:
            job_files = TRAINING_DATA_FILES
        
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

                    posting = JobPosting(
                        posting_id=f"{Path(filepath).stem}_{idx}",
                        company=job.get('company', 'Unknown'),
                        title=job.get('title', ''),
                        url=job.get('url', ''),
                        skills=skills,
                    )

                    self.graph.add_posting(posting)

                print(f"[OK] {Path(filepath).name}: {len(data)} loaded")
            except Exception as e:
                print(f"[FAIL] {filepath} load failed: {e}")

    def build_graph(self):
        """그래프 구축"""
        print("\n[그래프 구축]")
        print(f"  노드: {self.graph.G.number_of_nodes()}개")
        print(f"  엣지: {self.graph.G.number_of_edges()}개")

        print("\n[스킬 동시 출현 엣지 추가]")
        self.graph.build_skill_cooccurrence(min_cooccur=2)
        print(f"  엣지: {self.graph.G.number_of_edges()}개 (업데이트)")

    def build_matchers(self):
        """클러스터링 + SBERT 인덱싱"""
        print("\n[Louvain 클러스터링]")
        self.cluster_matcher = ClusterMatcher(self.graph.G)
        
        print("\n[SBERT Description 인덱싱]")
        self.sbert_matcher = SbertDescriptionMatcher(self.job_descriptions)
        
        self.matcher = JobMatcher(
            self.graph,
            self.cluster_matcher,
            self.sbert_matcher,
            self.job_descriptions,
        )

    def match_new_job(
        self,
        new_posting: NewJobPosting,
        ppr_top_n: int = 20,
        final_top_k: int = 2,
    ) -> List[JobMatchResult]:
        """새 채용공고 매칭"""
        if not self.matcher:
            raise ValueError("build_matchers()를 먼저 실행해야 합니다.")
        
        print("\n" + "="*60)
        print("직무 매칭 시작")
        print("="*60)
        
        results = self.matcher.match_job(
            new_posting,
            ppr_top_n=ppr_top_n,
            final_top_k=final_top_k,
        )
        return results
    
    def _convert_to_db_format(self, match_result: JobMatchResult) -> Dict[str, Any]:
        """
        JobMatchResult를 DB 저장용 JSON 형식으로 변환
        
        Returns:
            {
                "position": "Software Development",
                "industry": "Front-end Development",
                "sim_score": 0.6541,
                "sim_skill_matching": ["react", "git", "css"]
            }
        """
        return {
            "position": match_result.job_name,
            "industry": match_result.industry,
            "sim_score": round(match_result.final_score, 4),
            "sim_skill_matching": match_result.matching_skills,
        }

    def match_company_jobs(
        self,
        company_json_file: str,
        ppr_top_n: int = 20,
        final_top_k: int = 2,
    ) -> List[Dict]:
        """
        회사 전체 채용공고 매칭
        
        Returns:
            List[Dict] - 각 항목은 다음 키를 가짐:
                - 'posting': NewJobPosting 객체
                - 'matches': List[JobMatchResult] (전체 매칭 결과)
                - 'db_result': Dict (DB 저장용, 1등만 포함) 또는 None
        """
        with open(company_json_file, 'r', encoding='utf-8') as f:
            jobs_data = json.load(f)
        
        print(f"\n{'='*80}")
        print(f"[INFO] Company job matching: {len(jobs_data)} postings")
        print(f"파일: {Path(company_json_file).name}")
        print(f"{'='*80}")
        
        all_results = []
        
        for idx, job in enumerate(jobs_data):
            # 스킬 추출
            skills = []
            skill_info = job.get('skill_set_info', {})
            if isinstance(skill_info, dict):
                skill_set = skill_info.get('skill_set', [])
                if isinstance(skill_set, list):
                    skills = skill_set
            
            if not skills:
                continue
            
            # ---------- 🔧 변경 포인트 2: description에 본문 전체 우선 사용 ----------
            # 크롤러에서 본문 전체를 job["description"]이나 유사 키로 넣어준다 가정
            raw_body = (
                job.get('description')
                or job.get('content')
                or job.get('본문')
                or ""
            )
            
            new_posting = NewJobPosting(
                posting_id=f"new_{idx}",
                company=job.get('company', 'Unknown'),
                title=job.get('title', ''),
                skills=skills,
                url=job.get('url', ''),
                # 여기에는 "본문 전체"를 넣어두고, SBERT 쿼리에서는 title과 합쳐서 사용
                description=raw_body,
            )
            
            try:
                matches = self.match_new_job(
                    new_posting,
                    ppr_top_n=ppr_top_n,
                    final_top_k=final_top_k,
                )
                
                print(f"\n{'>'*40}")
                print(f"[{idx+1}/{len(jobs_data)}] {new_posting.company} - {new_posting.title}")
                print(f"스킬: {', '.join(new_posting.skills[:5])}{'...' if len(new_posting.skills) > 5 else ''}")
                print()
                
                for i, result in enumerate(matches, 1):
                    print(f"  {i}위. {result.job_name} / {result.industry}")
                    print(f"       점수: {result.final_score:.4f} | 매칭: {', '.join(result.matching_skills[:3])}")
                
                # DB 저장용 데이터 (1등만)
                db_result = None
                if matches and len(matches) > 0:
                    db_result = self._convert_to_db_format(matches[0])
                
                all_results.append({
                    'posting': new_posting,
                    'matches': matches,
                    'db_result': db_result,
                })
                
            except Exception as e:
                print(f"  [ERROR] Matching failed: {e}")
                # 에러 발생 시에도 DB 결과는 None으로 저장
                all_results.append({
                    'posting': new_posting,
                    'matches': [],
                    'db_result': None,
                })
        
        print(f"\n{'='*80}")
        print(f"[DONE] Complete: {len(all_results)} job postings matched")
        print(f"{'='*80}")
        
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: List[Dict]):
        """매칭 결과 요약"""
        job_counter = Counter()
        industry_counter = Counter()
        
        for item in results:
            if item['matches']:
                top_match = item['matches'][0]
                job_counter[top_match.job_name] += 1
                industry_counter[top_match.industry] += 1
        
        print(f"\n{'='*80}")
        print("[SUMMARY] Matching results")
        print(f"{'='*80}")
        
        print("\n[직무별 분포 (Top 10)]")
        for job_name, count in job_counter.most_common(10):
            print(f"  {job_name}: {count}개 ({count/len(results)*100:.1f}%)")
        
        print("\n[산업별 분포]")
        for industry, count in industry_counter.most_common():
            print(f"  {industry}: {count}개 ({count/len(results)*100:.1f}%)")
        
        print(f"\n{'='*80}")


# ============================================================================
# Main Execution (주석 처리 - FastAPI에서 사용할 예정)
# ============================================================================

# def main():
#     """메인 실행"""
# 
#     # 로그 파일명 생성 (타임스탬프 포함)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file = f"job_matching_v7_results_{timestamp}.txt"
# 
#     print("="*80)
#     print("직무 매칭 시스템 v7 - SBERT DESCRIPTION MATCHING")
#     print(f"로그 파일: {log_file}")
#     print("="*80)
# 
#     # app/core/job_matching에서 AI_Lab/data로 접근
#     base_dir = Path(__file__).parent.parent.parent.parent
#     data_dir = base_dir / "AI_Lab" / "data"
# 
#     # 로그 파일 활성화
#     system = JobMatchingSystem(log_file=log_file)
# 
#     print("\n[1/4] 직무 정의 로드")
#     system.load_job_descriptions(str(data_dir / 'new_job_description.json'))
# 
#     print("\n[2/4] 학습 데이터 로드")
#     training_files = [
#         str(data_dir / 'hanwha_jobs.json'),
#         str(data_dir / 'kakao_jobs.json'),
#         str(data_dir / 'line_jobs.json'),
#         str(data_dir / 'naver_jobs.json'),
#     ]
#     system.load_training_data(training_files)
# 
#     print("\n[3/4] 그래프 구축")
#     system.build_graph()
# 
#     print("\n[4/4] Matchers 초기화")
#     system.build_matchers()
# 
#     print("\n" + "="*80)
#     print("[OK] System ready!")
#     print("="*80)
# 
#     # line_jobs.json 안에 'description'(본문 전체) 필드까지 들어가 있으면
#     # SBERT가 제목+본문 기반으로 매칭 수행
#     results = system.match_company_jobs(
#         str(data_dir / 'line_jobs.json'),
#         ppr_top_n=20,
#         final_top_k=2,
#     )
# 
#     # DB 저장용 JSON 파일 생성 (1등 결과만)
#     json_output_file = f"job_matching_v7_db_results_{timestamp}.json"
#     db_results = []
#     
#     for result in results:
#         if result.get('db_result'):
#             # 원본 채용공고 정보와 DB 결과를 함께 저장
#             db_entry = {
#                 'company': result['posting'].company,
#                 'title': result['posting'].title,
#                 'url': result['posting'].url,
#                 **result['db_result']  # sim_position, sim_industry, sim_score, sim_skill_matching
#             }
#             db_results.append(db_entry)
#     
#     # JSON 파일로 저장
#     with open(json_output_file, 'w', encoding='utf-8') as f:
#         json.dump(db_results, f, ensure_ascii=False, indent=2)
#     
#     print(f"\n{'='*80}")
#     print(f"로그 파일: {log_file}")
#     print(f"DB 결과 JSON: {json_output_file} ({len(db_results)}개 결과)")
#     print(f"{'='*80}")
# 
# 
# if __name__ == '__main__':
#     main()