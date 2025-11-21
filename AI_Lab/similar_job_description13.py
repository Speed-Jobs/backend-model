"""
직무 매칭 시스템 v13 - PPR 필터링 전용 버전

핵심 전략:
1. PPR은 "필터링 전용" - specific_skills만 사용, 점수 버림, 상위 20개만 추출
2. Jaccard + SBERT로만 점수 계산 - 전체 스킬(common + specific) 사용
3. 가중치 없이 단순 합산 (0~2 범위)

주요 변경사항 (v12 → v13):
1. PPR 역할 변경
   - 기존: 점수 계산에 포함 (50%)
   - 변경: 후보 필터링 전용 (점수 버림)
   - 이유: 정규화 시 Hallucination 발생
   - PPR 계산: specific_skills만 사용 (직무 특화 스킬로 필터링)

2. Jaccard 부활
   - Weighted Jaccard 재도입 (common 0.33 + specific 0.67)
   - 필터링 제거 (모든 후보 계산)
   - Jaccard 계산: 전체 스킬(common + specific) 사용

3. 점수 계산
   - Final = Jaccard + SBERT (0~2 범위)
   - 가중치 없음 (단순 합산)

점수 구성:
- PPR: 필터링 전용 (specific_skills로 상위 20개 추출)
- Jaccard: 스킬 직접 매칭 (common + specific 전체, 0~1)
- SBERT: 의미 유사도 (0~1)
- Final: Jaccard + SBERT (0~2)

장점:
- PPR Hallucination 제거
- Jaccard로 스킬 매칭 정확도 향상
- SBERT 높은 성능 활용
- 단순하고 투명한 점수 체계

Usage:
    python similar_job_description13.py
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
from sentence_transformers import SentenceTransformer

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
    specific_only_skills: List[str] = field(default_factory=list)  # PPR 전용
    skill_set_description: str = ""  # 주요 업무 설명 (SBERT에 사용)
    common_skill_set_description: str = ""  # 공통 스킬 설명 (SBERT에 사용)

    def __post_init__(self):
        # v13: all_skills는 전체 (Jaccard/매칭용), specific_only_skills는 PPR 전용
        self.all_skills = list(set(self.common_skills + self.specific_skills))
        self.specific_only_skills = list(set(self.specific_skills))

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
    pagerank_score: float = 0.0  # 로깅용, 점수 계산에는 미포함
    sbert_score: float = 0.0
    
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
        model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2",
    ):
        self.job_descriptions = job_descriptions

        print(f"[SBERT] 모델 로딩 중... ({model_name})")
        self.model = SentenceTransformer(model_name)

        print(f"[SBERT] 직무 definition 임베딩 생성 중... (직무 정의 + industry + skill_set_description + 공통_skill_set_description)")
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
            
            # 4. 공통 Skill Set Description 추가 (프로그래밍 언어, 협업 도구 등)
            if jd.common_skill_set_description:
                parts.append(f"공통 기술: {jd.common_skill_set_description}")
            
            # 모든 정보 결합 (정보가 없으면 job_name만 사용)
            combined_text = "\n\n".join(parts) if parts else jd.job_name
            corpus.append(combined_text)

        # normalize_embeddings=True → 코사인 유사도 = 내적
        self.job_embeddings = self.model.encode(
            corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        print(f"[OK] {len(job_descriptions)}개 직무 정의 임베딩 완료 (industry + skill_set_description + 공통_skill_set_description 포함)")

    def calculate_similarity_no_normalize(self, query_text: str) -> Dict[str, float]:
        """
        새 공고의 쿼리 텍스트와 직무 정의의 의미 유사도 계산 (정규화 제거)

        Returns:
            Dict[job_name, raw_similarity_score] - 원본 코사인 유사도 (0~1 변환만)
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

        # [-1, 1] → [0, 1] 변환만 (max 정규화 제거!)
        sims = (sims + 1.0) / 2.0

        result = {}
        for i, jd in enumerate(self.job_descriptions):
            result[jd.job_name] = float(sims[i])

        return result


# ============================================================================
# Job Matcher (핵심 로직)
# ============================================================================

class JobMatcher:
    """새 채용공고 → 직무 매칭 (PPR 필터링 전용)"""

    def __init__(
        self,
        graph: JobPostingGraph,
        sbert_matcher: SbertDescriptionMatcher,
        job_descriptions: List[JobDescription],
    ):
        self.graph = graph
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
        
        v13: PPR 필터링 전용 + Jaccard + SBERT
        - PPR: 상위 20개 추출 (점수 버림)
        - Jaccard + SBERT: 단순 합산 (0~2)
        - 필터링 제거 (모든 후보 계산)
        """
        print(f"\n[Stage 1] PPR 기반 1차 필터링 (상위 {ppr_top_n}개 추출, 점수 버림)")
        
        # Stage 1: PPR로 상위 N개 직무 추출 (점수는 버림!)
        ppr_candidates = self._get_ppr_top_jobs(new_posting, ppr_top_n)
        
        if not ppr_candidates:
            print("  ! PPR 후보 없음, 전체 직무 대상으로 진행")
            ppr_candidates = [(jd, 0.0) for jd in self.job_descriptions]
        
        print(f"  [OK] {len(ppr_candidates)} jobs selected")
        
        # SBERT 쿼리 텍스트 구성 (title + description)
        query_text = f"{new_posting.title}\n\n{new_posting.description}".strip()
        
        # Stage 2: SBERT 계산 (전체 직무 대상)
        print(f"\n[Stage 2] SBERT + Jaccard 점수 계산")
        sbert_scores = self.sbert_matcher.calculate_similarity_no_normalize(query_text)
        
        if query_text:
            top_sbert = max(sbert_scores.items(), key=lambda x: x[1])
            print(f"  - SBERT 1등: {top_sbert[0]} (점수: {top_sbert[1]:.4f})")
        else:
            print(f"  ! Description/Title 없음 - SBERT 점수 모두 0")
        
        # Stage 3: PPR 후보들에 대해 Jaccard + SBERT 계산
        results = []
        
        for job_desc, ppr_score in ppr_candidates:
            # Weighted Jaccard 계산
            jaccard = self._calculate_weighted_jaccard(new_posting.skills, job_desc)
            
            # SBERT 점수
            sbert = sbert_scores.get(job_desc.job_name, 0.0)
            
            # 매칭 스킬 분석
            new_skills_norm = set(self.graph._normalize_skill(s) for s in new_posting.skills)
            job_skills_norm = set(self.graph._normalize_skill(s) for s in job_desc.all_skills)
            
            matching_skills = list(new_skills_norm & job_skills_norm)
            missing_skills = list(job_skills_norm - new_skills_norm)
            
            # v13: 필터링 제거! 모든 후보 계산
            
            # 최종 점수: Jaccard + SBERT (0~2 범위)
            final_score = jaccard + sbert
            
            result = JobMatchResult(
                job_name=job_desc.job_name,
                industry=job_desc.industry,
                final_score=final_score,
                jaccard_score=jaccard,
                pagerank_score=ppr_score,  # 로깅용
                sbert_score=sbert,
                matching_skills=matching_skills[:10],
                missing_skills=missing_skills[:5],
                job_definition=job_desc.job_definition,
                reason=self._generate_reason(matching_skills, jaccard, sbert),
            )
            
            results.append(result)
        
        # 정렬 및 Top-K 반환
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        print(f"  [OK] Final top {min(final_top_k, len(results))} returned")
        if results:
            print(
                "  - 1등: "
                f"{results[0].job_name}/{results[0].industry}\n"
                "         점수: "
                f"{results[0].final_score:.4f} "
                f"(Jacc:{results[0].jaccard_score:.4f}, "
                f"SBERT:{results[0].sbert_score:.4f})"
            )
        else:
            print(f"  [WARNING] 매칭 가능한 직무 없음")
        
        return results[:final_top_k]
    
    def _get_ppr_top_jobs(self, new_posting: NewJobPosting, top_n: int) -> List[Tuple[JobDescription, float]]:
        """
        PPR로 상위 N개 직무 추출 (점수는 버림!)
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
                    for s in job_desc.specific_only_skills  # PPR은 specific만 사용
                ]
                
                ppr_scores = [ppr.get(node, 0.0) for node in skill_nodes]
                avg_ppr = np.mean(ppr_scores) if ppr_scores else 0.0
                
                job_ppr_scores.append((job_desc, avg_ppr))
            
            # 정규화 없이 그냥 정렬만
            job_ppr_scores.sort(key=lambda x: x[1], reverse=True)
            top_candidates = job_ppr_scores[:top_n]
            
            if top_candidates:
                print(
                    f"  - PPR 1등: {top_candidates[0][0].job_name}/"
                    f"{top_candidates[0][0].industry} (원본 점수: {top_candidates[0][1]:.6f})"
                )
                print(
                    f"  - PPR {len(top_candidates)}등: {top_candidates[-1][0].job_name}/"
                    f"{top_candidates[-1][0].industry} (원본 점수: {top_candidates[-1][1]:.6f})"
                )
            
            return top_candidates
        
        except Exception as e:
            print(f"  ! PPR 계산 실패: {e}")
            return []

    def _calculate_jaccard(self, skills1: List[str], skills2: List[str]) -> float:
        """기본 Jaccard 계산 (단일 스킬 리스트 비교)"""
        set1 = set(self.graph._normalize_skill(s) for s in skills1)
        set2 = set(self.graph._normalize_skill(s) for s in skills2)
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_weighted_jaccard(self, new_posting_skills: List[str], job_desc: JobDescription) -> float:
        """
        가중치 적용 Jaccard 계산
        
        - common_skills: 0.33 가중치 (모든 직무 공통, 덜 중요)
        - specific_skills: 0.67 가중치 (직무 특화, 매우 중요)
        
        Args:
            new_posting_skills: 채용공고의 스킬 리스트
            job_desc: 직무 정의 객체
            
        Returns:
            float (0~1): 가중치 적용된 Jaccard 점수
        """
        # 1. Common skills Jaccard
        jaccard_common = self._calculate_jaccard(
            new_posting_skills, 
            job_desc.common_skills
        )
        
        # 2. Specific skills Jaccard
        jaccard_specific = self._calculate_jaccard(
            new_posting_skills, 
            job_desc.specific_skills
        )
        
        # 3. 가중 평균 (specific에 2배 가중치)
        weighted_jaccard = 0.33 * jaccard_common + 0.67 * jaccard_specific
        
        return weighted_jaccard

    def _generate_reason(self, matching_skills: List[str], jaccard: float, sbert: float) -> str:
        """매칭 이유 생성"""
        num_matches = len(matching_skills)
        
        if sbert > 0.5 and jaccard > 0.3:
            return f"의미 + 스킬 매칭 강함 (SBERT: {sbert:.3f}, Jacc: {jaccard:.3f}), 스킬 {num_matches}개"
        elif sbert > 0.5:
            return f"Description 의미 매칭 강함 (SBERT: {sbert:.3f}), 스킬 {num_matches}개"
        elif num_matches >= 5:
            return f"매칭 스킬 {num_matches}개 (Jacc: {jaccard:.3f})"
        elif num_matches >= 3:
            return f"매칭 스킬 {num_matches}개: {', '.join(matching_skills[:3])} (Jacc: {jaccard:.3f})"
        elif num_matches > 0:
            return f"매칭 스킬: {', '.join(matching_skills)} (Jacc: {jaccard:.3f})"
        else:
            return f"의미 유사도 기반 (SBERT: {sbert:.3f})"


# ============================================================================
# Main System
# ============================================================================

class JobMatchingSystem:
    """통합 직무 매칭 시스템 v13 (PPR 필터링 전용)"""

    def __init__(self, log_file: Optional[str] = None):
        self.graph = JobPostingGraph()
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

    def load_job_descriptions(self, filepath: str):
        """직무 정의 로드"""
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
                common_skill_set_description=item.get('공통_skill_set_description', ''),
            )
            self.job_descriptions.append(job_desc)
        
        print(f"[OK] Job descriptions loaded: {len(self.job_descriptions)}")

    def load_training_data(self, job_files: List[str]):
        """기존 채용공고 로드"""
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
        print(f"  [NOTE] v13: 스킬 동시 출현 엣지 생성 생략")
        print(f"  [NOTE] v13: PPR은 specific_skills만, Jaccard는 전체 스킬 사용")

    def build_matchers(self):
        """SBERT 인덱싱"""
        print("\n[SBERT Description 인덱싱]")
        self.sbert_matcher = SbertDescriptionMatcher(self.job_descriptions)
        
        self.matcher = JobMatcher(
            self.graph,
            self.sbert_matcher,
            self.job_descriptions,
        )
        
        print("[NOTE] v13: PPR은 필터링 전용 (점수 버림)")
        print("[NOTE] v13: Jaccard + SBERT 단순 합산 (0~2)")
        print("[NOTE] v13: 필터링 제거 (모든 후보 계산)")

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
            
            # description에 본문 전체 우선 사용
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
# Main Execution
# ============================================================================

def main():
    """메인 실행"""

    # 로그 파일명 생성 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"job_matching_v13_results_{timestamp}.txt"

    print("="*80)
    print("직무 매칭 시스템 v13 - PPR 필터링 전용 버전")
    print(f"로그 파일: {log_file}")
    print("="*80)

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"

    # 로그 파일 활성화
    system = JobMatchingSystem(log_file=log_file)

    print("\n[1/4] 직무 정의 로드")
    system.load_job_descriptions(str(data_dir / 'new_job_description.json'))

    print("\n[2/4] 학습 데이터 로드")
    training_files = [
        str(data_dir / 'hanwha_jobs.json'),
        str(data_dir / 'kakao_jobs.json'),
        str(data_dir / 'line_jobs.json'),
        str(data_dir / 'naver_jobs.json'),
    ]
    system.load_training_data(training_files)

    print("\n[3/4] 그래프 구축")
    system.build_graph()

    print("\n[4/4] Matchers 초기화")
    system.build_matchers()

    print("\n" + "="*80)
    print("[OK] System ready!")
    print("="*80)

    # 매칭 수행
    results = system.match_company_jobs(
        str(data_dir / 'line_jobs.json'),
        ppr_top_n=20,
        final_top_k=2,
    )

    # DB 저장용 JSON 파일 생성 (1등 결과만)
    json_output_file = f"job_matching_v13_db_results_{timestamp}.json"
    db_results = []
    
    for result in results:
        if result.get('db_result'):
            # 원본 채용공고 정보와 DB 결과를 함께 저장
            db_entry = {
                'company': result['posting'].company,
                'title': result['posting'].title,
                'url': result['posting'].url,
                **result['db_result']  # position, industry, sim_score, sim_skill_matching
            }
            db_results.append(db_entry)
    
    # JSON 파일로 저장
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(db_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"로그 파일: {log_file}")
    print(f"DB 결과 JSON: {json_output_file} ({len(db_results)}개 결과)")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

