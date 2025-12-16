"""Reranker Tool

Re-ranks search results based on entity matching and relevance.
"""

from typing import List, Dict, Any, Optional


class Reranker:
    """Reranks search results to improve accuracy"""
    
    def __init__(self):
        pass
    
    def rerank(
        self,
        results: List[Dict[str, Any]],
        extracted_entities: Dict[str, Any],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank results based on entity matching
        
        Args:
            results: Original search results from VectorDB
            extracted_entities: Entities extracted from query (company_name, year, etc.)
            query: Original query string
            
        Returns:
            Reranked results with adjusted scores
        """
        if not results:
            return results
        
        company_name = extracted_entities.get("company_name")
        year = extracted_entities.get("year")
        period = extracted_entities.get("period")
        
        # Apply reranking
        reranked = []
        for result in results:
            score = result.get("score", 0.0)
            boost = 0.0
            reasons = []
            
            # Extract text and metadata
            text = result.get("text", "").lower()
            metadata = result.get("metadata", {})
            
            # Company name matching (강력한 부스트)
            if company_name:
                company_lower = company_name.lower()
                company_found = False
                
                # 제목에 회사명이 정확히 포함된 경우 (매우 강한 부스트)
                if f"[{company_lower}]" in text or f"({company_lower})" in text:
                    boost += 0.3
                    reasons.append(f"Title contains exact company name: [{company_name}]")
                    company_found = True
                # 텍스트에 회사명 포함 (중간 부스트)
                elif company_lower in text:
                    boost += 0.15
                    reasons.append(f"Text contains company name: {company_name}")
                    company_found = True
                
                # 회사명이 없는 경우 완전히 제외 (점수를 -1로 만들어서 제거)
                if not company_found:
                    boost = -1.0  # 점수를 음수로 만들어서 필터링됨
                    reasons.append(f"EXCLUDED: Missing company name '{company_name}'")
            
            # Year matching
            if year and str(year) in text:
                boost += 0.05
                reasons.append(f"Year match: {year}")
            
            # Period matching (상반기/하반기)
            if period:
                period_lower = period.lower()
                if period_lower in text:
                    boost += 0.05
                    reasons.append(f"Period match: {period}")
            
            # Apply boost
            adjusted_score = min(1.0, max(0.0, score + boost))
            
            # 회사명이 지정되었는데 없는 경우 제외 (음수 점수)
            if adjusted_score <= 0.0 and company_name:
                continue  # 결과에서 완전히 제외
            
            reranked.append({
                **result,
                "original_score": score,
                "score": adjusted_score,
                "boost": boost,
                "rerank_reasons": reasons
            })
        
        # Sort by adjusted score
        reranked.sort(key=lambda x: x["score"], reverse=True)
        
        # Log reranking results
        excluded_count = len(results) - len(reranked)
        print(f"[Reranker] Reranked {len(results)} results → {len(reranked)} results (excluded: {excluded_count})")
        
        if company_name:
            print(f"[Reranker] Company filter: '{company_name}' - results without company name were excluded")
        
        for i, r in enumerate(reranked[:3], 1):
            print(f"  {i}. Score: {r['score']:.3f} (orig: {r['original_score']:.3f}, boost: {r['boost']:+.3f})")
            if r.get("rerank_reasons"):
                for reason in r["rerank_reasons"]:
                    print(f"     - {reason}")
        
        return reranked

