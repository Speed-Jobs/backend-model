"""
회사 그룹 매핑 설정
프론트엔드에서 받은 키워드를 DB의 회사명 패턴으로 변환
"""
from typing import List

COMPANY_GROUPS = {
    "toss": ["토스%", "토스뱅크%", "토스증권%", "비바리퍼블리카%", "AICC%"],
    "kakao": ["카카오%"],
    "hanwha": ["한화시스템%", "한화시스템템%", "한화시스템/ICT%", "한화시스템·ICT%"],
    "hyundai autoever": ["현대오토에버%"],
    "woowahan": ["우아한%", "배달의민족", "배민"],
    "coupang": ["쿠팡%", "Coupang%"],
    "line": ["LINE%", "라인%"],
    "naver": ["NAVER%", "네이버%"],
    "lg cns": ["LG_CNS%", "LG CNS%"],
}


def get_company_patterns(keyword: str) -> List[str]:
    """
    키워드를 회사명 패턴 리스트로 변환
    
    Args:
        keyword: 프론트엔드에서 받은 키워드 (예: "toss", "kakao")
        
    Returns:
        회사명 패턴 리스트 (예: ["토스%", "토스뱅크%", ...])
        키워드가 COMPANY_GROUPS에 없으면 [f"{keyword}%"] 반환
    """
    keyword_lower = keyword.lower().strip()
    
    # COMPANY_GROUPS에서 패턴 찾기
    if keyword_lower in COMPANY_GROUPS:
        return COMPANY_GROUPS[keyword_lower]
    
    # 없으면 키워드를 그대로 패턴으로 사용
    return [f"{keyword}%"]

