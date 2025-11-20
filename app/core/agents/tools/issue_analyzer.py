"""
Phase 2 Tools: 문제점 분석 도구
Phase 1 평가 데이터에서 개선이 필요한 부분을 추출하는 도구들
"""

from langchain_core.tools import tool
from typing import Dict, Any, List
import json
import re
from pathlib import Path


def _extract_data_from_dict(data: dict) -> tuple:
    """딕셔너리에서 평가 데이터 추출"""
    return (
        data.get("original_text", ""),
        data.get("keywords", []),
        data.get("keyword_count", 0),
        data.get("reasoning", "")
    )


@tool
def analyze_readability_issues(json_filename: str) -> str:
    """
    가독성 평가 데이터에서 문제점을 분석합니다.
    
    Args:
        json_filename: 평가 JSON 파일명
        
    Returns:
        가독성 문제점 요약 (전문용어, 일관성, 문법)
    """
    data_dir = Path("data/report")
    file_path = data_dir / json_filename
    
    if not file_path.exists():
        return f"[오류] 파일을 찾을 수 없습니다: {json_filename}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        readability = data.get('raw_evaluation_results', {}).get('readability', {})
        
        jargon_data = readability.get('jargon', {})
        consistency_data = readability.get('consistency', {})
        grammar_data = readability.get('grammar', {})
        
        # 데이터 추출
        _, jargon_keywords, jargon_count, jargon_reasoning = _extract_data_from_dict(jargon_data)
        _, consistency_keywords, consistency_count, consistency_reasoning = _extract_data_from_dict(consistency_data)
        _, grammar_keywords, grammar_count, grammar_reasoning = _extract_data_from_dict(grammar_data)
        
        result = f"""
=== 가독성 문제점 분석 ===

[사내 전문 용어 문제]
- 발견된 전문 용어 수: {jargon_count}개
- 문제가 되는 용어: {', '.join(jargon_keywords) if jargon_keywords else '없음'}
- 개선 방향: 일반인이 이해하기 어려운 전문 용어를 쉬운 표현으로 대체하거나 설명 추가

[문단 일관성 문제]
- 발견된 일관성 이슈 수: {consistency_count}개
- 문제가 되는 부분: {', '.join(consistency_keywords) if consistency_keywords else '없음'}
- 개선 방향: 문단 간 논리적 흐름 개선, 중복 내용 정리

[문법 정확성 문제]
- 발견된 문법 오류 수: {grammar_count}개
- 문제가 되는 부분: {', '.join(grammar_keywords) if grammar_keywords else '없음'}
- 개선 방향: 맞춤법, 띄어쓰기, 문장 구조 교정

[가독성 종합]
- 총 문제점: {jargon_count + consistency_count + grammar_count}개
- 우선 개선 필요: {'전문 용어' if jargon_count >= max(consistency_count, grammar_count) else '일관성' if consistency_count >= grammar_count else '문법'}
"""
        return result.strip()
        
    except Exception as e:
        return f"[오류] 분석 오류: {str(e)}"


@tool
def analyze_specificity_issues(json_filename: str) -> str:
    """
    구체성 평가 데이터에서 문제점을 분석합니다.
    
    Args:
        json_filename: 평가 JSON 파일명
        
    Returns:
        구체성 문제점 요약 (담당업무, 자격요건, 키워드, 필수항목)
    """
    data_dir = Path("data/report")
    file_path = data_dir / json_filename
    
    if not file_path.exists():
        return f"[오류] 파일을 찾을 수 없습니다: {json_filename}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        specificity = data.get('raw_evaluation_results', {}).get('specificity', {})
        
        responsibility_data = specificity.get('responsibility', {})
        qualification_data = specificity.get('qualification', {})
        keyword_data = specificity.get('keyword_relevance', {})
        required_data = specificity.get('required_fields', {})
        
        # 데이터 추출
        _, resp_keywords, resp_count, resp_reasoning = _extract_data_from_dict(responsibility_data)
        _, qual_keywords, qual_count, qual_reasoning = _extract_data_from_dict(qualification_data)
        _, kw_keywords, kw_count, kw_reasoning = _extract_data_from_dict(keyword_data)
        _, req_keywords, req_count, req_reasoning = _extract_data_from_dict(required_data)
        
        result = f"""
=== 구체성 문제점 분석 ===

[담당 업무 구체성]
- 분석된 항목 수: {resp_count}개
- 구체적인 업무: {', '.join(resp_keywords) if resp_keywords else '없음'}
- 개선 방향: 추상적인 업무 설명을 구체적인 작업과 책임으로 세분화

[자격 요건 구체성]
- 분석된 항목 수: {qual_count}개
- 명시된 요건: {', '.join(qual_keywords) if qual_keywords else '없음'}
- 개선 방향: 모호한 요건을 측정 가능한 기준으로 명확화

[키워드 적합성]
- 분석된 키워드 수: {kw_count}개
- 주요 키워드: {', '.join(kw_keywords) if kw_keywords else '없음'}
- 개선 방향: 직무와 관련된 핵심 키워드 보강

[필수 항목 포함 여부]
- 포함된 항목 수: {req_count}개
- 포함된 항목: {', '.join(req_keywords) if req_keywords else '없음'}
- 개선 방향: 누락된 필수 정보 추가 (급여, 근무지, 고용형태 등)

[구체성 종합]
- 총 분석 항목: {resp_count + qual_count + kw_count + req_count}개
- 가장 부족한 영역: {'담당업무' if resp_count <= min(qual_count, kw_count, req_count) else '자격요건' if qual_count <= min(kw_count, req_count) else '키워드' if kw_count <= req_count else '필수항목'}
"""
        return result.strip()
        
    except Exception as e:
        return f"[오류] 분석 오류: {str(e)}"


@tool
def analyze_attractiveness_issues(json_filename: str) -> str:
    """
    매력도 평가 데이터에서 문제점을 분석합니다.
    
    Args:
        json_filename: 평가 JSON 파일명
        
    Returns:
        매력도 문제점 요약 (특별 콘텐츠 포함 여부 및 충실도)
    """
    data_dir = Path("data/report")
    file_path = data_dir / json_filename
    
    if not file_path.exists():
        return f"[오류] 파일을 찾을 수 없습니다: {json_filename}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        attractiveness = data.get('raw_evaluation_results', {}).get('attractiveness', {})
        
        content_count_data = attractiveness.get('content_count', {})
        content_quality_data = attractiveness.get('content_quality', {})
        
        # 데이터 추출
        _, count_keywords, count_num, count_reasoning = _extract_data_from_dict(content_count_data)
        _, quality_keywords, quality_num, quality_reasoning = _extract_data_from_dict(content_quality_data)
        
        result = f"""
=== 매력도 문제점 분석 ===

[특별 콘텐츠 포함 여부]
- 포함된 콘텐츠 수: {count_num}개
- 포함된 콘텐츠: {', '.join(count_keywords) if count_keywords else '없음'}
- 개선 방향: 회사 비전, 복지혜택, 성장기회, 팀 문화 등의 콘텐츠 추가

[특별 콘텐츠 충실도]
- 평가된 항목 수: {quality_num}개
- 평가 대상: {', '.join(quality_keywords) if quality_keywords else '없음'}
- 개선 방향: 추상적인 설명을 구체적인 사례와 수치로 보강

[매력도 종합]
- 특별 콘텐츠 포함: {count_num}개 {'(부족)' if count_num < 3 else '(양호)' if count_num < 5 else '(우수)'}
- 콘텐츠 충실도: {quality_num}개 항목 평가됨
- 우선 개선: {'콘텐츠 추가 필요' if count_num < quality_num else '기존 콘텐츠 품질 향상'}

[추천 개선 방향]
- 회사만의 독특한 복지나 문화 강조
- 구체적인 성장 경로와 교육 프로그램 소개
- 실제 팀원 인터뷰나 업무 환경 사진 추가
"""
        return result.strip()
        
    except Exception as e:
        return f"[오류] 분석 오류: {str(e)}"


@tool
def get_overall_improvement_summary(json_filename: str) -> str:
    """
    전체 평가 데이터를 종합하여 개선이 가장 필요한 영역을 요약합니다.
    
    Args:
        json_filename: 평가 JSON 파일명
        
    Returns:
        종합 개선 우선순위 및 핵심 액션 아이템
    """
    data_dir = Path("data/report")
    file_path = data_dir / json_filename
    
    if not file_path.exists():
        return f"[오류] 파일을 찾을 수 없습니다: {json_filename}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_results = data.get('raw_evaluation_results', {})
        
        # 각 영역별 문제 개수 추정
        readability_issues = 0
        specificity_issues = 0
        attractiveness_issues = 0
        
        # 가독성
        for key in ['jargon', 'consistency', 'grammar']:
            eval_data = raw_results.get('readability', {}).get(key, {})
            readability_issues += eval_data.get('keyword_count', 0)
        
        # 구체성
        for key in ['responsibility', 'qualification', 'keyword_relevance', 'required_fields']:
            eval_data = raw_results.get('specificity', {}).get(key, {})
            specificity_issues += eval_data.get('keyword_count', 0)
        
        # 매력도
        for key in ['content_count', 'content_quality']:
            eval_data = raw_results.get('attractiveness', {}).get(key, {})
            attractiveness_issues += eval_data.get('keyword_count', 0)
        
        # 우선순위 결정
        issues = [
            ('가독성', readability_issues),
            ('구체성', specificity_issues),
            ('매력도', attractiveness_issues)
        ]
        issues.sort(key=lambda x: x[1], reverse=True)
        
        result = f"""
=== 종합 개선 우선순위 ===

[개선 필요 수준]
1위. {issues[0][0]}: {issues[0][1]}개 항목
2위. {issues[1][0]}: {issues[1][1]}개 항목
3위. {issues[2][0]}: {issues[2][1]}개 항목

[핵심 액션 아이템]
1. [{issues[0][0]}] 최우선 개선 - {issues[0][1]}개 항목 처리
2. [{issues[1][0]}] 2차 개선 - {issues[1][1]}개 항목 보완
3. [{issues[2][0]}] 마무리 개선 - {issues[2][1]}개 항목 추가

[AI 공고 생성 시 고려사항]
- {issues[0][0]} 영역에 특히 집중하여 개선
- 원본 공고의 핵심 정보는 유지하되, 표현 방식 개선
- 부족한 정보는 업계 표준이나 일반적인 내용으로 보완
- 최종 공고는 지원자 입장에서 명확하고 매력적이어야 함
"""
        return result.strip()
        
    except Exception as e:
        return f"[오류] 분석 오류: {str(e)}"

