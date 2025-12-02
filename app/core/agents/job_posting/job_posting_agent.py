"""
Phase 2: AI 채용 공고 생성 Agent
평가 데이터를 바탕으로 개선된 채용 공고를 생성하는 Agent
"""

from typing import Optional, Dict, Any
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv

# Schemas
from app.schemas.agent.schemas_report import JobPostingDetailReport

# Phase 2 Tools
from app.core.agents.job_posting.tools.issue_analyzer import (
    analyze_readability_issues,
    analyze_specificity_issues,
    analyze_attractiveness_issues,
    get_overall_improvement_summary,
)

# Utils
from app.utils.agents.evaluation.json_loader import (
    load_evaluation_json,
    list_available_evaluations,
)

# DB
from app.db.crud.post import get_post_by_id
from app.db.config.base import get_db

load_dotenv(override=True)


def _format_evaluation_feedback(raw_results: Dict[str, Any]) -> str:
    """평가 결과를 피드백 형식으로 포맷팅"""
    feedback = []
    
    # 가독성 피드백
    readability = raw_results.get('readability', {})
    jargon = readability.get('jargon', {})
    consistency = readability.get('consistency', {})
    grammar = readability.get('grammar', {})
    
    # jargon이 dictionary인지 확인 (에러 메시지 string일 수 있음)
    if isinstance(jargon, dict) and jargon.get('keyword_count', 0) > 0:
        feedback.append(f"[가독성 - 전문용어] {jargon.get('keyword_count')}개 발견: {', '.join(jargon.get('keywords', []))}")
        feedback.append(f"  → {jargon.get('reasoning', '')[:100]}...")
    
    if isinstance(consistency, dict) and consistency.get('keyword_count', 0) > 0:
        feedback.append(f"[가독성 - 일관성] {consistency.get('keyword_count')}개 문제: {', '.join(consistency.get('keywords', []))}")
        feedback.append(f"  → {consistency.get('reasoning', '')[:100]}...")
    
    if isinstance(grammar, dict) and grammar.get('keyword_count', 0) > 0:
        feedback.append(f"[가독성 - 문법] {grammar.get('keyword_count')}개 오류: {', '.join(grammar.get('keywords', []))}")
        feedback.append(f"  → {grammar.get('reasoning', '')[:100]}...")
    
    # 구체성 피드백
    specificity = raw_results.get('specificity', {})
    responsibility = specificity.get('responsibility', {})
    qualification = specificity.get('qualification', {})
    
    if isinstance(responsibility, dict) and responsibility.get('keyword_count', 0) > 0:
        feedback.append(f"[구체성 - 담당업무] {', '.join(responsibility.get('keywords', [])[:3])}...")
        feedback.append(f"  → {responsibility.get('reasoning', '')[:100]}...")
    
    if isinstance(qualification, dict) and qualification.get('keyword_count', 0) > 0:
        feedback.append(f"[구체성 - 자격요건] {', '.join(qualification.get('keywords', [])[:3])}...")
        feedback.append(f"  → {qualification.get('reasoning', '')[:100]}...")
    
    # 매력도 피드백
    attractiveness = raw_results.get('attractiveness', {})
    content_count = attractiveness.get('content_count', {})
    content_quality = attractiveness.get('content_quality', {})
    
    if isinstance(content_count, dict) and content_count.get('keyword_count', 0) > 0:
        feedback.append(f"[매력도 - 특별콘텐츠] {content_count.get('keyword_count')}개 포함: {', '.join(content_count.get('keywords', []))}")
        feedback.append(f"  → {content_count.get('reasoning', '')[:100]}...")
    
    if isinstance(content_quality, dict) and content_quality.get('keyword_count', 0) > 0:
        feedback.append(f"[매력도 - 콘텐츠품질] {', '.join(content_quality.get('keywords', [])[:3])}...")
        feedback.append(f"  → {content_quality.get('reasoning', '')[:100]}...")
    
    return "\n".join(feedback) if feedback else "평가 결과에 특별한 문제점이 발견되지 않았습니다."


def create_job_posting_generator_agent(
    llm_model: str = "gpt-4o",
):
    """
    AI 채용 공고 개선 Agent 생성
    
    Args:
        llm_model: 사용할 LLM 모델
        
    Returns:
        Agent: 실행 가능한 Agent
    """
    
    # Tools 정의
    tools = [
        list_available_evaluations,
        load_evaluation_json,
        analyze_readability_issues,
        analyze_specificity_issues,
        analyze_attractiveness_issues,
        get_overall_improvement_summary,
    ]
    
    # System Prompt - JobPostingDetailReport 형식으로 출력
    system_prompt = """당신은 채용 공고 정보 추출 및 구조화 전문가입니다.

**작업 프로세스:**
1. load_evaluation_json으로 원본 채용 공고와 평가 데이터를 로드
2. analyze_readability_issues, analyze_specificity_issues, analyze_attractiveness_issues로 문제점 분석
3. 원본 채용공고에서 다음 20개 필드의 정보를 추출하여 JSON 형식으로 반환

**추출할 필드 (JobPostingDetailReport 형식):**
1. company_name (필수): 회사명
2. position (필수): 채용 직무/포지션
3. employment_type: 고용 형태 (정규직, 계약직, 인턴 등)
4. work_location: 근무지
5. deadline: 마감일 (YYYY-MM-DD 형식)
6. company_introduction: 회사 소개
7. main_responsibilities: 주요 업무 (담당업무)
8. required_qualifications: 자격 요건 (필수 요건)
9. preferred_qualifications: 우대 사항
10. tech_stack: 기술 스택 (리스트 형식, 예: ["Python", "Django"])
11. team_introduction: 팀 소개
12. development_culture: 개발 문화
13. tools: 사용 도구 (리스트 형식, 예: ["Jira", "GitLab"])
14. project_introduction: 프로젝트 소개
15. growth_opportunities: 성장 기회
16. recruitment_process: 전형 절차
17. work_conditions: 근무 조건
18. benefits: 복리후생
19. application_method: 지원 방법
20. additional_info: 기타 사항

**출력 형식:**
- 반드시 유효한 JSON 형식으로 출력
- 필수 필드(company_name, position)는 반드시 포함
- 선택 필드는 정보가 없으면 null로 설정
- tech_stack과 tools는 배열 형식으로 출력
- 날짜는 YYYY-MM-DD 형식으로 통일
- 모든 텍스트는 공백 정리 및 정규화

**중요:**
- 원본 공고의 정보를 정확하게 추출
- 정보가 없는 필드는 null로 설정
- JSON만 출력하고 설명이나 메타 정보는 포함하지 않음"""
    
    # LLM 생성 (max_tokens 설정하여 출력이 잘리지 않도록)
    llm = ChatOpenAI(
        model=llm_model,
        temperature=0.3,
        max_tokens=16384,  # 충분한 토큰 할당
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Agent 생성 (LangChain v1.0)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    
    return agent


async def generate_improved_job_posting_async(
    json_filename: Optional[str] = None,
    llm_model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    비동기로 개선된 채용 공고를 생성합니다.
    
    Args:
        json_filename: 처리할 JSON 파일명 (None이면 사용 가능한 파일 목록 확인)
        llm_model: 사용할 LLM 모델
        
    Returns:
        Dict: 생성 결과
            {
                "status": "success" | "error",
                "improved_posting": "개선된 공고 텍스트",
                "original_file": "처리한 JSON 파일명",
                "message": "결과 메시지"
            }
    """
    try:
        # JSON 파일명이 없으면 자동으로 첫 번째 파일 찾기
        if json_filename is None:
            data_dir = Path("data/report")
            json_files = sorted(data_dir.glob("post_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            if json_files:
                json_filename = json_files[0].name
            else:
                return {
                    "status": "error",
                    "improved_posting": "",
                    "original_file": None,
                    "message": "처리할 JSON 파일이 없습니다."
                }
        
        # JSON 파일에서 원본 공고와 평가 결과 가져오기
        original_content = ""
        post_id = None
        company = ""
        title = ""
        evaluation_summary = ""
        
        if json_filename:
            data_dir = Path("data/report")
            file_path = data_dir / json_filename
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metadata = data.get('metadata', {})
                    post_id = metadata.get('post_id')
                    company = metadata.get('company', '')
                    title = metadata.get('title', '')
                    
                    # DB에서 원본 공고 가져오기
                    if post_id:
                        try:
                            db = next(get_db())
                            post = get_post_by_id(db, post_id)
                            if post:
                                original_content = post.description or ""
                        except Exception as e:
                            print(f"[Job Posting Generator] Failed to load original post: {e}")
                    
                    # 평가 결과 요약
                    raw_results = data.get('raw_evaluation_results', {})
                    evaluation_summary = _format_evaluation_feedback(raw_results)
        
        # Agent 생성
        agent = create_job_posting_generator_agent(llm_model=llm_model)
        
        # 요청 메시지 구성
        if json_filename:
            request = f"""'{json_filename}' 파일을 분석하여 채용공고 정보를 구조화된 JSON 형식으로 추출하세요.

작업 지시:
1. load_evaluation_json으로 원본 채용 공고 로드
2. 원본 채용공고에서 20개 필드의 정보를 추출
3. JobPostingDetailReport 형식의 JSON으로 반환

중요: 
- 반드시 유효한 JSON 형식으로만 출력
- 필수 필드(company_name, position)는 반드시 포함
- 정보가 없는 필드는 null로 설정
- tech_stack과 tools는 배열 형식
- JSON 외의 설명이나 메타 정보는 포함하지 않음"""
        else:
            request = "사용 가능한 평가 데이터를 확인하고 처리하세요."
        
        # Agent 실행
        result = await agent.ainvoke({
            "messages": [
                {"role": "user", "content": request}
            ]
        })
        
        # 결과 추출
        if "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            output = last_message.get("content", "") if isinstance(last_message, dict) else str(last_message.content)
        else:
            output = str(result)
        
        # JSON 추출 및 파싱
        job_posting_report = None
        try:
            # JSON 부분만 추출 (정규식 사용)
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                json_data = json.loads(json_text)
                # JobPostingDetailReport로 변환
                job_posting_report = JobPostingDetailReport(**json_data)
            else:
                # JSON이 없으면 직접 파싱 시도
                json_data = json.loads(output)
                job_posting_report = JobPostingDetailReport(**json_data)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"[Job Posting Generator] JSON 파싱 실패: {e}")
            print(f"원본 출력: {output[:500]}...")
            # 파싱 실패 시 에러 반환
            return {
                "status": "error",
                "data": None,
                "original_file": json_filename,
                "message": f"JSON 파싱 실패: {str(e)}"
            }
        
        return {
            "status": "success",
            "data": job_posting_report.dict() if job_posting_report else None,
            "original_file": json_filename,
            "title": title,
            "company": company,
            "message": "채용공고 정보 추출 완료"
        }
        
    except Exception as e:
        print(f"[Job Posting Generator] Error: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "data": None,
            "original_file": json_filename,
            "message": f"생성 실패: {str(e)}"
        }


def generate_improved_job_posting(
    json_filename: Optional[str] = None,
    llm_model: str = "gpt-4o"
) -> Dict[str, Any]:
    """
    동기적으로 개선된 채용 공고를 생성합니다. (비동기 래퍼)
    
    Args:
        json_filename: 처리할 JSON 파일명
        llm_model: 사용할 LLM 모델
        
    Returns:
        Dict: 생성 결과
    """
    import asyncio
    return asyncio.run(generate_improved_job_posting_async(json_filename, llm_model))

