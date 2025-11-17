"""
Phase 2 Tools: JSON 데이터 로더
Agent가 Phase 1의 평가 데이터를 읽어오는 도구
"""

from langchain_core.tools import tool
from typing import Dict, Any, Optional
import json
import os
from pathlib import Path
from app.db.crud.post import get_post_by_id
from app.db.config.base import get_db


@tool
def load_evaluation_json(json_filename: str) -> str:
    """
    Phase 1에서 생성된 평가 JSON 파일을 읽어옵니다.
    
    Args:
        json_filename: 읽을 JSON 파일명 (예: "post_123_평가데이터.json")
        
    Returns:
        JSON 데이터를 문자열로 반환 (Agent가 읽기 쉽도록)
    """
    data_dir = Path("data/report")
    file_path = data_dir / json_filename
    
    if not file_path.exists():
        return f"[오류] 파일을 찾을 수 없습니다: {json_filename}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Agent가 이해하기 쉬운 형태로 포맷팅
        metadata = data.get('metadata', {})
        raw_results = data.get('raw_evaluation_results', {})
        
        # DB에서 원본 공고 가져오기
        post_id = metadata.get('post_id')
        original_content = ""
        if post_id:
            try:
                db = next(get_db())
                post = get_post_by_id(db, post_id)
                if post:
                    original_content = post.description or ""
                    print(f"[JSON Loader - DEBUG] 원본 공고 로드 성공: {len(original_content)}자")
                    print(f"[JSON Loader - DEBUG] 미리보기: {original_content[:200]}...")
                else:
                    print(f"[JSON Loader - DEBUG] Post를 찾을 수 없음: post_id={post_id}")
            except Exception as e:
                print(f"[JSON Loader] Failed to load original post: {e}")
        
        # 가독성 데이터
        readability = raw_results.get('readability', {})
        jargon = readability.get('jargon', {})
        consistency = readability.get('consistency', {})
        grammar = readability.get('grammar', {})
        
        # 구체성 데이터
        specificity = raw_results.get('specificity', {})
        responsibility = specificity.get('responsibility', {})
        qualification = specificity.get('qualification', {})
        keyword_relevance = specificity.get('keyword_relevance', {})
        required_fields = specificity.get('required_fields', {})
        
        # 매력도 데이터
        attractiveness = raw_results.get('attractiveness', {})
        content_count = attractiveness.get('content_count', {})
        content_quality = attractiveness.get('content_quality', {})
        
        result = f"""
=== 평가 데이터 로드 완료 ===
파일명: {json_filename}

[기본 정보]
- 공고 ID: {metadata.get('post_id', 'N/A')}
- 회사 ID: {metadata.get('company_id', 'N/A')}
- 제목: {metadata.get('title', 'N/A')}
- 회사: {metadata.get('company', 'N/A')}
- 평가 시간: {metadata.get('evaluated_at', 'N/A')}

[원본 채용 공고 내용]
{original_content}...
(총 {len(original_content)}자)

[평가 결과 - 구조화된 데이터]

[가독성 평가]

1. 사내 전문 용어:
   - 키워드 개수: {jargon.get('keyword_count', 0)}개
   - 발견된 용어: {', '.join(jargon.get('keywords', [])) if jargon.get('keywords') else '없음'}
   - 판단 근거: {jargon.get('reasoning', '없음')}

2. 문단 일관성:
   - 문제 개수: {consistency.get('keyword_count', 0)}개
   - 문제 부분: {', '.join(consistency.get('keywords', [])) if consistency.get('keywords') else '없음'}
   - 판단 근거: {consistency.get('reasoning', '없음')}

3. 문법 정확성:
   - 오류 개수: {grammar.get('keyword_count', 0)}개
   - 오류 부분: {', '.join(grammar.get('keywords', [])) if grammar.get('keywords') else '없음'}
   - 판단 근거: {grammar.get('reasoning', '없음')}

[구체성 평가]

1. 담당 업무 구체성:
   - 분석 항목: {responsibility.get('keyword_count', 0)}개
   - 주요 키워드: {', '.join(responsibility.get('keywords', [])) if responsibility.get('keywords') else '없음'}
   - 판단 근거: {responsibility.get('reasoning', '없음')}

2. 자격 요건 구체성:
   - 분석 항목: {qualification.get('keyword_count', 0)}개
   - 주요 키워드: {', '.join(qualification.get('keywords', [])) if qualification.get('keywords') else '없음'}
   - 판단 근거: {qualification.get('reasoning', '없음')}

3. 키워드 적합성:
   - 분석 항목: {keyword_relevance.get('keyword_count', 0)}개
   - 주요 키워드: {', '.join(keyword_relevance.get('keywords', [])) if keyword_relevance.get('keywords') else '없음'}
   - 판단 근거: {keyword_relevance.get('reasoning', '없음')}

4. 필수 항목 포함:
   - 포함 항목: {required_fields.get('keyword_count', 0)}개
   - 항목 목록: {', '.join(required_fields.get('keywords', [])) if required_fields.get('keywords') else '없음'}
   - 판단 근거: {required_fields.get('reasoning', '없음')}

[매력도 평가]

1. 특별 콘텐츠 포함:
   - 포함 개수: {content_count.get('keyword_count', 0)}개
   - 콘텐츠 종류: {', '.join(content_count.get('keywords', [])) if content_count.get('keywords') else '없음'}
   - 판단 근거: {content_count.get('reasoning', '없음')}

2. 특별 콘텐츠 충실도:
   - 평가 항목: {content_quality.get('keyword_count', 0)}개
   - 주요 요소: {', '.join(content_quality.get('keywords', [])) if content_quality.get('keywords') else '없음'}
   - 판단 근거: {content_quality.get('reasoning', '없음')}
"""
        return result.strip()
        
    except json.JSONDecodeError as e:
        return f"[오류] JSON 파싱 오류: {str(e)}"
    except Exception as e:
        return f"[오류] 파일 읽기 오류: {str(e)}"


@tool
def list_available_evaluations() -> str:
    """
    data/report/ 디렉토리에 있는 평가 JSON 파일 목록을 반환합니다.
    
    Returns:
        사용 가능한 JSON 파일 목록
    """
    data_dir = Path("data/report")
    
    if not data_dir.exists():
        return "[오류] data/report 디렉토리가 존재하지 않습니다."
    
    json_files = list(data_dir.glob("*.json"))
    
    if not json_files:
        return "[안내] 평가 데이터가 없습니다. Phase 1을 먼저 실행하세요."
    
    result = "=== 사용 가능한 평가 데이터 ===\n\n"
    for i, file_path in enumerate(json_files, 1):
        # 파일명에서 기본 정보 추출
        filename = file_path.name
        result += f"{i}. {filename}\n"
        
        # 파일 내용에서 추가 정보 로드
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                result += f"   - 제목: {data.get('title', 'N/A')}\n"
                result += f"   - 회사: {data.get('company', 'N/A')}\n"
                result += f"   - 평가 시간: {data.get('timestamp', 'N/A')}\n\n"
        except:
            result += "   (파일 읽기 실패)\n\n"
    
    return result.strip()


@tool
def delete_evaluation_json(json_filename: str) -> str:
    """
    처리 완료된 평가 JSON 파일을 삭제합니다.
    
    Args:
        json_filename: 삭제할 JSON 파일명
        
    Returns:
        삭제 결과 메시지
    """
    data_dir = Path("data/report")
    file_path = data_dir / json_filename
    
    if not file_path.exists():
        return f"[오류] 파일을 찾을 수 없습니다: {json_filename}"
    
    try:
        os.remove(file_path)
        return f"[완료] 파일 삭제 완료: {json_filename}"
    except Exception as e:
        return f"[오류] 파일 삭제 오류: {str(e)}"

