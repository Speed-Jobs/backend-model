"""
All Tools 통합 테스트
DB에서 id 2, 3에 해당되는 공고를 불러와서 모든 tools를 테스트합니다.
- Readability Tools (가독성)
- Specificity Tools (구체성)
- Attractiveness Tools (매력도)
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 파일 로드
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from app.db.config.base import SessionLocal
# 모든 모델을 import하여 SQLAlchemy 관계 초기화
from app.models.company import Company
from app.models.post import Post
from app.models.industry import Industry
from app.models.position import Position
from app.models.skill import Skill
from app.models.post_skill import PostSkill
from app.models.position_skill import PositionSkill
from app.models.industry_skill import IndustrySkill
from app.models.dashboard_stat import DashboardStat

# Readability Tools
from app.core.agents.tools.tool_readability import (
    measure_company_jargon_frequency,
    measure_paragraph_consistency,
    measure_grammar_accuracy
)

# Specificity Tools
from app.core.agents.tools.tool_specificity import (
    measure_responsibility_specificity,
    measure_qualification_specificity,
    measure_keyword_relevance,
    measure_required_fields_count
)

# Attractiveness Tools
from app.core.agents.tools.tool_attractiveness import (
    measure_special_content_count,
    measure_special_content_quality
)


def test_all_tools():
    """모든 Tools 통합 테스트"""
    # DB 세션 생성
    db = SessionLocal()
    
    try:
        # 테스트할 Post ID 리스트
        test_post_ids = [2, 3]
        
        for post_id in test_post_ids:
            print("\n" + "="*100)
            print(f"테스트 Post ID: {post_id}")
            print("="*100)
            
            # Post 조회
            post = db.query(Post).filter(Post.id == post_id).first()
            
            if not post:
                print(f"✗ Post ID {post_id}를 찾을 수 없습니다.")
                continue
            
            # 회사명 가져오기
            company_name = post.company.name if post.company else ""
            
            print(f"\n[공고 정보]")
            print(f"제목: {post.title}")
            print(f"회사명: {company_name}")
            print(f"Description 길이: {len(post.description) if post.description else 0} 글자")
            
            if not post.description:
                print(f"✗ Post ID {post_id}에 description이 없습니다.")
                continue
            
            # Description 미리보기
            print(f"\n[Description 미리보기 (처음 300자)]")
            print("-"*100)
            print(post.description[:300] + "..." if len(post.description) > 300 else post.description)
            print("-"*100)
            
            # ========== Readability Tools 테스트 ==========
            print(f"\n{'#'*100}")
            print("### READABILITY TOOLS (가독성) ###")
            print(f"{'#'*100}")
            
            print(f"\n{'='*100}")
            print("[Readability 1] 사내 전문 용어 빈도수 측정")
            print("="*100)
            try:
                result = measure_company_jargon_frequency.invoke({
                    "job_description": post.description,
                    "company_name": company_name
                })
                print(result)
            except Exception as e:
                print(f"✗ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{'='*100}")
            print("[Readability 2] 문단 일관성 측정")
            print("="*100)
            try:
                result = measure_paragraph_consistency.invoke({
                    "job_description": post.description
                })
                print(result)
            except Exception as e:
                print(f"✗ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{'='*100}")
            print("[Readability 3] 문법 정확성 측정")
            print("="*100)
            try:
                result = measure_grammar_accuracy.invoke({
                    "job_description": post.description
                })
                print(result)
            except Exception as e:
                print(f"✗ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            
            # ========== Specificity Tools 테스트 ==========
            print(f"\n{'#'*100}")
            print("### SPECIFICITY TOOLS (구체성) ###")
            print(f"{'#'*100}")
            
            print(f"\n{'='*100}")
            print("[Specificity 1] 담당 업무 구체성 측정")
            print("="*100)
            try:
                result = measure_responsibility_specificity.invoke({
                    "job_description": post.description
                })
                print(result)
            except Exception as e:
                print(f"✗ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{'='*100}")
            print("[Specificity 2] 자격요건 구체성 측정")
            print("="*100)
            try:
                result = measure_qualification_specificity.invoke({
                    "job_description": post.description
                })
                print(result)
            except Exception as e:
                print(f"✗ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{'='*100}")
            print("[Specificity 3] 키워드 관련성 측정")
            print("="*100)
            try:
                result = measure_keyword_relevance.invoke({
                    "job_description": post.description
                })
                print(result)
            except Exception as e:
                print(f"✗ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{'='*100}")
            print("[Specificity 4] 필수 필드 개수 측정")
            print("="*100)
            try:
                result = measure_required_fields_count.invoke({
                    "job_description": post.description
                })
                print(result)
            except Exception as e:
                print(f"✗ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            
            # ========== Attractiveness Tools 테스트 ==========
            print(f"\n{'#'*100}")
            print("### ATTRACTIVENESS TOOLS (매력도) ###")
            print(f"{'#'*100}")
            
            print(f"\n{'='*100}")
            print("[Attractiveness 1] 특별 콘텐츠 포함 여부 측정")
            print("="*100)
            try:
                result = measure_special_content_count.invoke({
                    "job_description": post.description
                })
                print(result)
            except Exception as e:
                print(f"✗ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{'='*100}")
            print("[Attractiveness 2] 특별 콘텐츠 품질 측정")
            print("="*100)
            try:
                result = measure_special_content_quality.invoke({
                    "job_description": post.description
                })
                print(result)
            except Exception as e:
                print(f"✗ 오류 발생: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"\n{'='*100}")
            print(f"Post ID {post_id} 테스트 완료")
            print("="*100)
        
        print("\n" + "="*100)
        print("🎉 모든 테스트 완료!")
        print("="*100)
        
    except Exception as e:
        print(f"\n✗ 전체 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        print("\n[문제 해결 방법]")
        print("1. Kubernetes 포트 포워딩 확인:")
        print("   kubectl port-forward -n skala-practice svc/speedjobs-mysql 3306:3306")
        print("\n2. 포트 포워딩이 이미 실행 중이면 재시작:")
        print("   기존 포트 포워딩 프로세스 종료 후 다시 실행")
        print("\n3. MySQL 서버 상태 확인:")
        print("   kubectl get pods -n skala-practice | grep mysql")
    finally:
        try:
            db.close()
        except:
            pass


if __name__ == "__main__":
    test_all_tools()

