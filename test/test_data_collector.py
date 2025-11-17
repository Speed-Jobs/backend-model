"""Data Collector 테스트 스크립트 (Phase 1)"""
import sys
from pathlib import Path
import json

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.evaluation import collect_multiple_posts

def main():
    """Data Collector로 2개의 채용 공고 데이터 수집"""

    # 평가할 채용 공고 ID
    post_ids = [2, 3]

    print("="*80)
    print(f"채용 공고 평가 시작: Post IDs {post_ids}")
    print("="*80)
    print()

    try:
        # Data Collector 실행 (Phase 1: 원형 데이터만 수집)
        results = collect_multiple_posts(post_ids=post_ids)

        print("\n" + "="*80)
        print("평가 결과:")
        print("="*80)

        for post_key, result in results.items():
            print(f"\n### {post_key.upper()} ###")

            # Check if error occurred
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue

            print(f"Post ID: {result['post_id']}")
            print(f"Title: {result['title']}")
            print(f"Company: {result['company']}")

            print(f"\n📊 Tool 원형 결과:")
            print("-"*80)
            
            if 'raw_results' in result:
                raw_results = result['raw_results']
            elif 'raw_evaluation_results' in result:
                raw_results = result['raw_evaluation_results']
            else:
                raw_results = {}
                
                print(f"\n[가독성 도구 결과]")
                for tool_name, tool_result in raw_results.get('readability', {}).items():
                    print(f"\n  • {tool_name}:")
                    if isinstance(tool_result, dict):
                        print(f"    - 키워드 개수: {tool_result.get('keyword_count', 0)}개")
                        print(f"    - 키워드: {', '.join(tool_result.get('keywords', [])) if tool_result.get('keywords') else '없음'}")
                        reasoning = tool_result.get('reasoning', '')
                        print(f"    - 판단 근거: {reasoning[:100]}..." if len(reasoning) > 100 else f"    - 판단 근거: {reasoning}")
                    else:
                        print(f"    {tool_result}")
                
                print(f"\n[구체성 도구 결과]")
                for tool_name, tool_result in raw_results.get('specificity', {}).items():
                    print(f"\n  • {tool_name}:")
                    if isinstance(tool_result, dict):
                        print(f"    - 키워드 개수: {tool_result.get('keyword_count', 0)}개")
                        print(f"    - 키워드: {', '.join(tool_result.get('keywords', [])) if tool_result.get('keywords') else '없음'}")
                        reasoning = tool_result.get('reasoning', '')
                        print(f"    - 판단 근거: {reasoning[:100]}..." if len(reasoning) > 100 else f"    - 판단 근거: {reasoning}")
                    else:
                        print(f"    {tool_result}")
                
                print(f"\n[매력도 도구 결과]")
                for tool_name, tool_result in raw_results.get('attractiveness', {}).items():
                    print(f"\n  • {tool_name}:")
                    if isinstance(tool_result, dict):
                        print(f"    - 키워드 개수: {tool_result.get('keyword_count', 0)}개")
                        print(f"    - 키워드: {', '.join(tool_result.get('keywords', [])) if tool_result.get('keywords') else '없음'}")
                        reasoning = tool_result.get('reasoning', '')
                        print(f"    - 판단 근거: {reasoning[:100]}..." if len(reasoning) > 100 else f"    - 판단 근거: {reasoning}")
                    else:
                        print(f"    {tool_result}")

            print(f"\n💾 저장된 파일:")
            print("="*80)
            if 'saved_file' in result:
                print(f"원형 데이터: {result['saved_file']}")
            print("="*80)
            print("\n✅ Phase 1 완료: 원형 데이터 수집 및 저장")
            print("📌 Phase 2에서 이 데이터를 사용하여 보고서를 생성할 수 있습니다.")

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
