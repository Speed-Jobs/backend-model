"""Data Collector 테스트 스크립트 (Phase 1)"""
import sys
from pathlib import Path

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.orchestrator.data_collector import collect_multiple_posts

def main():
    """Data Collector로 2개의 채용 공고 데이터 수집"""

    # 평가할 채용 공고 ID
    post_ids = [2, 3]

    print("="*80)
    print(f"채용 공고 평가 시작: Post IDs {post_ids}")
    print("="*80)
    print()

    try:
        # Data Collector 실행
        results = collect_multiple_posts(
            post_ids=post_ids,
            llm_model="gpt-4o"  # Phase 1은 gpt-4o-mini 사용
        )

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
                
                print(f"\n[가독성 도구 결과]")
                for tool_name, tool_result in raw_results.get('readability', {}).items():
                    print(f"\n  • {tool_name}:")
                    print(f"    {tool_result[:300]}..." if len(str(tool_result)) > 300 else f"    {tool_result}")
                
                print(f"\n[구체성 도구 결과]")
                for tool_name, tool_result in raw_results.get('specificity', {}).items():
                    print(f"\n  • {tool_name}:")
                    print(f"    {tool_result[:300]}..." if len(str(tool_result)) > 300 else f"    {tool_result}")
                
                print(f"\n[매력도 도구 결과]")
                for tool_name, tool_result in raw_results.get('attractiveness', {}).items():
                    print(f"\n  • {tool_name}:")
                    print(f"    {tool_result[:300]}..." if len(str(tool_result)) > 300 else f"    {tool_result}")

            print(f"\n📋 종합 평가 보고서:")
            print("="*80)
            print(result['summary'])
            print("="*80)

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
