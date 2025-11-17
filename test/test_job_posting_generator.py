"""
Phase 2 Agent 테스트: AI 채용 공고 생성
"""

import asyncio
from app.core.agents import generate_improved_job_posting_async


async def test_generator():
    """
    AI 채용 공고 생성 Agent 테스트
    """
    print("=" * 80)
    print("Phase 2: AI 채용 공고 생성 Agent 테스트")
    print("=" * 80)
    
    # data/report/ 에 있는 JSON 파일을 자동으로 찾아 처리
    result = await generate_improved_job_posting_async(
        json_filename=None,  # None이면 사용 가능한 첫 번째 파일 처리
        llm_model="gpt-4o"
    )
    
    print("\n" + "=" * 80)
    print("개선된 AI 채용 공고 (전체)")
    print("=" * 80)
    print(f"제목: {result.get('title', 'N/A')}")
    print(f"회사: {result.get('company', 'N/A')}")
    print(f"처리 파일: {result['original_file']}")
    print("=" * 80)
    print(result['improved_posting'])
    print("\n" + "=" * 80)
    print(f"상태: {result['status']}")
    print(f"메시지: {result['message']}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_generator())

