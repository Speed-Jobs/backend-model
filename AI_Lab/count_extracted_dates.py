"""
채용 일정 추출 결과에서 각 필드별로 날짜가 추출된 항목의 개수를 집계하는 스크립트
"""
import json
from pathlib import Path


def count_extracted_dates(json_file_path: str):
    """
    JSON 파일을 읽어서 각 필드별로 빈 배열이 아닌 항목의 개수를 집계합니다.
    
    Args:
        json_file_path: 분석할 JSON 파일 경로
    """
    # JSON 파일 읽기
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 각 필드별로 빈 배열이 아닌 항목 개수 세기
    counts = {
        'application_date': 0,
        'document_screening_date': 0,
        'first_interview': 0,
        'second_interview': 0,
        'join_date': 0
    }
    
    # 각 필드별로 추출된 post_id 리스트
    post_ids = {
        'application_date': [],
        'document_screening_date': [],
        'first_interview': [],
        'second_interview': [],
        'join_date': []
    }
    
    for item in data:
        post_id = item.get('post_id')
        if post_id is None:
            continue
            
        if item.get('application_date') and len(item['application_date']) > 0:
            counts['application_date'] += 1
            post_ids['application_date'].append(post_id)
        if item.get('document_screening_date') and len(item['document_screening_date']) > 0:
            counts['document_screening_date'] += 1
            post_ids['document_screening_date'].append(post_id)
        if item.get('first_interview') and len(item['first_interview']) > 0:
            counts['first_interview'] += 1
            post_ids['first_interview'].append(post_id)
        if item.get('second_interview') and len(item['second_interview']) > 0:
            counts['second_interview'] += 1
            post_ids['second_interview'].append(post_id)
        if item.get('join_date') and len(item['join_date']) > 0:
            counts['join_date'] += 1
            post_ids['join_date'].append(post_id)
    
    # 결과 출력
    print(f"=== {Path(json_file_path).name} 분석 결과 ===")
    print(f"전체 항목 수: {len(data)}")
    print(f"\n각 필드별 추출된 항목 수:")
    print(f"  - application_date: {counts['application_date']} ({counts['application_date']/len(data)*100:.1f}%)")
    print(f"  - document_screening_date: {counts['document_screening_date']} ({counts['document_screening_date']/len(data)*100:.1f}%)")
    print(f"  - first_interview: {counts['first_interview']} ({counts['first_interview']/len(data)*100:.1f}%)")
    print(f"  - second_interview: {counts['second_interview']} ({counts['second_interview']/len(data)*100:.1f}%)")
    print(f"  - join_date: {counts['join_date']} ({counts['join_date']/len(data)*100:.1f}%)")
    
    print(f"\n각 필드별 추출된 post_id:")
    for field, ids in post_ids.items():
        print(f"  - {field}: {ids}")
    
    return counts, post_ids


if __name__ == "__main__":
    # 기본 파일 경로
    default_file = "output/recruitment_schedules_all_competitors_20251202_101818.json"
    
    import sys
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = default_file
    
    count_extracted_dates(json_file)

