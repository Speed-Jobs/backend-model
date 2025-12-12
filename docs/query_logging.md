# SQL 쿼리 로깅 시스템

RAG 검색에서 "statistics_with_stats" 라우팅 시 LLM이 생성하는 SQL 쿼리를 모니터링하고 분석할 수 있는 로깅 시스템입니다.

## 📋 개요

이 시스템은 다음을 자동으로 로깅합니다:
1. **라우팅 결정**: 질문 분석 결과 및 라우팅 전략
2. **SQL 쿼리 생성**: LLM이 생성한 SQL 쿼리
3. **실행 결과**: 쿼리 실행 시간, 결과 개수, 성공/실패 여부

## 📂 로그 파일 위치

```
logs/sql_queries/
├── sql_queries_2024-12-12.jsonl          # SQL 쿼리 로그 (날짜별)
├── sql_queries_2024-12-13.jsonl
├── routing_decisions_2024-12-12.jsonl    # 라우팅 결정 로그 (날짜별)
└── routing_decisions_2024-12-13.jsonl
```

## 📊 로그 구조

### SQL 쿼리 로그 (sql_queries_*.jsonl)

```json
{
  "timestamp": "2024-12-12T10:30:45.123456",
  "question": "2025년 하반기 토스 채용공고 총 몇개야?",
  "route_decision": "statistics_with_stats",
  "extracted_entities": {
    "company_name": "토스",
    "year": 2025,
    "period": "하반기"
  },
  "query_info": {
    "query_type": "채용공고 개수 집계",
    "generated_sql": "SELECT COUNT(*) as count FROM posts WHERE ...",
    "llm_response": "원본 LLM 응답..."
  },
  "execution": {
    "success": true,
    "execution_time_ms": 45.23,
    "result_count": 25,
    "error": null
  }
}
```

### 라우팅 결정 로그 (routing_decisions_*.jsonl)

```json
{
  "timestamp": "2024-12-12T10:30:44.000000",
  "type": "routing_decision",
  "question": "2025년 하반기 토스 채용공고 총 몇개야?",
  "route_decision": "statistics_with_stats",
  "extracted_entities": {
    "company_name": "토스",
    "year": 2025,
    "period": "하반기"
  },
  "params": {
    "needs_stats": true,
    "top_k": 5,
    "reason": "질문에 '총 몇개'라는 집계 키워드가 포함되어 통계 쿼리가 필요함"
  },
  "llm_response": "원본 LLM 응답..."
}
```

## 🔍 로그 조회 방법

### 1. 명령줄 도구 사용

```bash
# 오늘의 SQL 쿼리 로그 보기
python -m app.utils.view_query_logs view

# 특정 날짜의 로그 보기
python -m app.utils.view_query_logs view --date 2024-12-12

# 라우팅 결정 로그 보기
python -m app.utils.view_query_logs view --type routing_decisions

# 최근 10개만 보기
python -m app.utils.view_query_logs view --limit 10

# 통계 보기
python -m app.utils.view_query_logs stats

# 특정 날짜의 통계
python -m app.utils.view_query_logs stats --date 2024-12-12

# 키워드로 검색
python -m app.utils.view_query_logs search --keyword "토스"
```

### 2. Python 코드에서 사용

```python
from app.utils.query_logger import get_query_logger

# QueryLogger 인스턴스 가져오기
logger = get_query_logger()

# 오늘의 로그 읽기
logs = logger.read_logs()

# 특정 날짜의 로그 읽기
logs = logger.read_logs(date="2024-12-12", log_type="sql_queries")

# 통계 가져오기
stats = logger.get_statistics()
print(f"Total queries: {stats['total_queries']}")
print(f"Success rate: {stats['success_rate']}")
```

### 3. 직접 파일 읽기

JSONL 형식이므로 각 줄이 독립적인 JSON 객체입니다:

```python
import json

with open('logs/sql_queries/sql_queries_2024-12-12.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        log_entry = json.loads(line)
        print(log_entry['question'])
        print(log_entry['query_info']['generated_sql'])
```

## 📈 통계 및 모니터링

### 일일 통계 예시

```
📊 SQL QUERY STATISTICS - 2024-12-12
================================================================================

📈 Overall Statistics:
  • Total Queries: 42
  • Successful: 40
  • Failed: 2
  • Success Rate: 95.2%
  • Avg Execution Time: 78.45ms

📊 Query Types Distribution:
  • 채용공고 개수 집계: 15 (35.7%)
  • 기술스택 통계: 12 (28.6%)
  • 회사별 채용 트렌드: 10 (23.8%)
  • 기간별 채용 추이: 5 (11.9%)

🎯 Route Decisions Distribution:
  • statistics_with_stats: 42 (100.0%)
```

## 🎯 활용 방안

### 1. 쿼리 품질 모니터링
- LLM이 생성한 SQL 쿼리의 정확성 확인
- 에러가 발생한 쿼리 패턴 분석
- 쿼리 실행 시간 최적화

### 2. 라우팅 정확도 분석
- 질문 유형별 라우팅 결정 패턴 파악
- 엔티티 추출 정확도 확인
- 라우팅 로직 개선 방향 도출

### 3. 사용자 패턴 분석
- 자주 묻는 질문 유형 파악
- 시간대별 쿼리 분포 분석
- 인기 있는 회사/직무 키워드 추출

### 4. 시스템 성능 분석
- 쿼리 실행 시간 추이 모니터링
- 실패율 추적 및 원인 분석
- 병목 지점 식별

## 🔧 설정

### 로그 디렉토리 변경

```python
from app.utils.query_logger import QueryLogger

# 커스텀 로그 디렉토리 사용
logger = QueryLogger(log_dir="custom_logs/queries")
```

### 로그 레벨 조정

로거는 자동으로 다음을 기록합니다:
- ✅ 모든 쿼리 생성 시도
- ✅ 실행 성공/실패 여부
- ✅ 에러 메시지 및 스택 트레이스

콘솔 출력을 비활성화하려면 `query_logger.py`의 `print()` 문을 주석 처리하세요.

## 📝 예제

### 예제 1: 실패한 쿼리 찾기

```python
from app.utils.query_logger import get_query_logger

logger = get_query_logger()
logs = logger.read_logs()

failed_queries = [log for log in logs if not log['execution']['success']]

for log in failed_queries:
    print(f"Question: {log['question']}")
    print(f"SQL: {log['query_info']['generated_sql']}")
    print(f"Error: {log['execution']['error']}")
    print("-" * 80)
```

### 예제 2: 느린 쿼리 분석

```python
from app.utils.query_logger import get_query_logger

logger = get_query_logger()
logs = logger.read_logs()

# 100ms 이상 소요된 쿼리 찾기
slow_queries = [
    log for log in logs 
    if log['execution'].get('execution_time_ms', 0) > 100
]

# 실행 시간 순으로 정렬
slow_queries.sort(
    key=lambda x: x['execution'].get('execution_time_ms', 0),
    reverse=True
)

for log in slow_queries[:10]:  # Top 10
    exec_time = log['execution']['execution_time_ms']
    print(f"{exec_time:.2f}ms - {log['question']}")
```

### 예제 3: 회사별 쿼리 통계

```python
from app.utils.query_logger import get_query_logger
from collections import Counter

logger = get_query_logger()
logs = logger.read_logs()

# 회사명 추출
companies = [
    log['extracted_entities'].get('company_name')
    for log in logs
    if log['extracted_entities'].get('company_name')
]

# 빈도수 계산
company_counts = Counter(companies)

print("회사별 쿼리 수:")
for company, count in company_counts.most_common(10):
    print(f"  {company}: {count}")
```

## 🚀 자동화

### Cron Job으로 일일 리포트 생성

```bash
# 매일 자정에 전날 통계 이메일 발송
0 0 * * * python -m app.utils.view_query_logs stats --date $(date -d "yesterday" +\%Y-\%m-\%d) | mail -s "Daily Query Stats" admin@example.com
```

### 로그 파일 정리

오래된 로그를 정기적으로 정리:

```bash
# 30일 이전 로그 삭제
find logs/sql_queries/ -name "*.jsonl" -mtime +30 -delete
```

## 📞 문의

로깅 시스템 관련 문의사항이나 개선 제안이 있으시면 개발팀에 연락해주세요.

