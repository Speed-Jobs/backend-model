# RAG Agent System Migration Guide

vectorDB-server의 Agent 기반 RAG 시스템을 backend-model로 통합한 마이그레이션 가이드입니다.

## 📁 디렉터리 구조

```
backend-model/
├── app/
│   ├── core/
│   │   └── agents/
│   │       ├── chatbot/          # 기존 챗봇 에이전트
│   │       ├── dashboard/        # 기존 대시보드 에이전트
│   │       ├── job_posting/      # 기존 채용공고 에이전트
│   │       └── rag/              # 🆕 새로운 RAG 에이전트 시스템
│   │           ├── __init__.py
│   │           ├── base_agent.py       # Base Agent 클래스
│   │           ├── orchestrator.py     # LangGraph 워크플로우 오케스트레이터
│   │           ├── subagents/          # 전문화된 Sub-agents
│   │           │   ├── __init__.py
│   │           │   ├── router_agent.py         # 라우팅 & 엔티티 추출
│   │           │   ├── vector_search_agent.py  # VectorDB 검색
│   │           │   ├── web_search_agent.py     # 웹 검색
│   │           │   ├── sql_analysis_agent.py   # SQL 통계 분석
│   │           │   └── generator_agent.py      # 답변 생성
│   │           ├── tools/              # Agent 도구들
│   │           │   ├── __init__.py
│   │           │   ├── vector_search.py        # VectorDB 검색 도구
│   │           │   ├── web_search.py           # 웹 검색 도구 (Tavily)
│   │           │   ├── database_query.py       # SQL 쿼리 실행 도구
│   │           │   └── helpers.py              # 유틸리티 함수
│   │           ├── prompts/            # Prompt 템플릿
│   │           │   ├── __init__.py
│   │           │   └── system_prompts.py       # 모든 시스템 프롬프트
│   │           └── memory/             # Memory/State 관리
│   │               ├── __init__.py
│   │               └── states.py               # Agent State 정의
│   │
│   └── routers/
│       └── rag_retrieval.py          # 🆕 RAG 엔드포인트
│
├── requirements.txt                   # RAG 의존성 추가됨
└── RAG_MIGRATION_GUIDE.md            # 이 파일
```

## 🔧 마이그레이션 내용

### 1. Agent 시스템 통합

vectorDB-server의 agent 개념을 backend-model의 `app/core/agents/rag/` 디렉터리로 통합했습니다.

**기존 (vectorDB-server):**
```
vectorDB-server/app/agents/
├── base_agent.py
├── orchestrator.py
└── subagents/
```

**현재 (backend-model):**
```
backend-model/app/core/agents/rag/
├── base_agent.py
├── orchestrator.py
└── subagents/
```

### 2. 라우팅 분리

Execute 같은 수행 기능은 agent 내부에 유지하되, HTTP 엔드포인트는 라우터로 분리했습니다.

- **Agent 실행**: `orchestrator.execute()` - 워크플로우 실행
- **HTTP 엔드포인트**: `app/routers/rag_retrieval.py` - FastAPI 라우터

### 3. 주요 변경 사항

#### Import 경로 변경
```python
# Before (vectorDB-server)
from app.agents.base_agent import BaseAgent
from app.tools.vector_search import VectorSearchTool
from app.prompts.system_prompts import ROUTER_SYSTEM_PROMPT

# After (backend-model)
from app.core.agents.rag.base_agent import BaseAgent
from app.core.agents.rag.tools.vector_search import VectorSearchTool
from app.core.agents.rag.prompts.system_prompts import ROUTER_SYSTEM_PROMPT
```

#### 설정 관리
- vectorDB-server는 `settings` 객체 사용
- backend-model은 환경변수 직접 사용 (`os.getenv()`)

```python
# Before
from app.core.config import settings
api_key = settings.OPENAI_API_KEY

# After
import os
api_key = os.getenv("OPENAI_API_KEY")
```

## 🚀 사용 방법

### 1. 의존성 설치

```bash
cd backend-model
pip install -r requirements.txt
```

새로 추가된 의존성:
- `langgraph>=0.2.0` - 멀티 에이전트 워크플로우
- `tavily-python>=0.3.0` - 웹 검색 API
- `pymysql>=1.0.0` - MySQL 드라이버

### 2. 환경 변수 설정

`.env` 파일에 다음 변수들을 추가하세요:

```bash
# AI/LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here

# Database Configuration (이미 설정되어 있을 수 있음)
DB_HOST=localhost
DB_PORT=3306
DB_USER=admin
DB_PASSWORD=admin
DB_NAME=speedjobs
```

### 3. 서버 실행

```bash
cd backend-model
python -m app.main
```

서버는 `http://localhost:7777`에서 실행됩니다.

### 4. API 엔드포인트 사용

#### RAG 검색 엔드포인트

```bash
POST /rag/search
```

**요청 예시:**
```bash
curl -X POST "http://localhost:7777/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "2025년 하반기 토스 채용공고 총 몇개야?"
  }'
```

**응답 예시:**
```json
{
  "query": "2025년 하반기 토스 채용공고 총 몇개야?",
  "answer": "2025년 하반기 토스 채용공고는 총 25개입니다...",
  "sources": [],
  "route_decision": "statistics_with_stats",
  "total_sources": 0
}
```

#### Health Check 엔드포인트

```bash
GET /rag/health
```

**응답 예시:**
```json
{
  "status": "healthy",
  "components": {
    "openai": "configured",
    "tavily": "configured",
    "database": "configured"
  }
}
```

## 🔍 RAG Agent 워크플로우

```
사용자 질문
    ↓
RouterAgent (엔티티 추출 & 라우팅 결정)
    ↓
┌───────────────┼───────────────┐
↓               ↓               ↓
VectorSearch   WebSearch   SQLAnalysis
Agent          Agent       Agent
↓               ↓               ↓
└───────────────┼───────────────┘
                ↓
        GeneratorAgent
                ↓
        종합 답변 + 출처
```

### Agent 역할

1. **RouterAgent**: 질문 분석 및 최적 경로 결정
2. **VectorSearchAgent**: VectorDB(Qdrant)에서 관련 문서 검색
3. **WebSearchAgent**: Tavily API로 웹 검색
4. **SQLAnalysisAgent**: MySQL 통계 쿼리 생성 및 실행
5. **GeneratorAgent**: 수집된 정보를 종합하여 최종 답변 생성

## ⚠️ 주의사항

### VectorSearchTool 구현 필요

현재 `app/core/agents/rag/tools/vector_search.py`는 placeholder 구현입니다.
실제 VectorDB retriever를 구현해야 합니다:

```python
# TODO: Implement VectorDB retriever
# Example:
from app.services.vector_retriever import VectorRetriever

class VectorSearchTool:
    def __init__(self):
        self.retriever = VectorRetriever()  # 실제 구현 필요
```

기존 chatbot agent의 vector_search를 참고할 수 있습니다:
- [app/core/agents/chatbot/tools/vector_search.py](app/core/agents/chatbot/tools/vector_search.py)

### 환경 변수 설정

RAG 시스템을 사용하려면 다음 환경 변수가 필수입니다:
- `OPENAI_API_KEY` - GPT-4o API 키
- `TAVILY_API_KEY` - 웹 검색 API 키
- Database 설정 (`DB_*`)

설정되지 않은 경우 해당 기능이 동작하지 않습니다.

## 📝 통합 체크리스트

- [x] Agent 클래스 마이그레이션 (base_agent, orchestrator)
- [x] Subagent 마이그레이션 (router, vector_search, web_search, sql_analysis, generator)
- [x] Tools 마이그레이션 (vector_search, web_search, database_query, helpers)
- [x] Prompts 마이그레이션 (system_prompts)
- [x] Memory/State 마이그레이션 (states)
- [x] RAG 라우터 생성 (rag_retrieval.py)
- [x] main.py에 라우터 등록
- [x] requirements.txt 업데이트
- [ ] VectorSearchTool 실제 구현 연결
- [ ] 환경 변수 설정 (.env)
- [ ] 통합 테스트

## 🔗 관련 문서

- [vectorDB-server AGENT_ARCHITECTURE.md](../vectorDB-server/AGENT_ARCHITECTURE.md)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Tavily API](https://tavily.com/)

## 📞 도움말

질문이나 문제가 있으면 팀 슬랙 채널 또는 이슈 트래커에 문의하세요.
