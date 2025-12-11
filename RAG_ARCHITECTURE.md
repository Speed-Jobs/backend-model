# RAG 아키텍처 - 서비스 분리 구조

## 📐 설계 원칙

**"Agent의 execute 로직은 vectorDB-server에서만 실행"**

backend-model은 비즈니스 로직과 라우팅에만 집중하고, RAG AI Agent의 실제 실행은 vectorDB-server에 위임합니다.

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend/Client                           │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP Request
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              backend-model (Port 7777)                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  POST /rag/search                                     │   │
│  │  - 요청 검증                                           │   │
│  │  - vectorDB-server로 프록시                           │   │
│  │  - 응답 전달                                           │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP Proxy (httpx)
                         ↓
┌─────────────────────────────────────────────────────────────┐
│           vectorDB-server (Port 8000)                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  POST /search/agentic                                 │   │
│  │                                                        │   │
│  │  ┌─────────────────────────────────────────────┐     │   │
│  │  │  AgenticRAGOrchestrator                      │     │   │
│  │  │  (LangGraph Workflow)                        │     │   │
│  │  │                                               │     │   │
│  │  │  1. RouterAgent                              │     │   │
│  │  │     - 엔티티 추출                             │     │   │
│  │  │     - 라우팅 결정                             │     │   │
│  │  │                                               │     │   │
│  │  │  2. VectorSearchAgent / WebSearchAgent       │     │   │
│  │  │     - VectorDB 검색                           │     │   │
│  │  │     - 웹 검색 (Tavily)                        │     │   │
│  │  │                                               │     │   │
│  │  │  3. SQLAnalysisAgent                         │     │   │
│  │  │     - SQL 쿼리 생성                           │     │   │
│  │  │     - MySQL 통계 분석                         │     │   │
│  │  │                                               │     │   │
│  │  │  4. GeneratorAgent                           │     │   │
│  │  │     - GPT-4o로 답변 생성                      │     │   │
│  │  └─────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 📂 디렉터리 구조

### backend-model (Port 7777)
```
backend-model/
├── app/
│   ├── routers/
│   │   └── rag_retrieval.py          # 🔀 RAG 프록시 라우터 (간단)
│   ├── core/
│   │   └── agents/
│   │       ├── chatbot/               # 기존 챗봇 에이전트
│   │       ├── dashboard/             # 기존 대시보드 에이전트
│   │       └── job_posting/           # 기존 채용공고 에이전트
│   └── ...
└── requirements.txt                   # httpx만 추가됨
```

### vectorDB-server (Port 8000)
```
vectorDB-server/
├── app/
│   ├── agents/                         # 🤖 RAG Agent 실제 구현
│   │   ├── base_agent.py
│   │   ├── orchestrator.py            # LangGraph 워크플로우
│   │   └── subagents/
│   │       ├── router_agent.py        # execute() 메서드
│   │       ├── vector_search_agent.py # execute() 메서드
│   │       ├── web_search_agent.py    # execute() 메서드
│   │       ├── sql_analysis_agent.py  # execute() 메서드
│   │       └── generator_agent.py     # execute() 메서드
│   ├── tools/                          # Agent 도구들
│   ├── prompts/                        # 프롬프트 템플릿
│   ├── memory/                         # State 관리
│   └── routers/
│       └── retrieval.py                # /search/agentic 엔드포인트
└── requirements.txt                    # langgraph, tavily 등
```

## 🔄 요청 흐름

### 1. 클라이언트 요청
```bash
POST http://localhost:7777/rag/search
Content-Type: application/json

{
  "text": "2025년 하반기 토스 채용공고 총 몇개야?"
}
```

### 2. backend-model 처리
```python
# app/routers/rag_retrieval.py

@router.post("/search")
async def agentic_rag_search(query: AgenticRAGQuery):
    # 🔀 vectorDB-server로 프록시
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{VECTORDB_SERVER_URL}/search/agentic",
            json={"text": query.text}
        )
        return response.json()
```

### 3. vectorDB-server 실행
```python
# vectorDB-server/app/routers/retrieval.py

@router.post("/search/agentic")
async def agentic_search(query: AgenticRAGQuery, db: Session = Depends(get_db)):
    # 🤖 Agent 실행
    service = AgenticRAGService()
    result = await service.search(query=query.text, db=db)
    return result
```

### 4. Agent Workflow 실행
```python
# vectorDB-server/app/agents/orchestrator.py

class AgenticRAGOrchestrator:
    async def execute(self, state: AgenticRAGState):
        # 1. RouterAgent.execute() 실행
        # 2. VectorSearchAgent.execute() 또는 WebSearchAgent.execute()
        # 3. SQLAnalysisAgent.execute() (필요시)
        # 4. GeneratorAgent.execute()
        return final_state
```

## ✅ 장점

### 1. **명확한 관심사 분리**
- **backend-model**: 비즈니스 로직, 라우팅, DB 모델
- **vectorDB-server**: RAG, AI Agent, LLM 처리

### 2. **코드 중복 제거**
- Agent 로직은 vectorDB-server에만 존재
- backend-model은 단순 프록시 (100줄 미만)

### 3. **독립적 배포**
- 각 서비스를 독립적으로 배포/스케일 가능
- AI 기능 업데이트 시 vectorDB-server만 재배포

### 4. **성능 최적화**
- vectorDB-server를 GPU 서버에 배포 가능
- backend-model은 일반 서버에서 운영

### 5. **에러 격리**
- AI Agent 오류가 backend-model에 영향 없음
- 각 서비스의 health check 독립적

## 🔧 설정

### backend-model `.env`
```bash
# VectorDB Server URL
VECTORDB_SERVER_URL=http://localhost:8000

# Database (기존 설정 유지)
DB_HOST=localhost
DB_PORT=3306
DB_USER=admin
DB_PASSWORD=admin
DB_NAME=speedjobs
```

### vectorDB-server `.env`
```bash
# AI/LLM
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key

# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=admin
DB_PASSWORD=admin
DB_NAME=speedjobs

# VectorDB
QDRANT_URL=http://localhost:6333
```

## 🚀 실행 순서

```bash
# 1. vectorDB-server 실행 (먼저!)
cd vectorDB-server
python -m app.main
# → http://localhost:8000

# 2. backend-model 실행
cd backend-model
python -m app.main
# → http://localhost:7777
```

## 📊 비교: 이전 vs 현재

### 이전 설계 (통합)
```
backend-model/
├── app/core/agents/rag/          ❌ 복잡한 Agent 로직
│   ├── orchestrator.py           ❌ LangGraph 실행
│   ├── subagents/                ❌ 5개 Agent + execute()
│   ├── tools/                    ❌ VectorDB, Web, SQL tools
│   └── prompts/                  ❌ 모든 프롬프트
└── requirements.txt              ❌ langgraph, tavily 의존성
```

### 현재 설계 (분리)
```
backend-model/
├── app/routers/
│   └── rag_retrieval.py          ✅ 100줄 미만 프록시
└── requirements.txt              ✅ httpx만 추가

vectorDB-server/
├── app/agents/                   ✅ 모든 Agent 로직
├── app/tools/                    ✅ 모든 Tools
└── requirements.txt              ✅ AI 관련 의존성
```

## 🎯 결론

**"Execute는 vectorDB-server에서, 라우팅은 backend-model에서"**

이 구조는 각 서비스의 책임을 명확히 하고, 유지보수와 확장을 쉽게 만듭니다.

## 🔗 관련 문서

- [RAG_PROXY_GUIDE.md](RAG_PROXY_GUIDE.md) - 프록시 설정 가이드
- [vectorDB-server/AGENT_ARCHITECTURE.md](../vectorDB-server/AGENT_ARCHITECTURE.md) - Agent 아키텍처 상세
