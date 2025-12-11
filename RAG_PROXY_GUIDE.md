# RAG Proxy Guide

backend-model은 RAG 요청을 vectorDB-server로 프록시합니다.

## 🏗️ 아키텍처

```
Frontend/Client
      ↓
backend-model (Port 7777)
  └─ /rag/search → Proxy
      ↓
vectorDB-server (Port 8000)
  └─ /search/agentic → RAG Agent 실행
      ↓
    LangGraph + GPT-4o + Tavily + SQL
```

## 🚀 설정

### 1. 환경 변수

`.env` 파일에 vectorDB-server URL 추가:

```bash
# VectorDB Server URL
VECTORDB_SERVER_URL=http://localhost:8000
```

### 2. vectorDB-server 실행

먼저 vectorDB-server를 실행해야 합니다:

```bash
cd ../vectorDB-server
python -m app.main
```

vectorDB-server는 `http://localhost:8000`에서 실행됩니다.

### 3. backend-model 실행

```bash
cd backend-model
python -m app.main
```

backend-model은 `http://localhost:7777`에서 실행됩니다.

## 📡 API 사용

### RAG 검색 (Proxy)

```bash
POST http://localhost:7777/rag/search

{
  "text": "2025년 하반기 토스 채용공고 총 몇개야?"
}
```

이 요청은 자동으로 vectorDB-server로 전달됩니다:
```
backend-model → http://localhost:8000/search/agentic
```

### Health Check

```bash
GET http://localhost:7777/rag/health
```

vectorDB-server의 연결 상태를 확인합니다.

**정상 응답:**
```json
{
  "status": "healthy",
  "vectordb_server": "connected",
  "vectordb_server_url": "http://localhost:8000",
  "vectordb_server_health": {
    // vectorDB-server의 health 정보
  }
}
```

**연결 실패:**
```json
{
  "status": "unhealthy",
  "vectordb_server": "disconnected",
  "vectordb_server_url": "http://localhost:8000",
  "error": "Cannot connect to vectorDB-server"
}
```

## ⚠️ 주의사항

1. **vectorDB-server 필수**: RAG 기능을 사용하려면 vectorDB-server가 실행 중이어야 합니다.

2. **포트 충돌 방지**:
   - backend-model: 7777
   - vectorDB-server: 8000

3. **타임아웃**: RAG 검색은 최대 120초까지 대기합니다.

## 🔍 에러 처리

### 503: Service Unavailable
```json
{
  "detail": "Cannot connect to vectorDB-server at http://localhost:8000. Please ensure it's running."
}
```
**해결**: vectorDB-server를 실행하세요.

### 500: Internal Server Error
vectorDB-server에서 오류가 발생했습니다. vectorDB-server 로그를 확인하세요.

## 📂 파일 구조

```
backend-model/
├── app/
│   └── routers/
│       └── rag_retrieval.py    # RAG 프록시 라우터 (간단함)
│
vectorDB-server/
├── app/
│   ├── agents/                  # RAG Agent 실제 구현
│   │   ├── orchestrator.py
│   │   └── subagents/
│   ├── tools/
│   ├── prompts/
│   └── routers/
│       └── retrieval.py         # /search/agentic 엔드포인트
```

## 🎯 장점

1. **관심사 분리**:
   - backend-model: 비즈니스 로직 + 라우팅
   - vectorDB-server: RAG/AI 전문 처리

2. **간단한 유지보수**:
   - RAG 관련 코드는 vectorDB-server에만 존재
   - backend-model은 프록시 역할만 수행

3. **독립적 배포**:
   - 각 서버를 독립적으로 배포/스케일 가능

4. **명확한 책임**:
   - Agent execute는 vectorDB-server에서만 실행

## 🔗 관련 문서

- [vectorDB-server/AGENT_ARCHITECTURE.md](../vectorDB-server/AGENT_ARCHITECTURE.md)
- [vectorDB-server/README.md](../vectorDB-server/README.md)
